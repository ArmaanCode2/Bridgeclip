"""
Whisper speech-to-text on this computer.

Two ways to run it, chosen in Settings:

* ``builtin``: inside the engine's Python, with ``faster-whisper`` when it is
  installed (several times faster on a CPU) or ``openai-whisper``.
* ``server``: any OpenAI-compatible ``/v1/audio/transcriptions`` server that
  can return word timestamps (faster-whisper-server, LocalAI, ...).

Both return the OpenAI ``verbose_json`` shape the rest of the pipeline already
parses: ``{"text", "language", "words": [{"word", "start", "end"}]}``, in
seconds. Captions, cut snapping and pacing all depend on word timings, so
nothing here invents a timestamp: a backend that cannot produce them fails
with a clear reason instead.
"""

import asyncio
import glob
import importlib.util
import inspect
import json
import logging
import math
import os
import re
import sys
import threading
import wave
from pathlib import Path
from typing import Any, Callable, Optional
from urllib.parse import urlsplit

logger = logging.getLogger(__name__)

# Model names the built-in backends accept. Anything else is refused up front.
BUILTIN_MODELS = (
    "tiny", "tiny.en", "base", "base.en", "small", "small.en", "medium", "medium.en",
    "large", "large-v1", "large-v2", "large-v3", "turbo", "large-v3-turbo",
)
DEFAULT_MODEL = "small"

MAX_SERVER_AUDIO_BYTES = 64 * 1024 * 1024
MAX_SERVER_RESPONSE_BYTES = 8 * 1024 * 1024
# A CPU-only server can need minutes for one five-minute chunk.
SERVER_READ_TIMEOUT_SECONDS = 1800.0

# Whisper hears the end of its prompt, so keep vocabulary hints short.
MAX_PROMPT_CHARS = 400
# A word is never shown for less than this, so it cannot flash by unseen.
MIN_WORD_SECONDS = 0.04
# faster-whisper: 30-second windows of speech used to detect the language.
LANGUAGE_DETECTION_SEGMENTS = 3

_lock = threading.Lock()
_models: dict[tuple[str, str, bool], Any] = {}
# In "auto" mode, set once a GPU turned out not to work (for example its CUDA
# libraries are not installed), so the rest of the run stays on the CPU instead
# of failing again. Never consulted when the user chose CPU or GPU.
_cuda_disabled = False
DEVICES = ("auto", "cpu", "gpu")
_GPU_FAILURE_MARKERS = ("cublas", "cudnn", "cufft", "cuda", "cudart", "nvrtc", "out of memory", ".dll")


class LocalWhisperError(Exception):
    """Local transcription failed. ``reason`` is a stable code for error mapping."""

    def __init__(self, message: str, reason: str):
        super().__init__(message)
        self.reason = reason


_dll_directories: list[Any] = []


def prepare_gpu_libraries() -> list[str]:
    """Let CTranslate2 find NVIDIA's cuBLAS and cuDNN from their pip packages.

    On Windows, ``pip install nvidia-cublas-cu12 nvidia-cudnn-cu12`` puts the
    DLLs in ``site-packages/nvidia/*/bin``, which Windows does not search.
    CTranslate2 loads them when the GPU is first used, so the folders go on
    both the DLL search list and PATH. Elsewhere this does nothing.
    """
    if sys.platform != "win32":
        return []
    found: list[str] = []
    for entry in sys.path:
        if not entry or not os.path.isdir(os.path.join(entry, "nvidia")):
            continue
        for folder in sorted(glob.glob(os.path.join(entry, "nvidia", "*", "bin"))):
            if folder in found or not os.path.isdir(folder):
                continue
            found.append(folder)
            try:
                _dll_directories.append(os.add_dll_directory(folder))
            except OSError:
                pass
    if found:
        path = os.environ.get("PATH", "")
        missing = [folder for folder in found if folder not in path.split(os.pathsep)]
        if missing:
            os.environ["PATH"] = os.pathsep.join([*missing, path]) if path else os.pathsep.join(missing)
    return found


prepare_gpu_libraries()


def detect_backend() -> Optional[str]:
    """The built-in backend available to this Python, fastest first."""
    if importlib.util.find_spec("faster_whisper") is not None:
        return "faster-whisper"
    if importlib.util.find_spec("whisper") is not None:
        try:
            import whisper
        except Exception:
            return None
        if hasattr(whisper, "load_model"):
            return "openai-whisper"
    return None


class _GpuFailure(Exception):
    """The GPU was in use and failed. Handled according to the chosen device."""


def _looks_like_gpu_failure(exc: BaseException) -> bool:
    return any(marker in str(exc).lower() for marker in _GPU_FAILURE_MARKERS)


def _cuda_available(backend: str) -> bool:
    """Whether a CUDA device is visible to the backend. Only called for auto and gpu."""
    try:
        if backend == "faster-whisper":
            import ctranslate2
            return ctranslate2.get_cuda_device_count() > 0
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def describe_environment() -> dict[str, Any]:
    """What built-in Whisper can use here: the backend and whether CUDA is visible.

    The desktop app's "Check Whisper" runs this file directly, so what it reports
    is exactly what a job will do.
    """
    backend = detect_backend()
    return {"backend": backend, "cuda": bool(backend and _cuda_available(backend))}


def normalize_model(name: str) -> str:
    model = (name or "").strip() or DEFAULT_MODEL
    if model not in BUILTIN_MODELS:
        raise LocalWhisperError(
            f"Unknown Whisper model. Choose one of: {', '.join(BUILTIN_MODELS)}", "local_whisper_model")
    return model


def vocabulary_prompt(keyterms: Optional[list[str]]) -> Optional[str]:
    """A short prompt that nudges Whisper toward the spelling of names and jargon."""
    terms = [t for t in (keyterms or []) if isinstance(t, str) and t.strip()]
    if not terms:
        return None
    prompt = "Vocabulary: " + ", ".join(terms) + "."
    return prompt[:MAX_PROMPT_CHARS]


def _read_wav(path: str) -> Any:
    """Float32 mono samples at 16 kHz from the engine's own PCM WAV chunks.

    Reading the samples directly keeps Whisper from shelling out to ffmpeg.
    """
    import numpy as np

    try:
        with wave.open(path, "rb") as wav:
            if wav.getsampwidth() != 2 or wav.getnchannels() != 1 or wav.getframerate() != 16000:
                raise LocalWhisperError("Audio was not prepared as 16 kHz mono PCM", "local_whisper_failed")
            frames = wav.readframes(wav.getnframes())
    except (OSError, EOFError, wave.Error):
        raise LocalWhisperError("Could not read the audio for Whisper", "local_whisper_failed") from None
    return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0


def _load_model(backend: str, name: str, progress: Optional[Callable[[str], None]], use_cuda: bool) -> Any:
    key = (backend, name, use_cuda)
    with _lock:
        cached = _models.get(key)
        if cached is not None:
            return cached
        if progress:
            progress(f"Loading Whisper {name} (the first run downloads the model)...")
        try:
            if backend == "faster-whisper":
                from faster_whisper import WhisperModel
                # faster-whisper names the turbo model in full.
                model_id = "large-v3-turbo" if name == "turbo" else name
                model = WhisperModel(model_id, device="cuda" if use_cuda else "cpu",
                                     compute_type="float16" if use_cuda else "int8")
            else:
                import whisper
                model = whisper.load_model(name, device="cuda" if use_cuda else "cpu")
        except LocalWhisperError:
            raise
        except Exception as exc:
            logger.error("Could not load Whisper model %s: %s", name, exc, exc_info=True)
            if use_cuda and _looks_like_gpu_failure(exc):
                raise _GpuFailure() from None
            raise LocalWhisperError("Whisper model could not be loaded", "local_whisper_model") from None
        _models[key] = model
        logger.info("Loaded %s model %s on %s", backend, name, "cuda" if use_cuda else "cpu")
        return model


def _finite(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def normalize_words(raw_words: list[Any], duration: float) -> list[dict[str, Any]]:
    """Clean Whisper word timings so the pipeline can trust them.

    Whisper can emit words with equal start and end, with an end past the audio,
    or with a word that overlaps the next one. Overlaps would make two caption
    words visible at once and zero-length words would flash, so times are made
    monotonic and clamped to the audio. Words are never dropped for timing alone
    unless they fall wholly outside the audio.
    """
    cleaned: list[dict[str, Any]] = []
    for raw in raw_words:
        if not isinstance(raw, dict):
            continue
        word = " ".join(str(raw.get("word", "")).split())
        start, end = _finite(raw.get("start")), _finite(raw.get("end"))
        if not word or start is None or end is None or start >= duration:
            continue
        start = max(0.0, start)
        end = min(max(end, start), duration)
        cleaned.append({"word": word, "start": start, "end": end})
    cleaned.sort(key=lambda w: (w["start"], w["end"]))
    for index, word in enumerate(cleaned):
        following = cleaned[index + 1]["start"] if index + 1 < len(cleaned) else duration
        # A word ends no later than the next one starts, and lasts at least
        # MIN_WORD_SECONDS when there is room for that.
        limit = max(following, word["start"])
        shortest_end = min(word["start"] + MIN_WORD_SECONDS, limit)
        word["end"] = max(min(word["end"], limit), shortest_end)
    return cleaned


def _run_builtin(path: str, language: Optional[str], prompt: Optional[str], model_name: str,
                 progress: Optional[Callable[[str], None]], device: str = "auto") -> dict[str, Any]:
    """Transcribe on this computer.

    ``device`` is the user's choice: "cpu" never touches CUDA, "gpu" requires it
    and reports a clear error if it cannot be used, and "auto" tries the GPU and
    continues on the CPU if it fails.
    """
    global _cuda_disabled
    backend = detect_backend()
    if backend is None:
        raise LocalWhisperError(
            "Local Whisper is not installed for the engine's Python. Install faster-whisper or openai-whisper.",
            "local_whisper_missing")
    name = normalize_model(model_name)
    device = device if device in DEVICES else "auto"

    if device == "cpu":
        return _attempt(backend, name, path, language, prompt, progress, use_cuda=False)

    if device == "gpu":
        gpu_error = LocalWhisperError(
            "Local Whisper could not use the GPU. Choose Auto or CPU, or install the CUDA libraries.", "local_whisper_gpu")
        if not _cuda_available(backend):
            raise gpu_error
        try:
            return _attempt(backend, name, path, language, prompt, progress, use_cuda=True)
        except _GpuFailure:
            raise gpu_error from None

    for use_cuda in ((not _cuda_disabled and _cuda_available(backend)), False):
        try:
            return _attempt(backend, name, path, language, prompt, progress, use_cuda=use_cuda)
        except _GpuFailure:
            logger.warning("Whisper could not use the GPU; continuing on the CPU")
            _cuda_disabled = True
            with _lock:
                _models.pop((backend, name, True), None)
            if progress:
                progress("The GPU could not be used for Whisper; continuing on the CPU...")
    raise LocalWhisperError("Local Whisper transcription failed", "local_whisper_failed")


def _echoes_prompt(text: str, prompt: Optional[str]) -> bool:
    """Whisper sometimes writes its prompt out over quiet audio ("Vocabulary: A, B.").

    The vocabulary prompt goes before every window, so such a segment is dropped.
    """
    if not prompt:
        return False
    words = re.findall(r"\w+", text.lower())
    return bool(words) and words[0] == "vocabulary" and set(words) <= set(re.findall(r"\w+", prompt.lower()))


def _accepts(function: Callable[..., Any], name: str) -> bool:
    """Whether `function` has a parameter called `name`."""
    try:
        return name in inspect.signature(function).parameters
    except (TypeError, ValueError):
        return False


def _attempt(backend: str, name: str, path: str, language: Optional[str], prompt: Optional[str],
             progress: Optional[Callable[[str], None]], use_cuda: bool) -> dict[str, Any]:
    """One transcription on a fixed device. GPU trouble raises _GpuFailure."""
    try:
        return _transcribe_with(backend, name, path, language, prompt, progress, use_cuda)
    except (LocalWhisperError, _GpuFailure):
        raise
    except Exception as exc:
        if use_cuda and _looks_like_gpu_failure(exc):
            logger.warning("Whisper GPU failure: %s", type(exc).__name__)
            raise _GpuFailure() from None
        logger.error("Local Whisper failed: %s", exc, exc_info=True)
        raise LocalWhisperError("Local Whisper transcription failed", "local_whisper_failed") from None


def _transcribe_with(backend: str, name: str, path: str, language: Optional[str], prompt: Optional[str],
                     progress: Optional[Callable[[str], None]], use_cuda: bool) -> dict[str, Any]:
    model = _load_model(backend, name, progress, use_cuda)
    audio = _read_wav(path)
    duration = len(audio) / 16000.0
    raw_words: list[dict[str, Any]] = []
    texts: list[str] = []
    detected: Optional[str] = language
    probability: Optional[float] = None
    if backend == "faster-whisper":
        # Hotwords go before every 30-second window; an initial prompt only
        # reaches the first one when windows do not condition on each other.
        segments, info = model.transcribe(
            audio, language=language, word_timestamps=True, hotwords=prompt,
            condition_on_previous_text=False, vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500}, beam_size=5,
            # Detect from up to three 30-second windows of speech, not just the first,
            # so a music intro is less likely to decide the language.
            language_detection_segments=LANGUAGE_DETECTION_SEGMENTS,
        )
        detected = getattr(info, "language", None) or language
        if language is None:
            probability = _finite(getattr(info, "language_probability", None))
        for segment in segments:
            if _echoes_prompt(segment.text, prompt):
                continue
            texts.append(segment.text.strip())
            for word in segment.words or []:
                raw_words.append({"word": word.word, "start": word.start, "end": word.end})
    else:
        # openai-whisper 20250625 and later can repeat the prompt before every
        # window too; older releases refuse the option, so it is sent only when accepted.
        carry = {"carry_initial_prompt": True} if prompt and _accepts(model.transcribe, "carry_initial_prompt") else {}
        result = model.transcribe(
            audio, language=language, word_timestamps=True, initial_prompt=prompt,
            condition_on_previous_text=False, fp16=use_cuda, verbose=None, **carry,
        )
        detected = result.get("language") or language
        for segment in result.get("segments", []):
            # Whisper reports silence as speech now and then. A segment it
            # itself rates as probably silent and low confidence is noise.
            if segment.get("no_speech_prob", 0) > 0.6 and segment.get("avg_logprob", 0) < -1.0:
                continue
            if _echoes_prompt(str(segment.get("text", "")), prompt):
                continue
            texts.append(str(segment.get("text", "")).strip())
            raw_words.extend(segment.get("words", []))
    result: dict[str, Any] = {"text": " ".join(t for t in texts if t), "language": detected,
                              "words": normalize_words(raw_words, duration)}
    if probability is not None:
        result["language_probability"] = probability
    return result


def server_endpoint(url: str) -> str:
    """The transcription URL for a server base URL such as http://127.0.0.1:8000/v1."""
    raw = (url or "").strip().rstrip("/")
    parts = urlsplit(raw)
    if parts.scheme not in ("http", "https") or not parts.hostname or parts.username or parts.password:
        raise LocalWhisperError("Local Whisper server URL is not configured", "local_whisper_unreachable")
    if raw.endswith("/audio/transcriptions"):
        return raw
    return f"{raw}/audio/transcriptions" if raw.endswith("/v1") else f"{raw}/v1/audio/transcriptions"


async def _transcribe_server(settings: Any, path: str, language: Optional[str], prompt: Optional[str],
                             model: str) -> dict[str, Any]:
    import httpx

    endpoint = server_endpoint(getattr(settings, "local_whisper_url", ""))
    if os.path.getsize(path) > MAX_SERVER_AUDIO_BYTES:
        raise LocalWhisperError("Audio chunk is too large for the server", "local_whisper_failed")
    audio = await asyncio.to_thread(Path(path).read_bytes)
    data: dict[str, Any] = {
        "model": model or "whisper-1",
        "response_format": "verbose_json",
        "timestamp_granularities[]": ["word", "segment"],
    }
    if language:
        data["language"] = language
    if prompt:
        data["prompt"] = prompt
    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(SERVER_READ_TIMEOUT_SECONDS, connect=5.0), follow_redirects=False,
        ) as client:
            async with client.stream(
                "POST", endpoint, data=data, files={"file": (os.path.basename(path), audio, "audio/wav")},
                headers={"Accept-Encoding": "identity"},
            ) as response:
                body = bytearray()
                async for chunk in response.aiter_raw():
                    if len(chunk) > MAX_SERVER_RESPONSE_BYTES - len(body):
                        raise LocalWhisperError("Server response was too large", "local_whisper_failed")
                    body.extend(chunk)
                status = response.status_code
    except LocalWhisperError:
        raise
    except (httpx.TimeoutException, httpx.TransportError):
        raise LocalWhisperError("Local Whisper server could not be reached", "local_whisper_unreachable") from None
    if status == 404:
        raise LocalWhisperError("Local Whisper server has no transcription endpoint at this URL", "local_whisper_unreachable")
    if status != 200:
        raise LocalWhisperError(f"Local Whisper server returned HTTP {status}", "local_whisper_failed")
    try:
        result = json.loads(body)
    except (ValueError, UnicodeError, RecursionError):
        raise LocalWhisperError("Local Whisper server returned invalid JSON", "local_whisper_failed") from None
    if not isinstance(result, dict):
        raise LocalWhisperError("Local Whisper server returned an invalid response", "local_whisper_failed")

    raw_words = result.get("words")
    if not raw_words and isinstance(result.get("segments"), list):
        raw_words = [w for seg in result["segments"] if isinstance(seg, dict) for w in (seg.get("words") or [])]
    text = result.get("text") if isinstance(result.get("text"), str) else ""
    if not raw_words:
        if text.strip():
            raise LocalWhisperError(
                "Local Whisper server did not return word timestamps", "local_whisper_no_words")
        raw_words = []
    with wave.open(path, "rb") as wav:
        duration = wav.getnframes() / float(wav.getframerate() or 16000)
    return {
        "text": text,
        "language": result.get("language") if isinstance(result.get("language"), str) else language,
        "words": normalize_words(raw_words, duration),
    }


async def transcribe(settings: Any, audio_path: str, language: Optional[str], keyterms: Optional[list[str]],
                     model: str, progress: Optional[Callable[[str], None]] = None) -> dict[str, Any]:
    """Transcribe one prepared chunk with word timestamps, using the chosen mode."""
    prompt = vocabulary_prompt(keyterms)
    if getattr(settings, "local_whisper_mode", "builtin") == "server":
        return await _transcribe_server(settings, audio_path, language, prompt, model)
    if (model or "").strip().endswith(".en"):
        # English-only models cannot transcribe another language.
        language = "en"
    loop = asyncio.get_running_loop()

    def notify(message: str) -> None:
        # Whisper runs on a worker thread; progress callbacks belong to the loop.
        if progress:
            loop.call_soon_threadsafe(progress, message)

    device = getattr(settings, "local_whisper_device", "auto")
    return await asyncio.to_thread(_run_builtin, audio_path, language, prompt, model, notify, device)
