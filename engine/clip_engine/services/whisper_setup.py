"""
One-click setup for built-in Whisper, run by the desktop app with the engine's Python.

    python whisper_setup.py download <model>
    python whisper_setup.py gpu-test <model>
    python whisper_setup.py transcribe <model> <device> <language> <wav>...

``download`` fetches a model into the cache the engine reads from (the
Hugging Face hub cache for faster-whisper, ``~/.cache/whisper`` for
openai-whisper), so the first clip job does not stall on a silent download.
``gpu-test`` loads an installed model on the GPU and runs it on one second of
silence, which is the only reliable way to know that the CUDA libraries work.

Output is one JSON object per line on stdout:

    {"event": "progress", "done": <bytes>, "total": <bytes>}
    {"event": "done"}
    {"event": "error", "reason": "<code>", "message": "<plain words>"}

Messages are fixed strings, so no path or library error text reaches the app.
The app stops a download by ending this process; a partial Hugging Face
download resumes next time, a partial openai-whisper file is discarded.

This file imports nothing from the engine package, like local_whisper.py,
which it loads from its own folder.
"""

import errno
import fnmatch
import hashlib
import importlib.util
import json
import os
import sys
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any, Callable, Optional

# Predictable byte-by-byte downloads, and nothing written to the console.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# faster-whisper's own model table (faster_whisper/utils.py), for the models the app lists.
FASTER_WHISPER_REPOS = {
    "tiny.en": "Systran/faster-whisper-tiny.en",
    "tiny": "Systran/faster-whisper-tiny",
    "base.en": "Systran/faster-whisper-base.en",
    "base": "Systran/faster-whisper-base",
    "small.en": "Systran/faster-whisper-small.en",
    "small": "Systran/faster-whisper-small",
    "medium.en": "Systran/faster-whisper-medium.en",
    "medium": "Systran/faster-whisper-medium",
    "large-v3": "Systran/faster-whisper-large-v3",
    "turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
}
# The files faster_whisper.utils.download_model asks for.
FASTER_WHISPER_FILES = ["config.json", "preprocessor_config.json", "model.bin", "tokenizer.json", "vocabulary.*"]

PROGRESS_INTERVAL_SECONDS = 0.5
CHUNK_BYTES = 1024 * 1024

MESSAGES = {
    "usage": "Unknown setup command.",
    "unknown_model": "Choose a model from the list.",
    "missing_backend": "Whisper is not installed for BridgeClip's Python.",
    "network": "Could not download the model. Check your internet connection and try again.",
    "disk": "There is not enough free disk space for this model.",
    "checksum": "The download was damaged. Try again.",
    "failed": "Could not download the model.",
    "not_installed": "Install the model before testing the GPU.",
    "gpu_unavailable": "No NVIDIA GPU with CUDA was found.",
    "gpu_cublas": "The GPU libraries are missing (cuBLAS). Use Enable GPU to install them.",
    "gpu_cudnn": "The GPU libraries are missing (cuDNN). Use Enable GPU to install them.",
    "gpu_memory": "The model does not fit in your video memory. Pick a smaller model or use CPU.",
    "gpu_failed": "Whisper could not run on the GPU.",
    "transcribe_model": "The Whisper model could not be loaded. Install it in Settings.",
    "transcribe_failed": "Local Whisper could not transcribe the audio.",
    "transcribe_not_installed": "The Whisper model is not installed. Install it in Settings.",
}


def emit(event: str, **fields: Any) -> None:
    print(json.dumps({"event": event, **fields}), flush=True)


def fail(reason: str) -> int:
    emit("error", reason=reason, message=MESSAGES.get(reason, MESSAGES["failed"]))
    return 1


def load_local_whisper() -> Any:
    path = Path(__file__).with_name("local_whisper.py")
    spec = importlib.util.spec_from_file_location("bridgeclip_local_whisper", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def classify_download_error(exc: BaseException) -> str:
    if isinstance(exc, OSError) and exc.errno == errno.ENOSPC:
        return "disk"
    name = type(exc).__name__.lower()
    text = str(exc).lower()
    if any(word in name for word in ("connect", "timeout", "http", "network", "offline", "localentrynotfound", "urlerror")):
        return "network"
    if "no space" in text:
        return "disk"
    return "failed"


def _regular_file_bytes(folder: str) -> int:
    """Bytes in the regular files directly inside a folder; links are not counted."""
    total = 0
    try:
        with os.scandir(folder) as entries:
            for entry in entries:
                if entry.is_file(follow_symlinks=False):
                    # os.stat, not entry.stat: on Windows the directory listing keeps a stale
                    # size for a file that is still being written.
                    total += os.stat(entry.path, follow_symlinks=False).st_size
    except OSError:
        pass
    return total


def _watch(measure: Callable[[], int], total: int) -> Callable[[], None]:
    """Reports progress twice a second until the returned function is called."""
    stop = threading.Event()

    def run() -> None:
        last = -1
        while not stop.wait(PROGRESS_INTERVAL_SECONDS):
            done = min(total, measure()) if total else measure()
            if done != last:
                emit("progress", done=done, total=total)
                last = done

    thread = threading.Thread(target=run, daemon=True)
    thread.start()

    def finish() -> None:
        stop.set()
        thread.join(timeout=2)

    return finish


def download_faster_whisper(name: str) -> int:
    from huggingface_hub import HfApi, snapshot_download
    from huggingface_hub.constants import HF_HUB_CACHE

    repo = FASTER_WHISPER_REPOS[name]
    info = HfApi().model_info(repo, files_metadata=True)
    wanted = [s for s in info.siblings or [] if any(fnmatch.fnmatch(s.rfilename, p) for p in FASTER_WHISPER_FILES)]
    total = sum(int(s.size or 0) for s in wanted)
    folder = os.path.join(HF_HUB_CACHE, "models--" + repo.replace("/", "--"))
    # Downloads land in blobs/ (as .incomplete files first). Where Windows cannot make links,
    # finished files move on into the snapshot folder, so both are measured.
    measure = lambda: _regular_file_bytes(os.path.join(folder, "blobs")) + \
        _regular_file_bytes(os.path.join(folder, "snapshots", info.sha or ""))
    emit("progress", done=min(total, measure()), total=total)
    finish = _watch(measure, total)
    try:
        # No pinned revision: downloading "main" also records refs/main, which faster-whisper
        # needs to find the model offline.
        snapshot_download(repo, allow_patterns=FASTER_WHISPER_FILES)
    finally:
        finish()
    emit("progress", done=total, total=total)
    return 0


def openai_whisper_root() -> str:
    # The folder whisper.load_model downloads to by default.
    default = os.path.join(os.path.expanduser("~"), ".cache")
    return os.path.join(os.getenv("XDG_CACHE_HOME", default), "whisper")


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def download_openai_whisper(name: str) -> int:
    import whisper

    url = whisper._MODELS["turbo" if name == "turbo" else name]
    expected = url.split("/")[-2]
    root = openai_whisper_root()
    os.makedirs(root, exist_ok=True)
    target = os.path.join(root, os.path.basename(url))
    if os.path.isfile(target) and _sha256(target) == expected:
        size = os.path.getsize(target)
        emit("progress", done=size, total=size)
        return 0
    partial = target + ".part"
    with urllib.request.urlopen(url, timeout=30) as response:
        total = int(response.headers.get("Content-Length") or 0)
        done = 0
        last_report = 0.0
        with open(partial, "wb") as output:
            while True:
                block = response.read(CHUNK_BYTES)
                if not block:
                    break
                output.write(block)
                done += len(block)
                now = time.monotonic()
                if now - last_report >= PROGRESS_INTERVAL_SECONDS:
                    emit("progress", done=done, total=total)
                    last_report = now
    if _sha256(partial) != expected:
        os.remove(partial)
        return fail("checksum")
    os.replace(partial, target)
    emit("progress", done=done, total=total or done)
    return 0


def download(name: str) -> int:
    local_whisper = load_local_whisper()
    if name not in FASTER_WHISPER_REPOS:
        return fail("unknown_model")
    backend = local_whisper.detect_backend()
    if backend is None:
        return fail("missing_backend")
    try:
        code = download_faster_whisper(name) if backend == "faster-whisper" else download_openai_whisper(name)
    except Exception as exc:  # noqa: BLE001 - every failure becomes one fixed message
        return fail(classify_download_error(exc))
    if code == 0:
        emit("done", backend=backend)
    return code


def classify_gpu_error(exc: BaseException) -> str:
    text = f"{type(exc).__name__} {exc}".lower()
    if "out of memory" in text:
        return "gpu_memory"
    if "cublas" in text:
        return "gpu_cublas"
    if "cudnn" in text:
        return "gpu_cudnn"
    return "gpu_failed"


def gpu_test(name: str, run: Optional[Callable[[Any, str, str], None]] = None) -> int:
    local_whisper = load_local_whisper()
    if name not in FASTER_WHISPER_REPOS:
        return fail("unknown_model")
    backend = local_whisper.detect_backend()
    if backend is None:
        return fail("missing_backend")
    if not local_whisper._cuda_available(backend):
        return fail("gpu_unavailable")
    try:
        (run or _run_on_gpu)(local_whisper, backend, name)
    except FileNotFoundError:
        return fail("not_installed")
    except Exception as exc:  # noqa: BLE001 - every failure becomes one fixed message
        return fail(classify_gpu_error(exc))
    emit("done", backend=backend)
    return 0


def _run_on_gpu(local_whisper: Any, backend: str, name: str) -> None:
    """Loads the model on the GPU and transcribes 30 seconds of silence the way a job does.

    30 seconds is one full Whisper window, and beam size, word timestamps and float16
    match a clip job, so the memory it needs is close to a real run.
    """
    import numpy as np

    silence = np.zeros(16000 * 30, dtype=np.float32)
    if backend == "faster-whisper":
        from faster_whisper import WhisperModel

        model_id = "large-v3-turbo" if name == "turbo" else name
        try:
            model = WhisperModel(model_id, device="cuda", compute_type="float16", local_files_only=True)
        except Exception as exc:
            # A model that is not downloaded surfaces as the hub's LocalEntryNotFoundError.
            if type(exc).__name__ == "LocalEntryNotFoundError":
                raise FileNotFoundError(name) from None
            raise
        segments, _ = model.transcribe(silence, beam_size=5, word_timestamps=True, vad_filter=False,
                                       condition_on_previous_text=False, language="en")
        list(segments)
        return
    import whisper

    if not os.path.isfile(os.path.join(openai_whisper_root(), os.path.basename(whisper._MODELS[name]))):
        raise FileNotFoundError(name)
    model = whisper.load_model(name, device="cuda")
    model.transcribe(silence, fp16=True, verbose=None, condition_on_previous_text=False)


TRANSCRIBE_REASONS = {
    "local_whisper_missing": "missing_backend",
    "local_whisper_model": "transcribe_model",
    "local_whisper_gpu": "gpu_failed",
}


def model_installed(local_whisper: Any, name: str) -> bool:
    """Whether the engine's Whisper already has this model, so using it downloads nothing."""
    backend = local_whisper.detect_backend()
    try:
        if backend == "faster-whisper":
            from huggingface_hub import try_to_load_from_cache

            return isinstance(try_to_load_from_cache(FASTER_WHISPER_REPOS[name], "model.bin"), str)
        if backend == "openai-whisper":
            import whisper

            return os.path.isfile(os.path.join(openai_whisper_root(), os.path.basename(whisper._MODELS[name])))
    except Exception:  # noqa: BLE001 - an unreadable cache counts as not installed
        return False
    return False


def transcribe(model: str, device: str, language: str, paths: list[str],
               run: Optional[Callable[..., dict]] = None, installed: Optional[Callable[[str], bool]] = None) -> int:
    """Transcribes short WAV files (16 kHz mono PCM) for the app, one text line per file.

    Used for writing titles and captions, so only the text is returned. The model
    loads once and stays loaded for every file.
    """
    local_whisper = load_local_whisper()
    if model not in FASTER_WHISPER_REPOS:
        return fail("unknown_model")
    if device not in ("auto", "cpu", "gpu"):
        return fail("usage")
    if local_whisper.detect_backend() is None:
        return fail("missing_backend")
    # Titles and captions never trigger a silent multi-gigabyte download.
    if not (installed or (lambda name: model_installed(local_whisper, name)))(model):
        return fail("transcribe_not_installed")
    chosen_language = None if language in ("", "auto") else language
    if model.endswith(".en"):
        chosen_language = "en"
    for index, path in enumerate(paths):
        try:
            result = (run or local_whisper._run_builtin)(path, chosen_language, None, model, None, device)
        except Exception as exc:  # noqa: BLE001 - every failure becomes one fixed message
            return fail(TRANSCRIBE_REASONS.get(getattr(exc, "reason", ""), "transcribe_failed"))
        emit("text", index=index, text=str(result.get("text") or ""))
    emit("done")
    return 0


def main(argv: list[str]) -> int:
    if len(argv) >= 6 and argv[1] == "transcribe":
        return transcribe(argv[2], argv[3], argv[4], argv[5:])
    if len(argv) != 3 or argv[1] not in ("download", "gpu-test"):
        return fail("usage")
    return download(argv[2]) if argv[1] == "download" else gpu_test(argv[2])


if __name__ == "__main__":
    sys.exit(main(sys.argv))
