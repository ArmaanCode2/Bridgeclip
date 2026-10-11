"""Offline tests for local Whisper transcription and its pipeline integration."""

import asyncio
import json
import math
import struct
import threading
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from clip_engine.error_policy import (
    LOCAL_WHISPER_ERRORS,
    safe_failure_code,
    safe_job_error_text,
    safe_processing_error,
)
from clip_engine.services import local_whisper
from clip_engine.services import transcription_service as ts
from clip_engine.services.local_whisper import LocalWhisperError, normalize_words


def write_wav(path, seconds, rate=16000):
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(struct.pack("<h", 0) * int(seconds * rate))


def run(coro):
    return asyncio.run(coro)


class TestNormalizeWords:
    def test_overlapping_words_are_made_sequential(self):
        words = normalize_words([
            {"word": " Hello", "start": 0.0, "end": 1.0},
            {"word": "world.", "start": 0.6, "end": 1.4},
        ], 5.0)
        assert [w["word"] for w in words] == ["Hello", "world."]
        assert words[0]["end"] <= words[1]["start"]

    def test_end_before_start_is_repaired(self):
        words = normalize_words([{"word": "a", "start": 2.0, "end": 1.0}], 5.0)
        assert words[0]["end"] >= words[0]["start"]

    def test_zero_length_words_get_a_visible_duration_when_there_is_room(self):
        words = normalize_words([{"word": "a", "start": 1.0, "end": 1.0}, {"word": "b", "start": 2.0, "end": 2.5}], 5.0)
        assert words[0]["end"] - words[0]["start"] >= local_whisper.MIN_WORD_SECONDS - 1e-9

    def test_times_are_clamped_to_the_audio_so_the_parser_does_not_reject_them(self):
        words = normalize_words([{"word": "a", "start": 4.9, "end": 7.0}], 5.0)
        assert words[0]["end"] <= 5.0

    def test_words_wholly_outside_the_audio_or_with_bad_numbers_are_dropped(self):
        words = normalize_words([
            {"word": "late", "start": 6.0, "end": 7.0},
            {"word": "nan", "start": float("nan"), "end": 1.0},
            {"word": "", "start": 1.0, "end": 2.0},
            {"word": "inf", "start": 1.0, "end": math.inf},
            "garbage",
            {"word": "ok", "start": 1.0, "end": 2.0},
        ], 5.0)
        assert [w["word"] for w in words] == ["ok"]

    def test_unsorted_input_is_sorted(self):
        words = normalize_words([{"word": "b", "start": 2.0, "end": 3.0}, {"word": "a", "start": 0.0, "end": 1.0}], 5.0)
        assert [w["word"] for w in words] == ["a", "b"]

    def test_output_uses_plain_floats(self):
        # numpy scalars would fail the parser's `type(value) in (int, float)` check.
        numpy = pytest.importorskip("numpy")
        words = normalize_words([{"word": "a", "start": numpy.float32(0.5), "end": numpy.float64(1.5)}], 5.0)
        assert type(words[0]["start"]) is float and type(words[0]["end"]) is float

    def test_parser_accepts_normalized_output(self):
        service = ts.TranscriptionService.__new__(ts.TranscriptionService)
        service.settings = SimpleNamespace(transcription_model="m")
        words = normalize_words([{"word": "Hi", "start": 0.0, "end": 0.0}, {"word": "there.", "start": 0.4, "end": 99.0}], 3.0)
        result = service._parse_openrouter_response({"text": "Hi there.", "words": words}, 3.0, "small")
        assert result.segments[0].text == "Hi there."


class TestModelAndPrompt:
    @pytest.mark.parametrize("name", ["tiny", "small", "medium", "large-v3", "turbo", ""])
    def test_known_models_are_accepted(self, name):
        assert local_whisper.normalize_model(name) in local_whisper.BUILTIN_MODELS

    @pytest.mark.parametrize("name", ["../../etc/passwd", "gpt-4", "small; rm", "C:\\models\\x.pt"])
    def test_unknown_models_are_refused(self, name):
        with pytest.raises(LocalWhisperError) as raised:
            local_whisper.normalize_model(name)
        assert raised.value.reason == "local_whisper_model"

    def test_vocabulary_prompt(self):
        assert local_whisper.vocabulary_prompt(None) is None
        assert local_whisper.vocabulary_prompt(["Bridgemind", "Zernio"]) == "Vocabulary: Bridgemind, Zernio."
        assert len(local_whisper.vocabulary_prompt(["x" * 40] * 100)) <= local_whisper.MAX_PROMPT_CHARS


class TestBuiltinBackend:
    def fake_openai_whisper(self, monkeypatch, result):
        class Model:
            def transcribe(self, audio, **kwargs):
                self.kwargs = kwargs
                return result

        model = Model()
        monkeypatch.setattr(local_whisper, "detect_backend", lambda: "openai-whisper")
        monkeypatch.setattr(local_whisper, "_load_model", lambda backend, name, progress, use_cuda: model)
        monkeypatch.setattr(local_whisper, "_cuda_available", lambda backend: False)
        return model

    def test_returns_normalized_words_and_language(self, tmp_path, monkeypatch):
        write_wav(tmp_path / "a.wav", 4)
        model = self.fake_openai_whisper(monkeypatch, {"language": "en", "segments": [
            {"text": " Hello world.", "no_speech_prob": 0.01, "avg_logprob": -0.2,
             "words": [{"word": " Hello", "start": 0.1, "end": 0.5}, {"word": " world.", "start": 0.6, "end": 1.1}]}]})
        out = local_whisper._run_builtin(str(tmp_path / "a.wav"), None, "Vocabulary: X.", "small", None)
        assert out["language"] == "en" and out["text"] == "Hello world."
        assert [w["word"] for w in out["words"]] == ["Hello", "world."]
        # Settings that matter for caption quality and for hallucination control.
        assert model.kwargs["word_timestamps"] is True
        assert model.kwargs["condition_on_previous_text"] is False
        assert model.kwargs["initial_prompt"] == "Vocabulary: X."
        assert model.kwargs["fp16"] is False

    def test_segments_whisper_itself_rates_as_silence_are_dropped(self, tmp_path, monkeypatch):
        write_wav(tmp_path / "a.wav", 4)
        self.fake_openai_whisper(monkeypatch, {"language": "en", "segments": [
            {"text": "Thanks for watching!", "no_speech_prob": 0.9, "avg_logprob": -1.5,
             "words": [{"word": "Thanks", "start": 0.0, "end": 0.5}]},
            {"text": "Real speech.", "no_speech_prob": 0.1, "avg_logprob": -0.3,
             "words": [{"word": "Real", "start": 1.0, "end": 1.4}, {"word": "speech.", "start": 1.5, "end": 2.0}]}]})
        out = local_whisper._run_builtin(str(tmp_path / "a.wav"), "en", None, "small", None)
        assert out["text"] == "Real speech."
        assert [w["word"] for w in out["words"]] == ["Real", "speech."]

    def test_missing_backend_is_a_clear_error(self, tmp_path, monkeypatch):
        write_wav(tmp_path / "a.wav", 1)
        monkeypatch.setattr(local_whisper, "detect_backend", lambda: None)
        with pytest.raises(LocalWhisperError) as raised:
            local_whisper._run_builtin(str(tmp_path / "a.wav"), None, None, "small", None)
        assert raised.value.reason == "local_whisper_missing"

    def test_backend_crash_does_not_leak_details(self, tmp_path, monkeypatch):
        write_wav(tmp_path / "a.wav", 1)

        class Model:
            def transcribe(self, *args, **kwargs):
                raise RuntimeError("CUDA out of memory at C:\\Users\\someone\\secret")

        monkeypatch.setattr(local_whisper, "detect_backend", lambda: "openai-whisper")
        monkeypatch.setattr(local_whisper, "_load_model", lambda *a: Model())
        monkeypatch.setattr(local_whisper, "_cuda_available", lambda backend: False)
        with pytest.raises(LocalWhisperError) as raised:
            local_whisper._run_builtin(str(tmp_path / "a.wav"), None, None, "small", None)
        assert raised.value.reason == "local_whisper_failed" and "secret" not in str(raised.value)

    def test_wrong_audio_format_is_refused(self, tmp_path):
        write_wav(tmp_path / "a.wav", 1, rate=44100)
        with pytest.raises(LocalWhisperError):
            local_whisper._read_wav(str(tmp_path / "a.wav"))

    def test_faster_whisper_is_preferred_when_present(self, monkeypatch):
        import importlib.util
        real = importlib.util.find_spec
        monkeypatch.setattr(importlib.util, "find_spec",
                            lambda name, *a: object() if name == "faster_whisper" else real(name, *a))
        assert local_whisper.detect_backend() == "faster-whisper"


class DeviceHarness:
    """A fake backend that records which device each load and transcription used."""

    def __init__(self, monkeypatch, backend="faster-whisper", cuda_present=True, gpu_failure=None):
        self.loads = []
        self.transcribes = []
        self.cuda_checks = 0
        self.gpu_failure = gpu_failure
        harness = self

        class Model:
            def __init__(self, use_cuda):
                self.use_cuda = use_cuda

            def transcribe(self, audio, **kwargs):
                harness.transcribes.append(self.use_cuda)
                if self.use_cuda and harness.gpu_failure is not None:
                    raise harness.gpu_failure
                if backend == "faster-whisper":
                    word = SimpleNamespace(word=" Hi.", start=0.1, end=0.4)
                    return iter([SimpleNamespace(text=" Hi.", words=[word])]), SimpleNamespace(language="en")
                return {"language": "en", "segments": [{"text": "Hi.", "no_speech_prob": 0, "avg_logprob": 0,
                                                       "words": [{"word": " Hi.", "start": 0.1, "end": 0.4}]}]}

        def load(backend_name, name, progress, use_cuda):
            harness.loads.append(use_cuda)
            return Model(use_cuda)

        def cuda_available(backend_name):
            harness.cuda_checks += 1
            return cuda_present

        monkeypatch.setattr(local_whisper, "detect_backend", lambda: backend)
        monkeypatch.setattr(local_whisper, "_load_model", load)
        monkeypatch.setattr(local_whisper, "_cuda_available", cuda_available)
        monkeypatch.setattr(local_whisper, "_cuda_disabled", False)
        local_whisper._models.clear()


GPU_LIBRARY_ERROR = RuntimeError("Library cublas64_12.dll is not found or cannot be loaded")


class TestGpuFallback:
    """In auto mode a GPU whose libraries are missing must not fail the job."""

    def test_missing_cuda_library_falls_back_to_cpu_and_succeeds(self, tmp_path, monkeypatch):
        write_wav(tmp_path / "a.wav", 2)
        messages = []
        h = DeviceHarness(monkeypatch, gpu_failure=GPU_LIBRARY_ERROR)
        out = local_whisper._run_builtin(str(tmp_path / "a.wav"), None, None, "small", messages.append)
        assert [w["word"] for w in out["words"]] == ["Hi."]
        assert h.transcribes == [True, False] and h.loads == [True, False]
        assert any("CPU" in m for m in messages)
        assert local_whisper._cuda_disabled is True

    def test_the_cpu_choice_is_remembered_for_later_chunks(self, tmp_path, monkeypatch):
        write_wav(tmp_path / "a.wav", 2)
        h = DeviceHarness(monkeypatch, gpu_failure=RuntimeError("cuDNN failed"))
        local_whisper._run_builtin(str(tmp_path / "a.wav"), None, None, "small", None)
        local_whisper._run_builtin(str(tmp_path / "a.wav"), None, None, "small", None)
        assert h.transcribes == [True, False, False]  # the second chunk never touches the GPU

    def test_a_gpu_failure_while_loading_the_model_also_falls_back(self, tmp_path, monkeypatch):
        write_wav(tmp_path / "a.wav", 2)
        h = DeviceHarness(monkeypatch)
        real_load = local_whisper._load_model

        def load(backend, name, progress, use_cuda):
            if use_cuda:
                h.loads.append(True)
                raise local_whisper._GpuFailure()
            return real_load(backend, name, progress, use_cuda)

        monkeypatch.setattr(local_whisper, "_load_model", load)
        out = local_whisper._run_builtin(str(tmp_path / "a.wav"), None, None, "small", None)
        assert out["words"] and h.transcribes == [False]

    def test_other_failures_are_not_retried_or_blamed_on_the_gpu(self, tmp_path, monkeypatch):
        write_wav(tmp_path / "a.wav", 2)
        h = DeviceHarness(monkeypatch, gpu_failure=ValueError("bad input"))
        with pytest.raises(LocalWhisperError) as raised:
            local_whisper._run_builtin(str(tmp_path / "a.wav"), None, None, "small", None)
        assert raised.value.reason == "local_whisper_failed" and h.transcribes == [True]
        assert local_whisper._cuda_disabled is False

    def test_no_cuda_device_means_cpu_without_any_error(self, tmp_path, monkeypatch):
        write_wav(tmp_path / "a.wav", 2)
        h = DeviceHarness(monkeypatch, cuda_present=False)
        out = local_whisper._run_builtin(str(tmp_path / "a.wav"), None, None, "small", None)
        assert out["words"] and h.loads == [False]


class TestDeviceModes:
    """The user's Device setting: auto, cpu or gpu."""

    def test_cpu_never_touches_cuda_even_when_it_would_fail(self, tmp_path, monkeypatch):
        write_wav(tmp_path / "a.wav", 2)
        h = DeviceHarness(monkeypatch, gpu_failure=GPU_LIBRARY_ERROR)
        out = local_whisper._run_builtin(str(tmp_path / "a.wav"), None, None, "small", None, device="cpu")
        assert out["words"]
        assert h.cuda_checks == 0 and h.loads == [False] and h.transcribes == [False]

    @pytest.mark.parametrize("backend", ["faster-whisper", "openai-whisper"])
    def test_cpu_forces_cpu_settings_for_both_backends(self, tmp_path, monkeypatch, backend):
        write_wav(tmp_path / "a.wav", 2)
        h = DeviceHarness(monkeypatch, backend=backend)
        local_whisper._run_builtin(str(tmp_path / "a.wav"), None, None, "small", None, device="cpu")
        assert h.cuda_checks == 0 and h.loads == [False]

    def test_cpu_does_not_use_the_remembered_gpu_failure_or_a_cached_gpu_model(self, tmp_path, monkeypatch):
        write_wav(tmp_path / "a.wav", 2)
        h = DeviceHarness(monkeypatch)
        local_whisper._models[("faster-whisper", "small", True)] = object()
        local_whisper._run_builtin(str(tmp_path / "a.wav"), None, None, "small", None, device="cpu")
        assert h.loads == [False]

    def test_gpu_with_a_working_gpu_uses_it(self, tmp_path, monkeypatch):
        write_wav(tmp_path / "a.wav", 2)
        h = DeviceHarness(monkeypatch)
        out = local_whisper._run_builtin(str(tmp_path / "a.wav"), None, None, "small", None, device="gpu")
        assert out["words"] and h.loads == [True] and h.transcribes == [True]

    def test_gpu_failure_is_an_error_not_a_silent_cpu_run(self, tmp_path, monkeypatch):
        write_wav(tmp_path / "a.wav", 2)
        h = DeviceHarness(monkeypatch, gpu_failure=GPU_LIBRARY_ERROR)
        with pytest.raises(LocalWhisperError) as raised:
            local_whisper._run_builtin(str(tmp_path / "a.wav"), None, None, "small", None, device="gpu")
        assert raised.value.reason == "local_whisper_gpu"
        assert h.transcribes == [True]  # never retried on the CPU
        assert "cublas" not in str(raised.value).lower()

    def test_gpu_failure_while_loading_is_also_an_error(self, tmp_path, monkeypatch):
        write_wav(tmp_path / "a.wav", 2)
        DeviceHarness(monkeypatch)

        def load(*args):
            raise local_whisper._GpuFailure()

        monkeypatch.setattr(local_whisper, "_load_model", load)
        with pytest.raises(LocalWhisperError) as raised:
            local_whisper._run_builtin(str(tmp_path / "a.wav"), None, None, "small", None, device="gpu")
        assert raised.value.reason == "local_whisper_gpu"

    def test_gpu_without_a_cuda_device_is_a_clear_error_before_loading_anything(self, tmp_path, monkeypatch):
        write_wav(tmp_path / "a.wav", 2)
        h = DeviceHarness(monkeypatch, cuda_present=False)
        with pytest.raises(LocalWhisperError) as raised:
            local_whisper._run_builtin(str(tmp_path / "a.wav"), None, None, "small", None, device="gpu")
        assert raised.value.reason == "local_whisper_gpu" and h.loads == []

    def test_gpu_mode_ignores_an_earlier_auto_fallback_flag(self, tmp_path, monkeypatch):
        write_wav(tmp_path / "a.wav", 2)
        h = DeviceHarness(monkeypatch)
        monkeypatch.setattr(local_whisper, "_cuda_disabled", True)
        local_whisper._run_builtin(str(tmp_path / "a.wav"), None, None, "small", None, device="gpu")
        assert h.loads == [True]

    def test_a_non_gpu_failure_in_gpu_mode_is_not_reported_as_a_gpu_problem(self, tmp_path, monkeypatch):
        write_wav(tmp_path / "a.wav", 2)
        DeviceHarness(monkeypatch, gpu_failure=ValueError("bad input"))
        with pytest.raises(LocalWhisperError) as raised:
            local_whisper._run_builtin(str(tmp_path / "a.wav"), None, None, "small", None, device="gpu")
        assert raised.value.reason == "local_whisper_failed"

    def test_unknown_device_values_behave_as_auto(self, tmp_path, monkeypatch):
        write_wav(tmp_path / "a.wav", 2)
        h = DeviceHarness(monkeypatch, gpu_failure=GPU_LIBRARY_ERROR)
        local_whisper._run_builtin(str(tmp_path / "a.wav"), None, None, "small", None, device="turbo-boost")
        assert h.transcribes == [True, False]

    def test_the_setting_reaches_the_backend_through_transcribe(self, tmp_path, monkeypatch):
        write_wav(tmp_path / "a.wav", 2)
        h = DeviceHarness(monkeypatch)
        settings = SimpleNamespace(local_whisper_mode="builtin", local_whisper_device="cpu")
        run(local_whisper.transcribe(settings, str(tmp_path / "a.wav"), None, None, "small"))
        assert h.cuda_checks == 0 and h.loads == [False]

    def test_models_are_cached_per_device(self, monkeypatch):
        calls = []

        class FakeModule:
            @staticmethod
            def load_model(name, device):
                calls.append(device)
                return object()

        monkeypatch.setitem(__import__("sys").modules, "whisper", FakeModule)
        local_whisper._models.clear()
        local_whisper._load_model("openai-whisper", "small", None, False)
        local_whisper._load_model("openai-whisper", "small", None, False)
        local_whisper._load_model("openai-whisper", "small", None, True)
        assert calls == ["cpu", "cuda"]
        local_whisper._models.clear()

    def test_real_backend_constructor_arguments(self, monkeypatch):
        seen = {}

        class FakeWhisperModel:
            def __init__(self, model_id, device, compute_type):
                seen.update(model_id=model_id, device=device, compute_type=compute_type)

        monkeypatch.setitem(__import__("sys").modules, "faster_whisper", SimpleNamespace(WhisperModel=FakeWhisperModel))
        local_whisper._models.clear()
        local_whisper._load_model("faster-whisper", "turbo", None, False)
        assert seen == {"model_id": "large-v3-turbo", "device": "cpu", "compute_type": "int8"}
        local_whisper._models.clear()
        local_whisper._load_model("faster-whisper", "small", None, True)
        assert seen == {"model_id": "small", "device": "cuda", "compute_type": "float16"}
        local_whisper._models.clear()

    def test_openai_whisper_decodes_in_fp32_on_cpu(self, tmp_path, monkeypatch):
        write_wav(tmp_path / "a.wav", 2)
        kwargs = {}

        class Model:
            def transcribe(self, audio, **k):
                kwargs.update(k)
                return {"language": "en", "segments": []}

        DeviceHarness(monkeypatch, backend="openai-whisper")
        monkeypatch.setattr(local_whisper, "_load_model", lambda *a: Model())
        local_whisper._run_builtin(str(tmp_path / "a.wav"), None, None, "small", None, device="cpu")
        assert kwargs["fp16"] is False


class FakeWhisperServer:
    """A local OpenAI-compatible transcription server for tests."""

    def __init__(self, status=200, body=None):
        outer = self
        self.requests = []
        self.status = status
        self.body = body if body is not None else {
            "text": "Hello world.", "language": "en",
            "words": [{"word": "Hello", "start": 0.1, "end": 0.5}, {"word": "world.", "start": 0.6, "end": 1.0}]}

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                outer.requests.append((self.path, self.rfile.read(length)))
                payload = json.dumps(outer.body).encode()
                self.send_response(outer.status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args):
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/v1"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def server():
    started = []

    def make(**kwargs):
        started.append(FakeWhisperServer(**kwargs))
        return started[-1]

    yield make
    for item in started:
        item.close()


def server_settings(url, **extra):
    return SimpleNamespace(local_whisper_mode="server", local_whisper_url=url, **extra)


class TestServerBackend:
    @pytest.mark.parametrize("url,expected", [
        ("http://h:8000/v1", "http://h:8000/v1/audio/transcriptions"),
        ("http://h:8000/v1/", "http://h:8000/v1/audio/transcriptions"),
        ("http://h:8000", "http://h:8000/v1/audio/transcriptions"),
        ("http://h:8000/v1/audio/transcriptions", "http://h:8000/v1/audio/transcriptions"),
    ])
    def test_endpoint_forms(self, url, expected):
        assert local_whisper.server_endpoint(url) == expected

    @pytest.mark.parametrize("url", ["", "ftp://h/v1", "http://u:p@h/v1", "garbage"])
    def test_bad_urls_are_refused(self, url):
        with pytest.raises(LocalWhisperError) as raised:
            local_whisper.server_endpoint(url)
        assert raised.value.reason == "local_whisper_unreachable"

    def test_transcribes_through_the_server(self, tmp_path, server):
        fake = server()
        write_wav(tmp_path / "a.wav", 3)
        out = run(local_whisper.transcribe(server_settings(fake.url), str(tmp_path / "a.wav"), "en", ["Zernio"], "whisper-1"))
        assert out["text"] == "Hello world." and [w["word"] for w in out["words"]] == ["Hello", "world."]
        path, body = fake.requests[0]
        assert path == "/v1/audio/transcriptions"
        for needle in (b'name="model"', b"verbose_json", b"timestamp_granularities[]", b"word", b'name="language"', b"Zernio"):
            assert needle in body

    def test_words_nested_in_segments_are_accepted(self, tmp_path, server):
        fake = server(body={"text": "Hi.", "segments": [{"words": [{"word": "Hi.", "start": 0.0, "end": 0.4}]}]})
        write_wav(tmp_path / "a.wav", 2)
        out = run(local_whisper.transcribe(server_settings(fake.url), str(tmp_path / "a.wav"), None, None, "m"))
        assert out["words"][0]["word"] == "Hi."

    def test_text_without_word_timestamps_is_a_specific_error(self, tmp_path, server):
        fake = server(body={"text": "Hello world.", "segments": [{"text": "Hello world.", "start": 0, "end": 1}]})
        write_wav(tmp_path / "a.wav", 2)
        with pytest.raises(LocalWhisperError) as raised:
            run(local_whisper.transcribe(server_settings(fake.url), str(tmp_path / "a.wav"), None, None, "m"))
        assert raised.value.reason == "local_whisper_no_words"

    def test_silence_is_not_an_error(self, tmp_path, server):
        fake = server(body={"text": "", "words": []})
        write_wav(tmp_path / "a.wav", 2)
        out = run(local_whisper.transcribe(server_settings(fake.url), str(tmp_path / "a.wav"), None, None, "m"))
        assert out["words"] == []

    @pytest.mark.parametrize("status,reason", [(404, "local_whisper_unreachable"), (500, "local_whisper_failed"),
                                               (401, "local_whisper_failed")])
    def test_http_errors(self, tmp_path, server, status, reason):
        fake = server(status=status, body={"error": "x"})
        write_wav(tmp_path / "a.wav", 1)
        with pytest.raises(LocalWhisperError) as raised:
            run(local_whisper.transcribe(server_settings(fake.url), str(tmp_path / "a.wav"), None, None, "m"))
        assert raised.value.reason == reason

    def test_nothing_listening_is_unreachable(self, tmp_path, server):
        fake = server()
        url = fake.url
        fake.close()
        write_wav(tmp_path / "a.wav", 1)
        with pytest.raises(LocalWhisperError) as raised:
            run(local_whisper.transcribe(server_settings(url), str(tmp_path / "a.wav"), None, None, "m"))
        assert raised.value.reason == "local_whisper_unreachable"


class TestServiceIntegration:
    """TranscriptionService with local transcription, end to end (Whisper itself faked)."""

    def make_service(self, **settings):
        service = ts.TranscriptionService.__new__(ts.TranscriptionService)
        service.settings = SimpleNamespace(
            transcription_source="local", local_whisper_mode="builtin", local_whisper_model="small",
            local_whisper_url="", openrouter_api_key=None, clipping_mode="quality", **settings)
        return service

    def test_local_run_needs_no_openrouter_key_and_reports_zero_cost(self, tmp_path, monkeypatch):
        write_wav(tmp_path / "a.wav", 6)
        calls = []

        async def fake(settings, path, language, keyterms, model, progress=None):
            calls.append((language, model))
            return {"text": "Hello there.", "language": "en", "words": [
                {"word": "Hello", "start": 1.0, "end": 1.4}, {"word": "there.", "start": 1.5, "end": 2.0}]}

        monkeypatch.setattr(local_whisper, "transcribe", fake)
        result = run(self.make_service().transcribe_audio(str(tmp_path / "a.wav"), timeline_offset_seconds=10.0))
        assert result.api_costs.provider == "local" and result.api_costs.estimated_cost_usd == 0.0
        assert result.api_costs.cost_incomplete is False and result.api_costs.model == "small"
        # Word times are shifted onto the source video's clock.
        assert result.segments[0].words[0].start_time_ms == 11000
        assert result.language == "en" and calls[0][1] == "small"

    @staticmethod
    def words(count, start=1.0):
        """`count` short words inside the core of every chunk, so none is dropped as overlap."""
        return [{"word": f"Wort{i}.", "start": start + i * 0.04, "end": start + i * 0.04 + 0.03} for i in range(count)]

    def test_later_chunks_reuse_the_detected_language(self, tmp_path, monkeypatch):
        write_wav(tmp_path / "a.wav", 12)
        monkeypatch.setattr(ts, "TRANSCRIPTION_CHUNK_SECONDS", 5)
        languages = []

        async def fake(settings, path, language, keyterms, model, progress=None):
            languages.append(language)
            return {"text": "w", "language": "de", "words": self.words(ts.MIN_WORDS_TO_LOCK_LANGUAGE)}

        monkeypatch.setattr(local_whisper, "transcribe", fake)
        result = run(self.make_service().transcribe_audio(str(tmp_path / "a.wav")))
        assert languages[0] is None and set(languages[1:]) == {"de"}
        # Every chunk's words were kept (each "Wort." ends a sentence, so each is a segment).
        assert len(result.segments) == ts.MIN_WORDS_TO_LOCK_LANGUAGE * len(languages)

    def test_a_chunk_with_little_speech_does_not_decide_the_language(self, tmp_path, monkeypatch):
        # A stream that opens with music: Whisper guesses a language from a few words.
        write_wav(tmp_path / "a.wav", 14)
        monkeypatch.setattr(ts, "TRANSCRIPTION_CHUNK_SECONDS", 5)
        languages = []
        replies = [("ja", 2), ("en", ts.MIN_WORDS_TO_LOCK_LANGUAGE), ("en", 5)]

        async def fake(settings, path, language, keyterms, model, progress=None):
            detected, count = replies[len(languages)]
            languages.append(language)
            return {"text": "w", "language": detected, "words": self.words(count)}

        monkeypatch.setattr(local_whisper, "transcribe", fake)
        result = run(self.make_service().transcribe_audio(str(tmp_path / "a.wav")))
        assert languages == [None, None, "en"]
        # The job reports the language it settled on, not the intro's guess.
        assert result.language == "en"

    def test_an_unsure_detection_does_not_decide_the_language(self, tmp_path, monkeypatch):
        # A sung intro decoded as Japanese can produce plenty of words; Whisper's own
        # confidence is what gives it away.
        write_wav(tmp_path / "a.wav", 14)
        monkeypatch.setattr(ts, "TRANSCRIPTION_CHUNK_SECONDS", 5)
        languages = []
        replies = [("ja", 0.42), ("en", 0.97), ("en", 0.99)]

        async def fake(settings, path, language, keyterms, model, progress=None):
            detected, probability = replies[len(languages)]
            languages.append(language)
            return {"text": "w", "language": detected, "language_probability": probability,
                    "words": self.words(ts.MIN_WORDS_TO_LOCK_LANGUAGE + 5)}

        monkeypatch.setattr(local_whisper, "transcribe", fake)
        result = run(self.make_service().transcribe_audio(str(tmp_path / "a.wav")))
        assert languages == [None, None, "en"] and result.language == "en"

    def test_a_chosen_language_is_used_for_every_chunk(self, tmp_path, monkeypatch):
        write_wav(tmp_path / "a.wav", 12)
        monkeypatch.setattr(ts, "TRANSCRIPTION_CHUNK_SECONDS", 5)
        languages = []

        async def fake(settings, path, language, keyterms, model, progress=None):
            languages.append(language)
            return {"text": "w", "language": "en", "words": self.words(30)}

        monkeypatch.setattr(local_whisper, "transcribe", fake)
        run(self.make_service(local_whisper_language="hi").transcribe_audio(str(tmp_path / "a.wav")))
        assert languages == ["hi", "hi", "hi"]

    def test_whisper_errors_become_transcription_errors_with_safe_text(self, tmp_path, monkeypatch):
        write_wav(tmp_path / "a.wav", 2)

        async def fake(*args, **kwargs):
            raise LocalWhisperError("internal path C:\\secret", "local_whisper_model")

        monkeypatch.setattr(local_whisper, "transcribe", fake)
        with pytest.raises(ts.TranscriptionError) as raised:
            run(self.make_service().transcribe_audio(str(tmp_path / "a.wav")))
        assert raised.value.reason == "local_whisper_model"
        assert safe_processing_error(raised.value) == "Local Whisper model could not be loaded"
        assert safe_failure_code(raised.value) == "transcription.local_whisper_model"

    def test_cloud_run_still_requires_the_openrouter_key(self, tmp_path):
        write_wav(tmp_path / "a.wav", 2)
        service = self.make_service()
        service.settings.transcription_source = "cloud"
        with pytest.raises(ts.TranscriptionProviderError):
            run(service.transcribe_audio(str(tmp_path / "a.wav")))


@pytest.mark.parametrize("reason,text", list(LOCAL_WHISPER_ERRORS.items()))
def test_every_local_reason_has_a_safe_message_and_code(reason, text):
    error = ts.TranscriptionError("detail with secrets", reason=reason)
    assert safe_processing_error(error) == text
    assert safe_job_error_text(text) == text
    assert safe_failure_code(error) == f"transcription.{reason}"


def test_device_setting_defaults_to_auto_and_rejects_unknown_values():
    from clip_engine.config import Settings

    assert Settings(_env_file=None).local_whisper_device == "auto"
    for value in ("auto", "cpu", "gpu"):
        assert Settings(_env_file=None, local_whisper_device=value).local_whisper_device == value
    with pytest.raises(Exception):
        Settings(_env_file=None, local_whisper_device="tpu")


def test_describe_environment_reports_backend_and_cuda(monkeypatch):
    monkeypatch.setattr(local_whisper, "detect_backend", lambda: "faster-whisper")
    monkeypatch.setattr(local_whisper, "_cuda_available", lambda backend: backend == "faster-whisper")
    assert local_whisper.describe_environment() == {"backend": "faster-whisper", "cuda": True}
    monkeypatch.setattr(local_whisper, "detect_backend", lambda: None)
    assert local_whisper.describe_environment() == {"backend": None, "cuda": False}


class TestGpuLibraries:
    def test_nvidia_pip_folders_are_added_on_windows(self, tmp_path, monkeypatch):
        for name in ("cublas", "cudnn"):
            (tmp_path / "nvidia" / name / "bin").mkdir(parents=True)
        (tmp_path / "nvidia" / "not-a-library").mkdir()
        added = []
        monkeypatch.setattr(local_whisper.sys, "platform", "win32")
        monkeypatch.setattr(local_whisper.sys, "path", ["", str(tmp_path / "missing"), str(tmp_path)])
        monkeypatch.setattr(local_whisper.os, "add_dll_directory", lambda folder: added.append(folder), raising=False)
        monkeypatch.setenv("PATH", "C:\\Windows")
        found = local_whisper.prepare_gpu_libraries()
        expected = [str(tmp_path / "nvidia" / "cublas" / "bin"), str(tmp_path / "nvidia" / "cudnn" / "bin")]
        assert found == expected and added == expected
        path = local_whisper.os.environ["PATH"].split(local_whisper.os.pathsep)
        assert path[:2] == expected and path[-1] == "C:\\Windows"
        # A second call does not add the folders to PATH again.
        local_whisper.prepare_gpu_libraries()
        assert local_whisper.os.environ["PATH"].split(local_whisper.os.pathsep).count(expected[0]) == 1

    def test_nothing_happens_elsewhere(self, tmp_path, monkeypatch):
        (tmp_path / "nvidia" / "cublas" / "bin").mkdir(parents=True)
        monkeypatch.setattr(local_whisper.sys, "platform", "linux")
        monkeypatch.setattr(local_whisper.sys, "path", [str(tmp_path)])
        assert local_whisper.prepare_gpu_libraries() == []


class TestEnglishOnlyModels:
    def test_an_english_only_model_is_always_told_english(self, tmp_path, monkeypatch):
        write_wav(tmp_path / "a.wav", 1)
        seen = []
        monkeypatch.setattr(local_whisper, "_run_builtin",
                            lambda path, language, prompt, model, progress, device: seen.append((model, language)) or {})
        settings = SimpleNamespace(local_whisper_mode="builtin", local_whisper_device="cpu")
        run(local_whisper.transcribe(settings, str(tmp_path / "a.wav"), "hi", None, "small.en"))
        run(local_whisper.transcribe(settings, str(tmp_path / "a.wav"), "hi", None, "small"))
        assert seen == [("small.en", "en"), ("small", "hi")]


def test_the_language_setting_accepts_whisper_codes_and_falls_back_to_auto():
    from clip_engine.config import Settings

    assert Settings(_env_file=None).local_whisper_language == "auto"
    assert Settings(_env_file=None, local_whisper_language="HI").local_whisper_language == "hi"
    for bad in ("en-US", "english", "", None, "e"):
        assert Settings(_env_file=None, local_whisper_language=bad).local_whisper_language == "auto"


class TestVocabularyReachesEveryWindow:
    def test_faster_whisper_gets_the_names_as_hotwords(self, tmp_path, monkeypatch):
        write_wav(tmp_path / "a.wav", 4)
        seen = {}

        class Model:
            def transcribe(self, audio, **kwargs):
                seen.update(kwargs)
                word = SimpleNamespace(word=" Zernio.", start=0.1, end=0.6)
                return iter([SimpleNamespace(text=" Zernio.", words=[word])]), SimpleNamespace(language="en")

        monkeypatch.setattr(local_whisper, "detect_backend", lambda: "faster-whisper")
        monkeypatch.setattr(local_whisper, "_load_model", lambda backend, name, progress, use_cuda: Model())
        monkeypatch.setattr(local_whisper, "_cuda_available", lambda backend: False)
        out = local_whisper._run_builtin(str(tmp_path / "a.wav"), "en", "Vocabulary: Zernio.", "small", None)
        assert out["text"] == "Zernio."
        # An initial prompt reaches only the first 30 seconds when windows do not
        # condition on each other; hotwords go before every window.
        assert seen["hotwords"] == "Vocabulary: Zernio."
        assert seen.get("initial_prompt") is None
        assert seen["condition_on_previous_text"] is False

    @pytest.mark.parametrize("supported", [True, False])
    def test_openai_whisper_carries_the_prompt_when_it_can(self, tmp_path, monkeypatch, supported):
        write_wav(tmp_path / "a.wav", 4)
        seen = {}
        result = {"language": "en", "segments": []}

        class New:
            def transcribe(self, audio, *, language=None, word_timestamps=False, initial_prompt=None,
                           carry_initial_prompt=False, condition_on_previous_text=True, fp16=True, verbose=None):
                seen.update(initial_prompt=initial_prompt, carry=carry_initial_prompt)
                return result

        class Old:
            def transcribe(self, audio, *, language=None, word_timestamps=False, initial_prompt=None,
                           condition_on_previous_text=True, fp16=True, verbose=None):
                seen.update(initial_prompt=initial_prompt, carry=None)
                return result

        model = New() if supported else Old()
        monkeypatch.setattr(local_whisper, "detect_backend", lambda: "openai-whisper")
        monkeypatch.setattr(local_whisper, "_load_model", lambda backend, name, progress, use_cuda: model)
        monkeypatch.setattr(local_whisper, "_cuda_available", lambda backend: False)
        local_whisper._run_builtin(str(tmp_path / "a.wav"), "en", "Vocabulary: Zernio.", "small", None)
        assert seen == {"initial_prompt": "Vocabulary: Zernio.", "carry": True if supported else None}


class TestPromptEcho:
    def test_a_segment_that_only_repeats_the_vocabulary_is_dropped(self, tmp_path, monkeypatch):
        write_wav(tmp_path / "a.wav", 4)

        def seg(text, start, end):
            return SimpleNamespace(text=text, words=[SimpleNamespace(word=w, start=start, end=end) for w in text.split()])

        class Model:
            def transcribe(self, audio, **kwargs):
                return iter([seg(" Vocabulary: Zernio.", 0.1, 0.5), seg(" Vocabulary matters to Zernio.", 1.0, 2.0)]), SimpleNamespace(language="en")

        monkeypatch.setattr(local_whisper, "detect_backend", lambda: "faster-whisper")
        monkeypatch.setattr(local_whisper, "_load_model", lambda backend, name, progress, use_cuda: Model())
        monkeypatch.setattr(local_whisper, "_cuda_available", lambda backend: False)
        out = local_whisper._run_builtin(str(tmp_path / "a.wav"), "en", "Vocabulary: Zernio.", "small", None)
        assert out["text"] == "Vocabulary matters to Zernio."
        assert local_whisper._echoes_prompt(" Vocabulary: Zernio.", None) is False
