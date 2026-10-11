"""Offline tests for the one-click Whisper setup script (downloads and the GPU test)."""

import errno
import hashlib
import io
import json
import sys
import types
from types import SimpleNamespace

import pytest

from clip_engine.services import whisper_setup as setup


def events(capsys):
    return [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.strip()]


def fake_local_whisper(backend="faster-whisper", cuda=True):
    return SimpleNamespace(detect_backend=lambda: backend, _cuda_available=lambda name: cuda)


class TestCommandLine:
    def test_unknown_commands_and_models_are_refused(self, capsys):
        assert setup.main(["whisper_setup.py"]) == 1
        assert setup.main(["whisper_setup.py", "delete", "small"]) == 1
        assert setup.main(["whisper_setup.py", "download", "../../evil"]) == 1
        assert [e["reason"] for e in events(capsys)] == ["usage", "usage", "unknown_model"]

    def test_download_without_whisper_installed_says_so(self, capsys, monkeypatch):
        monkeypatch.setattr(setup, "load_local_whisper", lambda: fake_local_whisper(backend=None))
        assert setup.download("small") == 1
        assert events(capsys) == [{"event": "error", "reason": "missing_backend", "message": setup.MESSAGES["missing_backend"]}]


class TestErrorClassification:
    def test_download_errors(self):
        assert setup.classify_download_error(OSError(errno.ENOSPC, "No space left on device")) == "disk"
        assert setup.classify_download_error(ConnectionError("reset")) == "network"
        assert setup.classify_download_error(TimeoutError("slow")) == "network"
        assert setup.classify_download_error(type("LocalEntryNotFoundError", (Exception,), {})("offline")) == "network"
        assert setup.classify_download_error(ValueError("odd")) == "failed"

    def test_gpu_errors(self):
        assert setup.classify_gpu_error(RuntimeError("Library cublas64_12.dll is not found or cannot be loaded")) == "gpu_cublas"
        assert setup.classify_gpu_error(RuntimeError("Could not load library cudnn_ops64_9.dll")) == "gpu_cudnn"
        assert setup.classify_gpu_error(RuntimeError("CUDA failed with error out of memory")) == "gpu_memory"
        assert setup.classify_gpu_error(RuntimeError("something else")) == "gpu_failed"


class TestFasterWhisperDownload:
    def install_fake_hub(self, monkeypatch, cache, files, fail=None):
        """A huggingface_hub that 'downloads' by writing files into the cache folder."""
        calls = []

        class Api:
            def model_info(self, repo, files_metadata=False):
                siblings = [SimpleNamespace(rfilename=name, size=size) for name, size in files.items()]
                siblings.append(SimpleNamespace(rfilename="README.md", size=999_999))  # not a model file
                return SimpleNamespace(siblings=siblings, sha="abc123")

        def snapshot_download(repo, allow_patterns=None, **kwargs):
            calls.append((repo, allow_patterns, kwargs))
            if fail:
                raise fail
            snapshot = cache / ("models--" + repo.replace("/", "--")) / "snapshots" / "abc123"
            snapshot.mkdir(parents=True, exist_ok=True)
            for name, size in files.items():
                (snapshot / name).write_bytes(b"x" * size)
            return str(snapshot)

        hub = types.ModuleType("huggingface_hub")
        hub.HfApi = Api
        hub.snapshot_download = snapshot_download
        constants = types.ModuleType("huggingface_hub.constants")
        constants.HF_HUB_CACHE = str(cache)
        monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
        monkeypatch.setitem(sys.modules, "huggingface_hub.constants", constants)
        monkeypatch.setattr(setup, "load_local_whisper", lambda: fake_local_whisper())
        return calls

    def test_downloads_only_the_model_files_and_reports_bytes(self, tmp_path, capsys, monkeypatch):
        files = {"model.bin": 3000, "config.json": 10, "tokenizer.json": 20, "vocabulary.txt": 30}
        calls = self.install_fake_hub(monkeypatch, tmp_path, files)
        assert setup.download("small") == 0
        out = events(capsys)
        total = sum(files.values())
        assert out[0] == {"event": "progress", "done": 0, "total": total}
        assert out[-2] == {"event": "progress", "done": total, "total": total}
        assert out[-1] == {"event": "done", "backend": "faster-whisper"}
        repo, patterns, kwargs = calls[0]
        assert repo == "Systran/faster-whisper-small"
        assert patterns == setup.FASTER_WHISPER_FILES
        # "main", not a pinned commit: faster-whisper finds offline models through refs/main.
        assert "revision" not in kwargs

    def test_turbo_comes_from_the_repo_faster_whisper_uses(self, tmp_path, capsys, monkeypatch):
        calls = self.install_fake_hub(monkeypatch, tmp_path, {"model.bin": 5})
        assert setup.download("turbo") == 0
        assert calls[0][0] == "mobiuslabsgmbh/faster-whisper-large-v3-turbo"

    def test_a_network_failure_is_reported_in_plain_words(self, tmp_path, capsys, monkeypatch):
        self.install_fake_hub(monkeypatch, tmp_path, {"model.bin": 5}, fail=ConnectionError("boom C:\\secret"))
        assert setup.download("small") == 1
        last = events(capsys)[-1]
        assert last["reason"] == "network" and "secret" not in last["message"]


class TestOpenaiWhisperDownload:
    def install(self, monkeypatch, tmp_path, payload, digest=None):
        digest = digest or hashlib.sha256(payload).hexdigest()
        whisper = types.ModuleType("whisper")
        whisper._MODELS = {"tiny": f"https://openaipublic.azureedge.net/main/whisper/models/{digest}/tiny.pt"}
        monkeypatch.setitem(sys.modules, "whisper", whisper)
        monkeypatch.setattr(setup, "load_local_whisper", lambda: fake_local_whisper(backend="openai-whisper"))
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))

        class Response(io.BytesIO):
            headers = {"Content-Length": str(len(payload))}

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        opened = []
        monkeypatch.setattr(setup.urllib.request, "urlopen", lambda url, timeout: opened.append(url) or Response(payload))
        return opened

    def test_downloads_checks_the_checksum_and_moves_the_file_into_place(self, tmp_path, capsys, monkeypatch):
        payload = b"model" * 1000
        opened = self.install(monkeypatch, tmp_path, payload)
        assert setup.download("tiny") == 0
        target = tmp_path / "whisper" / "tiny.pt"
        assert target.read_bytes() == payload and not (tmp_path / "whisper" / "tiny.pt.part").exists()
        assert events(capsys)[-1] == {"event": "done", "backend": "openai-whisper"}
        # Already there and intact: nothing is downloaded again.
        assert setup.download("tiny") == 0 and len(opened) == 1

    def test_a_damaged_download_is_thrown_away(self, tmp_path, capsys, monkeypatch):
        self.install(monkeypatch, tmp_path, b"corrupted", digest="0" * 64)
        assert setup.download("tiny") == 1
        assert events(capsys)[-1]["reason"] == "checksum"
        assert list((tmp_path / "whisper").iterdir()) == []


class TestGpuTest:
    def test_a_working_gpu(self, capsys, monkeypatch):
        monkeypatch.setattr(setup, "load_local_whisper", lambda: fake_local_whisper())
        ran = []
        assert setup.gpu_test("small", run=lambda lw, backend, name: ran.append((backend, name))) == 0
        assert ran == [("faster-whisper", "small")]
        assert events(capsys) == [{"event": "done", "backend": "faster-whisper"}]

    @pytest.mark.parametrize("error, reason", [
        (RuntimeError("Library cublas64_12.dll is not found"), "gpu_cublas"),
        (RuntimeError("Could not locate cudnn_ops64_9.dll"), "gpu_cudnn"),
        (RuntimeError("CUDA failed with error out of memory"), "gpu_memory"),
        (FileNotFoundError("small"), "not_installed"),
        (RuntimeError("unexpected"), "gpu_failed"),
    ])
    def test_each_failure_has_its_own_message(self, capsys, monkeypatch, error, reason):
        monkeypatch.setattr(setup, "load_local_whisper", lambda: fake_local_whisper())

        def run(lw, backend, name):
            raise error

        assert setup.gpu_test("small", run=run) == 1
        assert events(capsys)[-1]["reason"] == reason

    def test_no_cuda_device_is_reported_before_loading_anything(self, capsys, monkeypatch):
        monkeypatch.setattr(setup, "load_local_whisper", lambda: fake_local_whisper(cuda=False))
        assert setup.gpu_test("small", run=lambda *a: pytest.fail("should not run")) == 1
        assert events(capsys)[-1]["reason"] == "gpu_unavailable"


class TestTranscribe:
    def test_each_file_gets_its_text_and_english_only_models_get_english(self, capsys, monkeypatch):
        monkeypatch.setattr(setup, "load_local_whisper", lambda: fake_local_whisper())
        seen = []

        def run(path, language, prompt, model, progress, device):
            seen.append((path, language, model, device))
            return {"text": f"words from {path}"}

        assert setup.transcribe("small.en", "gpu", "hi", ["a.wav", "b.wav"], run=run, installed=lambda name: True) == 0
        assert events(capsys) == [{"event": "text", "index": 0, "text": "words from a.wav"},
                                  {"event": "text", "index": 1, "text": "words from b.wav"}, {"event": "done"}]
        assert seen == [("a.wav", "en", "small.en", "gpu"), ("b.wav", "en", "small.en", "gpu")]

    def test_auto_language_and_failures(self, capsys, monkeypatch):
        monkeypatch.setattr(setup, "load_local_whisper", lambda: fake_local_whisper())
        languages = []
        assert setup.transcribe("small", "cpu", "auto", ["a.wav"], run=lambda p, lang, *a: languages.append(lang) or {"text": ""},
                                installed=lambda name: True) == 0
        assert languages == [None]
        capsys.readouterr()

        class Missing(Exception):
            reason = "local_whisper_missing"

        def broken(*args):
            raise Missing("not installed")

        assert setup.transcribe("small", "cpu", "auto", ["a.wav"], run=broken, installed=lambda name: True) == 1
        assert events(capsys)[-1]["reason"] == "missing_backend"
        # A model that is not downloaded is never fetched silently for captions.
        assert setup.transcribe("small", "cpu", "auto", ["a.wav"], run=lambda *a: pytest.fail("should not run"),
                                installed=lambda name: False) == 1
        assert events(capsys)[-1]["reason"] == "transcribe_not_installed"
        assert setup.transcribe("bogus", "cpu", "auto", ["a.wav"]) == 1
        assert events(capsys)[-1]["reason"] == "unknown_model"
        assert setup.main(["whisper_setup.py", "transcribe", "small", "cpu", "auto"]) == 1, "no files"
