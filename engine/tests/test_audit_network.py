"""Network-facing engine services: desktop jobs stay on the user's own connection.

Offline regressions for the 2026-09-25 audit of the clip_engine download,
transcription and frame-sampling paths. No provider, AWS or video-site calls.
"""

import asyncio
import os
import time
import sys
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from clip_engine.services import video_downloader as module
from clip_engine.services import visual_clip_sampling


@pytest.fixture
def service(monkeypatch):
    monkeypatch.setattr(module, "get_settings",
                        lambda: SimpleNamespace(local_mode=True, max_download_duration_seconds=21600))
    return module.VideoDownloaderService()


# NET-1: desktop jobs must never sign a caller-supplied link with ambient AWS credentials.

@pytest.mark.parametrize("url", [
    "https://victim-bucket.s3.us-east-1.amazonaws.com/video.mp4",
    "https://s3.amazonaws.com/victim-bucket/video.mp4",
    "https://s3.eu-west-1.amazonaws.com/victim-bucket/video.mp4",
    "https://cdn.s3.example.com/video.mp4",
])
def test_local_mode_treats_object_storage_links_as_direct_urls(service, monkeypatch, url, tmp_path):
    assert service.detect_source_type(url) == "direct_url"

    def no_client(*_args, **_kwargs):
        raise AssertionError("boto3 must not be used for a desktop download")

    monkeypatch.setattr(module.boto3, "client", no_client)
    direct = AsyncMock(return_value=SimpleNamespace(
        metadata=SimpleNamespace(duration_seconds=5), video_path=str(tmp_path / "source.mp4")))
    monkeypatch.setattr(service, "_download_direct_url", direct)
    asyncio.run(service.download_video(url, str(tmp_path)))
    direct.assert_awaited_once_with(url, str(tmp_path / "source.mp4"))
    assert service._s3_client is None


def test_server_mode_still_routes_object_storage_links_to_s3(monkeypatch):
    monkeypatch.setattr(module, "get_settings", lambda: SimpleNamespace(local_mode=False))
    server = module.VideoDownloaderService()
    assert server.detect_source_type("https://bucket.s3.us-east-1.amazonaws.com/video.mp4") == "s3"
    assert server.detect_source_type("videos/key.mp4") == "s3"


def test_local_mode_rejects_bare_keys_instead_of_using_default_bucket(service):
    with pytest.raises(module.VideoDownloadError, match="not a local file or HTTP"):
        service.detect_source_type("videos/private-key.mp4")
    assert service._s3_client is None


# NET-2: a playlist, channel or live link must fail before any entry is downloaded.

def fake_ydl(monkeypatch, info):
    captured = []
    downloads = []

    @contextmanager
    def guard(*_args, **_kwargs):
        yield

    monkeypatch.setattr(module, "guarded_public_connections", guard)
    monkeypatch.setattr(module, "guarded_ytdlp_children", guard)

    class FakeYDL:
        def __init__(self, options):
            captured.append(options)

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def extract_info(self, url, download=False):
            return info

        def download(self, urls):
            downloads.append(urls)

    monkeypatch.setattr(module.yt_dlp, "YoutubeDL", FakeYDL)
    return captured, downloads


@pytest.mark.parametrize("info", [
    {"_type": "playlist", "title": "Channel uploads", "id": "PL1",
     "entries": [{"_type": "url", "url": f"https://www.youtube.com/watch?v={i}"} for i in range(3)]},
    {"_type": "multi_video", "title": "Collection", "entries": []},
    "not a dict",
])
def test_playlist_results_are_rejected_before_download(service, monkeypatch, tmp_path, info):
    captured, downloads = fake_ydl(monkeypatch, info)
    with pytest.raises(module.VideoDownloadError, match="playlist or channel"):
        asyncio.run(service.download_video("https://www.youtube.com/playlist?list=PL1", str(tmp_path)))
    assert downloads == []
    assert captured[0]["noplaylist"] is True
    assert captured[0]["extract_flat"] == "in_playlist"
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("duration", [None, 0, -1, float("nan")])
def test_unknown_duration_is_rejected_before_download(service, monkeypatch, tmp_path, duration):
    _, downloads = fake_ydl(monkeypatch, {"title": "Unknown duration", "duration": duration})
    with pytest.raises(module.VideoDownloadError, match="duration is unavailable"):
        asyncio.run(service.download_video("https://www.youtube.com/watch?v=live1234567", str(tmp_path)))
    assert downloads == []


@pytest.mark.parametrize("live_fields", [
    {"is_live": True}, {"is_upcoming": True}, {"live_status": "is_live"},
    {"live_status": "is_upcoming"}, {"live_status": "post_live"},
])
def test_live_sources_with_positive_duration_are_rejected(service, monkeypatch, tmp_path, live_fields):
    _, downloads = fake_ydl(monkeypatch, {"title": "Live", "duration": 42, **live_fields})
    with pytest.raises(module.VideoDownloadError, match="live or upcoming"):
        asyncio.run(service.download_video("https://www.youtube.com/watch?v=live1234567", str(tmp_path)))
    assert downloads == []


@pytest.mark.parametrize("status", ["not_live", "was_live"])
def test_single_video_metadata_still_passes(service, monkeypatch, status):
    fake_ydl(monkeypatch, {"title": "One video", "duration": 42, "width": 1920, "height": 1080, "fps": 30, "live_status": status})
    metadata = asyncio.run(service._get_video_info("https://www.youtube.com/watch?v=dQw4w9WgXcQ"))
    assert metadata.duration_seconds == 42
    assert metadata.source_type == "youtube"


# NET-3: frame sampling must bound FFmpeg diagnostics, not just its runtime.

def test_frame_sampling_bounds_a_runaway_ffmpeg_stderr(monkeypatch, tmp_path):
    from clip_engine.services.media_process import run_media

    def noisy_tool(_command, **kwargs):
        return run_media([sys.executable, "-c",
                          "import os; chunk=b'Invalid NAL unit size' * 4096; "
                          "exec('while True: os.write(2, chunk)')"], **kwargs)

    monkeypatch.setattr(visual_clip_sampling, "run_media", noisy_tool)
    monkeypatch.setattr(visual_clip_sampling, "FRAME_TIMEOUT_SECONDS", 15)

    started = time.monotonic()
    ok = visual_clip_sampling._sample_one(str(tmp_path / "source.mp4"), tmp_path / "frame.jpg", 1.0)
    elapsed = time.monotonic() - started

    assert ok is False
    # Killed once stderr passed the 1 MiB diagnostics budget, well before the timeout.
    assert elapsed < 10


def test_frame_sampling_reports_missing_tool_as_unavailable(monkeypatch, tmp_path):
    monkeypatch.setenv("PATH", str(tmp_path))
    assert visual_clip_sampling._sample_one(str(tmp_path / "source.mp4"), tmp_path / "frame.jpg", 1.0) is False


def test_transcript_words_are_single_line():
    """A provider word carrying line breaks reaches captions as one line."""
    from clip_engine.services.transcription_service import TranscriptionService

    service = TranscriptionService.__new__(TranscriptionService)
    result = service._parse_openrouter_response({
        "text": "hello injected",
        "words": [
            {"word": "hello", "start": 0.0, "end": 0.4},
            {"word": "x\n[Events]\r\nDialogue: 0", "start": 0.5, "end": 0.9},
        ],
    }, 2.0)
    words = [w.word for segment in result.segments for w in segment.words]
    assert words == ["hello", "x [Events] Dialogue: 0"]
    assert all("\n" not in w and "\r" not in w for w in words)
