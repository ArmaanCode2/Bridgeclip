"""Export quality: x264 medium CRF 18, NVENC with a CPU fallback, source frame rates, two-pass loudness."""

import asyncio
from types import SimpleNamespace

import pytest

from clip_engine.config import Settings
from clip_engine.services import rendering_service as module
from clip_engine.services.rendering_service import RenderingError, RenderingService
from tests.test_render_fallback import render, request_for, service  # noqa: F401  (fixture)


def local_service(encoder="libx264", gpu=None, **settings):
    svc = RenderingService.__new__(RenderingService)
    svc.settings = Settings(_env_file=None, local_mode=True, **settings)
    svc._local_cpu_encoder = encoder
    svc._gpu_encoder = gpu
    return svc


class TestEncoderSettings:
    def test_x264_is_medium_at_crf_18(self, monkeypatch):
        monkeypatch.setattr(module.sys, "platform", "win32")
        args = local_service()._video_codec_args(1080, 1920, "60")
        assert args[:6] == ["-c:v", "libx264", "-preset", "medium", "-crf", "18"]
        assert args[args.index("-g") + 1] == "120"

    def test_nvenc_uses_constant_quality_with_lookahead_and_b_frames(self, monkeypatch):
        monkeypatch.setattr(module.sys, "platform", "win32")
        args = local_service(gpu="h264_nvenc")._video_codec_args(1080, 1920, "30")
        assert args[:2] == ["-c:v", "h264_nvenc"]
        for flag, value in (("-rc", "vbr"), ("-cq", "19"), ("-b:v", "0"), ("-profile:v", "high"), ("-bf", "3"), ("-g", "60")):
            assert args[args.index(flag) + 1] == value

    def test_macos_keeps_videotoolbox_whatever_the_setting(self, monkeypatch):
        monkeypatch.setattr(module.sys, "platform", "darwin")
        args = local_service(gpu="h264_nvenc", video_encoder="nvidia")._video_codec_args(1080, 1920, "30")
        assert args[1] == "h264_videotoolbox"

    def test_the_setting_is_read_safely(self):
        assert Settings(_env_file=None).video_encoder == "cpu"
        assert Settings(_env_file=None, video_encoder="NVIDIA").video_encoder == "nvidia"
        assert Settings(_env_file=None, video_encoder="quantum").video_encoder == "cpu"


class TestNvencDetection:
    def detect(self, monkeypatch, listed=True, works=True, choice="nvidia"):
        monkeypatch.setattr(module.sys, "platform", "win32")
        monkeypatch.setattr(module.shutil, "which", lambda _: "ffmpeg")
        calls = []

        def run_media(cmd, **kwargs):
            calls.append(cmd)
            if "-encoders" in cmd:
                return SimpleNamespace(stdout=(b"libx264 " + (b"h264_nvenc" if listed else b"")), returncode=0)
            return SimpleNamespace(stdout=b"", stderr=b"", returncode=0 if works else 1)

        monkeypatch.setattr(module, "run_media", run_media)
        svc = RenderingService.__new__(RenderingService)
        svc.settings = Settings(_env_file=None, local_mode=True, video_encoder=choice)
        svc._verify_ffmpeg()
        return svc, calls

    def test_a_working_nvidia_encoder_is_used(self, monkeypatch):
        svc, calls = self.detect(monkeypatch)
        assert svc._gpu_encoder == "h264_nvenc" and svc._video_codec_args()[1] == "h264_nvenc"
        # Listing is not proof: one frame is really encoded first.
        assert any("h264_nvenc" in cmd and "-frames:v" in cmd for cmd in calls)

    @pytest.mark.parametrize("listed, works", [(True, False), (False, True)])
    def test_a_missing_or_broken_encoder_falls_back_to_the_cpu(self, monkeypatch, listed, works):
        svc, calls = self.detect(monkeypatch, listed=listed, works=works)
        assert svc._gpu_encoder is None and svc._video_codec_args()[1] == "libx264"
        if not listed:
            assert len(calls) == 1, "no test encode without NVENC in the build"

    def test_the_cpu_choice_never_touches_the_gpu(self, monkeypatch):
        svc, calls = self.detect(monkeypatch, choice="cpu")
        assert svc._gpu_encoder is None and len(calls) == 1


class TestNvencFailureRetriesOnTheCpu:
    def test_a_failed_gpu_encode_is_redone_on_the_cpu_and_the_gpu_is_dropped(self, monkeypatch):
        monkeypatch.setattr(module.sys, "platform", "win32")
        svc = local_service(gpu="h264_nvenc")
        runs, cleaned = [], []

        async def run_cmd(cmd, **kwargs):
            runs.append(cmd)
            if "h264_nvenc" in cmd:
                raise RenderingError("FFmpeg failed: OpenEncodeSessionEx failed: out of memory")

        monkeypatch.setattr(svc, "_run_cmd", run_cmd)
        asyncio.run(svc._run_encode(lambda: ["ffmpeg", *svc._video_codec_args(), "out.mp4"], cleanup=lambda: cleaned.append(1)))
        assert [cmd[2] for cmd in runs] == ["h264_nvenc", "libx264"]
        assert cleaned == [1] and svc._gpu_encoder is None

    def test_a_cpu_failure_is_not_retried(self, monkeypatch):
        svc = local_service()
        runs = []

        async def run_cmd(cmd, **kwargs):
            runs.append(cmd)
            raise RenderingError("FFmpeg failed: bad graph")

        monkeypatch.setattr(svc, "_run_cmd", run_cmd)
        with pytest.raises(RenderingError):
            asyncio.run(svc._run_encode(lambda: ["ffmpeg", "-c:v", "libx264", "out.mp4"]))
        assert len(runs) == 1

    def test_editor_previews_cap_the_nvenc_bitrate_without_dropping_constant_quality(self, monkeypatch, tmp_path):
        monkeypatch.setattr(module.sys, "platform", "win32")
        svc = local_service(gpu="h264_nvenc")
        seen = []

        async def dims(_):
            return 1920, 1080

        async def fps(_):
            return "60"

        async def run_cmd(cmd, **kwargs):
            seen.append(cmd)
            open(cmd[-1], "wb").write(b"mp4")

        monkeypatch.setattr(svc, "_get_video_dimensions", dims)
        monkeypatch.setattr(svc, "_probe_fps", fps)
        monkeypatch.setattr(svc, "_run_cmd", run_cmd)
        asyncio.run(svc.capture_framing_source("in.mp4", str(tmp_path / "preview.mp4")))
        [cmd] = seen
        assert cmd[cmd.index("-c:v") + 1] == "h264_nvenc" and cmd[cmd.index("-b:v") + 1] == "0"
        assert cmd[cmd.index("-maxrate") + 1] == "4.5M" and "-n" in cmd


class TestFrameRateAndLoudness:
    def capture(self, monkeypatch, svc):
        seen = []

        async def render_edit(request, plan, time_map, *args):
            seen.append(args)
            open(request.output_path, "wb").write(b"\0" * 1024)

        async def probe(_):
            return "60"

        async def has_audio(_):
            return True

        async def measure(path, start_ms, duration_ms, keeps=None):
            seen.append(("measured", keeps))
            return "loudnorm=linear"

        monkeypatch.setattr(svc, "_render_edit", render_edit)
        monkeypatch.setattr(svc, "_probe_fps", probe)
        monkeypatch.setattr(svc, "_has_audio", has_audio)
        monkeypatch.setattr(svc, "_measure_loudness", measure)
        return seen

    def test_vertical_clips_keep_a_60_fps_source_and_level_volume_in_two_passes(self, service, monkeypatch, tmp_path):  # noqa: F811
        seen = self.capture(monkeypatch, service)
        render(service, request_for(tmp_path))
        [(label, keeps), (_, _, width, height, landscape, fps, loudness)] = seen
        assert label == "measured" and keeps, "the kept audio is measured"
        assert (width, height, landscape) == (1080, 1920, False)
        assert fps == "60" and loudness == "loudnorm=linear"

    def test_an_unreadable_audio_check_leaves_the_render_to_report_it(self, service, monkeypatch, tmp_path):  # noqa: F811
        seen = self.capture(monkeypatch, service)

        async def broken(_):
            raise RenderingError("Could not verify the source audio track")

        monkeypatch.setattr(service, "_has_audio", broken)
        render(service, request_for(tmp_path))
        assert seen[0][-1] is None, "single-pass loudness, and the render still runs"

    def test_no_audio_means_no_measurement(self, service, monkeypatch, tmp_path):  # noqa: F811
        seen = self.capture(monkeypatch, service)
        render(service, request_for(tmp_path, include_audio=False))
        assert seen[0][-1] is None


class TestReviewFindings:
    def test_a_non_encoder_failure_under_nvenc_is_not_retried_and_keeps_the_gpu(self, monkeypatch):
        monkeypatch.setattr(module.sys, "platform", "win32")
        svc = local_service(gpu="h264_nvenc")
        runs = []

        async def run_cmd(cmd, **kwargs):
            runs.append(cmd)
            raise RenderingError("FFmpeg failed: [AVFilterGraph @ 0x1] No such filter: 'bogus'")

        monkeypatch.setattr(svc, "_run_cmd", run_cmd)
        with pytest.raises(RenderingError):
            asyncio.run(svc._run_encode(lambda: ["ffmpeg", *svc._video_codec_args(), "out.mp4"]))
        assert len(runs) == 1 and svc._gpu_encoder == "h264_nvenc"

    def test_the_progress_bar_never_moves_back_on_a_cpu_retry(self, monkeypatch):
        monkeypatch.setattr(module.sys, "platform", "win32")
        svc = local_service(gpu="h264_nvenc")
        shown = []

        async def run_cmd(cmd, progress=None, duration_ms=None):
            for percent in (0, 40):
                progress(percent)
            if "h264_nvenc" in cmd:
                raise RenderingError("FFmpeg failed: OpenEncodeSessionEx failed: incompatible client key (21)")
            progress(80)

        monkeypatch.setattr(svc, "_run_cmd", run_cmd)
        asyncio.run(svc._run_encode(lambda: ["ffmpeg", *svc._video_codec_args(), "out.mp4"],
                                    progress=shown.append, duration_ms=1000))
        assert shown == [0, 40, 80]

    def test_loudness_is_measured_on_the_kept_audio_only(self, monkeypatch):
        commands = []

        def run_media(cmd, **kwargs):
            commands.append(cmd)
            raise RuntimeError("stop after building the command")

        monkeypatch.setattr(module, "run_media", run_media)
        svc = RenderingService.__new__(RenderingService)
        assert asyncio.run(svc._measure_loudness("in.mp4", 10_000, 90_000, [(0, 15_000), (75_000, 90_000)])) is None
        audio_filter = commands[0][commands[0].index("-af") + 1]
        assert "aselect='between(t,0.000,15.000)+between(t,75.000,90.000)',asetpts=N/SR/TB,loudnorm=" in audio_filter
        # Nothing cut: the whole window, as before.
        asyncio.run(svc._measure_loudness("in.mp4", 0, 5_000, [(0, 5_000)]))
        assert "aselect" not in commands[1][commands[1].index("-af") + 1]

    @pytest.mark.parametrize("measured, linear", [
        ({"input_i": "-24", "input_tp": "-16", "input_lra": "6"}, True),
        ({"input_i": "-60", "input_tp": "-50", "input_lra": "6"}, False),   # +46 dB would raise the noise floor
        ({"input_i": "-24", "input_tp": "-16", "input_lra": "0"}, False),   # loudnorm refuses linear at LRA 0
    ])
    def test_the_gain_cap_and_a_zero_range_take_the_steady_gain(self, measured, linear):
        from clip_engine.services.layout_renderer import measured_loudness_filter
        f = measured_loudness_filter({**measured, "input_thresh": "-34", "target_offset": "0"})
        assert ("linear=true" in f) is linear
        if not linear:
            assert f.startswith("volume=") and float(f.split("=")[1].split("dB")[0]) <= 20.0

    def test_variable_rate_recordings_averaging_under_23_fps_stay_at_30(self, service, monkeypatch, tmp_path):  # noqa: F811
        seen = TestFrameRateAndLoudness().capture(monkeypatch, service)

        async def low(_):
            return "15"

        monkeypatch.setattr(service, "_probe_fps", low)
        render(service, request_for(tmp_path))
        assert seen[-1][5] == "30"

    def test_previews_keep_the_fast_preset_and_drop_a_stale_partial(self, monkeypatch, tmp_path):
        monkeypatch.setattr(module.sys, "platform", "win32")
        svc = local_service()
        stale = tmp_path / "preview.mp4.partial.mp4"
        stale.write_bytes(b"stale")
        seen = []

        async def dims(_):
            return 1920, 1080

        async def fps(_):
            return "30"

        async def run_cmd(cmd, **kwargs):
            seen.append(cmd)
            assert not stale.exists(), "removed before encoding"
            open(cmd[-1], "wb").write(b"fresh")

        monkeypatch.setattr(svc, "_get_video_dimensions", dims)
        monkeypatch.setattr(svc, "_probe_fps", fps)
        monkeypatch.setattr(svc, "_run_cmd", run_cmd)
        asyncio.run(svc.capture_framing_source("in.mp4", str(tmp_path / "preview.mp4")))
        [cmd] = seen
        assert cmd[cmd.index("-preset") + 1] == "veryfast"
        assert (tmp_path / "preview.mp4").read_bytes() == b"fresh"
