"""
Continuing a run from the transcript an earlier run saved: the planner gets the
same transcript, nothing is transcribed, and a transcript that cannot stand in
for this run's stops the job with a fixed message instead of guessing.
"""

import asyncio
import json
import os
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from clip_engine.error_policy import SAVED_TRANSCRIPT_ERROR, safe_failure_code, safe_job_error_text, safe_processing_error
from clip_engine.services import ai_clipping_pipeline as pipeline_module
from clip_engine.services.ai_client import AiClientError
from clip_engine.services.ai_clipping_pipeline import AIClippingPipeline, ClippingJobRequest, JobStatus
from clip_engine.services.intelligence_planner import ClipPlanResponse, ClipPlanSegment
from clip_engine.services.rendering_service import RenderingService, RenderResult
from clip_engine.services.transcription_service import (
    SavedTranscriptError,
    TranscriptionResult,
    TranscriptSegment,
    TranscriptWord,
    load_saved_transcript,
)


def segments():
    return [
        TranscriptSegment(0, 1200, "Hello there.", speaker_label="A",
                          words=[TranscriptWord("Hello", 0, 500), TranscriptWord("there.", 520, 1200)]),
        TranscriptSegment(61_000, 62_000, "Later.", words=[TranscriptWord("Later.", 61_000, 62_000)],
                          audio_events=["(laughter)"]),
    ]


def write_transcript(path, source_range=(None, None), status="available", rows=None, duration=300.0):
    rows = segments() if rows is None else rows
    data = {"segments": [asdict(s) for s in rows], "full_text": "Hello there. Later.", "language": "en",
            "status": status, "captions_available": bool(rows), "source_range": list(source_range),
            "source_duration_seconds": duration}
    path.write_text(json.dumps(data), encoding="utf-8")
    return str(path)


def test_round_trip_keeps_every_word_speaker_and_event(tmp_path):
    result, status, covered = load_saved_transcript(write_transcript(tmp_path / "transcript.json"), (None, None), 300.0)
    assert (status, covered) == ("available", (None, None))
    assert result.segments == segments()
    assert result.full_text == "Hello there. Later." and result.language == "en"
    # The earlier run already paid for it; this run reports no transcription cost.
    assert result.api_costs is None


@pytest.mark.parametrize("saved, wanted, usable", [
    ((None, None), (None, None), True),
    ((None, None), (30.0, 90.0), True),        # a full transcript covers any range
    ((30.0, 90.0), (30.0, 90.0), True),
    ((30.0, 90.0), (30.004, 89.996), True),    # float noise
    ((30.0, 90.0), (None, None), False),       # Jev now needs the whole source
    ((30.0, 90.0), (20.0, 90.0), False),
    ((30.0, 90.0), (30.0, 120.0), False),
    ((0.0, 300.0), (None, None), True),        # ends at the video's end
    ((None, 120.0), (None, None), False),
])
def test_the_saved_range_must_cover_what_this_run_needs(tmp_path, saved, wanted, usable):
    path = write_transcript(tmp_path / "transcript.json", saved)
    if usable:
        assert load_saved_transcript(path, wanted, 300.0)[0].segments == segments()
    else:
        with pytest.raises(SavedTranscriptError):
            load_saved_transcript(path, wanted, 300.0)


@pytest.mark.parametrize("field", ["source_range", "source_duration_seconds"])
def test_an_older_transcript_without_its_range_or_length_is_not_guessed_at(tmp_path, field):
    path = tmp_path / "transcript.json"
    write_transcript(path)
    data = json.loads(path.read_text())
    del data[field]
    path.write_text(json.dumps(data))
    with pytest.raises(SavedTranscriptError):
        load_saved_transcript(str(path), (None, None), 300.0)


@pytest.mark.parametrize("duration, usable", [(300.0, True), (300.9, True), (299.2, True), (301.5, False), (240.0, False)])
def test_a_transcript_of_another_version_of_the_video_is_refused(tmp_path, duration, usable):
    # A file re-exported to the same path, or a re-uploaded link, usually changes length.
    path = write_transcript(tmp_path / "transcript.json", duration=duration)
    if usable:
        load_saved_transcript(path, (None, None), 300.0)
    else:
        with pytest.raises(SavedTranscriptError, match="different version"):
            load_saved_transcript(path, (None, None), 300.0)


def test_a_saved_silent_source_stays_silent(tmp_path):
    path = write_transcript(tmp_path / "transcript.json", status="no_speech", rows=[])
    result, status, _ = load_saved_transcript(path, (None, None), 300.0)
    assert (result.segments, status) == ([], "no_speech")


def test_a_quiet_part_of_a_full_transcript_is_reported_as_no_speech(tmp_path):
    # Speech at 0-1.2 s and 61-62 s only, so a fresh run of 5-60 s would plan from frames too.
    path = write_transcript(tmp_path / "transcript.json")
    assert load_saved_transcript(path, (5.0, 60.0), 300.0)[1] == "no_speech"
    assert load_saved_transcript(path, (5.0, 61.5), 300.0)[1] == "available"
    assert load_saved_transcript(path, (5.0, 60.0), 300.0)[2] == (None, None)


@pytest.mark.parametrize("mutate", [
    lambda d: d.update(segments="nope"),
    lambda d: d.update(status="maybe"),
    lambda d: d["segments"][0].update(start_time_ms=-1),
    lambda d: d["segments"][0].update(start_time_ms=1.5),
    lambda d: d["segments"][0].update(end_time_ms=-5),
    lambda d: d["segments"][0].update(text=None),
    lambda d: d["segments"][0].update(speaker_label=7),
    lambda d: d["segments"][0]["words"][0].update(end_time_ms=None),
    lambda d: d["segments"][0]["words"].append("word"),
    lambda d: d["segments"][1].update(audio_events=[3]),
    lambda d: d.update(language=["en"]),
    lambda d: d.update(source_range=["0", None]),
    lambda d: d.update(source_range=[float("nan"), None]),
    lambda d: d.update(source_duration_seconds="300"),
])
def test_a_damaged_transcript_is_refused(tmp_path, mutate):
    path = tmp_path / "transcript.json"
    write_transcript(path)
    data = json.loads(path.read_text())
    mutate(data)
    path.write_text(json.dumps(data))
    with pytest.raises(SavedTranscriptError):
        load_saved_transcript(str(path), (None, None), 300.0)


def test_unreadable_or_missing_files_are_refused(tmp_path):
    for path in (tmp_path / "missing.json", tmp_path):
        with pytest.raises(SavedTranscriptError):
            load_saved_transcript(str(path), (None, None), 300.0)
    (tmp_path / "transcript.json").write_text("{not json")
    with pytest.raises(SavedTranscriptError):
        load_saved_transcript(str(tmp_path / "transcript.json"), (None, None), 300.0)


def test_failure_is_reported_with_fixed_text():
    error = SavedTranscriptError("Saved transcript covers a different part of the video")
    assert safe_processing_error(error) == SAVED_TRANSCRIPT_ERROR
    assert safe_failure_code(error) == "transcription.saved_transcript_unusable"
    assert safe_job_error_text(SAVED_TRANSCRIPT_ERROR) == SAVED_TRANSCRIPT_ERROR


@pytest.fixture
def local_pipeline(monkeypatch, tmp_path):
    from unittest.mock import AsyncMock
    monkeypatch.setattr(pipeline_module.CoherenceReviewer, "prepare", AsyncMock(return_value=True))
    monkeypatch.setattr(RenderingService, "_verify_ffmpeg", lambda self: None)
    settings = pipeline_module.get_settings()
    monkeypatch.setattr(settings, "local_mode", True)
    monkeypatch.setattr(settings, "local_output_dir", str(tmp_path / "out"))
    monkeypatch.setattr(settings.__class__, "temp_directory", property(lambda self: str(tmp_path / "work")))
    calls = SimpleNamespace(transcribed=0, planned=[])

    def build(plan_fails=False):
        pipeline = AIClippingPipeline()
        pipeline.local_mode = True

        async def download(url, output_dir):
            meta = SimpleNamespace(title="Test", duration_seconds=300.0, width=1920, height=1080)
            return SimpleNamespace(video_path=str(tmp_path / "source.mp4"), metadata=meta, file_size_bytes=1)

        async def transcribe(video_path, work_dir, keyterms=None, **_range):
            calls.transcribed += 1
            return TranscriptionResult(segments=segments(), full_text="Hello there. Later.", language="en")

        async def plan(**kwargs):
            calls.planned.append(kwargs["transcript_result"].segments)
            if plan_fails:
                raise AiClientError("synthetic", reason="model_not_found")
            return ClipPlanResponse(segments=[ClipPlanSegment(0, 20_000, 0.8)], total_clips=1)

        async def render(request):
            open(request.output_path, "wb").write(b"mp4")
            return RenderResult(output_path=request.output_path, file_size_bytes=3, duration_ms=20_000)

        monkeypatch.setattr(pipeline.video_downloader, "download_video", download)
        monkeypatch.setattr(pipeline.transcription_service, "transcribe", transcribe)
        monkeypatch.setattr(pipeline.intelligence_planner, "plan_clips", plan)
        monkeypatch.setattr(pipeline.rendering_service, "render_clip", render)
        return pipeline

    return build, calls, tmp_path / "out"


def test_a_run_that_failed_at_planning_continues_from_its_transcript(local_pipeline):
    build, calls, out = local_pipeline
    failed = asyncio.run(build(plan_fails=True).process_video(ClippingJobRequest(video_url="local.mp4", job_id="first")))
    assert failed.status == JobStatus.FAILED and failed.failure_stage == "planning"
    saved = out / "first" / "transcript.json"
    assert json.loads(saved.read_text())["source_range"] == [None, None]

    resumed = asyncio.run(build().process_video(ClippingJobRequest(
        video_url="local.mp4", job_id="second", resume_transcript_path=str(saved))))
    assert resumed.status == JobStatus.COMPLETED, resumed.error
    assert calls.transcribed == 1
    assert calls.planned[1] == calls.planned[0] == segments()
    # The new run keeps its own copy, so it can be continued from too.
    assert json.loads((out / "second" / "transcript.json").read_text())["segments"] == \
        json.loads(saved.read_text())["segments"]
    assert "total_estimated_cost_usd" in resumed.output.metrics["api_costs"]
    assert "transcription" not in resumed.output.metrics["api_costs"]
    assert resumed.output.metrics["requested_settings"]["transcription_model"] == "saved transcript"


def test_an_advanced_run_continues_without_a_transcription_model(local_pipeline, monkeypatch):
    # Main drops the cloud transcription model from a continued Advanced run; its stats must survive.
    build, calls, out = local_pipeline
    asyncio.run(build(plan_fails=True).process_video(ClippingJobRequest(video_url="local.mp4", job_id="first")))
    settings = pipeline_module.get_settings()
    monkeypatch.setattr(settings, "clipping_mode", "advanced")
    monkeypatch.setattr(settings, "advanced_transcription_model", "")
    monkeypatch.setattr(settings, "transcription_source", "cloud")
    resumed = asyncio.run(build().process_video(ClippingJobRequest(
        video_url="local.mp4", job_id="second", resume_transcript_path=str(out / "first" / "transcript.json"))))
    assert resumed.status == JobStatus.COMPLETED, resumed.error
    assert "api_costs" in resumed.output.metrics and "pipeline_stages" in resumed.output.metrics


def test_a_continued_run_keeps_the_range_its_transcript_covers(local_pipeline):
    build, calls, out = local_pipeline
    asyncio.run(build(plan_fails=True).process_video(ClippingJobRequest(video_url="local.mp4", job_id="first")))
    resumed = asyncio.run(build().process_video(ClippingJobRequest(
        video_url="local.mp4", job_id="second", start_time_seconds=0, end_time_seconds=120,
        resume_transcript_path=str(out / "first" / "transcript.json"))))
    assert resumed.status == JobStatus.COMPLETED, resumed.error
    # Still the whole source, so a later run without a range can continue from this copy.
    assert json.loads((out / "second" / "transcript.json").read_text())["source_range"] == [None, None]


def test_a_transcript_for_another_range_stops_the_run(local_pipeline):
    build, calls, out = local_pipeline
    asyncio.run(build(plan_fails=True).process_video(ClippingJobRequest(
        video_url="local.mp4", job_id="first", start_time_seconds=30, end_time_seconds=90)))
    saved = out / "first" / "transcript.json"
    assert json.loads(saved.read_text())["source_range"] == [30, 90]

    result = asyncio.run(build().process_video(ClippingJobRequest(
        video_url="local.mp4", job_id="second", start_time_seconds=10, end_time_seconds=90,
        resume_transcript_path=str(saved))))
    assert result.status == JobStatus.FAILED
    assert (result.error, result.failure_stage) == (SAVED_TRANSCRIPT_ERROR, "transcription")
    assert result.failure_code == "transcription.saved_transcript_unusable"
    assert calls.transcribed == 1 and len(calls.planned) == 1
    assert not os.path.exists(out / "second" / "transcript.json")
