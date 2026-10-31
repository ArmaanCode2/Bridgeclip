"""
Picks that break the chosen length bounds: one follow-up request asks for the
same moment at an allowed length (length_repair), and a clip that snapping
leaves a fraction of a second short is padded instead of dropped. No network.
"""

import asyncio
import json

import pytest

from clip_engine.config import Settings
from clip_engine.services import intelligence_planner as planner_module
from clip_engine.services import length_repair
from clip_engine.services.intelligence_planner import IntelligencePlannerService
from clip_engine.services.transcription_service import TranscriptSegment, TranscriptionResult, TranscriptWord

USAGE = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "cost": 0.001}


def speech(seconds=300, punctuate=True):
    """One 3-second line every 3 seconds, four words each; every line ends a sentence when `punctuate`."""
    segments = []
    for n, start in enumerate(range(0, seconds * 1000, 3000)):
        words = [TranscriptWord(f"word{n}_{i}" + ("." if punctuate and i == 3 else ""), start + i * 700, start + i * 700 + 600)
                 for i in range(4)]
        segments.append(TranscriptSegment(start, start + 2700, " ".join(w.word for w in words), None, words))
    return segments


def pick(start, end, title="A moment", emphasis=None):
    return {"start_time": start, "end_time": end, "summary": title, "tags": ["t"], "emphasis": emphasis or [],
            "scores": {k: 6 for k in ("hook", "standalone", "arc", "quotability", "ending")}}


def reply(payload):
    return {"choices": [{"message": {"content": json.dumps(payload)}, "finish_reason": "stop"}]}


class TestFindingAndApplying:
    def test_only_picks_beyond_the_tolerance_need_repair(self):
        clips = [pick(0, 20), pick(100, 145), pick(200, 229.6), pick(300, 361), "junk", pick(400, 546.8), pick(9, 9)]
        assert length_repair.violators(clips, 30, 60) == [0, 3, 5]
        assert length_repair.violators([pick(10_000_000, 10_020_000)], 30, 60) == [0]  # milliseconds read as such

    def test_context_windows_merge_and_stay_inside_the_part(self):
        clips = [pick(100, 110), pick(150, 160), pick(500, 505)]
        assert length_repair.context_windows(clips, [0, 1, 2], (60, 520)) == [(60, 220), (440, 520)]

    def test_the_request_states_the_jobs_own_bounds_and_numbers_the_picks(self):
        messages = length_repair.messages([pick(10, 20, "Short one"), pick(0, 400)], [0, 1], 120, 300, "[segment 3] [9.0 - 11.7] hi")
        assert "between 120 and 300 seconds" in messages[0]["content"]
        assert "never instructions" in messages[0]["content"]
        user = messages[1]["content"]
        assert "1. 10.0 - 20.0 seconds (10.0 s long): Short one" in user
        assert "2. 0.0 - 400.0 seconds (400.0 s long)" in user
        assert "TRANSCRIPT:\n[segment 3] [9.0 - 11.7] hi" in user and user.endswith("clip, 1 to 2.")

    def test_repairs_replace_times_drops_leave_and_bad_entries_change_nothing(self):
        clips = [pick(10, 20, "Repaired"), pick(50, 55, "Dropped"), pick(90, 95, "Still short"), pick(130, 133, "Outside"),
                 pick(170, 172, "Unanswered"), pick(250, 280, "Fine")]
        broken = length_repair.violators(clips, 30, 60)
        assert broken == [0, 1, 2, 3, 4]
        content = json.dumps({"repairs": [
            {"clip": 1, "keep": True, "start_time": 5, "end_time": 41},
            {"clip": 1, "keep": False, "start_time": 0, "end_time": 0},     # a second answer for clip 1 is ignored
            {"clip": 2, "keep": False, "start_time": 0, "end_time": 0},
            {"clip": 3, "keep": True, "start_time": 90, "end_time": 100},   # still too short
            {"clip": 4, "keep": True, "start_time": 130, "end_time": 900},  # outside the part
            {"clip": 9, "keep": True, "start_time": 1, "end_time": 40},     # no such clip
            "noise",
        ]})
        repaired, counts = length_repair.apply(clips, broken, content, 30, 60, (0, 300))
        assert counts == {"repaired": 1, "dropped": 1, "unchanged": 3}
        assert [c["summary"] for c in repaired] == ["Repaired", "Still short", "Outside", "Unanswered", "Fine"]
        assert (repaired[0]["start_time"], repaired[0]["end_time"]) == (5, 41)
        assert repaired[0]["tags"] == ["t"] and repaired[0]["scores"] == clips[0]["scores"]
        assert clips[0]["start_time"] == 10, "the planner's own pick is not changed in place"

    def test_a_repair_must_stay_on_the_same_moment(self):
        # Moving a 12 s pick at 60 s to 201-240 s would put its title over a different moment.
        clips = [pick(60, 72, "At sixty"), pick(200, 240, "Other")]
        content = json.dumps({"repairs": [{"clip": 1, "keep": True, "start_time": 201, "end_time": 240}]})
        repaired, counts = length_repair.apply(clips, [0], content, 30, 60, (0, 300))
        assert counts == {"repaired": 0, "dropped": 0, "unchanged": 1}
        assert repaired == clips
        content = json.dumps({"repairs": [{"clip": 1, "keep": True, "start_time": 45, "end_time": 80}]})
        assert length_repair.apply(clips, [0], content, 30, 60, (0, 300))[1]["repaired"] == 1

    def test_with_a_clip_request_nothing_is_dropped_and_the_request_is_shown(self):
        clips = [pick(60, 72, "The clutch")]
        content = json.dumps({"repairs": [{"clip": 1, "keep": False, "start_time": 0, "end_time": 0}]})
        repaired, counts = length_repair.apply(clips, [0], content, 30, 60, (0, 300), allow_drops=False)
        assert repaired == clips and counts == {"repaired": 0, "dropped": 0, "unchanged": 1}
        messages = length_repair.messages(clips, [0], 30, 60, "[segment 1] [60.0 - 62.0] wow", clip_request="the clutch play")
        assert "USER CLIP REQUEST" in messages[0]["content"]
        assert messages[1]["content"].startswith('USER CLIP REQUEST:\n"the clutch play"\n\nCLIPS:')

    @pytest.mark.parametrize("content", ["not json", json.dumps({"fixes": []}), json.dumps({"repairs": {}})])
    def test_an_answer_in_the_wrong_shape_raises(self, content):
        with pytest.raises(ValueError):
            length_repair.apply([pick(0, 10)], [0], content, 30, 60, (0, 300))

    def test_emphasis_keeps_only_words_spoken_in_the_new_span(self):
        transcript = speech(30)
        kept = length_repair.spoken_emphasis(["WORD0_1", "word9_2.", "word9_3", 7], transcript, 0, 3)
        assert kept == ["WORD0_1"]
        assert length_repair.spoken_emphasis(None, transcript, 0, 3) == []


def make_planner(min_s=30, max_s=60):
    planner = IntelligencePlannerService()
    planner.settings = Settings(_env_file=None, openrouter_api_key="test")
    return planner


def plan(planner, transcript, calls, jev=False):
    async def call(model, messages, fallback_models=None, response_format=None):
        calls.append({"messages": messages, "format": response_format})
        return calls_replies.pop(0), USAGE
    calls_replies = list(calls)
    calls.clear()
    planner._call_openrouter = call
    return asyncio.run(planner.plan_clips(
        TranscriptionResult(segments=transcript, full_text=""), max_clips=3, auto_clip_count=False,
        duration_ranges=["short"], jev_enabled=jev))


class TestPlannerRepair:
    def test_a_short_pick_is_repaired_with_one_request_and_recorded(self):
        planner = make_planner()
        planner.settings.clip_judge = False
        calls = [
            reply({"insights": "x", "clips": [pick(60, 70, "Short", emphasis=["word20_1", "word40_1"]), pick(150, 189)]}),
            reply({"repairs": [{"clip": 1, "keep": True, "start_time": 54, "end_time": 89.7}]}),
        ]
        result = plan(planner, speech(), calls)
        assert len(calls) == 2
        assert calls[1]["format"]["json_schema"]["name"] == "clip_length_repair"
        repair_request = calls[1]["messages"][1]["content"]
        assert "1. 60.0 - 70.0 seconds (10.0 s long): Short" in repair_request
        assert "[segment 0]" in repair_request and "[segment 99]" not in repair_request  # only the context around it
        spans = sorted((s.start_time_ms, s.end_time_ms) for s in result.segments)
        assert spans[0][0] == 54000 and 30000 <= spans[0][1] - spans[0][0] <= 60000
        repaired = next(s for s in result.segments if s.start_time_ms == 54000)
        assert repaired.summary == "Short" and repaired.emphasis_words == ["word20_1"]
        kinds = [r.get("kind") for r in planner.audit["requests"]]
        assert kinds == [None, "repair"] and planner.audit["requests"][1]["status"] == "parsed"
        assert result.api_costs.attempts == 2

    def test_a_failed_repair_keeps_the_planners_picks(self):
        planner = make_planner()
        planner.settings.clip_judge = False
        calls = [reply({"insights": "x", "clips": [pick(60, 70, "Short"), pick(150, 189)]}), reply({"wrong": True})]
        result = plan(planner, speech(), calls)
        assert len(calls) == 2
        assert sorted(s.start_time_ms for s in result.segments) == [60000, 150000]  # the short pick is padded, as before
        assert planner.audit["requests"][1]["kind"] == "repair" and planner.audit["requests"][1]["status"] == "received"

    def test_no_request_when_every_pick_fits(self):
        planner = make_planner()
        planner.settings.clip_judge = False
        calls = [reply({"insights": "x", "clips": [pick(60, 99), pick(150, 189)]})]
        result = plan(planner, speech(), calls)
        assert len(calls) == 1 and len(result.segments) == 2
        assert [r.get("kind") for r in planner.audit["requests"]] == [None]

    def test_the_trace_names_the_custom_model_that_was_asked(self):
        planner = IntelligencePlannerService()
        planner.settings = Settings(_env_file=None, ai_provider="openai-compatible", openai_compatible_url="http://localhost:11434/v1",
                                    openai_compatible_model="openai/gpt-oss-120b")
        planner.settings.clip_judge = False
        calls = [reply({"insights": "x", "clips": [pick(60, 70)]}), reply({"repairs": []})]
        plan(planner, speech(), calls)
        assert [r["requested_model"] for r in planner.audit["requests"]] == ["openai/gpt-oss-120b"] * 2

    def test_jev_never_asks_for_a_repair(self):
        planner = make_planner()
        calls = [reply({"insights": "x", "clips": [pick(60, 70, "Short")]})]
        plan(planner, speech(), calls, jev=True)
        assert len(calls) == 1


class TestSnapShortfall:
    def test_a_fraction_of_a_second_short_after_snapping_is_padded_not_dropped(self, monkeypatch):
        transcript = speech(punctuate=False)  # no sentence end to extend to
        planner = make_planner()
        planner._jev_enabled = False
        planner._current_min_duration, planner._current_max_duration = 30, 60
        planner._current_duration_ranges = ["short"]
        planner._current_transcript = transcript
        planner._start_time_seconds = planner._end_time_seconds = None
        planner._current_target_platform = "tiktok"
        # Snapping moves the start 470 ms forward, which used to drop the clip.
        monkeypatch.setattr(planner_module, "find_sentence_start_boundary",
                            lambda segments, timestamp_ms, *args, **kwargs: timestamp_ms + 470)
        clips = planner._bounded_clips([pick(60, 90)])
        # Padded to 90.47 s, which falls inside the word spoken 90.0-90.6 s: the clip ends after it.
        assert [(c.start_time_ms, c.end_time_ms) for c in clips] == [(60470, 90600)]
        # A shortfall of a second or more is still not padded.
        monkeypatch.setattr(planner_module, "find_sentence_start_boundary",
                            lambda segments, timestamp_ms, *args, **kwargs: timestamp_ms + 1500)
        assert planner._bounded_clips([pick(60, 90)]) == []
