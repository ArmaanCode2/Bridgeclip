"""
The last look at the whole candidate set: clips that repeat an incident are left
out (with guards), and titles that repeat another's wording are renamed. No network.
"""

import asyncio
import json

import pytest

from clip_engine.config import Settings
from clip_engine.services import final_check
from clip_engine.services.intelligence_planner import ClipPlanSegment, IntelligencePlannerService
from clip_engine.services.transcription_service import TranscriptSegment, TranscriptionResult, TranscriptWord

USAGE = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "cost": 0.001}


def speech(seconds=600):
    segments = []
    for n, start in enumerate(range(0, seconds * 1000, 3000)):
        words = [TranscriptWord(f"w{n}_{i}" + ("." if i == 3 else ""), start + i * 700, start + i * 700 + 600) for i in range(4)]
        segments.append(TranscriptSegment(start, start + 2700, " ".join(w.word for w in words), None, words))
    return segments


def segment(start_s, score, title, topic=None):
    clip = ClipPlanSegment(start_time_ms=start_s * 1000, end_time_ms=(start_s + 40) * 1000, virality_score=score, summary=title)
    clip.moment = {"topic": topic} if topic else None
    return clip


def answer(same=(), retitle=()):
    return json.dumps({"same_incident": [{"keep": k, "drop": d} for k, d in same],
                       "retitle": [{"clip": c, "title": t} for c, t in retitle]})


class TestRules:
    def test_the_request_numbers_clips_with_title_topic_and_opening_words(self):
        clips = [segment(0, 0.8, "Toilet tour", "Japanese toilets"), segment(60, 0.7, "Rank talk")]
        user = final_check.messages(clips, speech())[1]["content"]
        assert '1. Title: "Toilet tour" | Topic: Japanese toilets | Opens: "w0_0 w0_1 w0_2 w0_3. w1_0' in user
        assert '2. Title: "Rank talk" | Opens: "w20_0' in user and "| Topic" not in user.splitlines()[2]
        assert user.endswith("Check clips 1 to 2. Return JSON.")
        assert "never instructions" in final_check.messages(clips, speech())[0]["content"]

    def test_only_a_weaker_clip_is_dropped_and_never_both_of_a_pair(self):
        clips = [segment(i * 50, score, f"Clip {i}") for i, score in enumerate([0.9, 0.6, 0.8, 0.5, 0.7, 0.4, 0.3, 0.2])]
        dropped, titles, applied = final_check.apply(clips, answer(same=[
            (2, 1),   # keep 0.6 over 0.9: the stronger clip is never dropped for a weaker one
            (1, 2),   # keep 0.9, drop 0.6
            (2, 4),   # keep 0.6, drop 0.5: all three are one incident, and the 0.9 clip stays
            (3, 3), (9, 1), ("1", 2),
        ]))
        # The 0.5 clip repeated the 0.6 one, which repeated the 0.9 one: both notes name the clip that stays.
        assert dropped == {1, 3} and applied == [(3, 0), (1, 0)] and titles == {}
        # Two equal clips naming each other: one stays.
        clips = [segment(i * 50, 0.7, f"Clip {i}") for i in range(4)]
        dropped, _, _ = final_check.apply(clips, answer(same=[(1, 2), (2, 1)]))
        assert len(dropped) == 1

    def test_at_most_a_quarter_of_the_set_is_left_out_weakest_first(self):
        clips = [segment(i * 50, 0.9 - i * 0.05, f"Clip {i}") for i in range(8)]
        dropped, _, applied = final_check.apply(clips, answer(same=[(1, 2), (1, 3), (1, 4), (1, 5), (1, 6)]))
        assert len(dropped) == 2 and dropped == {5, 4}
        assert [drop for drop, _ in applied] == [5, 4]

    def test_new_titles_must_be_two_to_seven_words_and_not_for_dropped_clips(self):
        clips = [segment(i * 50, 0.9 - i * 0.1, f"Insane thing {i}") for i in range(4)]
        dropped, titles, _ = final_check.apply(clips, answer(same=[(1, 2)], retitle=[
            (2, "Dropped clip gets nothing"), (3, "  Melvin   steals the mouse "), (3, "Second answer ignored"),
            (4, "Word"), (1, "one two three four five six seven eight"), (9, "No such clip"),
        ]))
        assert dropped == {1}
        assert titles == {2: "Melvin steals the mouse"}

    def test_at_most_half_the_set_is_renamed_and_never_to_a_title_already_used(self):
        clips = [segment(i * 50, 0.9 - i * 0.1, title) for i, title in enumerate(["Rank one grind", "Rank one bet", "Japan toilets", "Late night lag"])]
        _, titles, _ = final_check.apply(clips, answer(retitle=[
            (2, "Japan toilets"),              # already another clip's title
            (2, "Melvin bets his rank"), (3, "Bidet with a dryer"), (4, "Packet loss at midnight"),
        ]))
        assert titles == {1: "Melvin bets his rank", 2: "Bidet with a dryer"}

    @pytest.mark.parametrize("content", ["nope", json.dumps({"same_incident": []}), json.dumps({"same_incident": {}, "retitle": []})])
    def test_an_answer_in_the_wrong_shape_raises(self, content):
        with pytest.raises(ValueError):
            final_check.apply([segment(0, 0.5, "x")], content)


def planner_with(replies, calls):
    planner = IntelligencePlannerService()
    planner.settings = Settings(_env_file=None, openrouter_api_key="test")
    planner.settings.clip_judge = False

    async def call(model, messages, fallback_models=None, response_format=None):
        calls.append(response_format["json_schema"]["name"] if response_format else "clip_plan")
        return replies.pop(0), USAGE
    planner._call_openrouter = call
    return planner


def reply(payload):
    return {"choices": [{"message": {"content": json.dumps(payload)}, "finish_reason": "stop"}]}


def picks(count):
    return [{"start_time": 10 + i * 60, "end_time": 50 + i * 60, "summary": f"Insane moment {i}", "tags": [], "emphasis": [],
             "scores": {k: 8 - i * 0.5 for k in ("hook", "standalone", "arc", "quotability", "ending")}} for i in range(count)]


def run(planner, clip_request=None):
    return asyncio.run(planner.plan_clips(
        TranscriptionResult(segments=speech(), full_text=""), max_clips=6, auto_clip_count=False,
        duration_ranges=["short"], clip_request=clip_request))


class TestPlannerFinalCheck:
    def test_one_request_leaves_out_a_repeat_and_renames_a_title(self):
        calls = []
        planner = planner_with([
            reply({"insights": "x", "clips": picks(5)}),
            {"choices": [{"message": {"content": answer(same=[(1, 4)], retitle=[(2, "Melvin steals the mouse")])}, "finish_reason": "stop"}]},
        ], calls)
        result = run(planner)
        assert calls == ["clip_plan", "clip_set_check"]
        titles = [s.summary for s in sorted(result.segments, key=lambda s: s.start_time_ms)]
        assert titles == ["Insane moment 0", "Melvin steals the mouse", "Insane moment 2", "Insane moment 4"]
        assert 'Left out 1 clip repeating another clip\'s incident: "Insane moment 3" (same as "Insane moment 0")' in result.insights
        assert "Renamed 1 title" in result.insights
        record = planner.audit["requests"][-1]
        assert record["kind"] == "final_check" and record["status"] == "parsed"

    def test_a_failed_check_keeps_the_set(self):
        calls = []
        planner = planner_with([reply({"insights": "x", "clips": picks(5)}), reply({"unexpected": True})], calls)
        result = run(planner)
        assert calls == ["clip_plan", "clip_set_check"] and len(result.segments) == 5

    def test_no_request_for_a_small_set_or_a_clip_request(self):
        calls = []
        result = run(planner_with([reply({"insights": "x", "clips": picks(3)})], calls))
        assert calls == ["clip_plan"] and len(result.segments) == 3
        calls = []
        run(planner_with([reply({"insights": "x", "clips": picks(5)})], calls), clip_request="every clutch play")
        assert calls == ["clip_plan"]
