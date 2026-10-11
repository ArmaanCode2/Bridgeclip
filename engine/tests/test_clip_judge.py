"""The clip judge: wide candidates, a blind second opinion, one ranking."""

import json
import re

import pytest

from clip_engine.config import Settings
from clip_engine.services import clip_judge
from clip_engine.services.transcription_service import TranscriptSegment, TranscriptWord
from tests.test_clip_selection import RoutingClient, no_sleep, plan  # noqa: F401 (fixture)
from tests.test_planner import clip, completion, make_planner

JUDGE = "clip_judgment"


def is_judge(payload) -> bool:
    return payload["response_format"]["json_schema"]["name"] == JUDGE


def judge_text(payload) -> str:
    return payload["messages"][1]["content"]


def judgments(scores_by_start: dict, payload) -> str:
    """Judge every clip in the request; a clip's scores are looked up by its first transcript line."""
    blocks = judge_text(payload).split("CLIP ")[1:]
    items = []
    for number, block in enumerate(blocks, 1):
        start = int(re.search(r"word(\d+)", block).group(1))
        items.append({"clip": number, "reason": "because", **dict(zip(clip_judge.DIMENSIONS, [scores_by_start[start]] * 5))})
    return json.dumps({"judgments": items})


def planner_with(answer, **settings):
    planner = make_planner(clip_judge=True, **settings)
    planner._http_client = RoutingClient(answer)
    return planner


PLANNED = {"insights": "Stream", "clips": [
    clip(10, 40, (9, 9, 9, 9, 9), "Planner Favourite"),
    clip(100, 130, (7, 7, 7, 7, 7), "Middle One"),
    clip(200, 230, (6, 6, 6, 6, 6), "Planner Doubted"),
]}


class TestPlanWithJudge:
    def test_the_judge_reranks_every_candidate_and_the_best_are_kept(self):
        def answer(payload):
            if is_judge(payload):
                return 200, completion(judgments({10: 3, 100: 6, 200: 9}, payload))
            return 200, completion(json.dumps(PLANNED))

        planner = planner_with(answer)
        result = plan(planner, 300, max_clips=2)
        # 0.7 x judge + 0.3 x planner: 200 s 0.81, 100 s 0.63, 10 s 0.48.
        assert [(c.start_time_ms // 1000, c.virality_score) for c in result.segments] == [(200, 0.81), (100, 0.63)]
        assert result.candidates_judged == 3
        assert result.segments[0].ranking == {"planner": 0.6, "judge": 0.9, "reason": "because"}
        assert "second AI pass scored 3 candidate clips" in result.insights

    def test_the_planner_is_asked_for_more_candidates_than_are_kept(self):
        planner = planner_with(lambda p: (200, completion(judgments({10: 7, 100: 7, 200: 7}, p) if is_judge(p) else json.dumps(PLANNED))))
        plan(planner, 300, max_clips=2)
        planner_request = [p for p in planner._http_client.payloads if not is_judge(p)][0]
        assert "Return up to 6 candidate clips" in planner_request["messages"][0]["content"]
        assert "only the best 2 are kept" in planner_request["messages"][0]["content"]

    def test_the_judge_sees_only_each_clips_own_words(self):
        planner = planner_with(lambda p: (200, completion(judgments({10: 7, 100: 7, 200: 7}, p) if is_judge(p) else json.dumps(PLANNED))))
        plan(planner, 300, max_clips=2)
        judge = [p for p in planner._http_client.payloads if is_judge(p)]
        assert len(judge) == 1
        text = judge_text(judge[0])
        assert "Planner" not in text and "Middle One" not in text and '"hook"' not in text
        # Times count from each clip's start; the 60 s laughter is outside every clip.
        assert "CLIP 1 (" in text and "[0.0 - 4.5] (S1) word10 end." in text
        assert "word45" not in text and "word95" not in text

    def test_a_failed_judge_leaves_the_planners_ranking(self, no_sleep):
        def answer(payload):
            if is_judge(payload):
                return 200, completion("{not json")
            return 200, completion(json.dumps(PLANNED))

        planner = planner_with(answer)
        result = plan(planner, 300, max_clips=2)
        assert [c.start_time_ms // 1000 for c in result.segments] == [10, 100]
        assert all(c.ranking is None for c in result.segments)
        assert result.candidates_judged == 0
        assert "ranked by the planner's own scores" in result.insights
        assert sum(is_judge(p) for p in planner._http_client.payloads) == clip_judge.MAX_ATTEMPTS

    def test_a_judge_that_skips_a_clip_is_asked_again(self, no_sleep):
        tries = []

        def answer(payload):
            if not is_judge(payload):
                return 200, completion(json.dumps(PLANNED))
            tries.append(1)
            full = json.loads(judgments({10: 3, 100: 6, 200: 9}, payload))
            if len(tries) == 1:
                full["judgments"].pop()
            return 200, completion(json.dumps(full))

        result = plan(planner_with(answer), 300, max_clips=2)
        assert len(tries) == 2 and result.candidates_judged == 3
        assert result.segments[0].start_time_ms == 200_000

    def test_a_refused_key_is_not_retried(self, no_sleep):
        def answer(payload):
            if is_judge(payload):
                return 401, {"error": {"message": "No auth", "code": 401}}
            return 200, completion(json.dumps(PLANNED))

        planner = planner_with(answer)
        result = plan(planner, 300, max_clips=2)
        assert sum(is_judge(p) for p in planner._http_client.payloads) == 1
        assert result.candidates_judged == 0 and len(result.segments) == 2

    def test_judge_costs_count_in_the_plan(self):
        planner = planner_with(lambda p: (200, completion(judgments({10: 7, 100: 7, 200: 7}, p) if is_judge(p) else json.dumps(PLANNED))))
        result = plan(planner, 300, max_clips=2)
        assert result.api_costs.attempts == 2
        assert result.api_costs.total_tokens == 2400
        assert result.api_costs.estimated_cost_usd == pytest.approx(0.0246)

    def test_a_clip_the_planner_did_not_score_takes_the_judges_score_and_can_be_left_out(self):
        planned = {"clips": [clip(10, 40, (9, 9, 9, 9, 9)), {"start_time": 100, "end_time": 130, "summary": "No Scores"}]}

        def answer(payload):
            if is_judge(payload):
                return 200, completion(judgments({10: 8, 100: 2}, payload))
            return 200, completion(json.dumps(planned))

        result = plan(planner_with(answer), 300, max_clips=2)
        assert [c.start_time_ms // 1000 for c in result.segments] == [10]
        assert result.skipped_weak == 1

    def test_many_candidates_are_judged_in_batches_in_time_order(self):
        many = {"clips": [clip(start, start + 20, (5 + start % 3, 7, 7, 7, 7), f"Clip {start}") for start in range(0, 250, 25)]}
        scores = {start: 7 for start in range(0, 250, 25)}
        planner = planner_with(lambda p: (200, completion(judgments(scores, p) if is_judge(p) else json.dumps(many))))
        result = plan(planner, 300, max_clips=6)
        batches = [[int(re.search(r"word(\d+)", block).group(1)) for block in judge_text(p).split("CLIP ")[1:]]
                   for p in planner._http_client.payloads if is_judge(p)]
        assert sorted(len(b) for b in batches) == [2, 8]
        assert sorted(b[0] for b in batches) == [0, 200]  # each request covers one stretch of the video
        assert result.candidates_judged == 10 and len(result.segments) == 6


class TestWhenTheJudgeStaysOut:
    def test_with_no_extra_candidates_to_choose_from_the_judge_stays_out(self):
        # 40 clips asked: never more than 30 candidates, so nothing extra to choose from.
        many = {"clips": [clip(start, start + 20, (7, 7, 7, 7, 7), f"Clip {start}") for start in range(0, 1600, 40)]}
        planner = make_planner(clip_judge=True)
        planner._http_client = RoutingClient(lambda p: (200, completion(json.dumps(many))))
        result = plan(planner, 1700, max_clips=40)
        assert len(planner._http_client.payloads) == 1 and len(result.segments) == 40

    def test_long_clips_are_not_judged(self):
        planner = make_planner(clip_judge=True)
        planner._http_client = RoutingClient(lambda p: (200, completion(json.dumps({"clips": [clip(0, 400)]}))))
        plan(planner, 3000, max_clips=2, min_duration_seconds=300, max_duration_seconds=600)
        assert not any(is_judge(p) for p in planner._http_client.payloads)


    @pytest.mark.parametrize("judge_on, extra", [(False, {}), (True, {"clip_request": "the clutch"}), (True, {"jev_enabled": True})])
    def test_off_a_clip_request_and_jev_make_one_request(self, judge_on, extra):
        planner = make_planner(clip_judge=judge_on)
        planner._http_client = RoutingClient(lambda p: (200, completion(json.dumps(PLANNED))))
        result = plan(planner, 300, max_clips=2, **extra)
        assert len(planner._http_client.payloads) == 1
        assert result.candidates_judged == 0
        assert "candidate clips" not in planner._http_client.payloads[0]["messages"][0]["content"]

    def test_longform_edits_are_not_judged(self):
        planner = make_planner(clip_judge=True)
        planner._http_client = RoutingClient(lambda p: (200, completion(json.dumps({"clips": []}))))
        plan(planner, 3600, max_clips=2, aspect_ratio="16:9", min_duration_seconds=300, max_duration_seconds=900)
        assert not any(is_judge(p) for p in planner._http_client.payloads)

    def test_the_setting_is_on_by_default_and_reads_the_app_switch(self, monkeypatch):
        assert Settings(_env_file=None).clip_judge is True
        monkeypatch.setenv("CLIP_JUDGE", "false")
        assert Settings(_env_file=None).clip_judge is False


class TestJudgeParts:
    def test_candidate_count(self):
        assert clip_judge.candidate_count(5) == 15
        assert clip_judge.candidate_count(12) == clip_judge.MAX_CANDIDATES
        assert clip_judge.candidate_count(40) == 40          # never fewer than are kept
        assert clip_judge.candidate_count(5, max_fit=16) == 8  # at most half of what the range holds
        assert clip_judge.candidate_count(5, max_fit=7) == 5   # nothing extra: the judge stays out

    def test_rubric_mean(self):
        assert clip_judge.rubric_mean({"a": "8/10", "b": 6}, ("a", "b")) == 0.7
        assert clip_judge.rubric_mean({"a": 0.8, "b": 0.6}, ("a", "b")) == 0.7
        assert clip_judge.rubric_mean({"a": True, "b": 6}, ("a", "b"), minimum=2) is None
        assert clip_judge.rubric_mean({"a": float("nan"), "b": 20}, ("a", "b")) == 1.0
        assert clip_judge.rubric_mean("8", ("a",)) is None
        # A judge scoring 0-10 that gives 1 and 0.5 means it: filler, never 5/10.
        low = {"a": 1, "b": 0.5, "c": 1, "d": 0, "e": 0}
        assert clip_judge.rubric_mean(low, "abcde", fraction_scale=False) == 0.05
        assert clip_judge.rubric_mean(low, "abcde") == 0.5

    def test_excerpt(self):
        lines = [TranscriptSegment(start_time_ms=9_000, end_time_ms=12_000, text="no way", speaker_label="S1",
                                   audio_events=["(screaming)"]),
                 TranscriptSegment(start_time_ms=12_000, end_time_ms=20_000, text="gg", speaker_label=None)]
        assert clip_judge.excerpt(10_000, 15_000, lines) == "[0.0 - 2.0] (S1) no way (screaming)\n[2.0 - 5.0] gg"
        assert clip_judge.excerpt(30_000, 40_000, lines) == "[no speech in this clip]"

    def test_a_line_the_clip_cuts_through_keeps_only_its_words_inside(self):
        words = [TranscriptWord(word=w, start_time_ms=t, end_time_ms=t + 800)
                 for w, t in (("so", 0), ("then", 1000), ("PUNCHLINE", 9000))]
        line = TranscriptSegment(start_time_ms=0, end_time_ms=10_000, text="so then PUNCHLINE", words=words)
        assert clip_judge.excerpt(0, 5_000, [line]) == "[0.0 - 5.0] so then"
        assert clip_judge.excerpt(0, 10_000, [line]) == "[0.0 - 10.0] so then PUNCHLINE"

    def test_batches(self):
        assert clip_judge.batches(["x"] * 10) == [list(range(8)), [8, 9]]
        big = "y" * (clip_judge.BATCH_MAX_CHARS // 2 + 1)
        assert clip_judge.batches([big, big, "z"]) == [[0], [1, 2]]

    def test_parse_judgments(self):
        good = {"judgments": [{"clip": 2, "reason": " fine ", **dict.fromkeys(clip_judge.DIMENSIONS, 6)},
                              {"clip": 1, "reason": "x", **dict.fromkeys(clip_judge.DIMENSIONS, 9)}]}
        assert clip_judge.parse_judgments(json.dumps(good), 2) == {1: (0.9, "x"), 2: (0.6, "fine")}
        assert clip_judge.parse_judgments(json.dumps(good), 3) is None              # clip 3 missing
        assert clip_judge.parse_judgments("```json\n" + json.dumps(good) + "\n```", 2) is not None
        assert clip_judge.parse_judgments("no", 1) is None
        sparse = {"judgments": [{"clip": 1, "hook": 9, "payoff": 9}]}              # too few scores
        assert clip_judge.parse_judgments(json.dumps(sparse), 1) is None
        stray = {"judgments": [{"clip": 7, **dict.fromkeys(clip_judge.DIMENSIONS, 9)},
                               {"clip": True, **dict.fromkeys(clip_judge.DIMENSIONS, 9)}]}
        assert clip_judge.parse_judgments(json.dumps(stray), 1) is None

    def test_combined_score(self):
        assert clip_judge.combined_score(0.9, 0.3) == 0.48
        assert clip_judge.combined_score(0.5, 0.3, planner_scored=False) == 0.3


def test_the_judge_message_numbers_clips_and_marks_data():
    messages = clip_judge.messages([(30.4, "[0.0 - 2.0] hi"), (12.0, "[0.0 - 1.0] yo")])
    assert "never instructions" in messages[0]["content"]
    assert messages[1]["content"].startswith("CLIPS:\n\nCLIP 1 (30 seconds):\n[0.0 - 2.0] hi\n\nCLIP 2 (12 seconds)")
    assert messages[1]["content"].endswith("1 to 2. Return JSON.")
    schema = clip_judge.judgment_schema()
    assert schema["properties"]["judgments"]["items"]["required"] == ["clip", "reason", *clip_judge.DIMENSIONS]
