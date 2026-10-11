"""Clip selection: up to N, a minimum score, long sources searched in parts, sentence starts."""

import asyncio
import json
import re
from types import SimpleNamespace

import httpx
import pytest

from clip_engine.services import intelligence_planner as planner_module
from clip_engine.services.intelligence_planner import ClipPlanSegment
from clip_engine.services.transcription_service import TranscriptSegment, TranscriptWord, find_sentence_start_boundary
from tests.test_planner import clip, completion, make_planner, make_transcript


@pytest.fixture
def no_sleep(monkeypatch):
    async def instant(_):
        return None
    monkeypatch.setattr("clip_engine.services.intelligence_planner.asyncio.sleep", instant)


class RoutingClient(httpx.AsyncClient):
    """Answers each planner request from what it asks for, whatever order requests arrive in."""

    def __init__(self, answer):
        self.payloads = []

        class Body(httpx.AsyncByteStream):
            def __init__(self, body):
                self.body = body

            async def __aiter__(self):
                yield json.dumps(self.body).encode()

        async def handle(request):
            payload = json.loads(request.content)
            self.payloads.append(payload)
            status, body = answer(payload)
            return httpx.Response(status, stream=Body(body))

        super().__init__(base_url="https://example.invalid", transport=httpx.MockTransport(handle))


def user_text(payload) -> str:
    return "\n".join(part["text"] for part in payload["messages"][1]["content"] if part["type"] == "text")


def part_span(payload) -> tuple[float, float]:
    """First and last transcript times (seconds) a request carries."""
    times = [(float(a), float(b)) for a, b in re.findall(r"\] \[([\d.]+) - ([\d.]+)\]", user_text(payload))]
    return times[0][0], times[-1][1]


def plan(planner, seconds, **kwargs):
    args = dict(transcript_result=make_transcript(seconds), video_metadata=SimpleNamespace(duration_seconds=seconds),
                max_clips=6, auto_clip_count=False, min_duration_seconds=15, max_duration_seconds=60)
    args.update(kwargs)
    return asyncio.run(planner.plan_clips(**args))


class TestUpToNAndMinimumScore:
    def scored(self, *scores):
        return [ClipPlanSegment(i * 100_000, i * 100_000 + 30_000, s) for i, s in enumerate(scores)]

    def test_weak_clips_are_dropped_and_the_best_two_stay_when_none_pass(self):
        planner = make_planner()
        kept = planner._finalize_clips(self.scored(0.9, 0.5, 0.7, 0.4), clip_count=10, min_score=0.6)
        assert [c.virality_score for c in kept] == [0.9, 0.7]
        kept = planner._finalize_clips(self.scored(0.5, 0.3, 0.55, 0.2), clip_count=10, min_score=0.6)
        assert [c.virality_score for c in kept] == [0.55, 0.5]
        assert len(planner._finalize_clips(self.scored(0.5), clip_count=10, min_score=0.6)) == 1

    def test_a_plan_reports_what_it_left_out(self):
        planner = make_planner()
        planner._http_client = RoutingClient(lambda p: (200, completion(json.dumps({"insights": "Podcast", "clips": [
            clip(10, 40, (9, 9, 9, 9, 9)), clip(100, 130, (5, 5, 5, 5, 5)), clip(200, 230, (4, 4, 4, 4, 4))]}))))
        result = plan(planner, 300)
        assert [round(c.virality_score, 2) for c in result.segments] == [0.9]
        assert result.skipped_weak == 2
        assert "2 weaker moments scored below 60/100" in result.insights

    def test_a_clip_the_model_did_not_score_is_never_dropped_for_it(self):
        planner = make_planner()
        unscored = {"start_time": 100, "end_time": 130, "summary": "No Scores Given"}
        planner._http_client = RoutingClient(lambda p: (200, completion(json.dumps({"clips": [
            clip(10, 40, (9, 9, 9, 9, 9)), unscored]}))))
        result = plan(planner, 300)
        assert len(result.segments) == 2 and result.skipped_weak == 0

    @pytest.mark.parametrize("kind", ["clip_request", "jev"])
    def test_a_clip_request_and_jev_review_have_no_score_floor(self, kind):
        planner = make_planner()
        planner._http_client = RoutingClient(lambda p: (200, completion(json.dumps({"clips": [
            clip(10, 40, (4, 4, 4, 4, 4)), clip(100, 130, (3, 3, 3, 3, 3)), clip(200, 230, (2, 2, 2, 2, 2))]}))))
        extra = {"clip_request": "the pricing debate"} if kind == "clip_request" else {"jev_enabled": True}
        result = plan(planner, 300, **extra)
        assert len(result.segments) == 3 and result.skipped_weak == 0

    def test_the_count_never_exceeds_what_the_person_asked_for(self):
        planner = make_planner()
        assert planner.calculate_optimal_clip_count(3600, 1, auto_clip_count=True) == 1
        assert planner.calculate_optimal_clip_count(3600, None, auto_clip_count=True) >= 3

    def test_the_minimum_score_setting_is_bounded(self):
        from clip_engine.config import Settings
        assert Settings(_env_file=None).min_clip_score == 0.6
        assert Settings(_env_file=None, min_clip_score="0.7").min_clip_score == 0.7
        # Written the way the app shows scores.
        assert Settings(_env_file=None, min_clip_score="60").min_clip_score == 0.6
        assert Settings(_env_file=None, min_clip_score="150").min_clip_score == 1.0
        assert Settings(_env_file=None, min_clip_score="nope").min_clip_score == 0.6

    def test_gaming_streams_are_a_content_type(self):
        prompt = make_planner()._build_system_prompt(5, 15, 60, None)
        assert "Gaming/Livestream: Prioritize clutch plays" in prompt
        assert "quotability by how replayable the reaction" in prompt


class TestLongSourcesAreSearchedInParts:
    def answer_inside(self, scores_by_part=None, outside=False):
        """Two clips from the middle of whatever part was asked; optionally one outside it."""
        def answer(payload):
            first, last = part_span(payload)
            middle = (first + last) / 2
            number = int(re.search(r"part (\d+) of", user_text(payload)).group(1)) if "part " in user_text(payload) else 1
            score = (scores_by_part or {}).get(number, 8)
            clips = [clip(middle, middle + 30, (score,) * 5), clip(middle + 100, middle + 130, (7,) * 5)]
            if outside:
                clips.append(clip(last + 300, last + 330, (10,) * 5))
            return 200, completion(json.dumps({"insights": f"Part {number}", "clips": clips}))
        return answer

    def test_a_short_source_is_one_request(self):
        planner = make_planner()
        planner._http_client = RoutingClient(self.answer_inside())
        plan(planner, 1800)
        [payload] = planner._http_client.payloads
        assert "this transcript is part" not in user_text(payload).lower()

    def test_ninety_minutes_go_in_overlapping_parts_with_their_own_line_numbers(self, no_sleep):
        planner = make_planner()
        planner._http_client = RoutingClient(self.answer_inside())
        seen = []
        result = plan(planner, 5400, max_clips=6, progress=seen.append)
        payloads = sorted(planner._http_client.payloads, key=lambda p: part_span(p)[0])
        spans = [part_span(p) for p in payloads]
        assert len(payloads) == 4
        # Parts of at most ~25 minutes (the last takes the tail), overlapping by about a minute.
        assert spans[0][0] == 0 and spans[-1][1] >= 5395
        for (a0, a1), (b0, b1) in zip(spans, spans[1:]):
            assert 0 < a1 - b0 <= 65 and a1 - a0 <= 25 * 60
        for payload, (first, _) in zip(payloads, spans):
            text = user_text(payload)
            # Lines keep their numbers in the whole transcript.
            assert f"[segment {int(first // 5)}] [{first:.1f}" in text
            assert re.search(r"part \d of 4, covering", text)
            assert "Return up to " in payload["messages"][0]["content"]
        assert len(seen) == 4 and all(m.startswith("Searching part") for m in seen)
        # Ranked together: the best clips of every part, capped at the six asked for.
        assert len(result.segments) == 6
        assert result.api_costs.attempts == 4 and result.api_costs.estimated_cost_usd == pytest.approx(4 * 0.0123)
        assert all(f"Part {n} of 4" in result.insights for n in range(1, 5))

    def test_the_best_part_wins_the_ranking_and_clips_outside_a_part_are_dropped(self, no_sleep):
        planner = make_planner()
        planner._http_client = RoutingClient(self.answer_inside(scores_by_part={3: 10}, outside=True))
        result = plan(planner, 5400, max_clips=1)
        [best] = result.segments
        third = sorted(planner._http_client.payloads, key=lambda p: part_span(p)[0])[2]
        first, last = part_span(third)
        assert first * 1000 <= best.start_time_ms <= last * 1000

    def test_one_part_failing_keeps_the_others(self, no_sleep):
        planner = make_planner()
        good = self.answer_inside()

        def answer(payload):
            if part_span(payload)[0] == 0:
                return 503, {"error": {"message": "overloaded"}}
            return good(payload)

        planner._http_client = RoutingClient(answer)
        result = plan(planner, 5400)
        assert result.segments and "Part 1 of 4 (0-25 min) could not be searched" in result.insights

    def test_a_rejected_key_stops_the_whole_plan(self, no_sleep):
        planner = make_planner()
        planner._http_client = RoutingClient(lambda p: (401, {"error": {"message": "bad key"}}))
        with pytest.raises(planner_module.IntelligencePlanningError):
            plan(planner, 5400)

    def test_dense_speech_is_split_before_it_overflows_a_small_context(self):
        planner = make_planner()
        words = " ".join(["talking"] * 60)
        segments = [TranscriptSegment(t * 1000, t * 1000 + 4500, words) for t in range(0, 20 * 60, 5)]
        windows = planner._search_windows(segments)
        assert len(windows) > 1
        for first, last in windows:
            assert len(planner._build_transcript_text(segments[first:last], start_index=first)) <= planner_module.SEARCH_WINDOW_MAX_CHARS + 500

    def test_jev_with_a_preferred_range_stays_one_request(self, no_sleep):
        planner = make_planner()
        planner._http_client = RoutingClient(self.answer_inside())
        plan(planner, 5400, jev_enabled=True, start_time_seconds=600, end_time_seconds=900)
        assert len(planner._http_client.payloads) == 1


def words(*items):
    return [TranscriptWord(word=w, start_time_ms=a, end_time_ms=b) for w, a, b in items]


class TestSentenceStarts:
    def test_a_start_inside_a_sentence_moves_back_to_its_beginning(self):
        segment = TranscriptSegment(0, 2800, "And that was it. Now the big secret", words=words(
            ("And", 0, 300), ("that", 300, 600), ("was", 600, 900), ("it.", 900, 1200),
            ("Now", 1500, 1800), ("the", 1800, 2000), ("big", 2000, 2300), ("secret", 2300, 2800)))
        # Inside "big": the sentence "Now the big secret" starts at 1.5 s.
        assert find_sentence_start_boundary([segment], 2100) == 1500

    def test_a_pause_counts_as_a_sentence_start_without_punctuation(self):
        segment = TranscriptSegment(0, 2200, "yeah go watch this", words=words(
            ("yeah", 0, 300), ("go", 300, 600), ("watch", 1600, 1900), ("this", 1900, 2200)))
        assert find_sentence_start_boundary([segment], 2000) == 1600

    def test_with_no_start_behind_it_moves_forward_to_the_next_sentence(self):
        rambling = [(f"w{i}", i * 300, i * 300 + 300) for i in range(19)] + [("done.", 5700, 6000), ("Watch", 6100, 6400)]
        segment = TranscriptSegment(0, 6400, "", words=words(*rambling))
        assert find_sentence_start_boundary([segment], 4850, max_adjustment_ms=3000, max_forward_ms=2000) == 6100
        # Without leave to move forward it still never starts mid-word.
        assert find_sentence_start_boundary([segment], 4850, max_adjustment_ms=3000) == 4800

    def test_the_planner_moves_forward_only_when_the_clip_stays_long_enough(self, no_sleep):
        # make_transcript: a sentence every 5 s ("wordN" then "end." 2-4.5 s in).
        for end, expected_start in ((60, 15_000), (29, 12_000)):
            planner = make_planner()
            planner._http_client = RoutingClient(lambda p, e=end: (200, completion(json.dumps({"clips": [clip(14, e)]}))))
            [segment] = plan(planner, 300).segments
            assert segment.start_time_ms == expected_start, end


class TestReviewFindings:
    def test_a_long_break_in_speech_does_not_multiply_the_parts(self):
        planner = make_planner()
        talk = [TranscriptSegment(t * 1000, t * 1000 + 4500, "talking here.") for t in range(0, 600, 5)]
        later = [TranscriptSegment(t * 1000, t * 1000 + 4500, "back again.") for t in range(6600, 7200, 5)]
        windows = planner._search_windows(talk + later)
        # Two stretches of ten minutes around a 100-minute break: two parts, no near-copies.
        assert windows == [(0, len(talk)), (len(talk), len(talk) + len(later))]

    def test_insights_that_are_not_text_do_not_break_the_plan(self, no_sleep):
        planner = make_planner()
        planner._http_client = RoutingClient(lambda p: (200, completion(json.dumps({
            "insights": {"type": "Gaming/Livestream"}, "clips": [clip(10, 40)]}))))
        result = plan(planner, 300)
        assert "Gaming/Livestream" in result.insights and len(result.segments) == 1

    def test_a_rounded_line_time_keeps_its_own_sentence(self):
        segments = [TranscriptSegment(10_000, 10_600, "Yeah.", words=words(("Yeah.", 10_000, 10_600))),
                    TranscriptSegment(12_340, 14_000, "Watch this jump", words=words(
                        ("Watch", 12_340, 12_700), ("this", 12_700, 13_000), ("jump", 13_000, 14_000)))]
        # The model copied "[12.3 - 14.0]": it means "Watch", not the "Yeah." before it.
        assert find_sentence_start_boundary(segments, 12_300) == 12_340
        assert find_sentence_start_boundary(segments, 12_000) == 12_340

    def test_the_first_word_after_a_long_silence_starts_a_sentence(self):
        segments = [TranscriptSegment(60_000, 80_000, "earlier talk", words=words(("earlier", 60_000, 70_000), ("talk", 70_000, 80_000))),
                    TranscriptSegment(99_000, 101_000, "did you see that", words=words(
                        ("did", 99_000, 99_400), ("you", 99_400, 99_700), ("see", 99_700, 100_200), ("that", 100_200, 101_000)))]
        assert find_sentence_start_boundary(segments, 100_000) == 99_000

    def test_clips_snapped_back_before_a_part_still_count(self, no_sleep, monkeypatch):
        planner = make_planner()
        # Each start lands in a sentence that began 2.5 s earlier, before the part's first line.
        monkeypatch.setattr(planner_module, "find_sentence_start_boundary", lambda segments, timestamp_ms, *a, **k: timestamp_ms - 2500)

        def answer(payload):
            first, _ = part_span(payload)
            return 200, completion(json.dumps({"clips": [clip(first + 0.5, first + 40)]}))

        planner._http_client = RoutingClient(answer)
        result = plan(planner, 5400, max_clips=10)
        assert len(result.segments) == 4

    def test_scores_written_oddly_are_still_read(self):
        planner = make_planner()
        dims = ("hook", "standalone", "arc", "quotability", "ending")
        assert planner._score_clip({"scores": dict.fromkeys(dims, "8/10")}) == 0.8
        assert planner._score_clip({"scores": dict.fromkeys(dims, 0.75)}) == 0.75
        assert planner._score_clip({"scores": dict.fromkeys(dims, 7)}) == 0.7

    def test_a_rejected_key_stops_the_parts_still_waiting(self, no_sleep):
        planner = make_planner(ai_provider="openai-compatible", openai_compatible_url="http://localhost:1234/v1",
                               openai_compatible_model="m")
        planner._http_client = RoutingClient(lambda p: (401, {"error": {"message": "bad key"}}))
        with pytest.raises(planner_module.IntelligencePlanningError):
            plan(planner, 5400)
        # One part at a time on a custom endpoint: the first refusal stops the rest.
        assert len(planner._http_client.payloads) == 1

    def test_parts_that_all_fail_or_find_nothing_report_the_failure(self, no_sleep):
        planner = make_planner()

        def answer(payload):
            if part_span(payload)[0] == 0:
                return 503, {"error": {"message": "overloaded"}}
            return 200, completion(json.dumps({"clips": []}))

        planner._http_client = RoutingClient(answer)
        with pytest.raises(planner_module.IntelligencePlanningError):
            plan(planner, 5400)

    def test_review_mode_on_a_long_source_searches_in_parts(self, no_sleep):
        planner = make_planner()

        def answer(payload):
            first, last = part_span(payload)
            middle = (first + last) / 2
            return 200, completion(json.dumps({"clips": [clip(middle, middle + 30)]}))

        planner._http_client = RoutingClient(answer)
        result = plan(planner, 5400, jev_enabled=True)
        assert len(planner._http_client.payloads) == 4
        assert "Preferred discovery range" in planner._http_client.payloads[0]["messages"][0]["content"]
        assert len(result.segments) == 4 and result.skipped_weak == 0
