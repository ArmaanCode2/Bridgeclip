"""Offline tests for BridgeClip's OpenRouter transcription parsing and vocabulary hints."""

import pytest

from clip_engine.error_policy import safe_failure_code, safe_processing_error
from clip_engine.services.transcription_service import (
    TranscriptionError,
    TranscriptionProviderError,
    TranscriptionService,
    normalize_keyterms,
)


def parse(words, *, duration=20.0, text="spoken words", usage=None):
    service = TranscriptionService.__new__(TranscriptionService)
    response = {"text": text, "words": words, "language": "en", "usage": usage or {}}
    return service._parse_openrouter_response(response, duration)


def word(text, start, end, speaker=None):
    return {"word": text, "start": start, "end": end, "speaker": speaker}


def test_provider_http_400_keeps_a_safe_actionable_code():
    error = TranscriptionProviderError("bad_request", 400)
    assert safe_processing_error(error) == "Transcription request rejected by provider"
    assert safe_failure_code(error) == "transcription.bad_request"
    assert error.status_code == 400


def test_missing_word_timestamps_has_its_own_code():
    error = TranscriptionError("Transcription response did not include word timestamps", reason="missing_word_timestamps")
    assert safe_processing_error(error) == "Transcription response lacked word timestamps"
    assert safe_failure_code(error) == "transcription.missing_word_timestamps"


class TestParseTranscriptionResponse:
    def test_splits_on_speaker_change_and_sentence_end(self):
        result = parse([
            word("So", 0.0, 0.2, 0), word("anyway.", 0.3, 0.8, 0),
            word("No", 1.0, 1.2, 1), word("way.", 1.2, 1.6, 1),
        ])
        assert [s.text for s in result.segments] == ["So anyway.", "No way."]
        assert [s.speaker_label for s in result.segments] == ["S1", "S2"]

    def test_timestamps_stay_on_audio_timeline(self):
        result = parse([word("Hi.", 4.0, 8.0)])
        assert result.segments[0].start_time_ms == 4000
        assert result.segments[0].end_time_ms == 8000

    def test_no_diarization_leaves_speaker_blank(self):
        result = parse([word("Just", 0.0, 0.2), word("me.", 0.2, 0.5)])
        assert len(result.segments) == 1
        assert result.segments[0].speaker_label is None

    @pytest.mark.parametrize("words", [[word("bad", None, 1.0)], [word("bad", 2.0, 1.0)], [word("bad", 0.0, 25.0)]])
    def test_rejects_invalid_timestamps(self, words):
        with pytest.raises(TranscriptionProviderError):
            parse(words)

    def test_requires_word_timings_for_spoken_text(self):
        with pytest.raises(Exception, match="word timestamps"):
            parse([], text="Hello")

    def test_uses_returned_usage(self):
        result = parse([word("Hi.", 0.0, 0.2)], usage={"seconds": 3.5, "cost": 0.01})
        assert result.api_costs.audio_duration_seconds == 3.5
        assert result.api_costs.estimated_cost_usd == 0.01


class TestNormalizeKeyterms:
    def test_enforces_provider_limits(self):
        terms = ["  BridgeClip  ", "bridgeclip", "one two three four five six", "bad<chars>", "x" * 80, "", None]
        cleaned = normalize_keyterms(terms)
        assert cleaned[0] == "BridgeClip"
        assert "bridgeclip" not in cleaned
        assert not any("six" in t or "<" in t for t in cleaned)
        assert all(len(t) < 50 for t in cleaned)
        assert len(cleaned) == 2

    def test_caps_count(self):
        assert len(normalize_keyterms([f"term{i}" for i in range(1500)])) == 200
