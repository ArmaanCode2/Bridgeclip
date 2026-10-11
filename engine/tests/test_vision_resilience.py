"""The facecam AI stays on after a passing failure, and goes off only when it cannot work."""

import asyncio
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from clip_engine.services import layout_analyzer as la
from clip_engine.services.ai_client import AiClientError
from clip_engine.services.layout_analyzer import LayoutAnalyzer, LayoutType, ShotLayout


def keyframe(shade):
    image = np.full((90, 160, 3), shade, dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", image)
    assert ok
    return encoded.tobytes()


def analyzer_with(monkeypatch, outcomes):
    """An analyzer on Gemini whose vision calls return or raise the given outcomes in turn."""
    analyzer = LayoutAnalyzer()
    analyzer.settings = SimpleNamespace(
        ai_provider="gemini", gemini_api_key="key", gemini_model="gemini-test", layout_vision_enabled=True,
        layout_vision_model="google/gemini", layout_vision_reasoning_effort="low",
        get_layout_vision_fallback_models=lambda: [],
    )
    calls = []

    async def fake_completion(client, payload, settings=None):
        outcome = outcomes[min(len(calls), len(outcomes) - 1)]
        calls.append(payload)
        if isinstance(outcome, Exception):
            raise outcome
        return {"model": "gemini-test", "choices": [{"message": {"content": '{"layout": "talking_head"}'}}]}, {"cost": 0.0}

    async def fake_client():
        return None

    monkeypatch.setattr(la, "ai_chat_completion", fake_completion)
    monkeypatch.setattr(analyzer, "_get_client", fake_client)
    return analyzer, calls


def classify(analyzer, shade):
    heuristic = ShotLayout(start_ms=0, end_ms=1000, layout=LayoutType.TALKING_HEAD)
    return asyncio.run(analyzer._vision_classify(keyframe(shade), [], heuristic))


def bad_request():
    return AiClientError("AI provider rejected the request", reason="bad_request")


def test_one_rejected_frame_does_not_turn_vision_off(monkeypatch):
    analyzer, calls = analyzer_with(monkeypatch, [bad_request(), None])
    assert classify(analyzer, 10)[0] is None
    assert analyzer._vision_enabled()
    assert classify(analyzer, 200)[0] == {"layout": "talking_head"}
    assert len(calls) == 2


def test_the_same_rejection_twice_in_a_row_turns_vision_off(monkeypatch):
    analyzer, calls = analyzer_with(monkeypatch, [bad_request(), bad_request()])
    classify(analyzer, 10)
    classify(analyzer, 200)
    assert not analyzer._vision_enabled()


def test_a_success_in_between_resets_the_count(monkeypatch):
    analyzer, calls = analyzer_with(monkeypatch, [bad_request(), None, bad_request()])
    for shade in (10, 120, 240):
        classify(analyzer, shade)
    assert analyzer._vision_enabled()


@pytest.mark.parametrize("reason", ["auth", "model_not_found", "endpoint_not_found", "model_missing", "url_missing"])
def test_a_wrong_key_model_or_url_turns_vision_off_at_once(monkeypatch, reason):
    analyzer, calls = analyzer_with(monkeypatch, [AiClientError("x", reason=reason)])
    classify(analyzer, 10)
    assert not analyzer._vision_enabled()


@pytest.mark.parametrize("reason", ["timeout", "rate_limit", "quota"])
def test_a_model_too_slow_or_out_of_quota_stops_frame_checks(monkeypatch, reason):
    analyzer, calls = analyzer_with(monkeypatch, [AiClientError("x", reason=reason)])
    classify(analyzer, 10)
    assert not analyzer._vision_enabled()


def test_frame_checks_on_a_custom_endpoint_wait_at_most_five_minutes():
    analyzer = LayoutAnalyzer()
    analyzer.settings = SimpleNamespace(ai_provider="openai-compatible", openai_compatible_url="http://localhost:1234/v1",
                                        openai_compatible_api_key="", openai_compatible_timeout_seconds=7200.0)
    client = asyncio.run(analyzer._get_client())
    assert client.timeout.read == la.VISION_TIMEOUT_LIMIT_SECONDS == 300.0
    asyncio.run(client.aclose())
