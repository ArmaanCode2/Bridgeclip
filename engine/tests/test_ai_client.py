"""Offline tests for the Gemini / OpenAI-compatible chat client."""

import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest

from clip_engine.config import Settings
from clip_engine.services import ai_client
from clip_engine.services.ai_client import (
    GEMINI_OPENAI_BASE_URL,
    AiClientError,
    ai_chat_completion,
    base_url_for,
    get_ai_client,
    prepare_payload,
    supports_images,
    tolerant_json_parse,
)
from clip_engine.services.openrouter import json_schema_format

SCHEMA = {"type": "object", "properties": {"clips": {"type": "array", "items": {"type": "string"}}},
          "required": ["clips"], "additionalProperties": False}


def settings(**overrides):
    values = dict(
        ai_provider="gemini", gemini_api_key="gem-key", gemini_model="gemini-test",
        openai_compatible_url="http://localhost:11434/v1", openai_compatible_api_key="",
        openai_compatible_model="llama-test", openai_compatible_vision=False,
        openrouter_api_key="or-key", openrouter_base_url="https://openrouter.ai/api/v1",
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def request_payload():
    return {
        "model": "anthropic/claude-opus-5.5",
        "messages": [{"role": "system", "content": "sys"}, {"role": "user", "content": "plan these"}],
        "max_tokens": 1000,
        "response_format": json_schema_format("clip_plan", SCHEMA),
        "plugins": [{"id": "response-healing"}],
        "provider": {"require_parameters": True},
        "models": ["a/b"],
        "reasoning": {"effort": "high", "exclude": True},
    }


def ok_body(content='{"clips": []}', model="gemini-test"):
    return {"model": model, "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}


class _Body(httpx.AsyncByteStream):
    def __init__(self, data):
        self.data = data

    async def __aiter__(self):
        yield self.data


def reply(status, json=None, content=None, headers=None):
    """A response that is streamed, as a real server's is.

    httpx.MockTransport reads responses up front, which the client's streaming
    reader (aiter_raw) would then refuse to read a second time.
    """
    import json as jsonlib

    data = content if content is not None else jsonlib.dumps(json if json is not None else {}).encode()
    return httpx.Response(status, headers=headers, stream=_Body(data))


class _Transport(httpx.AsyncBaseTransport):
    def __init__(self, handler):
        self.handler = handler

    async def handle_async_request(self, request):
        await request.aread()
        return self.handler(request)


def client_for(handler):
    return httpx.AsyncClient(base_url="https://provider.test/v1", transport=_Transport(handler))


@pytest.fixture(autouse=True)
def fresh_state(monkeypatch):
    ai_client._working_level.clear()
    ai_client._token_settings.clear()
    ai_client._no_temperature.clear()

    async def no_sleep(_seconds):
        return None

    monkeypatch.setattr(ai_client.asyncio, "sleep", no_sleep)


def run(coro):
    return asyncio.run(coro)


class TestPreparePayload:
    def test_openrouter_payload_is_untouched(self):
        payload = request_payload()
        assert prepare_payload(settings(ai_provider="openrouter"), payload) is payload

    def test_gemini_drops_openrouter_fields_and_keeps_strict_schema(self):
        prepared = prepare_payload(settings(), request_payload())
        for field in ("plugins", "provider", "models", "reasoning"):
            assert field not in prepared
        assert prepared["model"] == "gemini-test"
        # Gemini accepts structured output, so it must not be downgraded up front.
        assert prepared["response_format"]["type"] == "json_schema"
        assert prepared["temperature"] == 0.2

    def test_openai_compatible_uses_configured_model(self):
        prepared = prepare_payload(settings(ai_provider="openai-compatible"), request_payload())
        assert prepared["model"] == "llama-test"

    @pytest.mark.parametrize("model", ["o3-mini", "gpt-5", "openai/o1"])
    def test_openai_reasoning_models_get_no_temperature(self, model):
        prepared = prepare_payload(settings(ai_provider="openai-compatible", openai_compatible_model=model), request_payload())
        assert "temperature" not in prepared

    @pytest.mark.parametrize("model", ["llama3.1", "gemma-2-27b", "qwen2.5-coder"])
    def test_ordinary_models_keep_temperature(self, model):
        prepared = prepare_payload(settings(ai_provider="openai-compatible", openai_compatible_model=model), request_payload())
        assert prepared["temperature"] == 0.2

    def test_openai_compatible_without_a_model_is_a_clear_error(self):
        with pytest.raises(AiClientError) as raised:
            prepare_payload(settings(ai_provider="openai-compatible", openai_compatible_model=""), request_payload())
        assert raised.value.reason == "model_missing"


class TestClientConstruction:
    def test_gemini_uses_google_endpoint_and_bearer_key(self):
        client = get_ai_client(settings())
        assert str(client.base_url).rstrip("/") == GEMINI_OPENAI_BASE_URL
        assert client.headers["authorization"] == "Bearer gem-key"

    def test_openai_compatible_without_key_sends_no_authorization(self):
        client = get_ai_client(settings(ai_provider="openai-compatible"))
        assert str(client.base_url).rstrip("/") == "http://localhost:11434/v1"
        assert "authorization" not in client.headers

    def test_openai_compatible_with_key_sends_it(self):
        client = get_ai_client(settings(ai_provider="openai-compatible", openai_compatible_api_key="sk-x"))
        assert client.headers["authorization"] == "Bearer sk-x"

    def test_openrouter_client_matches_the_original_headers(self):
        client = get_ai_client(settings(ai_provider="openrouter"))
        assert client.headers["authorization"] == "Bearer or-key"
        assert client.headers["x-title"] == "BridgeClip AI Clipping Agent"

    @pytest.mark.parametrize("url", ["", "ftp://host/v1", "http://user:pw@host/v1", "not a url"])
    def test_bad_endpoint_urls_are_refused(self, url):
        with pytest.raises(AiClientError) as raised:
            base_url_for(settings(ai_provider="openai-compatible", openai_compatible_url=url))
        assert raised.value.reason == "url_missing"

    def test_image_support_follows_the_provider(self):
        assert supports_images(settings(ai_provider="openrouter"))
        assert supports_images(settings(ai_provider="gemini"))
        assert not supports_images(settings(ai_provider="openai-compatible"))
        assert supports_images(settings(ai_provider="openai-compatible", openai_compatible_vision=True))


class TestStructuredOutputLadder:
    def test_schema_request_is_sent_first_and_succeeds(self):
        seen = []

        def handler(request):
            seen.append(json.loads(request.content))
            return reply(200, json=ok_body())

        body, usage = run(ai_chat_completion(client_for(handler), prepare_payload(settings(), request_payload())))
        assert len(seen) == 1
        assert seen[0]["response_format"]["type"] == "json_schema"
        assert usage["prompt_tokens"] == 10 and usage["cost"] is None
        # A server can accept response_format and ignore it, so the schema is in the prompt too.
        assert '"clips"' in seen[0]["messages"][-1]["content"]

    def test_falls_back_to_json_mode_and_puts_the_schema_in_the_prompt(self):
        seen = []

        def handler(request):
            payload = json.loads(request.content)
            seen.append(payload)
            if payload.get("response_format", {}).get("type") == "json_schema":
                return reply(400, json={"error": {"message": "unsupported schema"}})
            return reply(200, json=ok_body())

        run(ai_chat_completion(client_for(handler), prepare_payload(settings(), request_payload())))
        assert [r.get("response_format", {}).get("type") for r in seen] == ["json_schema", "json_object"]
        # Without the schema in the prompt a JSON-mode model has no idea what to return.
        assert '"clips"' in seen[1]["messages"][-1]["content"]

    def test_falls_back_to_plain_text_when_json_mode_is_also_refused(self):
        seen = []

        def handler(request):
            payload = json.loads(request.content)
            seen.append(payload)
            if "response_format" in payload:
                return reply(422, json={"error": {"message": "nope"}})
            return reply(200, json=ok_body())

        run(ai_chat_completion(client_for(handler), prepare_payload(settings(), request_payload())))
        assert len(seen) == 3 and "response_format" not in seen[2]
        assert '"clips"' in seen[2]["messages"][-1]["content"]

    def test_remembers_the_level_that_worked(self):
        seen = []

        def handler(request):
            payload = json.loads(request.content)
            seen.append(payload)
            if payload.get("response_format", {}).get("type") == "json_schema":
                return reply(400, json={})
            return reply(200, json=ok_body())

        client = client_for(handler)
        prepared = prepare_payload(settings(), request_payload())
        run(ai_chat_completion(client, prepared))
        run(ai_chat_completion(client, prepared))
        assert len(seen) == 3  # the second call goes straight to JSON mode

    def test_original_messages_are_not_mutated(self):
        def handler(request):
            payload = json.loads(request.content)
            if payload.get("response_format", {}).get("type") == "json_schema":
                return reply(400, json={})
            return reply(200, json=ok_body())

        prepared = prepare_payload(settings(), request_payload())
        run(ai_chat_completion(client_for(handler), prepared))
        assert prepared["messages"][-1]["content"] == "plan these"

    def test_vision_messages_get_the_schema_as_an_extra_text_part(self):
        seen = []

        def handler(request):
            payload = json.loads(request.content)
            seen.append(payload)
            if payload.get("response_format", {}).get("type") == "json_schema":
                return reply(400, json={})
            return reply(200, json=ok_body())

        payload = request_payload()
        payload["messages"] = [{"role": "user", "content": [
            {"type": "text", "text": "what is in the frame"},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AAAA"}}]}]
        run(ai_chat_completion(client_for(handler), prepare_payload(settings(), payload)))
        parts = seen[1]["messages"][0]["content"]
        assert parts[1]["type"] == "image_url" and parts[-1]["type"] == "text" and '"clips"' in parts[-1]["text"]


class TestOutputTokenAdaptation:
    """OpenAI-style servers reject the planner's 32,000-token request in two ways."""

    def seen_after(self, error_message, accept):
        seen = []

        def handler(request):
            payload = json.loads(request.content)
            seen.append(payload)
            if accept(payload):
                return reply(200, json=ok_body())
            return reply(400, json={"error": {"message": error_message}})

        prepared = prepare_payload(settings(ai_provider="openai-compatible"), request_payload())
        prepared["max_tokens"] = 32000
        client = client_for(handler)
        run(ai_chat_completion(client, prepared))
        run(ai_chat_completion(client, prepared))
        return seen

    def test_max_tokens_is_swapped_for_max_completion_tokens(self):
        seen = self.seen_after(
            "Unsupported parameter: 'max_tokens' is not supported with this model. Use 'max_completion_tokens' instead.",
            lambda p: "max_completion_tokens" in p and "max_tokens" not in p)
        assert "max_tokens" in seen[0] and "max_completion_tokens" in seen[1]
        # The structured-output format is untouched: only the token parameter changed.
        assert seen[1]["response_format"]["type"] == "json_schema"
        # Learned once: the second call does not repeat the failed attempt.
        assert len(seen) == 3 and "max_completion_tokens" in seen[2]

    def test_a_too_large_limit_is_lowered_to_what_the_model_allows(self):
        seen = self.seen_after(
            "max_tokens is too large: 32000. This model supports at most 16384 completion tokens, whereas you provided 32000.",
            lambda p: p.get("max_tokens", 0) <= 16384)
        assert seen[1]["max_tokens"] == 16384 and seen[1]["response_format"]["type"] == "json_schema"
        assert len(seen) == 3 and seen[2]["max_tokens"] == 16384

    def test_a_limit_larger_than_the_request_is_not_applied(self):
        def handler(request):
            return reply(400, json={"error": {"message": "max_tokens must be at most 99999"}})

        with pytest.raises(AiClientError):
            run(ai_chat_completion(client_for(handler), prepare_payload(settings(), request_payload())))

    def test_unrelated_400s_still_use_the_structured_output_ladder(self):
        seen = self.seen_after("something else is wrong", lambda p: p.get("response_format", {}).get("type") != "json_schema")
        assert seen[0]["response_format"]["type"] == "json_schema" and seen[1]["response_format"]["type"] == "json_object"


class TestErrorMapping:
    @pytest.mark.parametrize("status,reason,retryable", [
        (401, "auth", False), (403, "auth", False), (402, "quota", False),
        (404, "endpoint_not_found", False), (429, "rate_limit", True),
        (500, "unavailable", True), (503, "unavailable", True),
    ])
    def test_status_codes_map_to_stable_reasons(self, status, reason, retryable):
        def handler(request):
            return reply(status, json={"error": {"message": "secret detail"}})

        with pytest.raises(AiClientError) as raised:
            run(ai_chat_completion(client_for(handler), prepare_payload(settings(), request_payload())))
        assert raised.value.reason == reason and raised.value.retryable is retryable
        # Provider text is kept for logs but never put in the user-facing message.
        assert "secret detail" not in str(raised.value)
        assert "secret detail" in raised.value.detail

    def test_a_400_that_no_level_can_fix_is_reported(self):
        def handler(request):
            return reply(400, json={"error": {"message": "context too long"}})

        with pytest.raises(AiClientError) as raised:
            run(ai_chat_completion(client_for(handler), prepare_payload(settings(), request_payload())))
        assert raised.value.reason == "bad_request"

    def test_network_failure_is_retryable(self):
        def handler(request):
            raise httpx.ConnectError("down")

        with pytest.raises(AiClientError) as raised:
            run(ai_chat_completion(client_for(handler), prepare_payload(settings(), request_payload())))
        assert raised.value.reason == "network" and raised.value.retryable

    def test_error_inside_a_200_body_is_retryable(self):
        def handler(request):
            return reply(200, json={"error": {"message": "upstream"}})

        with pytest.raises(AiClientError) as raised:
            run(ai_chat_completion(client_for(handler), prepare_payload(settings(), request_payload())))
        assert raised.value.retryable

    def test_garbage_body_is_an_invalid_response(self):
        def handler(request):
            return reply(200, content=b"<html>proxy login</html>")

        with pytest.raises(AiClientError) as raised:
            run(ai_chat_completion(client_for(handler), prepare_payload(settings(), request_payload())))
        assert raised.value.reason == "invalid_response"

    def test_rate_limit_with_short_retry_after_is_retried(self):
        calls = []

        def handler(request):
            calls.append(1)
            if len(calls) < 3:
                return reply(429, headers={"Retry-After": "2"}, json={})
            return reply(200, json=ok_body())

        run(ai_chat_completion(client_for(handler), prepare_payload(settings(), request_payload())))
        assert len(calls) == 3

    def test_rate_limit_with_long_retry_after_is_not_waited_out(self):
        calls = []

        def handler(request):
            calls.append(1)
            return reply(429, headers={"Retry-After": "3600"}, json={})

        with pytest.raises(AiClientError) as raised:
            run(ai_chat_completion(client_for(handler), prepare_payload(settings(), request_payload())))
        assert len(calls) == 1 and raised.value.reason == "rate_limit"

    def test_oversized_response_is_refused(self, monkeypatch):
        monkeypatch.setattr(ai_client, "MAX_CHAT_RESPONSE_BYTES", 100)

        def handler(request):
            return reply(200, content=b"x" * 500)

        with pytest.raises(AiClientError):
            run(ai_chat_completion(client_for(handler), prepare_payload(settings(), request_payload())))

    def test_redirects_are_not_followed(self):
        def handler(request):
            return reply(302, headers={"Location": "http://169.254.169.254/"})

        with pytest.raises(AiClientError) as raised:
            run(ai_chat_completion(client_for(handler), prepare_payload(settings(), request_payload())))
        assert raised.value.reason == "unavailable" and not raised.value.retryable


class TestTolerantJson:
    def test_plain_json(self):
        assert tolerant_json_parse('{"a": 1}') == {"a": 1}

    def test_code_fence(self):
        assert tolerant_json_parse('```json\n{"clips": [1, 2]}\n```') == {"clips": [1, 2]}

    def test_prose_and_trailing_commas(self):
        raw = 'Here is the plan:\n{"clips": [{"start": 10, "end": 20,},],}\nHope this helps!'
        assert tolerant_json_parse(raw) == {"clips": [{"start": 10, "end": 20}]}

    def test_bare_array(self):
        assert tolerant_json_parse("Result: [1, 2, 3]") == [1, 2, 3]

    @pytest.mark.parametrize("raw", ["", "no json here", "{broken"])
    def test_unrecoverable_input_raises_value_error(self, raw):
        with pytest.raises(ValueError):
            tolerant_json_parse(raw)


class TestSettingsGuards:
    def test_only_web_research_is_forced_off_for_other_providers(self):
        # Review and visual context run on the chosen model; web research is an OpenRouter tool.
        s = Settings(_env_file=None, ai_provider="gemini", jev_enabled=True,
                     jev_visual_context=True, source_context_web_research=True)
        assert (s.jev_enabled, s.jev_visual_context, s.source_context_web_research) == (True, True, False)

    def test_beta_features_keep_their_values_on_openrouter(self):
        s = Settings(_env_file=None, ai_provider="openrouter", jev_enabled=True, source_context_web_research=True)
        assert s.jev_enabled and s.source_context_web_research

    def test_unknown_provider_is_rejected(self):
        with pytest.raises(Exception):
            Settings(_env_file=None, ai_provider="grok")

    def test_defaults_keep_existing_behaviour(self):
        s = Settings(_env_file=None)
        assert s.ai_provider == "openrouter" and s.transcription_source == "cloud"
        assert s.transcription_provider == "openrouter"


class TestPhase2AReliability:
    """Problems found with real Gemini and local servers."""

    def test_reasoning_before_the_answer_is_ignored(self):
        assert tolerant_json_parse('<think>compare {a} with [b]</think>\n{"clips": [1]}') == {"clips": [1]}
        assert tolerant_json_parse('<THINKING>x{</THINKING>```json\n[1, 2]\n```') == [1, 2]
        assert ai_client.strip_thinking("no reasoning here") == "no reasoning here"

    def test_a_reply_cut_off_inside_its_reasoning_has_no_json(self):
        with pytest.raises(ValueError):
            tolerant_json_parse('<think>still weighing {"clips": [1]} against')

    def test_a_rejected_temperature_is_dropped_and_remembered(self):
        seen = []

        def handler(request):
            payload = json.loads(request.content)
            seen.append(payload)
            if "temperature" in payload:
                return reply(400, json={"error": {"message": "Unsupported value: 'temperature' does not support 0.2 with this model."}})
            return reply(200, json=ok_body())

        client = client_for(handler)
        payload = prepare_payload(settings(ai_provider="openai-compatible"), request_payload())
        run(ai_chat_completion(client, payload))
        assert ["temperature" in p for p in seen] == [True, False]
        # Structured output was not weakened to get past it.
        assert seen[1]["response_format"]["type"] == "json_schema"
        run(ai_chat_completion(client, payload))
        assert len(seen) == 3 and "temperature" not in seen[2], "the next request leaves it out from the start"

    def test_a_missing_model_and_a_wrong_url_path_are_told_apart(self):
        def missing_model(request):
            return reply(404, json={"error": {"message": 'model "llama9" not found, try pulling it first'}})

        def wrong_path(request):
            return reply(404, content=b"404 page not found")

        for handler, reason, text in ((missing_model, "model_not_found", "AI model was not found"),
                                      (wrong_path, "endpoint_not_found", "AI endpoint was not found")):
            with pytest.raises(AiClientError) as raised:
                run(ai_chat_completion(client_for(handler), prepare_payload(settings(), request_payload())))
            assert raised.value.reason == reason and str(raised.value) == text

    @pytest.mark.parametrize("body", [
        [{"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "details": [
            {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "40s"}]}}],
        {"error": {"code": 429, "message": "Quota exceeded for metric. Please retry in 40.0s."}},
    ])
    def test_gemini_waits_as_long_as_the_error_body_asks(self, body, monkeypatch):
        waits, calls = [], []

        async def record_sleep(seconds):
            waits.append(seconds)

        monkeypatch.setattr(ai_client.asyncio, "sleep", record_sleep)

        def handler(request):
            calls.append(1)
            return reply(429, json=body) if len(calls) == 1 else reply(200, json=ok_body())

        run(ai_chat_completion(client_for(handler), prepare_payload(settings(), request_payload())))
        assert waits == [40.0] and len(calls) == 2

    def test_a_wait_longer_than_the_limit_is_not_waited_out(self):
        calls = []

        def handler(request):
            calls.append(1)
            return reply(429, json={"error": {"details": [{"retryDelay": "86400s"}]}})

        with pytest.raises(AiClientError) as raised:
            run(ai_chat_completion(client_for(handler), prepare_payload(settings(), request_payload())))
        assert len(calls) == 1 and raised.value.retry_after == 86400.0

    def test_a_custom_endpoint_waits_as_long_as_its_setting_says(self):
        assert ai_client.request_timeout(settings(ai_provider="openai-compatible", openai_compatible_timeout_seconds=3600.0), 120.0) == 3600.0
        assert ai_client.request_timeout(settings(ai_provider="openai-compatible"), 120.0) == ai_client.DEFAULT_OPENAI_COMPATIBLE_TIMEOUT_SECONDS
        assert ai_client.request_timeout(settings(ai_provider="gemini", openai_compatible_timeout_seconds=3600.0), 120.0) == 120.0
        client = get_ai_client(settings(ai_provider="openai-compatible", openai_compatible_timeout_seconds=900.0), timeout=120.0)
        assert client.timeout.read == 900.0
        run(client.aclose())

    def test_the_wait_setting_is_bounded(self):
        assert Settings(_env_file=None).openai_compatible_timeout_seconds == 1800.0
        assert Settings(_env_file=None, openai_compatible_timeout_seconds="600").openai_compatible_timeout_seconds == 600.0
        assert Settings(_env_file=None, openai_compatible_timeout_seconds=5).openai_compatible_timeout_seconds == 60.0
        assert Settings(_env_file=None, openai_compatible_timeout_seconds=10 ** 6).openai_compatible_timeout_seconds == 7200.0
        assert Settings(_env_file=None, openai_compatible_timeout_seconds="soon").openai_compatible_timeout_seconds == 1800.0


class TestReviewFixes:
    """Cases an adversarial review of Phase 2A found."""

    def test_a_reply_whose_template_opened_the_reasoning_is_read(self):
        # Qwen3-Thinking and DeepSeek-R1 templates open <think> themselves; the reply has only the end.
        assert tolerant_json_parse('weigh {a} and [b]\n</think>\n{"clips": ["x"]}') == {"clips": ["x"]}

    def test_a_title_that_mentions_the_tags_is_left_alone(self):
        raw = '{"title": "Why models emit <think> then </think> tags"}'
        assert tolerant_json_parse(raw) == {"title": "Why models emit <think> then </think> tags"}
        assert tolerant_json_parse('<think>ok</think>' + raw)["title"] == "Why models emit <think> then </think> tags"

    def test_a_400_that_only_mentions_temperature_does_not_drop_it(self):
        seen = []

        def handler(request):
            payload = json.loads(request.content)
            seen.append(payload)
            if len(seen) == 1:
                return reply(400, json={"error": {"message": "Invalid schema; temperature, top_p and max_tokens are fine"}})
            return reply(200, json=ok_body())

        run(ai_chat_completion(client_for(handler), prepare_payload(settings(ai_provider="openai-compatible"), request_payload())))
        assert "temperature" in seen[1] and seen[1]["response_format"]["type"] == "json_object"

    def test_two_requests_at_once_both_drop_temperature_without_weakening_the_format(self):
        seen = []

        async def both():
            def handler(request):
                payload = json.loads(request.content)
                seen.append(payload)
                if "temperature" in payload:
                    return reply(400, json={"error": {"message": "temperature is not supported by this model"}})
                return reply(200, json=ok_body())

            client = client_for(handler)
            payload = prepare_payload(settings(ai_provider="openai-compatible"), request_payload())
            await asyncio.gather(ai_chat_completion(client, payload), ai_chat_completion(client, payload))

        run(both())
        assert all(p["response_format"]["type"] == "json_schema" for p in seen)
        assert not ai_client._working_level, "structured output was never stepped down"

    @pytest.mark.parametrize("model", ["gemini-3.8-flash", "gemini-3.7-flash", "google/gemini-3.1-pro-preview", "models/gemini-3-pro"])
    def test_gemini_3_keeps_its_default_temperature(self, model):
        for provider, field in (("gemini", "gemini_model"), ("openai-compatible", "openai_compatible_model")):
            payload = prepare_payload(settings(ai_provider=provider, **{field: model}), {**request_payload(), "temperature": 0.0})
            assert "temperature" not in payload, (provider, model)

    def test_older_gemini_and_other_models_keep_temperature(self):
        for model in ("gemini-2.5-flash", "llama3.1", "qwen3:8b"):
            payload = prepare_payload(settings(ai_provider="openai-compatible", openai_compatible_model=model), request_payload())
            assert payload["temperature"] == 0.2, model

    def test_a_daily_gemini_quota_is_not_waited_out(self):
        calls = []

        def handler(request):
            calls.append(1)
            return reply(429, json=[{"error": {"code": 429, "message": "Please retry in 41s.", "details": [
                {"@type": "type.googleapis.com/google.rpc.QuotaFailure", "violations": [
                    {"quotaMetric": "generativelanguage.googleapis.com/generate_content_free_tier_requests",
                     "quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}]},
                {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "41s"}]}}])

        with pytest.raises(AiClientError) as raised:
            run(ai_chat_completion(client_for(handler), prepare_payload(settings(), request_payload())))
        assert len(calls) == 1 and raised.value.reason == "rate_limit" and raised.value.retry_after is None

    def test_a_404_with_a_working_model_list_is_a_wrong_model(self):
        def handler(request):
            if request.method == "GET" and request.url.path.endswith("/models"):
                return reply(200, json={"data": []})
            return reply(404, content=b"Not Found")

        with pytest.raises(AiClientError) as raised:
            run(ai_chat_completion(client_for(handler), prepare_payload(settings(), request_payload())))
        assert raised.value.reason == "model_not_found"

    def test_a_top_level_message_is_read(self):
        def handler(request):
            return reply(404, json={"object": "error", "message": "The model `x` does not exist."})

        with pytest.raises(AiClientError) as raised:
            run(ai_chat_completion(client_for(handler), prepare_payload(settings(), request_payload())))
        assert raised.value.reason == "model_not_found" and "does not exist" in raised.value.detail

    def test_a_slow_answer_is_not_retried_but_an_unreachable_server_is(self):
        for exc, reason, retryable in ((httpx.ReadTimeout("slow"), "timeout", False),
                                       (httpx.ConnectTimeout("down"), "network", True),
                                       (httpx.ConnectError("refused"), "network", True)):
            def handler(request, exc=exc):
                raise exc

            with pytest.raises(AiClientError) as raised:
                run(ai_chat_completion(client_for(handler), prepare_payload(settings(), request_payload())))
            assert (raised.value.reason, raised.value.retryable) == (reason, retryable), type(exc).__name__

    def test_frame_checks_wait_less_than_the_planner(self):
        custom = settings(ai_provider="openai-compatible", openai_compatible_timeout_seconds=3600.0)
        assert ai_client.request_timeout(custom, 120.0, limit=300.0) == 300.0
        assert ai_client.request_timeout(settings(ai_provider="gemini"), 120.0, limit=300.0) == 120.0


def test_a_temperature_that_may_only_be_the_default_is_dropped():
    seen = []

    def handler(request):
        payload = json.loads(request.content)
        seen.append(payload)
        if "temperature" in payload:
            return reply(400, json={"error": {"message": "temperature may only be set to 1"}})
        return reply(200, json=ok_body())

    run(ai_chat_completion(client_for(handler), prepare_payload(settings(ai_provider="openai-compatible"), request_payload())))
    assert ["temperature" in p for p in seen] == [True, False]
