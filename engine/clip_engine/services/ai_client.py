"""
Chat-completions client for AI providers other than OpenRouter.

OpenRouter keeps using ``openrouter.chat_completion`` unchanged. This module
serves Google Gemini (through Google's OpenAI-compatible endpoint) and any
user-supplied OpenAI-compatible endpoint (OpenAI, Ollama, LM Studio, vLLM...).

Compatible servers differ in what they accept. The JSON schema is always
written into the prompt as well, because some servers accept ``response_format``
and then ignore it without an error. A request starts with strict JSON-schema
structured output and, only if the server rejects that with a 400/422, steps
down to JSON mode and then to plain text. The level that worked is remembered
for the rest of the run, as are a rejected ``temperature`` and output-token
limits the server reported.
"""

import asyncio
import copy
import json
import logging
import re
from typing import Any, Optional
from urllib.parse import urlsplit

import httpx

from clip_engine.services.openrouter import (
    MAX_CHAT_RESPONSE_BYTES,
    RETRYABLE_STATUS_CODES,
    OpenRouterError,
    message_text,
)

logger = logging.getLogger(__name__)

GEMINI_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"
DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"

# Longest wait honoured for a 429, and how often to wait, before giving up and
# letting the caller's own retry policy take over. Gemini's per-minute limits
# ask for up to about a minute.
MAX_RATE_LIMIT_WAIT_SECONDS = 90.0
MAX_RATE_LIMIT_RETRIES = 2
# How long an OpenAI-compatible endpoint may take to answer when no setting says.
# Local models on a CPU can need many minutes for a long transcript.
DEFAULT_OPENAI_COMPATIBLE_TIMEOUT_SECONDS = 1800.0

# OpenAI-style reasoning models reject a temperature value, and Google asks for
# Gemini 3 models to keep their default of 1.0: lower values can make them loop.
_NO_TEMPERATURE_MODEL = re.compile(r"(^|[/:._-])(o\d|gpt-5)|(^|/)(models/)?gemini-3", re.IGNORECASE)
# A 400 about temperature that says the value is not accepted, rather than one
# that only mentions it among other parameters.
_TEMPERATURE_REFUSED = ("unsupported", "not supported", "does not support", "only the default", "only default", "not allowed",
                        "may only")

# (base_url, model) -> structured-output level that the server accepted.
_working_level: dict[tuple[str, str], int] = {}
# (base_url, model) -> output-token settings the server told us it needs:
# {"param": "max_tokens" | "max_completion_tokens", "limit": int | None}.
_token_settings: dict[tuple[str, str], dict[str, Any]] = {}
MAX_TOKEN_ADJUSTMENTS = 2
# (base_url, model) pairs whose server rejected a temperature value.
_no_temperature: set[tuple[str, str]] = set()


class AiClientError(OpenRouterError):
    """A non-OpenRouter provider request failed.

    ``reason`` is a short stable code. ``detail`` is the provider's own message
    (truncated); it is for logs only and is never shown to the user, because it
    can echo request content.
    """

    def __init__(self, message: str, retryable: bool = False, reason: str = "error",
                 status: Optional[int] = None, detail: str = "", retry_after: Optional[float] = None):
        super().__init__(message, retryable=retryable)
        self.reason = reason
        self.status = status
        self.detail = detail
        self.retry_after = retry_after


def provider_of(settings: Any) -> str:
    provider = getattr(settings, "ai_provider", "openrouter")
    return provider if provider in ("openrouter", "gemini", "openai-compatible") else "openrouter"


def uses_openrouter(settings: Any) -> bool:
    return provider_of(settings) == "openrouter"


def supports_images(settings: Any) -> bool:
    """Whether the active provider's model can be sent images."""
    provider = provider_of(settings)
    if provider == "gemini":
        return True
    if provider == "openai-compatible":
        return bool(getattr(settings, "openai_compatible_vision", False))
    return True


def provider_label(settings: Any) -> str:
    return {"gemini": "Gemini", "openai-compatible": "your AI endpoint"}.get(provider_of(settings), "OpenRouter")


def active_model(settings: Any) -> str:
    """The model id sent to the active non-OpenRouter provider."""
    provider = provider_of(settings)
    if provider == "gemini":
        return (getattr(settings, "gemini_model", "") or DEFAULT_GEMINI_MODEL).strip()
    if provider == "openai-compatible":
        model = (getattr(settings, "openai_compatible_model", "") or "").strip()
        if not model:
            raise AiClientError("AI model is not configured", reason="model_missing")
        return model
    return str(getattr(settings, "planner_model", ""))


def has_credentials(settings: Any) -> bool:
    """True when the active provider has what it needs to be called."""
    provider = provider_of(settings)
    if provider == "gemini":
        return bool(getattr(settings, "gemini_api_key", None))
    if provider == "openai-compatible":
        return bool(getattr(settings, "openai_compatible_url", None))
    return bool(getattr(settings, "openrouter_api_key", None))


def base_url_for(settings: Any) -> str:
    provider = provider_of(settings)
    if provider == "gemini":
        return GEMINI_OPENAI_BASE_URL
    if provider == "openai-compatible":
        raw = (getattr(settings, "openai_compatible_url", "") or "").strip().rstrip("/")
        parts = urlsplit(raw)
        if parts.scheme not in ("http", "https") or not parts.hostname or parts.username or parts.password:
            raise AiClientError("AI endpoint URL is not configured", reason="url_missing")
        return raw
    return str(settings.openrouter_base_url)


def request_timeout(settings: Any, default: float, limit: Optional[float] = None) -> float:
    """Seconds to wait for a reply. A custom endpoint uses its own setting, since a
    local model can be far slower than a hosted one; ``limit`` caps it for calls
    made many times per job, such as one per video frame."""
    if provider_of(settings) != "openai-compatible":
        return default
    value = getattr(settings, "openai_compatible_timeout_seconds", None)
    seconds = float(value) if isinstance(value, (int, float)) and value > 0 else DEFAULT_OPENAI_COMPATIBLE_TIMEOUT_SECONDS
    return min(seconds, limit) if limit else seconds


def get_ai_client(settings: Any, timeout: float = 600.0, limit: Optional[float] = None) -> httpx.AsyncClient:
    """Create the HTTP client for the active provider."""
    provider = provider_of(settings)
    timeout = request_timeout(settings, timeout, limit)
    if provider == "openrouter":
        return httpx.AsyncClient(
            base_url=settings.openrouter_base_url,
            timeout=httpx.Timeout(timeout, connect=30.0),
            headers={
                "Authorization": f"Bearer {settings.openrouter_api_key}",
                "HTTP-Referer": "https://github.com/bridge-mind/bridgeclip",
                "X-Title": "BridgeClip AI Clipping Agent",
            },
        )
    if provider == "gemini":
        key = getattr(settings, "gemini_api_key", "") or ""
    else:
        key = getattr(settings, "openai_compatible_api_key", "") or ""
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    return httpx.AsyncClient(
        base_url=base_url_for(settings),
        timeout=httpx.Timeout(timeout, connect=30.0),
        headers=headers,
    )


def prepare_payload(settings: Any, payload: dict[str, Any], is_vision: bool = False) -> dict[str, Any]:
    """Adapt an OpenRouter-shaped payload for the active provider.

    Drops OpenRouter-only routing fields and replaces the OpenRouter model id
    with the model the user configured. The structured-output format is kept;
    ``ai_chat_completion`` handles servers that reject it.
    """
    if uses_openrouter(settings):
        return payload
    request = dict(payload)
    for key in ("plugins", "provider", "models", "reasoning"):
        request.pop(key, None)
    request["model"] = active_model(settings)
    if _NO_TEMPERATURE_MODEL.search(request["model"]):
        request.pop("temperature", None)
    elif "temperature" not in request:
        request["temperature"] = 0.2
    return request


def _schema_of(payload: dict[str, Any]) -> Optional[dict[str, Any]]:
    response_format = payload.get("response_format")
    if isinstance(response_format, dict) and response_format.get("type") == "json_schema":
        schema = (response_format.get("json_schema") or {}).get("schema")
        if isinstance(schema, dict):
            return schema
    return None


def _with_schema_in_prompt(messages: list[dict], schema: dict[str, Any]) -> list[dict]:
    """Copy of ``messages`` whose last user turn also states the JSON schema."""
    instruction = (
        "Reply with one JSON value and nothing else: no prose and no code fences. "
        "It must validate against this JSON Schema:\n" + json.dumps(schema, separators=(",", ":"))
    )
    result = copy.deepcopy(messages)
    for message in reversed(result):
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, list):
            content.append({"type": "text", "text": instruction})
        else:
            message["content"] = f"{content or ''}\n\n{instruction}"
        return result
    result.append({"role": "user", "content": instruction})
    return result


def _variant(payload: dict[str, Any], level: int) -> dict[str, Any]:
    """Payload at structured-output ``level``: 0 schema, 1 JSON mode, 2 plain.

    The schema is in the prompt at every level: a server can accept
    ``response_format`` and ignore it, and the model then has nothing else to go on.
    """
    schema = _schema_of(payload)
    if schema is None:
        return payload
    variant = dict(payload)
    variant["messages"] = _with_schema_in_prompt(payload.get("messages") or [], schema)
    if level == 0:
        return variant
    if level == 1:
        variant["response_format"] = {"type": "json_object"}
    else:
        variant.pop("response_format", None)
    return variant


_LEADING_THINKING = re.compile(r"^\s*<(think|thinking|reasoning)>[\s\S]*?</\1>", re.IGNORECASE)
_OPENING_THINKING = re.compile(r"^\s*<(think|thinking|reasoning)>", re.IGNORECASE)
_CLOSING_THINKING = re.compile(r"</(think|thinking|reasoning)>", re.IGNORECASE)


def strip_thinking(content: str) -> str:
    """Text without the reasoning that models such as DeepSeek-R1 and Qwen3 write
    before the answer, in <think>...</think>. Its braces would otherwise be read
    as the start of the JSON.

    Only reasoning before the answer is removed, so a clip title that mentions
    the tags is left alone. Some chat templates open the block themselves, and
    the reply then has only the closing tag.
    """
    text = content
    while True:
        block = _LEADING_THINKING.match(text)
        if not block:
            break
        text = text[block.end():]
    if _OPENING_THINKING.match(text):
        # A reply cut off inside its reasoning has no answer at all.
        return ""
    closing = _CLOSING_THINKING.search(text)
    if closing and not text.lstrip().startswith(("{", "[", "`")):
        text = text[closing.end():]
    return text


def tolerant_json_parse(content: str) -> Any:
    """Parse JSON that may follow reasoning, or be wrapped in code fences, prose or trailing commas."""
    if not content:
        raise ValueError("Empty content")
    text = strip_thinking(content).strip()
    if not text:
        raise ValueError("Empty content")

    fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()

    try:
        return json.loads(text)
    except (json.JSONDecodeError, UnicodeError, RecursionError):
        pass

    end = max(text.rfind("]"), text.rfind("}"))
    starts = [i for i in (text.find("["), text.find("{")) if 0 <= i < end]
    if starts:
        candidate = text[min(starts):end + 1]
        for attempt in (candidate, re.sub(r",\s*([\]}])", r"\1", candidate)):
            try:
                return json.loads(attempt)
            except (json.JSONDecodeError, UnicodeError, RecursionError):
                continue

    raise ValueError("Could not extract JSON from content")


_LIMIT_NUMBER = re.compile(
    r"(?:at most|maximum(?: allowed| value)?(?: is| of)?|limit(?: is| of)?|less than or equal to|up to|supports? (?:at most|up to))"
    r"\D{0,20}(\d[\d,]{2,8})", re.IGNORECASE)


def _apply_token_settings(payload: dict[str, Any], learned: Optional[dict[str, Any]]) -> dict[str, Any]:
    """Payload with the output-token parameter and limit this server needs."""
    if not learned or ("max_tokens" not in payload and "max_completion_tokens" not in payload):
        return payload
    adjusted = dict(payload)
    value = adjusted.pop("max_tokens", None) or adjusted.pop("max_completion_tokens", None)
    if learned.get("limit"):
        value = min(int(value), int(learned["limit"]))
    adjusted[learned.get("param", "max_tokens")] = value
    return adjusted


def _learn_token_settings(error: "AiClientError", payload: dict[str, Any],
                          learned: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """What a 400 about output tokens says to change, or None if it says nothing.

    Newer OpenAI models reject ``max_tokens`` in favour of
    ``max_completion_tokens``, and most hosted models cap output well below
    the planner's request. Both come back as a 400 naming the parameter.
    """
    detail = error.detail.lower()
    if "max_tokens" not in detail and "max_completion_tokens" not in detail:
        return None
    current = dict(learned or {"param": "max_tokens", "limit": None})
    if "max_completion_tokens" in detail and "max_tokens" in detail and current["param"] == "max_tokens"             and ("unsupported" in detail or "not supported" in detail or "instead" in detail):
        current["param"] = "max_completion_tokens"
        return current
    found = _LIMIT_NUMBER.search(error.detail)
    requested = payload.get("max_tokens") or payload.get("max_completion_tokens")
    if found and requested:
        limit = int(found.group(1).replace(",", ""))
        if 0 < limit < int(requested):
            current["limit"] = limit
            return current
    return None


async def ai_chat_completion(
    client: httpx.AsyncClient,
    payload: dict[str, Any],
    settings: Any = None,
) -> tuple[dict, dict]:
    """POST a chat completion, stepping down structured-output support if needed."""
    from clip_engine.services.run_diagnostics import model_request

    model = str(payload.get("model", ""))
    key = (str(client.base_url), model)
    start = _working_level.get(key, 0)
    has_schema = _schema_of(payload) is not None
    last_level = 2 if has_schema else 0

    with model_request(model) as call:
        level = start if has_schema else 0
        adjustments = 0
        while True:
            sendable = _apply_token_settings(_variant(payload, level), _token_settings.get(key))
            if key in _no_temperature:
                sendable = {k: v for k, v in sendable.items() if k != "temperature"}
            try:
                body, usage = await _send_with_rate_limit_retry(client, sendable)
                break
            except AiClientError as error:
                if error.reason == "bad_request":
                    # Some models accept only their default temperature. That is
                    # fixed by leaving it out, not by weakening structured output.
                    detail = error.detail.lower()
                    if "temperature" in sendable and "temperature" in detail and any(word in detail for word in _TEMPERATURE_REFUSED):
                        # Without temperature the retry cannot hit this again, even
                        # when another request added the key first.
                        _no_temperature.add(key)
                        logger.info("AI endpoint rejected temperature; retrying without it")
                        continue
                    # A 400 about the output-token parameter is fixed by changing
                    # that parameter, not by weakening structured output.
                    learned = _learn_token_settings(error, payload, _token_settings.get(key))
                    if learned is not None and adjustments < MAX_TOKEN_ADJUSTMENTS:
                        adjustments += 1
                        _token_settings[key] = learned
                        logger.info("AI endpoint needs output tokens as %s", learned)
                        continue
                    if level < last_level:
                        level += 1
                        logger.info("AI endpoint rejected structured output (%s); retrying at level %d", error.detail[:200], level)
                        continue
                raise
        if has_schema and level != start:
            _working_level[key] = level
        call.update(success=True, input_tokens=usage.get("prompt_tokens"),
                    output_tokens=usage.get("completion_tokens"), cost_usd=usage.get("cost"))
        return body, usage


async def own_model_chat(settings: Any, payload: dict[str, Any], *, timeout: float,
                         limit: Optional[float] = None, is_vision: bool = False) -> tuple[dict, dict]:
    """One chat completion with the person's Gemini or OpenAI-compatible model.

    For features that were written against OpenRouter (the editorial judge, clip
    repair, visual context): the OpenRouter-shaped payload is adapted by
    prepare_payload, so the configured model is used and OpenRouter-only fields
    are dropped. OpenRouter itself keeps using its own call sites.
    """
    async with get_ai_client(settings, timeout=timeout, limit=limit) as client:
        return await ai_chat_completion(client, prepare_payload(settings, payload, is_vision=is_vision), settings)


async def _send_with_rate_limit_retry(client: httpx.AsyncClient, payload: dict[str, Any]) -> tuple[dict, dict]:
    for attempt in range(MAX_RATE_LIMIT_RETRIES + 1):
        try:
            return await _send_chat_completion(client, payload)
        except AiClientError as error:
            wait = error.retry_after
            if error.reason != "rate_limit" or attempt == MAX_RATE_LIMIT_RETRIES or wait is None or wait > MAX_RATE_LIMIT_WAIT_SECONDS:
                raise
            logger.info("AI provider rate limited; waiting %.0fs", wait)
            await asyncio.sleep(max(wait, 1.0))
    raise AiClientError("AI provider rate limit reached", retryable=True, reason="rate_limit")


def _provider_detail(content: bytes) -> str:
    try:
        body = json.loads(content)
    except (ValueError, UnicodeError, RecursionError):
        return ""
    error = body.get("error") if isinstance(body, dict) else None
    if isinstance(body, list) and body and isinstance(body[0], dict):
        error = body[0].get("error")
    if isinstance(error, dict):
        return str(error.get("message") or "")[:300]
    if not error and isinstance(body, dict):
        # Some servers (SGLang, older vLLM) put the text at the top level.
        error = body.get("message") or body.get("detail")
    return str(error or "")[:300]


def _retry_after_seconds(response: httpx.Response) -> Optional[float]:
    try:
        value = float(response.headers.get("retry-after", ""))
    except ValueError:
        return None
    return value if value >= 0 else None


_DELAY = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*s\s*$")
_RETRY_IN = re.compile(r"retry in (\d+(?:\.\d+)?)\s*s", re.IGNORECASE)


def _retry_delay_from_body(content: bytes) -> Optional[float]:
    """The wait a 429 body asks for. Gemini sends it as RetryInfo's ``retryDelay``
    ("40s") in the error details, or only in the message ("Please retry in 40.5s.").

    A daily quota (a QuotaFailure whose quota id or metric is per day) is not
    lifted by waiting a minute, even when the body suggests one, so it gets no wait.
    """
    try:
        body = json.loads(content)
    except (ValueError, UnicodeError, RecursionError):
        return None
    delay: Optional[float] = None
    stack, seen = [body], 0
    while stack and seen < 200:
        item = stack.pop()
        seen += 1
        if isinstance(item, dict):
            for name in ("quotaId", "quotaMetric"):
                if "perday" in str(item.get(name, "")).lower():
                    return None
            value = item.get("retryDelay")
            if delay is None and isinstance(value, str) and _DELAY.match(value):
                delay = float(_DELAY.match(value).group(1))
            stack.extend(item.values())
        elif isinstance(item, list):
            stack.extend(item)
        elif isinstance(item, str) and delay is None:
            found = _RETRY_IN.search(item)
            if found:
                delay = float(found.group(1))
    return delay


async def _models_endpoint_answers(client: httpx.AsyncClient) -> bool:
    """Whether GET {base}/models works, which tells a wrong model from a wrong URL."""
    try:
        async with client.stream("GET", "/models", headers={"Accept-Encoding": "identity"},
                                 follow_redirects=False, timeout=10.0) as response:
            return response.status_code == 200
    except (httpx.TimeoutException, httpx.TransportError):
        return False


async def _send_chat_completion(client: httpx.AsyncClient, payload: dict[str, Any]) -> tuple[dict, dict]:
    model = payload.get("model", "")
    try:
        async with client.stream(
            "POST", "/chat/completions", json=payload,
            headers={"Accept-Encoding": "identity"}, follow_redirects=False,
        ) as response:
            if response.headers.get("content-encoding", "identity").lower() != "identity":
                raise AiClientError("AI provider returned an unsupported response encoding", reason="invalid_response")
            content = bytearray()
            async for chunk in response.aiter_raw():
                if len(chunk) > MAX_CHAT_RESPONSE_BYTES - len(content):
                    raise AiClientError("AI provider response exceeds the size limit", reason="invalid_response")
                content.extend(chunk)
            status = response.status_code
            retry_after = _retry_after_seconds(response)
    except httpx.ConnectTimeout:
        raise AiClientError("AI provider could not be reached", retryable=True, reason="network") from None
    except httpx.TimeoutException:
        # Asking again would wait just as long: a slow model needs a longer wait
        # setting or a smaller model, not a retry.
        raise AiClientError("AI provider took too long to answer", reason="timeout") from None
    except httpx.TransportError:
        raise AiClientError("AI provider could not be reached", retryable=True, reason="network") from None

    if status != 200:
        detail = _provider_detail(bytes(content))
        if status in (401, 403):
            raise AiClientError("AI provider rejected the API key", reason="auth", status=status, detail=detail)
        if status == 402:
            raise AiClientError("AI provider account is out of credit", reason="quota", status=status, detail=detail)
        if status == 404:
            # A missing model is usually named in the message. Otherwise, a model
            # list at the same base URL means the URL is right and the model is not;
            # no list means nothing answers here, usually a base URL without /v1.
            if "model" in detail.lower() or await _models_endpoint_answers(client):
                raise AiClientError("AI model was not found", reason="model_not_found", status=status, detail=detail)
            raise AiClientError("AI endpoint was not found", reason="endpoint_not_found", status=status, detail=detail)
        if status == 429:
            if retry_after is None:
                retry_after = _retry_delay_from_body(bytes(content))
            raise AiClientError("AI provider rate limit or quota reached", retryable=True, reason="rate_limit",
                                status=status, detail=detail, retry_after=retry_after)
        if status in (400, 422):
            raise AiClientError("AI provider rejected the request", reason="bad_request", status=status, detail=detail)
        raise AiClientError("AI provider is temporarily unavailable", retryable=status in RETRYABLE_STATUS_CODES,
                            reason="unavailable", status=status, detail=detail)

    try:
        body = json.loads(content)
    except (ValueError, UnicodeError, RecursionError):
        raise AiClientError("AI provider returned an invalid response", retryable=True, reason="invalid_response") from None
    if isinstance(body, list) and len(body) == 1:
        body = body[0]
    if not isinstance(body, dict):
        raise AiClientError("AI provider returned an invalid response", retryable=True, reason="invalid_response")
    if body.get("error"):
        raise AiClientError("AI provider returned an error", retryable=True, reason="unavailable",
                            detail=str(body["error"])[:300])
    if not body.get("choices"):
        raise AiClientError("AI provider returned an empty response", retryable=True, reason="invalid_response")

    usage = body.get("usage") or {}
    cost = usage.get("cost")
    usage_data = {
        "prompt_tokens": usage.get("prompt_tokens") or 0,
        "completion_tokens": usage.get("completion_tokens") or 0,
        "total_tokens": usage.get("total_tokens") or 0,
        "cost": float(cost) if isinstance(cost, (int, float)) else None,
    }
    logger.info(
        "AI usage (%s): %s prompt, %s completion tokens",
        body.get("model", model), usage_data["prompt_tokens"], usage_data["completion_tokens"],
    )
    return body, usage_data
