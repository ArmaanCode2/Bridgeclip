"""Bounded OpenRouter Decisions adapter for Jev. No media, credentials or provider errors in records.

Contract: https://openrouter.ai/docs/guides/community/jev; pricing snapshot: 2026-09-24.
One instance belongs to one job. Cancellation propagates; failures are data.
"""
import asyncio
import copy
import email.utils
import hashlib
import json
import math
import re
import time
from datetime import datetime, timezone
from typing import Literal, TypedDict

import httpx

MODEL = 'typesafe/jev-1.13'
RULE_VERSION = 'editorial-v2-openrouter'
ENDPOINT = 'https://openrouter.ai/api/alpha/decisions'
INPUT_USD_PER_TOKEN = .042 / 1_000_000
# Request bodies are UTF-8 JSON without ASCII escaping, so an excerpt in any
# script costs its real size. The budget is in bytes, not characters.
MAX_REQUEST_BYTES = 64_000
MAX_RESPONSE_BYTES = 128_000
# Transient provider responses (408, 429 and any 5xx, including 529) get at
# most two more attempts. Retry-After is honored up to MAX_RETRY_AFTER seconds;
# a longer requested wait ends the request instead of stalling the job.
RETRY_DELAYS = (1.0, 3.0)
MAX_RETRY_AFTER = 20.0
# Consecutive failed requests after which the endpoint counts as unavailable
# for the rest of the job, instead of timing out candidate by candidate.
MAX_CONSECUTIVE_FAILURES = 4
_sleep = asyncio.sleep  # Indirection so tests can skip real backoff.


class Question(TypedDict):
    type: Literal['noul', 'choice', 'score']
    instructions: str
    criteria: dict | list


def noul(instructions: str, yes: str, no: str) -> Question:
    return {'type': 'noul', 'instructions': instructions, 'criteria': {'true': yes, 'false': no}}


def choice(instructions: str, criteria: dict[str, str]) -> Question:
    return {'type': 'choice', 'instructions': instructions, 'criteria': criteria}


def score(instructions: str, levels: list[str]) -> Question:
    return {'type': 'score', 'instructions': instructions, 'criteria': levels}


def _number(value, low=0, high=1):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError('Invalid judgment number')
    return value


def _unreported_choice(options, selected):
    """The API may omit probabilities. Assume the least informative distribution
    that still ranks the reported choice first, so thresholds stay conservative:
    a bare two-way "sufficient" is 51%, never certainty."""
    if len(options) == 1:
        return {selected: 1.0}
    top = 1 / len(options) + .01
    return {o: top if o == selected else (1 - top) / (len(options) - 1) for o in sorted(options)}


def _unreported_score(value, levels):
    """A distribution consistent with a reported score whose probabilities were omitted."""
    low = min(int(value), levels - 1)
    high = min(low + 1, levels - 1)
    fraction = value - low if high > low else 0.0
    return {str(i): (1 - fraction if i == low else fraction if i == high else 0.0) for i in range(levels)}


def validate_answers(raw, questions):
    if not isinstance(raw, dict) or set(raw) != set(questions):
        raise ValueError('Missing judgments')
    answers = {}
    for key, question in questions.items():
        answer = raw[key]
        kind = question['type']
        if not isinstance(answer, dict) or answer.get('type') != kind:
            raise ValueError('Incorrect judgment type')
        if kind == 'noul':
            answers[key] = {'type': kind, 'noul': _number(answer.get('noul'))}
            continue
        options = set(question['criteria']) if kind == 'choice' else {str(i) for i in range(len(question['criteria']))}
        # `probabilities` and `confidence` are optional in the Decisions API.
        probabilities = answer.get('probabilities')
        reported = probabilities is not None
        if reported:
            if not isinstance(probabilities, dict) or set(probabilities) != options:
                raise ValueError('Incorrect judgment options')
            probabilities = {k: _number(v) for k, v in probabilities.items()}
            if abs(sum(probabilities.values()) - 1) > .005:
                raise ValueError('Invalid probability distribution')
        confidence = answer.get('confidence')
        clean = {'type': kind, 'confidence': None if confidence is None else _number(confidence)}
        if kind == 'choice':
            selected = answer.get('choice')
            if selected not in options:
                raise ValueError('Invalid selected option')
            if not reported:
                probabilities = _unreported_choice(options, selected)
            if probabilities[selected] < max(probabilities.values()) - .005:
                raise ValueError('Invalid selected option')
            clean['choice'] = selected
        else:
            value = _number(answer.get('score'), 0, len(options) - 1)
            if not reported:
                probabilities = _unreported_score(value, len(options))
            if abs(value - sum(int(k) * v for k, v in probabilities.items())) > .02:
                raise ValueError('Inconsistent score')
            clean.update(score=value, legend={str(i): level for i, level in enumerate(question['criteria'])})
        clean['probabilities'] = probabilities
        if not reported:
            clean['probabilities_reported'] = False
        answers[key] = clean
    return answers


def _retry_after(value):
    """Seconds requested by a Retry-After header (delta-seconds or HTTP date), or None."""
    if not value:
        return None
    value = value.strip()
    try:
        seconds = float(value)
    except ValueError:
        try:
            when = email.utils.parsedate_to_datetime(value)
        except (TypeError, ValueError, IndexError):
            return None
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        seconds = (when - datetime.now(timezone.utc)).total_seconds()
    return max(0.0, seconds) if math.isfinite(seconds) else None


class _Transient(Exception):
    """A retryable provider response. Carries no provider text."""

    def __init__(self, retry_after=None):
        super().__init__('Transient provider failure')
        self.retry_after = retry_after


class _OutOfCredits(Exception):
    """OpenRouter answered 402: the account or key has insufficient credit."""


class JevService:
    def __init__(self, api_key='', *, transport=None, max_requests=64, token_budget=180_000, timeout=15):
        self._api_key = api_key or ''
        self._transport = transport
        self._semaphore = asyncio.Semaphore(2)
        self._cache = {}
        # Bounded by request count, token budget and a per-request timeout,
        # never by cumulative latency: a slow provider must not change which
        # clips are approved.
        self.max_requests, self.token_budget, self.timeout = max_requests, token_budget, timeout
        self.requests = self.reserved_tokens = self.input_tokens = self.output_tokens = 0
        self.estimated_cost_usd = 0.0
        self.request_seconds = 0.0  # Diagnostics only.
        self.consecutive_failures = 0
        self.out_of_credits = False

    @classmethod
    def from_settings(cls, settings, *, required=False):
        key = getattr(settings, 'openrouter_api_key', None) if required or getattr(settings, 'jev_enabled', False) else None
        return cls(key)

    @property
    def enabled(self):
        return bool(self._api_key)

    async def evaluate(self, state, questions: dict[str, Question]):
        # Source metadata and web research are untrusted and can carry prompt
        # injections; Jev judges transcript evidence only.
        if isinstance(state, dict) and 'source_context' in state:
            state = {k: v for k, v in state.items() if k != 'source_context'}
        payload = {'model': MODEL, 'state': state, 'questions': questions}
        encoded = json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False, ensure_ascii=False).encode('utf-8')
        cache_id = hashlib.sha256(RULE_VERSION.encode() + encoded).hexdigest()
        record = {'status': 'disabled', 'model': MODEL, 'requested_model': MODEL, 'rule_version': RULE_VERSION,
                  'cache_id': cache_id, 'cache_hit': False, 'latency_ms': 0, 'attempts': 0,
                  'input_tokens': None, 'output_tokens': None, 'cost_usd': None,
                  'estimated_cost_usd': None, 'answers': {}, 'questions': questions}
        if not self.enabled:
            return record
        # UTF-8 bytes are a conservative input-token reservation; reserve room
        # for the fixed, bounded answer schema too. No invented API token knobs.
        reserve = len(encoded) + 128 * len(questions)
        if not 1 <= len(questions) <= 16 or len(encoded) > MAX_REQUEST_BYTES:
            return {**record, 'status': 'evidence_limit'}
        async with self._semaphore:
            if cache_id in self._cache:
                return {**copy.deepcopy(self._cache[cache_id]), 'cache_hit': True, 'latency_ms': 0, 'attempts': 0,
                        'input_tokens': 0, 'output_tokens': 0, 'cost_usd': 0.0, 'estimated_cost_usd': 0.0}
            if self.out_of_credits:
                return {**record, 'status': 'out_of_credits'}
            if self.consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                return {**record, 'status': 'unavailable'}
            if self.requests >= self.max_requests or self.reserved_tokens + reserve > self.token_budget:
                return {**record, 'status': 'budget_exhausted'}
            self.requests += 1
            self.reserved_tokens += reserve
            record.update(status='unavailable', attempts=1)
            from .run_diagnostics import model_request
            with model_request(MODEL) as usage_call:
                return await self._evaluate_request(encoded, record, questions, reserve, usage_call)

    async def _post(self, encoded):
        """One HTTP attempt inside its own timeout. Returns the parsed body."""
        async with asyncio.timeout(self.timeout):
            async with httpx.AsyncClient(transport=self._transport, timeout=self.timeout, follow_redirects=False) as client:
                try:
                    async with client.stream('POST', ENDPOINT, headers={'Authorization': f'Bearer {self._api_key}',
                                              'Content-Type': 'application/json', 'Accept-Encoding': 'identity'}, content=encoded) as response:
                        status = response.status_code
                        if status == 402:
                            raise _OutOfCredits()
                        if status in (408, 429) or 500 <= status <= 599:
                            raise _Transient(_retry_after(response.headers.get('retry-after')))
                        response.raise_for_status()
                        if response.headers.get('content-encoding', 'identity').lower() != 'identity':
                            raise ValueError('Unsupported response encoding')
                        body = bytearray()
                        async for chunk in response.aiter_bytes():
                            body.extend(chunk)
                            if len(body) > MAX_RESPONSE_BYTES:
                                raise ValueError('Oversized response')
                except httpx.ConnectError:
                    # Nothing was sent, so a new connection is safe to try.
                    raise _Transient() from None
        return json.loads(body)

    async def _post_with_retries(self, encoded, record):
        for backoff in (*RETRY_DELAYS, None):
            try:
                return await self._post(encoded)
            except _Transient as error:
                wait = None if backoff is None else max(backoff, error.retry_after or 0)
                if wait is None or wait > MAX_RETRY_AFTER or self.requests >= self.max_requests:
                    raise
                await _sleep(wait)  # Cancellation propagates.
                self.requests += 1
                record['attempts'] += 1

    async def _evaluate_request(self, encoded, record, questions, reserve, usage_call):
        now = time.monotonic()
        try:
            data = await self._post_with_retries(encoded, record)
            actual_model = data.get('model')
            if not isinstance(actual_model, str) or not re.fullmatch(re.escape(MODEL) + r'(?:-\d{8})?', actual_model):
                raise ValueError('Unexpected model version')
            record['model'] = actual_model
            usage = data.get('usage', {})
            input_tokens = _number(usage.get('input_tokens'), 0, 1_000_000)
            output_tokens = _number(usage.get('output_tokens'), 0, 1_000_000)
            if not isinstance(input_tokens, int) or not isinstance(output_tokens, int):
                raise ValueError('Invalid token usage')
            self.input_tokens += input_tokens
            self.output_tokens += output_tokens
            # Replace the byte-based reservation with reported usage, so scripts
            # with multi-byte characters are not charged several times over.
            self.reserved_tokens += input_tokens + output_tokens - reserve
            estimate = input_tokens * INPUT_USD_PER_TOKEN
            billed = usage.get('cost')
            if billed is not None:
                billed = _number(billed, 0, 1000)
            self.estimated_cost_usd += billed if billed is not None else estimate
            usage_call.update(input_tokens=input_tokens, output_tokens=output_tokens, cost_usd=billed)
            record.update(input_tokens=input_tokens, output_tokens=output_tokens, cost_usd=billed, estimated_cost_usd=estimate)
            record.update(answers=validate_answers(data.get('answers'), questions), status='success')
        except asyncio.CancelledError:
            raise
        except _OutOfCredits:
            self.out_of_credits = True
            record['status'] = 'out_of_credits'
        except (_Transient, httpx.HTTPError, TimeoutError, ValueError, TypeError, AttributeError):
            pass  # Never retain response text, request headers or exception strings.
        elapsed = time.monotonic() - now
        self.request_seconds += elapsed
        record['latency_ms'] = round(elapsed * 1000)
        usage_call['success'] = record['status'] == 'success'
        if record['status'] == 'success':
            self.consecutive_failures = 0
            self._cache[record['cache_id']] = copy.deepcopy(record)
        else:
            self.consecutive_failures += 1
        return record
