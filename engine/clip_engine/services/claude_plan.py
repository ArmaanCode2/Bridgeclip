"""
Claude Code as an AI provider: requests answered by the person's own Claude plan.

BridgeClip runs the ``claude`` program the person signed in to, one hidden run per
request. ``ai_client.get_ai_client`` gives callers an httpx client whose transport
starts that run instead of opening a connection, so the planner, judges and reviews
send their requests exactly as they do to any other provider.

Each run has no tools, saves no session, reads none of the person's Claude Code
settings, starts in an empty folder and sees only the environment the app chose
(src/main/claude-plan/runner.ts builds both). Nothing from this process's environment
reaches it, because one API key there would quietly move the run off the plan and onto
per-token billing. A run that reports it is not on the plan stops every later request all the same.

Some steps can be skipped when their model request fails, which suits a passing
failure but would hide a plan that has run out. So a failure that every later request
would hit too is remembered: later requests fail with it without starting a run, and
the pipeline calls ``raise_if_stopped`` before it moves on. One bridge process runs one
job, so this lasts exactly as long as the job.

Every run gets an id the app chose, written to the job's folder before it starts. A run
saves no session, so normally that list points at nothing. It is there in case a Claude
Code update changes that: src/main/claude-plan/sessions.ts removes those ids when the job
is deleted.
"""

import asyncio
import ctypes
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import uuid
import weakref
from typing import Any, Optional

import httpx

from clip_engine.error_policy import AI_PROVIDER_ERRORS
from clip_engine.services.ai_client import AiClientError, _schema_of

logger = logging.getLogger(__name__)

# Most Claude Code runs at once for one job. Parts of a long video and their reviews
# queue here, whichever step sent them.
MAX_RUNS = 2
# Claude Code's own system prompt describes a coding agent, so a request without one
# still replaces it.
DEFAULT_SYSTEM_PROMPT = "Do what the user asks. Reply with the answer only."
# How long to wait for a killed run's output, in case something it started holds the pipes.
_KILLED_RUN_SECONDS = 5

_DATA_URL = re.compile(r"^data:(image/[a-z0-9.+-]+);base64,(.+)$", re.IGNORECASE | re.DOTALL)

# Claude Code's error code or its words -> the app's failure. Its own text can echo the
# request, so it stays in ``detail``. The first match wins, and none of these is retried:
# asking again cannot sign anyone in or lift a limit. src/main/claude-plan/chat.ts has the
# same rows.
_FAILURES = (
    ({"authentication_failed"}, re.compile(r"not logged in|/login|invalid api key|authentication_error|oauth token|not authenticated", re.IGNORECASE), "claude_auth"),
    ({"rate_limit", "billing_error"}, re.compile(r"rate limit|usage limit|session limit|weekly limit|hit your \S* ?limit", re.IGNORECASE), "plan_limit"),
    ({"model_not_found"}, re.compile(r"selected model", re.IGNORECASE), "claude_model"),
    (set(), re.compile(r"prompt is too long", re.IGNORECASE), "too_long"),
    (set(), re.compile(r"unknown option|unknown argument", re.IGNORECASE), "outdated"),
)
# Every later request of the job would fail the same way, or should not be sent at all.
_STOPS_THE_JOB = frozenset({"claude_auth", "claude_billing", "not_installed", "outdated", "plan_limit"})

_stopped_by: Optional[str] = None
_slots: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Semaphore]" = weakref.WeakKeyDictionary()


def _error(reason: str, detail: str = "") -> AiClientError:
    """One of the fixed failures from error_policy, the only text the person sees."""
    return AiClientError(AI_PROVIDER_ERRORS[reason], reason=reason, detail=detail[:300])


def raise_if_stopped() -> None:
    """Fail the job if the plan stopped answering, whichever step first met it."""
    if _stopped_by is not None:
        raise _error(_stopped_by)


def _stop_job(reason: str) -> None:
    global _stopped_by
    if reason in _STOPS_THE_JOB and _stopped_by is None:
        _stopped_by = reason


def _slot() -> asyncio.Semaphore:
    loop = asyncio.get_running_loop()
    if loop not in _slots:
        _slots[loop] = asyncio.Semaphore(MAX_RUNS)
    return _slots[loop]


def _runner(settings: Any) -> dict[str, Any]:
    """How to start Claude Code, as worked out by the main process.

    Never guessed here: a missing piece fails the request rather than falling back to
    this process's environment.
    """
    try:
        runner = json.loads(getattr(settings, "claude_plan_runner", None) or "")
    except ValueError:
        runner = None
    if isinstance(runner, dict):
        command, env, cwd = runner.get("command"), runner.get("env"), runner.get("cwd")
        if (isinstance(command, list) and command and all(isinstance(part, str) and part for part in command)
                and isinstance(env, dict) and all(isinstance(value, str) for value in env.values())
                and isinstance(cwd, str) and cwd and isinstance(runner.get("sessions"), (str, type(None)))):
            return runner
    raise _error("not_installed")


def _image(url: str) -> dict[str, Any]:
    found = _DATA_URL.match(url)
    if not found:
        raise _error("bad_request", "Image is not embedded")
    return {"type": "image", "source": {"type": "base64", "media_type": found[1], "data": found[2]}}


def _parts(content: Any) -> list[dict[str, Any]]:
    """An OpenAI-style message content as Claude content blocks."""
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    parts: list[dict[str, Any]] = []
    for part in content or []:
        if part.get("type") == "text":
            parts.append({"type": "text", "text": part.get("text", "")})
        elif part.get("type") == "image_url":
            parts.append(_image((part.get("image_url") or {}).get("url", "")))
        else:
            raise _error("bad_request", f"Unknown part {part.get('type')}")
    return parts


def _split_messages(messages: list[dict]) -> tuple[str, list[dict[str, Any]]]:
    """(system prompt, content of the one user message a run takes)."""
    system: list[str] = []
    content: list[dict[str, Any]] = []
    for message in messages:
        parts = _parts(message.get("content"))
        if message.get("role") == "system":
            system.extend(part["text"] for part in parts if part["type"] == "text")
        else:
            content.extend(parts)
    return "\n\n".join(system) or DEFAULT_SYSTEM_PROMPT, content


def _list_session(runner: dict[str, Any], session_id: str) -> None:
    """Adds a run to the job's list before it starts, so a crashed run is listed too."""
    path = runner.get("sessions")
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as file:
            file.write(session_id + "\n")
    except OSError:
        logger.warning("Could not list a Claude Code run in the job folder")


def _arguments(settings: Any, system_file: str, schema: Optional[dict[str, Any]], session_id: str) -> list[str]:
    arguments = [
        "-p",
        "--model", settings.claude_plan_model,
        "--effort", settings.claude_plan_effort,
        "--tools", "",
        "--strict-mcp-config",
        # The person's settings file could name a key or remap a model. Signing in still works.
        "--setting-sources", "",
        "--settings", json.dumps({"disableAllHooks": True}),
        "--disable-slash-commands",
        "--no-session-persistence",
        "--session-id", session_id,
        "--permission-mode", "dontAsk",
        # A long prompt could outgrow a Windows command line, so it goes in a file.
        "--system-prompt-file", system_file,
        "--input-format", "stream-json",
        "--output-format", "stream-json",
        "--verbose",
    ]
    if schema is not None:
        arguments += ["--json-schema", json.dumps(schema, separators=(",", ":"))]
    return arguments


class _BasicLimit(ctypes.Structure):
    _fields_ = [("user_time", ctypes.c_int64 * 2), ("flags", ctypes.c_ulong), ("working_set", ctypes.c_size_t * 2),
                ("process_limit", ctypes.c_ulong), ("affinity", ctypes.c_size_t), ("priority", ctypes.c_ulong), ("scheduling", ctypes.c_ulong)]


class _ExtendedLimit(ctypes.Structure):
    _fields_ = [("basic", _BasicLimit), ("io", ctypes.c_uint64 * 6), ("memory", ctypes.c_size_t * 4)]


_JOB_EXTENDED_LIMIT = 9
_JOB_KILL_ON_CLOSE = 0x2000


def _tie_to_engine(process: subprocess.Popen) -> Optional[int]:
    """A Windows job object ending the run with this process, even if the engine crashes.

    Without one a run outlives a crash and keeps spending the plan. It is a safety net,
    so a run still goes ahead if Windows refuses.
    """
    if sys.platform != "win32":
        return None
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateJobObjectW.restype = ctypes.c_void_p
    kernel32.CreateJobObjectW.argtypes = (ctypes.c_void_p, ctypes.c_wchar_p)
    kernel32.SetInformationJobObject.argtypes = (ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_ulong)
    kernel32.AssignProcessToJobObject.argtypes = (ctypes.c_void_p, ctypes.c_void_p)
    kernel32.CloseHandle.argtypes = (ctypes.c_void_p,)
    job = kernel32.CreateJobObjectW(None, None)
    if not job:
        logger.warning("Could not tie a Claude Code run to the engine")
        return None
    limit = _ExtendedLimit()
    limit.basic.flags = _JOB_KILL_ON_CLOSE
    if (kernel32.SetInformationJobObject(job, _JOB_EXTENDED_LIMIT, ctypes.byref(limit), ctypes.sizeof(limit))
            and kernel32.AssignProcessToJobObject(job, int(process._handle))):
        return job
    logger.warning("Could not tie a Claude Code run to the engine")
    kernel32.CloseHandle(job)
    return None


def _release(job: Optional[int]) -> None:
    """Closing the last handle ends whatever is still running in the job."""
    if job is not None:
        kernel32 = ctypes.WinDLL("kernel32")
        kernel32.CloseHandle.argtypes = (ctypes.c_void_p,)
        kernel32.CloseHandle(job)


def _finish(process: subprocess.Popen, stdin: bytes, timeout: float) -> tuple[bytes, bytes]:
    try:
        return process.communicate(stdin, timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        try:
            process.communicate(timeout=_KILLED_RUN_SECONDS)
        except subprocess.TimeoutExpired:
            logger.warning("A stopped Claude Code run left its output open")
        raise _error("claude_timeout") from None


async def _run(command: list[str], stdin: bytes, runner: dict[str, Any], timeout: float) -> tuple[str, str]:
    """(stdout, stderr) of one run. A cancelled request stops its run."""
    flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    try:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   cwd=runner["cwd"], env=runner["env"], creationflags=flags)
    except OSError as error:
        raise _error("not_installed", type(error).__name__) from None
    job = _tie_to_engine(process)
    try:
        stdout, stderr = await asyncio.to_thread(_finish, process, stdin, timeout)
    except asyncio.CancelledError:
        process.kill()
        raise
    finally:
        _release(job)
    return stdout.decode("utf-8", "replace"), stderr.decode("utf-8", "replace")


def _events(stdout: str) -> list[dict[str, Any]]:
    events = []
    # Only "\n" ends a line: str.splitlines would also split an answer on the separators JSON leaves unescaped.
    for line in stdout.split("\n"):
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if isinstance(event, dict):
            events.append(event)
    return events


def _limits(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """What Claude Code reported about the plan's limits during a run."""
    return [event.get("rate_limit_info") or {} for event in events if event.get("type") == "rate_limit_event"]


def _failure(events: list[dict[str, Any]], result: Optional[dict[str, Any]], stderr: str) -> AiClientError:
    if result is not None and str(result.get("subtype", "")).startswith("error_max_structured_output"):
        return AiClientError(AI_PROVIDER_ERRORS["invalid_response"], retryable=True, reason="invalid_response")
    code = next((event["error"] for event in events if event.get("type") == "assistant" and event.get("error")), None)
    text = " ".join(part for part in (str((result or {}).get("result") or ""), stderr) if part)
    if any(limit.get("status") == "rejected" for limit in _limits(events)):
        return _error("plan_limit", text)
    for codes, words, reason in _FAILURES:
        if code in codes or words.search(text):
            return _error(reason, text)
    return AiClientError(AI_PROVIDER_ERRORS["unavailable"], retryable=True, reason="unavailable", detail=text[:300])


def _answer(result: dict[str, Any], model: str) -> dict[str, Any]:
    """A successful run as an OpenAI-style reply. It carries no price: the plan pays,
    and Claude Code's ``total_cost_usd`` is a local estimate."""
    structured = result.get("structured_output")
    content = json.dumps(structured) if structured is not None else result.get("result")
    if not content:
        raise AiClientError(AI_PROVIDER_ERRORS["invalid_response"], retryable=True, reason="invalid_response")
    usage = result.get("usage") or {}
    prompt = sum(int(usage.get(key) or 0) for key in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"))
    completion = int(usage.get("output_tokens") or 0)
    return {
        "model": next(iter(result.get("modelUsage") or {}), model),
        "choices": [{"index": 0, "finish_reason": "length" if result.get("stop_reason") == "max_tokens" else "stop",
                     "message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion},
    }


async def _ask(settings: Any, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    runner = _runner(settings)
    system, content = _split_messages(payload.get("messages") or [])
    stdin = json.dumps({"type": "user", "message": {"role": "user", "content": content}}).encode() + b"\n"
    # In the job's own work folder when it has one, so a forced stop leaves nothing behind.
    descriptor, system_file = tempfile.mkstemp(prefix="claude-prompt-", suffix=".txt", dir=os.environ.get("BRIDGECLIP_WORK_ROOT") or None)
    try:
        with os.fdopen(descriptor, "wb") as file:
            file.write(system.encode("utf-8"))
        session_id = str(uuid.uuid4())
        _list_session(runner, session_id)
        command = [*runner["command"], *_arguments(settings, system_file, _schema_of(payload), session_id)]
        stdout, stderr = await _run(command, stdin, runner, timeout)
    finally:
        try:
            os.unlink(system_file)
        except OSError:
            logger.warning("Could not remove a Claude Code prompt file")
    events = _events(stdout)
    if any(event.get("apiKeySource") not in (None, "none") for event in events if event.get("type") == "system"):
        raise _error("claude_billing")
    result = next((event for event in reversed(events) if event.get("type") == "result"), None)
    if result is None or result.get("is_error"):
        raise _failure(events, result, stderr)
    # Paid overflow past the plan's limit. This answer is kept; nothing more is sent.
    if any(limit.get("isUsingOverage") for limit in _limits(events)):
        _stop_job("plan_limit")
    return _answer(result, str(payload.get("model", "")))


class ClaudePlanTransport(httpx.AsyncBaseTransport):
    """Answers chat-completion requests with Claude Code instead of a connection."""

    def __init__(self, settings: Any, timeout: float):
        self.settings = settings
        self.timeout = timeout

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        payload = json.loads(await request.aread())
        async with _slot():
            raise_if_stopped()
            try:
                body = await _ask(self.settings, payload, self.timeout)
            except AiClientError as error:
                _stop_job(error.reason)
                raise
        return httpx.Response(200, headers={"content-type": "application/json"}, stream=httpx.ByteStream(json.dumps(body).encode()))
