"""Offline tests for the Claude plan provider. A fake `claude` program replays what the real one prints."""

import asyncio
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from clip_engine.config import Settings
from clip_engine.services import ai_client, claude_plan
from clip_engine.services.ai_client import (
    AiClientError,
    active_model,
    get_ai_client,
    has_credentials,
    own_model_chat,
    provider_of,
    request_timeout,
    supports_images,
)
from clip_engine.services.openrouter import json_schema_format

FAKE_CLAUDE = r'''
import json, os, sys, time

argv = sys.argv[1:]
spec = json.load(open(os.environ["FAKE_SPEC"], encoding="utf-8"))
record = {"argv": argv, "cwd": os.getcwd(), "env": dict(os.environ), "stdin": sys.stdin.read(), "started": time.time()}
if "--system-prompt-file" in argv:
    path = argv[argv.index("--system-prompt-file") + 1]
    record["system_prompt_file"] = path
    record["system_prompt"] = open(path, encoding="utf-8").read()
time.sleep(spec.get("sleep", 0))
record["finished"] = time.time()
folder = os.environ["FAKE_RECORDS"]
with open(os.path.join(folder, f"{os.getpid()}-{time.time_ns()}.json"), "w", encoding="utf-8") as file:
    json.dump(record, file)
sys.stdout.reconfigure(encoding="utf-8")
for line in spec.get("stdout", []):
    print(json.dumps(line) if isinstance(line, dict) else line, flush=True)
sys.stderr.write(spec.get("stderr", ""))
sys.exit(spec.get("exit", 0))
'''

CRASHING_ENGINE = r'''
import asyncio, json, os
from types import SimpleNamespace
from clip_engine.services.ai_client import own_model_chat

settings = SimpleNamespace(ai_provider="claude-plan", claude_plan_model="sonnet", claude_plan_effort="high", claude_plan_vision=False,
                           claude_plan_runner=os.environ["CLAUDE_RUNNER"])

async def main():
    request = asyncio.create_task(own_model_chat(settings, {"model": "x", "messages": [{"role": "user", "content": "Hello"}]}, timeout=30))
    await asyncio.sleep(1.5)
    os._exit(7)

asyncio.run(main())
'''
UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
SCHEMA = {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"], "additionalProperties": False}
# Fields a real run reports; the rest of its events are shaped the same way.
INIT = {"type": "system", "subtype": "init", "model": "claude-sonnet-5-5", "apiKeySource": "none", "tools": ["StructuredOutput"]}


@pytest.fixture(autouse=True)
def fresh_job(monkeypatch):
    """One process runs one job, so a stopped job would otherwise carry into the next test."""
    monkeypatch.setattr(claude_plan, "_stopped_by", None)


def result_line(**fields):
    return {"type": "result", "subtype": "success", "is_error": False, "stop_reason": "tool_use", "terminal_reason": "completed",
            "api_error_status": None, "total_cost_usd": 0.42, "session_id": "not-saved",
            "usage": {"input_tokens": 2, "cache_creation_input_tokens": 1195, "cache_read_input_tokens": 10, "output_tokens": 73},
            "modelUsage": {"claude-sonnet-5-5": {"inputTokens": 2, "outputTokens": 73}}, **fields}


def limit_event(**info):
    return {"type": "rate_limit_event", "rate_limit_info": {
        "status": "allowed", "rateLimitType": "five_hour", "overageStatus": "rejected", "overageDisabledReason": "org_level_disabled",
        "isUsingOverage": False, **info}}


def answered(**fields):
    return [INIT,
            {"type": "assistant", "message": {"content": []}},
            limit_event(),
            result_line(**fields)]


def failed(text, code=None, status=None):
    """What a run that fails inside Claude Code prints: the exit code is 1 and stderr can be empty."""
    return [INIT,
            {"type": "assistant", "error": code, "message": {"content": [{"type": "text", "text": text}]}},
            result_line(is_error=True, result=text, stop_reason="stop_sequence", terminal_reason="api_error", api_error_status=status)]


class FakeClaude:
    def __init__(self, folder: Path):
        self.folder = folder
        self.records_folder = folder / "records"
        self.records_folder.mkdir()
        self.workspace = folder / "workspace"
        self.workspace.mkdir()
        self.program = folder / "fake_claude.py"
        self.program.write_text(FAKE_CLAUDE, encoding="utf-8")
        self.spec = folder / "spec.json"
        self.answers(answered(structured_output={"answer": "yes"}, result='{"answer":"yes"}'))

    def answers(self, stdout, stderr="", code=0, sleep=0):
        self.spec.write_text(json.dumps({"stdout": stdout, "stderr": stderr, "exit": code, "sleep": sleep}), encoding="utf-8")

    def settings(self, **overrides):
        env = {key: os.environ[key] for key in ("SYSTEMROOT", "PATH") if key in os.environ}
        runner = {"command": [sys.executable, str(self.program)], "cwd": str(self.workspace),
                  "env": {**env, "FAKE_SPEC": str(self.spec), "FAKE_RECORDS": str(self.records_folder)}}
        values = dict(ai_provider="claude-plan", claude_plan_model="sonnet", claude_plan_effort="high", claude_plan_vision=False,
                      claude_plan_runner=json.dumps(runner))
        values.update(overrides)
        return SimpleNamespace(**values)

    def records(self):
        return [json.loads(path.read_text(encoding="utf-8")) for path in sorted(self.records_folder.iterdir())]

    def last(self):
        return self.records()[-1]


@pytest.fixture
def claude(tmp_path):
    return FakeClaude(tmp_path)


def payload(**overrides):
    return {"model": "ignored", "max_tokens": 100, "temperature": 0.2, "messages": [
        {"role": "system", "content": "Pick the best clips."},
        {"role": "user", "content": "Here is the transcript."},
    ], "response_format": json_schema_format("answer", SCHEMA), **overrides}


def ask(settings, request=None, timeout=30.0):
    return asyncio.run(own_model_chat(settings, request or payload(), timeout=timeout))


def after(argv, flag):
    return argv[argv.index(flag) + 1]


class TestRun:
    def test_a_request_runs_claude_code_with_the_chosen_model_and_effort_and_nothing_else(self, claude):
        ask(claude.settings(claude_plan_model="opus", claude_plan_effort="max"))
        argv = claude.last()["argv"]
        assert argv[0] == "-p"
        assert after(argv, "--model") == "opus"
        assert after(argv, "--effort") == "max"
        # No tools: words in a transcript can only ever be read, never acted on.
        assert after(argv, "--tools") == ""
        assert "--strict-mcp-config" in argv
        assert "--disable-slash-commands" in argv
        # Not even the person's own Claude Code settings: one could name a key or remap a model.
        assert after(argv, "--setting-sources") == ""
        assert json.loads(after(argv, "--settings")) == {"disableAllHooks": True}
        assert after(argv, "--permission-mode") == "dontAsk"

    def test_a_run_saves_no_session(self, claude):
        ask(claude.settings())
        assert "--no-session-persistence" in claude.last()["argv"]

    def test_a_run_starts_in_the_folder_the_app_chose(self, claude):
        ask(claude.settings())
        assert Path(claude.last()["cwd"]).resolve() == claude.workspace.resolve()

    def test_only_the_environment_the_app_chose_reaches_claude_code(self, claude, monkeypatch):
        # Any of these would move the run off the plan, onto a key billed per token.
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-secret")
        monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "secret")
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-secret")
        ask(claude.settings())
        env = claude.last()["env"]
        assert not {"ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "OPENROUTER_API_KEY"} & set(env)

    def test_the_system_prompt_goes_in_a_file_that_is_removed_afterwards(self, claude):
        ask(claude.settings())
        record = claude.last()
        assert record["system_prompt"] == "Pick the best clips."
        assert not Path(record["system_prompt_file"]).exists()
        assert "Pick the best clips." not in record["argv"]
        assert list(claude.workspace.iterdir()) == []

    def test_a_request_without_a_system_prompt_still_replaces_claude_codes_own(self, claude):
        ask(claude.settings(), payload(messages=[{"role": "user", "content": "Hello"}]))
        assert claude.last()["system_prompt"].strip()

    def test_the_request_goes_in_on_stdin_as_one_user_message(self, claude):
        ask(claude.settings())
        record = claude.last()
        assert after(record["argv"], "--input-format") == "stream-json"
        assert after(record["argv"], "--output-format") == "stream-json"
        assert "--verbose" in record["argv"]
        lines = record["stdin"].strip().splitlines()
        assert len(lines) == 1
        message = json.loads(lines[0])
        assert message["type"] == "user" and message["message"]["role"] == "user"
        text = message["message"]["content"][0]
        assert text["type"] == "text" and text["text"].startswith("Here is the transcript.")

    def test_text_parts_of_one_message_stay_in_order(self, claude):
        request = payload(messages=[{"role": "user", "content": [{"type": "text", "text": "First"}, {"type": "text", "text": "Second"}]}],
                          response_format=None)
        ask(claude.settings(), request)
        content = json.loads(claude.last()["stdin"])["message"]["content"]
        assert [part["text"] for part in content] == ["First", "Second"]

    def test_images_are_handed_over_as_image_blocks(self, claude):
        request = payload(messages=[{"role": "user", "content": [
            {"type": "text", "text": "What is this?"},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,QUJD", "detail": "low"}},
        ]}], response_format=None)
        ask(claude.settings(claude_plan_vision=True), request)
        content = json.loads(claude.last()["stdin"])["message"]["content"]
        assert content[1] == {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "QUJD"}}

    def test_an_image_that_is_not_embedded_is_refused_without_a_run(self, claude):
        request = payload(messages=[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "https://example.com/a.jpg"}}]}],
                          response_format=None)
        with pytest.raises(AiClientError) as caught:
            ask(claude.settings(claude_plan_vision=True), request)
        assert caught.value.reason == "bad_request"
        assert claude.records() == []


class TestAnswer:
    def test_a_schema_request_uses_structured_output(self, claude):
        body, _ = ask(claude.settings())
        assert json.loads(after(claude.last()["argv"], "--json-schema")) == SCHEMA
        assert json.loads(body["choices"][0]["message"]["content"]) == {"answer": "yes"}
        assert body["choices"][0]["finish_reason"] == "stop"

    def test_a_request_without_a_schema_returns_the_text(self, claude):
        claude.answers(answered(result="Plain answer"))
        body, _ = ask(claude.settings(), payload(response_format=None))
        assert "--json-schema" not in claude.last()["argv"]
        assert body["choices"][0]["message"]["content"] == "Plain answer"

    def test_a_run_that_ends_without_structured_output_falls_back_to_its_text(self, claude):
        claude.answers(answered(result='{"answer":"from text"}'))
        body, _ = ask(claude.settings())
        assert json.loads(body["choices"][0]["message"]["content"]) == {"answer": "from text"}

    def test_tokens_are_counted_and_no_price_is_claimed(self, claude):
        body, usage = ask(claude.settings())
        # The plan pays for these: total_cost_usd is a client-side estimate, not a bill.
        assert usage["cost"] is None
        assert usage["prompt_tokens"] == 2 + 1195 + 10
        assert usage["completion_tokens"] == 73
        assert usage["total_tokens"] == 2 + 1195 + 10 + 73
        assert body["model"] == "claude-sonnet-5-5"

    def test_a_cut_off_answer_says_so(self, claude):
        claude.answers(answered(result="{", stop_reason="max_tokens"))
        body, _ = ask(claude.settings(), payload(response_format=None))
        assert body["choices"][0]["finish_reason"] == "length"

    def test_lines_that_are_not_json_are_skipped(self, claude):
        claude.answers(["Claude Code update available"] + answered(structured_output={"answer": "yes"}))
        body, _ = ask(claude.settings())
        assert json.loads(body["choices"][0]["message"]["content"]) == {"answer": "yes"}


class TestFailures:
    @pytest.mark.parametrize("stdout, stderr, reason, retryable, words", [
        (failed("Not logged in · Please run /login", "authentication_failed"), "", "claude_auth", False, "not signed in"),
        (failed("Your session has expired", "authentication_failed"), "", "claude_auth", False, "not signed in"),
        (failed("There's an issue with the selected model (opus). It may not exist or you may not have access to it.", "model_not_found", 404),
         "[claude-code:unrecognized_model] {}", "claude_model", False, "model"),
        (failed("You've hit your session limit · resets 3:45pm", "rate_limit", 429), "", "plan_limit", False, "usage limit"),
        (failed("You’ve hit your weekly limit · resets Monday"), "", "plan_limit", False, "usage limit"),
        (failed("Prompt is too long", "invalid_request", 400), "", "too_long", False, "too long"),
        (failed("Internal server error", "server_error", 500), "", "unavailable", True, "temporarily unavailable"),
        ([], "error: unknown option '--no-session-persistence'", "outdated", False, "too old"),
        ([], "Error: --json-schema is not valid JSON", "unavailable", True, "temporarily unavailable"),
        ([], "", "unavailable", True, "temporarily unavailable"),
    ])
    def test_a_failed_run_becomes_a_plain_app_error(self, claude, stdout, stderr, reason, retryable, words):
        claude.answers(stdout, stderr=stderr, code=1)
        with pytest.raises(AiClientError) as caught:
            ask(claude.settings())
        assert caught.value.reason == reason
        assert caught.value.retryable is retryable
        assert words in str(caught.value).lower()

    def test_claude_codes_own_words_are_kept_for_logs_only(self, claude):
        claude.answers(failed("Not logged in · Please run /login", "authentication_failed"), code=1)
        with pytest.raises(AiClientError) as caught:
            ask(claude.settings())
        assert "login" in caught.value.detail
        assert "login" not in str(caught.value)

    def test_a_run_that_reports_an_error_and_exits_zero_still_fails(self, claude):
        claude.answers(failed("You've hit your session limit · resets 3:45pm", "rate_limit"), code=0)
        with pytest.raises(AiClientError) as caught:
            ask(claude.settings())
        assert caught.value.reason == "plan_limit"

    def test_a_run_that_gives_up_on_the_schema_can_be_retried(self, claude):
        claude.answers(answered(subtype="error_max_structured_output_retries", is_error=True), code=1)
        with pytest.raises(AiClientError) as caught:
            ask(claude.settings())
        assert caught.value.reason == "invalid_response"
        assert caught.value.retryable is True

    def test_a_rejected_limit_is_a_plan_limit_whatever_words_come_with_it(self, claude):
        claude.answers([INIT, limit_event(status="rejected"), result_line(is_error=True, result="Something new")], code=1)
        with pytest.raises(AiClientError) as caught:
            ask(claude.settings())
        assert caught.value.reason == "plan_limit"

    def test_a_run_with_no_answer_at_all_can_be_retried(self, claude):
        claude.answers(answered())
        with pytest.raises(AiClientError) as caught:
            ask(claude.settings())
        assert caught.value.reason == "invalid_response"
        assert caught.value.retryable is True

    def test_a_program_that_cannot_be_started_is_reported_as_not_installed(self, claude):
        runner = json.loads(claude.settings().claude_plan_runner)
        runner["command"] = [str(claude.folder / "missing.exe")]
        with pytest.raises(AiClientError) as caught:
            ask(claude.settings(claude_plan_runner=json.dumps(runner)))
        assert caught.value.reason == "not_installed"
        assert caught.value.retryable is False

    @pytest.mark.parametrize("runner", [None, "", "not json", "[]", '{"command": []}', '{"command": ["claude"], "cwd": "x"}',
                                        '{"command": "claude", "cwd": "x", "env": {}}', '{"command": ["claude"], "env": {}, "cwd": ""}'])
    def test_missing_launch_details_never_fall_back_to_the_engines_environment(self, claude, runner):
        with pytest.raises(AiClientError) as caught:
            ask(claude.settings(claude_plan_runner=runner))
        assert caught.value.reason == "not_installed"
        assert claude.records() == []

    def test_a_run_that_takes_too_long_is_stopped(self, claude):
        claude.answers(answered(structured_output={"answer": "late"}), sleep=20)
        started = time.monotonic()
        with pytest.raises(AiClientError) as caught:
            asyncio.run(own_model_chat(claude.settings(), payload(), timeout=30, limit=0.5))
        assert caught.value.reason == "claude_timeout"
        assert time.monotonic() - started < 10
        assert claude.records() == []

    def test_cancelling_a_request_stops_the_run(self, claude):
        claude.answers(answered(structured_output={"answer": "late"}), sleep=2)

        async def scenario():
            task = asyncio.create_task(own_model_chat(claude.settings(), payload(), timeout=30))
            await asyncio.sleep(0.8)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            await asyncio.sleep(2)

        asyncio.run(scenario())
        # The fake writes its record after sleeping, so a killed run never gets to.
        assert claude.records() == []


class TestStoppingTheJob:
    def test_after_a_plan_limit_no_more_runs_start(self, claude):
        claude.answers(failed("You've hit your session limit · resets 3:45pm", "rate_limit", 429), code=1)
        with pytest.raises(AiClientError):
            ask(claude.settings())
        started = len(claude.records())
        claude.answers(answered(structured_output={"answer": "yes"}))
        with pytest.raises(AiClientError) as caught:
            ask(claude.settings())
        assert caught.value.reason == "plan_limit"
        assert len(claude.records()) == started

    def test_a_step_that_swallowed_the_failure_cannot_hide_it(self, claude):
        claude.answers(failed("Not logged in · Please run /login", "authentication_failed"), code=1)
        with pytest.raises(AiClientError):
            ask(claude.settings())
        with pytest.raises(AiClientError) as caught:
            claude_plan.raise_if_stopped()
        assert caught.value.reason == "claude_auth"

    def test_a_passing_failure_does_not_stop_the_job(self, claude):
        claude.answers(failed("Internal server error", "server_error", 500), code=1)
        with pytest.raises(AiClientError):
            ask(claude.settings())
        claude_plan.raise_if_stopped()
        claude.answers(answered(structured_output={"answer": "yes"}))
        assert ask(claude.settings())[0]["choices"]

    def test_a_refused_model_does_not_stop_the_job_by_itself(self, claude):
        # Planning fails on it anyway, and the same model can pass a smaller request.
        claude.answers(failed("Prompt is too long", "invalid_request", 400), code=1)
        with pytest.raises(AiClientError):
            ask(claude.settings())
        claude_plan.raise_if_stopped()

    def test_a_clip_review_that_lost_the_plan_is_not_marked_as_reviewed(self):
        from clip_engine.services import manual_editor

        async def judge(title, ranges, report, kind):
            # coherence_review swallows the failure and answers "unavailable".
            claude_plan._stopped_by = "plan_limit"
            report["coherence"] = {"attempts": [{}]}
            return False

        candidate = {"title": "A clip", "ranges": [(0, 1000)]}
        with pytest.raises(AiClientError) as caught:
            asyncio.run(manual_editor.review_candidate(candidate, SimpleNamespace(judge=judge)))
        assert caught.value.reason == "plan_limit"
        assert "review" not in candidate

    @pytest.mark.parametrize("reason", ["plan_limit", "claude_auth", "claude_billing", "claude_model", "claude_timeout", "not_installed", "outdated"])
    def test_framing_stops_asking_when_the_plan_cannot_answer(self, reason):
        from clip_engine.services.layout_analyzer import VISION_FATAL_REASONS

        assert reason in VISION_FATAL_REASONS

    def test_paid_overflow_keeps_the_answer_and_sends_nothing_more(self, claude):
        claude.answers([INIT, limit_event(status="allowed_warning", overageStatus="allowed", isUsingOverage=True),
                        result_line(structured_output={"answer": "yes"})])
        body, _ = ask(claude.settings())
        assert json.loads(body["choices"][0]["message"]["content"]) == {"answer": "yes"}
        with pytest.raises(AiClientError) as caught:
            claude_plan.raise_if_stopped()
        assert caught.value.reason == "plan_limit"

    def test_a_run_billed_to_an_api_key_stops_the_job(self, claude):
        claude.answers([{**INIT, "apiKeySource": "ANTHROPIC_API_KEY"}, result_line(structured_output={"answer": "yes"})])
        with pytest.raises(AiClientError) as caught:
            ask(claude.settings())
        assert caught.value.reason == "claude_billing"
        with pytest.raises(AiClientError):
            claude_plan.raise_if_stopped()

    @pytest.mark.skipif(sys.platform != "win32", reason="job objects are a Windows feature")
    def test_a_crashed_engine_takes_its_run_with_it(self, claude):
        claude.answers(answered(structured_output={"answer": "late"}), sleep=4)
        program = claude.folder / "crashing_engine.py"
        program.write_text(CRASHING_ENGINE, encoding="utf-8")
        env = {**os.environ, "PYTHONPATH": str(Path(claude_plan.__file__).parents[2]), "CLAUDE_RUNNER": claude.settings().claude_plan_runner}
        finished = subprocess.run([sys.executable, str(program)], env=env, capture_output=True, timeout=30)
        assert finished.returncode == 7, finished.stderr.decode(errors="replace")[-500:]
        time.sleep(5)
        # The fake writes its record after sleeping, so a run ended with the engine never gets to.
        assert claude.records() == []


class TestSessionList:
    def settings_with_list(self, claude, path):
        runner = json.loads(claude.settings().claude_plan_runner)
        return claude.settings(claude_plan_runner=json.dumps({**runner, "sessions": str(path)}))

    def test_every_run_gets_its_own_id_and_the_job_lists_it(self, claude):
        sessions = claude.folder / "claude-sessions.txt"
        settings = self.settings_with_list(claude, sessions)
        ask(settings)
        ask(settings)
        used = [after(record["argv"], "--session-id") for record in claude.records()]
        assert len(set(used)) == 2
        assert all(UUID.fullmatch(session) for session in used)
        assert sorted(sessions.read_text(encoding="utf-8").split()) == sorted(used)

    def test_a_run_is_listed_before_it_starts_so_a_failed_run_is_listed_too(self, claude):
        sessions = claude.folder / "claude-sessions.txt"
        claude.answers(failed("You've hit your session limit · resets 3:45pm", "rate_limit"), code=1)
        with pytest.raises(AiClientError):
            ask(self.settings_with_list(claude, sessions))
        assert sessions.read_text(encoding="utf-8").split() == [after(claude.last()["argv"], "--session-id")]

    def test_a_list_that_cannot_be_written_does_not_stop_the_job(self, claude):
        ask(self.settings_with_list(claude, claude.folder / "no-such-folder" / "claude-sessions.txt"))
        assert len(claude.records()) == 1

    def test_runs_still_save_no_session(self, claude):
        ask(self.settings_with_list(claude, claude.folder / "claude-sessions.txt"))
        assert "--no-session-persistence" in claude.last()["argv"]


class TestReviewAndEdit:
    def test_a_review_run_records_that_the_plan_picked_its_clips(self):
        from clip_engine.services.ai_clipping_pipeline import AIClippingPipeline
        from clip_engine.services.intelligence_planner import PlanningApiCosts

        plan = SimpleNamespace(api_costs=PlanningApiCosts(provider="claude-plan", model="claude-sonnet-5-5"))
        costs = AIClippingPipeline._review_costs(SimpleNamespace(settings=SimpleNamespace(ai_provider="claude-plan")), plan, 0.0)
        assert costs["planning"]["provider"] == "claude-plan"
        assert costs["total_estimated_cost_usd"] == 0.0

    def test_other_providers_keep_their_review_costs_as_they_were(self):
        from clip_engine.services.ai_clipping_pipeline import AIClippingPipeline
        from clip_engine.services.intelligence_planner import PlanningApiCosts

        plan = SimpleNamespace(api_costs=PlanningApiCosts(provider="gemini", model="gemini-test"))
        costs = AIClippingPipeline._review_costs(SimpleNamespace(settings=SimpleNamespace(ai_provider="gemini")), plan, 0.5)
        assert costs == {"total_estimated_cost_usd": 0.5, "cost_incomplete": True}


class TestAnswerText:
    @pytest.mark.parametrize("character", ["\u0085", "\u2028", "\u2029", "\x0b"])
    def test_a_line_separator_inside_an_answer_does_not_lose_it(self, claude, character):
        line = json.dumps(result_line(structured_output={"answer": f"a{character}b"}), ensure_ascii=False)
        claude.answers([INIT, line])
        body, _ = ask(claude.settings())
        assert json.loads(body["choices"][0]["message"]["content"]) == {"answer": f"a{character}b"}

    def test_the_prompt_file_stays_in_the_jobs_work_folder(self, claude, monkeypatch, tmp_path):
        work = tmp_path / "job-work"
        work.mkdir()
        monkeypatch.setenv("BRIDGECLIP_WORK_ROOT", str(work))
        ask(claude.settings())
        assert Path(claude.last()["system_prompt_file"]).parent.resolve() == work.resolve()
        assert list(work.iterdir()) == []


class TestProvider:
    def test_it_is_a_provider_like_the_others(self, claude):
        settings = claude.settings(claude_plan_model="haiku")
        assert provider_of(settings) == "claude-plan"
        assert active_model(settings) == "haiku"
        assert has_credentials(settings) is True
        assert has_credentials(claude.settings(claude_plan_runner=None)) is False

    def test_it_reads_images_only_when_the_person_said_so(self, claude):
        assert supports_images(claude.settings(claude_plan_vision=False)) is False
        assert supports_images(claude.settings(claude_plan_vision=True)) is True

    def test_it_waits_as_long_as_a_slow_effort_level_needs_within_a_callers_limit(self, claude):
        settings = claude.settings()
        assert request_timeout(settings, 600.0) == ai_client.DEFAULT_CLAUDE_PLAN_TIMEOUT_SECONDS
        assert request_timeout(settings, 600.0, limit=60.0) == 60.0

    def test_the_client_needs_no_network_and_no_key(self, claude):
        client = get_ai_client(claude.settings(), timeout=30)
        assert isinstance(client._transport, claude_plan.ClaudePlanTransport)
        assert "authorization" not in client.headers
        asyncio.run(client.aclose())

    def test_settings_read_it_from_the_environment(self, monkeypatch):
        monkeypatch.setenv("AI_PROVIDER", "claude-plan")
        monkeypatch.setenv("CLAUDE_PLAN_MODEL", "opus")
        monkeypatch.setenv("CLAUDE_PLAN_EFFORT", "xhigh")
        monkeypatch.setenv("CLAUDE_PLAN_VISION", "true")
        monkeypatch.setenv("CLAUDE_PLAN_RUNNER", '{"command": ["claude"]}')
        settings = Settings(_env_file=None)
        assert (settings.ai_provider, settings.claude_plan_model, settings.claude_plan_effort) == ("claude-plan", "opus", "xhigh")
        assert settings.claude_plan_vision is True
        assert settings.claude_plan_runner == '{"command": ["claude"]}'

    def test_settings_refuse_a_model_or_effort_outside_the_lists(self):
        with pytest.raises(ValueError):
            Settings(_env_file=None, claude_plan_model="fable")
        with pytest.raises(ValueError):
            Settings(_env_file=None, claude_plan_effort="--dangerously-skip-permissions")

    def test_web_research_stays_off_as_on_every_provider_but_openrouter(self):
        assert Settings(_env_file=None, ai_provider="claude-plan", source_context_web_research=True).source_context_web_research is False

    def test_at_most_two_runs_go_at_once(self, claude):
        claude.answers(answered(structured_output={"answer": "yes"}), sleep=0.5)

        async def scenario():
            await asyncio.gather(*(own_model_chat(claude.settings(), payload(), timeout=30) for _ in range(5)))

        asyncio.run(scenario())
        records = claude.records()
        assert len(records) == 5
        spans = [(record["started"], record["finished"]) for record in records]
        most = max(sum(1 for start, end in spans if start <= moment < end) for moment, _ in spans)
        assert most <= claude_plan.MAX_RUNS == 2

    def test_a_planner_request_runs_through_the_same_path(self, claude):
        from clip_engine.services.intelligence_planner import IntelligencePlannerService

        settings = Settings(_env_file=None, ai_provider="claude-plan", claude_plan_model="sonnet", claude_plan_effort="high",
                            claude_plan_runner=claude.settings().claude_plan_runner)
        claude.answers(answered(structured_output={"segments": []}))
        planner = IntelligencePlannerService()
        planner.settings = settings

        async def scenario():
            try:
                return await planner._call_openrouter("ignored", [{"role": "system", "content": "Plan."},
                                                                  {"role": "user", "content": "Transcript."}])
            finally:
                await planner.close()

        body, usage = asyncio.run(scenario())
        assert json.loads(body["choices"][0]["message"]["content"]) == {"segments": []}
        assert usage["cost"] is None
        record = claude.last()
        assert after(record["argv"], "--model") == "sonnet"
        assert record["system_prompt"] == "Plan."
        assert isinstance(json.loads(after(record["argv"], "--json-schema")), dict)
