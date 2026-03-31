"""Tests for Phase 5: agent interface, tools registry, memory manager."""

import asyncio
import json
import os
import pytest
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

from craving_mind.agent.interface import CLIProvider, LLMResponse, MockProvider, AgentInterface
from craving_mind.agent.tools import ToolsRegistry
from craving_mind.agent.memory import MemoryManager
from craving_mind.orchestrator.budget import BudgetManager


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

BASE_CONFIG = {
    "budget": {
        "base_tokens": 50000,
        "circuit_breaker_pct": 0.15,
        "venture_decay": 0.5,
        "rnd_lambda": 0.0001,
        "rnd_max_pct": 0.30,
        "rnd_min_success_rate": 0.50,
        "critical_starvation_pct": 0.10,
    },
    "memory": {
        "graveyard_ttl_epochs": 10,
        "bible_max_weight_pct": 0.20,
    },
    "sandbox": {
        "timeout_seconds": 5,
        "allowed_imports": ["re", "math", "collections"],
    },
}


@pytest.fixture
def budget():
    bm = BudgetManager(BASE_CONFIG)
    bm.start_epoch(1)
    return bm


@pytest.fixture
def agent_dir(tmp_path):
    return str(tmp_path / "agent_workspace")


@pytest.fixture
def memory(agent_dir):
    return MemoryManager(BASE_CONFIG, agent_dir)


# ---------------------------------------------------------------------------
# LLMResponse
# ---------------------------------------------------------------------------

class TestLLMResponse:
    def test_fields(self):
        r = LLMResponse(
            content="hello",
            tool_calls=[{"name": "run_compress", "arguments": {}}],
            usage={"input_tokens": 10, "output_tokens": 5},
            stop_reason="end_turn",
        )
        assert r.content == "hello"
        assert len(r.tool_calls) == 1
        assert r.usage["input_tokens"] == 10
        assert r.stop_reason == "end_turn"

    def test_empty_tool_calls(self):
        r = LLMResponse(content="x", tool_calls=[], usage={}, stop_reason="end_turn")
        assert r.tool_calls == []


# ---------------------------------------------------------------------------
# MockProvider
# ---------------------------------------------------------------------------

class TestMockProvider:
    def test_returns_canned_responses_in_order(self):
        r1 = LLMResponse("first", [], {"input_tokens": 10, "output_tokens": 5}, "end_turn")
        r2 = LLMResponse("second", [], {"input_tokens": 20, "output_tokens": 10}, "end_turn")
        provider = MockProvider([r1, r2])

        got1 = provider.chat([{"role": "user", "content": "hi"}])
        got2 = provider.chat([{"role": "user", "content": "bye"}])

        assert got1.content == "first"
        assert got2.content == "second"

    def test_default_response_after_exhaustion(self):
        provider = MockProvider([])
        resp = provider.chat([{"role": "user", "content": "hi"}])
        assert resp.content == "mock response"
        assert resp.stop_reason == "end_turn"

    def test_records_call_history(self):
        provider = MockProvider()
        msgs = [{"role": "user", "content": "test"}]
        tools = [{"name": "run_compress"}]
        provider.chat(msgs, tools=tools, system="sys prompt")

        assert len(provider.call_history) == 1
        call = provider.call_history[0]
        assert call["messages"] == msgs
        assert call["tools"] == tools
        assert call["system"] == "sys prompt"

    def test_multiple_calls_tracked(self):
        provider = MockProvider()
        for _ in range(3):
            provider.chat([{"role": "user", "content": "msg"}])
        assert len(provider.call_history) == 3


# ---------------------------------------------------------------------------
# MemoryManager
# ---------------------------------------------------------------------------

class TestMemoryManager:
    def test_read_nonexistent_returns_empty(self, memory):
        assert memory.read_file("bible.md") == ""

    def test_write_then_read(self, memory):
        memory.write_file("bible.md", "# My Bible\nSome content.")
        assert memory.read_file("bible.md") == "# My Bible\nSome content."

    def test_write_creates_agent_dir(self, tmp_path):
        new_dir = str(tmp_path / "new_agent")
        m = MemoryManager(BASE_CONFIG, new_dir)
        assert os.path.isdir(new_dir)

    def test_backup_captures_all_files(self, memory):
        memory.write_file("bible.md", "bible content")
        memory.write_file("compress.py", "def compress(t, r): return t")
        backup = memory.backup()
        assert backup["bible.md"] == "bible content"
        assert backup["compress.py"] == "def compress(t, r): return t"
        assert backup["graveyard.md"] == ""

    def test_restore_writes_back(self, memory):
        memory.write_file("bible.md", "original")
        backup = memory.backup()
        memory.write_file("bible.md", "overwritten")
        memory.restore(backup)
        assert memory.read_file("bible.md") == "original"

    def test_restore_skips_empty_files(self, memory):
        memory.write_file("bible.md", "keep me")
        backup = {"bible.md": "", "compress.py": ""}
        memory.restore(backup)
        # Empty string in backup → skip write, so "keep me" stays
        assert memory.read_file("bible.md") == "keep me"

    def test_init_from_inheritance_compress(self, memory):
        code = "def compress(t, r): return t[:int(len(t)*r)]"
        memory.init_from_inheritance(prev_compress=code)
        assert memory.read_file("compress.py") == code

    def test_init_from_inheritance_graveyard_tagged(self, memory):
        grave = "<!-- AMENDMENT: epoch=1 -->\nold idea\n<!-- /AMENDMENT -->"
        memory.init_from_inheritance(prev_graveyard=grave)
        stored = memory.read_file("graveyard.md")
        assert "inherited" in stored

    def test_cleanup_graveyard_removes_expired(self, memory):
        content = (
            "<!-- AMENDMENT epoch=1 -->\nold entry\n<!-- /AMENDMENT -->\n"
            "<!-- AMENDMENT epoch=15 -->\nnew entry\n<!-- /AMENDMENT -->\n"
        )
        memory.write_file("graveyard.md", content)
        memory.cleanup_graveyard(current_epoch=20)
        remaining = memory.read_file("graveyard.md")
        # epoch=1 is 19 epochs ago (> TTL=10), epoch=15 is 5 epochs ago (≤ TTL)
        assert "old entry" not in remaining
        assert "new entry" in remaining

    def test_cleanup_graveyard_empty(self, memory):
        memory.cleanup_graveyard(current_epoch=5)  # no error on empty


# ---------------------------------------------------------------------------
# ToolsRegistry
# ---------------------------------------------------------------------------

class TestToolsRegistry:
    @pytest.fixture
    def mock_sandbox(self):
        sb = MagicMock()
        from craving_mind.agent.sandbox import SandboxResult
        sb.run_compress.return_value = SandboxResult(
            success=True, output='{"result": "compressed"}', error="", return_value="compressed"
        )
        sb.run_script.return_value = SandboxResult(
            success=True, output="script ran", error=""
        )
        sb.validate_imports.return_value = (True, "")
        return sb

    @pytest.fixture
    def registry(self, mock_sandbox, memory, budget):
        return ToolsRegistry(mock_sandbox, memory, budget)

    def test_get_tool_definitions_returns_five(self, registry):
        tools = registry.get_tool_definitions()
        assert len(tools) == 5

    def test_tool_names(self, registry):
        names = {t["name"] for t in registry.get_tool_definitions()}
        assert names == {"run_compress", "read_file", "write_file", "run_script", "audit_budget"}

    def test_each_tool_has_input_schema(self, registry):
        for tool in registry.get_tool_definitions():
            assert "input_schema" in tool
            assert "type" in tool["input_schema"]

    def test_execute_read_file(self, registry, memory):
        memory.write_file("bible.md", "hello bible")
        result = registry.execute("read_file", {"filename": "bible.md"})
        assert result["content"] == "hello bible"

    def test_execute_write_file(self, registry, memory):
        registry.execute("write_file", {"filename": "compress.py", "content": "# code"})
        assert memory.read_file("compress.py") == "# code"

    def test_execute_write_file_forbidden_import_rejected(self, mock_sandbox, memory, budget):
        mock_sandbox.validate_imports.return_value = (False, "Forbidden import: anthropic")
        registry = ToolsRegistry(mock_sandbox, memory, budget)
        result = registry.execute(
            "write_file",
            {"filename": "compress.py", "content": "from anthropic import Anthropic\n"},
        )
        assert result["success"] is False
        assert "anthropic" in result["error"].lower()
        # File must NOT have been saved
        assert memory.read_file("compress.py") != "from anthropic import Anthropic\n"

    def test_execute_write_file_smoke_test_passed(self, mock_sandbox, memory, budget):
        mock_smoke = MagicMock()
        mock_smoke.run.return_value = (True, [])
        registry = ToolsRegistry(mock_sandbox, memory, budget, smoke_test=mock_smoke)
        result = registry.execute(
            "write_file", {"filename": "compress.py", "content": "def compress(t, r): return t"}
        )
        assert result["success"] is True
        assert result.get("smoke_test") == "PASSED"

    def test_execute_write_file_smoke_test_failed(self, mock_sandbox, memory, budget):
        mock_smoke = MagicMock()
        mock_smoke.run.return_value = (False, ["Sample 0: RuntimeError: boom"])
        registry = ToolsRegistry(mock_sandbox, memory, budget, smoke_test=mock_smoke)
        result = registry.execute(
            "write_file", {"filename": "compress.py", "content": "def compress(t, r): raise RuntimeError('boom')"}
        )
        assert result["success"] is True  # file was saved but smoke failed
        assert result.get("smoke_test") == "FAILED"
        assert len(result["errors"]) > 0

    def test_execute_write_non_compress_skips_validation(self, mock_sandbox, memory, budget):
        # validate_imports should NOT be called for bible.md
        registry = ToolsRegistry(mock_sandbox, memory, budget)
        registry.execute("write_file", {"filename": "bible.md", "content": "some notes"})
        mock_sandbox.validate_imports.assert_not_called()
        assert memory.read_file("bible.md") == "some notes"

    def test_execute_run_compress(self, registry, mock_sandbox, memory):
        memory.write_file("compress.py", "def compress(t, r): return t")
        result = registry.execute("run_compress", {"text": "hello world", "target_ratio": 0.5})
        assert result["success"] is True
        assert result["output"] == "compressed"
        mock_sandbox.run_compress.assert_called_once()

    def test_execute_run_script(self, registry, mock_sandbox):
        result = registry.execute("run_script", {"code": "print('hi')"})
        assert result["success"] is True
        assert result["output"] == "script ran"

    def test_execute_audit_budget(self, registry, budget):
        result = registry.execute("audit_budget", {})
        assert "remaining" in result
        assert "pulse" in result
        assert result["remaining"] == budget.remaining

    def test_execute_unknown_tool(self, registry):
        result = registry.execute("nonexistent_tool", {})
        assert "error" in result
        assert "nonexistent_tool" in result["error"]


# ---------------------------------------------------------------------------
# AgentInterface
# ---------------------------------------------------------------------------

class TestAgentInterface:
    @pytest.fixture
    def mock_sandbox(self):
        sb = MagicMock()
        from craving_mind.agent.sandbox import SandboxResult
        sb.run_compress.return_value = SandboxResult(
            success=True, output='{"result": "compressed"}', error="", return_value="compressed"
        )
        sb.run_script.return_value = SandboxResult(success=True, output="", error="")
        return sb

    @pytest.fixture
    def provider(self):
        return MockProvider()

    @pytest.fixture
    def interface(self, provider, budget, mock_sandbox, memory):
        registry = ToolsRegistry(mock_sandbox, memory, budget)
        iface = AgentInterface(BASE_CONFIG, provider, budget, mock_sandbox, registry)
        iface.start_epoch(1, system_prompt="You are Crav.")
        return iface

    def test_start_epoch_resets_conversation(self, interface):
        interface.conversation.append({"role": "user", "content": "old"})
        interface.start_epoch(2, system_prompt="new prompt")
        assert interface.conversation == []
        assert interface._system_prompt == "new prompt"

    def test_send_task_includes_pulse_and_ratio(self, interface, provider):
        interface.send_task("Hello world text", 0.5)
        last_call = provider.call_history[-1]
        user_msg = last_call["messages"][-1]["content"]
        assert "[B:" in user_msg          # pulse present
        assert "50%" in user_msg          # ratio formatted
        assert "Hello world text" in user_msg

    def test_send_task_appends_to_conversation(self, interface):
        interface.send_task("some text", 0.7)
        # user message + assistant response
        assert len(interface.conversation) >= 2
        assert interface.conversation[0]["role"] == "user"
        assert interface.conversation[1]["role"] == "assistant"

    def test_send_task_returns_dict_with_expected_keys(self, interface):
        result = interface.send_task("some text", 0.5)
        assert "content" in result
        assert "tool_calls" in result
        assert "tool_results" in result
        assert "tokens_spent" in result
        assert "is_oom" in result

    def test_send_task_deducts_tokens_from_budget(self, interface, budget):
        before = budget.remaining
        interface.send_task("text", 0.5)
        assert budget.remaining < before

    def test_send_feedback_appends_to_conversation(self, interface):
        interface.send_task("text", 0.5)
        before = len(interface.conversation)
        interface.send_feedback({"score": 0.9, "pass": True})
        assert len(interface.conversation) == before + 1
        last = interface.conversation[-1]
        assert last["role"] == "user"
        assert "score" in last["content"]

    def test_request_rnd_includes_pulse(self, interface, provider):
        interface.request_rnd()
        last_call = provider.call_history[-1]
        user_msg = last_call["messages"][-1]["content"]
        assert "[B:" in user_msg
        assert "R&D" in user_msg

    def test_tool_calls_handled_and_added_to_conversation(self, provider, budget, mock_sandbox, memory):
        """When provider returns tool calls, results are appended to conversation."""
        from craving_mind.agent.sandbox import SandboxResult
        mock_sandbox.run_compress.return_value = SandboxResult(
            success=True, output="ok", error="", return_value="compressed text"
        )
        memory.write_file("compress.py", "def compress(t, r): return t")

        canned = LLMResponse(
            content="",
            tool_calls=[{
                "id": "tc_001",
                "name": "run_compress",
                "arguments": {"text": "hello", "target_ratio": 0.5},
            }],
            usage={"input_tokens": 100, "output_tokens": 50},
            stop_reason="tool_use",
        )
        provider = MockProvider([canned])
        registry = ToolsRegistry(mock_sandbox, memory, budget)
        iface = AgentInterface(BASE_CONFIG, provider, budget, mock_sandbox, registry)
        iface.start_epoch(1, "sys")

        result = iface.send_task("hello", 0.5)

        assert len(result["tool_calls"]) == 1
        assert len(result["tool_results"]) == 1
        assert result["tool_results"][0]["name"] == "run_compress"
        # Conversation should have: user, assistant, user(tool_results)
        assert len(iface.conversation) == 3
        assert iface.conversation[2]["role"] == "user"

    def test_oom_flag_when_budget_exhausted(self, provider, mock_sandbox, memory):
        """is_oom=True when budget runs out."""
        config = {**BASE_CONFIG, "budget": {**BASE_CONFIG["budget"], "base_tokens": 100}}
        bm = BudgetManager(config)
        bm.start_epoch(1)

        # Provide a response that costs more than the budget
        canned = LLMResponse(
            content="response",
            tool_calls=[],
            usage={"input_tokens": 200, "output_tokens": 200},
            stop_reason="end_turn",
        )
        prov = MockProvider([canned])
        registry = ToolsRegistry(mock_sandbox, memory, bm)
        iface = AgentInterface(config, prov, bm, mock_sandbox, registry)
        iface.start_epoch(1, "sys")

        result = iface.send_task("text", 0.5)
        assert result["is_oom"] is True

    def test_pass_tools_to_provider(self, interface, provider):
        interface.send_task("text", 0.5)
        last_call = provider.call_history[-1]
        assert last_call["tools"] is not None
        assert len(last_call["tools"]) == 5

    def test_system_prompt_passed_to_provider(self, interface, provider):
        interface.send_task("text", 0.5)
        last_call = provider.call_history[-1]
        assert last_call["system"] == "You are Crav."


# ---------------------------------------------------------------------------
# CLIProvider
# ---------------------------------------------------------------------------

def _make_sdk_types(text_response: str, session_id: str = "sess-001", usage: dict = None):
    """Build minimal SDK message objects for mocking query()."""
    from claude_code_sdk.types import AssistantMessage, TextBlock, ResultMessage

    assistant_msg = AssistantMessage(
        content=[TextBlock(text=text_response)],
        model="claude-haiku-4-5-20251001",
    )
    result_msg = ResultMessage(
        subtype="success",
        duration_ms=100,
        duration_api_ms=90,
        is_error=False,
        num_turns=1,
        session_id=session_id,
        usage=usage or {"input_tokens": 20, "output_tokens": 10},
    )
    return [assistant_msg, result_msg]


async def _async_gen(items):
    for item in items:
        yield item


class TestCLIProvider:
    def test_parse_response_valid_json_no_tools(self):
        p = CLIProvider()
        content, calls = p._parse_response('{"content": "hello", "tool_calls": []}')
        assert content == "hello"
        assert calls == []

    def test_parse_response_valid_json_with_tool_call(self):
        p = CLIProvider()
        payload = json.dumps({
            "content": "thinking",
            "tool_calls": [{"name": "run_compress", "arguments": {"text": "hi", "target_ratio": 0.5}}],
        })
        content, calls = p._parse_response(payload)
        assert content == "thinking"
        assert len(calls) == 1
        assert calls[0]["name"] == "run_compress"
        assert calls[0]["id"] == "cli_0000"
        assert calls[0]["arguments"]["target_ratio"] == 0.5

    def test_parse_response_strips_markdown_fences(self):
        p = CLIProvider()
        fenced = '```json\n{"content": "ok", "tool_calls": []}\n```'
        content, calls = p._parse_response(fenced)
        assert content == "ok"
        assert calls == []

    def test_parse_response_invalid_json_returns_raw(self):
        p = CLIProvider()
        content, calls = p._parse_response("not json at all")
        assert content == "not json at all"
        assert calls == []

    def test_build_prompt_includes_system(self):
        p = CLIProvider()
        prompt = p._build_prompt([], None, system="You are Crav.")
        assert "You are Crav." in prompt

    def test_build_prompt_includes_tool_definitions(self):
        p = CLIProvider()
        tools = [{"name": "run_compress", "description": "compress text"}]
        prompt = p._build_prompt([], tools, system="")
        assert "run_compress" in prompt
        assert "tool_calls" in prompt

    def test_build_prompt_includes_messages(self):
        p = CLIProvider()
        messages = [
            {"role": "user", "content": "compress this"},
            {"role": "assistant", "content": '{"content": "ok", "tool_calls": []}'},
        ]
        prompt = p._build_prompt(messages, None, system="")
        assert "compress this" in prompt
        assert "USER" in prompt
        assert "ASSISTANT" in prompt

    def test_build_prompt_handles_tool_result_list_content(self):
        p = CLIProvider()
        messages = [{
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "tc_001", "content": '{"success": true}'},
            ],
        }]
        prompt = p._build_prompt(messages, None, system="")
        assert "tc_001" in prompt
        assert "success" in prompt

    def test_new_session_clears_session_id(self):
        p = CLIProvider()
        p._session_id = "old-session"
        p.new_session()
        assert p._session_id is None

    def test_chat_returns_llm_response(self):
        p = CLIProvider(model="haiku")
        response_json = json.dumps({"content": "compressed text", "tool_calls": []})
        sdk_messages = _make_sdk_types(response_json)

        with patch("craving_mind.agent.interface.CLIProvider.chat") as mock_chat:
            mock_chat.return_value = LLMResponse(
                content="compressed text",
                tool_calls=[],
                usage={"input_tokens": 20, "output_tokens": 10},
                stop_reason="end_turn",
            )
            resp = p.chat([{"role": "user", "content": "compress this"}])
            # Direct test via _parse_response + manual call simulation
        # Test via the actual parse path instead
        content, calls = p._parse_response(response_json)
        assert content == "compressed text"
        assert calls == []

    def test_chat_with_tool_call_via_query_mock(self):
        """Full chat() integration with mocked SDK query."""
        p = CLIProvider(model="haiku")
        tool_payload = json.dumps({
            "content": "I'll compress that",
            "tool_calls": [{"name": "run_compress", "arguments": {"text": "hello", "target_ratio": 0.5}}],
        })
        sdk_messages = _make_sdk_types(tool_payload, session_id="sess-abc", usage={"input_tokens": 50, "output_tokens": 30})

        async def fake_query(*, prompt, options):
            for m in sdk_messages:
                yield m

        with patch("craving_mind.agent.interface.query", fake_query):
            resp = p.chat(
                messages=[{"role": "user", "content": "hello"}],
                tools=[{"name": "run_compress"}],
                system="You are Crav.",
            )

        assert resp.content == "I'll compress that"
        assert len(resp.tool_calls) == 1
        assert resp.tool_calls[0]["name"] == "run_compress"
        assert resp.stop_reason == "tool_use"
        assert resp.usage["input_tokens"] == 50
        assert resp.usage["output_tokens"] == 30
        # Session ID captured for conversation continuity
        assert p._session_id == "sess-abc"

    def test_chat_no_tools_returns_end_turn(self):
        p = CLIProvider(model="haiku")
        response_json = json.dumps({"content": "done", "tool_calls": []})
        sdk_messages = _make_sdk_types(response_json)

        async def fake_query(*, prompt, options):
            for m in sdk_messages:
                yield m

        with patch("craving_mind.agent.interface.query", fake_query):
            resp = p.chat([{"role": "user", "content": "hello"}])

        assert resp.stop_reason == "end_turn"
        assert resp.tool_calls == []

    def test_chat_resumes_session_on_second_call(self):
        p = CLIProvider(model="haiku")
        p._session_id = "prev-session"

        captured_options = {}

        async def fake_query(*, prompt, options):
            captured_options["resume"] = options.resume
            for m in _make_sdk_types('{"content": "ok", "tool_calls": []}'):
                yield m

        with patch("craving_mind.agent.interface.query", fake_query):
            p.chat([{"role": "user", "content": "hi"}])

        assert captured_options["resume"] == "prev-session"

    def test_chat_estimates_tokens_when_usage_missing(self):
        p = CLIProvider(model="haiku")
        from claude_code_sdk.types import AssistantMessage, TextBlock, ResultMessage
        response_json = '{"content": "hello world", "tool_calls": []}'
        msgs = [
            AssistantMessage(content=[TextBlock(text=response_json)], model="haiku"),
            ResultMessage(
                subtype="success", duration_ms=50, duration_api_ms=40,
                is_error=False, num_turns=1, session_id="s1",
                usage=None,  # no usage data
            ),
        ]

        async def fake_query(*, prompt, options):
            for m in msgs:
                yield m

        with patch("craving_mind.agent.interface.query", fake_query):
            resp = p.chat([{"role": "user", "content": "hi"}])

        assert resp.usage["input_tokens"] >= 1
        assert resp.usage["output_tokens"] >= 1

    def test_chat_strips_claudecode_env(self):
        """CLAUDECODE env var is removed from options.env to allow nested calls."""
        p = CLIProvider(model="haiku")
        captured_env = {}

        async def fake_query(*, prompt, options):
            captured_env.update(options.env)
            for m in _make_sdk_types('{"content": "ok", "tool_calls": []}'):
                yield m

        with patch.dict(os.environ, {"CLAUDECODE": "1"}):
            with patch("craving_mind.agent.interface.query", fake_query):
                p.chat([{"role": "user", "content": "hi"}])

        assert "CLAUDECODE" not in captured_env

    def test_chat_missing_sdk_raises_runtime_error(self):
        """When query is None (SDK not installed), chat() raises RuntimeError."""
        p = CLIProvider()
        with patch("craving_mind.agent.interface.query", None):
            with pytest.raises(RuntimeError, match="claude-code-sdk"):
                p.chat([{"role": "user", "content": "hi"}])
