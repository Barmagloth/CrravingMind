"""Agent interface and abstract LLM provider."""

import asyncio
import json
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

try:
    from claude_code_sdk import query as _sdk_query
    from claude_code_sdk import ClaudeCodeOptions as _SdkOptions
    from claude_code_sdk.types import AssistantMessage as _AssistantMessage
    from claude_code_sdk.types import TextBlock as _TextBlock
    from claude_code_sdk.types import ResultMessage as _ResultMessage
    _SDK_AVAILABLE = True
except ImportError:
    _sdk_query = None  # type: ignore[assignment]
    _SdkOptions = None  # type: ignore[assignment]
    _AssistantMessage = None  # type: ignore[assignment]
    _TextBlock = None  # type: ignore[assignment]
    _ResultMessage = None  # type: ignore[assignment]
    _SDK_AVAILABLE = False

# Alias used in tests as patch target: craving_mind.agent.interface.query
query = _sdk_query


@dataclass
class LLMResponse:
    content: str
    tool_calls: list  # [{id, name, arguments}]
    usage: dict       # {input_tokens, output_tokens}
    stop_reason: str  # "end_turn", "tool_use", etc.


class LLMProvider(ABC):
    @abstractmethod
    def chat(self, messages: list, tools: list = None, system: str = "", max_tokens: int = 4096) -> LLMResponse:
        pass


class AnthropicProvider(LLMProvider):
    """Anthropic Claude API provider."""

    def __init__(self, model: str, api_key: str = None):
        self.model = model
        self._client = None
        self._api_key = api_key

    def _get_client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic(api_key=self._api_key)
        return self._client

    def chat(self, messages, tools=None, system="", max_tokens=4096) -> LLMResponse:
        client = self._get_client()
        kwargs = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = tools

        response = client.messages.create(**kwargs)

        content = ""
        tool_calls = []
        for block in response.content:
            if block.type == "text":
                content += block.text
            elif block.type == "tool_use":
                tool_calls.append({
                    "id": block.id,
                    "name": block.name,
                    "arguments": block.input,
                })

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            usage={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
            stop_reason=response.stop_reason,
        )


class CLIProvider(LLMProvider):
    """LLM provider that uses the claude CLI via claude-code-sdk.

    Designed for users who have Claude Code installed but no Anthropic API key.
    Tool calls are simulated via structured JSON in the response text.

    Session continuity within an epoch is maintained by resuming the same
    claude session via session_id, so the model retains context across turns.
    """

    _RESPONSE_SCHEMA = {
        "type": "object",
        "properties": {
            "content": {"type": "string"},
            "tool_calls": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "arguments": {"type": "object"},
                    },
                    "required": ["name", "arguments"],
                },
            },
        },
        "required": ["content", "tool_calls"],
    }

    def __init__(self, model: str = "haiku"):
        self.model = model
        self._session_id: str | None = None

    def new_session(self) -> None:
        """Drop the current session so the next call starts fresh."""
        self._session_id = None

    def _build_prompt(
        self, messages: list, tools: list | None, system: str
    ) -> str:
        """Format the full conversation + tool spec into one prompt string."""
        parts: list[str] = []

        if system:
            parts.append(f"[SYSTEM]\n{system}\n[/SYSTEM]\n")

        if tools:
            tool_specs = json.dumps(tools, indent=2)
            parts.append(
                "You have access to the following tools. "
                "When you want to call a tool, respond ONLY with a JSON object "
                "matching this exact schema — no markdown fences, no surrounding text:\n"
                '{"content": "<thinking or empty string>", '
                '"tool_calls": [{"name": "<tool_name>", "arguments": {<args>}}]}\n'
                "If you do NOT need to call a tool, respond with the same JSON schema "
                'but with an empty tool_calls array: {"content": "<your reply>", "tool_calls": []}\n\n'
                f"Available tools:\n{tool_specs}\n"
            )
        else:
            parts.append(
                "Respond ONLY with a JSON object: "
                '{"content": "<your reply>", "tool_calls": []}\n'
            )

        for msg in messages:
            role = msg["role"].upper()
            content = msg["content"]
            if isinstance(content, list):
                # Tool result messages use a list of content blocks.
                text_parts = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "tool_result":
                        text_parts.append(
                            f"[Tool result for {block.get('tool_use_id', '')}]: "
                            f"{block.get('content', '')}"
                        )
                    else:
                        text_parts.append(str(block))
                content = "\n".join(text_parts)
            parts.append(f"[{role}]\n{content}")

        parts.append("[ASSISTANT]")
        return "\n\n".join(parts)

    def _parse_response(self, raw_text: str) -> tuple[str, list]:
        """Extract (content, tool_calls) from the model's response text.

        Returns (raw_text, []) if JSON cannot be parsed, so the caller still
        gets something useful rather than crashing.
        """
        text = raw_text.strip()

        # Strip markdown code fences if present.
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()

        try:
            data = json.loads(text)
            content = data.get("content", "")
            raw_calls = data.get("tool_calls", [])
            tool_calls = []
            for i, tc in enumerate(raw_calls):
                tool_calls.append({
                    "id": f"cli_{i:04d}",
                    "name": tc.get("name", ""),
                    "arguments": tc.get("arguments", {}),
                })
            return content, tool_calls
        except (json.JSONDecodeError, AttributeError):
            return raw_text, []

    def chat(
        self,
        messages: list,
        tools: list = None,
        system: str = "",
        max_tokens: int = 4096,
    ) -> LLMResponse:
        # `query` is the module-level alias — tests can patch it at
        # craving_mind.agent.interface.query and this method picks it up.
        import craving_mind.agent.interface as _mod
        _query = _mod.query
        if _query is None:
            raise RuntimeError(
                "claude-code-sdk is not installed. "
                "Run: pip install claude-code-sdk"
            )
        if _SdkOptions is None:
            raise RuntimeError(
                "claude-code-sdk is not installed. "
                "Run: pip install claude-code-sdk"
            )

        prompt = self._build_prompt(messages, tools, system)

        # Remove CLAUDECODE so we can run nested from within a Claude Code session.
        env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}
        options = _SdkOptions(
            model=self.model,
            allowed_tools=[],         # no filesystem tools — text-only
            permission_mode="bypassPermissions",
            env=env,
            resume=self._session_id,
        )

        collected_text: list[str] = []
        usage_data: dict = {}

        async def _run() -> None:
            nonlocal usage_data
            async for msg in _query(prompt=prompt, options=options):
                if _AssistantMessage and isinstance(msg, _AssistantMessage):
                    for block in msg.content:
                        if _TextBlock and isinstance(block, _TextBlock):
                            collected_text.append(block.text)
                elif _ResultMessage and isinstance(msg, _ResultMessage):
                    if msg.usage:
                        usage_data = msg.usage
                    if msg.session_id:
                        self._session_id = msg.session_id

        asyncio.run(_run())

        raw_text = "".join(collected_text)
        content, tool_calls = self._parse_response(raw_text)

        # Estimate token usage from text length if SDK didn't report it.
        input_tokens = usage_data.get("input_tokens") or max(1, len(prompt) // 4)
        output_tokens = usage_data.get("output_tokens") or max(1, len(raw_text) // 4)

        stop_reason = "tool_use" if tool_calls else "end_turn"

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            usage={"input_tokens": int(input_tokens), "output_tokens": int(output_tokens)},
            stop_reason=stop_reason,
        )


class MockProvider(LLMProvider):
    """Mock provider for testing. Returns canned responses."""

    def __init__(self, responses: list = None):
        self.responses = responses or []
        self.call_history = []
        self._call_index = 0

    def chat(self, messages, tools=None, system="", max_tokens=4096) -> LLMResponse:
        self.call_history.append({"messages": list(messages), "tools": tools, "system": system})
        if self._call_index < len(self.responses):
            resp = self.responses[self._call_index]
            self._call_index += 1
            return resp
        return LLMResponse(
            content="mock response",
            tool_calls=[],
            usage={"input_tokens": 100, "output_tokens": 50},
            stop_reason="end_turn",
        )


class AgentInterface:
    """Manages conversation with Crav (the agent LLM)."""

    def __init__(self, config: dict, provider: LLMProvider, budget_manager, sandbox, tools_registry):
        self.config = config
        self.provider = provider
        self.budget = budget_manager
        self.sandbox = sandbox
        self.tools = tools_registry
        self.conversation: list = []
        self.crav_id: str = "Crav-001"
        self._system_prompt: str = ""

    def start_epoch(self, epoch: int, system_prompt: str):
        """Reset conversation for new epoch."""
        self.conversation = []
        self._system_prompt = system_prompt

    def send_task(self, source_text: str, target_ratio: float) -> dict:
        """Send a compression task to Crav. Returns the tool call results."""
        pulse = self.budget.pulse_string()
        user_msg = f"{pulse}\n\nCompress the following text to {target_ratio:.0%} of its original length.\n\n{source_text}"
        self.conversation.append({"role": "user", "content": user_msg})
        return self._run_turn()

    def send_feedback(self, feedback: dict):
        """Send Judge feedback to Crav after a task."""
        msg = f"Task result: {json.dumps(feedback)}"
        self.conversation.append({"role": "user", "content": msg})

    def request_rnd(self) -> dict:
        """Ask Crav to do R&D (analyze results, update compress.py)."""
        pulse = self.budget.pulse_string()
        msg = f"{pulse}\n\nYou have R&D budget available. Analyze your recent results and improve compress.py if needed."
        self.conversation.append({"role": "user", "content": msg})
        return self._run_turn()

    def _run_turn(self) -> dict:
        """Execute one turn: send to LLM, handle tool calls."""
        max_tokens = max(1, min(4096, self.budget.remaining // 4))

        response = self.provider.chat(
            messages=self.conversation,
            tools=self.tools.get_tool_definitions(),
            system=self._system_prompt,
            max_tokens=max_tokens,
        )

        actual_tokens = response.usage["input_tokens"] + response.usage["output_tokens"]
        alive = self.budget.spend(actual_tokens)

        # Add assistant message to conversation
        self.conversation.append({"role": "assistant", "content": response.content})

        # Handle tool calls and add results to conversation
        tool_results = []
        if response.tool_calls:
            tool_result_contents = []
            for tc in response.tool_calls:
                result = self.tools.execute(tc["name"], tc["arguments"])
                tool_results.append({
                    "tool_call_id": tc["id"],
                    "name": tc["name"],
                    "result": result,
                })
                tool_result_contents.append({
                    "type": "tool_result",
                    "tool_use_id": tc["id"],
                    "content": json.dumps(result),
                })
            self.conversation.append({"role": "user", "content": tool_result_contents})

        return {
            "content": response.content,
            "tool_calls": response.tool_calls,
            "tool_results": tool_results,
            "tokens_spent": actual_tokens,
            "is_oom": not alive,
        }
