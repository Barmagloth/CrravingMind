"""Agent interface and abstract LLM provider."""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


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
