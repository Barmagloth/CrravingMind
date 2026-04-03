"""Agent interface and abstract LLM provider."""

import asyncio
import json
import logging
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

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


def _patch_sdk_parser() -> None:
    """Monkey-patch claude-code-sdk to skip unknown message types instead of raising.

    SDK 0.0.25 raises MessageParseError on 'rate_limit_event' messages (and any
    other future types it doesn't recognise). Patching parse_message to return
    None for unknowns lets us skip them gracefully rather than retrying the whole
    query. The async-for loop in CLIProvider._run() already skips None messages.
    """
    try:
        from claude_code_sdk._internal import message_parser
        from claude_code_sdk._errors import MessageParseError

        _original_parse = message_parser.parse_message

        def _patched_parse(data):  # type: ignore[no-untyped-def]
            try:
                return _original_parse(data)
            except MessageParseError as exc:
                msg_type = data.get("type", "<unknown>") if isinstance(data, dict) else "<non-dict>"
                logger.debug(
                    "SDK parse_message skipped unrecognised message type %r: %s",
                    msg_type, exc,
                )
                return None

        message_parser.parse_message = _patched_parse
        logger.debug("claude-code-sdk message_parser patched to skip unknown types")
    except Exception:  # pragma: no cover — only fails if SDK internals change
        logger.debug("Could not patch claude-code-sdk message_parser", exc_info=True)


_patch_sdk_parser()

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
                '{"content": "<brief note>", '
                '"tool_calls": [{"name": "<tool_name>", "arguments": {<args>}}]}\n'
                "If you do NOT need to call a tool, respond with the same JSON schema "
                'but with an empty tool_calls array: {"content": "<your reply>", "tool_calls": []}\n\n'
                "CRITICAL: Keep your ENTIRE JSON response under 800 characters. "
                "For edit_file, use the SHORTEST unique old_string (1-3 lines, not the whole function). "
                "Keep content field to 1-2 sentences max.\n\n"
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

    def _trim_conversation(self, messages: list, max_chars: int = 20000) -> list:
        """Keep first message + recent messages so total stays within max_chars."""
        total = sum(len(str(m.get("content", ""))) for m in messages)
        if total <= max_chars:
            return messages
        trimmed = [messages[0]] if messages else []
        remaining = max_chars - len(str(messages[0].get("content", ""))) if messages else max_chars
        for msg in reversed(messages[1:]):
            msg_len = len(str(msg.get("content", "")))
            if remaining - msg_len > 0:
                trimmed.insert(1, msg)
                remaining -= msg_len
            else:
                break
        return trimmed

    def _parse_response(self, raw_text: str) -> tuple[str, list]:
        """Extract (content, tool_calls) from the model's response text.

        Handles three failure modes of the CLI provider:
        1. Clean single JSON object (happy path).
        2. Multi-turn roleplay: model generates [USER]/[ASSISTANT] continuations
           after its first JSON — we extract only the first JSON object.
        3. Truncated JSON: model ran out of tokens mid-output — we attempt
           partial recovery but return (raw_text, []) if hopeless.

        Returns (raw_text, []) if JSON cannot be parsed, so the caller still
        gets something useful rather than crashing.
        """
        text = raw_text.strip()

        # Strip markdown code fences if present.
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()

        # Try the happy path first: entire text is valid JSON.
        data = self._try_parse_json(text)

        # Retry with repaired escapes (\s → \\s etc.) before expensive
        # brace-counting extraction.
        if data is None and text.startswith("{"):
            data = self._try_parse_json(self._repair_json_escapes(text))

        if data is None:
            # Multi-turn roleplay: model continued past its first JSON object.
            # Find the first top-level {...} by brace-counting.
            data = self._extract_first_json_object(text)

        if data is None:
            preview = raw_text[:300].replace('\n', '\\n')
            logger.warning(
                "CLIProvider: could not parse any JSON from response "
                "(%d chars): %.300s",
                len(raw_text), preview,
            )
            return raw_text, []

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

    @staticmethod
    def _try_parse_json(text: str) -> dict | None:
        """Try json.loads; return dict or None.

        Uses strict=False to accept literal newlines inside JSON strings —
        models often output code with real line breaks instead of \\n escapes.
        """
        try:
            data = json.loads(text, strict=False)
            if isinstance(data, dict):
                return data
        except (json.JSONDecodeError, ValueError):
            pass
        return None

    @staticmethod
    def _repair_json_escapes(text: str) -> str:
        """Fix invalid JSON escape sequences produced by the model.

        Models embedding code strings (regex patterns like \\s+, \\w+, \\d+)
        often write them as \\s, \\w, \\d — which are not valid JSON escapes.
        This doubles the backslash so json.loads accepts them.
        """
        _VALID_JSON_ESCAPES = frozenset('"\\/bfnrtu')
        result: list[str] = []
        i = 0
        while i < len(text):
            if text[i] == '\\' and i + 1 < len(text):
                next_ch = text[i + 1]
                if next_ch in _VALID_JSON_ESCAPES:
                    result.append(text[i : i + 2])
                else:
                    # Invalid escape like \s → turn into \\s
                    result.append('\\\\')
                    result.append(next_ch)
                i += 2
            else:
                result.append(text[i])
                i += 1
        return ''.join(result)

    @staticmethod
    def _extract_first_json_object(text: str) -> dict | None:
        """Extract the first balanced {...} from text and parse it.

        Uses brace-counting with awareness of JSON string literals
        (skips braces inside quoted strings).  If json.loads fails on
        the extracted candidate, retries after repairing common invalid
        escape sequences (\\s, \\w, etc. from code embedded in strings).
        """
        start = text.find("{")
        if start == -1:
            return None

        depth = 0
        in_string = False
        escape = False

        for i in range(start, len(text)):
            ch = text[i]

            if escape:
                escape = False
                continue

            if ch == "\\":
                if in_string:
                    escape = True
                continue

            if ch == '"':
                in_string = not in_string
                continue

            if in_string:
                continue

            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : i + 1]
                    try:
                        data = json.loads(candidate, strict=False)
                        if isinstance(data, dict):
                            return data
                    except (json.JSONDecodeError, ValueError):
                        pass
                    # Retry with repaired escape sequences.
                    try:
                        repaired = CLIProvider._repair_json_escapes(candidate)
                        data = json.loads(repaired, strict=False)
                        if isinstance(data, dict):
                            logger.debug(
                                "CLIProvider: JSON parsed after escape repair"
                            )
                            return data
                    except (json.JSONDecodeError, ValueError) as exc:
                        logger.warning(
                            "CLIProvider: balanced JSON found but unparseable "
                            "even after repair (%d chars): %s — first 200: %.200s",
                            len(candidate), exc,
                            candidate[:200].replace('\n', '\\n'),
                        )
                    return None

        # Braces never balanced — either genuinely truncated, or the brace
        # tracker lost sync (e.g. unescaped quotes in code strings).
        # Fallback: try json.loads + repair on the entire text from '{'.
        if start < len(text):
            tail = text[start:]
            for attempt_text in (tail, CLIProvider._repair_json_escapes(tail)):
                try:
                    data = json.loads(attempt_text, strict=False)
                    if isinstance(data, dict):
                        logger.debug(
                            "CLIProvider: unbalanced braces but json.loads "
                            "succeeded on tail (%d chars)", len(tail),
                        )
                        return data
                except (json.JSONDecodeError, ValueError):
                    pass
            logger.debug(
                "CLIProvider: braces never balanced, depth=%d at end "
                "(%d chars from '{')", depth, len(tail),
            )
        return None

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

        messages = self._trim_conversation(messages)

        # On resumed sessions the model already has system prompt and tool
        # definitions from the first call.  Re-sending them wastes ~900
        # tokens per call (system ~500 + tools ~400).  Only include them
        # on the very first call (no session_id yet).
        is_resumed = self._session_id is not None
        if is_resumed:
            prompt = self._build_prompt(messages, tools=None, system="")
        else:
            prompt = self._build_prompt(messages, tools, system)

        logger.info(
            "CLIProvider.chat: prompt_len=%d max_tokens=%d session_id=%s",
            len(prompt), max_tokens, self._session_id,
        )

        # Remove CLAUDECODE so we can run nested from within a Claude Code session.
        env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}
        options = _SdkOptions(
            model=self.model,
            allowed_tools=[],         # no filesystem tools — text-only
            permission_mode="bypassPermissions",
            env=env,
            resume=self._session_id,
        )

        MAX_RETRIES = 5
        INITIAL_BACKOFF = 2.0

        collected_text: list[str] = []
        usage_data: dict = {}

        async def _run() -> None:
            nonlocal usage_data
            for attempt in range(MAX_RETRIES):
                collected_text.clear()
                usage_data.clear()
                try:
                    logger.debug(
                        "CLIProvider: attempt %d/%d, prompt_len=%d",
                        attempt + 1, MAX_RETRIES, len(prompt),
                    )
                    async for msg in _query(prompt=prompt, options=options):
                        if msg is None:
                            continue  # patched-out unknown message type (e.g. rate_limit_event)
                        if _AssistantMessage and isinstance(msg, _AssistantMessage):
                            for block in msg.content:
                                if _TextBlock and isinstance(block, _TextBlock):
                                    collected_text.append(block.text)
                                    logger.debug(
                                        "CLIProvider: text chunk %d chars", len(block.text)
                                    )
                        elif _ResultMessage and isinstance(msg, _ResultMessage):
                            if msg.usage:
                                usage_data.update(msg.usage if isinstance(msg.usage, dict)
                                                  else vars(msg.usage))
                            if msg.session_id:
                                self._session_id = msg.session_id
                    total_chars = sum(len(t) for t in collected_text)
                    logger.info(
                        "CLIProvider: response received, total_chars=%d (~%d tokens)",
                        total_chars,
                        max(1, total_chars // 4),
                    )
                    if total_chars == 0 and attempt < MAX_RETRIES - 1:
                        logger.warning(
                            "CLIProvider: empty response on attempt %d/%d — retrying",
                            attempt + 1, MAX_RETRIES,
                        )
                        # Drop session so next attempt starts fresh.
                        self._session_id = None
                        wait = INITIAL_BACKOFF * (2 ** attempt)
                        await asyncio.sleep(wait)
                        continue
                    return  # success — exit retry loop
                except Exception as e:
                    error_str = str(e).lower()
                    if "rate_limit" in error_str or "unknown message type" in error_str:
                        wait = INITIAL_BACKOFF * (2 ** attempt)
                        logger.warning(
                            "CLIProvider: rate-limited (attempt %d/%d), waiting %.1fs: %s",
                            attempt + 1, MAX_RETRIES, wait, e,
                        )
                        await asyncio.sleep(wait)
                        if attempt == MAX_RETRIES - 1:
                            logger.error(
                                "CLIProvider: all %d retries exhausted: %s", MAX_RETRIES, e
                            )
                            raise
                    else:
                        logger.error("CLIProvider: non-retryable error: %s", e)
                        raise

        asyncio.run(_run())

        raw_text = "".join(collected_text)
        content, tool_calls = self._parse_response(raw_text)
        logger.debug(
            "CLIProvider: parsed content_len=%d tool_calls=%d",
            len(content), len(tool_calls),
        )

        # Estimate token usage from text length if SDK didn't report it.
        # The CLI SDK rarely reports input_tokens, so we estimate from the
        # prompt we built.  For resumed sessions, the prompt only contains
        # the new user message (tools/system omitted), which underestimates
        # the real context but is the best local approximation we have.
        if usage_data.get("input_tokens"):
            input_tokens = usage_data["input_tokens"]
        else:
            input_tokens = max(1, len(prompt) // 4)
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
        """Reset conversation for new epoch.

        Drops the CLI session so the next call starts fresh — the new agent
        gets the updated system prompt, tool definitions, and graveyard
        instead of inheriting behavioural ruts from the previous epoch.
        """
        self.conversation = []
        self._system_prompt = system_prompt
        if hasattr(self.provider, "new_session"):
            self.provider.new_session()

    def send_task(self, source_text: str, target_ratio: float) -> dict:
        """Send a compression task to Crav. Returns the tool call results.

        DEPRECATED — prefer the auto-compress path in EpochRunner which
        runs compress.py directly and sends only metrics via send_metrics().
        Kept for backward compatibility with tests.
        """
        pulse = self.budget.pulse_string()
        user_msg = f"{pulse}\n\nCompress the following text to {target_ratio:.0%} of its original length.\n\n{source_text}"
        self.conversation.append({"role": "user", "content": user_msg})
        return self._run_turn()

    def send_feedback(self, feedback: dict):
        """Append Judge feedback to conversation (no LLM call)."""
        msg = f"Task result: {json.dumps(feedback)}"
        self.conversation.append({"role": "user", "content": msg})

    def send_metrics(self, task_idx: int, tasks_total: int, feedback: dict) -> dict:
        """Send task metrics to Crav and let it react (improve compress.py).

        The agent never sees the source text — only numerical feedback.
        It gets a turn to read/write files and run scripts to improve
        compress.py based on the metrics.

        Conversation is trimmed to the last assistant summary before each
        new task to prevent unbounded prompt growth across tasks.
        """
        # Trim conversation: keep only the last assistant message (if any)
        # so the agent has minimal context. It can always read_file to
        # refresh its knowledge. This prevents prompt_len from growing
        # 4k→7k+ across tasks, eating the entire budget.
        self._trim_to_last_summary()

        pulse = self.budget.pulse_string()
        msg = (
            f"{pulse}\n\n"
            f"Task {task_idx}/{tasks_total} result: "
            f"ratio={feedback.get('compression_ratio', 0):.2f} "
            f"a={feedback.get('semantic_score', 0):.2f} "
            f"b={feedback.get('entity_score', 0):.2f} "
            f"{'PASS' if feedback.get('pass') else 'FAIL'}"
        )
        self.conversation.append({"role": "user", "content": msg})
        return self._run_turn()

    def _trim_to_last_summary(self) -> None:
        """Keep only the last assistant message in conversation.

        Between tasks, the conversation accumulates tool results (read_file
        echoes full compress.py ~5000 chars) and multiple round-trips.
        By task 6, prompt_len can hit 7600+ tokens — most of the budget.

        Trimming to just the last assistant summary gives the agent a
        one-line reminder of what happened, while keeping prompt small.
        """
        if len(self.conversation) <= 2:
            return

        # Find last assistant message.
        last_assistant = None
        for msg in reversed(self.conversation):
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                # Truncate to 200 chars — just a summary, not full tool output.
                last_assistant = {"role": "assistant", "content": content[:200]}
                break

        if last_assistant:
            self.conversation = [last_assistant]
        else:
            self.conversation = []

    def request_rnd(self) -> dict:
        """Ask Crav to do R&D (analyze results, update compress.py)."""
        pulse = self.budget.pulse_string()
        msg = f"{pulse}\n\nYou have R&D budget available. Analyze your recent results and improve compress.py if needed."
        self.conversation.append({"role": "user", "content": msg})
        return self._run_turn()

    def request_last_words(self) -> str:
        """Ask the dying agent for a 1-line epitaph: what it tried, why it failed.

        Uses a single LLM call with max_tokens capped low.
        Free — does NOT consume agent budget (system-level operation).

        The model already remembers the entire epoch via CLI session resume —
        just send the bare request as the next conversation turn.
        """
        msg = (
            "Epoch over. Summarize for the graveyard in 2-3 SHORT sentences (max 300 chars total): "
            "what strategy you used, what you changed in compress.py, best scores achieved, "
            "and why it wasn't enough. Be specific (mention function names, thresholds, scores). "
            "No tools, just text. No JSON wrapper."
        )
        self.conversation.append({"role": "user", "content": msg})

        response = self.provider.chat(
            messages=self.conversation,
            tools=[],
            system="",
            max_tokens=200,
        )
        self.conversation.append({"role": "assistant", "content": response.content})

        # Return just the text, stripped of any JSON wrapper.
        text = response.content.strip()
        # CLI provider may still wrap in JSON.
        if text.startswith("{"):
            try:
                import json as _json
                data = _json.loads(text)
                text = data.get("content", text)
            except (ValueError, AttributeError):
                pass
        return self._truncate_at_word_boundary(text, 300)

    @staticmethod
    def _truncate_at_word_boundary(text: str, limit: int) -> str:
        """Truncate text at a word boundary, ending with '…' if shortened."""
        if len(text) <= limit:
            return text
        # Find last space before limit.
        cut = text.rfind(" ", 0, limit)
        if cut <= 0:
            cut = limit
        return text[:cut].rstrip(".,;:—- ") + "…"

    # Maximum LLM round-trips per _run_turn call (read → fix → compress → done).
    _MAX_TOOL_ROUNDS = 3
    # Maximum fraction of remaining budget a single _run_turn may consume.
    _MAX_TURN_BUDGET_FRACTION = 0.20

    def _run_turn(self) -> dict:
        """Execute one turn with a tool-use loop.

        The agent may need several LLM round-trips to accomplish a task
        (e.g. read_file → write_file → run_compress).  We loop until the
        model stops requesting tools, the budget runs out, we hit
        _MAX_TOOL_ROUNDS, or this turn consumed >20% of remaining budget.
        """
        all_tool_calls: list = []
        all_tool_results: list = []
        total_tokens = 0
        final_content = ""
        alive = True
        turn_budget_cap = int(self.budget.remaining * self._MAX_TURN_BUDGET_FRACTION)

        for _round in range(self._MAX_TOOL_ROUNDS):
            max_tokens = max(1, min(4096, self.budget.remaining // 4))

            response = self.provider.chat(
                messages=self.conversation,
                tools=self.tools.get_tool_definitions(),
                system=self._system_prompt,
                max_tokens=max_tokens,
            )

            step_tokens = response.usage["input_tokens"] + response.usage["output_tokens"]
            total_tokens += step_tokens
            alive = self.budget.spend(step_tokens)
            final_content = response.content

            # Add assistant message to conversation.
            self.conversation.append({"role": "assistant", "content": response.content})

            if not response.tool_calls:
                # Model finished — no more tools to call.
                break

            # Execute tool calls, feed results back into conversation.
            all_tool_calls.extend(response.tool_calls)
            tool_result_contents = []
            for tc in response.tool_calls:
                result = self.tools.execute(tc["name"], tc["arguments"])
                all_tool_results.append({
                    "tool_call_id": tc["id"],
                    "name": tc["name"],
                    "result": result,
                })
                # Truncate large tool results to save context window.
                # read_file returns full file content (~5000 chars for compress.py)
                # which bloats the conversation across tasks.
                result_str = json.dumps(result)
                if len(result_str) > 3000:
                    result_str = result_str[:3000] + '..."}'
                tool_result_contents.append({
                    "type": "tool_result",
                    "tool_use_id": tc["id"],
                    "content": result_str,
                })
            self.conversation.append({"role": "user", "content": tool_result_contents})

            # Free reads: if the only tool call this round was reading
            # graveyard.md, refund the tokens — agent shouldn't pay to
            # learn from past failures.
            _FREE_READS = {"graveyard.md"}
            is_free_round = all(
                tc["name"] == "read_file"
                and tc["arguments"].get("filename") in _FREE_READS
                for tc in response.tool_calls
            )
            if is_free_round:
                self.budget.refund(step_tokens)
                total_tokens -= step_tokens
                logger.debug("Free read round — refunded %d tokens", step_tokens)

            if not alive:
                # Budget exhausted — stop looping.
                break

            # Per-turn circuit breaker: stop if this turn already consumed
            # a large fraction of the budget (prevents runaway tool loops).
            if total_tokens > turn_budget_cap:
                logger.info(
                    "Turn budget cap reached (%d > %d) — stopping tool loop",
                    total_tokens, turn_budget_cap,
                )
                break

        return {
            "content": final_content,
            "tool_calls": all_tool_calls,
            "tool_results": all_tool_results,
            "tokens_spent": total_tokens,
            "is_oom": not alive,
        }
