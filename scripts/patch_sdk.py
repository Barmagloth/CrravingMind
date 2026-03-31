#!/usr/bin/env python3
"""Patch claude-code-sdk to fix two bugs.

Bug 1 — rate_limit_event crash:
    The SDK raises MessageParseError on unknown message types, but Claude CLI
    can emit 'rate_limit_event' which is not yet handled by the SDK.

Bug 2 — WinError 206 (command line too long):
    The SDK passes the entire prompt as a CLI argument:
        claude --print -- <entire_prompt>
    On Windows, CreateProcess has a 32 767-character limit on the command
    line.  With conversation history accumulating across tasks the 4th task
    can exceed this limit.  The fix moves the prompt off the command line
    onto stdin instead.

Run after `pip install` or `uv sync` to re-apply the patches:
    python scripts/patch_sdk.py
"""
import inspect

# --- Patch message_parser.py ---

import claude_code_sdk._internal.message_parser as mp

mp_path = inspect.getfile(mp)
with open(mp_path) as f:
    mp_content = f.read()

if "rate_limit_event" in mp_content:
    print(f"message_parser.py already patched: {mp_path}")
else:
    mp_content = mp_content.replace(
        "        case _:\n"
        "            raise MessageParseError(f\"Unknown message type: {message_type}\", data)\n",
        "        case \"rate_limit_event\":\n"
        "            return None  # Skip rate limit events (SDK bug workaround)\n"
        "\n"
        "        case _:\n"
        "            raise MessageParseError(f\"Unknown message type: {message_type}\", data)\n",
    )
    with open(mp_path, "w") as f:
        f.write(mp_content)
    print(f"Patched message_parser.py: {mp_path}")

# --- Patch client.py ---

import claude_code_sdk._internal.client as cl

cl_path = inspect.getfile(cl)
with open(cl_path) as f:
    cl_content = f.read()

if "if result is not None" in cl_content:
    print(f"client.py already patched: {cl_path}")
elif "yield parse_message(data)" in cl_content:
    cl_content = cl_content.replace(
        "                yield parse_message(data)",
        "                result = parse_message(data)\n"
        "                if result is not None:\n"
        "                    yield result",
    )
    with open(cl_path, "w") as f:
        f.write(cl_content)
    print(f"Patched client.py: {cl_path}")
else:
    print("WARNING: client.py pattern not found — manual inspection required.")

# --- Patch subprocess_cli.py (Bug 2: WinError 206) ---

import claude_code_sdk._internal.transport.subprocess_cli as sc  # noqa: E402

sc_path = inspect.getfile(sc)
with open(sc_path) as f:
    sc_content = f.read()

_SC_SENTINEL = "_stdin_prompt = str(self._prompt)"

if _SC_SENTINEL in sc_content:
    print(f"subprocess_cli.py already patched: {sc_path}")
else:
    # 1. _build_command: replace "--print -- prompt" with "--print" + stash
    sc_content = sc_content.replace(
        '            # String mode: use --print with the prompt\n'
        '            cmd.extend(["--print", "--", str(self._prompt)])\n',
        '            # String mode: use --print; prompt sent via stdin to avoid\n'
        '            # WinError 206 (Windows 32 767-char command-line limit).\n'
        '            cmd.extend(["--print"])\n'
        '            self._stdin_prompt = str(self._prompt)\n',
    )
    # 2. connect: write _stdin_prompt to stdin before closing
    sc_content = sc_content.replace(
        '            elif not self._is_streaming and self._process.stdin:\n'
        '                # String mode: close stdin immediately\n'
        '                await self._process.stdin.aclose()\n',
        '            elif not self._is_streaming and self._process.stdin:\n'
        '                # String mode: write prompt via stdin then close\n'
        '                # (avoids WinError 206 — Windows cmd line length limit).\n'
        '                _sp = getattr(self, "_stdin_prompt", None)\n'
        '                if _sp is not None:\n'
        '                    await self._process.stdin.send(_sp.encode("utf-8"))\n'
        '                await self._process.stdin.aclose()\n',
    )
    if _SC_SENTINEL not in sc_content:
        print("WARNING: subprocess_cli.py pattern not found — manual inspection required.")
    else:
        with open(sc_path, "w") as f:
            f.write(sc_content)
        print(f"Patched subprocess_cli.py: {sc_path}")

# --- Verify ---

from claude_code_sdk._internal.message_parser import parse_message  # noqa: E402

result = parse_message({"type": "rate_limit_event"})
assert result is None, f"Expected None, got {result!r}"
print("Verification OK — rate_limit_event returns None.")

from claude_code_sdk._internal.transport.subprocess_cli import SubprocessCLITransport  # noqa: E402
import inspect as _insp

_connect_src = _insp.getsource(SubprocessCLITransport.connect)
assert "_stdin_prompt" in _connect_src or "_sp" in _connect_src, (
    "subprocess_cli.py stdin patch not detected in connect()"
)
print("Verification OK — SubprocessCLITransport.connect uses stdin for prompt.")
