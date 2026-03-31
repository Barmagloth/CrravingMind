#!/usr/bin/env python3
"""Patch claude-code-sdk to handle rate_limit_event messages.

The SDK raises MessageParseError on unknown message types, but Claude CLI
can emit 'rate_limit_event' which is not yet handled by the SDK.

Run after `pip install` or `uv sync` to re-apply the patch:
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

# --- Verify ---

from claude_code_sdk._internal.message_parser import parse_message  # noqa: E402

result = parse_message({"type": "rate_limit_event"})
assert result is None, f"Expected None, got {result!r}"
print("Verification OK — rate_limit_event returns None.")
