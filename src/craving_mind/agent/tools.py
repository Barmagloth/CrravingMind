"""Tool definitions available to the agent."""


class ToolsRegistry:
    """Registry of tools available to Crav."""

    def __init__(self, sandbox, memory_manager, budget_manager, smoke_test=None):
        self.sandbox = sandbox
        self.memory = memory_manager
        self.budget = budget_manager
        self.smoke = smoke_test
        self._phase: int = 1

    def get_tool_definitions(self) -> list:
        """Return Anthropic-format tool definitions."""
        # bible.md is only available from Phase 2 onwards.
        if self._phase >= 2:
            rw_files = ["bible.md", "graveyard.md", "compress.py"]
        else:
            rw_files = ["graveyard.md", "compress.py"]

        return [
            {
                "name": "run_compress",
                "description": "Run your compress() function on sample input.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "Input string"},
                        "target_ratio": {"type": "number", "description": "Target ratio (0.0-1.0)"},
                    },
                    "required": ["text", "target_ratio"],
                },
            },
            {
                "name": "read_file",
                "description": "Read a file from your workspace.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "filename": {
                            "type": "string",
                            "enum": rw_files,
                        },
                    },
                    "required": ["filename"],
                },
            },
            {
                "name": "write_file",
                "description": (
                    "Overwrite a file with full content. "
                    "EXPENSIVE — prefer edit_file for small changes to compress.py."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "filename": {
                            "type": "string",
                            "enum": rw_files,
                        },
                        "content": {"type": "string"},
                    },
                    "required": ["filename", "content"],
                },
            },
            {
                "name": "edit_file",
                "description": (
                    "Replace a substring in compress.py. CHEAP — send only the changed lines. "
                    "old_string must match exactly (including whitespace). Max 500 chars."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "old_string": {"type": "string", "description": "Exact text to find in compress.py"},
                        "new_string": {"type": "string", "description": "Replacement text"},
                    },
                    "required": ["old_string", "new_string"],
                },
            },
            {
                "name": "run_script",
                "description": "Run a Python script with access to your workspace files.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "code": {"type": "string", "description": "Python code to execute"},
                    },
                    "required": ["code"],
                },
            },
            {
                "name": "audit_budget",
                "description": "Check your remaining token budget and cost of last operation.",
                "input_schema": {
                    "type": "object",
                    "properties": {},
                },
            },
        ]

    def execute(self, tool_name: str, arguments: dict) -> dict:
        """Execute a tool call and return result."""
        if tool_name == "run_compress":
            compress_code = self.memory.read_file("compress.py")
            result = self.sandbox.run_compress(
                compress_code, arguments["text"], arguments["target_ratio"]
            )
            return {
                "success": result.success,
                "output": result.return_value or "",
                "error": result.error,
            }

        elif tool_name == "read_file":
            filename = arguments["filename"]
            if filename == "bible.md" and self._phase < 2:
                return {"error": "bible.md is not available in Phase 1."}
            content = self.memory.read_file(filename)
            return {"content": content}

        elif tool_name == "write_file":
            filename = arguments["filename"]
            content = arguments["content"]

            if filename == "bible.md" and self._phase < 2:
                return {"error": "bible.md is not available in Phase 1."}

            if filename == "compress.py":
                # 1. Validate imports before writing.
                ok, err = self.sandbox.validate_imports(content)
                if not ok:
                    return {
                        "success": False,
                        "error": f"Forbidden import: {err}. File NOT saved.",
                    }

                # 2. Smoke test BEFORE writing — don't break what works.
                if self.smoke is not None:
                    passed, errors = self.smoke.run(content)
                    if not passed:
                        return {
                            "success": False,
                            "error": f"Smoke test FAILED — file NOT saved. Errors: {'; '.join(errors[:3])}",
                        }

                # 3. All checks passed — safe to write.
                self.memory.write_file(filename, content)
                return {"success": True, "smoke_test": "PASSED"}

            self.memory.write_file(filename, content)
            return {"success": True}

        elif tool_name == "edit_file":
            old_str = arguments.get("old_string", "")
            new_str = arguments.get("new_string", "")

            # Reject oversized old_string — edit is for diffs, not full rewrites.
            _MAX_OLD_STR = 500
            if len(old_str) > _MAX_OLD_STR:
                return {
                    "success": False,
                    "error": (
                        f"old_string too long ({len(old_str)} chars, max {_MAX_OLD_STR}). "
                        "Send only the lines you are changing, not the whole file. "
                        "Use write_file for full rewrites."
                    ),
                }

            current = self.memory.read_file("compress.py")
            if not current:
                return {"success": False, "error": "compress.py does not exist yet. Use write_file."}

            count = current.count(old_str)
            if count == 0:
                return {"success": False, "error": "old_string not found in compress.py. Read it first."}
            if count > 1:
                return {
                    "success": False,
                    "error": f"old_string matches {count} locations — must be unique. Add more context.",
                }

            patched = current.replace(old_str, new_str, 1)

            # Same validation as write_file: imports → smoke → save.
            ok, err = self.sandbox.validate_imports(patched)
            if not ok:
                return {"success": False, "error": f"Forbidden import: {err}. Edit NOT applied."}

            if self.smoke is not None:
                passed, errors = self.smoke.run(patched)
                if not passed:
                    return {
                        "success": False,
                        "error": f"Smoke test FAILED — edit NOT applied. Errors: {'; '.join(errors[:3])}",
                    }

            self.memory.write_file("compress.py", patched)
            return {"success": True, "smoke_test": "PASSED"}

        elif tool_name == "run_script":
            result = self.sandbox.run_script(arguments["code"], self.memory.agent_dir)
            return {"success": result.success, "output": result.output, "error": result.error}

        elif tool_name == "audit_budget":
            return {
                "remaining": self.budget.remaining,
                "last_cost": self.budget.last_step_cost,
                "pulse": self.budget.pulse_string(),
            }

        return {"error": f"Unknown tool: {tool_name}"}
