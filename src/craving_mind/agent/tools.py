"""Tool definitions available to the agent."""


class ToolsRegistry:
    """Registry of tools available to Crav."""

    def __init__(self, sandbox, memory_manager, budget_manager, smoke_test=None):
        self.sandbox = sandbox
        self.memory = memory_manager
        self.budget = budget_manager
        self.smoke = smoke_test

    def get_tool_definitions(self) -> list:
        """Return Anthropic-format tool definitions."""
        return [
            {
                "name": "run_compress",
                "description": "Run your compress() function on the given text.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "Text to compress"},
                        "target_ratio": {"type": "number", "description": "Target compression ratio (0.0-1.0)"},
                    },
                    "required": ["text", "target_ratio"],
                },
            },
            {
                "name": "read_file",
                "description": "Read a file from your workspace (bible.md, graveyard.md, compress.py).",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "filename": {
                            "type": "string",
                            "enum": ["bible.md", "graveyard.md", "compress.py"],
                        },
                    },
                    "required": ["filename"],
                },
            },
            {
                "name": "write_file",
                "description": "Write content to a file in your workspace.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "filename": {
                            "type": "string",
                            "enum": ["bible.md", "graveyard.md", "compress.py"],
                        },
                        "content": {"type": "string"},
                    },
                    "required": ["filename", "content"],
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
            content = self.memory.read_file(arguments["filename"])
            return {"content": content}

        elif tool_name == "write_file":
            filename = arguments["filename"]
            content = arguments["content"]

            if filename == "compress.py":
                ok, err = self.sandbox.validate_imports(content)
                if not ok:
                    return {
                        "success": False,
                        "error": f"Forbidden import in compress.py: {err}. Only pure Python + whitelist allowed.",
                    }

            self.memory.write_file(filename, content)

            if filename == "compress.py" and self.smoke is not None:
                passed, errors = self.smoke.run(content)
                if not passed:
                    return {"success": True, "smoke_test": "FAILED", "errors": errors}
                return {"success": True, "smoke_test": "PASSED"}

            return {"success": True}

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
