import ast
import os
import shutil
import subprocess
import sys
import tempfile
import json
from pathlib import Path
from typing import Optional
from dataclasses import dataclass


@dataclass
class SandboxResult:
    success: bool
    output: str  # stdout
    error: str   # stderr or exception message
    return_value: Optional[str] = None  # captured return value if any
    timed_out: bool = False


class Sandbox:
    """Isolated execution environment for agent code (compress.py, DIY scripts)."""

    def __init__(self, config: dict):
        self.timeout = config['sandbox']['timeout_seconds']
        self.allowed_imports = set(config['sandbox']['allowed_imports'])
        # Always allow stdlib modules that are commonly needed
        self._stdlib_allowed = {
            'builtins', 'abc', 'copy', 'typing', 'dataclasses',
            'os.path',  # but NOT os itself unless in allowed
        }

    def validate_imports(self, code: str) -> tuple[bool, str]:
        """Check AST for forbidden imports. Returns (ok, error_message)."""
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return False, f"SyntaxError: {e}"

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split('.')[0]
                    if root not in self.allowed_imports and root not in self._stdlib_allowed:
                        return False, f"Forbidden import: {alias.name}"
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    root = node.module.split('.')[0]
                    if root not in self.allowed_imports and root not in self._stdlib_allowed:
                        return False, f"Forbidden import: from {node.module}"

        return True, ""

    def run_compress(self, compress_code: str, text: str, target_ratio: float) -> SandboxResult:
        """Run compress(text, target_ratio) from the given code in isolation."""

        # Validate imports first
        ok, err = self.validate_imports(compress_code)
        if not ok:
            return SandboxResult(success=False, output="", error=err)

        # Create temp directory
        with tempfile.TemporaryDirectory(prefix="craving_sandbox_") as tmpdir:
            # Write compress.py
            compress_path = Path(tmpdir) / "compress.py"
            compress_path.write_text(compress_code, encoding='utf-8')

            # Write runner script that imports and calls compress()
            runner_code = '''
import sys
import json
sys.path.insert(0, sys.argv[1])
from compress import compress

text = json.loads(sys.argv[2])
ratio = float(sys.argv[3])

result = compress(text, ratio)
# Output as JSON for safe parsing
print(json.dumps({"result": result}))
'''
            runner_path = Path(tmpdir) / "_runner.py"
            runner_path.write_text(runner_code, encoding='utf-8')

            # Run in subprocess
            try:
                proc = subprocess.run(
                    [sys.executable, str(runner_path), tmpdir, json.dumps(text), str(target_ratio)],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    cwd=tmpdir,
                    env=self._restricted_env()
                )

                if proc.returncode == 0:
                    try:
                        output = json.loads(proc.stdout.strip())
                        return SandboxResult(
                            success=True,
                            output=proc.stdout,
                            error=proc.stderr,
                            return_value=output.get("result", "")
                        )
                    except json.JSONDecodeError:
                        return SandboxResult(
                            success=False,
                            output=proc.stdout,
                            error="Failed to parse compress() output as JSON"
                        )
                else:
                    return SandboxResult(
                        success=False,
                        output=proc.stdout,
                        error=proc.stderr
                    )

            except subprocess.TimeoutExpired:
                return SandboxResult(
                    success=False,
                    output="",
                    error=f"Execution timed out after {self.timeout}s",
                    timed_out=True
                )

    def run_script(self, script_code: str, agent_dir: str) -> SandboxResult:
        """Run a DIY script with access to agent files (bible.md, graveyard.md)."""

        ok, err = self.validate_imports(script_code)
        if not ok:
            return SandboxResult(success=False, output="", error=err)

        with tempfile.TemporaryDirectory(prefix="craving_diy_") as tmpdir:
            # Copy agent files to tmpdir
            agent_path = Path(agent_dir)
            for f in ['bible.md', 'graveyard.md', 'compress.py']:
                src = agent_path / f
                if src.exists():
                    shutil.copy2(src, Path(tmpdir) / f)

            # Write script
            script_path = Path(tmpdir) / "_diy_script.py"
            script_path.write_text(script_code, encoding='utf-8')

            try:
                proc = subprocess.run(
                    [sys.executable, str(script_path)],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    cwd=tmpdir,
                    env=self._restricted_env()
                )

                # Copy modified files back
                if proc.returncode == 0:
                    for f in ['bible.md', 'graveyard.md']:
                        modified = Path(tmpdir) / f
                        if modified.exists():
                            shutil.copy2(modified, agent_path / f)

                return SandboxResult(
                    success=proc.returncode == 0,
                    output=proc.stdout,
                    error=proc.stderr
                )

            except subprocess.TimeoutExpired:
                return SandboxResult(
                    success=False,
                    output="",
                    error=f"DIY script timed out after {self.timeout}s",
                    timed_out=True
                )

    def _restricted_env(self) -> dict:
        """Create a restricted environment for subprocess."""
        env = os.environ.copy()
        # Remove sensitive vars
        for key in ['ANTHROPIC_API_KEY', 'OPENAI_API_KEY', 'HOME', 'USERPROFILE']:
            env.pop(key, None)
        # Disable network (best effort — real isolation would need containers)
        env['no_proxy'] = '*'
        env['HTTP_PROXY'] = 'http://0.0.0.0:0'
        env['HTTPS_PROXY'] = 'http://0.0.0.0:0'
        return env
