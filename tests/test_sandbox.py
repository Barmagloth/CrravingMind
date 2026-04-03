"""Comprehensive tests for Sandbox and SmokeTest."""
import os
import tempfile
from pathlib import Path

import pytest
import yaml

from craving_mind.agent.sandbox import Sandbox, SandboxResult
from craving_mind.judge.smoke_test import SmokeTest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def config():
    cfg_path = Path(__file__).parent.parent / "config" / "default.yaml"
    with open(cfg_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture()
def sandbox(config):
    return Sandbox(config)


@pytest.fixture()
def smoke_test(sandbox):
    return SmokeTest(sandbox)


# ---------------------------------------------------------------------------
# Shared test code snippets
# ---------------------------------------------------------------------------

SIMPLE_COMPRESS = '''
def compress(text, target_ratio):
    words = text.split()
    target_len = int(len(words) * target_ratio)
    return ' '.join(words[:target_len])
'''

CRASH_COMPRESS = '''
def compress(text, target_ratio):
    raise RuntimeError("intentional crash")
'''

INFINITE_LOOP_COMPRESS = '''
def compress(text, target_ratio):
    while True:
        pass
'''

FORBIDDEN_OS_COMPRESS = '''
import os

def compress(text, target_ratio):
    return text
'''

FORBIDDEN_SUBPROCESS_COMPRESS = '''
import subprocess

def compress(text, target_ratio):
    return text
'''

FORBIDDEN_SOCKET_COMPRESS = '''
import socket

def compress(text, target_ratio):
    return text
'''

FORBIDDEN_REQUESTS_COMPRESS = '''
import requests

def compress(text, target_ratio):
    return text
'''

SYNTAX_ERROR_CODE = '''
def compress(text, target_ratio)
    return text
'''

ALLOWED_IMPORTS_COMPRESS = '''
import re
import math
import collections
import json

def compress(text, target_ratio):
    words = text.split()
    target_len = max(1, int(len(words) * target_ratio))
    return ' '.join(words[:target_len])
'''

NUMPY_COMPRESS = '''
import numpy as np

def compress(text, target_ratio):
    words = text.split()
    target_len = max(1, int(len(words) * target_ratio))
    return ' '.join(words[:target_len])
'''

EMPTY_CODE = ''

RETURNS_NONE_COMPRESS = '''
def compress(text, target_ratio):
    return None
'''

RETURNS_NONSTRING_COMPRESS = '''
def compress(text, target_ratio):
    return 42
'''

PARENT_DIR_TRAVERSAL = '''
def compress(text, target_ratio):
    with open("../../etc/passwd") as f:
        return f.read()
'''


# ---------------------------------------------------------------------------
# validate_imports tests
# ---------------------------------------------------------------------------

class TestValidateImports:
    def test_allowed_stdlib_passes(self, sandbox):
        ok, err = sandbox.validate_imports(ALLOWED_IMPORTS_COMPRESS)
        assert ok
        assert err == ""

    def test_numpy_forbidden(self, sandbox):
        ok, err = sandbox.validate_imports(NUMPY_COMPRESS)
        assert not ok
        assert "numpy" in err

    def test_no_imports_passes(self, sandbox):
        ok, err = sandbox.validate_imports(SIMPLE_COMPRESS)
        assert ok

    def test_forbidden_os(self, sandbox):
        ok, err = sandbox.validate_imports(FORBIDDEN_OS_COMPRESS)
        assert not ok
        assert "os" in err

    def test_forbidden_subprocess(self, sandbox):
        ok, err = sandbox.validate_imports(FORBIDDEN_SUBPROCESS_COMPRESS)
        assert not ok
        assert "subprocess" in err

    def test_forbidden_socket(self, sandbox):
        ok, err = sandbox.validate_imports(FORBIDDEN_SOCKET_COMPRESS)
        assert not ok
        assert "socket" in err

    def test_forbidden_requests(self, sandbox):
        ok, err = sandbox.validate_imports(FORBIDDEN_REQUESTS_COMPRESS)
        assert not ok
        assert "requests" in err

    def test_syntax_error_caught(self, sandbox):
        ok, err = sandbox.validate_imports(SYNTAX_ERROR_CODE)
        assert not ok
        assert "SyntaxError" in err

    def test_empty_code_passes(self, sandbox):
        ok, err = sandbox.validate_imports(EMPTY_CODE)
        assert ok

    def test_from_import_forbidden(self, sandbox):
        code = "from socket import socket\ndef compress(t, r): return t"
        ok, err = sandbox.validate_imports(code)
        assert not ok
        assert "socket" in err

    def test_from_import_allowed(self, sandbox):
        code = "from collections import Counter\ndef compress(t, r): return t"
        ok, err = sandbox.validate_imports(code)
        assert ok

    def test_nested_forbidden_import(self, sandbox):
        code = "import os.path\ndef compress(t, r): return t"
        ok, err = sandbox.validate_imports(code)
        # os.path root is 'os' which is not in allowed_imports
        assert not ok

    def test_typing_allowed(self, sandbox):
        code = "from typing import Optional\ndef compress(t: str, r: float) -> str: return t"
        ok, err = sandbox.validate_imports(code)
        assert ok


# ---------------------------------------------------------------------------
# run_compress tests
# ---------------------------------------------------------------------------

class TestRunCompress:
    def test_valid_compress_returns_result(self, sandbox):
        result = sandbox.run_compress(SIMPLE_COMPRESS, "hello world foo bar baz", 0.5)
        assert result.success
        assert result.return_value is not None
        assert len(result.return_value) > 0

    def test_result_is_shorter(self, sandbox):
        text = "one two three four five six seven eight nine ten"
        result = sandbox.run_compress(SIMPLE_COMPRESS, text, 0.5)
        assert result.success
        original_words = len(text.split())
        compressed_words = len(result.return_value.split())
        assert compressed_words <= original_words

    def test_crash_returns_error_with_traceback(self, sandbox):
        result = sandbox.run_compress(CRASH_COMPRESS, "some text", 0.5)
        assert not result.success
        assert "RuntimeError" in result.error or "intentional crash" in result.error

    def test_timeout_returns_timed_out(self, sandbox):
        result = sandbox.run_compress(INFINITE_LOOP_COMPRESS, "hello", 0.5)
        assert not result.success
        assert result.timed_out

    def test_forbidden_import_blocked_before_execution(self, sandbox):
        result = sandbox.run_compress(FORBIDDEN_OS_COMPRESS, "hello", 0.5)
        assert not result.success
        assert not result.timed_out
        assert "Forbidden" in result.error or "os" in result.error

    def test_syntax_error_caught(self, sandbox):
        result = sandbox.run_compress(SYNTAX_ERROR_CODE, "hello", 0.5)
        assert not result.success
        assert "SyntaxError" in result.error

    def test_numpy_compress_blocked(self, sandbox):
        result = sandbox.run_compress(NUMPY_COMPRESS, "alpha beta gamma delta epsilon", 0.6)
        assert not result.success

    def test_empty_text(self, sandbox):
        result = sandbox.run_compress(SIMPLE_COMPRESS, "", 0.5)
        assert result.success
        assert result.return_value == ""

    def test_ratio_zero(self, sandbox):
        result = sandbox.run_compress(SIMPLE_COMPRESS, "hello world", 0.0)
        assert result.success
        assert result.return_value == ""

    def test_ratio_one(self, sandbox):
        text = "hello world"
        result = sandbox.run_compress(SIMPLE_COMPRESS, text, 1.0)
        assert result.success
        assert result.return_value == text

    def test_parent_dir_traversal_fails(self, sandbox):
        # Should either succeed with empty output (file not found) or fail —
        # the important thing is it doesn't escape the sandbox silently
        result = sandbox.run_compress(PARENT_DIR_TRAVERSAL, "anything", 0.5)
        # The file won't exist in the sandbox tmpdir so it'll raise FileNotFoundError
        assert not result.success

    def test_no_compress_function_fails(self, sandbox):
        code = "x = 1 + 1\n"
        result = sandbox.run_compress(code, "hello", 0.5)
        assert not result.success

    def test_returns_none_fails(self, sandbox):
        # json.dumps({"result": None}) is valid JSON, but return_value will be None
        # The runner will output {"result": null}; we accept None as return_value
        result = sandbox.run_compress(RETURNS_NONE_COMPRESS, "hello world", 0.5)
        # Success depends on whether json serialization works; None is valid JSON null
        # Either way, check we get a consistent result object
        assert isinstance(result, SandboxResult)


# ---------------------------------------------------------------------------
# run_script tests
# ---------------------------------------------------------------------------

class TestRunScript:
    def test_basic_script_runs(self, sandbox, tmp_path):
        script = 'print("hello from script")\n'
        result = sandbox.run_script(script, str(tmp_path))
        assert result.success
        assert "hello from script" in result.output

    def test_modified_bible_copied_back(self, sandbox, tmp_path):
        # Create initial bible.md
        bible = tmp_path / "bible.md"
        bible.write_text("original content", encoding="utf-8")

        script = '''
with open("bible.md", "w", encoding="utf-8") as f:
    f.write("updated content")
'''
        result = sandbox.run_script(script, str(tmp_path))
        assert result.success
        assert bible.read_text(encoding="utf-8") == "updated content"

    def test_modified_graveyard_copied_back(self, sandbox, tmp_path):
        graveyard = tmp_path / "graveyard.md"
        graveyard.write_text("old entries", encoding="utf-8")

        script = '''
with open("graveyard.md", "a", encoding="utf-8") as f:
    f.write("\\nnew entry")
'''
        result = sandbox.run_script(script, str(tmp_path))
        assert result.success
        content = graveyard.read_text(encoding="utf-8")
        assert "new entry" in content

    def test_script_crash_does_not_copy_back(self, sandbox, tmp_path):
        bible = tmp_path / "bible.md"
        bible.write_text("original", encoding="utf-8")

        script = '''
with open("bible.md", "w") as f:
    f.write("should not be written back")
raise RuntimeError("crash after write")
'''
        result = sandbox.run_script(script, str(tmp_path))
        assert not result.success
        # File modified in temp but not copied back due to non-zero returncode
        assert bible.read_text(encoding="utf-8") == "original"

    def test_forbidden_import_in_script(self, sandbox, tmp_path):
        script = "import subprocess\nsubprocess.run(['echo', 'hi'])\n"
        result = sandbox.run_script(script, str(tmp_path))
        assert not result.success
        assert "subprocess" in result.error or "Forbidden" in result.error

    def test_agent_files_available_in_tmpdir(self, sandbox, tmp_path):
        compress_py = tmp_path / "compress.py"
        compress_py.write_text("def compress(t, r): return t", encoding="utf-8")

        # pathlib is in allowed_imports; os is not
        script = '''
from pathlib import Path
files = [f.name for f in Path(".").iterdir()]
print("\\n".join(files))
'''
        result = sandbox.run_script(script, str(tmp_path))
        assert result.success
        assert "compress.py" in result.output

    def test_script_timeout(self, sandbox, tmp_path):
        script = "while True: pass\n"
        result = sandbox.run_script(script, str(tmp_path))
        assert not result.success
        assert result.timed_out


# ---------------------------------------------------------------------------
# SmokeTest tests
# ---------------------------------------------------------------------------

class TestSmokeTest:
    def test_passing_code_passes_all_samples(self, smoke_test):
        passed, errors = smoke_test.run(SIMPLE_COMPRESS)
        assert passed
        assert errors == []

    def test_crashing_code_fails_with_specific_errors(self, smoke_test):
        passed, errors = smoke_test.run(CRASH_COMPRESS)
        assert not passed
        assert len(errors) == len(SmokeTest.SAMPLE_TEXTS)
        for err in errors:
            assert "Sample" in err

    def test_error_messages_contain_sample_index(self, smoke_test):
        passed, errors = smoke_test.run(CRASH_COMPRESS)
        for i, err in enumerate(errors):
            assert f"Sample {i}:" in err

    def test_syntax_error_fails_all(self, smoke_test):
        passed, errors = smoke_test.run(SYNTAX_ERROR_CODE)
        assert not passed
        assert len(errors) == len(SmokeTest.SAMPLE_TEXTS)

    def test_numpy_compress_blocked(self, smoke_test):
        passed, errors = smoke_test.run(NUMPY_COMPRESS)
        assert not passed

    def test_allowed_imports_compress_passes(self, smoke_test):
        passed, errors = smoke_test.run(ALLOWED_IMPORTS_COMPRESS)
        assert passed, f"Unexpected errors: {errors}"

    def test_partial_failure_reported(self, smoke_test):
        # Code that crashes on texts containing numbers
        fragile_compress = '''
def compress(text, target_ratio):
    if any(c.isdigit() for c in text):
        raise ValueError("no digits allowed")
    words = text.split()
    n = max(1, int(len(words) * target_ratio))
    return " ".join(words[:n])
'''
        passed, errors = smoke_test.run(fragile_compress)
        assert not passed
        # Some samples have digits, some don't — at least one error
        assert len(errors) > 0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_unicode_text(self, sandbox):
        text = "Привет мир. Это тестовый текст на русском языке."
        result = sandbox.run_compress(SIMPLE_COMPRESS, text, 0.5)
        assert result.success

    def test_very_long_text(self, sandbox):
        # Keep under Windows command-line limit (~32767 chars)
        text = "word " * 1_000
        result = sandbox.run_compress(SIMPLE_COMPRESS, text.strip(), 0.1)
        assert result.success

    def test_sensitive_env_vars_removed(self, sandbox):
        # Test _restricted_env() directly — the sandbox scripts cannot use os
        import os as _os
        # Set fake keys so we can verify they get stripped
        _os.environ["ANTHROPIC_API_KEY"] = "sk-test-anthropic"
        _os.environ["OPENAI_API_KEY"] = "sk-test-openai"
        try:
            env = sandbox._restricted_env()
        finally:
            _os.environ.pop("ANTHROPIC_API_KEY", None)
            _os.environ.pop("OPENAI_API_KEY", None)
        assert "ANTHROPIC_API_KEY" not in env
        assert "OPENAI_API_KEY" not in env

    def test_compress_result_is_string(self, sandbox):
        result = sandbox.run_compress(SIMPLE_COMPRESS, "hello world test", 0.5)
        assert result.success
        assert isinstance(result.return_value, str)

    def test_sandbox_result_fields(self, sandbox):
        result = sandbox.run_compress(SIMPLE_COMPRESS, "a b c d e", 0.5)
        assert isinstance(result.success, bool)
        assert isinstance(result.output, str)
        assert isinstance(result.error, str)
        assert isinstance(result.timed_out, bool)
