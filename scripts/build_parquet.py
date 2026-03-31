#!/usr/bin/env python
"""Build frozen benchmark Parquet from source texts.

Tries (in order):
  1. Anthropic SDK  (ANTHROPIC_API_KEY env var)
  2. claude CLI subprocess  (not usable inside another Claude Code session)
  3. MockBenchmarkGenerator  (deterministic fallback, always works)

Saves progress to data/benchmarks/build_checkpoint.json so interrupted
runs can be resumed.

Usage:
    python scripts/build_parquet.py
    python scripts/build_parquet.py --output data/benchmarks/benchmark_v1.parquet
    python scripts/build_parquet.py --mock   # force mock mode
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCES_DIR = PROJECT_ROOT / "data" / "sources"
BENCHMARKS_DIR = PROJECT_ROOT / "data" / "benchmarks"
CHECKPOINT_PATH = BENCHMARKS_DIR / "build_checkpoint.json"
DEFAULT_OUTPUT = BENCHMARKS_DIR / "benchmark_v1.parquet"

# ---------------------------------------------------------------------------
# LLM prompt templates (matching generator.py conventions)
# ---------------------------------------------------------------------------
_QUESTION_SYSTEM = (
    "You are a rigorous QA engineer. Generate exactly 10 questions about the given text."
    " Cover: general meaning, specific numbers, names, conditions, and logical connections."
    " Return a JSON array of 10 question strings, nothing else."
)

_ANSWER_SYSTEM = (
    "You are a precise reading comprehension assistant."
    " Answer each question using only the provided context."
    " Return a JSON array of 10 answer strings matching the question order, nothing else."
)

_COMBINED_PROMPT = """\
Given this text, generate exactly 10 questions and 10 reference answers. \
The questions should test both general understanding and specific details \
(numbers, dates, names, facts). Answer each question using ONLY information \
from the text.

Respond in this EXACT JSON format:
{{"questions": ["q1", "q2", ...], "answers": ["a1", "a2", ...]}}

TEXT:
{source_text}"""


# ---------------------------------------------------------------------------
# Entity extraction (regex fallback when spaCy not available)
# ---------------------------------------------------------------------------

def _extract_entities_regex(text: str) -> list[str]:
    """Extract numbers and capitalised multi-word tokens as a simple fallback."""
    entities: set[str] = set()
    # Numbers (integers and decimals)
    for m in re.finditer(r"\b\d+(?:[.,]\d+)*\b", text):
        entities.add(m.group())
    # Capitalised words (likely proper nouns) — skip sentence-start words
    for m in re.finditer(r"\b[A-Z][a-z]{1,}\b", text):
        entities.add(m.group())
    return sorted(entities)


def _extract_entities(text: str) -> list[str]:
    try:
        import spacy  # type: ignore
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text)
        return sorted({ent.text.strip() for ent in doc.ents if ent.text.strip()})
    except Exception:
        return _extract_entities_regex(text)


# ---------------------------------------------------------------------------
# JSON list parser (same logic as BenchmarkGenerator._parse_json_list)
# ---------------------------------------------------------------------------

def _parse_json_list(raw: str, expected_length: int) -> list[str]:
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        raw = "\n".join(ln for ln in lines if not ln.startswith("```")).strip()
    try:
        result = json.loads(raw)
        if isinstance(result, list):
            items = [str(x) for x in result]
            while len(items) < expected_length:
                items.append("")
            return items[:expected_length]
    except (json.JSONDecodeError, ValueError):
        pass
    lines = [ln.strip().lstrip("-•0123456789. ") for ln in raw.splitlines() if ln.strip()]
    while len(lines) < expected_length:
        lines.append("")
    return lines[:expected_length]


# ---------------------------------------------------------------------------
# LLM backends
# ---------------------------------------------------------------------------

def _ask_anthropic_sdk(prompt: str, system: str, model: str = "claude-haiku-4-5-20251001") -> str | None:
    """Use the Anthropic Python SDK (requires ANTHROPIC_API_KEY)."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    try:
        import anthropic  # type: ignore
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model=model,
            max_tokens=2048,
            temperature=0,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text
    except Exception as exc:
        print(f"  [warn] Anthropic SDK error: {exc}", file=sys.stderr)
        return None


def _ask_claude_cli(prompt: str) -> str | None:
    """Call the claude CLI subprocess (fails inside a nested Claude Code session)."""
    try:
        result = subprocess.run(
            ["claude", "-p", prompt, "--output-format", "json"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            return data.get("result", "")
        stderr = result.stderr[:200]
        if "nested" in stderr.lower() or "CLAUDECODE" in stderr:
            # Permanent failure — stop trying CLI for the rest of the run
            _ask_claude_cli._disabled = True  # type: ignore[attr-defined]
        return None
    except Exception:
        return None


_ask_claude_cli._disabled = False  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Single-record generation
# ---------------------------------------------------------------------------

def _generate_record_llm(source_text: str, hidden_type: str) -> dict | None:
    """Try to generate a record with a real LLM. Returns None if all backends fail."""
    prompt = _COMBINED_PROMPT.format(source_text=source_text[:8000])

    raw: str | None = None

    # 1. Anthropic SDK
    raw = _ask_anthropic_sdk(prompt, _QUESTION_SYSTEM)

    # 2. claude CLI (if SDK failed)
    if raw is None and not _ask_claude_cli._disabled:  # type: ignore[attr-defined]
        raw = _ask_claude_cli(prompt)

    if raw is None:
        return None

    # Parse combined JSON response
    try:
        raw_stripped = raw.strip()
        if raw_stripped.startswith("```"):
            lines = raw_stripped.splitlines()
            raw_stripped = "\n".join(ln for ln in lines if not ln.startswith("```")).strip()
        data = json.loads(raw_stripped)
        questions = [str(q) for q in data.get("questions", [])]
        answers = [str(a) for a in data.get("answers", [])]
    except (json.JSONDecodeError, AttributeError):
        # Try treating as a questions-only array
        questions = _parse_json_list(raw, 10)
        answers = [""] * 10

    # Pad / truncate to 10
    while len(questions) < 10:
        questions.append("")
    questions = questions[:10]
    while len(answers) < 10:
        answers.append("")
    answers = answers[:10]

    reference_entities = [sorted(_extract_entities(ans)) for ans in answers]
    target_ratio = round(random.uniform(0.2, 0.6), 4)

    return {
        "source_text": source_text,
        "hidden_type": hidden_type,
        "questions": json.dumps(questions),
        "reference_answers": json.dumps(answers),
        "reference_entities": json.dumps(reference_entities),
        "target_ratio": target_ratio,
    }


def _generate_record_mock(source_text: str, hidden_type: str, rng: random.Random) -> dict:
    """Deterministic mock record (no LLM calls)."""
    # Import mock data from generator
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from craving_mind.benchmark.generator import MockBenchmarkGenerator
    from craving_mind.judge.entities import EntityExtractor

    # Reuse the generator's mock data but with per-text entities from the source
    mock_q = [
        "What is the main topic discussed in the text?",
        "What specific numbers or quantities are mentioned?",
        "Who are the key entities or people mentioned?",
        "What conditions or requirements are stated?",
        "What is the logical structure of the argument?",
        "What conclusion is reached in the text?",
        "What cause-and-effect relationships are described?",
        "What time periods or dates are referenced?",
        "What locations or places are mentioned?",
        "What actions or processes are described?",
    ]
    # Build answers from the actual source text (use regex to extract factual snippets)
    sentences = [s.strip() for s in re.split(r"[.!?]", source_text) if len(s.strip()) > 20]
    mock_a: list[str] = []
    for i in range(10):
        if sentences:
            mock_a.append(sentences[i % len(sentences)][:200])
        else:
            mock_a.append("Information not found in the provided text.")

    reference_entities = [sorted(_extract_entities_regex(ans)) for ans in mock_a]
    target_ratio = round(rng.uniform(0.2, 0.6), 4)

    return {
        "source_text": source_text,
        "hidden_type": hidden_type,
        "questions": json.dumps(mock_q),
        "reference_answers": json.dumps(mock_a),
        "reference_entities": json.dumps(reference_entities),
        "target_ratio": target_ratio,
    }


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _load_checkpoint() -> dict:
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH, encoding="utf-8") as fh:
            return json.load(fh)
    return {"completed": [], "records": []}


def _save_checkpoint(state: dict) -> None:
    BENCHMARKS_DIR.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_PATH, "w", encoding="utf-8") as fh:
        json.dump(state, fh, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Source loader
# ---------------------------------------------------------------------------

def _collect_sources() -> list[dict[str, str]]:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from craving_mind.benchmark.sources import load_texts_from_dir

    records: list[dict[str, str]] = []
    for hidden_type in ("discourse", "needle", "code"):
        type_dir = SOURCES_DIR / hidden_type
        if not type_dir.is_dir():
            continue
        loaded = load_texts_from_dir(str(type_dir), hidden_type)
        records.extend(loaded)
        print(f"  {hidden_type}: {len(loaded)} files")
    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build benchmark_v1.parquet from source texts.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output .parquet path.")
    parser.add_argument("--mock", action="store_true", help="Force MockBenchmarkGenerator (no LLM).")
    parser.add_argument("--reset", action="store_true", help="Ignore existing checkpoint and start fresh.")
    args = parser.parse_args(argv)

    output_path = Path(args.output)
    BENCHMARKS_DIR.mkdir(parents=True, exist_ok=True)

    # Collect all source texts
    print("Collecting sources ...")
    source_records = _collect_sources()
    if not source_records:
        print("[error] No source records found.", file=sys.stderr)
        return 1
    total = len(source_records)
    print(f"Total: {total} source texts\n")

    # Load or reset checkpoint
    state = {} if args.reset else _load_checkpoint()
    completed: set[str] = set(state.get("completed", []))
    rows: list[dict] = list(state.get("records", []))

    if completed:
        print(f"Resuming from checkpoint: {len(completed)}/{total} already done.\n")

    # Determine LLM mode.
    # Fall back to mock only when BOTH the SDK key is absent AND the CLI is
    # known-broken (e.g. nested Claude Code session).  By default the CLI is
    # not disabled, so a plain terminal run without ANTHROPIC_API_KEY will
    # still attempt the claude CLI subprocess.
    has_api_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
    cli_disabled: bool = _ask_claude_cli._disabled  # type: ignore[attr-defined]
    use_mock = args.mock or (not has_api_key and cli_disabled)

    rng = random.Random(42)
    if use_mock:
        print("[mode] MockBenchmarkGenerator (no real LLM calls)\n")
    elif has_api_key:
        print("[mode] Anthropic SDK  model=claude-haiku-4-5-20251001\n")
    else:
        print("[mode] claude CLI subprocess\n")

    # Process each source text
    for idx, rec in enumerate(source_records, start=1):
        # Use (hidden_type, idx) as the unique key — avoids prefix collisions
        src_key = f"{rec['hidden_type']}::{idx}"
        filename = f"{rec['hidden_type']}/{idx:03d}"

        if src_key in completed:
            continue

        print(f"Processing {idx}/{total}: {filename} ...", end=" ", flush=True)

        if use_mock:
            row = _generate_record_mock(rec["source_text"], rec["hidden_type"], rng)
        else:
            row = _generate_record_llm(rec["source_text"], rec["hidden_type"])
            if row is None:
                print(f"LLM failed — falling back to mock")
                row = _generate_record_mock(rec["source_text"], rec["hidden_type"], rng)

        rows.append(row)
        completed.add(src_key)
        print("done")

        # Save checkpoint every 10 texts
        if len(rows) % 10 == 0:
            _save_checkpoint({"completed": list(completed), "records": rows})
            print(f"  [checkpoint] {len(rows)}/{total} saved.")

    # Final checkpoint save
    _save_checkpoint({"completed": list(completed), "records": rows})

    # Write Parquet
    print(f"\nWriting {len(rows)} records to {output_path} ...")
    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(str(output_path), index=False)
    print("Done.\n")

    # Verify
    print("Verifying Parquet ...")
    df_check = pd.read_parquet(str(output_path))
    print(f"  Rows : {len(df_check)}")
    print(f"  Schema:\n{df_check.dtypes.to_string()}")

    # Clean up checkpoint
    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()
        print("\n  Checkpoint removed.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
