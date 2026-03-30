"""Tests for craving_mind.benchmark — generator, loader, sources."""

from __future__ import annotations

import json
import random
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from craving_mind.benchmark.generator import MockBenchmarkGenerator, BenchmarkGenerator
from craving_mind.benchmark.loader import BenchmarkLoader
from craving_mind.benchmark.sources import load_texts_from_dir, list_available_sources


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FIXTURES = Path(__file__).parent / "fixtures"
SAMPLE_TEXTS_DIR = FIXTURES / "sample_texts"


def _make_parquet(path: str, n: int = 5) -> list[dict]:
    """Write a small synthetic benchmark Parquet and return the records."""
    rng = random.Random(0)
    records = []
    types = ["discourse", "needle", "code"]
    for i in range(n):
        questions = [f"Question {i}-{j}?" for j in range(10)]
        answers = [f"Answer {i}-{j}." for j in range(10)]
        entities = [[f"entity{i}{j}"] for j in range(10)]
        records.append(
            {
                "source_text": f"Source text number {i}. Contains important info.",
                "hidden_type": types[i % len(types)],
                "questions": json.dumps(questions),
                "reference_answers": json.dumps(answers),
                "reference_entities": json.dumps(entities),
                "target_ratio": round(rng.uniform(0.2, 0.6), 4),
            }
        )
    df = pd.DataFrame(records)
    df.to_parquet(path, index=False)
    return records


# ---------------------------------------------------------------------------
# BenchmarkLoader tests
# ---------------------------------------------------------------------------

class TestBenchmarkLoader:
    def test_load_frozen_returns_correct_count(self, tmp_path):
        p = str(tmp_path / "bench.parquet")
        _make_parquet(p, n=5)
        loader = BenchmarkLoader()
        tasks = loader.load_frozen(p)
        assert len(tasks) == 5

    def test_load_frozen_deserialises_questions(self, tmp_path):
        p = str(tmp_path / "bench.parquet")
        _make_parquet(p, n=3)
        loader = BenchmarkLoader()
        tasks = loader.load_frozen(p)
        for t in tasks:
            assert isinstance(t["questions"], list)
            assert len(t["questions"]) == 10

    def test_load_frozen_deserialises_reference_answers(self, tmp_path):
        p = str(tmp_path / "bench.parquet")
        _make_parquet(p, n=2)
        loader = BenchmarkLoader()
        tasks = loader.load_frozen(p)
        for t in tasks:
            assert isinstance(t["reference_answers"], list)
            assert all(isinstance(a, str) for a in t["reference_answers"])

    def test_load_frozen_deserialises_reference_entities(self, tmp_path):
        p = str(tmp_path / "bench.parquet")
        _make_parquet(p, n=2)
        loader = BenchmarkLoader()
        tasks = loader.load_frozen(p)
        for t in tasks:
            assert isinstance(t["reference_entities"], list)
            assert all(isinstance(e, list) for e in t["reference_entities"])

    def test_load_frozen_preserves_target_ratio(self, tmp_path):
        p = str(tmp_path / "bench.parquet")
        orig = _make_parquet(p, n=3)
        loader = BenchmarkLoader()
        tasks = loader.load_frozen(p)
        for task, rec in zip(tasks, orig):
            assert task["target_ratio"] == pytest.approx(rec["target_ratio"])

    def test_load_frozen_missing_ratio_filled(self, tmp_path):
        """If target_ratio is NaN/None in Parquet, loader should assign a random value."""
        p = str(tmp_path / "bench.parquet")
        records = [
            {
                "source_text": "text",
                "hidden_type": "discourse",
                "questions": json.dumps(["q"] * 10),
                "reference_answers": json.dumps(["a"] * 10),
                "reference_entities": json.dumps([[] for _ in range(10)]),
                "target_ratio": None,
            }
        ]
        df = pd.DataFrame(records)
        df.to_parquet(p, index=False)
        loader = BenchmarkLoader()
        tasks = loader.load_frozen(p)
        assert tasks[0]["target_ratio"] is not None
        assert 0.0 <= tasks[0]["target_ratio"] <= 1.0

    def test_get_epoch_tasks_combines_frozen_and_dynamic(self, tmp_path):
        p = str(tmp_path / "bench.parquet")
        _make_parquet(p, n=4)
        loader = BenchmarkLoader()
        frozen = loader.load_frozen(p)
        dynamic = [{"source_text": "dyn", "hidden_type": "code",
                    "questions": [], "reference_answers": [],
                    "reference_entities": [], "target_ratio": 0.4}]
        combined = loader.get_epoch_tasks(frozen, dynamic)
        assert len(combined) == 5

    def test_get_epoch_tasks_no_dynamic(self, tmp_path):
        p = str(tmp_path / "bench.parquet")
        _make_parquet(p, n=3)
        loader = BenchmarkLoader()
        frozen = loader.load_frozen(p)
        combined = loader.get_epoch_tasks(frozen)
        assert len(combined) == 3

    def test_get_epoch_tasks_is_dynamic_flag(self, tmp_path):
        p = str(tmp_path / "bench.parquet")
        _make_parquet(p, n=2)
        loader = BenchmarkLoader()
        frozen = loader.load_frozen(p)
        dynamic = [{"source_text": "d", "hidden_type": "needle",
                    "questions": [], "reference_answers": [],
                    "reference_entities": [], "target_ratio": 0.3}]
        combined = loader.get_epoch_tasks(frozen, dynamic)
        frozen_flags = [t["is_dynamic"] for t in combined if not t["is_dynamic"]]
        dynamic_flags = [t["is_dynamic"] for t in combined if t["is_dynamic"]]
        assert len(frozen_flags) == 2
        assert len(dynamic_flags) == 1

    def test_get_epoch_tasks_all_dynamic_no_frozen(self):
        loader = BenchmarkLoader()
        dynamic = [
            {"source_text": f"d{i}", "hidden_type": "code",
             "questions": [], "reference_answers": [],
             "reference_entities": [], "target_ratio": 0.4}
            for i in range(3)
        ]
        combined = loader.get_epoch_tasks([], dynamic)
        assert len(combined) == 3
        assert all(t["is_dynamic"] for t in combined)

    def test_select_frozen_subset_returns_sample(self, tmp_path):
        p = str(tmp_path / "bench.parquet")
        _make_parquet(p, n=10)
        loader = BenchmarkLoader()
        frozen = loader.load_frozen(p)
        subset = loader.select_frozen_subset(frozen, tasks_per_epoch=4)
        assert len(subset) == 4

    def test_select_frozen_subset_when_pool_smaller_than_requested(self, tmp_path):
        p = str(tmp_path / "bench.parquet")
        _make_parquet(p, n=3)
        loader = BenchmarkLoader()
        frozen = loader.load_frozen(p)
        subset = loader.select_frozen_subset(frozen, tasks_per_epoch=100)
        assert len(subset) == 3

    def test_select_frozen_subset_single_task(self, tmp_path):
        p = str(tmp_path / "bench.parquet")
        _make_parquet(p, n=1)
        loader = BenchmarkLoader()
        frozen = loader.load_frozen(p)
        subset = loader.select_frozen_subset(frozen, tasks_per_epoch=1)
        assert len(subset) == 1

    def test_get_epoch_tasks_does_not_mutate_input(self, tmp_path):
        p = str(tmp_path / "bench.parquet")
        _make_parquet(p, n=3)
        loader = BenchmarkLoader()
        frozen = loader.load_frozen(p)
        frozen_copy = [dict(t) for t in frozen]
        loader.get_epoch_tasks(frozen)
        # original dicts should not have is_dynamic added
        for t in frozen_copy:
            assert "is_dynamic" not in t


# ---------------------------------------------------------------------------
# BenchmarkGenerator (Mock) tests
# ---------------------------------------------------------------------------

class TestMockBenchmarkGenerator:
    def test_generate_record_structure(self):
        gen = MockBenchmarkGenerator()
        rec = gen.generate_record("Some source text here.", "discourse")
        assert set(rec.keys()) == {
            "source_text", "hidden_type", "questions",
            "reference_answers", "reference_entities", "target_ratio",
        }

    def test_generate_record_questions_is_json_list(self):
        gen = MockBenchmarkGenerator()
        rec = gen.generate_record("Text.", "needle")
        qs = json.loads(rec["questions"])
        assert isinstance(qs, list)
        assert len(qs) == 10

    def test_generate_record_answers_is_json_list(self):
        gen = MockBenchmarkGenerator()
        rec = gen.generate_record("Text.", "code")
        ans = json.loads(rec["reference_answers"])
        assert isinstance(ans, list)
        assert len(ans) == 10

    def test_generate_record_entities_is_json_list_of_lists(self):
        gen = MockBenchmarkGenerator()
        rec = gen.generate_record("Text.", "discourse")
        ents = json.loads(rec["reference_entities"])
        assert isinstance(ents, list)
        assert len(ents) == 10
        assert all(isinstance(e, list) for e in ents)

    def test_generate_record_target_ratio_in_range(self):
        gen = MockBenchmarkGenerator()
        for _ in range(10):
            rec = gen.generate_record("Text.", "needle")
            assert 0.2 <= rec["target_ratio"] <= 0.6

    def test_generate_record_hidden_type_preserved(self):
        gen = MockBenchmarkGenerator()
        for hidden_type in ("discourse", "needle", "code"):
            rec = gen.generate_record("Text.", hidden_type)
            assert rec["hidden_type"] == hidden_type

    def test_generate_benchmark_writes_parquet(self, tmp_path):
        gen = MockBenchmarkGenerator()
        sources = [
            {"source_text": f"Text {i}.", "hidden_type": "discourse"}
            for i in range(3)
        ]
        out = str(tmp_path / "bench.parquet")
        gen.generate_benchmark(sources, out)
        assert Path(out).exists()

    def test_parquet_round_trip(self, tmp_path):
        gen = MockBenchmarkGenerator(seed=0)
        sources = [
            {"source_text": "Alpha bravo charlie.", "hidden_type": "discourse"},
            {"source_text": "Delta echo foxtrot.", "hidden_type": "needle"},
            {"source_text": "def foo(): return 42", "hidden_type": "code"},
        ]
        out = str(tmp_path / "round_trip.parquet")
        gen.generate_benchmark(sources, out)

        loader = BenchmarkLoader()
        tasks = loader.load_frozen(out)
        assert len(tasks) == 3
        for task in tasks:
            assert isinstance(task["questions"], list)
            assert len(task["questions"]) == 10
            assert isinstance(task["reference_answers"], list)
            assert isinstance(task["reference_entities"], list)
            assert 0.2 <= task["target_ratio"] <= 0.6

    def test_parquet_round_trip_source_text_preserved(self, tmp_path):
        gen = MockBenchmarkGenerator(seed=1)
        text = "The quick brown fox jumps over the lazy dog."
        sources = [{"source_text": text, "hidden_type": "discourse"}]
        out = str(tmp_path / "rt2.parquet")
        gen.generate_benchmark(sources, out)
        loader = BenchmarkLoader()
        tasks = loader.load_frozen(out)
        assert tasks[0]["source_text"] == text

    def test_generate_dynamic_batch_count(self):
        gen = MockBenchmarkGenerator()
        batch = gen.generate_dynamic_batch(5)
        assert len(batch) == 5

    def test_generate_dynamic_batch_record_structure(self):
        gen = MockBenchmarkGenerator()
        batch = gen.generate_dynamic_batch(3)
        for rec in batch:
            assert "source_text" in rec
            assert "hidden_type" in rec
            qs = json.loads(rec["questions"])
            assert len(qs) == 10

    def test_generate_dynamic_batch_respects_types(self):
        gen = MockBenchmarkGenerator()
        batch = gen.generate_dynamic_batch(20, hidden_types=["discourse"])
        for rec in batch:
            assert rec["hidden_type"] == "discourse"

    def test_deterministic_with_same_seed(self):
        gen1 = MockBenchmarkGenerator(seed=99)
        gen2 = MockBenchmarkGenerator(seed=99)
        r1 = gen1.generate_record("Same text.", "discourse")
        r2 = gen2.generate_record("Same text.", "discourse")
        assert r1["target_ratio"] == r2["target_ratio"]

    def test_parse_json_list_strips_code_fences(self):
        raw = '```json\n["a", "b"]\n```'
        result = BenchmarkGenerator._parse_json_list(raw, expected_length=2)
        assert result == ["a", "b"]

    def test_parse_json_list_pads_short_list(self):
        raw = json.dumps(["q1", "q2"])
        result = BenchmarkGenerator._parse_json_list(raw, expected_length=5)
        assert len(result) == 5
        assert result[2] == ""

    def test_parse_json_list_truncates_long_list(self):
        raw = json.dumps([f"q{i}" for i in range(15)])
        result = BenchmarkGenerator._parse_json_list(raw, expected_length=10)
        assert len(result) == 10


# ---------------------------------------------------------------------------
# sources.py tests
# ---------------------------------------------------------------------------

class TestLoadTextsFromDir:
    def test_loads_txt_files(self):
        records = load_texts_from_dir(str(SAMPLE_TEXTS_DIR / "discourse"), "discourse")
        assert len(records) >= 2

    def test_assigns_correct_hidden_type(self):
        records = load_texts_from_dir(str(SAMPLE_TEXTS_DIR / "discourse"), "discourse")
        assert all(r["hidden_type"] == "discourse" for r in records)

    def test_loads_md_files(self):
        records = load_texts_from_dir(str(SAMPLE_TEXTS_DIR / "discourse"), "discourse")
        # doc2.md should be loaded
        texts = [r["source_text"] for r in records]
        assert any("Federal Reserve" in t for t in texts)

    def test_needle_dir(self):
        records = load_texts_from_dir(str(SAMPLE_TEXTS_DIR / "needle"), "needle")
        assert len(records) >= 1
        assert all(r["hidden_type"] == "needle" for r in records)

    def test_nonexistent_dir_returns_empty(self):
        records = load_texts_from_dir("/nonexistent/path/xyz", "discourse")
        assert records == []

    def test_empty_dir_returns_empty(self, tmp_path):
        records = load_texts_from_dir(str(tmp_path), "discourse")
        assert records == []

    def test_ignores_non_text_files(self, tmp_path):
        (tmp_path / "data.csv").write_text("a,b,c")
        (tmp_path / "image.png").write_bytes(b"\x89PNG")
        records = load_texts_from_dir(str(tmp_path), "discourse")
        assert records == []

    def test_skips_empty_files(self, tmp_path):
        (tmp_path / "empty.txt").write_text("   ")
        (tmp_path / "real.txt").write_text("Some actual content here.")
        records = load_texts_from_dir(str(tmp_path), "discourse")
        assert len(records) == 1

    def test_returns_source_text_content(self):
        records = load_texts_from_dir(str(SAMPLE_TEXTS_DIR / "needle"), "needle")
        for r in records:
            assert len(r["source_text"]) > 0


class TestListAvailableSources:
    def test_counts_types(self):
        counts = list_available_sources(str(SAMPLE_TEXTS_DIR))
        assert "discourse" in counts
        assert counts["discourse"] >= 2

    def test_includes_needle(self):
        counts = list_available_sources(str(SAMPLE_TEXTS_DIR))
        assert "needle" in counts
        assert counts["needle"] >= 1

    def test_empty_subdir_counted_as_zero(self):
        counts = list_available_sources(str(SAMPLE_TEXTS_DIR))
        # empty_type subdir has no files
        assert counts.get("empty_type", 0) == 0

    def test_nonexistent_root_returns_empty(self):
        counts = list_available_sources("/nonexistent/path")
        assert counts == {}

    def test_returns_dict(self):
        counts = list_available_sources(str(SAMPLE_TEXTS_DIR))
        assert isinstance(counts, dict)


# ---------------------------------------------------------------------------
# Integration: full pipeline frozen + dynamic epoch
# ---------------------------------------------------------------------------

class TestEpochIntegration:
    def test_epoch_with_frozen_and_dynamic(self, tmp_path):
        gen = MockBenchmarkGenerator(seed=7)
        sources = [
            {"source_text": f"Frozen text {i}.", "hidden_type": "discourse"}
            for i in range(4)
        ]
        out = str(tmp_path / "frozen.parquet")
        gen.generate_benchmark(sources, out)

        loader = BenchmarkLoader()
        frozen = loader.load_frozen(out)
        subset = loader.select_frozen_subset(frozen, tasks_per_epoch=3)

        dynamic = gen.generate_dynamic_batch(2, hidden_types=["needle"])
        # dynamic records have JSON-serialised fields; deserialise for consistency
        for rec in dynamic:
            for field in ("questions", "reference_answers", "reference_entities"):
                if isinstance(rec[field], str):
                    rec[field] = json.loads(rec[field])

        epoch_tasks = loader.get_epoch_tasks(subset, dynamic)
        assert len(epoch_tasks) == 5

        frozen_count = sum(1 for t in epoch_tasks if not t["is_dynamic"])
        dynamic_count = sum(1 for t in epoch_tasks if t["is_dynamic"])
        assert frozen_count == 3
        assert dynamic_count == 2
