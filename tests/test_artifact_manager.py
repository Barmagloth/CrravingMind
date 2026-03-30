"""Tests for ArtifactManager (Phase 8)."""

import json
import os
import pytest

from craving_mind.orchestrator.artifact_manager import ArtifactManager


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def artifacts_dir(tmp_path):
    return str(tmp_path / "artifacts")


@pytest.fixture
def manager(artifacts_dir):
    return ArtifactManager(artifacts_dir)


def _meta(epoch: int = 0, score: float = 0.9, crav_id: str = "Crav-001") -> dict:
    return {
        "epoch": epoch,
        "crav_id": crav_id,
        "mean_score": score,
        "semantic_score": score,
        "entity_score": score,
        "score_by_type": {"discourse": score},
        "mean_compression_ratio": 0.45,
        "success_rate": score,
    }


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

class TestInit:
    def test_creates_artifacts_dir(self, artifacts_dir):
        ArtifactManager(artifacts_dir)
        assert os.path.isdir(artifacts_dir)

    def test_default_manifest_path_inside_artifacts_dir(self, artifacts_dir):
        am = ArtifactManager(artifacts_dir)
        assert am.manifest_path == os.path.join(artifacts_dir, "manifest.jsonl")

    def test_custom_manifest_path(self, tmp_path, artifacts_dir):
        custom = str(tmp_path / "custom_manifest.jsonl")
        am = ArtifactManager(artifacts_dir, manifest_path=custom)
        assert am.manifest_path == custom

    def test_fresh_version_starts_at_zero(self, manager):
        assert manager._current_version == 0
        assert manager.next_version == 1

    def test_loads_version_from_existing_manifest(self, artifacts_dir):
        # Pre-write a manifest with version 5.
        manifest = os.path.join(artifacts_dir, "manifest.jsonl")
        os.makedirs(artifacts_dir, exist_ok=True)
        with open(manifest, "w") as f:
            f.write(json.dumps({"version": 5, "filename": "x.py"}) + "\n")
            f.write(json.dumps({"version": 3, "filename": "y.py"}) + "\n")
        am = ArtifactManager(artifacts_dir, manifest_path=manifest)
        assert am._current_version == 5
        assert am.next_version == 6


# ---------------------------------------------------------------------------
# export()
# ---------------------------------------------------------------------------

class TestExport:
    def test_version_increments_on_each_export(self, manager):
        e1 = manager.export("def compress(t, r): return t", _meta(epoch=0, score=0.9))
        e2 = manager.export("def compress(t, r): return t[:10]", _meta(epoch=1, score=0.91))
        assert e1["version"] == 1
        assert e2["version"] == 2

    def test_file_created(self, manager, artifacts_dir):
        entry = manager.export("# code", _meta(epoch=0, score=0.88))
        assert os.path.isfile(entry["filepath"])

    def test_file_contains_header_and_code(self, manager):
        code = "def compress(t, r): return t"
        entry = manager.export(code, _meta(epoch=2, score=0.92, crav_id="Crav-007"))
        content = open(entry["filepath"], encoding="utf-8").read()
        assert "CravingMind Artifact v1" in content
        assert "Epoch: 2" in content
        assert "Crav: Crav-007" in content
        assert code in content

    def test_filename_encodes_version_epoch_score(self, manager):
        entry = manager.export("# v1", _meta(epoch=3, score=0.856))
        assert "v0001" in entry["filename"]
        assert "epoch0003" in entry["filename"]
        assert "0.856" in entry["filename"]

    def test_manifest_appended(self, manager):
        manager.export("# a", _meta(epoch=0, score=0.9))
        manager.export("# b", _meta(epoch=1, score=0.91))
        history = manager.get_history()
        assert len(history) == 2

    def test_entry_contains_all_metadata_fields(self, manager):
        meta = _meta(epoch=1, score=0.88, crav_id="Crav-042")
        entry = manager.export("# code", meta)
        for key in ("version", "filename", "filepath", "timestamp",
                    "epoch", "crav_id", "mean_score", "semantic_score",
                    "entity_score", "score_by_type", "mean_compression_ratio",
                    "success_rate"):
            assert key in entry, f"Missing key: {key}"

    def test_timestamp_is_string(self, manager):
        entry = manager.export("# x", _meta())
        assert isinstance(entry["timestamp"], str)
        assert len(entry["timestamp"]) > 0

    def test_multiple_exports_unique_filenames(self, manager):
        e1 = manager.export("# v1", _meta(epoch=0, score=0.9))
        e2 = manager.export("# v2", _meta(epoch=1, score=0.91))
        assert e1["filename"] != e2["filename"]


# ---------------------------------------------------------------------------
# get_best()
# ---------------------------------------------------------------------------

class TestGetBest:
    def test_returns_none_when_empty(self, manager):
        assert manager.get_best() is None

    def test_returns_best_by_mean_score(self, manager):
        manager.export("# a", _meta(epoch=0, score=0.80))
        manager.export("# b", _meta(epoch=1, score=0.95))
        manager.export("# c", _meta(epoch=2, score=0.85))
        best = manager.get_best("mean_score")
        assert best["mean_score"] == pytest.approx(0.95)

    def test_returns_best_by_custom_metric(self, manager):
        meta1 = {**_meta(epoch=0, score=0.80), "entity_score": 0.70}
        meta2 = {**_meta(epoch=1, score=0.85), "entity_score": 0.99}
        manager.export("# a", meta1)
        manager.export("# b", meta2)
        best = manager.get_best("entity_score")
        assert best["entity_score"] == pytest.approx(0.99)

    def test_single_entry_is_best(self, manager):
        manager.export("# only", _meta(score=0.77))
        best = manager.get_best()
        assert best is not None
        assert best["mean_score"] == pytest.approx(0.77)


# ---------------------------------------------------------------------------
# get_latest()
# ---------------------------------------------------------------------------

class TestGetLatest:
    def test_returns_none_when_empty(self, manager):
        assert manager.get_latest() is None

    def test_returns_highest_version(self, manager):
        manager.export("# v1", _meta(epoch=0, score=0.80))
        manager.export("# v2", _meta(epoch=1, score=0.90))
        latest = manager.get_latest()
        assert latest["version"] == 2

    def test_single_entry_is_latest(self, manager):
        manager.export("# x", _meta())
        assert manager.get_latest()["version"] == 1


# ---------------------------------------------------------------------------
# get_history()
# ---------------------------------------------------------------------------

class TestGetHistory:
    def test_returns_empty_list_when_no_manifest(self, manager):
        assert manager.get_history() == []

    def test_returns_all_entries_in_order(self, manager):
        manager.export("# a", _meta(epoch=0, score=0.80))
        manager.export("# b", _meta(epoch=1, score=0.90))
        manager.export("# c", _meta(epoch=2, score=0.85))
        history = manager.get_history()
        assert len(history) == 3
        assert [e["version"] for e in history] == [1, 2, 3]

    def test_manifest_roundtrip(self, artifacts_dir):
        """Write entries, reload manager from disk, verify history intact."""
        am = ArtifactManager(artifacts_dir)
        am.export("# v1", _meta(epoch=0, score=0.88))
        am.export("# v2", _meta(epoch=1, score=0.92))

        # Reload fresh instance from same directory.
        am2 = ArtifactManager(artifacts_dir)
        assert am2._current_version == 2
        history = am2.get_history()
        assert len(history) == 2
        assert history[0]["epoch"] == 0
        assert history[1]["epoch"] == 1


# ---------------------------------------------------------------------------
# has_changed()
# ---------------------------------------------------------------------------

class TestHasChanged:
    def test_identical_code_returns_false(self, manager):
        code = "def compress(t, r): return t"
        assert manager.has_changed(code, code) is False

    def test_whitespace_only_difference_returns_false(self, manager):
        a = "def compress(t, r): return t\n"
        b = "def compress(t, r): return t   "
        assert manager.has_changed(a, b) is False

    def test_different_code_returns_true(self, manager):
        a = "def compress(t, r): return t"
        b = "def compress(t, r): return t[:10]"
        assert manager.has_changed(a, b) is True

    def test_empty_vs_nonempty_returns_true(self, manager):
        assert manager.has_changed("", "def compress(t, r): return t") is True

    def test_both_empty_returns_false(self, manager):
        assert manager.has_changed("", "") is False
