"""Tests for Phase 7: dashboard — storage, metrics, server."""

import json
import os
import tempfile

import pytest
from fastapi.testclient import TestClient

from craving_mind.dashboard.storage import MetricsStorage
from craving_mind.dashboard.metrics import MetricsCollector
from craving_mind.dashboard.server import DashboardServer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

BASE_CONFIG = {
    "budget": {"base_tokens": 50000},
    "memory": {"bible_max_weight_pct": 0.20},
    "phases": {"phase2_start": 11, "phase3_start": 26},
    "judge": {"pass_threshold": 0.85, "dynamic_multiplier": 1.3},
    "dashboard": {"port": 8080, "update_interval_seconds": 2},
}


def _write_jsonl(path: str, rows: list[dict]):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


@pytest.fixture
def run_dir(tmp_path):
    return str(tmp_path)


@pytest.fixture
def populated_run_dir(tmp_path):
    """Run dir with epoch_log, task_log, manifest, checkpoint."""
    epoch_rows = [
        {
            "epoch": 0,
            "success_rate": 0.72,
            "frozen_success_rate": 0.75,
            "dynamic_success_rate": 0.68,
            "overfit_gap": 0.07,
            "tasks_completed": 9,
            "tasks_total": 10,
            "saved_tokens": 3000,
            "is_oom": False,
            "drift_detected": False,
            "artifact_path": None,
            "ts": "2025-01-01T00:00:00Z",
        },
        {
            "epoch": 1,
            "success_rate": 0.88,
            "frozen_success_rate": 0.90,
            "dynamic_success_rate": 0.85,
            "overfit_gap": 0.05,
            "tasks_completed": 10,
            "tasks_total": 10,
            "saved_tokens": 5000,
            "is_oom": False,
            "drift_detected": False,
            "artifact_path": "artifacts/compress_v0001_epoch0001_0.880.py",
            "ts": "2025-01-01T00:01:00Z",
        },
    ]
    _write_jsonl(str(tmp_path / "epoch_log.jsonl"), epoch_rows)

    task_rows = [
        {"epoch": 0, "task_score": 0.80, "passed": True, "hidden_type": "discourse",
         "is_dynamic": False, "compression_ratio": 0.5, "semantic_score": 0.82,
         "entity_score": 0.78, "tokens_spent": 500},
        {"epoch": 0, "task_score": 0.60, "passed": False, "hidden_type": "needle",
         "is_dynamic": True, "compression_ratio": 0.45, "semantic_score": 0.65,
         "entity_score": 0.55, "tokens_spent": 600},
        {"epoch": 1, "task_score": 0.91, "passed": True, "hidden_type": "code",
         "is_dynamic": False, "compression_ratio": 0.4, "semantic_score": 0.93,
         "entity_score": 0.89, "tokens_spent": 450},
    ]
    _write_jsonl(str(tmp_path / "task_log.jsonl"), task_rows)

    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()
    manifest_rows = [
        {
            "version": 1,
            "filename": "compress_v0001_epoch0001_0.880.py",
            "filepath": str(artifacts_dir / "compress_v0001_epoch0001_0.880.py"),
            "epoch": 1,
            "crav_id": "Crav-001",
            "mean_score": 0.880,
            "semantic_score": 0.90,
            "entity_score": 0.86,
            "score_by_type": {"discourse": 0.88, "needle": 0.85, "code": 0.91},
            "success_rate": 0.88,
        }
    ]
    _write_jsonl(str(artifacts_dir / "manifest.jsonl"), manifest_rows)

    # Write artifact file
    (artifacts_dir / "compress_v0001_epoch0001_0.880.py").write_text(
        "# Artifact v1\ndef compress(text, ratio):\n    return text[:int(len(text)*ratio)]\n",
        encoding="utf-8",
    )

    # Checkpoint
    checkpoint = {"epoch": 1, "success_rate": 0.88, "saved_tokens": 5000, "is_oom": False}
    (tmp_path / "checkpoint.json").write_text(json.dumps(checkpoint), encoding="utf-8")

    # Agent workspace with files
    ws = tmp_path / "agent_workspace"
    ws.mkdir()
    (ws / "compress.py").write_text("def compress(text, ratio):\n    return text\n", encoding="utf-8")
    (ws / "bible.md").write_text("# Bible\n\n## Rules\n- rule 1\n", encoding="utf-8")
    (ws / "graveyard.md").write_text("# Graveyard\n", encoding="utf-8")

    return str(tmp_path)


# ---------------------------------------------------------------------------
# MetricsStorage tests
# ---------------------------------------------------------------------------

class TestMetricsStorage:
    def test_get_epoch_history_empty(self, run_dir):
        s = MetricsStorage(run_dir)
        assert s.get_epoch_history() == []

    def test_get_epoch_history(self, populated_run_dir):
        s = MetricsStorage(populated_run_dir)
        history = s.get_epoch_history()
        assert len(history) == 2
        assert history[0]["epoch"] == 0
        assert history[1]["epoch"] == 1

    def test_get_task_history_all(self, populated_run_dir):
        s = MetricsStorage(populated_run_dir)
        tasks = s.get_task_history()
        assert len(tasks) == 3

    def test_get_task_history_by_epoch(self, populated_run_dir):
        s = MetricsStorage(populated_run_dir)
        tasks = s.get_task_history(epoch=0)
        assert len(tasks) == 2
        assert all(t["epoch"] == 0 for t in tasks)

    def test_get_task_history_epoch_filter_empty(self, populated_run_dir):
        s = MetricsStorage(populated_run_dir)
        assert s.get_task_history(epoch=99) == []

    def test_get_artifact_history(self, populated_run_dir):
        s = MetricsStorage(populated_run_dir)
        arts = s.get_artifact_history()
        assert len(arts) == 1
        assert arts[0]["version"] == 1

    def test_get_artifact_history_empty(self, run_dir):
        s = MetricsStorage(run_dir)
        assert s.get_artifact_history() == []

    def test_get_checkpoint(self, populated_run_dir):
        s = MetricsStorage(populated_run_dir)
        ck = s.get_checkpoint()
        assert ck is not None
        assert ck["epoch"] == 1

    def test_get_checkpoint_missing(self, run_dir):
        s = MetricsStorage(run_dir)
        assert s.get_checkpoint() is None

    def test_get_latest_epoch(self, populated_run_dir):
        s = MetricsStorage(populated_run_dir)
        latest = s.get_latest_epoch()
        assert latest is not None
        assert latest["epoch"] == 1

    def test_get_latest_epoch_empty(self, run_dir):
        s = MetricsStorage(run_dir)
        assert s.get_latest_epoch() is None

    def test_handles_malformed_jsonl(self, tmp_path):
        path = str(tmp_path / "epoch_log.jsonl")
        with open(path, "w") as f:
            f.write('{"epoch": 0}\n')
            f.write('NOT JSON\n')
            f.write('{"epoch": 1}\n')
        s = MetricsStorage(str(tmp_path))
        history = s.get_epoch_history()
        assert len(history) == 2


# ---------------------------------------------------------------------------
# MetricsCollector tests
# ---------------------------------------------------------------------------

class TestMetricsCollector:
    def test_empty_state(self, run_dir):
        storage = MetricsStorage(run_dir)
        collector = MetricsCollector(storage, BASE_CONFIG)
        state = collector.get_dashboard_state()
        assert "health" in state
        assert "efficiency" in state
        assert "evolution" in state
        assert "artifact" in state
        assert "live" in state
        assert "epoch_history" in state

    def test_state_with_data(self, populated_run_dir):
        storage = MetricsStorage(populated_run_dir)
        collector = MetricsCollector(storage, BASE_CONFIG)
        state = collector.get_dashboard_state()

        # Health
        health = state["health"]
        assert health["phase"] == 1  # epoch 1 < phase2_start=11
        assert health["oom_count"] == 0

        # Efficiency
        eff = state["efficiency"]
        assert eff["success_rate_pct"] > 0

        # Live
        live = state["live"]
        assert live["current_epoch"] == 1
        assert live["total_epochs"] == 2

        # Artifact
        art = state["artifact"]
        assert art["best_mean_score"] == pytest.approx(0.880)
        assert art["best_epoch"] == 1
        assert art["latest_version_str"] == "v0001"

    def test_phase_detection(self, tmp_path):
        # epoch 15 → phase 2
        rows = [{"epoch": i, "success_rate": 0.9, "frozen_success_rate": 0.9,
                 "dynamic_success_rate": 0.9, "overfit_gap": 0.0,
                 "tasks_completed": 10, "tasks_total": 10,
                 "saved_tokens": 0, "is_oom": False, "drift_detected": False,
                 "artifact_path": None, "ts": ""} for i in range(15)]
        _write_jsonl(str(tmp_path / "epoch_log.jsonl"), rows)
        storage = MetricsStorage(str(tmp_path))
        collector = MetricsCollector(storage, BASE_CONFIG)
        state = collector.get_dashboard_state()
        assert state["health"]["phase"] == 2

    def test_epoch_history_for_charts(self, populated_run_dir):
        storage = MetricsStorage(populated_run_dir)
        collector = MetricsCollector(storage, BASE_CONFIG)
        state = collector.get_dashboard_state()
        hist = state["epoch_history"]
        assert len(hist) == 2
        assert all("success_rate" in e for e in hist)

    def test_artifact_metrics_empty(self, run_dir):
        storage = MetricsStorage(run_dir)
        collector = MetricsCollector(storage, BASE_CONFIG)
        state = collector.get_dashboard_state()
        art = state["artifact"]
        assert art["latest_version"] is None
        assert art["best_mean_score"] == 0.0


# ---------------------------------------------------------------------------
# DashboardServer tests
# ---------------------------------------------------------------------------

class TestDashboardServer:
    def test_app_created(self, run_dir):
        server = DashboardServer(BASE_CONFIG, run_dir)
        assert server.app is not None

    def test_index_returns_html(self, run_dir):
        server = DashboardServer(BASE_CONFIG, run_dir)
        client = TestClient(server.app)
        resp = client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        assert "CravingMind" in resp.text

    def test_api_epochs_empty(self, run_dir):
        server = DashboardServer(BASE_CONFIG, run_dir)
        client = TestClient(server.app)
        resp = client.get("/api/epochs")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_api_epochs_with_data(self, populated_run_dir):
        server = DashboardServer(BASE_CONFIG, populated_run_dir)
        client = TestClient(server.app)
        resp = client.get("/api/epochs")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2

    def test_api_tasks(self, populated_run_dir):
        server = DashboardServer(BASE_CONFIG, populated_run_dir)
        client = TestClient(server.app)
        resp = client.get("/api/tasks/0")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2

    def test_api_artifacts(self, populated_run_dir):
        server = DashboardServer(BASE_CONFIG, populated_run_dir)
        client = TestClient(server.app)
        resp = client.get("/api/artifacts")
        assert resp.status_code == 200
        assert len(resp.json()) == 1

    def test_api_file_compress_py(self, populated_run_dir):
        server = DashboardServer(BASE_CONFIG, populated_run_dir)
        client = TestClient(server.app)
        resp = client.get("/api/files/compress.py")
        assert resp.status_code == 200
        assert "compress" in resp.text

    def test_api_file_bible_md(self, populated_run_dir):
        server = DashboardServer(BASE_CONFIG, populated_run_dir)
        client = TestClient(server.app)
        resp = client.get("/api/files/bible.md")
        assert resp.status_code == 200
        assert "Bible" in resp.text

    def test_api_file_graveyard_md(self, populated_run_dir):
        server = DashboardServer(BASE_CONFIG, populated_run_dir)
        client = TestClient(server.app)
        resp = client.get("/api/files/graveyard.md")
        assert resp.status_code == 200

    def test_api_file_disallowed(self, run_dir):
        server = DashboardServer(BASE_CONFIG, run_dir)
        client = TestClient(server.app)
        resp = client.get("/api/files/secrets.txt")
        assert resp.status_code == 404

    def test_api_file_missing_returns_placeholder(self, run_dir):
        server = DashboardServer(BASE_CONFIG, run_dir)
        client = TestClient(server.app)
        resp = client.get("/api/files/compress.py")
        assert resp.status_code == 200
        assert "not yet created" in resp.text

    def test_api_artifact_version(self, populated_run_dir):
        server = DashboardServer(BASE_CONFIG, populated_run_dir)
        client = TestClient(server.app)
        resp = client.get("/api/artifacts/1")
        assert resp.status_code == 200
        assert "compress" in resp.text

    def test_api_artifact_version_not_found(self, populated_run_dir):
        server = DashboardServer(BASE_CONFIG, populated_run_dir)
        client = TestClient(server.app)
        resp = client.get("/api/artifacts/999")
        assert resp.status_code == 404

    def test_html_contains_chart_js(self, run_dir):
        server = DashboardServer(BASE_CONFIG, run_dir)
        client = TestClient(server.app)
        resp = client.get("/")
        assert "Chart.js" in resp.text or "chart.umd.min.js" in resp.text

    def test_html_contains_websocket_code(self, run_dir):
        server = DashboardServer(BASE_CONFIG, run_dir)
        client = TestClient(server.app)
        resp = client.get("/")
        assert "WebSocket" in resp.text

    def test_html_contains_file_viewer(self, run_dir):
        server = DashboardServer(BASE_CONFIG, run_dir)
        client = TestClient(server.app)
        resp = client.get("/")
        assert "compress.py" in resp.text
        assert "bible.md" in resp.text
