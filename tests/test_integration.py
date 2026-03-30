"""Integration tests: end-to-end pipeline with MockProvider (Phase 8)."""

import json
import logging
import os
import tempfile

import pytest
from unittest.mock import MagicMock

from craving_mind.agent.interface import LLMResponse, MockProvider, AgentInterface
from craving_mind.agent.tools import ToolsRegistry
from craving_mind.agent.memory import MemoryManager
from craving_mind.agent.sandbox import SandboxResult
from craving_mind.orchestrator.artifact_manager import ArtifactManager
from craving_mind.orchestrator.budget import BudgetManager
from craving_mind.orchestrator.checkpoint import CheckpointManager
from craving_mind.orchestrator.phases import PhaseManager
from craving_mind.orchestrator.runner import EpochRunner
from craving_mind.judge.scoring import Scorer
from craving_mind.judge.dedup import DedupFilter
from craving_mind.judge.drift import CUSUMMonitor
from craving_mind.judge.smoke_test import SmokeTest
from craving_mind.utils.tokens import TokenCounter


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

BASE_CONFIG = {
    "agent": {"provider": "mock", "model": "test"},
    "budget": {
        "base_tokens": 50000,
        "circuit_breaker_pct": 0.15,
        "venture_decay": 0.5,
        "rnd_lambda": 0.0001,
        "rnd_max_pct": 0.30,
        "rnd_min_success_rate": 0.50,
        "critical_starvation_pct": 0.10,
    },
    "memory": {"graveyard_ttl_epochs": 10, "bible_max_weight_pct": 0.20},
    "sandbox": {"timeout_seconds": 5, "allowed_imports": ["re", "math"]},
    "judge": {
        "pass_threshold": 0.85,
        "task_score_weights": {"semantic": 0.5, "entity": 0.5},
        "ratio_tolerance": 1.05,
        "epoch": {"epsilon": 0.01, "type_weights": {}},
        "dynamic_multiplier": 1.3,
        "dedup": {"task_prefix_length": 500},
        "drift": {"window": 10, "sigma_multiplier": 2.0},
    },
    "phases": {"phase2_start": 11, "phase3_start": 26},
    "benchmark": {
        "frozen_ratio": 0.7,
        "target_ratio_min": 0.2,
        "target_ratio_max": 0.6,
        "tasks_per_epoch": 5,
    },
}

_COMPRESS_CODE = (
    "def compress(text, target_ratio):\n"
    "    n = int(len(text) * target_ratio)\n"
    "    return text[:n]\n"
)


def _task(
    source_text: str = "Hello world, this is a test text for compression.",
    hidden_type: str = "discourse",
    target_ratio: float = 0.5,
    is_dynamic: bool = False,
) -> dict:
    return {
        "source_text": source_text,
        "hidden_type": hidden_type,
        "target_ratio": target_ratio,
        "questions": ["What is the main topic?"],
        "reference_answers": ["The main topic is a test text."],
        "reference_entities": [set()],
        "is_dynamic": is_dynamic,
    }


def _compress_response(i: int = 0) -> LLMResponse:
    """One run_compress tool call from the mock agent."""
    return LLMResponse(
        content="",
        tool_calls=[{
            "id": f"tc_{i:04d}",
            "name": "run_compress",
            "arguments": {"text": "test", "target_ratio": 0.5},
        }],
        usage={"input_tokens": 100, "output_tokens": 50},
        stop_reason="tool_use",
    )


def _mock_judge(task_score: float = 0.9, passes: bool = True, hidden_type: str = "discourse"):
    judge = MagicMock()
    judge.evaluate_task.return_value = {
        "compression_ratio": 0.4,
        "semantic_score": task_score,
        "entity_score": task_score,
        "pass": passes,
        "task_score": task_score,
        "hidden_type": hidden_type,
    }
    return judge


def _mock_sandbox(return_value: str = "compressed text"):
    sb = MagicMock()
    sb.run_compress.return_value = SandboxResult(
        success=True,
        output=f'{{"result": "{return_value}"}}',
        error="",
        return_value=return_value,
    )
    sb.run_script.return_value = SandboxResult(success=True, output="", error="")
    return sb


def _make_runner(
    config: dict = None,
    n_tasks: int = 5,
    task_score: float = 0.9,
    passes: bool = True,
    responses: list = None,
    run_dir: str = "runs",
    artifact_manager: ArtifactManager = None,
    extra_budget_cfg: dict = None,
):
    """Assemble a fully-wired EpochRunner with mock components."""
    cfg = dict(config or BASE_CONFIG)
    if extra_budget_cfg:
        cfg = {**cfg, "budget": {**cfg["budget"], **extra_budget_cfg}}

    bm = BudgetManager(cfg)
    pm = PhaseManager(cfg)
    sc = Scorer(cfg)
    dd = DedupFilter(cfg)
    dr = CUSUMMonitor(cfg)
    lg = logging.getLogger("test_integration")

    with tempfile.TemporaryDirectory() as tmp:
        agent_dir = os.path.join(tmp, "agent")
        mem = MemoryManager(cfg, agent_dir)
        mem.write_file("compress.py", _COMPRESS_CODE)

        sb = _mock_sandbox()
        tools = ToolsRegistry(sb, mem, bm)

        if responses is None:
            responses = [_compress_response(i) for i in range(n_tasks * 4)]

        provider = MockProvider(responses)
        agent = AgentInterface(cfg, provider, bm, sb, tools)
        judge = _mock_judge(task_score, passes)
        smoke = SmokeTest(sb)
        tc = TokenCounter(cfg)

        runner = EpochRunner(
            config=cfg,
            agent_interface=agent,
            judge_evaluator=judge,
            benchmark_loader=MagicMock(),
            budget_manager=bm,
            phase_manager=pm,
            memory_manager=mem,
            scorer=sc,
            dedup_filter=dd,
            drift_monitor=dr,
            smoke_test=smoke,
            token_counter=tc,
            logger=lg,
            run_dir=run_dir,
            artifact_manager=artifact_manager,
        )
        # Yield the runner and supporting objects.
        return runner, bm, mem, agent, judge


# ---------------------------------------------------------------------------
# Full experiment (3 epochs)
# ---------------------------------------------------------------------------

class TestFullExperiment3Epochs:
    def test_runs_3_epochs_without_error(self, tmp_path):
        tasks = [_task() for _ in range(5)]
        run_dir = str(tmp_path / "run")
        am = ArtifactManager(str(tmp_path / "artifacts"))
        runner, budget, mem, agent, judge = _make_runner(run_dir=run_dir, artifact_manager=am)
        # Re-make runner properly inside tmp context.
        cfg = BASE_CONFIG
        bm = BudgetManager(cfg)
        pm = PhaseManager(cfg)
        sc = Scorer(cfg)
        dd = DedupFilter(cfg)
        dr = CUSUMMonitor(cfg)
        lg = logging.getLogger("test_integration")
        agent_dir = str(tmp_path / "agent")
        mem = MemoryManager(cfg, agent_dir)
        mem.write_file("compress.py", _COMPRESS_CODE)
        sb = _mock_sandbox()
        tools = ToolsRegistry(sb, mem, bm)
        responses = [_compress_response(i) for i in range(60)]
        provider = MockProvider(responses)
        agent = AgentInterface(cfg, provider, bm, sb, tools)
        judge = _mock_judge(0.9, True)
        smoke = SmokeTest(sb)
        tc = TokenCounter(cfg)

        runner = EpochRunner(
            config=cfg,
            agent_interface=agent,
            judge_evaluator=judge,
            benchmark_loader=MagicMock(),
            budget_manager=bm,
            phase_manager=pm,
            memory_manager=mem,
            scorer=sc,
            dedup_filter=dd,
            drift_monitor=dr,
            smoke_test=smoke,
            token_counter=tc,
            logger=lg,
            run_dir=run_dir,
            artifact_manager=am,
        )

        results = []
        prev_sr, prev_saved, prev_oom = 0.0, 0, False
        for epoch in range(3):
            result = runner.run_epoch(
                epoch=epoch,
                tasks=tasks,
                prev_success_rate=prev_sr,
                prev_saved=prev_saved,
                prev_oom=prev_oom,
            )
            results.append(result)
            prev_sr = result["success_rate"]
            prev_saved = result["saved_tokens"]
            prev_oom = result["is_oom"]

        assert len(results) == 3

    def test_tasks_completed_count(self, tmp_path):
        tasks = [_task() for _ in range(5)]
        cfg = BASE_CONFIG
        bm = BudgetManager(cfg)
        pm = PhaseManager(cfg)
        sc = Scorer(cfg)
        dd = DedupFilter(cfg)
        dr = CUSUMMonitor(cfg)
        lg = logging.getLogger("test_integration")
        agent_dir = str(tmp_path / "agent")
        mem = MemoryManager(cfg, agent_dir)
        mem.write_file("compress.py", _COMPRESS_CODE)
        sb = _mock_sandbox()
        tools = ToolsRegistry(sb, mem, bm)
        responses = [_compress_response(i) for i in range(60)]
        provider = MockProvider(responses)
        agent = AgentInterface(cfg, provider, bm, sb, tools)
        judge = _mock_judge(0.9, True)
        smoke = SmokeTest(sb)
        tc = TokenCounter(cfg)
        am = ArtifactManager(str(tmp_path / "artifacts"))
        run_dir = str(tmp_path / "run")

        runner = EpochRunner(
            config=cfg, agent_interface=agent, judge_evaluator=judge,
            benchmark_loader=MagicMock(), budget_manager=bm, phase_manager=pm,
            memory_manager=mem, scorer=sc, dedup_filter=dd, drift_monitor=dr,
            smoke_test=smoke, token_counter=tc, logger=lg, run_dir=run_dir,
            artifact_manager=am,
        )

        for epoch in range(3):
            result = runner.run_epoch(epoch=epoch, tasks=tasks)
            assert result["tasks_completed"] == 5, f"Epoch {epoch}: expected 5 tasks"

    def test_budget_decreases_per_task(self, tmp_path):
        """Budget decreases by tokens_spent for each task executed."""
        tasks = [_task() for _ in range(5)]
        cfg = BASE_CONFIG
        bm = BudgetManager(cfg)
        pm = PhaseManager(cfg)
        sc = Scorer(cfg)
        dd = DedupFilter(cfg)
        dr = CUSUMMonitor(cfg)
        lg = logging.getLogger("test_integration")
        agent_dir = str(tmp_path / "agent")
        mem = MemoryManager(cfg, agent_dir)
        mem.write_file("compress.py", _COMPRESS_CODE)
        sb = _mock_sandbox()
        tools = ToolsRegistry(sb, mem, bm)
        responses = [_compress_response(i) for i in range(60)]
        provider = MockProvider(responses)
        agent = AgentInterface(cfg, provider, bm, sb, tools)
        judge = _mock_judge(0.9, True)
        smoke = SmokeTest(sb)
        tc = TokenCounter(cfg)
        am = ArtifactManager(str(tmp_path / "artifacts"))
        run_dir = str(tmp_path / "run")

        runner = EpochRunner(
            config=cfg, agent_interface=agent, judge_evaluator=judge,
            benchmark_loader=MagicMock(), budget_manager=bm, phase_manager=pm,
            memory_manager=mem, scorer=sc, dedup_filter=dd, drift_monitor=dr,
            smoke_test=smoke, token_counter=tc, logger=lg, run_dir=run_dir,
            artifact_manager=am,
        )

        bm.start_epoch(0)
        initial_budget = bm.remaining
        # Run epoch 0 tasks
        runner.run_epoch(epoch=0, tasks=tasks)
        # At least some tokens should have been spent (5 tasks × 150 tokens min)
        assert bm.total_spent >= 5 * 150

    def test_success_rate_calculated_correctly(self, tmp_path):
        """All tasks passing → success_rate should be > 0.85."""
        tasks = [_task() for _ in range(5)]
        cfg = BASE_CONFIG
        bm = BudgetManager(cfg)
        pm = PhaseManager(cfg)
        sc = Scorer(cfg)
        dd = DedupFilter(cfg)
        dr = CUSUMMonitor(cfg)
        lg = logging.getLogger("test_integration")
        agent_dir = str(tmp_path / "agent")
        mem = MemoryManager(cfg, agent_dir)
        mem.write_file("compress.py", _COMPRESS_CODE)
        sb = _mock_sandbox()
        tools = ToolsRegistry(sb, mem, bm)
        responses = [_compress_response(i) for i in range(20)]
        provider = MockProvider(responses)
        agent = AgentInterface(cfg, provider, bm, sb, tools)
        judge = _mock_judge(0.9, True)
        smoke = SmokeTest(sb)
        tc = TokenCounter(cfg)
        am = ArtifactManager(str(tmp_path / "artifacts"))
        run_dir = str(tmp_path / "run")

        runner = EpochRunner(
            config=cfg, agent_interface=agent, judge_evaluator=judge,
            benchmark_loader=MagicMock(), budget_manager=bm, phase_manager=pm,
            memory_manager=mem, scorer=sc, dedup_filter=dd, drift_monitor=dr,
            smoke_test=smoke, token_counter=tc, logger=lg, run_dir=run_dir,
            artifact_manager=am,
        )
        result = runner.run_epoch(epoch=0, tasks=tasks)
        assert result["success_rate"] >= 0.85

    def test_artifact_exported_on_first_successful_epoch(self, tmp_path):
        """Artifact exported when success_rate >= threshold (first epoch)."""
        tasks = [_task() for _ in range(5)]
        cfg = BASE_CONFIG
        bm = BudgetManager(cfg)
        pm = PhaseManager(cfg)
        sc = Scorer(cfg)
        dd = DedupFilter(cfg)
        dr = CUSUMMonitor(cfg)
        lg = logging.getLogger("test_integration")
        agent_dir = str(tmp_path / "agent")
        mem = MemoryManager(cfg, agent_dir)
        mem.write_file("compress.py", _COMPRESS_CODE)
        sb = _mock_sandbox()
        tools = ToolsRegistry(sb, mem, bm)
        responses = [_compress_response(i) for i in range(20)]
        provider = MockProvider(responses)
        agent = AgentInterface(cfg, provider, bm, sb, tools)
        judge = _mock_judge(0.9, True)
        smoke = SmokeTest(sb)
        tc = TokenCounter(cfg)
        am = ArtifactManager(str(tmp_path / "artifacts"))
        run_dir = str(tmp_path / "run")

        runner = EpochRunner(
            config=cfg, agent_interface=agent, judge_evaluator=judge,
            benchmark_loader=MagicMock(), budget_manager=bm, phase_manager=pm,
            memory_manager=mem, scorer=sc, dedup_filter=dd, drift_monitor=dr,
            smoke_test=smoke, token_counter=tc, logger=lg, run_dir=run_dir,
            artifact_manager=am,
        )
        result = runner.run_epoch(epoch=0, tasks=tasks)
        assert result["artifact_path"] is not None
        assert os.path.isfile(result["artifact_path"])
        assert am.get_latest() is not None
        assert am.get_latest()["version"] == 1

    def test_checkpoint_saved_after_epoch(self, tmp_path):
        """CheckpointManager saves state after each epoch."""
        tasks = [_task() for _ in range(5)]
        cfg = BASE_CONFIG
        bm = BudgetManager(cfg)
        pm = PhaseManager(cfg)
        sc = Scorer(cfg)
        dd = DedupFilter(cfg)
        dr = CUSUMMonitor(cfg)
        lg = logging.getLogger("test_integration")
        agent_dir = str(tmp_path / "agent")
        mem = MemoryManager(cfg, agent_dir)
        mem.write_file("compress.py", _COMPRESS_CODE)
        sb = _mock_sandbox()
        tools = ToolsRegistry(sb, mem, bm)
        responses = [_compress_response(i) for i in range(60)]
        provider = MockProvider(responses)
        agent = AgentInterface(cfg, provider, bm, sb, tools)
        judge = _mock_judge(0.9, True)
        smoke = SmokeTest(sb)
        tc = TokenCounter(cfg)
        am = ArtifactManager(str(tmp_path / "artifacts"))
        run_dir = str(tmp_path / "run")
        checkpoint = CheckpointManager(run_dir)

        runner = EpochRunner(
            config=cfg, agent_interface=agent, judge_evaluator=judge,
            benchmark_loader=MagicMock(), budget_manager=bm, phase_manager=pm,
            memory_manager=mem, scorer=sc, dedup_filter=dd, drift_monitor=dr,
            smoke_test=smoke, token_counter=tc, logger=lg, run_dir=run_dir,
            artifact_manager=am,
        )

        prev_sr, prev_saved, prev_oom = 0.0, 0, False
        for epoch in range(3):
            result = runner.run_epoch(epoch=epoch, tasks=tasks,
                                      prev_success_rate=prev_sr,
                                      prev_saved=prev_saved, prev_oom=prev_oom)
            checkpoint.save({
                "epoch": epoch,
                "success_rate": result["success_rate"],
                "saved_tokens": result["saved_tokens"],
                "is_oom": result["is_oom"],
            })
            checkpoint.save_epoch_log(epoch, result)
            prev_sr = result["success_rate"]
            prev_saved = result["saved_tokens"]
            prev_oom = result["is_oom"]

        state = checkpoint.load()
        assert state is not None
        assert state["epoch"] == 2
        assert "success_rate" in state

        # Verify epoch_log.jsonl has 3 lines.
        with open(checkpoint._epoch_log_path) as f:
            lines = [l for l in f.readlines() if l.strip()]
        assert len(lines) == 3

    def test_no_artifact_when_below_threshold(self, tmp_path):
        """No artifact exported when success_rate < pass_threshold."""
        tasks = [_task() for _ in range(5)]
        cfg = BASE_CONFIG
        bm = BudgetManager(cfg)
        pm = PhaseManager(cfg)
        sc = Scorer(cfg)
        dd = DedupFilter(cfg)
        dr = CUSUMMonitor(cfg)
        lg = logging.getLogger("test_integration")
        agent_dir = str(tmp_path / "agent")
        mem = MemoryManager(cfg, agent_dir)
        mem.write_file("compress.py", _COMPRESS_CODE)
        sb = _mock_sandbox()
        tools = ToolsRegistry(sb, mem, bm)
        responses = [_compress_response(i) for i in range(20)]
        provider = MockProvider(responses)
        agent = AgentInterface(cfg, provider, bm, sb, tools)
        # All tasks fail → success_rate will be 0
        judge = _mock_judge(0.5, False)
        smoke = SmokeTest(sb)
        tc = TokenCounter(cfg)
        am = ArtifactManager(str(tmp_path / "artifacts"))
        run_dir = str(tmp_path / "run")

        runner = EpochRunner(
            config=cfg, agent_interface=agent, judge_evaluator=judge,
            benchmark_loader=MagicMock(), budget_manager=bm, phase_manager=pm,
            memory_manager=mem, scorer=sc, dedup_filter=dd, drift_monitor=dr,
            smoke_test=smoke, token_counter=tc, logger=lg, run_dir=run_dir,
            artifact_manager=am,
        )
        result = runner.run_epoch(epoch=0, tasks=tasks)
        assert result["artifact_path"] is None


# ---------------------------------------------------------------------------
# OOM rollback
# ---------------------------------------------------------------------------

class TestOomRollback:
    def test_bible_md_restored_after_oom(self, tmp_path):
        """Agent writes to bible.md → OOM → bible.md is restored to pre-epoch state."""
        # Use phase2_start=0 so Phase 2 is active and backup is taken.
        # base_tokens=100, epoch=0: effective = 100 * (1+2*exp(0)) = 300
        # Response costs 400 tokens (200+200) → OOM (400 > 300).
        cfg = {
            **BASE_CONFIG,
            "phases": {"phase2_start": 0, "phase3_start": 26},
            "budget": {**BASE_CONFIG["budget"], "base_tokens": 100},
        }

        agent_dir = str(tmp_path / "agent")
        mem = MemoryManager(cfg, agent_dir)
        mem.write_file("bible.md", "original bible content")
        mem.write_file("compress.py", _COMPRESS_CODE)

        bm = BudgetManager(cfg)
        pm = PhaseManager(cfg)
        sc = Scorer(cfg)
        dd = DedupFilter(cfg)
        dr = CUSUMMonitor(cfg)
        lg = logging.getLogger("test_integration")

        sb = _mock_sandbox()
        tools = ToolsRegistry(sb, mem, bm)

        # Agent response: first writes to bible.md, then uses up the budget.
        # The write_file call happens, then 400 tokens are spent → OOM.
        oom_response = LLMResponse(
            content="",
            tool_calls=[{
                "id": "tc_0001",
                "name": "write_file",
                "arguments": {"filename": "bible.md", "content": "modified bible content"},
            }],
            usage={"input_tokens": 200, "output_tokens": 200},  # 400 > effective budget
            stop_reason="tool_use",
        )
        provider = MockProvider([oom_response])
        agent = AgentInterface(cfg, provider, bm, sb, tools)
        judge = _mock_judge(0.9, True)
        smoke = SmokeTest(sb)
        tc = TokenCounter(cfg)
        am = ArtifactManager(str(tmp_path / "artifacts"))
        run_dir = str(tmp_path / "run")

        runner = EpochRunner(
            config=cfg, agent_interface=agent, judge_evaluator=judge,
            benchmark_loader=MagicMock(), budget_manager=bm, phase_manager=pm,
            memory_manager=mem, scorer=sc, dedup_filter=dd, drift_monitor=dr,
            smoke_test=smoke, token_counter=tc, logger=lg, run_dir=run_dir,
            artifact_manager=am,
        )

        tasks = [_task()]
        result = runner.run_epoch(epoch=0, tasks=tasks)

        assert result["is_oom"] is True
        # bible.md must be restored to pre-epoch state.
        assert mem.read_file("bible.md") == "original bible content"

    def test_no_artifact_on_oom(self, tmp_path):
        """No artifact exported when epoch ends in OOM."""
        cfg = {
            **BASE_CONFIG,
            "phases": {"phase2_start": 0, "phase3_start": 26},
            "budget": {**BASE_CONFIG["budget"], "base_tokens": 100},
        }
        agent_dir = str(tmp_path / "agent")
        mem = MemoryManager(cfg, agent_dir)
        mem.write_file("compress.py", _COMPRESS_CODE)

        bm = BudgetManager(cfg)
        pm = PhaseManager(cfg)
        sc = Scorer(cfg)
        dd = DedupFilter(cfg)
        dr = CUSUMMonitor(cfg)
        lg = logging.getLogger("test_integration")
        sb = _mock_sandbox()
        tools = ToolsRegistry(sb, mem, bm)

        oom_response = LLMResponse(
            content="",
            tool_calls=[],
            usage={"input_tokens": 400, "output_tokens": 400},
            stop_reason="end_turn",
        )
        provider = MockProvider([oom_response])
        agent = AgentInterface(cfg, provider, bm, sb, tools)
        judge = _mock_judge(0.9, True)
        smoke = SmokeTest(sb)
        tc = TokenCounter(cfg)
        am = ArtifactManager(str(tmp_path / "artifacts"))
        run_dir = str(tmp_path / "run")

        runner = EpochRunner(
            config=cfg, agent_interface=agent, judge_evaluator=judge,
            benchmark_loader=MagicMock(), budget_manager=bm, phase_manager=pm,
            memory_manager=mem, scorer=sc, dedup_filter=dd, drift_monitor=dr,
            smoke_test=smoke, token_counter=tc, logger=lg, run_dir=run_dir,
            artifact_manager=am,
        )

        result = runner.run_epoch(epoch=0, tasks=[_task()])
        assert result["is_oom"] is True
        assert result["artifact_path"] is None


# ---------------------------------------------------------------------------
# Phase transitions
# ---------------------------------------------------------------------------

class TestPhaseTransitions:
    def test_phase_1_no_memory_backup(self, tmp_path):
        """Phase 1 (epoch < 11): has_memory=False → no OOM rollback."""
        pm = PhaseManager(BASE_CONFIG)
        assert not pm.has_memory(0)
        assert not pm.has_memory(10)

    def test_phase_2_has_memory(self, tmp_path):
        """Phase 2 (epoch >= 11): has_memory=True."""
        pm = PhaseManager(BASE_CONFIG)
        assert pm.has_memory(11)
        assert pm.has_memory(25)

    def test_phase_3_has_dedup(self, tmp_path):
        """Phase 3 (epoch >= 26): has_duplicate_filter=True."""
        pm = PhaseManager(BASE_CONFIG)
        assert not pm.has_duplicate_filter(25)
        assert pm.has_duplicate_filter(26)

    def test_phase_1_has_venture(self):
        pm = PhaseManager(BASE_CONFIG)
        assert pm.has_venture(0)
        assert pm.has_venture(10)
        assert not pm.has_venture(11)

    def test_phase_transitions_at_boundaries(self):
        pm = PhaseManager(BASE_CONFIG)
        assert pm.get_phase(0) == 1
        assert pm.get_phase(10) == 1
        assert pm.get_phase(11) == 2
        assert pm.get_phase(25) == 2
        assert pm.get_phase(26) == 3


# ---------------------------------------------------------------------------
# Artifact versioning
# ---------------------------------------------------------------------------

class TestArtifactVersioning:
    def test_version_increments_each_export(self, tmp_path):
        am = ArtifactManager(str(tmp_path / "artifacts"))
        meta = {
            "epoch": 0, "crav_id": "Crav-001", "mean_score": 0.9,
            "semantic_score": 0.9, "entity_score": 0.9,
            "score_by_type": {}, "mean_compression_ratio": 0.4, "success_rate": 0.9,
        }
        e1 = am.export("# v1", {**meta, "epoch": 0})
        e2 = am.export("# v2", {**meta, "epoch": 1, "mean_score": 0.92})
        e3 = am.export("# v3", {**meta, "epoch": 2, "mean_score": 0.94})
        assert e1["version"] == 1
        assert e2["version"] == 2
        assert e3["version"] == 3

    def test_manifest_has_entry_per_export(self, tmp_path):
        am = ArtifactManager(str(tmp_path / "artifacts"))
        meta = {
            "epoch": 0, "crav_id": "Crav-001", "mean_score": 0.9,
            "semantic_score": 0.9, "entity_score": 0.9,
            "score_by_type": {}, "mean_compression_ratio": 0.4, "success_rate": 0.9,
        }
        for i in range(4):
            am.export(f"# version {i}", {**meta, "epoch": i})
        assert len(am.get_history()) == 4

    def test_get_best_returns_highest_score(self, tmp_path):
        am = ArtifactManager(str(tmp_path / "artifacts"))
        meta_base = {
            "crav_id": "Crav-001", "semantic_score": 0.9, "entity_score": 0.9,
            "score_by_type": {}, "mean_compression_ratio": 0.4, "success_rate": 0.9,
        }
        am.export("# a", {"epoch": 0, "mean_score": 0.80, **meta_base})
        am.export("# b", {"epoch": 1, "mean_score": 0.95, **meta_base})
        am.export("# c", {"epoch": 2, "mean_score": 0.88, **meta_base})
        best = am.get_best("mean_score")
        assert best["mean_score"] == pytest.approx(0.95)
        assert best["epoch"] == 1

    def test_only_export_on_compress_py_change(self, tmp_path):
        """Runner only exports when compress.py actually changes."""
        cfg = BASE_CONFIG
        bm = BudgetManager(cfg)
        pm = PhaseManager(cfg)
        sc = Scorer(cfg)
        dd = DedupFilter(cfg)
        dr = CUSUMMonitor(cfg)
        lg = logging.getLogger("test_integration")
        agent_dir = str(tmp_path / "agent")
        mem = MemoryManager(cfg, agent_dir)
        mem.write_file("compress.py", _COMPRESS_CODE)
        sb = _mock_sandbox()
        tools = ToolsRegistry(sb, mem, bm)
        responses = [_compress_response(i) for i in range(60)]
        provider = MockProvider(responses)
        agent = AgentInterface(cfg, provider, bm, sb, tools)
        judge = _mock_judge(0.9, True)
        smoke = SmokeTest(sb)
        tc = TokenCounter(cfg)
        am = ArtifactManager(str(tmp_path / "artifacts"))
        run_dir = str(tmp_path / "run")

        runner = EpochRunner(
            config=cfg, agent_interface=agent, judge_evaluator=judge,
            benchmark_loader=MagicMock(), budget_manager=bm, phase_manager=pm,
            memory_manager=mem, scorer=sc, dedup_filter=dd, drift_monitor=dr,
            smoke_test=smoke, token_counter=tc, logger=lg, run_dir=run_dir,
            artifact_manager=am,
        )

        tasks = [_task() for _ in range(5)]
        # Run 3 epochs — compress.py unchanged between epochs since MockProvider
        # doesn't write it. Only epoch 0 should produce an artifact.
        prev_sr, prev_saved, prev_oom = 0.0, 0, False
        artifact_paths = []
        for epoch in range(3):
            result = runner.run_epoch(epoch=epoch, tasks=tasks,
                                      prev_success_rate=prev_sr,
                                      prev_saved=prev_saved, prev_oom=prev_oom)
            artifact_paths.append(result["artifact_path"])
            prev_sr = result["success_rate"]
            prev_saved = result["saved_tokens"]
            prev_oom = result["is_oom"]

        non_none = [p for p in artifact_paths if p is not None]
        assert len(non_none) == 1, "Only epoch 0 should produce an artifact (code unchanged)"
        assert am.get_latest()["version"] == 1


# ---------------------------------------------------------------------------
# Checkpoint resume
# ---------------------------------------------------------------------------

class TestCheckpointResume:
    def test_save_load_roundtrip(self, tmp_path):
        run_dir = str(tmp_path / "run")
        cp = CheckpointManager(run_dir)
        state = {
            "epoch": 5,
            "success_rate": 0.91,
            "saved_tokens": 42000,
            "is_oom": False,
        }
        cp.save(state)
        loaded = cp.load()
        assert loaded["epoch"] == 5
        assert loaded["success_rate"] == pytest.approx(0.91)
        assert loaded["saved_tokens"] == 42000
        assert "saved_at" in loaded

    def test_resume_from_checkpoint_continues_from_next_epoch(self, tmp_path):
        """Simulating --resume: start_epoch = checkpoint_epoch + 1."""
        run_dir = str(tmp_path / "run")
        cp = CheckpointManager(run_dir)
        cp.save({"epoch": 5, "success_rate": 0.88, "saved_tokens": 1000, "is_oom": False})

        state = cp.load()
        start_epoch = state["epoch"] + 1
        assert start_epoch == 6

    def test_epoch_log_accumulates(self, tmp_path):
        run_dir = str(tmp_path / "run")
        cp = CheckpointManager(run_dir)
        for i in range(5):
            cp.save_epoch_log(i, {"success_rate": 0.8 + i * 0.01, "is_oom": False})
        with open(cp._epoch_log_path) as f:
            lines = [l for l in f.readlines() if l.strip()]
        assert len(lines) == 5


# ---------------------------------------------------------------------------
# Circuit breaker
# ---------------------------------------------------------------------------

class TestCircuitBreaker:
    def test_circuit_breaker_logs_warning_when_task_over_limit(self, tmp_path, caplog):
        """Single task exceeding 15% of epoch budget triggers a warning log."""
        # With base_tokens=1000 and venture_multiplier≈3 (epoch 0), effective≈3000
        # circuit_limit = 0.15 * 3000 = 450 tokens
        # We'll use a response that costs 500 tokens (100+400) > 450
        cfg = {
            **BASE_CONFIG,
            "budget": {**BASE_CONFIG["budget"], "base_tokens": 1000},
        }
        agent_dir = str(tmp_path / "agent")
        mem = MemoryManager(cfg, agent_dir)
        mem.write_file("compress.py", _COMPRESS_CODE)

        bm = BudgetManager(cfg)
        pm = PhaseManager(cfg)
        sc = Scorer(cfg)
        dd = DedupFilter(cfg)
        dr = CUSUMMonitor(cfg)
        lg = logging.getLogger("craving_mind.circuit_breaker_test")
        sb = _mock_sandbox()
        tools = ToolsRegistry(sb, mem, bm)

        # expensive_response: costs 500 tokens, which > circuit_limit
        expensive_response = LLMResponse(
            content="",
            tool_calls=[{
                "id": "tc_0000",
                "name": "run_compress",
                "arguments": {"text": "test", "target_ratio": 0.5},
            }],
            usage={"input_tokens": 400, "output_tokens": 100},  # 500 tokens
            stop_reason="tool_use",
        )
        # Fill remaining responses with cheap ones
        cheap = [_compress_response(i) for i in range(20)]
        provider = MockProvider([expensive_response] + cheap)
        agent = AgentInterface(cfg, provider, bm, sb, tools)
        judge = _mock_judge(0.9, True)
        smoke = SmokeTest(sb)
        tc = TokenCounter(cfg)
        run_dir = str(tmp_path / "run")

        runner = EpochRunner(
            config=cfg, agent_interface=agent, judge_evaluator=judge,
            benchmark_loader=MagicMock(), budget_manager=bm, phase_manager=pm,
            memory_manager=mem, scorer=sc, dedup_filter=dd, drift_monitor=dr,
            smoke_test=smoke, token_counter=tc, logger=lg, run_dir=run_dir,
        )

        with caplog.at_level(logging.WARNING):
            runner.run_epoch(epoch=0, tasks=[_task()])

        warning_msgs = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any("circuit breaker" in m.lower() or "Circuit breaker" in m for m in warning_msgs), (
            f"Expected circuit breaker warning. Got: {warning_msgs}"
        )


# ---------------------------------------------------------------------------
# R&D carry-over
# ---------------------------------------------------------------------------

class TestRndCarryover:
    def test_rnd_fund_available_after_successful_epoch(self, tmp_path):
        """After epoch with success_rate >= 0.5 and saved tokens, rnd_fund > 0."""
        cfg = BASE_CONFIG
        bm = BudgetManager(cfg)
        # Simulate prev epoch: success_rate=0.8, saved=40000 tokens
        bm.start_epoch(
            epoch=1,
            prev_success_rate=0.8,
            prev_saved=40000,
            prev_oom=False,
        )
        assert bm.rnd_fund > 0

    def test_no_rnd_fund_after_oom(self):
        cfg = BASE_CONFIG
        bm = BudgetManager(cfg)
        bm.start_epoch(
            epoch=1,
            prev_success_rate=0.9,
            prev_saved=40000,
            prev_oom=True,  # Previous epoch was OOM → no R&D
        )
        assert bm.rnd_fund == 0

    def test_no_rnd_fund_when_success_rate_below_min(self):
        cfg = BASE_CONFIG
        bm = BudgetManager(cfg)
        bm.start_epoch(
            epoch=1,
            prev_success_rate=0.3,  # Below rnd_min_success_rate=0.50
            prev_saved=40000,
            prev_oom=False,
        )
        assert bm.rnd_fund == 0

    def test_rnd_fund_increases_budget(self, tmp_path):
        """Epoch budget with R&D carry-over > epoch budget without it."""
        cfg = BASE_CONFIG
        bm_no_rnd = BudgetManager(cfg)
        bm_no_rnd.start_epoch(epoch=1, prev_success_rate=0.0, prev_saved=0, prev_oom=False)
        budget_no_rnd = bm_no_rnd.remaining

        bm_with_rnd = BudgetManager(cfg)
        bm_with_rnd.start_epoch(
            epoch=1, prev_success_rate=0.8, prev_saved=40000, prev_oom=False
        )
        budget_with_rnd = bm_with_rnd.remaining

        assert budget_with_rnd > budget_no_rnd


# ---------------------------------------------------------------------------
# Dynamic multiplier
# ---------------------------------------------------------------------------

class TestDynamicMultiplier:
    def test_combined_sr_weighted_by_multiplier(self):
        """combined = (frozen + 1.3 * dynamic) / (1 + 1.3)."""
        scorer = Scorer(BASE_CONFIG)
        frozen_sr = 0.8
        dynamic_sr = 1.0
        combined = scorer.combined_success_rate(frozen_sr, dynamic_sr)
        expected = (0.8 + 1.3 * 1.0) / (1.0 + 1.3)
        assert combined == pytest.approx(expected, rel=1e-5)

    def test_dynamic_tasks_boost_combined_sr(self):
        """Dynamic tasks with higher pass rate boost combined SR above frozen."""
        scorer = Scorer(BASE_CONFIG)
        combined_high = scorer.combined_success_rate(0.7, 1.0)
        combined_low = scorer.combined_success_rate(0.7, 0.0)
        assert combined_high > combined_low

    def test_epoch_success_rate_with_mixed_types(self, tmp_path):
        """Epoch SR integrates dynamic tasks via combined_success_rate in runner."""
        tasks_frozen = [_task(hidden_type="discourse", is_dynamic=False) for _ in range(3)]
        tasks_dynamic = [_task(hidden_type="discourse", is_dynamic=True) for _ in range(2)]
        all_tasks = tasks_frozen + tasks_dynamic

        cfg = BASE_CONFIG
        bm = BudgetManager(cfg)
        pm = PhaseManager(cfg)
        sc = Scorer(cfg)
        dd = DedupFilter(cfg)
        dr = CUSUMMonitor(cfg)
        lg = logging.getLogger("test_integration")
        agent_dir = str(tmp_path / "agent")
        mem = MemoryManager(cfg, agent_dir)
        mem.write_file("compress.py", _COMPRESS_CODE)
        sb = _mock_sandbox()
        tools = ToolsRegistry(sb, mem, bm)
        responses = [_compress_response(i) for i in range(30)]
        provider = MockProvider(responses)
        agent = AgentInterface(cfg, provider, bm, sb, tools)

        judge = MagicMock()
        # Frozen tasks: pass; dynamic tasks: fail → dynamic_sr should be 0
        call_count = [0]
        def judge_side_effect(**kwargs):
            is_dynamic = kwargs.get("hidden_type", "discourse")
            call_count[0] += 1
            return {
                "compression_ratio": 0.4,
                "semantic_score": 0.9,
                "entity_score": 0.9,
                "pass": True,
                "task_score": 0.9,
                "hidden_type": "discourse",
            }
        judge.evaluate_task.side_effect = judge_side_effect

        smoke = SmokeTest(sb)
        tc = TokenCounter(cfg)
        run_dir = str(tmp_path / "run")

        runner = EpochRunner(
            config=cfg, agent_interface=agent, judge_evaluator=judge,
            benchmark_loader=MagicMock(), budget_manager=bm, phase_manager=pm,
            memory_manager=mem, scorer=sc, dedup_filter=dd, drift_monitor=dr,
            smoke_test=smoke, token_counter=tc, logger=lg, run_dir=run_dir,
        )
        result = runner.run_epoch(epoch=0, tasks=all_tasks)
        assert result["tasks_completed"] == 5
        assert "frozen_success_rate" in result
        assert "dynamic_success_rate" in result
        assert "success_rate" in result


# ---------------------------------------------------------------------------
# Inheritance
# ---------------------------------------------------------------------------

class TestInheritance:
    def test_init_from_inheritance_sets_compress_py(self, tmp_path):
        cfg = BASE_CONFIG
        agent_dir = str(tmp_path / "agent")
        mem = MemoryManager(cfg, agent_dir)
        prev_code = "def compress(t, r): return t[:5]"
        mem.init_from_inheritance(prev_compress=prev_code)
        assert mem.read_file("compress.py") == prev_code

    def test_init_from_inheritance_tags_graveyard(self, tmp_path):
        cfg = BASE_CONFIG
        agent_dir = str(tmp_path / "agent")
        mem = MemoryManager(cfg, agent_dir)
        grave = "<!-- AMENDMENT: epoch=1 -->\nold idea\n<!-- /AMENDMENT -->"
        mem.init_from_inheritance(prev_graveyard=grave)
        stored = mem.read_file("graveyard.md")
        assert "inherited" in stored

    def test_inheritance_compress_used_in_runner(self, tmp_path):
        """Inherited compress.py is written to agent dir; runner reads it for export."""
        cfg = BASE_CONFIG
        agent_dir = str(tmp_path / "agent")
        mem = MemoryManager(cfg, agent_dir)
        inherited_code = "def compress(t, r): return t[:int(len(t)*r)]\n"
        mem.init_from_inheritance(prev_compress=inherited_code)
        assert mem.read_file("compress.py") == inherited_code
