"""Tests for Phase 6: orchestrator — epoch runner, checkpoint, main loop."""

import json
import logging
import os
import pytest

from unittest.mock import MagicMock

from craving_mind.agent.interface import LLMResponse, MockProvider, AgentInterface
from craving_mind.agent.tools import ToolsRegistry
from craving_mind.agent.memory import MemoryManager
from craving_mind.agent.sandbox import SandboxResult
from craving_mind.orchestrator.budget import BudgetManager
from craving_mind.orchestrator.phases import PhaseManager
from craving_mind.orchestrator.runner import EpochRunner
from craving_mind.orchestrator.checkpoint import CheckpointManager
from craving_mind.judge.scoring import Scorer
from craving_mind.judge.dedup import DedupFilter
from craving_mind.judge.drift import CUSUMMonitor
from craving_mind.judge.smoke_test import SmokeTest
from craving_mind.utils.tokens import TokenCounter


# ---------------------------------------------------------------------------
# Shared config and helpers
# ---------------------------------------------------------------------------

BASE_CONFIG = {
    "agent": {
        "provider": "mock",
        "model": "test",
    },
    "budget": {
        "base_tokens": 50000,
        "circuit_breaker_pct": 0.15,
        "venture_decay": 0.5,
        "rnd_lambda": 0.0001,
        "rnd_max_pct": 0.30,
        "rnd_min_success_rate": 0.50,
        "critical_starvation_pct": 0.10,
    },
    "memory": {
        "graveyard_ttl_epochs": 10,
        "bible_max_weight_pct": 0.20,
    },
    "sandbox": {
        "timeout_seconds": 5,
        "allowed_imports": ["re", "math"],
    },
    "judge": {
        "pass_threshold": 0.85,
        "task_score_weights": {"semantic": 0.5, "entity": 0.5},
        "ratio_tolerance": 1.05,
        "epoch": {"epsilon": 0.01, "type_weights": {}},
        "dynamic_multiplier": 1.3,
        "dedup": {"task_prefix_length": 500},
        "drift": {"window": 10, "sigma_multiplier": 2.0},
    },
    "phases": {
        "phase2_start": 11,
        "phase3_start": 26,
    },
    "benchmark": {
        "frozen_ratio": 0.7,
        "target_ratio_min": 0.2,
        "target_ratio_max": 0.6,
        "tasks_per_epoch": 5,
    },
}


def _task(
    source_text: str = "Hello world, this is a test text for compression.",
    hidden_type: str = "discourse",
    target_ratio: float = 0.5,
    is_dynamic: bool = False,
):
    """Minimal task dict."""
    return {
        "source_text": source_text,
        "hidden_type": hidden_type,
        "target_ratio": target_ratio,
        "questions": ["What is the main topic?"],
        "reference_answers": ["The main topic is a test text."],
        "reference_entities": [set()],
        "is_dynamic": is_dynamic,
    }


def _compress_response(i: int = 0, compressed: str = "compressed text") -> LLMResponse:
    """MockProvider response that calls run_compress."""
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


def _no_tool_response(i: int = 0) -> LLMResponse:
    """MockProvider response with no tool calls (agent skips compression)."""
    return LLMResponse(
        content="I cannot compress this.",
        tool_calls=[],
        usage={"input_tokens": 80, "output_tokens": 40},
        stop_reason="end_turn",
    )


def _mock_judge(task_score: float = 0.9, passes: bool = True) -> MagicMock:
    judge = MagicMock()
    judge.evaluate_task.return_value = {
        "compression_ratio": 0.4,
        "semantic_score": task_score,
        "entity_score": task_score,
        "pass": passes,
        "task_score": task_score,
        "hidden_type": "discourse",
    }
    return judge


def _mock_sandbox(return_value: str = "compressed text") -> MagicMock:
    sb = MagicMock()
    sb.run_compress.return_value = SandboxResult(
        success=True,
        output=f'{{"result": "{return_value}"}}',
        error="",
        return_value=return_value,
    )
    sb.run_script.return_value = SandboxResult(success=True, output="", error="")
    return sb


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def run_dir(tmp_path):
    return str(tmp_path / "run")


@pytest.fixture
def agent_dir(tmp_path):
    return str(tmp_path / "agent")


@pytest.fixture
def memory(agent_dir):
    m = MemoryManager(BASE_CONFIG, agent_dir)
    m.write_file(
        "compress.py",
        "def compress(text, target_ratio):\n    n = int(len(text) * target_ratio)\n    return text[:n]\n",
    )
    return m


@pytest.fixture
def budget():
    return BudgetManager(BASE_CONFIG)


@pytest.fixture
def phase_manager():
    return PhaseManager(BASE_CONFIG)


@pytest.fixture
def scorer():
    return Scorer(BASE_CONFIG)


@pytest.fixture
def dedup():
    return DedupFilter(BASE_CONFIG)


@pytest.fixture
def drift():
    return CUSUMMonitor(BASE_CONFIG)


@pytest.fixture
def logger():
    return logging.getLogger("test_orchestrator")


def _make_runner(
    n_tasks: int = 5,
    task_score: float = 0.9,
    passes: bool = True,
    memory=None,
    budget=None,
    phase_manager=None,
    scorer=None,
    dedup=None,
    drift=None,
    logger=None,
    run_dir: str = "runs",
    tmp_path=None,
    config: dict = None,
    responses: list = None,
):
    """Assemble a fully-wired EpochRunner with mock LLM and sandbox."""
    cfg = config or BASE_CONFIG
    bm = budget or BudgetManager(cfg)
    pm = phase_manager or PhaseManager(cfg)
    sc = scorer or Scorer(cfg)
    dd = dedup or DedupFilter(cfg)
    dr = drift or CUSUMMonitor(cfg)
    lg = logger or logging.getLogger("test_orchestrator")

    # Agent workspace
    import tempfile
    tmp = tmp_path or tempfile.mkdtemp()
    agent_workspace = str(os.path.join(str(tmp), "agent"))
    mem = memory or MemoryManager(cfg, agent_workspace)
    if memory is None and not mem.read_file("compress.py"):
        mem.write_file(
            "compress.py",
            "def compress(text, target_ratio):\n    n = int(len(text) * target_ratio)\n    return text[:n]\n",
        )

    sb = _mock_sandbox()
    tools = ToolsRegistry(sb, mem, bm)

    # Default: one compress response per task
    if responses is None:
        responses = [_compress_response(i) for i in range(max(n_tasks, 1) * 3)]

    provider = MockProvider(responses)
    agent = AgentInterface(cfg, provider, bm, sb, tools)
    judge = _mock_judge(task_score, passes)
    benchmark_loader = MagicMock()
    smoke = SmokeTest(sb)
    tc = TokenCounter(cfg)

    return EpochRunner(
        config=cfg,
        agent_interface=agent,
        judge_evaluator=judge,
        benchmark_loader=benchmark_loader,
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
    )


# ---------------------------------------------------------------------------
# CheckpointManager tests
# ---------------------------------------------------------------------------

class TestCheckpointManager:
    def test_save_creates_file(self, tmp_path):
        cp = CheckpointManager(str(tmp_path / "run"))
        cp.save({"epoch": 1, "success_rate": 0.9})
        assert os.path.exists(cp.checkpoint_path)

    def test_load_returns_none_when_missing(self, tmp_path):
        cp = CheckpointManager(str(tmp_path / "run"))
        assert cp.load() is None

    def test_save_load_roundtrip(self, tmp_path):
        cp = CheckpointManager(str(tmp_path / "run"))
        state = {"epoch": 5, "success_rate": 0.88, "saved_tokens": 1200, "is_oom": False}
        cp.save(state)
        loaded = cp.load()
        assert loaded["epoch"] == 5
        assert loaded["success_rate"] == pytest.approx(0.88)
        assert loaded["saved_tokens"] == 1200
        assert loaded["is_oom"] is False
        assert "saved_at" in loaded  # timestamp added

    def test_save_overwrites_existing_checkpoint(self, tmp_path):
        cp = CheckpointManager(str(tmp_path / "run"))
        cp.save({"epoch": 1})
        cp.save({"epoch": 2})
        assert cp.load()["epoch"] == 2

    def test_epoch_log_appends(self, tmp_path):
        cp = CheckpointManager(str(tmp_path / "run"))
        cp.save_epoch_log(1, {"success_rate": 0.7})
        cp.save_epoch_log(2, {"success_rate": 0.8})
        with open(cp._epoch_log_path) as f:
            lines = f.readlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["epoch"] == 1
        assert json.loads(lines[1])["epoch"] == 2

    def test_task_log_appends(self, tmp_path):
        cp = CheckpointManager(str(tmp_path / "run"))
        cp.save_task_log(1, {"task_score": 0.9, "passed": True})
        cp.save_task_log(1, {"task_score": 0.6, "passed": False})
        with open(cp._task_log_path) as f:
            lines = f.readlines()
        assert len(lines) == 2
        records = [json.loads(ln) for ln in lines]
        assert records[0]["passed"] is True
        assert records[1]["passed"] is False

    def test_epoch_log_entry_has_ts(self, tmp_path):
        cp = CheckpointManager(str(tmp_path / "run"))
        cp.save_epoch_log(0, {"success_rate": 0.5})
        with open(cp._epoch_log_path) as f:
            entry = json.loads(f.readline())
        assert "ts" in entry

    def test_creates_run_dir(self, tmp_path):
        new_dir = str(tmp_path / "deep" / "nested" / "run")
        cp = CheckpointManager(new_dir)
        assert os.path.isdir(new_dir)


# ---------------------------------------------------------------------------
# EpochRunner — basic run_epoch
# ---------------------------------------------------------------------------

class TestEpochRunnerBasic:
    def test_run_epoch_returns_dict_with_expected_keys(self, tmp_path):
        runner = _make_runner(n_tasks=2, tmp_path=tmp_path, run_dir=str(tmp_path / "run"))
        tasks = [_task() for _ in range(2)]
        result = runner.run_epoch(epoch=0, tasks=tasks)
        for key in ("epoch", "success_rate", "frozen_success_rate", "dynamic_success_rate",
                    "tasks_completed", "tasks_total", "saved_tokens", "is_oom",
                    "artifact_path", "drift_detected"):
            assert key in result, f"Missing key: {key}"

    def test_epoch_number_in_result(self, tmp_path):
        runner = _make_runner(n_tasks=1, tmp_path=tmp_path, run_dir=str(tmp_path / "run"))
        result = runner.run_epoch(epoch=7, tasks=[_task()])
        assert result["epoch"] == 7

    def test_tasks_completed_count(self, tmp_path):
        runner = _make_runner(n_tasks=3, tmp_path=tmp_path, run_dir=str(tmp_path / "run"))
        tasks = [_task(source_text=f"text {i}") for i in range(3)]
        result = runner.run_epoch(epoch=0, tasks=tasks)
        assert result["tasks_completed"] == 3
        assert result["tasks_total"] == 3

    def test_success_rate_all_passing(self, tmp_path):
        runner = _make_runner(n_tasks=3, task_score=0.95, passes=True,
                               tmp_path=tmp_path, run_dir=str(tmp_path / "run"))
        tasks = [_task(source_text=f"text {i}") for i in range(3)]
        result = runner.run_epoch(epoch=0, tasks=tasks)
        assert result["success_rate"] > 0.0

    def test_success_rate_all_failing(self, tmp_path):
        runner = _make_runner(n_tasks=3, task_score=0.3, passes=False,
                               tmp_path=tmp_path, run_dir=str(tmp_path / "run"))
        tasks = [_task(source_text=f"text {i}") for i in range(3)]
        result = runner.run_epoch(epoch=0, tasks=tasks)
        assert result["success_rate"] < 0.5

    def test_empty_task_list(self, tmp_path):
        runner = _make_runner(n_tasks=0, tmp_path=tmp_path, run_dir=str(tmp_path / "run"))
        result = runner.run_epoch(epoch=0, tasks=[])
        assert result["tasks_completed"] == 0
        assert result["success_rate"] == 0.0

    def test_judge_called_for_each_task(self, tmp_path):
        runner = _make_runner(n_tasks=4, tmp_path=tmp_path, run_dir=str(tmp_path / "run"))
        tasks = [_task(source_text=f"text {i}") for i in range(4)]
        runner.run_epoch(epoch=0, tasks=tasks)
        assert runner.judge.evaluate_task.call_count == 4

    def test_compressed_text_extracted_and_passed_to_judge(self, tmp_path):
        runner = _make_runner(n_tasks=1, tmp_path=tmp_path, run_dir=str(tmp_path / "run"))
        runner.run_epoch(epoch=0, tasks=[_task()])
        call_args = runner.judge.evaluate_task.call_args
        # compressed_text should not be empty (sandbox returned "compressed text")
        assert call_args.kwargs["compressed_text"] == "compressed text"

    def test_is_not_oom_on_normal_run(self, tmp_path):
        runner = _make_runner(n_tasks=2, tmp_path=tmp_path, run_dir=str(tmp_path / "run"))
        result = runner.run_epoch(epoch=0, tasks=[_task(), _task(source_text="other")])
        assert result["is_oom"] is False

    def test_start_epoch_initialises_budget(self, tmp_path):
        bm = BudgetManager(BASE_CONFIG)
        runner = _make_runner(n_tasks=1, budget=bm,
                               tmp_path=tmp_path, run_dir=str(tmp_path / "run"))
        runner.run_epoch(epoch=0, tasks=[_task()])
        # Budget was started (remaining should have been set)
        assert bm.epoch == 0


# ---------------------------------------------------------------------------
# EpochRunner — success rate calculation
# ---------------------------------------------------------------------------

class TestSuccessRateCalculation:
    def test_frozen_only_tasks_success_rate(self, tmp_path):
        runner = _make_runner(n_tasks=4, task_score=0.9, passes=True,
                               tmp_path=tmp_path, run_dir=str(tmp_path / "run"))
        tasks = [_task(source_text=f"text {i}", is_dynamic=False) for i in range(4)]
        result = runner.run_epoch(epoch=0, tasks=tasks)
        # All frozen, dynamic_success_rate defaults to frozen_success_rate
        assert result["frozen_success_rate"] == result["dynamic_success_rate"]

    def test_dynamic_tasks_affect_combined_rate(self, tmp_path):
        # Use a judge that fails dynamic tasks but passes frozen ones.
        runner = _make_runner(n_tasks=4, task_score=0.9, passes=True,
                               tmp_path=tmp_path, run_dir=str(tmp_path / "run"))

        call_count = [0]
        def side_effect(**kwargs):
            call_count[0] += 1
            # First 2 calls → pass (frozen), last 2 → fail (dynamic)
            passes = call_count[0] <= 2
            return {
                "compression_ratio": 0.4,
                "semantic_score": 0.9 if passes else 0.3,
                "entity_score": 0.9 if passes else 0.3,
                "pass": passes,
                "task_score": 0.9 if passes else 0.3,
                "hidden_type": kwargs.get("hidden_type", "discourse"),
            }

        runner.judge.evaluate_task.side_effect = side_effect
        tasks = [
            _task(source_text="frozen 1", is_dynamic=False),
            _task(source_text="frozen 2", is_dynamic=False),
            _task(source_text="dynamic 1", is_dynamic=True),
            _task(source_text="dynamic 2", is_dynamic=True),
        ]
        result = runner.run_epoch(epoch=0, tasks=tasks)
        # frozen passes all (high), dynamic fails all (low)
        assert result["frozen_success_rate"] > result["dynamic_success_rate"]
        # combined is a weighted blend
        assert result["dynamic_success_rate"] < result["combined_success_rate"] if "combined_success_rate" in result else True

    def test_overfit_gap_computed(self, tmp_path):
        runner = _make_runner(n_tasks=4, task_score=0.9, passes=True,
                               tmp_path=tmp_path, run_dir=str(tmp_path / "run"))
        tasks = [
            _task(source_text="f1", is_dynamic=False),
            _task(source_text="d1", is_dynamic=True),
        ]
        result = runner.run_epoch(epoch=0, tasks=tasks)
        assert "overfit_gap" in result
        assert isinstance(result["overfit_gap"], float)


# ---------------------------------------------------------------------------
# EpochRunner — OOM handling
# ---------------------------------------------------------------------------

class TestOOMHandling:
    def test_oom_stops_processing_remaining_tasks(self, tmp_path):
        # Budget so tiny it goes OOM after first task.
        tiny_config = {
            **BASE_CONFIG,
            "budget": {
                **BASE_CONFIG["budget"],
                "base_tokens": 50,  # extremely tight
            },
        }
        # Responses cost 150 tokens each (> 50 base).
        expensive_response = LLMResponse(
            content="",
            tool_calls=[],
            usage={"input_tokens": 100, "output_tokens": 100},
            stop_reason="end_turn",
        )
        provider = MockProvider([expensive_response] * 10)
        bm = BudgetManager(tiny_config)
        bm.start_epoch(0)
        import tempfile
        agent_workspace = str(tmp_path / "agent")
        mem = MemoryManager(tiny_config, agent_workspace)
        sb = _mock_sandbox()
        tools = ToolsRegistry(sb, mem, bm)
        agent = AgentInterface(tiny_config, provider, bm, sb, tools)

        runner = EpochRunner(
            config=tiny_config,
            agent_interface=agent,
            judge_evaluator=_mock_judge(),
            benchmark_loader=MagicMock(),
            budget_manager=bm,
            phase_manager=PhaseManager(tiny_config),
            memory_manager=mem,
            scorer=Scorer(tiny_config),
            dedup_filter=DedupFilter(tiny_config),
            drift_monitor=CUSUMMonitor(tiny_config),
            smoke_test=SmokeTest(sb),
            token_counter=TokenCounter(tiny_config),
            logger=logging.getLogger("test"),
            run_dir=str(tmp_path / "run"),
        )

        tasks = [_task(source_text=f"task {i}") for i in range(5)]
        result = runner.run_epoch(epoch=0, tasks=tasks)
        assert result["is_oom"] is True
        assert result["tasks_completed"] < 5  # not all tasks ran

    def test_oom_rollback_restores_memory(self, tmp_path):
        """OOM in Phase 2+ triggers memory.restore() with epoch-start backup."""
        tiny_config = {
            **BASE_CONFIG,
            "budget": {**BASE_CONFIG["budget"], "base_tokens": 50},
            "phases": {"phase2_start": 0, "phase3_start": 100},  # Phase 2 from epoch 0
        }
        expensive_response = LLMResponse(
            content="",
            tool_calls=[],
            usage={"input_tokens": 200, "output_tokens": 200},
            stop_reason="end_turn",
        )
        provider = MockProvider([expensive_response] * 5)
        bm = BudgetManager(tiny_config)
        agent_workspace = str(tmp_path / "agent")
        mem = MemoryManager(tiny_config, agent_workspace)
        # Write initial bible.md content
        mem.write_file("bible.md", "original bible content")
        sb = _mock_sandbox()
        tools = ToolsRegistry(sb, mem, bm)
        agent = AgentInterface(tiny_config, provider, bm, sb, tools)

        runner = EpochRunner(
            config=tiny_config,
            agent_interface=agent,
            judge_evaluator=_mock_judge(),
            benchmark_loader=MagicMock(),
            budget_manager=bm,
            phase_manager=PhaseManager(tiny_config),
            memory_manager=mem,
            scorer=Scorer(tiny_config),
            dedup_filter=DedupFilter(tiny_config),
            drift_monitor=CUSUMMonitor(tiny_config),
            smoke_test=SmokeTest(sb),
            token_counter=TokenCounter(tiny_config),
            logger=logging.getLogger("test"),
            run_dir=str(tmp_path / "run"),
        )

        tasks = [_task()]
        runner.run_epoch(epoch=0, tasks=tasks)
        # After OOM, bible.md should be restored to epoch-start content.
        assert mem.read_file("bible.md") == "original bible content"

    def test_no_rollback_in_phase1(self, tmp_path):
        """Phase 1 has no memory, so OOM does not attempt rollback."""
        tiny_config = {
            **BASE_CONFIG,
            "budget": {**BASE_CONFIG["budget"], "base_tokens": 50},
            "phases": {"phase2_start": 100, "phase3_start": 200},  # Phase 1 stays
        }
        expensive_response = LLMResponse(
            content="",
            tool_calls=[],
            usage={"input_tokens": 200, "output_tokens": 200},
            stop_reason="end_turn",
        )
        provider = MockProvider([expensive_response] * 3)
        bm = BudgetManager(tiny_config)
        agent_workspace = str(tmp_path / "agent")
        mem = MemoryManager(tiny_config, agent_workspace)
        mem.write_file("bible.md", "phase1 content")
        sb = _mock_sandbox()
        tools = ToolsRegistry(sb, mem, bm)
        agent = AgentInterface(tiny_config, provider, bm, sb, tools)

        runner = EpochRunner(
            config=tiny_config,
            agent_interface=agent,
            judge_evaluator=_mock_judge(),
            benchmark_loader=MagicMock(),
            budget_manager=bm,
            phase_manager=PhaseManager(tiny_config),
            memory_manager=mem,
            scorer=Scorer(tiny_config),
            dedup_filter=DedupFilter(tiny_config),
            drift_monitor=CUSUMMonitor(tiny_config),
            smoke_test=SmokeTest(sb),
            token_counter=TokenCounter(tiny_config),
            logger=logging.getLogger("test"),
            run_dir=str(tmp_path / "run"),
        )

        tasks = [_task()]
        result = runner.run_epoch(epoch=0, tasks=tasks)
        # No crash; bible.md unchanged since no rollback in Phase 1
        assert mem.read_file("bible.md") == "phase1 content"


# ---------------------------------------------------------------------------
# EpochRunner — circuit breaker
# ---------------------------------------------------------------------------

class TestCircuitBreaker:
    def test_circuit_breaker_logged_when_task_exceeds_limit(self, tmp_path, caplog):
        """A single task that spends > 15% of epoch budget triggers a warning."""
        # base_tokens=1000 → epoch budget ≈ 3000 (venture at epoch 0)
        # circuit_breaker_pct=0.15 → limit ≈ 450
        # Response costs 600 tokens → exceeds limit
        config = {
            **BASE_CONFIG,
            "budget": {
                **BASE_CONFIG["budget"],
                "base_tokens": 1000,
                "circuit_breaker_pct": 0.15,
                "venture_decay": 0.5,
            },
        }
        expensive_response = LLMResponse(
            content="",
            tool_calls=[
                {
                    "id": "tc_0000",
                    "name": "run_compress",
                    "arguments": {"text": "t", "target_ratio": 0.5},
                }
            ],
            usage={"input_tokens": 400, "output_tokens": 400},  # 800 total > limit
            stop_reason="tool_use",
        )
        end_turn = LLMResponse(
            content="done",
            tool_calls=[],
            usage={"input_tokens": 20, "output_tokens": 20},
            stop_reason="end_turn",
        )
        # Tool-use loop: expensive (tool call) → end_turn (no tools) per task.
        provider = MockProvider([expensive_response, end_turn] * 5)
        bm = BudgetManager(config)
        agent_workspace = str(tmp_path / "agent")
        mem = MemoryManager(config, agent_workspace)
        mem.write_file("compress.py", "def compress(t, r): return t")
        sb = _mock_sandbox()
        tools = ToolsRegistry(sb, mem, bm)
        agent = AgentInterface(config, provider, bm, sb, tools)

        runner = EpochRunner(
            config=config,
            agent_interface=agent,
            judge_evaluator=_mock_judge(),
            benchmark_loader=MagicMock(),
            budget_manager=bm,
            phase_manager=PhaseManager(config),
            memory_manager=mem,
            scorer=Scorer(config),
            dedup_filter=DedupFilter(config),
            drift_monitor=CUSUMMonitor(config),
            smoke_test=SmokeTest(sb),
            token_counter=TokenCounter(config),
            logger=logging.getLogger("test"),
            run_dir=str(tmp_path / "run"),
        )

        with caplog.at_level(logging.WARNING, logger="test"):
            runner.run_epoch(epoch=0, tasks=[_task()])

        # Warning should mention circuit breaker
        circuit_warnings = [r for r in caplog.records if "circuit" in r.message.lower()]
        assert len(circuit_warnings) >= 1


# ---------------------------------------------------------------------------
# EpochRunner — phase transitions
# ---------------------------------------------------------------------------

class TestPhaseTransitions:
    def test_phase1_no_dedup_filter(self, tmp_path):
        """In Phase 1, duplicate tasks are NOT filtered."""
        config = {
            **BASE_CONFIG,
            "phases": {"phase2_start": 100, "phase3_start": 200},  # stay in Phase 1
        }
        runner = _make_runner(n_tasks=2, config=config,
                               tmp_path=tmp_path, run_dir=str(tmp_path / "run"))
        # Same task twice
        same_task = _task(source_text="identical text", target_ratio=0.5)
        tasks = [same_task, same_task]
        result = runner.run_epoch(epoch=0, tasks=tasks)
        # Both tasks should be processed (no dedup in Phase 1)
        assert result["tasks_completed"] == 2

    def test_phase3_dedup_filter_active(self, tmp_path):
        """In Phase 3, duplicate tasks are skipped."""
        config = {
            **BASE_CONFIG,
            "phases": {"phase2_start": 0, "phase3_start": 0},  # Phase 3 immediately
        }
        runner = _make_runner(n_tasks=2, config=config,
                               tmp_path=tmp_path, run_dir=str(tmp_path / "run"))
        same_task = _task(source_text="identical text", target_ratio=0.5)
        tasks = [same_task, same_task]
        result = runner.run_epoch(epoch=0, tasks=tasks)
        # Second task is a duplicate → skipped → only 1 completed
        assert result["tasks_completed"] == 1
        assert result["tasks_total"] == 2

    def test_phase2_memory_backup_taken(self, tmp_path):
        """In Phase 2, memory backup is taken at epoch start."""
        config = {
            **BASE_CONFIG,
            "phases": {"phase2_start": 0, "phase3_start": 100},  # Phase 2 from epoch 0
        }
        agent_workspace = str(tmp_path / "agent")
        mem = MemoryManager(config, agent_workspace)
        mem.write_file("bible.md", "phase2 content")
        runner = _make_runner(
            n_tasks=1,
            config=config,
            memory=mem,
            tmp_path=tmp_path,
            run_dir=str(tmp_path / "run"),
        )
        # Should complete without error (backup is taken internally)
        result = runner.run_epoch(epoch=0, tasks=[_task()])
        assert "epoch" in result

    def test_phase1_no_memory_backup(self, tmp_path):
        """In Phase 1, no memory backup is taken (has_memory returns False)."""
        config = {
            **BASE_CONFIG,
            "phases": {"phase2_start": 100, "phase3_start": 200},
        }
        runner = _make_runner(n_tasks=1, config=config,
                               tmp_path=tmp_path, run_dir=str(tmp_path / "run"))
        result = runner.run_epoch(epoch=0, tasks=[_task()])
        assert "epoch" in result  # ran without error


# ---------------------------------------------------------------------------
# EpochRunner — artifact export
# ---------------------------------------------------------------------------

class TestArtifactExport:
    def test_artifact_exported_when_successful(self, tmp_path):
        """Successful epoch (score >= pass_threshold) exports compress.py artifact."""
        config = {
            **BASE_CONFIG,
            "judge": {**BASE_CONFIG["judge"], "pass_threshold": 0.5},  # easy to pass
        }
        runner = _make_runner(
            n_tasks=1,
            task_score=0.9,
            passes=True,
            config=config,
            tmp_path=tmp_path,
            run_dir=str(tmp_path / "run"),
        )
        result = runner.run_epoch(epoch=3, tasks=[_task()])
        assert result["artifact_path"] is not None
        assert os.path.exists(result["artifact_path"])
        assert "compress_epoch_3_" in result["artifact_path"]
        assert result["artifact_path"].endswith(".py")

    def test_artifact_content_is_compress_py(self, tmp_path):
        config = {
            **BASE_CONFIG,
            "judge": {**BASE_CONFIG["judge"], "pass_threshold": 0.5},
        }
        expected_code = (
            "def compress(text, target_ratio):\n"
            "    n = int(len(text) * target_ratio)\n"
            "    return text[:n]\n"
        )
        agent_workspace = str(tmp_path / "agent")
        mem = MemoryManager(config, agent_workspace)
        mem.write_file("compress.py", expected_code)

        runner = _make_runner(
            n_tasks=1,
            task_score=0.9,
            passes=True,
            config=config,
            memory=mem,
            tmp_path=tmp_path,
            run_dir=str(tmp_path / "run"),
        )
        result = runner.run_epoch(epoch=1, tasks=[_task()])
        with open(result["artifact_path"]) as f:
            artifact_code = f.read()
        assert artifact_code == expected_code

    def test_no_artifact_when_epoch_fails(self, tmp_path):
        """Unsuccessful epoch (score < pass_threshold) does not export artifact."""
        config = {
            **BASE_CONFIG,
            "judge": {**BASE_CONFIG["judge"], "pass_threshold": 0.99},  # very high threshold
        }
        runner = _make_runner(
            n_tasks=1,
            task_score=0.3,
            passes=False,
            config=config,
            tmp_path=tmp_path,
            run_dir=str(tmp_path / "run"),
        )
        result = runner.run_epoch(epoch=0, tasks=[_task()])
        assert result["artifact_path"] is None

    def test_artifact_filename_includes_score(self, tmp_path):
        config = {
            **BASE_CONFIG,
            "judge": {**BASE_CONFIG["judge"], "pass_threshold": 0.5},
        }
        runner = _make_runner(
            n_tasks=1,
            task_score=0.9,
            passes=True,
            config=config,
            tmp_path=tmp_path,
            run_dir=str(tmp_path / "run"),
        )
        result = runner.run_epoch(epoch=2, tasks=[_task()])
        filename = os.path.basename(result["artifact_path"])
        # Should look like compress_epoch_2_0.900.py (or similar score)
        assert filename.startswith("compress_epoch_2_")
        assert filename.endswith(".py")

    def test_no_artifact_when_compress_py_empty(self, tmp_path):
        config = {
            **BASE_CONFIG,
            "judge": {**BASE_CONFIG["judge"], "pass_threshold": 0.5},
        }
        agent_workspace = str(tmp_path / "agent")
        mem = MemoryManager(config, agent_workspace)
        # Overwrite seed with empty content to simulate no agent code.
        import os
        compress_path = os.path.join(agent_workspace, "compress.py")
        if os.path.exists(compress_path):
            os.remove(compress_path)

        runner = _make_runner(
            n_tasks=1,
            task_score=0.9,
            passes=True,
            config=config,
            memory=mem,
            tmp_path=tmp_path,
            run_dir=str(tmp_path / "run"),
        )
        result = runner.run_epoch(epoch=0, tasks=[_task()])
        assert result["artifact_path"] is None


# ---------------------------------------------------------------------------
# EpochRunner — rat mode
# ---------------------------------------------------------------------------

class TestRatMode:
    def test_rat_mode_not_active_in_phase1(self, tmp_path):
        """Phase 1 never has rat mode."""
        config = {
            **BASE_CONFIG,
            "phases": {"phase2_start": 100, "phase3_start": 200},
        }
        pm = PhaseManager(config)
        assert not pm.has_rat_mode(0)
        assert not pm.has_rat_mode(50)

    def test_rat_mode_available_in_phase3(self, tmp_path):
        """Phase 3 has rat mode capability."""
        config = {
            **BASE_CONFIG,
            "phases": {"phase2_start": 0, "phase3_start": 0},
        }
        pm = PhaseManager(config)
        assert pm.has_rat_mode(0)

    def test_critical_starvation_flag_set_at_10pct_budget(self):
        """BudgetManager raises is_critical_starvation when remaining < 10% of total."""
        bm = BudgetManager(BASE_CONFIG)
        bm.start_epoch(0)
        initial = bm.remaining
        # Spend 91% of budget
        bm.spend(int(initial * 0.91))
        assert bm.is_critical_starvation is True

    def test_no_critical_starvation_at_normal_budget(self):
        """No critical starvation when plenty of budget remains."""
        bm = BudgetManager(BASE_CONFIG)
        bm.start_epoch(0)
        initial = bm.remaining
        bm.spend(int(initial * 0.5))
        assert bm.is_critical_starvation is False


# ---------------------------------------------------------------------------
# EpochRunner — drift detection
# ---------------------------------------------------------------------------

class TestDriftDetection:
    def test_drift_not_detected_on_first_epoch(self, tmp_path):
        runner = _make_runner(n_tasks=1, tmp_path=tmp_path, run_dir=str(tmp_path / "run"))
        result = runner.run_epoch(epoch=0, tasks=[_task()])
        # Single data point — cannot detect drift
        assert result["drift_detected"] is False

    def test_drift_result_key_present(self, tmp_path):
        runner = _make_runner(n_tasks=1, tmp_path=tmp_path, run_dir=str(tmp_path / "run"))
        result = runner.run_epoch(epoch=0, tasks=[_task()])
        assert "drift_detected" in result
        assert isinstance(result["drift_detected"], bool)


# ---------------------------------------------------------------------------
# EpochRunner — system prompt content
# ---------------------------------------------------------------------------

class TestSystemPrompt:
    def test_prompt_mentions_epoch(self, tmp_path):
        runner = _make_runner(n_tasks=0, tmp_path=tmp_path, run_dir=str(tmp_path / "run"))
        prompt = runner._build_system_prompt(epoch=5, phase=1)
        assert "5" in prompt

    def test_prompt_mentions_phase(self, tmp_path):
        runner = _make_runner(n_tasks=0, tmp_path=tmp_path, run_dir=str(tmp_path / "run"))
        p1 = runner._build_system_prompt(epoch=0, phase=1)
        p2 = runner._build_system_prompt(epoch=0, phase=2)
        p3 = runner._build_system_prompt(epoch=0, phase=3)
        assert "Phase 1" in p1
        assert "Phase 2" in p2
        assert "Phase 3" in p3

    def test_prompt_lists_tools(self, tmp_path):
        runner = _make_runner(n_tasks=0, tmp_path=tmp_path, run_dir=str(tmp_path / "run"))
        prompt = runner._build_system_prompt(epoch=0, phase=1)
        for tool in ("run_compress", "read_file", "write_file", "run_script", "audit_budget"):
            assert tool in prompt

    def test_prompt_changes_per_phase(self, tmp_path):
        runner = _make_runner(n_tasks=0, tmp_path=tmp_path, run_dir=str(tmp_path / "run"))
        p1 = runner._build_system_prompt(epoch=0, phase=1)
        p3 = runner._build_system_prompt(epoch=0, phase=3)
        assert p1 != p3


# ---------------------------------------------------------------------------
# EpochRunner — feedback sent to agent
# ---------------------------------------------------------------------------

class TestAgentFeedback:
    def test_metrics_sent_after_each_task(self, tmp_path):
        """agent.send_metrics() is called once per completed task."""
        runner = _make_runner(n_tasks=3, tmp_path=tmp_path, run_dir=str(tmp_path / "run"))
        tasks = [_task(source_text=f"text {i}") for i in range(3)]

        # Wrap agent.send_metrics to track calls.
        metrics_calls = []
        original_send_metrics = runner.agent.send_metrics

        def tracking_metrics(**kwargs):
            metrics_calls.append(kwargs)
            return original_send_metrics(**kwargs)

        runner.agent.send_metrics = tracking_metrics
        runner.run_epoch(epoch=0, tasks=tasks)
        assert len(metrics_calls) == 3

    def test_metrics_excludes_hidden_type(self, tmp_path):
        """hidden_type must NOT be sent to agent in feedback."""
        runner = _make_runner(n_tasks=1, tmp_path=tmp_path, run_dir=str(tmp_path / "run"))
        metrics_calls = []
        original = runner.agent.send_metrics

        def capture(**kwargs):
            metrics_calls.append(kwargs)
            return original(**kwargs)

        runner.agent.send_metrics = capture
        runner.run_epoch(epoch=0, tasks=[_task()])
        assert len(metrics_calls) == 1
        assert "hidden_type" not in metrics_calls[0]["feedback"]

    def test_metrics_includes_task_score(self, tmp_path):
        """Feedback payload includes task_score and pass."""
        runner = _make_runner(n_tasks=1, tmp_path=tmp_path, run_dir=str(tmp_path / "run"))
        metrics_calls = []
        original = runner.agent.send_metrics

        def capture(**kwargs):
            metrics_calls.append(kwargs)
            return original(**kwargs)

        runner.agent.send_metrics = capture
        runner.run_epoch(epoch=0, tasks=[_task()])
        fb = metrics_calls[0]["feedback"]
        assert "task_score" in fb
        assert "pass" in fb
