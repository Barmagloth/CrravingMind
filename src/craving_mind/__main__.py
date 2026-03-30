"""Entry point: python -m craving_mind --config config/default.yaml"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone

from craving_mind.utils.config import load_config
from craving_mind.utils.logging import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="craving-mind",
        description="CravingMind — Computational Darwinism via resource scarcity",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to config YAML",
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Run directory (auto-generated if not set)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=100,
        help="Max epochs to run (default: 100)",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock LLM provider (no API calls)",
    )
    parser.add_argument(
        "--provider",
        choices=["anthropic", "cli", "mock"],
        default=None,
        help="LLM provider override (anthropic | cli | mock); overrides config and --mock",
    )
    parser.add_argument(
        "--benchmark",
        default=None,
        help="Path to frozen Parquet benchmark file",
    )
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Start the web dashboard before running epochs",
    )
    parser.add_argument(
        "--dashboard-port",
        type=int,
        default=None,
        help="Dashboard port (overrides config)",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    # Determine run directory.
    if args.run_dir is None:
        ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_dir = os.path.join("runs", f"run_{ts}")
    else:
        run_dir = args.run_dir

    os.makedirs(run_dir, exist_ok=True)

    log_level = config.get("logging", {}).get("level", "INFO")
    setup_logging(run_dir=run_dir, level=log_level)
    logger = logging.getLogger("craving_mind")
    logger.info("CravingMind starting", extra={"run_dir": run_dir, "config": args.config})

    # Lazy imports — keep startup fast for --help.
    from craving_mind.orchestrator.artifact_manager import ArtifactManager
    from craving_mind.orchestrator.budget import BudgetManager
    from craving_mind.orchestrator.checkpoint import CheckpointManager
    from craving_mind.orchestrator.phases import PhaseManager
    from craving_mind.orchestrator.runner import EpochRunner
    from craving_mind.agent.interface import AgentInterface, AnthropicProvider, CLIProvider, MockProvider
    from craving_mind.agent.tools import ToolsRegistry
    from craving_mind.agent.memory import MemoryManager
    from craving_mind.agent.sandbox import Sandbox
    from craving_mind.judge.scoring import Scorer
    from craving_mind.judge.dedup import DedupFilter
    from craving_mind.judge.drift import CUSUMMonitor
    from craving_mind.judge.smoke_test import SmokeTest
    from craving_mind.judge.evaluator import ConcreteJudgeEvaluator
    from craving_mind.benchmark.loader import BenchmarkLoader
    from craving_mind.utils.tokens import TokenCounter

    # Resolve provider: --provider flag > --mock flag > config.agent.provider.
    agent_cfg = config.get("agent", {})
    effective_provider = args.provider or ("mock" if args.mock else agent_cfg.get("provider", "anthropic"))

    # LLM providers.
    if effective_provider == "mock":
        provider = MockProvider()
        judge_provider = MockProvider()
        logger.info("Using mock LLM provider")
    elif effective_provider == "cli":
        cli_model = agent_cfg.get("cli_model", "haiku")
        provider = CLIProvider(model=cli_model)
        judge_llm_cfg = config.get("judge", {}).get("llm", {})
        judge_provider = CLIProvider(model=agent_cfg.get("cli_model", "haiku"))
        logger.info("Using CLI LLM provider", extra={"model": cli_model})
    else:
        provider = AnthropicProvider(
            model=agent_cfg.get("model", "claude-sonnet-4-6"),
            api_key=agent_cfg.get("api_key"),
        )
        judge_llm_cfg = config.get("judge", {}).get("llm", {})
        judge_provider = AnthropicProvider(
            model=judge_llm_cfg.get("model", "claude-haiku-4-5-20251001"),
        )

    # Wiring.
    budget = BudgetManager(config)
    phase_manager = PhaseManager(config)
    sandbox = Sandbox(config)
    agent_dir = os.path.join(run_dir, "agent_workspace")
    memory = MemoryManager(config, agent_dir)
    tools = ToolsRegistry(sandbox, memory, budget)
    agent = AgentInterface(config, provider, budget, sandbox, tools)
    scorer = Scorer(config)
    dedup = DedupFilter(config)
    drift = CUSUMMonitor(config)
    smoke = SmokeTest(sandbox)
    token_counter = TokenCounter(config)
    benchmark_loader = BenchmarkLoader(config)
    checkpoint = CheckpointManager(run_dir)

    # Artifact manager (versioned exports).
    artifacts_dir = os.path.join(run_dir, "artifacts")
    artifact_manager = ArtifactManager(artifacts_dir)

    # Judge evaluator — loads ML models (embeddings + NER).
    logger.info("Loading judge evaluator (embeddings + NER)…")
    judge = ConcreteJudgeEvaluator(judge_provider, config=config)

    # Load benchmark tasks.
    all_frozen: list = []
    if args.benchmark:
        logger.info("Loading benchmark", extra={"path": args.benchmark})
        all_frozen = benchmark_loader.load_frozen(args.benchmark)
        logger.info("Benchmark loaded", extra={"tasks": len(all_frozen)})
    elif args.mock:
        # Generate a small mock benchmark for smoke-testing with --mock.
        logger.info("Generating mock benchmark (no --benchmark specified)")
        from craving_mind.benchmark.generator import MockBenchmarkGenerator
        mock_gen = MockBenchmarkGenerator(config)
        source_records = [
            {
                "source_text": (
                    f"This is mock source text number {i}. "
                    "It contains information about various topics "
                    "including science, history, and technology. " * 4
                ).strip(),
                "hidden_type": "discourse",
            }
            for i in range(10)
        ]
        for rec in source_records:
            raw = mock_gen.generate_record(rec["source_text"], rec["hidden_type"])
            task = dict(raw)
            for field in ("questions", "reference_answers", "reference_entities"):
                if isinstance(task.get(field), str):
                    task[field] = json.loads(task[field])
            all_frozen.append(task)
        logger.info("Mock benchmark generated", extra={"tasks": len(all_frozen)})
    else:
        logger.warning("No --benchmark specified — running with empty task pool")

    # Resume from checkpoint if requested.
    start_epoch = 0
    prev_success_rate = 0.0
    prev_saved = 0
    prev_oom = False

    if args.resume:
        state = checkpoint.load()
        if state:
            start_epoch = state.get("epoch", 0) + 1
            prev_success_rate = state.get("success_rate", 0.0)
            prev_saved = state.get("saved_tokens", 0)
            prev_oom = state.get("is_oom", False)
            logger.info("Resumed from checkpoint", extra={"start_epoch": start_epoch})
        else:
            logger.warning("--resume specified but no checkpoint found; starting fresh")

    # Start dashboard in a background thread if requested.
    if args.dashboard:
        from craving_mind.dashboard.server import DashboardServer
        import threading

        dashboard = DashboardServer(config, run_dir)
        port = args.dashboard_port or config.get("dashboard", {}).get("port", 8080)
        dash_thread = threading.Thread(
            target=dashboard.start,
            kwargs={"host": "0.0.0.0", "port": port},
            daemon=True,
        )
        dash_thread.start()
        print(f"Dashboard: http://localhost:{port}")

    runner = EpochRunner(
        config=config,
        agent_interface=agent,
        judge_evaluator=judge,
        benchmark_loader=benchmark_loader,
        budget_manager=budget,
        phase_manager=phase_manager,
        memory_manager=memory,
        scorer=scorer,
        dedup_filter=dedup,
        drift_monitor=drift,
        smoke_test=smoke,
        token_counter=token_counter,
        logger=logger,
        run_dir=run_dir,
        artifact_manager=artifact_manager,
    )

    tasks_per_epoch = config.get("benchmark", {}).get("tasks_per_epoch", 10)

    for epoch in range(start_epoch, args.max_epochs):
        epoch_tasks = benchmark_loader.select_frozen_subset(all_frozen, tasks_per_epoch)

        result = runner.run_epoch(
            epoch=epoch,
            tasks=epoch_tasks,
            prev_success_rate=prev_success_rate,
            prev_saved=prev_saved,
            prev_oom=prev_oom,
        )

        # Save checkpoint and logs.
        checkpoint.save({
            "epoch": epoch,
            "success_rate": result["success_rate"],
            "saved_tokens": result["saved_tokens"],
            "is_oom": result["is_oom"],
        })
        checkpoint.save_epoch_log(epoch, result)

        prev_success_rate = result["success_rate"]
        prev_saved = result["saved_tokens"]
        # Pass drift_detected as prev_oom proxy to suppress R&D fund on drift.
        prev_oom = result["is_oom"] or result.get("drift_detected", False)

        artifact_info = ""
        if result.get("artifact_path"):
            artifact_info = f" | artifact=v{artifact_manager._current_version}"

        print(
            f"Epoch {epoch:4d} | SR={result['success_rate']:.3f} "
            f"| frozen={result['frozen_success_rate']:.3f} "
            f"| saved={result['saved_tokens']:6d} "
            f"| oom={result['is_oom']}"
            f"{artifact_info}"
        )

    logger.info("Run complete", extra={"epochs_run": args.max_epochs - start_epoch})

    # Print summary.
    best = artifact_manager.get_best()
    if best:
        print(
            f"\nBest artifact: v{best['version']} "
            f"(epoch {best['epoch']}, score={best['mean_score']:.3f})"
        )
        print(f"  File: {best['filename']}")
    else:
        print("\nNo artifacts exported (compress.py unchanged or success_rate below threshold).")


if __name__ == "__main__":
    main()
