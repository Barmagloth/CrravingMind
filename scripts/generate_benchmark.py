#!/usr/bin/env python
"""CLI: generate benchmark dataset from raw sources.

Usage:
    python scripts/generate_benchmark.py \\
        --sources data/sources/ \\
        --output data/benchmarks/benchmark_v1.parquet \\
        --config config/default.yaml

    # Offline mock run (no LLM calls):
    python scripts/generate_benchmark.py \\
        --sources data/sources/ \\
        --output data/benchmarks/benchmark_v1.parquet \\
        --mock
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _load_config(config_path: str) -> dict:
    import yaml  # type: ignore

    with open(config_path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _collect_sources(sources_dir: str) -> list[dict[str, str]]:
    from craving_mind.benchmark.sources import load_texts_from_dir, list_available_sources

    counts = list_available_sources(sources_dir)
    if not counts:
        print(f"[warn] No source files found under {sources_dir!r}", file=sys.stderr)
        return []

    records: list[dict[str, str]] = []
    for hidden_type, n in counts.items():
        type_dir = str(Path(sources_dir) / hidden_type)
        loaded = load_texts_from_dir(type_dir, hidden_type)
        records.extend(loaded)
        print(f"  {hidden_type}: {len(loaded)}/{n} files loaded")
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate CravingMind benchmark Parquet.")
    parser.add_argument("--sources", default="data/sources/", help="Root dir with per-type subdirs.")
    parser.add_argument("--output", default="data/benchmarks/benchmark_v1.parquet", help="Output .parquet path.")
    parser.add_argument("--config", default="config/default.yaml", help="YAML config file.")
    parser.add_argument("--mock", action="store_true", help="Use mock LLM (no API calls).")
    args = parser.parse_args(argv)

    config: dict = {}
    if Path(args.config).exists():
        config = _load_config(args.config)
    else:
        print(f"[warn] Config not found at {args.config!r}, using defaults.", file=sys.stderr)

    print(f"Collecting sources from {args.sources!r} ...")
    source_records = _collect_sources(args.sources)

    if not source_records:
        print("[error] No source records collected. Aborting.", file=sys.stderr)
        return 1

    print(f"Total source records: {len(source_records)}")

    if args.mock:
        from craving_mind.benchmark.generator import MockBenchmarkGenerator

        generator = MockBenchmarkGenerator(config=config)
        print("[mock] Using MockBenchmarkGenerator (no LLM calls).")
    else:
        print("[error] Real LLM generator not yet wired. Use --mock for offline runs.", file=sys.stderr)
        return 1

    print(f"Generating benchmark → {args.output!r} ...")
    generator.generate_benchmark(source_records, args.output)
    print(f"Done. Wrote {len(source_records)} records to {args.output!r}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
