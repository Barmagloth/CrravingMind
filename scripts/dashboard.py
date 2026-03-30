"""Standalone dashboard viewer for CravingMind runs.

Usage:
    python scripts/dashboard.py --run-dir runs/run_001 --port 8766
    python scripts/dashboard.py --run-dir runs/run_001  # default port 8080
"""

import argparse
import os
import sys

# Make sure the package is importable when run from the project root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from craving_mind.utils.config import load_config
from craving_mind.dashboard.server import DashboardServer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CravingMind Dashboard — view a completed or in-progress run"
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Path to the run directory (contains epoch_log.jsonl, checkpoint.json, …)",
    )
    parser.add_argument(
        "--config",
        default="config/default.yaml",
        help="Path to config YAML (default: config/default.yaml)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Dashboard port (overrides config)",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind (default: 0.0.0.0)",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.run_dir):
        print(f"Error: run-dir does not exist: {args.run_dir}", file=sys.stderr)
        sys.exit(1)

    config = load_config(args.config)

    port = args.port or config.get("dashboard", {}).get("port", 8080)
    dashboard = DashboardServer(config, args.run_dir)

    print(f"CravingMind Dashboard")
    print(f"  Run dir : {os.path.abspath(args.run_dir)}")
    print(f"  URL     : http://localhost:{port}")
    print(f"  Press Ctrl+C to stop.")

    dashboard.start(host=args.host, port=port)


if __name__ == "__main__":
    main()
