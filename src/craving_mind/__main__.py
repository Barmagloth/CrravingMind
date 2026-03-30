"""Entry point: python -m craving_mind --config config/default.yaml"""

import argparse
import sys

from craving_mind import version
from craving_mind.utils.config import load_config
from craving_mind.utils.logging import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="craving-mind",
        description="CravingMind — computational Darwinism via resource scarcity",
    )
    parser.add_argument(
        "--config",
        default="config/default.yaml",
        help="Path to YAML config file (default: config/default.yaml)",
    )
    parser.add_argument(
        "--run-dir",
        default="runs",
        help="Directory for run outputs (default: runs)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    log_level = config.get("logging", {}).get("level", "INFO")
    setup_logging(run_dir=args.run_dir, level=log_level)

    import logging
    logger = logging.getLogger("craving_mind")

    top_keys = ", ".join(sorted(config.keys()))
    print(f"CravingMind v{version} — skeleton loaded, config OK")
    logger.info("Startup", extra={"config_path": args.config, "top_level_keys": top_keys})
    print(f"  config : {args.config}")
    print(f"  keys   : {top_keys}")
    print(f"  agent  : {config.get('agent', {}).get('model', '?')}")
    print(f"  judge  : {config.get('judge', {}).get('model', '?')}")


if __name__ == "__main__":
    main()
