"""Benchmark dataset loader (Parquet + frozen split)."""


class BenchmarkLoader:
    """Loads and splits the benchmark dataset into frozen and dynamic subsets."""

    def __init__(self, config: dict):
        self.config = config
        # TODO: implement
