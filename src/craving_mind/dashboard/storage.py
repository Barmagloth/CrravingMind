"""Persistent metrics storage (Parquet-backed)."""


class MetricsStorage:
    """Persists metrics to disk and supports historical queries."""

    def __init__(self, config: dict):
        self.config = config
        # TODO: implement
