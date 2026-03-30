"""CUSUM-based score drift monitor."""


class CUSUMMonitor:
    """Detects score distribution drift using a CUSUM control chart."""

    def __init__(self, config: dict):
        self.config = config
        # TODO: implement
