"""FastAPI/WebSocket dashboard server."""


class DashboardServer:
    """Serves real-time metrics over WebSocket on the configured port."""

    def __init__(self, config: dict):
        self.config = config
        # TODO: implement
