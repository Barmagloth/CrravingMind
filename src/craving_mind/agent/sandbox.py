"""Restricted execution sandbox for agent-generated code."""


class Sandbox:
    """Executes untrusted agent code in a restricted environment."""

    def __init__(self, config: dict):
        self.config = config
        # TODO: implement
