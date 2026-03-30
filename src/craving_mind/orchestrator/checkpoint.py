"""Checkpoint serialisation and restoration."""


class CheckpointManager:
    """Saves and loads run checkpoints for fault tolerance."""

    def __init__(self, config: dict):
        self.config = config
        # TODO: implement
