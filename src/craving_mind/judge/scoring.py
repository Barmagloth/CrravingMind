"""Composite scoring: semantic + entity overlap."""


class Scorer:
    """Combines semantic similarity and entity overlap into a final score."""

    def __init__(self, config: dict):
        self.config = config
        # TODO: implement
