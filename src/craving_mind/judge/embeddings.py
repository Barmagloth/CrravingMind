"""Sentence-transformer embedding model wrapper."""


class EmbeddingModel:
    """Wraps a sentence-transformer model for semantic similarity scoring."""

    def __init__(self, config: dict):
        self.config = config
        # TODO: implement
