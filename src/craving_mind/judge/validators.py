"""Type and format validators for judge inputs/outputs."""


class TypeValidator:
    """Validates structural types of agent responses before scoring."""

    def __init__(self, config: dict):
        self.config = config
        # TODO: implement
