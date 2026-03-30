"""Judge LLM evaluator: scores agent outputs against ground truth."""


class JudgeEvaluator:
    """Calls the judge LLM and returns structured evaluation results."""

    def __init__(self, config: dict):
        self.config = config
        # TODO: implement
