class TokenCounter:
    """Token counting abstraction. Uses provider-specific tokenizer for pre-estimation,
    actual usage from API response for billing."""

    def __init__(self, config: dict):
        self.provider = config["agent"]["provider"]
        self._tokenizer = None

    def estimate(self, text: str) -> int:
        """Pre-estimate token count. Uses simple heuristic if no tokenizer available."""
        # For Anthropic: ~4 chars per token is a reasonable estimate
        # TODO: use actual tokenizer when available
        return max(1, len(text) // 4)

    def actual_from_response(self, response: dict) -> int:
        """Extract actual usage from API response."""
        # Anthropic format
        if "usage" in response:
            usage = response["usage"]
            return usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
        return 0
