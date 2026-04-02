import math


class BudgetManager:
    """Manages token budget for an epoch: hard cap, venture, circuit breaker, R&D fund."""

    def __init__(self, config: dict):
        b = config["budget"]
        self.base_tokens = b["base_tokens"]
        self.circuit_breaker_pct = b["circuit_breaker_pct"]
        self.venture_decay = b["venture_decay"]
        self.rnd_lambda = b["rnd_lambda"]
        self.rnd_max_pct = b["rnd_max_pct"]
        self.rnd_min_success_rate = b["rnd_min_success_rate"]
        self.critical_starvation_pct = b["critical_starvation_pct"]

        # State
        self.remaining = 0
        self.epoch = 0
        self.rnd_fund = 0
        self.total_spent = 0
        self.last_step_cost = 0
        self.is_oom = False
        self.is_critical_starvation = False
        self._initial_epoch_budget = 0  # for circuit breaker

    def start_epoch(
        self,
        epoch: int,
        prev_success_rate: float = 0.0,
        prev_saved: int = 0,
        prev_oom: bool = False,
    ):
        """Initialize budget for a new epoch."""
        self.epoch = epoch
        self.is_oom = False
        self.is_critical_starvation = False
        self.total_spent = 0

        # Effective budget = base * venture_multiplier + rnd_fund
        effective = int(self.base_tokens * self.venture_multiplier(epoch))

        # R&D carry-over (Phase 2+, only if prev epoch was successful)
        if not prev_oom and prev_success_rate >= self.rnd_min_success_rate:
            self.rnd_fund = self.calculate_rnd_fund(prev_saved)
        else:
            self.rnd_fund = 0

        self.remaining = effective + self.rnd_fund
        self._initial_epoch_budget = self.remaining

    def venture_multiplier(self, epoch: int) -> float:
        """K = 1 + 2 * exp(-decay * epoch). Active in Phase 1."""
        return 1.0 + 2.0 * math.exp(-self.venture_decay * epoch)

    def calculate_rnd_fund(self, saved_tokens: int) -> int:
        """R&D fund = rnd_max_pct * base * (1 - exp(-lambda * saved)). Diminishing returns."""
        raw = self.rnd_max_pct * self.base_tokens * (1.0 - math.exp(-self.rnd_lambda * saved_tokens))
        return int(raw)

    def circuit_breaker_limit(self) -> int:
        """Max tokens any single task can consume: circuit_breaker_pct of initial epoch budget."""
        return int(self._initial_epoch_budget * self.circuit_breaker_pct)

    def spend(self, tokens: int) -> bool:
        """Deduct tokens. Returns False if OOM (budget exceeded)."""
        self.last_step_cost = tokens
        self.total_spent += tokens
        self.remaining -= tokens

        if self.remaining <= 0:
            self.is_oom = True
            self.remaining = 0
            return False

        # Check critical starvation: remaining < pct * effective_budget
        if self.remaining < self.critical_starvation_pct * (self.total_spent + self.remaining):
            self.is_critical_starvation = True

        return True

    def refund(self, tokens: int) -> None:
        """Return tokens to budget (e.g. free tool calls like graveyard reads)."""
        self.remaining += tokens
        self.total_spent = max(0, self.total_spent - tokens)
        # Un-trip starvation if refund brought us back above threshold.
        if self.remaining >= self.critical_starvation_pct * (self.total_spent + self.remaining):
            self.is_critical_starvation = False

    def can_afford(self, estimated_tokens: int) -> bool:
        """Pre-check if we can afford this call."""
        return estimated_tokens <= self.remaining

    def pulse_string(self) -> str:
        """Format: [B:14050|C:412]"""
        return f"[B:{self.remaining}|C:{self.last_step_cost}]"

    @property
    def saved_tokens(self) -> int:
        return max(0, self.remaining)

    @property
    def effective_budget(self) -> int:
        return self.total_spent + self.remaining
