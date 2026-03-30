import math
import pytest
from craving_mind.orchestrator.budget import BudgetManager


BASE_CONFIG = {
    "budget": {
        "base_tokens": 50000,
        "circuit_breaker_pct": 0.15,
        "venture_decay": 0.5,
        "rnd_lambda": 0.0001,
        "rnd_max_pct": 0.30,
        "rnd_min_success_rate": 0.50,
        "critical_starvation_pct": 0.10,
    }
}


@pytest.fixture
def bm():
    return BudgetManager(BASE_CONFIG)


# ── venture_multiplier ─────────────────────────────────────────────────────────

class TestVentureMultiplier:
    def test_epoch_0_is_exactly_3(self, bm):
        assert bm.venture_multiplier(0) == pytest.approx(3.0, abs=1e-9)

    def test_epoch_5_approx_1_16(self, bm):
        # K = 1 + 2*exp(-2.5) ≈ 1.1641
        expected = 1.0 + 2.0 * math.exp(-2.5)
        assert bm.venture_multiplier(5) == pytest.approx(expected, rel=1e-6)
        assert 1.16 < bm.venture_multiplier(5) < 1.17

    def test_epoch_10_approx_1_013(self, bm):
        # K = 1 + 2*exp(-5) ≈ 1.01348
        expected = 1.0 + 2.0 * math.exp(-5.0)
        assert bm.venture_multiplier(10) == pytest.approx(expected, rel=1e-6)
        assert 1.013 < bm.venture_multiplier(10) < 1.014

    def test_large_epoch_approaches_1(self, bm):
        assert bm.venture_multiplier(100) == pytest.approx(1.0, abs=1e-10)

    def test_monotonically_decreasing(self, bm):
        prev = bm.venture_multiplier(0)
        for e in range(1, 20):
            cur = bm.venture_multiplier(e)
            assert cur < prev
            prev = cur


# ── calculate_rnd_fund ─────────────────────────────────────────────────────────

class TestRndFund:
    def test_zero_saved_gives_zero_fund(self, bm):
        assert bm.calculate_rnd_fund(0) == 0

    def test_large_saved_approaches_30_pct_of_base(self, bm):
        # With very large saved, fund → rnd_max_pct * base_tokens = 15000
        max_fund = int(0.30 * 50000)
        big_fund = bm.calculate_rnd_fund(1_000_000)
        assert big_fund == pytest.approx(max_fund, abs=1)

    def test_partial_saved_less_than_max(self, bm):
        fund = bm.calculate_rnd_fund(5000)
        assert 0 < fund < int(0.30 * 50000)

    def test_diminishing_returns(self, bm):
        # Doubling saved should give less than double the fund
        fund_1 = bm.calculate_rnd_fund(10000)
        fund_2 = bm.calculate_rnd_fund(20000)
        assert fund_2 < 2 * fund_1

    def test_returns_int(self, bm):
        assert isinstance(bm.calculate_rnd_fund(5000), int)


# ── circuit_breaker_limit ──────────────────────────────────────────────────────

class TestCircuitBreakerLimit:
    def test_is_15_pct_of_initial_budget_at_epoch_0(self, bm):
        bm.start_epoch(0)
        initial = bm._initial_epoch_budget
        assert bm.circuit_breaker_limit() == int(initial * 0.15)

    def test_does_not_change_after_spending(self, bm):
        bm.start_epoch(1)
        limit_before = bm.circuit_breaker_limit()
        bm.spend(10000)
        limit_after = bm.circuit_breaker_limit()
        assert limit_before == limit_after

    def test_exact_15_pct(self, bm):
        bm.start_epoch(0)
        # epoch 0: effective = int(50000 * 3.0) = 150000, no rnd → initial = 150000
        # circuit = int(150000 * 0.15) = 22500
        assert bm.circuit_breaker_limit() == int(150000 * 0.15)

    def test_epoch_10_budget(self, bm):
        bm.start_epoch(10)
        k = bm.venture_multiplier(10)
        expected_initial = int(50000 * k)
        assert bm.circuit_breaker_limit() == int(expected_initial * 0.15)


# ── spend ──────────────────────────────────────────────────────────────────────

class TestSpend:
    def test_normal_spend_returns_true(self, bm):
        bm.start_epoch(1)
        assert bm.spend(1000) is True

    def test_spend_deducts_from_remaining(self, bm):
        bm.start_epoch(1)
        initial = bm.remaining
        bm.spend(1000)
        assert bm.remaining == initial - 1000

    def test_spend_accumulates_total(self, bm):
        bm.start_epoch(1)
        bm.spend(1000)
        bm.spend(2000)
        assert bm.total_spent == 3000

    def test_spend_updates_last_step_cost(self, bm):
        bm.start_epoch(1)
        bm.spend(777)
        assert bm.last_step_cost == 777

    def test_oom_on_exact_budget(self, bm):
        bm.start_epoch(1)
        total = bm.remaining
        result = bm.spend(total)
        assert result is False
        assert bm.is_oom is True
        assert bm.remaining == 0

    def test_oom_on_overspend(self, bm):
        bm.start_epoch(1)
        result = bm.spend(bm.remaining + 1)
        assert result is False
        assert bm.is_oom is True
        assert bm.remaining == 0

    def test_critical_starvation_triggers_below_10_pct(self, bm):
        bm.start_epoch(1)
        # Spend until remaining < 10% of effective budget
        # effective = 150000 (epoch 1: K≈1+2*exp(-0.5)≈2.213, ~110657)
        effective = bm._initial_epoch_budget
        # Spend 91% to leave 9%
        to_spend = int(effective * 0.91)
        bm.spend(to_spend)
        # remaining is 9% of effective, which is < 10% → critical starvation
        assert bm.is_critical_starvation is True

    def test_no_critical_starvation_above_threshold(self, bm):
        bm.start_epoch(1)
        effective = bm._initial_epoch_budget
        # Spend 85% to leave 15%
        bm.spend(int(effective * 0.85))
        assert bm.is_critical_starvation is False

    def test_remaining_clamped_to_zero_on_oom(self, bm):
        bm.start_epoch(1)
        bm.spend(bm.remaining * 10)
        assert bm.remaining == 0


# ── start_epoch ────────────────────────────────────────────────────────────────

class TestStartEpoch:
    def test_resets_oom_flag(self, bm):
        bm.start_epoch(1)
        bm.spend(bm.remaining + 1)
        assert bm.is_oom is True
        bm.start_epoch(2)
        assert bm.is_oom is False

    def test_resets_critical_starvation(self, bm):
        bm.start_epoch(1)
        bm.spend(int(bm.remaining * 0.95))
        bm.start_epoch(2)
        assert bm.is_critical_starvation is False

    def test_resets_total_spent(self, bm):
        bm.start_epoch(1)
        bm.spend(5000)
        bm.start_epoch(2)
        assert bm.total_spent == 0

    def test_no_rnd_fund_without_carry_over(self, bm):
        # First epoch: no prev epoch, default args
        bm.start_epoch(1)
        assert bm.rnd_fund == 0

    def test_rnd_fund_with_successful_prev_epoch(self, bm):
        # Simulate: epoch 1 saved 10000 tokens with success_rate=0.8
        bm.start_epoch(2, prev_success_rate=0.8, prev_saved=10000, prev_oom=False)
        expected = bm.calculate_rnd_fund(10000)
        assert bm.rnd_fund == expected
        assert bm.rnd_fund > 0

    def test_no_rnd_fund_when_prev_oom(self, bm):
        bm.start_epoch(2, prev_success_rate=0.9, prev_saved=10000, prev_oom=True)
        assert bm.rnd_fund == 0

    def test_no_rnd_fund_when_success_rate_below_threshold(self, bm):
        bm.start_epoch(2, prev_success_rate=0.49, prev_saved=10000, prev_oom=False)
        assert bm.rnd_fund == 0

    def test_rnd_fund_exactly_at_threshold(self, bm):
        bm.start_epoch(2, prev_success_rate=0.50, prev_saved=10000, prev_oom=False)
        assert bm.rnd_fund > 0

    def test_remaining_includes_rnd_fund(self, bm):
        bm.start_epoch(2, prev_success_rate=0.8, prev_saved=10000, prev_oom=False)
        k = bm.venture_multiplier(2)
        effective = int(50000 * k)
        rnd = bm.calculate_rnd_fund(10000)
        assert bm.remaining == effective + rnd

    def test_initial_epoch_budget_stored(self, bm):
        bm.start_epoch(0)
        assert bm._initial_epoch_budget == bm.remaining


# ── pulse_string ───────────────────────────────────────────────────────────────

class TestPulseString:
    def test_format(self, bm):
        bm.start_epoch(1)
        bm.spend(412)
        remaining = bm.remaining
        assert bm.pulse_string() == f"[B:{remaining}|C:412]"

    def test_initial_cost_zero(self, bm):
        bm.start_epoch(1)
        remaining = bm.remaining
        assert bm.pulse_string() == f"[B:{remaining}|C:0]"

    def test_updates_after_spend(self, bm):
        bm.start_epoch(1)
        bm.spend(1000)
        bm.spend(500)
        remaining = bm.remaining
        assert bm.pulse_string() == f"[B:{remaining}|C:500]"


# ── can_afford ─────────────────────────────────────────────────────────────────

class TestCanAfford:
    def test_can_afford_within_budget(self, bm):
        bm.start_epoch(1)
        assert bm.can_afford(100) is True

    def test_cannot_afford_over_budget(self, bm):
        bm.start_epoch(1)
        assert bm.can_afford(bm.remaining + 1) is False

    def test_can_afford_exact_remaining(self, bm):
        bm.start_epoch(1)
        assert bm.can_afford(bm.remaining) is True

    def test_can_afford_after_partial_spend(self, bm):
        bm.start_epoch(1)
        bm.spend(1000)
        assert bm.can_afford(500) is True
        assert bm.can_afford(bm.remaining + 1) is False


# ── properties ─────────────────────────────────────────────────────────────────

class TestProperties:
    def test_saved_tokens_equals_remaining(self, bm):
        bm.start_epoch(1)
        bm.spend(5000)
        assert bm.saved_tokens == bm.remaining

    def test_saved_tokens_not_negative(self, bm):
        bm.start_epoch(1)
        bm.spend(bm.remaining * 2)
        assert bm.saved_tokens == 0

    def test_effective_budget_equals_total_spent_plus_remaining(self, bm):
        bm.start_epoch(1)
        bm.spend(3000)
        assert bm.effective_budget == bm.total_spent + bm.remaining

    def test_effective_budget_constant_through_spending(self, bm):
        bm.start_epoch(1)
        initial_effective = bm.effective_budget
        bm.spend(10000)
        assert bm.effective_budget == initial_effective
