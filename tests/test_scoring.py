"""Tests for craving_mind.judge.scoring — Scorer class."""

import math
import pytest
from craving_mind.judge.scoring import Scorer


@pytest.fixture
def scorer():
    return Scorer()


# ------------------------------------------------------------------
# task_score
# ------------------------------------------------------------------

class TestTaskScore:
    def test_equal_weights_midpoint(self, scorer):
        assert scorer.task_score(0.8, 0.6) == pytest.approx(0.7)

    def test_equal_weights_perfect(self, scorer):
        assert scorer.task_score(1.0, 1.0) == pytest.approx(1.0)

    def test_equal_weights_zero(self, scorer):
        assert scorer.task_score(0.0, 0.0) == pytest.approx(0.0)

    def test_custom_weights_semantic_dominant(self, scorer):
        result = scorer.task_score(1.0, 0.0, weights={"semantic": 0.8, "entity": 0.2})
        assert result == pytest.approx(0.8)

    def test_custom_weights_entity_dominant(self, scorer):
        result = scorer.task_score(0.0, 1.0, weights={"semantic": 0.2, "entity": 0.8})
        assert result == pytest.approx(0.8)

    def test_zero_total_weight_returns_zero(self, scorer):
        result = scorer.task_score(1.0, 1.0, weights={"semantic": 0.0, "entity": 0.0})
        assert result == pytest.approx(0.0)


# ------------------------------------------------------------------
# is_pass
# ------------------------------------------------------------------

class TestIsPass:
    def test_exactly_at_threshold_passes(self, scorer):
        assert scorer.is_pass(0.85) is True

    def test_just_below_threshold_fails(self, scorer):
        assert scorer.is_pass(0.849) is False

    def test_above_threshold_passes(self, scorer):
        assert scorer.is_pass(0.9) is True

    def test_zero_fails(self, scorer):
        assert scorer.is_pass(0.0) is False

    def test_custom_threshold(self, scorer):
        assert scorer.is_pass(0.7, threshold=0.7) is True
        assert scorer.is_pass(0.699, threshold=0.7) is False


# ------------------------------------------------------------------
# epoch_success_rate
# ------------------------------------------------------------------

class TestEpochSuccessRate:
    def test_equal_types_all_pass(self, scorer):
        scores = {
            "discourse": [True, True, True],
            "needle":    [True, True, True],
            "code":      [True, True, True],
        }
        rate = scorer.epoch_success_rate(scores)
        assert rate == pytest.approx(1.0)

    def test_equal_types_mixed(self, scorer):
        # discourse=1.0, needle=0.5, code=0.5 → geometric mean
        scores = {
            "discourse": [True, True],
            "needle":    [True, False],
            "code":      [True, False],
        }
        rate = scorer.epoch_success_rate(scores)
        expected = math.exp((math.log(1.0) + math.log(0.5) + math.log(0.5)) / 3)
        assert rate == pytest.approx(expected)

    def test_one_type_at_zero_uses_epsilon(self, scorer):
        scores = {
            "discourse": [True, True],
            "needle":    [False, False],  # sr=0 → clamped to epsilon=0.01
        }
        epsilon = 0.01
        rate = scorer.epoch_success_rate(scores)
        expected = math.exp((math.log(1.0) + math.log(epsilon)) / 2)
        assert rate == pytest.approx(expected)

    def test_single_type_full_pass(self, scorer):
        scores = {"discourse": [True, True, True]}
        assert scorer.epoch_success_rate(scores) == pytest.approx(1.0)

    def test_single_type_full_fail(self, scorer):
        scores = {"discourse": [False, False]}
        epsilon = 0.01
        assert scorer.epoch_success_rate(scores) == pytest.approx(epsilon)

    def test_empty_scores_returns_zero(self, scorer):
        assert scorer.epoch_success_rate({}) == pytest.approx(0.0)

    def test_custom_epsilon(self, scorer):
        scores = {"needle": [False]}
        rate = scorer.epoch_success_rate(scores, epsilon=0.05)
        assert rate == pytest.approx(0.05)

    def test_custom_type_weights(self, scorer):
        # discourse weight=2, needle weight=1
        scores = {
            "discourse": [True, True],   # sr=1.0
            "needle":    [False, False],  # sr=0 → epsilon=0.01
        }
        epsilon = 0.01
        rate = scorer.epoch_success_rate(
            scores,
            type_weights={"discourse": 2.0, "needle": 1.0},
        )
        expected = math.exp((2 * math.log(1.0) + 1 * math.log(epsilon)) / 3)
        assert rate == pytest.approx(expected)


# ------------------------------------------------------------------
# combined_success_rate
# ------------------------------------------------------------------

class TestCombinedSuccessRate:
    def test_symmetric_inputs(self, scorer):
        # frozen == dynamic → result == both
        rate = scorer.combined_success_rate(0.8, 0.8, dynamic_multiplier=1.3)
        assert rate == pytest.approx(0.8)

    def test_dynamic_pulls_result_up(self, scorer):
        rate = scorer.combined_success_rate(0.5, 1.0, dynamic_multiplier=1.3)
        assert rate > 0.5
        assert rate < 1.0

    def test_dynamic_pulls_result_down(self, scorer):
        rate = scorer.combined_success_rate(1.0, 0.5, dynamic_multiplier=1.3)
        assert rate < 1.0
        assert rate > 0.5

    def test_formula_correctness(self, scorer):
        mult = 1.3
        frozen, dynamic = 0.6, 0.9
        expected = (frozen + mult * dynamic) / (1 + mult)
        assert scorer.combined_success_rate(frozen, dynamic, mult) == pytest.approx(expected)

    def test_uses_config_multiplier_by_default(self):
        cfg = {"judge": {"dynamic_multiplier": 2.0}}
        scorer = Scorer(cfg)
        frozen, dynamic = 0.6, 0.9
        expected = (frozen + 2.0 * dynamic) / (1 + 2.0)
        assert scorer.combined_success_rate(frozen, dynamic) == pytest.approx(expected)
