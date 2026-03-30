"""Tests for craving_mind.judge.validators — TypeValidator class."""

import pytest
from craving_mind.judge.validators import TypeValidator


@pytest.fixture
def tv():
    return TypeValidator()


# ------------------------------------------------------------------
# validate_needle
# ------------------------------------------------------------------

class TestValidateNeedle:
    def test_all_perfect_unchanged(self, tv):
        scores = [1.0, 1.0, 1.0]
        assert tv.validate_needle(scores) == [1.0, 1.0, 1.0]

    def test_one_missing_entity_zeroed(self, tv):
        scores = [1.0, 0.9, 1.0]
        result = tv.validate_needle(scores)
        assert result == [1.0, 0.0, 1.0]

    def test_all_partial_all_zeroed(self, tv):
        scores = [0.5, 0.8, 0.99]
        result = tv.validate_needle(scores)
        assert result == [0.0, 0.0, 0.0]

    def test_empty_list(self, tv):
        assert tv.validate_needle([]) == []

    def test_exact_1_0_passes(self, tv):
        assert tv.validate_needle([1.0]) == [1.0]

    def test_0_999_fails(self, tv):
        assert tv.validate_needle([0.999]) == [0.0]


# ------------------------------------------------------------------
# validate_code
# ------------------------------------------------------------------

class TestValidateCode:
    def test_cosine_ignored_entity_returned(self, tv):
        semantic = [0.2, 0.3, 0.4]
        entity   = [0.9, 0.8, 0.7]
        assert tv.validate_code(semantic, entity) == pytest.approx(entity)

    def test_high_cosine_low_entity(self, tv):
        result = tv.validate_code([1.0, 1.0], [0.1, 0.2])
        assert result == pytest.approx([0.1, 0.2])

    def test_empty_inputs(self, tv):
        assert tv.validate_code([], []) == []

    def test_returns_copy_not_reference(self, tv):
        entity = [0.5]
        result = tv.validate_code([0.9], entity)
        result[0] = 99.0
        assert entity[0] == pytest.approx(0.5)


# ------------------------------------------------------------------
# validate_discourse
# ------------------------------------------------------------------

class TestValidateDiscourse:
    def test_base_scoring(self, tv):
        semantic = [1.0, 0.0]
        entity   = [0.0, 1.0]
        result = tv.validate_discourse(semantic, entity)
        assert result == pytest.approx([0.5, 0.5])

    def test_all_perfect(self, tv):
        result = tv.validate_discourse([1.0, 1.0], [1.0, 1.0])
        assert result == pytest.approx([1.0, 1.0])

    def test_all_zero(self, tv):
        result = tv.validate_discourse([0.0, 0.0], [0.0, 0.0])
        assert result == pytest.approx([0.0, 0.0])

    def test_length_mismatch_raises(self, tv):
        with pytest.raises(ValueError):
            tv.validate_discourse([1.0], [1.0, 0.5])


# ------------------------------------------------------------------
# validate (dispatcher)
# ------------------------------------------------------------------

class TestValidateDispatcher:
    def test_dispatches_needle(self, tv):
        # One imperfect entity → score zeroed → mean = 0.5
        semantic = [1.0, 1.0]
        entity   = [1.0, 0.5]  # second question zeroed
        result = tv.validate("needle", semantic, entity)
        assert result == pytest.approx(0.5)

    def test_dispatches_code(self, tv):
        semantic = [0.0, 0.0]
        entity   = [0.8, 0.6]
        result = tv.validate("code", semantic, entity)
        assert result == pytest.approx(0.7)

    def test_dispatches_discourse(self, tv):
        semantic = [1.0, 0.0]
        entity   = [0.0, 1.0]
        result = tv.validate("discourse", semantic, entity)
        assert result == pytest.approx(0.5)

    def test_unknown_type_falls_back_to_discourse(self, tv):
        semantic = [0.8, 0.6]
        entity   = [0.4, 0.2]
        expected = tv.validate("discourse", semantic, entity)
        actual   = tv.validate("mystery_type", semantic, entity)
        assert actual == pytest.approx(expected)

    def test_empty_questions_returns_zero(self, tv):
        assert tv.validate("discourse", [], []) == pytest.approx(0.0)
