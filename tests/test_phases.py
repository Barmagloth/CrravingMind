import pytest
from craving_mind.orchestrator.phases import PhaseManager


BASE_CONFIG = {
    "phases": {
        "phase2_start": 11,
        "phase3_start": 26,
    }
}


@pytest.fixture
def pm():
    return PhaseManager(BASE_CONFIG)


# ── get_phase ──────────────────────────────────────────────────────────────────

class TestGetPhase:
    def test_epoch_1_is_phase_1(self, pm):
        assert pm.get_phase(1) == 1

    def test_epoch_10_is_phase_1(self, pm):
        assert pm.get_phase(10) == 1

    def test_epoch_11_is_phase_2(self, pm):
        assert pm.get_phase(11) == 2

    def test_epoch_25_is_phase_2(self, pm):
        assert pm.get_phase(25) == 2

    def test_epoch_26_is_phase_3(self, pm):
        assert pm.get_phase(26) == 3

    def test_epoch_100_is_phase_3(self, pm):
        assert pm.get_phase(100) == 3

    def test_epoch_0_is_phase_1(self, pm):
        assert pm.get_phase(0) == 1

    def test_boundary_10_to_11(self, pm):
        assert pm.get_phase(10) == 1
        assert pm.get_phase(11) == 2

    def test_boundary_25_to_26(self, pm):
        assert pm.get_phase(25) == 2
        assert pm.get_phase(26) == 3


# ── has_memory ─────────────────────────────────────────────────────────────────

class TestHasMemory:
    def test_phase_1_no_memory(self, pm):
        for epoch in [1, 5, 10]:
            assert pm.has_memory(epoch) is False

    def test_phase_2_has_memory(self, pm):
        for epoch in [11, 15, 25]:
            assert pm.has_memory(epoch) is True

    def test_phase_3_has_memory(self, pm):
        for epoch in [26, 50, 100]:
            assert pm.has_memory(epoch) is True


# ── has_rnd_fund ───────────────────────────────────────────────────────────────

class TestHasRndFund:
    def test_phase_1_no_rnd_fund(self, pm):
        for epoch in [1, 5, 10]:
            assert pm.has_rnd_fund(epoch) is False

    def test_phase_2_has_rnd_fund(self, pm):
        for epoch in [11, 20, 25]:
            assert pm.has_rnd_fund(epoch) is True

    def test_phase_3_has_rnd_fund(self, pm):
        assert pm.has_rnd_fund(26) is True


# ── has_rat_mode ───────────────────────────────────────────────────────────────

class TestHasRatMode:
    def test_phase_1_no_rat_mode(self, pm):
        for epoch in [1, 10]:
            assert pm.has_rat_mode(epoch) is False

    def test_phase_2_no_rat_mode(self, pm):
        for epoch in [11, 25]:
            assert pm.has_rat_mode(epoch) is False

    def test_phase_3_has_rat_mode(self, pm):
        for epoch in [26, 50, 100]:
            assert pm.has_rat_mode(epoch) is True


# ── has_scarring ───────────────────────────────────────────────────────────────

class TestHasScarring:
    def test_phase_1_no_scarring(self, pm):
        assert pm.has_scarring(10) is False

    def test_phase_2_no_scarring(self, pm):
        assert pm.has_scarring(25) is False

    def test_phase_3_has_scarring(self, pm):
        assert pm.has_scarring(26) is True


# ── has_duplicate_filter ───────────────────────────────────────────────────────

class TestHasDuplicateFilter:
    def test_phase_1_no_filter(self, pm):
        assert pm.has_duplicate_filter(1) is False

    def test_phase_2_no_filter(self, pm):
        assert pm.has_duplicate_filter(11) is False

    def test_phase_3_has_filter(self, pm):
        assert pm.has_duplicate_filter(26) is True


# ── has_venture ────────────────────────────────────────────────────────────────

class TestHasVenture:
    def test_phase_1_has_venture(self, pm):
        for epoch in [1, 5, 10]:
            assert pm.has_venture(epoch) is True

    def test_phase_2_no_venture(self, pm):
        for epoch in [11, 20, 25]:
            assert pm.has_venture(epoch) is False

    def test_phase_3_no_venture(self, pm):
        for epoch in [26, 50]:
            assert pm.has_venture(epoch) is False

    def test_boundary_10_has_venture(self, pm):
        assert pm.has_venture(10) is True

    def test_boundary_11_no_venture(self, pm):
        assert pm.has_venture(11) is False


# ── feature matrix: all flags per phase ───────────────────────────────────────

class TestFeatureMatrix:
    """Verify the complete feature flag table for representative epochs."""

    @pytest.mark.parametrize("epoch", [1, 5, 10])
    def test_phase_1_feature_set(self, pm, epoch):
        assert pm.get_phase(epoch) == 1
        assert pm.has_venture(epoch) is True
        assert pm.has_memory(epoch) is False
        assert pm.has_rnd_fund(epoch) is False
        assert pm.has_rat_mode(epoch) is False
        assert pm.has_scarring(epoch) is False
        assert pm.has_duplicate_filter(epoch) is False

    @pytest.mark.parametrize("epoch", [11, 18, 25])
    def test_phase_2_feature_set(self, pm, epoch):
        assert pm.get_phase(epoch) == 2
        assert pm.has_venture(epoch) is False
        assert pm.has_memory(epoch) is True
        assert pm.has_rnd_fund(epoch) is True
        assert pm.has_rat_mode(epoch) is False
        assert pm.has_scarring(epoch) is False
        assert pm.has_duplicate_filter(epoch) is False

    @pytest.mark.parametrize("epoch", [26, 50, 100])
    def test_phase_3_feature_set(self, pm, epoch):
        assert pm.get_phase(epoch) == 3
        assert pm.has_venture(epoch) is False
        assert pm.has_memory(epoch) is True
        assert pm.has_rnd_fund(epoch) is True
        assert pm.has_rat_mode(epoch) is True
        assert pm.has_scarring(epoch) is True
        assert pm.has_duplicate_filter(epoch) is True
