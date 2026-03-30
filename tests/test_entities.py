"""Tests for craving_mind.judge.entities — EntityExtractor class (fallback mode)."""

import pytest
from craving_mind.judge.entities import EntityExtractor


@pytest.fixture
def extractor():
    """Always uses the regex fallback — no spaCy dependency required."""
    ex = EntityExtractor()
    ex._use_fallback = True  # force fallback regardless of install state
    return ex


# ------------------------------------------------------------------
# entity_f1
# ------------------------------------------------------------------

class TestEntityF1:
    def test_perfect_match(self, extractor):
        ref = {"alice", "42", "london"}
        tst = {"alice", "42", "london"}
        assert extractor.entity_f1(ref, tst) == pytest.approx(1.0)

    def test_no_overlap(self, extractor):
        ref = {"alice", "london"}
        tst = {"bob", "paris"}
        assert extractor.entity_f1(ref, tst) == pytest.approx(0.0)

    def test_partial_overlap(self, extractor):
        ref = {"alice", "bob", "charlie"}
        tst = {"alice", "bob", "dave"}
        # P = 2/3, R = 2/3, F1 = 2/3
        assert extractor.entity_f1(ref, tst) == pytest.approx(2 / 3)

    def test_both_empty_returns_one(self, extractor):
        assert extractor.entity_f1(set(), set()) == pytest.approx(1.0)

    def test_ref_empty_tst_nonempty_returns_zero(self, extractor):
        assert extractor.entity_f1(set(), {"alice"}) == pytest.approx(0.0)

    def test_ref_nonempty_tst_empty_returns_zero(self, extractor):
        assert extractor.entity_f1({"alice"}, set()) == pytest.approx(0.0)

    def test_subset_ref(self, extractor):
        # test is a subset of ref
        ref = {"a", "b", "c"}
        tst = {"a", "b"}
        # P = 2/2 = 1.0, R = 2/3, F1 = 2*(1*2/3)/(1+2/3) = (4/3)/(5/3) = 4/5
        assert extractor.entity_f1(ref, tst) == pytest.approx(4 / 5)

    def test_superset_tst(self, extractor):
        ref = {"a", "b"}
        tst = {"a", "b", "c"}
        # P = 2/3, R = 1.0, F1 = 2*(2/3*1)/(2/3+1) = (4/3)/(5/3) = 4/5
        assert extractor.entity_f1(ref, tst) == pytest.approx(4 / 5)


# ------------------------------------------------------------------
# batch_entity_f1
# ------------------------------------------------------------------

class TestBatchEntityF1:
    def test_batch_matches_single(self, extractor):
        pairs = [
            ({"a", "b"}, {"a", "b"}),
            ({"x"}, {"y"}),
            (set(), set()),
        ]
        results = extractor.batch_entity_f1(pairs)
        assert results[0] == pytest.approx(1.0)
        assert results[1] == pytest.approx(0.0)
        assert results[2] == pytest.approx(1.0)

    def test_empty_batch(self, extractor):
        assert extractor.batch_entity_f1([]) == []


# ------------------------------------------------------------------
# extract (regex fallback)
# ------------------------------------------------------------------

class TestExtractFallback:
    def test_numbers_found(self, extractor):
        entities = extractor.extract("There are 42 apples and 3.14 pies.")
        assert "42" in entities
        assert "3.14" in entities

    def test_capitalised_words_found(self, extractor):
        entities = extractor.extract("Alice met Bob in London.")
        assert "alice" in entities
        assert "bob" in entities
        assert "london" in entities

    def test_lowercase_words_not_extracted(self, extractor):
        entities = extractor.extract("the quick brown fox")
        # No capitalised words, no numbers → set should be empty or minimal
        caps = {e for e in entities if e.isalpha()}
        assert len(caps) == 0

    def test_empty_text(self, extractor):
        assert extractor.extract("") == set()

    def test_returns_lowercased(self, extractor):
        entities = extractor.extract("Paris is beautiful.")
        assert "paris" in entities
        # Should NOT contain 'Paris' (capital)
        assert "Paris" not in entities
