"""Tests for craving_mind.judge.dedup — DedupFilter class."""

import pytest
from craving_mind.judge.dedup import DedupFilter


@pytest.fixture
def dedup():
    return DedupFilter()


# ------------------------------------------------------------------
# task_hash
# ------------------------------------------------------------------

class TestTaskHash:
    def test_hash_is_deterministic(self, dedup):
        h1 = dedup.task_hash("Hello world, this is a test.", 0.3)
        h2 = dedup.task_hash("Hello world, this is a test.", 0.3)
        assert h1 == h2

    def test_different_ratio_different_hash(self, dedup):
        h1 = dedup.task_hash("Hello world.", 0.3)
        h2 = dedup.task_hash("Hello world.", 0.5)
        assert h1 != h2

    def test_different_text_different_hash(self, dedup):
        h1 = dedup.task_hash("Text A.", 0.3)
        h2 = dedup.task_hash("Text B.", 0.3)
        assert h1 != h2

    def test_hash_is_sha256_hex(self, dedup):
        h = dedup.task_hash("test", 0.5)
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_prefix_only_first_500_chars(self, dedup):
        # Two texts that differ only after 500 chars should collide.
        prefix = "A" * 500
        h1 = dedup.task_hash(prefix + "XXXXXX", 0.3)
        h2 = dedup.task_hash(prefix + "YYYYYY", 0.3)
        assert h1 == h2

    def test_short_text_no_crash(self, dedup):
        h = dedup.task_hash("hi", 0.2)
        assert isinstance(h, str)

    def test_custom_prefix_length(self, dedup):
        # With prefix_length=3, texts that differ at position 4 collide.
        h1 = dedup.task_hash("abcXXX", 0.5, prefix_length=3)
        h2 = dedup.task_hash("abcYYY", 0.5, prefix_length=3)
        assert h1 == h2
        # But texts differing in the first 3 chars should not.
        h3 = dedup.task_hash("xyzXXX", 0.5, prefix_length=3)
        assert h1 != h3


# ------------------------------------------------------------------
# Duplicate task detection
# ------------------------------------------------------------------

class TestDuplicateTask:
    def test_unseen_task_not_duplicate(self, dedup):
        assert dedup.is_duplicate_task("Some text.", 0.3) is False

    def test_seen_task_is_duplicate(self, dedup):
        dedup.mark_task_seen("Some text.", 0.3)
        assert dedup.is_duplicate_task("Some text.", 0.3) is True

    def test_different_ratio_not_duplicate(self, dedup):
        dedup.mark_task_seen("Some text.", 0.3)
        assert dedup.is_duplicate_task("Some text.", 0.5) is False

    def test_different_text_not_duplicate(self, dedup):
        dedup.mark_task_seen("Text A.", 0.3)
        assert dedup.is_duplicate_task("Text B.", 0.3) is False

    def test_mark_twice_idempotent(self, dedup):
        dedup.mark_task_seen("Text.", 0.4)
        dedup.mark_task_seen("Text.", 0.4)
        assert dedup.is_duplicate_task("Text.", 0.4) is True


# ------------------------------------------------------------------
# Amendment blacklist
# ------------------------------------------------------------------

class TestAmendmentBlacklist:
    def test_unseen_amendment_not_blacklisted(self, dedup):
        assert dedup.is_duplicate_amendment("patch content A") is False

    def test_blacklisted_amendment_detected(self, dedup):
        dedup.blacklist_amendment("bad patch")
        assert dedup.is_duplicate_amendment("bad patch") is True

    def test_different_patch_not_blacklisted(self, dedup):
        dedup.blacklist_amendment("bad patch")
        assert dedup.is_duplicate_amendment("good patch") is False

    def test_minor_change_creates_new_hash(self, dedup):
        dedup.blacklist_amendment("patch v1")
        # Even a single char difference escapes the filter (intentional spec trade-off).
        assert dedup.is_duplicate_amendment("patch v1 ") is False

    def test_amendment_hash_deterministic(self, dedup):
        h1 = dedup.amendment_hash("some patch")
        h2 = dedup.amendment_hash("some patch")
        assert h1 == h2

    def test_amendment_hash_is_sha256(self, dedup):
        h = dedup.amendment_hash("x")
        assert len(h) == 64
