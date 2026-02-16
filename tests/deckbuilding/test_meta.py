"""Unit tests for meta-game tracker.

Following TDD: these tests define the contract for meta.py which
catalogs deck archetypes, tracks their win rates, and identifies
counter-strategies and rock-paper-scissors dynamics.

Components:
- Archetype: Represents a deck archetype with name, key cards, strategy
- MetaTracker: Tracks archetype popularity, win rates, and matchup trends
"""

import pytest

from deckbuilding.meta import Archetype, MetaTracker


# ============================================================
# Archetype Tests
# ============================================================


class TestArchetypeCreation:
    """Tests for Archetype dataclass."""

    def test_create_archetype(self):
        """Can create an archetype with name and description."""
        arch = Archetype(name="Aggro", description="Fast aggressive deck")
        assert arch.name == "Aggro"
        assert arch.description == "Fast aggressive deck"

    def test_archetype_with_key_cards(self):
        """Archetype can list key cards."""
        arch = Archetype(
            name="Big Minions",
            description="Large minion strategy",
            key_cards=["Giant", "Brute"],
        )
        assert "Giant" in arch.key_cards
        assert len(arch.key_cards) == 2

    def test_archetype_defaults(self):
        """Default archetype has empty key cards."""
        arch = Archetype(name="test", description="test")
        assert arch.key_cards == []

    def test_archetype_with_strategy_tag(self):
        """Archetype can have a strategy tag."""
        arch = Archetype(
            name="Control",
            description="Late game control",
            strategy="control",
        )
        assert arch.strategy == "control"

    def test_default_strategy_is_empty(self):
        """Default strategy is empty string."""
        arch = Archetype(name="test", description="test")
        assert arch.strategy == ""


# ============================================================
# MetaTracker Creation Tests
# ============================================================


class TestMetaTrackerCreation:
    """Tests for MetaTracker initialization."""

    def test_create_empty(self):
        """Can create an empty meta tracker."""
        tracker = MetaTracker()
        assert len(tracker.archetypes) == 0

    def test_register_archetype(self):
        """Can register an archetype."""
        tracker = MetaTracker()
        arch = Archetype(name="Aggro", description="Fast deck")
        tracker.register_archetype(arch)
        assert len(tracker.archetypes) == 1
        assert tracker.archetypes["Aggro"] is arch

    def test_register_multiple_archetypes(self):
        """Can register multiple archetypes."""
        tracker = MetaTracker()
        tracker.register_archetype(Archetype(name="Aggro", description="Fast"))
        tracker.register_archetype(Archetype(name="Control", description="Slow"))
        assert len(tracker.archetypes) == 2

    def test_duplicate_name_overwrites(self):
        """Registering same name overwrites the previous."""
        tracker = MetaTracker()
        tracker.register_archetype(Archetype(name="Aggro", description="v1"))
        tracker.register_archetype(Archetype(name="Aggro", description="v2"))
        assert len(tracker.archetypes) == 1
        assert tracker.archetypes["Aggro"].description == "v2"


# ============================================================
# MetaTracker Match Recording Tests
# ============================================================


class TestMetaTrackerMatchRecording:
    """Tests for recording match results between archetypes."""

    def _make_tracker(self):
        """Create tracker with Aggro and Control."""
        tracker = MetaTracker()
        tracker.register_archetype(Archetype(name="Aggro", description="Fast"))
        tracker.register_archetype(Archetype(name="Control", description="Slow"))
        return tracker

    def test_record_match(self):
        """Can record a match result."""
        tracker = self._make_tracker()
        tracker.record_match("Aggro", "Control", winner="Aggro")
        assert tracker.total_matches == 1

    def test_record_draw(self):
        """Can record a draw."""
        tracker = self._make_tracker()
        tracker.record_match("Aggro", "Control", winner=None)
        assert tracker.total_matches == 1

    def test_record_multiple_matches(self):
        """Can record many matches."""
        tracker = self._make_tracker()
        for _ in range(10):
            tracker.record_match("Aggro", "Control", winner="Aggro")
        for _ in range(5):
            tracker.record_match("Aggro", "Control", winner="Control")
        assert tracker.total_matches == 15

    def test_unknown_archetype_auto_registers(self):
        """Recording a match with unknown archetype auto-registers it."""
        tracker = MetaTracker()
        tracker.record_match("NewArch", "OtherArch", winner="NewArch")
        assert "NewArch" in tracker.archetypes
        assert "OtherArch" in tracker.archetypes


# ============================================================
# MetaTracker Win Rate Queries
# ============================================================


class TestMetaTrackerWinRates:
    """Tests for querying archetype win rates."""

    def _make_tracker_with_data(self):
        """Create tracker with match data."""
        tracker = MetaTracker()
        tracker.register_archetype(Archetype(name="Aggro", description="Fast"))
        tracker.register_archetype(Archetype(name="Control", description="Slow"))
        tracker.register_archetype(Archetype(name="Midrange", description="Mid"))

        # Aggro beats Control 7/10
        for _ in range(7):
            tracker.record_match("Aggro", "Control", winner="Aggro")
        for _ in range(3):
            tracker.record_match("Aggro", "Control", winner="Control")

        # Control beats Midrange 6/10
        for _ in range(6):
            tracker.record_match("Control", "Midrange", winner="Control")
        for _ in range(4):
            tracker.record_match("Control", "Midrange", winner="Midrange")

        # Midrange beats Aggro 8/10
        for _ in range(8):
            tracker.record_match("Midrange", "Aggro", winner="Midrange")
        for _ in range(2):
            tracker.record_match("Midrange", "Aggro", winner="Aggro")

        return tracker

    def test_overall_win_rate(self):
        """Can get overall win rate for an archetype."""
        tracker = self._make_tracker_with_data()
        wr = tracker.get_win_rate("Aggro")
        # Aggro: 7 wins vs Control + 2 wins vs Midrange = 9 out of 20
        assert wr == pytest.approx(9 / 20)

    def test_matchup_win_rate(self):
        """Can get win rate for a specific matchup."""
        tracker = self._make_tracker_with_data()
        wr = tracker.get_matchup_win_rate("Aggro", "Control")
        assert wr == pytest.approx(7 / 10)

    def test_matchup_win_rate_reverse(self):
        """Reverse matchup gives complement."""
        tracker = self._make_tracker_with_data()
        wr_ac = tracker.get_matchup_win_rate("Aggro", "Control")
        wr_ca = tracker.get_matchup_win_rate("Control", "Aggro")
        assert wr_ac + wr_ca == pytest.approx(1.0)

    def test_no_data_returns_zero(self):
        """Win rate with no data returns 0.0."""
        tracker = MetaTracker()
        assert tracker.get_win_rate("Unknown") == 0.0

    def test_no_matchup_data_returns_zero(self):
        """Matchup with no games returns 0.0."""
        tracker = self._make_tracker_with_data()
        assert tracker.get_matchup_win_rate("Aggro", "Unknown") == 0.0

    def test_popularity(self):
        """Can get archetype popularity (proportion of matches)."""
        tracker = self._make_tracker_with_data()
        pop = tracker.get_popularity("Aggro")
        # Aggro appears in 20 out of 30 total matches
        assert pop == pytest.approx(20 / 30)

    def test_popularity_no_matches(self):
        """Popularity with no matches returns 0.0."""
        tracker = MetaTracker()
        assert tracker.get_popularity("X") == 0.0


# ============================================================
# MetaTracker RPS Detection Tests
# ============================================================


class TestMetaTrackerCounters:
    """Tests for counter-strategy and RPS detection."""

    def _make_tracker_with_data(self):
        """Same data as win rate tests — clear RPS triangle."""
        tracker = MetaTracker()
        for name in ["Aggro", "Control", "Midrange"]:
            tracker.register_archetype(Archetype(name=name, description=name))

        for _ in range(7):
            tracker.record_match("Aggro", "Control", winner="Aggro")
        for _ in range(3):
            tracker.record_match("Aggro", "Control", winner="Control")
        for _ in range(6):
            tracker.record_match("Control", "Midrange", winner="Control")
        for _ in range(4):
            tracker.record_match("Control", "Midrange", winner="Midrange")
        for _ in range(8):
            tracker.record_match("Midrange", "Aggro", winner="Midrange")
        for _ in range(2):
            tracker.record_match("Midrange", "Aggro", winner="Aggro")
        return tracker

    def test_get_counters(self):
        """Can find archetypes that counter a given one."""
        tracker = self._make_tracker_with_data()
        counters = tracker.get_counters("Aggro", threshold=0.55)
        # Midrange beats Aggro 80% → is a counter
        assert "Midrange" in counters

    def test_get_counters_respects_threshold(self):
        """Only matchups above threshold are returned as counters."""
        tracker = self._make_tracker_with_data()
        counters = tracker.get_counters("Aggro", threshold=0.90)
        # 80% < 90%, so Midrange shouldn't qualify
        assert "Midrange" not in counters

    def test_get_favorable_matchups(self):
        """Can find archetypes that a given one beats."""
        tracker = self._make_tracker_with_data()
        favorable = tracker.get_favorable_matchups("Aggro", threshold=0.55)
        assert "Control" in favorable

    def test_get_meta_summary(self):
        """Can get a full meta summary with all archetypes."""
        tracker = self._make_tracker_with_data()
        summary = tracker.get_meta_summary()
        assert len(summary) == 3
        for entry in summary:
            assert "name" in entry
            assert "win_rate" in entry
            assert "popularity" in entry
            assert "matches" in entry

    def test_meta_summary_sorted_by_win_rate(self):
        """Meta summary is sorted by win rate descending."""
        tracker = self._make_tracker_with_data()
        summary = tracker.get_meta_summary()
        win_rates = [e["win_rate"] for e in summary]
        assert win_rates == sorted(win_rates, reverse=True)
