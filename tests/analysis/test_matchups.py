"""Unit tests for matchup matrix analysis.

Following TDD: these tests define the contract for matchups.py which
builds and queries deck-vs-deck win rate matrices.

Components:
- MatchupMatrix: NxN win rate matrix for deck/archetype matchups
"""

import pytest

from analysis.matchups import MatchupMatrix


# ============================================================
# MatchupMatrix Creation Tests
# ============================================================


class TestMatchupMatrixCreation:
    """Tests for MatchupMatrix initialization."""

    def test_create_empty(self):
        """Can create an empty matchup matrix."""
        matrix = MatchupMatrix()
        assert matrix.size == 0

    def test_create_with_labels(self):
        """Can create matrix with deck labels."""
        matrix = MatchupMatrix(labels=["Aggro", "Control", "Midrange"])
        assert matrix.size == 3
        assert matrix.labels == ["Aggro", "Control", "Midrange"]


# ============================================================
# MatchupMatrix Recording Tests
# ============================================================


class TestMatchupMatrixRecording:
    """Tests for recording results into the matrix."""

    def test_record_result(self):
        """Can record a win/loss result."""
        matrix = MatchupMatrix()
        matrix.record("Aggro", "Control", wins=7, losses=3)
        assert matrix.get_wins("Aggro", "Control") == 7
        assert matrix.get_losses("Aggro", "Control") == 3

    def test_record_accumulates(self):
        """Multiple records for same matchup accumulate."""
        matrix = MatchupMatrix()
        matrix.record("Aggro", "Control", wins=5, losses=2)
        matrix.record("Aggro", "Control", wins=3, losses=1)
        assert matrix.get_wins("Aggro", "Control") == 8
        assert matrix.get_losses("Aggro", "Control") == 3

    def test_record_auto_adds_labels(self):
        """Recording with new labels auto-adds them."""
        matrix = MatchupMatrix()
        matrix.record("A", "B", wins=1, losses=0)
        assert "A" in matrix.labels
        assert "B" in matrix.labels

    def test_record_symmetric(self):
        """A's wins vs B are B's losses vs A."""
        matrix = MatchupMatrix()
        matrix.record("A", "B", wins=7, losses=3)
        assert matrix.get_wins("B", "A") == 3
        assert matrix.get_losses("B", "A") == 7


# ============================================================
# MatchupMatrix Win Rate Queries
# ============================================================


class TestMatchupMatrixWinRates:
    """Tests for win rate calculations."""

    def test_win_rate(self):
        """Can get win rate for a specific matchup."""
        matrix = MatchupMatrix()
        matrix.record("A", "B", wins=7, losses=3)
        assert matrix.get_win_rate("A", "B") == pytest.approx(0.7)

    def test_win_rate_reverse(self):
        """Reverse matchup is complement."""
        matrix = MatchupMatrix()
        matrix.record("A", "B", wins=7, losses=3)
        assert matrix.get_win_rate("B", "A") == pytest.approx(0.3)

    def test_win_rate_no_data_returns_none(self):
        """Win rate with no games returns None."""
        matrix = MatchupMatrix()
        assert matrix.get_win_rate("X", "Y") is None

    def test_overall_win_rate(self):
        """Can get overall win rate across all matchups."""
        matrix = MatchupMatrix()
        matrix.record("A", "B", wins=7, losses=3)
        matrix.record("A", "C", wins=5, losses=5)
        wr = matrix.get_overall_win_rate("A")
        # 12 wins out of 20 games
        assert wr == pytest.approx(12 / 20)

    def test_overall_win_rate_no_data(self):
        """Overall win rate with no games returns 0.0."""
        matrix = MatchupMatrix()
        assert matrix.get_overall_win_rate("X") == 0.0

    def test_total_games_for_matchup(self):
        """Can get total games played for a matchup."""
        matrix = MatchupMatrix()
        matrix.record("A", "B", wins=7, losses=3)
        assert matrix.get_total_games("A", "B") == 10


# ============================================================
# MatchupMatrix Table Export
# ============================================================


class TestMatchupMatrixTable:
    """Tests for matrix table generation."""

    def test_to_dict(self):
        """Can export matrix as nested dict."""
        matrix = MatchupMatrix()
        matrix.record("A", "B", wins=7, losses=3)
        matrix.record("A", "C", wins=6, losses=4)
        matrix.record("B", "C", wins=5, losses=5)

        d = matrix.to_dict()
        assert d["A"]["B"] == pytest.approx(0.7)
        assert d["A"]["C"] == pytest.approx(0.6)
        assert d["B"]["A"] == pytest.approx(0.3)
        assert d["B"]["C"] == pytest.approx(0.5)

    def test_to_dict_empty(self):
        """Empty matrix returns empty dict."""
        matrix = MatchupMatrix()
        assert matrix.to_dict() == {}

    def test_get_best_matchup(self):
        """Can find the best matchup for a deck."""
        matrix = MatchupMatrix()
        matrix.record("A", "B", wins=7, losses=3)
        matrix.record("A", "C", wins=9, losses=1)
        best = matrix.get_best_matchup("A")
        assert best == "C"

    def test_get_worst_matchup(self):
        """Can find the worst matchup for a deck."""
        matrix = MatchupMatrix()
        matrix.record("A", "B", wins=7, losses=3)
        matrix.record("A", "C", wins=2, losses=8)
        worst = matrix.get_worst_matchup("A")
        assert worst == "C"

    def test_best_worst_no_data(self):
        """Best/worst with no data returns None."""
        matrix = MatchupMatrix()
        assert matrix.get_best_matchup("X") is None
        assert matrix.get_worst_matchup("X") is None

    def test_size_updates_on_record(self):
        """Matrix size grows as new labels are added."""
        matrix = MatchupMatrix()
        assert matrix.size == 0
        matrix.record("A", "B", wins=1, losses=0)
        assert matrix.size == 2
        matrix.record("A", "C", wins=1, losses=0)
        assert matrix.size == 3
