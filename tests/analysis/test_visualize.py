"""Unit tests for visualization data generation.

Following TDD: these tests define the contract for visualize.py which
generates chart-ready data for training curves, action distributions,
Elo progression, and matchup heatmaps.

Note: We test data generation, not rendering. Actual plotting
(matplotlib, etc.) is optional and tested separately if present.

Components:
- TrainingCurveData: Extracts training curve data from MetricsTracker
- ActionDistributionData: Summarizes action type frequencies
- EloProgressionData: Extracts Elo history for plotting
- MatchupHeatmapData: Converts MatchupMatrix to heatmap format
"""

import pytest

from analysis.visualize import (
    training_curve_data,
    action_distribution_data,
    elo_progression_data,
    matchup_heatmap_data,
)
from analysis.metrics import MetricsTracker
from analysis.matchups import MatchupMatrix
from agents.rl.self_play import EloTracker


# ============================================================
# Training Curve Data Tests
# ============================================================


class TestTrainingCurveData:
    """Tests for extracting training curve data."""

    def test_empty_tracker(self):
        """Empty tracker produces empty data."""
        tracker = MetricsTracker()
        data = training_curve_data(tracker)
        assert data["episodes"] == []
        assert data["rewards"] == []
        assert data["win_rates"] == []

    def test_basic_data(self):
        """Can extract rewards and lengths over episodes."""
        tracker = MetricsTracker()
        tracker.record_episode(reward=1.0, length=20)
        tracker.record_episode(reward=-1.0, length=30)
        tracker.record_episode(reward=1.0, length=25)
        data = training_curve_data(tracker)
        assert data["episodes"] == [1, 2, 3]
        assert data["rewards"] == [1.0, -1.0, 1.0]
        assert data["lengths"] == [20, 30, 25]

    def test_includes_losses(self):
        """Data includes policy/value loss if recorded."""
        tracker = MetricsTracker()
        tracker.record_episode(
            reward=1.0, length=20,
            losses={"policy_loss": 0.5, "value_loss": 0.3, "entropy": 0.1},
        )
        data = training_curve_data(tracker)
        assert data["policy_losses"] == [0.5]
        assert data["value_losses"] == [0.3]
        assert data["entropies"] == [0.1]

    def test_windowed_win_rate(self):
        """Can compute rolling win rate."""
        tracker = MetricsTracker()
        for r in [1, 1, -1, 1, -1]:
            tracker.record_episode(reward=float(r), length=10)
        data = training_curve_data(tracker, window=3)
        assert "rolling_win_rates" in data
        # Window of 3: first 2 entries have smaller windows
        assert len(data["rolling_win_rates"]) == 5

    def test_eval_win_rates_included(self):
        """Eval win rates from tracker are included."""
        tracker = MetricsTracker()
        tracker.record_eval(0.55)
        tracker.record_eval(0.60)
        data = training_curve_data(tracker)
        assert data["eval_win_rates"] == [0.55, 0.60]


# ============================================================
# Action Distribution Data Tests
# ============================================================


class TestActionDistributionData:
    """Tests for action distribution summaries."""

    def test_empty_actions(self):
        """Empty action list produces empty counts."""
        data = action_distribution_data([])
        assert data["total"] == 0

    def test_basic_counts(self):
        """Counts action types correctly."""
        actions = ["PLAY_CARD", "PLAY_CARD", "ATTACK", "END_TURN", "HERO_POWER"]
        data = action_distribution_data(actions)
        assert data["PLAY_CARD"] == 2
        assert data["ATTACK"] == 1
        assert data["END_TURN"] == 1
        assert data["HERO_POWER"] == 1
        assert data["total"] == 5

    def test_proportions(self):
        """Can get proportions instead of counts."""
        actions = ["PLAY_CARD", "PLAY_CARD", "ATTACK", "ATTACK"]
        data = action_distribution_data(actions, proportions=True)
        assert data["PLAY_CARD"] == pytest.approx(0.5)
        assert data["ATTACK"] == pytest.approx(0.5)

    def test_unknown_action_type_counted(self):
        """Unknown action types are still counted."""
        actions = ["PLAY_CARD", "UNKNOWN_TYPE"]
        data = action_distribution_data(actions)
        assert data["UNKNOWN_TYPE"] == 1


# ============================================================
# Elo Progression Data Tests
# ============================================================


class TestEloProgressionData:
    """Tests for Elo progression data extraction."""

    def test_empty_tracker(self):
        """Empty Elo tracker gives empty data."""
        tracker = EloTracker()
        data = elo_progression_data(tracker)
        assert data["matches"] == []
        assert data["agents"] == {}

    def test_single_match(self):
        """Data reflects one match's rating changes."""
        tracker = EloTracker()
        tracker.set_rating("a", 1500)
        tracker.set_rating("b", 1500)
        tracker.record_match("a", "b", winner="a")
        data = elo_progression_data(tracker)
        assert len(data["matches"]) == 1
        assert data["matches"][0]["winner"] == "a"

    def test_rating_progression(self):
        """Can extract rating values over time for an agent."""
        tracker = EloTracker()
        tracker.set_rating("a", 1500)
        tracker.set_rating("b", 1500)
        tracker.record_match("a", "b", winner="a")
        tracker.record_match("a", "b", winner="a")
        data = elo_progression_data(tracker)
        # Agent "a" should have 2 data points
        assert len(data["agents"]["a"]) == 2
        # Rating should increase
        assert data["agents"]["a"][1] > data["agents"]["a"][0]

    def test_multiple_agents(self):
        """Tracks multiple agents' progressions."""
        tracker = EloTracker()
        tracker.record_match("a", "b", winner="a")
        tracker.record_match("b", "c", winner="b")
        data = elo_progression_data(tracker)
        assert "a" in data["agents"]
        assert "b" in data["agents"]
        assert "c" in data["agents"]


# ============================================================
# Matchup Heatmap Data Tests
# ============================================================


class TestMatchupHeatmapData:
    """Tests for matchup heatmap data conversion."""

    def test_empty_matrix(self):
        """Empty matrix gives empty heatmap."""
        matrix = MatchupMatrix()
        data = matchup_heatmap_data(matrix)
        assert data["labels"] == []
        assert data["values"] == []

    def test_basic_heatmap(self):
        """Heatmap contains labels and 2D value grid."""
        matrix = MatchupMatrix()
        matrix.record("A", "B", wins=7, losses=3)
        matrix.record("A", "C", wins=6, losses=4)
        matrix.record("B", "C", wins=5, losses=5)
        data = matchup_heatmap_data(matrix)
        assert len(data["labels"]) == 3
        assert len(data["values"]) == 3
        assert len(data["values"][0]) == 3

    def test_diagonal_is_none_or_half(self):
        """Diagonal entries (mirror matchup) are 0.5 or None."""
        matrix = MatchupMatrix()
        matrix.record("A", "B", wins=7, losses=3)
        data = matchup_heatmap_data(matrix)
        # Find diagonal positions
        for i in range(len(data["labels"])):
            assert data["values"][i][i] in (None, 0.5)

    def test_heatmap_values_complement(self):
        """values[i][j] + values[j][i] == 1.0 for non-diagonal."""
        matrix = MatchupMatrix()
        matrix.record("A", "B", wins=7, losses=3)
        data = matchup_heatmap_data(matrix)
        labels = data["labels"]
        values = data["values"]
        i_a = labels.index("A")
        i_b = labels.index("B")
        v_ab = values[i_a][i_b]
        v_ba = values[i_b][i_a]
        if v_ab is not None and v_ba is not None:
            assert v_ab + v_ba == pytest.approx(1.0)
