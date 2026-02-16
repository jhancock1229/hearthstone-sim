"""Unit tests for analysis/metrics.py.

Tests cover:
- EpisodeMetrics dataclass
- compute_win_rate function
- MetricsTracker.record_episode
- MetricsTracker.record_eval
- MetricsTracker.summary (default and last_n)
- MetricsTracker.format_summary
"""

import pytest

from analysis.metrics import EpisodeMetrics, MetricsTracker, compute_win_rate


# ============================================================
# EpisodeMetrics
# ============================================================


class TestEpisodeMetrics:
    """Tests for the EpisodeMetrics dataclass."""

    def test_required_fields(self):
        ep = EpisodeMetrics(reward=1.0, length=20, win=True)
        assert ep.reward == 1.0
        assert ep.length == 20
        assert ep.win is True

    def test_default_losses(self):
        ep = EpisodeMetrics(reward=1.0, length=20, win=True)
        assert ep.policy_loss == 0.0
        assert ep.value_loss == 0.0
        assert ep.entropy == 0.0

    def test_custom_losses(self):
        ep = EpisodeMetrics(
            reward=-1.0, length=15, win=False,
            policy_loss=0.5, value_loss=0.3, entropy=0.1,
        )
        assert ep.policy_loss == 0.5
        assert ep.value_loss == 0.3
        assert ep.entropy == 0.1

    def test_negative_reward(self):
        ep = EpisodeMetrics(reward=-1.0, length=10, win=False)
        assert ep.reward == -1.0
        assert ep.win is False


# ============================================================
# compute_win_rate
# ============================================================


class TestComputeWinRate:
    """Tests for the compute_win_rate function."""

    def test_all_wins(self):
        assert compute_win_rate([1.0, 1.0, 1.0]) == 1.0

    def test_all_losses(self):
        assert compute_win_rate([-1.0, -1.0, -1.0]) == 0.0

    def test_mixed(self):
        assert compute_win_rate([1.0, -1.0, 1.0, -1.0]) == 0.5

    def test_empty_list(self):
        assert compute_win_rate([]) == 0.0

    def test_single_win(self):
        assert compute_win_rate([1.0]) == 1.0

    def test_single_loss(self):
        assert compute_win_rate([-1.0]) == 0.0

    def test_zero_reward_not_win(self):
        """Zero reward is not counted as a win."""
        assert compute_win_rate([0.0, 0.0]) == 0.0

    def test_fractional_positive_is_win(self):
        """Any positive reward counts as a win."""
        assert compute_win_rate([0.1, 0.5, -0.5]) == pytest.approx(2 / 3)


# ============================================================
# MetricsTracker Record Episode
# ============================================================


class TestMetricsTrackerRecording:
    """Tests for recording episodes."""

    def test_record_episode(self):
        tracker = MetricsTracker()
        tracker.record_episode(reward=1.0, length=20)
        assert len(tracker.episodes) == 1

    def test_record_multiple(self):
        tracker = MetricsTracker()
        tracker.record_episode(reward=1.0, length=20)
        tracker.record_episode(reward=-1.0, length=15)
        assert len(tracker.episodes) == 2

    def test_record_sets_win_from_reward(self):
        tracker = MetricsTracker()
        tracker.record_episode(reward=1.0, length=20)
        assert tracker.episodes[0].win is True
        tracker.record_episode(reward=-1.0, length=15)
        assert tracker.episodes[1].win is False

    def test_record_with_losses(self):
        tracker = MetricsTracker()
        tracker.record_episode(
            reward=1.0, length=20,
            losses={"policy_loss": 0.5, "value_loss": 0.3, "entropy": 0.1},
        )
        ep = tracker.episodes[0]
        assert ep.policy_loss == 0.5
        assert ep.value_loss == 0.3
        assert ep.entropy == 0.1

    def test_record_without_losses(self):
        tracker = MetricsTracker()
        tracker.record_episode(reward=1.0, length=20)
        ep = tracker.episodes[0]
        assert ep.policy_loss == 0.0
        assert ep.value_loss == 0.0
        assert ep.entropy == 0.0

    def test_record_partial_losses(self):
        tracker = MetricsTracker()
        tracker.record_episode(reward=1.0, length=20, losses={"policy_loss": 0.5})
        ep = tracker.episodes[0]
        assert ep.policy_loss == 0.5
        assert ep.value_loss == 0.0
        assert ep.entropy == 0.0

    def test_zero_reward_is_loss(self):
        tracker = MetricsTracker()
        tracker.record_episode(reward=0.0, length=10)
        assert tracker.episodes[0].win is False


# ============================================================
# MetricsTracker Record Eval
# ============================================================


class TestMetricsTrackerEval:
    """Tests for recording evaluation win rates."""

    def test_record_eval(self):
        tracker = MetricsTracker()
        tracker.record_eval(0.75)
        assert tracker.eval_win_rates == [0.75]

    def test_record_multiple_evals(self):
        tracker = MetricsTracker()
        tracker.record_eval(0.5)
        tracker.record_eval(0.6)
        tracker.record_eval(0.7)
        assert tracker.eval_win_rates == [0.5, 0.6, 0.7]

    def test_empty_eval_rates(self):
        tracker = MetricsTracker()
        assert tracker.eval_win_rates == []


# ============================================================
# MetricsTracker Summary
# ============================================================


class TestMetricsTrackerSummary:
    """Tests for the summary() method."""

    def test_empty_summary(self):
        tracker = MetricsTracker()
        s = tracker.summary()
        assert s["mean_reward"] == 0.0
        assert s["win_rate"] == 0.0
        assert s["mean_length"] == 0.0
        assert s["total_episodes"] == 0

    def test_summary_keys(self):
        tracker = MetricsTracker()
        tracker.record_episode(reward=1.0, length=20)
        s = tracker.summary()
        expected_keys = [
            "mean_reward", "win_rate", "mean_length",
            "mean_policy_loss", "mean_value_loss", "mean_entropy",
            "total_episodes",
        ]
        for key in expected_keys:
            assert key in s

    def test_summary_single_episode(self):
        tracker = MetricsTracker()
        tracker.record_episode(
            reward=1.0, length=20,
            losses={"policy_loss": 0.5, "value_loss": 0.3, "entropy": 0.1},
        )
        s = tracker.summary()
        assert float(s["mean_reward"]) == pytest.approx(1.0)
        assert float(s["win_rate"]) == pytest.approx(1.0)
        assert float(s["mean_length"]) == pytest.approx(20.0)
        assert float(s["mean_policy_loss"]) == pytest.approx(0.5)
        assert float(s["mean_value_loss"]) == pytest.approx(0.3)
        assert float(s["mean_entropy"]) == pytest.approx(0.1)
        assert s["total_episodes"] == 1

    def test_summary_multiple_episodes(self):
        tracker = MetricsTracker()
        tracker.record_episode(reward=1.0, length=20)
        tracker.record_episode(reward=-1.0, length=30)
        s = tracker.summary()
        assert float(s["mean_reward"]) == pytest.approx(0.0)
        assert float(s["win_rate"]) == pytest.approx(0.5)
        assert float(s["mean_length"]) == pytest.approx(25.0)
        assert s["total_episodes"] == 2

    def test_summary_last_n(self):
        """Summary with last_n only considers recent episodes."""
        tracker = MetricsTracker()
        # Record 5 losses then 5 wins
        for _ in range(5):
            tracker.record_episode(reward=-1.0, length=10)
        for _ in range(5):
            tracker.record_episode(reward=1.0, length=20)
        # last 5 should be all wins
        s = tracker.summary(last_n=5)
        assert float(s["win_rate"]) == pytest.approx(1.0)
        assert float(s["mean_reward"]) == pytest.approx(1.0)
        assert s["total_episodes"] == 10  # total is always all episodes

    def test_summary_last_n_larger_than_episodes(self):
        """last_n larger than episode count uses all episodes."""
        tracker = MetricsTracker()
        tracker.record_episode(reward=1.0, length=20)
        tracker.record_episode(reward=-1.0, length=30)
        s = tracker.summary(last_n=1000)
        assert float(s["win_rate"]) == pytest.approx(0.5)


# ============================================================
# MetricsTracker format_summary
# ============================================================


class TestMetricsTrackerFormatSummary:
    """Tests for the format_summary() method."""

    def test_format_returns_string(self):
        tracker = MetricsTracker()
        result = tracker.format_summary()
        assert isinstance(result, str)

    def test_format_empty_tracker(self):
        tracker = MetricsTracker()
        result = tracker.format_summary()
        assert "Episodes: 0" in result

    def test_format_contains_key_metrics(self):
        tracker = MetricsTracker()
        tracker.record_episode(reward=1.0, length=20)
        result = tracker.format_summary()
        assert "Win Rate" in result
        assert "Reward" in result
        assert "Length" in result
        assert "P.Loss" in result
        assert "V.Loss" in result
        assert "Entropy" in result

    def test_format_with_data(self):
        tracker = MetricsTracker()
        tracker.record_episode(reward=1.0, length=20)
        tracker.record_episode(reward=1.0, length=30)
        result = tracker.format_summary()
        assert "Episodes: 2" in result
        assert "100.0%" in result  # 100% win rate

    def test_format_respects_last_n(self):
        tracker = MetricsTracker()
        for _ in range(5):
            tracker.record_episode(reward=-1.0, length=10)
        for _ in range(5):
            tracker.record_episode(reward=1.0, length=20)
        result = tracker.format_summary(last_n=5)
        # Last 5 are all wins, but total_episodes is 10
        assert "Episodes: 10" in result
        assert "100.0%" in result
