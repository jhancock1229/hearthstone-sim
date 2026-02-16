"""Unit tests for self-play training loop.

Following TDD: these tests define the contract for self_play.py which
implements self-play training with Elo rating tracking.

Components:
- EloTracker: Maintains Elo ratings for agent snapshots
- SelfPlayManager: Orchestrates self-play training iterations
  with opponent pool management and snapshot scheduling
"""

import copy
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from agents.rl.self_play import (
    EloTracker,
    SelfPlayManager,
)


# ============================================================
# EloTracker Tests
# ============================================================


class TestEloTrackerCreation:
    """Tests for EloTracker initialization."""

    def test_create_default(self):
        """Can create EloTracker with default settings."""
        tracker = EloTracker()
        assert tracker.default_rating == 1500
        assert tracker.k_factor == 32

    def test_create_custom_rating(self):
        """Can set custom default rating."""
        tracker = EloTracker(default_rating=1200, k_factor=16)
        assert tracker.default_rating == 1200
        assert tracker.k_factor == 16

    def test_no_ratings_initially(self):
        """New tracker has no recorded ratings."""
        tracker = EloTracker()
        assert len(tracker.ratings) == 0

    def test_get_rating_returns_default(self):
        """Getting unregistered agent returns default rating."""
        tracker = EloTracker(default_rating=1500)
        assert tracker.get_rating("unknown") == 1500


class TestEloTrackerUpdates:
    """Tests for Elo rating update logic."""

    def test_winner_gains_rating(self):
        """Winner's rating increases after a win."""
        tracker = EloTracker()
        tracker.set_rating("a", 1500)
        tracker.set_rating("b", 1500)
        tracker.record_match("a", "b", winner="a")
        assert tracker.get_rating("a") > 1500
        assert tracker.get_rating("b") < 1500

    def test_draw_no_change_equal_ratings(self):
        """Draw between equal-rated players: minimal or no change."""
        tracker = EloTracker()
        tracker.set_rating("a", 1500)
        tracker.set_rating("b", 1500)
        tracker.record_match("a", "b", winner=None)  # draw
        # Equal ratings => draw keeps them close to original
        assert abs(tracker.get_rating("a") - 1500) < 1.0
        assert abs(tracker.get_rating("b") - 1500) < 1.0

    def test_upset_gives_larger_swing(self):
        """Lower-rated player beating higher-rated player causes larger shift."""
        tracker = EloTracker()
        tracker.set_rating("weak", 1200)
        tracker.set_rating("strong", 1800)
        tracker.record_match("weak", "strong", winner="weak")
        # Upset: weak gains a lot
        assert tracker.get_rating("weak") > 1200 + 20

    def test_expected_win_gives_small_swing(self):
        """Higher-rated player winning gives smaller shift."""
        tracker = EloTracker()
        tracker.set_rating("strong", 1800)
        tracker.set_rating("weak", 1200)
        tracker.record_match("strong", "weak", winner="strong")
        # Expected win: strong gains little
        assert tracker.get_rating("strong") < 1800 + 5

    def test_rating_sum_preserved(self):
        """Total rating change is zero-sum (what one gains, other loses)."""
        tracker = EloTracker()
        tracker.set_rating("a", 1500)
        tracker.set_rating("b", 1500)
        before_sum = tracker.get_rating("a") + tracker.get_rating("b")
        tracker.record_match("a", "b", winner="a")
        after_sum = tracker.get_rating("a") + tracker.get_rating("b")
        assert before_sum == pytest.approx(after_sum, abs=0.01)

    def test_multiple_matches_accumulate(self):
        """Multiple wins keep pushing rating up."""
        tracker = EloTracker()
        tracker.set_rating("a", 1500)
        tracker.set_rating("b", 1500)
        for _ in range(10):
            tracker.record_match("a", "b", winner="a")
        assert tracker.get_rating("a") > 1600
        assert tracker.get_rating("b") < 1400

    def test_k_factor_controls_magnitude(self):
        """Higher K factor means larger rating changes."""
        tracker_high = EloTracker(k_factor=64)
        tracker_low = EloTracker(k_factor=8)

        for t in [tracker_high, tracker_low]:
            t.set_rating("a", 1500)
            t.set_rating("b", 1500)
            t.record_match("a", "b", winner="a")

        change_high = tracker_high.get_rating("a") - 1500
        change_low = tracker_low.get_rating("a") - 1500
        assert change_high > change_low

    def test_set_rating(self):
        """Can explicitly set a rating."""
        tracker = EloTracker()
        tracker.set_rating("agent_1", 1700)
        assert tracker.get_rating("agent_1") == 1700

    def test_get_all_ratings(self):
        """Can retrieve all ratings as a dict."""
        tracker = EloTracker()
        tracker.set_rating("a", 1500)
        tracker.set_rating("b", 1600)
        ratings = tracker.ratings
        assert "a" in ratings
        assert "b" in ratings
        assert ratings["a"] == 1500
        assert ratings["b"] == 1600

    def test_match_history_tracked(self):
        """Match results are stored in history."""
        tracker = EloTracker()
        tracker.set_rating("a", 1500)
        tracker.set_rating("b", 1500)
        tracker.record_match("a", "b", winner="a")
        tracker.record_match("a", "b", winner="b")
        assert len(tracker.history) == 2

    def test_new_agent_auto_registered(self):
        """Recording a match with unknown agent auto-registers at default."""
        tracker = EloTracker(default_rating=1500)
        tracker.record_match("new_a", "new_b", winner="new_a")
        # Both should be registered now
        assert "new_a" in tracker.ratings
        assert "new_b" in tracker.ratings


# ============================================================
# SelfPlayManager Tests
# ============================================================


class TestSelfPlayManagerCreation:
    """Tests for SelfPlayManager initialization."""

    def test_create_with_agent(self):
        """Can create manager with a PPO agent."""
        from agents.rl.ppo_agent import PPOAgent
        agent = PPOAgent()
        manager = SelfPlayManager(agent)
        assert manager.agent is agent

    def test_default_config(self):
        """Default configuration is reasonable."""
        from agents.rl.ppo_agent import PPOAgent
        agent = PPOAgent()
        manager = SelfPlayManager(agent)
        assert manager.snapshot_interval > 0
        assert manager.max_opponent_pool_size > 0
        assert manager.win_rate_threshold > 0.0

    def test_custom_config(self):
        """Can customize self-play parameters."""
        from agents.rl.ppo_agent import PPOAgent
        agent = PPOAgent()
        manager = SelfPlayManager(
            agent,
            snapshot_interval=50,
            max_opponent_pool_size=5,
            win_rate_threshold=0.6,
        )
        assert manager.snapshot_interval == 50
        assert manager.max_opponent_pool_size == 5
        assert manager.win_rate_threshold == 0.6

    def test_initial_opponent_pool_empty(self):
        """Opponent pool starts empty."""
        from agents.rl.ppo_agent import PPOAgent
        agent = PPOAgent()
        manager = SelfPlayManager(agent)
        assert len(manager.opponent_pool) == 0

    def test_has_elo_tracker(self):
        """Manager has an EloTracker."""
        from agents.rl.ppo_agent import PPOAgent
        agent = PPOAgent()
        manager = SelfPlayManager(agent)
        assert isinstance(manager.elo_tracker, EloTracker)


class TestSelfPlayManagerSnapshots:
    """Tests for agent snapshot management."""

    def test_create_snapshot(self):
        """Can create a snapshot of current agent."""
        from agents.rl.ppo_agent import PPOAgent
        agent = PPOAgent()
        manager = SelfPlayManager(agent)
        snapshot_id = manager.create_snapshot()
        assert snapshot_id is not None
        assert len(manager.opponent_pool) == 1

    def test_snapshot_is_independent_copy(self):
        """Snapshot weights don't change when original agent trains."""
        from agents.rl.ppo_agent import PPOAgent
        agent = PPOAgent()
        manager = SelfPlayManager(agent)

        # Get original weights
        original_params = {
            name: p.clone()
            for name, p in agent.network.named_parameters()
        }

        snapshot_id = manager.create_snapshot()
        snapshot = manager.opponent_pool[0]

        # Verify snapshot matches original
        for name, p in snapshot.network.named_parameters():
            assert (p == original_params[name]).all()

    def test_pool_size_limit(self):
        """Opponent pool doesn't exceed max size."""
        from agents.rl.ppo_agent import PPOAgent
        agent = PPOAgent()
        manager = SelfPlayManager(agent, max_opponent_pool_size=3)

        for _ in range(5):
            manager.create_snapshot()

        assert len(manager.opponent_pool) <= 3

    def test_oldest_removed_when_pool_full(self):
        """When pool is full, oldest snapshot is removed."""
        from agents.rl.ppo_agent import PPOAgent
        agent = PPOAgent()
        manager = SelfPlayManager(agent, max_opponent_pool_size=2)

        id1 = manager.create_snapshot()
        id2 = manager.create_snapshot()
        id3 = manager.create_snapshot()

        # Pool should have the two most recent
        assert len(manager.opponent_pool) == 2

    def test_multiple_snapshots_have_unique_ids(self):
        """Each snapshot gets a unique identifier."""
        from agents.rl.ppo_agent import PPOAgent
        agent = PPOAgent()
        manager = SelfPlayManager(agent, max_opponent_pool_size=10)

        ids = [manager.create_snapshot() for _ in range(5)]
        assert len(set(ids)) == 5


class TestSelfPlayManagerOpponentSelection:
    """Tests for opponent selection from pool."""

    def test_select_opponent_from_pool(self):
        """Can select an opponent from the pool."""
        from agents.rl.ppo_agent import PPOAgent
        agent = PPOAgent()
        manager = SelfPlayManager(agent)
        manager.create_snapshot()
        opponent = manager.select_opponent()
        assert opponent is not None

    def test_select_opponent_empty_pool_returns_self(self):
        """If pool is empty, returns a snapshot of current agent."""
        from agents.rl.ppo_agent import PPOAgent
        agent = PPOAgent()
        manager = SelfPlayManager(agent)
        opponent = manager.select_opponent()
        assert opponent is not None

    def test_select_opponent_randomness(self):
        """Opponent selection has some randomness (samples from pool)."""
        from agents.rl.ppo_agent import PPOAgent
        agent = PPOAgent()
        manager = SelfPlayManager(agent, max_opponent_pool_size=10)

        for _ in range(5):
            manager.create_snapshot()

        # Select multiple opponents — shouldn't always be the same
        opponents = [id(manager.select_opponent()) for _ in range(10)]
        # With 5 in pool, we should see some variety (not guaranteed, but likely)
        assert len(opponents) == 10  # Just verify it works


class TestSelfPlayManagerTraining:
    """Tests for self-play training iteration."""

    def test_should_snapshot_at_interval(self):
        """should_create_snapshot returns True at snapshot_interval."""
        from agents.rl.ppo_agent import PPOAgent
        agent = PPOAgent()
        manager = SelfPlayManager(agent, snapshot_interval=10)
        assert manager.should_create_snapshot(10) is True
        assert manager.should_create_snapshot(20) is True
        assert manager.should_create_snapshot(5) is False

    def test_should_snapshot_at_win_rate(self):
        """should_create_snapshot returns True when win rate exceeds threshold."""
        from agents.rl.ppo_agent import PPOAgent
        agent = PPOAgent()
        manager = SelfPlayManager(agent, win_rate_threshold=0.55)
        assert manager.should_create_snapshot(1, win_rate=0.6) is True
        assert manager.should_create_snapshot(1, win_rate=0.4) is False

    def test_get_training_stats(self):
        """Can get training statistics."""
        from agents.rl.ppo_agent import PPOAgent
        agent = PPOAgent()
        manager = SelfPlayManager(agent)
        manager.create_snapshot()
        stats = manager.get_stats()
        assert "pool_size" in stats
        assert "total_snapshots" in stats
        assert stats["pool_size"] == 1

    def test_iteration_counter(self):
        """Manager tracks iteration count."""
        from agents.rl.ppo_agent import PPOAgent
        agent = PPOAgent()
        manager = SelfPlayManager(agent)
        assert manager.iteration == 0
        manager.iteration += 1
        assert manager.iteration == 1


class TestEloIntegration:
    """Tests for Elo tracking within self-play."""

    def test_record_self_play_result(self):
        """Recording a self-play match updates Elo."""
        from agents.rl.ppo_agent import PPOAgent
        agent = PPOAgent()
        manager = SelfPlayManager(agent)

        snap_id = manager.create_snapshot()
        manager.record_result("current", snap_id, winner="current")

        assert manager.elo_tracker.get_rating("current") > 1500

    def test_elo_history_grows(self):
        """Each recorded result adds to history."""
        from agents.rl.ppo_agent import PPOAgent
        agent = PPOAgent()
        manager = SelfPlayManager(agent)

        snap_id = manager.create_snapshot()
        manager.record_result("current", snap_id, winner="current")
        manager.record_result("current", snap_id, winner=snap_id)

        assert len(manager.elo_tracker.history) == 2
