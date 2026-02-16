"""Self-play training loop with Elo rating tracking.

Agent plays against copies of itself, improving iteratively.
Maintains a pool of past snapshots as opponents to prevent
catastrophic forgetting and ensure diverse training.

Components:
- EloTracker: Tracks Elo ratings for agent snapshots
- SelfPlayManager: Orchestrates self-play with snapshot scheduling

Usage:
    from agents.rl.ppo_agent import PPOAgent
    from agents.rl.self_play import SelfPlayManager

    agent = PPOAgent()
    manager = SelfPlayManager(agent, snapshot_interval=50)

    opponent = manager.select_opponent()
    # ... run episode agent vs opponent ...
    manager.record_result("current", snap_id, winner="current")

    if manager.should_create_snapshot(episode):
        manager.create_snapshot()
"""

import copy
import math
import random
from typing import Any, Dict, List, Optional, Tuple


class EloTracker:
    """Maintains Elo ratings for agent snapshots.

    Uses standard Elo formula with configurable K-factor.
    Ratings are zero-sum: what one player gains, the other loses.
    """

    def __init__(self, default_rating: float = 1500, k_factor: float = 32):
        """Initialize Elo tracker.

        Args:
            default_rating: Starting rating for new agents
            k_factor: Controls magnitude of rating changes per match
        """
        self.default_rating = default_rating
        self.k_factor = k_factor
        self.ratings: Dict[str, float] = {}
        self.history: List[Dict[str, Any]] = []

    def get_rating(self, agent_id: str) -> float:
        """Get current rating for an agent.

        Args:
            agent_id: Agent identifier

        Returns:
            Current Elo rating (default_rating if unregistered)
        """
        return self.ratings.get(agent_id, self.default_rating)

    def set_rating(self, agent_id: str, rating: float):
        """Explicitly set an agent's rating.

        Args:
            agent_id: Agent identifier
            rating: New rating value
        """
        self.ratings[agent_id] = rating

    def record_match(
        self, agent_a: str, agent_b: str, winner: Optional[str] = None
    ):
        """Record a match result and update ratings.

        Args:
            agent_a: First agent identifier
            agent_b: Second agent identifier
            winner: ID of winner, or None for draw
        """
        # Auto-register unknown agents
        if agent_a not in self.ratings:
            self.ratings[agent_a] = self.default_rating
        if agent_b not in self.ratings:
            self.ratings[agent_b] = self.default_rating

        ra = self.ratings[agent_a]
        rb = self.ratings[agent_b]

        # Expected scores
        ea = 1.0 / (1.0 + math.pow(10, (rb - ra) / 400.0))
        eb = 1.0 - ea

        # Actual scores
        if winner == agent_a:
            sa, sb = 1.0, 0.0
        elif winner == agent_b:
            sa, sb = 0.0, 1.0
        else:
            sa, sb = 0.5, 0.5

        # Update ratings
        self.ratings[agent_a] = ra + self.k_factor * (sa - ea)
        self.ratings[agent_b] = rb + self.k_factor * (sb - eb)

        # Record history
        self.history.append({
            "agent_a": agent_a,
            "agent_b": agent_b,
            "winner": winner,
            "rating_a": self.ratings[agent_a],
            "rating_b": self.ratings[agent_b],
        })


class SelfPlayManager:
    """Manages self-play training with opponent pool and snapshots.

    Maintains a pool of past agent snapshots as opponents.
    Schedules when to create new snapshots based on episode count
    or win rate threshold.
    """

    def __init__(
        self,
        agent: Any,
        snapshot_interval: int = 100,
        max_opponent_pool_size: int = 10,
        win_rate_threshold: float = 0.55,
    ):
        """Initialize self-play manager.

        Args:
            agent: The PPO agent being trained
            snapshot_interval: Create snapshot every N episodes
            max_opponent_pool_size: Max snapshots in opponent pool
            win_rate_threshold: Win rate triggering new snapshot
        """
        self.agent = agent
        self.snapshot_interval = snapshot_interval
        self.max_opponent_pool_size = max_opponent_pool_size
        self.win_rate_threshold = win_rate_threshold

        self.opponent_pool: List[Any] = []
        self.elo_tracker = EloTracker()
        self.iteration = 0
        self._snapshot_counter = 0

    def create_snapshot(self) -> str:
        """Create a frozen copy of the current agent and add to pool.

        Returns:
            Unique snapshot identifier
        """
        snapshot = copy.deepcopy(self.agent)
        self._snapshot_counter += 1
        snapshot_id = f"snapshot_{self._snapshot_counter}"

        # Evict oldest if pool is full
        if len(self.opponent_pool) >= self.max_opponent_pool_size:
            self.opponent_pool.pop(0)

        self.opponent_pool.append(snapshot)
        self.elo_tracker.set_rating(snapshot_id, self.elo_tracker.default_rating)

        return snapshot_id

    def select_opponent(self) -> Any:
        """Select an opponent from the pool.

        If pool is empty, returns a deep copy of the current agent.

        Returns:
            Agent instance to use as opponent
        """
        if not self.opponent_pool:
            return copy.deepcopy(self.agent)
        return random.choice(self.opponent_pool)

    def should_create_snapshot(
        self, episode: int, win_rate: Optional[float] = None
    ) -> bool:
        """Check whether a new snapshot should be created.

        Args:
            episode: Current episode number
            win_rate: Current win rate against opponent pool

        Returns:
            True if a snapshot should be created
        """
        if episode > 0 and episode % self.snapshot_interval == 0:
            return True
        if win_rate is not None and win_rate >= self.win_rate_threshold:
            return True
        return False

    def record_result(self, agent_id: str, opponent_id: str, winner: Optional[str] = None):
        """Record a self-play match result for Elo tracking.

        Args:
            agent_id: Current agent identifier
            opponent_id: Opponent snapshot identifier
            winner: ID of winner, or None for draw
        """
        self.elo_tracker.record_match(agent_id, opponent_id, winner=winner)

    def get_stats(self) -> Dict[str, Any]:
        """Get current self-play statistics.

        Returns:
            Dict with pool_size, total_snapshots, iteration, ratings
        """
        return {
            "pool_size": len(self.opponent_pool),
            "total_snapshots": self._snapshot_counter,
            "iteration": self.iteration,
            "ratings": dict(self.elo_tracker.ratings),
        }
