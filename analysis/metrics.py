"""Training and evaluation metrics.

Tracks per-episode and aggregate statistics for monitoring
training progress and comparing agents.

Usage:
    tracker = MetricsTracker()
    tracker.record_episode(reward=1.0, length=45, losses={...})
    summary = tracker.summary(last_n=100)
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class EpisodeMetrics:
    """Metrics for a single episode."""
    reward: float
    length: int
    win: bool
    policy_loss: float = 0.0
    value_loss: float = 0.0
    entropy: float = 0.0


def compute_win_rate(rewards: List[float]) -> float:
    """Compute win rate from a list of episode rewards.

    Args:
        rewards: List of terminal rewards (+1 win, -1 loss)

    Returns:
        Win rate as float between 0.0 and 1.0
    """
    if not rewards:
        return 0.0
    wins = sum(1 for r in rewards if r > 0)
    return wins / len(rewards)


class MetricsTracker:
    """Collects and aggregates training metrics over time.

    Tracks episode rewards, lengths, losses, and win rates.
    Provides rolling-window summaries for monitoring progress.
    """

    def __init__(self):
        self.episodes: List[EpisodeMetrics] = []
        self.eval_win_rates: List[float] = []

    def record_episode(
        self,
        reward: float,
        length: int,
        losses: Optional[Dict[str, float]] = None,
    ):
        """Record metrics for one episode.

        Args:
            reward: Total episode reward
            length: Number of steps in episode
            losses: Optional dict with policy_loss, value_loss, entropy
        """
        losses = losses or {}
        self.episodes.append(EpisodeMetrics(
            reward=reward,
            length=length,
            win=reward > 0,
            policy_loss=losses.get('policy_loss', 0.0),
            value_loss=losses.get('value_loss', 0.0),
            entropy=losses.get('entropy', 0.0),
        ))

    def record_eval(self, win_rate: float):
        """Record an evaluation win rate."""
        self.eval_win_rates.append(win_rate)

    def summary(self, last_n: int = 100) -> Dict[str, Any]:
        """Compute summary statistics over recent episodes.

        Args:
            last_n: Number of recent episodes to summarize

        Returns:
            Dict with mean_reward, win_rate, mean_length, mean_policy_loss,
            mean_value_loss, mean_entropy, total_episodes
        """
        if not self.episodes:
            return {
                'mean_reward': 0.0,
                'win_rate': 0.0,
                'mean_length': 0.0,
                'mean_policy_loss': 0.0,
                'mean_value_loss': 0.0,
                'mean_entropy': 0.0,
                'total_episodes': 0,
            }

        recent = self.episodes[-last_n:]

        return {
            'mean_reward': np.mean([e.reward for e in recent]),
            'win_rate': np.mean([1.0 if e.win else 0.0 for e in recent]),
            'mean_length': np.mean([e.length for e in recent]),
            'mean_policy_loss': np.mean([e.policy_loss for e in recent]),
            'mean_value_loss': np.mean([e.value_loss for e in recent]),
            'mean_entropy': np.mean([e.entropy for e in recent]),
            'total_episodes': len(self.episodes),
        }

    def format_summary(self, last_n: int = 100) -> str:
        """Format summary as human-readable string."""
        s = self.summary(last_n)
        return (
            f"Episodes: {s['total_episodes']} | "
            f"Win Rate: {s['win_rate']:.1%} | "
            f"Reward: {s['mean_reward']:.3f} | "
            f"Length: {s['mean_length']:.1f} | "
            f"P.Loss: {s['mean_policy_loss']:.4f} | "
            f"V.Loss: {s['mean_value_loss']:.4f} | "
            f"Entropy: {s['mean_entropy']:.4f}"
        )
