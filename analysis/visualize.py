"""Visualization data generation for training curves, action distributions,
Elo progression, and matchup heatmaps.

Generates chart-ready data structures. Actual rendering (matplotlib, etc.)
is optional and separate from this module.

Usage:
    from analysis.visualize import training_curve_data, matchup_heatmap_data
    from analysis.metrics import MetricsTracker

    tracker = MetricsTracker()
    # ... record episodes ...
    data = training_curve_data(tracker, window=50)
    # data["episodes"], data["rewards"], data["rolling_win_rates"]
"""

from typing import Any, Dict, List, Optional

from analysis.metrics import MetricsTracker
from analysis.matchups import MatchupMatrix
from agents.rl.self_play import EloTracker


def training_curve_data(
    tracker: MetricsTracker, window: int = 50
) -> Dict[str, Any]:
    """Extract training curve data from a MetricsTracker.

    Args:
        tracker: MetricsTracker with recorded episodes
        window: Rolling window size for win rate smoothing

    Returns:
        Dict with keys: episodes, rewards, lengths, policy_losses,
        value_losses, entropies, rolling_win_rates, eval_win_rates
    """
    episodes_list = tracker.episodes

    if not episodes_list:
        return {
            "episodes": [],
            "rewards": [],
            "lengths": [],
            "policy_losses": [],
            "value_losses": [],
            "entropies": [],
            "rolling_win_rates": [],
            "eval_win_rates": list(tracker.eval_win_rates),
            "win_rates": [],
        }

    episode_nums = list(range(1, len(episodes_list) + 1))
    rewards = [e.reward for e in episodes_list]
    lengths = [e.length for e in episodes_list]
    policy_losses = [e.policy_loss for e in episodes_list]
    value_losses = [e.value_loss for e in episodes_list]
    entropies = [e.entropy for e in episodes_list]

    # Rolling win rate
    rolling_win_rates = []
    for i in range(len(episodes_list)):
        start = max(0, i - window + 1)
        window_eps = episodes_list[start:i + 1]
        wins = sum(1 for e in window_eps if e.win)
        rolling_win_rates.append(wins / len(window_eps))

    # Cumulative win rates
    win_rates = []
    cumulative_wins = 0
    for i, e in enumerate(episodes_list):
        if e.win:
            cumulative_wins += 1
        win_rates.append(cumulative_wins / (i + 1))

    return {
        "episodes": episode_nums,
        "rewards": rewards,
        "lengths": lengths,
        "policy_losses": policy_losses,
        "value_losses": value_losses,
        "entropies": entropies,
        "rolling_win_rates": rolling_win_rates,
        "win_rates": win_rates,
        "eval_win_rates": list(tracker.eval_win_rates),
    }


def action_distribution_data(
    actions: List[str], proportions: bool = False
) -> Dict[str, Any]:
    """Summarize action type frequencies.

    Args:
        actions: List of action type strings (e.g., "PLAY_CARD", "ATTACK")
        proportions: If True, return proportions instead of counts

    Returns:
        Dict mapping action type to count (or proportion), plus "total"
    """
    counts: Dict[str, int] = {}
    for action in actions:
        counts[action] = counts.get(action, 0) + 1

    total = len(actions)

    if proportions and total > 0:
        result: Dict[str, Any] = {k: v / total for k, v in counts.items()}
    else:
        result = dict(counts)

    result["total"] = total
    return result


def elo_progression_data(tracker: EloTracker) -> Dict[str, Any]:
    """Extract Elo rating progression from an EloTracker.

    Args:
        tracker: EloTracker with recorded matches

    Returns:
        Dict with keys:
            matches: List of match history entries
            agents: Dict[agent_id] -> List of rating values over time
    """
    if not tracker.history:
        return {"matches": [], "agents": {}}

    # Build per-agent rating progression
    agents: Dict[str, List[float]] = {}

    for match in tracker.history:
        a = match["agent_a"]
        b = match["agent_b"]

        if a not in agents:
            agents[a] = []
        if b not in agents:
            agents[b] = []

        agents[a].append(match["rating_a"])
        agents[b].append(match["rating_b"])

    return {
        "matches": list(tracker.history),
        "agents": agents,
    }


def matchup_heatmap_data(matrix: MatchupMatrix) -> Dict[str, Any]:
    """Convert a MatchupMatrix to heatmap-ready format.

    Args:
        matrix: MatchupMatrix with recorded matchups

    Returns:
        Dict with keys:
            labels: List of deck/archetype names
            values: 2D list where values[i][j] is win rate of labels[i] vs labels[j]
                    Diagonal entries are 0.5 (mirror match).
                    Entries with no data are None.
    """
    labels = matrix.labels

    if not labels:
        return {"labels": [], "values": []}

    values = []
    for a in labels:
        row = []
        for b in labels:
            if a == b:
                row.append(0.5)
            else:
                wr = matrix.get_win_rate(a, b)
                row.append(wr)
        values.append(row)

    return {"labels": labels, "values": values}
