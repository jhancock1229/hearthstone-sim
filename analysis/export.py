"""Export results to CSV, JSON, or other formats for external analysis.

Provides export functions for training metrics, matchup matrices,
and meta-game summaries.

Usage:
    from analysis.export import export_training_csv, export_meta_json

    export_training_csv(tracker, "training_results.csv")
    export_meta_json(meta_tracker, "meta_summary.json")
"""

import csv
import json
from typing import Any, Dict

from analysis.metrics import MetricsTracker
from analysis.matchups import MatchupMatrix
from deckbuilding.meta import MetaTracker


def export_training_csv(tracker: MetricsTracker, path: str):
    """Export training metrics to CSV.

    Columns: episode, reward, length, win, policy_loss, value_loss, entropy

    Args:
        tracker: MetricsTracker with recorded episodes
        path: Output CSV file path
    """
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "episode", "reward", "length", "win",
            "policy_loss", "value_loss", "entropy",
        ])
        for i, ep in enumerate(tracker.episodes):
            writer.writerow([
                i + 1,
                ep.reward,
                ep.length,
                1 if ep.win else 0,
                ep.policy_loss,
                ep.value_loss,
                ep.entropy,
            ])


def export_matchups_csv(matrix: MatchupMatrix, path: str):
    """Export matchup matrix to CSV.

    First row/column are labels. Cell values are win rates.

    Args:
        matrix: MatchupMatrix with recorded matchups
        path: Output CSV file path
    """
    labels = matrix.labels
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        # Header row
        writer.writerow([""] + labels)
        for a in labels:
            row = [a]
            for b in labels:
                if a == b:
                    row.append("")
                else:
                    wr = matrix.get_win_rate(a, b)
                    row.append(f"{wr:.3f}" if wr is not None else "")
            writer.writerow(row)


def export_meta_json(tracker: MetaTracker, path: str):
    """Export meta-game summary to JSON.

    Args:
        tracker: MetaTracker with recorded matches
        path: Output JSON file path
    """
    data = {
        "total_matches": tracker.total_matches,
        "archetypes": tracker.get_meta_summary(),
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def export_training_json(tracker: MetricsTracker, path: str):
    """Export training metrics to JSON.

    Args:
        tracker: MetricsTracker with recorded episodes
        path: Output JSON file path
    """
    episodes = []
    for i, ep in enumerate(tracker.episodes):
        episodes.append({
            "episode": i + 1,
            "reward": ep.reward,
            "length": ep.length,
            "win": ep.win,
            "policy_loss": ep.policy_loss,
            "value_loss": ep.value_loss,
            "entropy": ep.entropy,
        })

    summary = tracker.summary()
    # Convert numpy values to plain floats for JSON serialization
    clean_summary: Dict[str, Any] = {}
    for k, v in summary.items():
        try:
            clean_summary[k] = float(v)
        except (TypeError, ValueError):
            clean_summary[k] = v

    data = {
        "episodes": episodes,
        "summary": clean_summary,
        "eval_win_rates": list(tracker.eval_win_rates),
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
