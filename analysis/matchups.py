"""Deck vs deck win rate matrix.

Builds and queries an NxN matchup matrix for tracking deck/archetype
win rates against each other.

Usage:
    from analysis.matchups import MatchupMatrix

    matrix = MatchupMatrix()
    matrix.record("Aggro", "Control", wins=7, losses=3)
    print(matrix.get_win_rate("Aggro", "Control"))  # 0.7
    print(matrix.to_dict())
"""

from collections import defaultdict
from typing import Any, Dict, List, Optional


class MatchupMatrix:
    """NxN win rate matrix for deck/archetype matchups.

    Stores wins and losses for each ordered pair (a, b).
    A's wins vs B are automatically B's losses vs A.
    """

    def __init__(self, labels: Optional[List[str]] = None):
        """Initialize matchup matrix.

        Args:
            labels: Optional initial list of deck/archetype labels
        """
        self._labels: List[str] = list(labels) if labels else []
        self._wins: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._losses: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    @property
    def labels(self) -> List[str]:
        """Get all labels in the matrix."""
        return list(self._labels)

    @property
    def size(self) -> int:
        """Number of distinct labels."""
        return len(self._labels)

    def _ensure_label(self, label: str):
        """Add label if not already present."""
        if label not in self._labels:
            self._labels.append(label)

    def record(self, deck_a: str, deck_b: str, wins: int, losses: int):
        """Record matchup results.

        Args:
            deck_a: First deck label
            deck_b: Second deck label
            wins: Number of wins for deck_a
            losses: Number of losses for deck_a (= wins for deck_b)
        """
        self._ensure_label(deck_a)
        self._ensure_label(deck_b)

        self._wins[deck_a][deck_b] += wins
        self._losses[deck_a][deck_b] += losses
        # Symmetric: A's wins are B's losses
        self._wins[deck_b][deck_a] += losses
        self._losses[deck_b][deck_a] += wins

    def get_wins(self, deck_a: str, deck_b: str) -> int:
        """Get number of wins for deck_a vs deck_b."""
        return self._wins.get(deck_a, {}).get(deck_b, 0)

    def get_losses(self, deck_a: str, deck_b: str) -> int:
        """Get number of losses for deck_a vs deck_b."""
        return self._losses.get(deck_a, {}).get(deck_b, 0)

    def get_total_games(self, deck_a: str, deck_b: str) -> int:
        """Get total games between deck_a and deck_b."""
        return self.get_wins(deck_a, deck_b) + self.get_losses(deck_a, deck_b)

    def get_win_rate(self, deck_a: str, deck_b: str) -> Optional[float]:
        """Get win rate for deck_a vs deck_b.

        Returns:
            Win rate between 0.0 and 1.0, or None if no games
        """
        total = self.get_total_games(deck_a, deck_b)
        if total == 0:
            return None
        return self.get_wins(deck_a, deck_b) / total

    def get_overall_win_rate(self, deck: str) -> float:
        """Get overall win rate for a deck across all opponents.

        Returns:
            Win rate between 0.0 and 1.0, or 0.0 if no games
        """
        total_wins = 0
        total_games = 0
        for opponent in self._labels:
            if opponent == deck:
                continue
            w = self.get_wins(deck, opponent)
            g = self.get_total_games(deck, opponent)
            total_wins += w
            total_games += g
        if total_games == 0:
            return 0.0
        return total_wins / total_games

    def get_best_matchup(self, deck: str) -> Optional[str]:
        """Find the opponent with highest win rate for this deck.

        Returns:
            Opponent label, or None if no data
        """
        best_wr = -1.0
        best_opp = None
        for opponent in self._labels:
            if opponent == deck:
                continue
            wr = self.get_win_rate(deck, opponent)
            if wr is not None and wr > best_wr:
                best_wr = wr
                best_opp = opponent
        return best_opp

    def get_worst_matchup(self, deck: str) -> Optional[str]:
        """Find the opponent with lowest win rate for this deck.

        Returns:
            Opponent label, or None if no data
        """
        worst_wr = 2.0
        worst_opp = None
        for opponent in self._labels:
            if opponent == deck:
                continue
            wr = self.get_win_rate(deck, opponent)
            if wr is not None and wr < worst_wr:
                worst_wr = wr
                worst_opp = opponent
        return worst_opp

    def to_dict(self) -> Dict[str, Dict[str, float]]:
        """Export matrix as nested dict of win rates.

        Returns:
            Dict[row_label][col_label] = win_rate
        """
        if not self._labels:
            return {}
        result = {}
        for a in self._labels:
            result[a] = {}
            for b in self._labels:
                if a == b:
                    result[a][b] = 0.5
                else:
                    wr = self.get_win_rate(a, b)
                    result[a][b] = wr if wr is not None else 0.5
        return result
