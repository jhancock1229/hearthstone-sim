"""Meta-game tracker: catalogs deck archetypes, tracks win rates,
identifies counter-strategies and rock-paper-scissors dynamics.

Components:
- Archetype: Represents a deck archetype with name, key cards, strategy
- MetaTracker: Tracks archetype popularity, win rates, and matchup trends

Usage:
    from deckbuilding.meta import Archetype, MetaTracker

    tracker = MetaTracker()
    tracker.register_archetype(Archetype(name="Aggro", description="Fast"))
    tracker.record_match("Aggro", "Control", winner="Aggro")
    print(tracker.get_win_rate("Aggro"))
    print(tracker.get_meta_summary())
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class Archetype:
    """Represents a deck archetype.

    Attributes:
        name: Archetype identifier
        description: Human-readable description
        key_cards: List of key card names that define this archetype
        strategy: Strategy tag (e.g., "aggro", "control", "combo")
    """
    name: str
    description: str
    key_cards: List[str] = field(default_factory=list)
    strategy: str = ""


class MetaTracker:
    """Tracks archetype popularity, win rates, and matchup dynamics.

    Records match results between archetypes and provides queries
    for win rates, popularity, counters, and meta summaries.
    """

    def __init__(self):
        self.archetypes: Dict[str, Archetype] = {}
        self._matches: List[Dict[str, Any]] = []
        # wins[a][b] = number of times a beat b
        self._wins: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        # games[a][b] = total games between a and b
        self._games: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    @property
    def total_matches(self) -> int:
        """Total number of recorded matches."""
        return len(self._matches)

    def register_archetype(self, archetype: Archetype):
        """Register or overwrite an archetype.

        Args:
            archetype: Archetype to register
        """
        self.archetypes[archetype.name] = archetype

    def _ensure_registered(self, name: str):
        """Auto-register an archetype if not already present."""
        if name not in self.archetypes:
            self.archetypes[name] = Archetype(name=name, description=name)

    def record_match(
        self, archetype_a: str, archetype_b: str, winner: Optional[str] = None
    ):
        """Record a match result.

        Args:
            archetype_a: First archetype name
            archetype_b: Second archetype name
            winner: Name of winner, or None for draw
        """
        self._ensure_registered(archetype_a)
        self._ensure_registered(archetype_b)

        self._matches.append({
            "a": archetype_a,
            "b": archetype_b,
            "winner": winner,
        })

        self._games[archetype_a][archetype_b] += 1
        self._games[archetype_b][archetype_a] += 1

        if winner == archetype_a:
            self._wins[archetype_a][archetype_b] += 1
        elif winner == archetype_b:
            self._wins[archetype_b][archetype_a] += 1
        # draw: no wins recorded for either

    def get_win_rate(self, archetype: str) -> float:
        """Get overall win rate for an archetype across all matchups.

        Args:
            archetype: Archetype name

        Returns:
            Win rate between 0.0 and 1.0, or 0.0 if no data
        """
        total_wins = 0
        total_games = 0
        for opponent in self._games.get(archetype, {}):
            total_wins += self._wins.get(archetype, {}).get(opponent, 0)
            total_games += self._games[archetype][opponent]
        if total_games == 0:
            return 0.0
        return total_wins / total_games

    def get_matchup_win_rate(self, archetype: str, opponent: str) -> float:
        """Get win rate for a specific matchup.

        Args:
            archetype: Archetype name
            opponent: Opponent archetype name

        Returns:
            Win rate between 0.0 and 1.0, or 0.0 if no data
        """
        games = self._games.get(archetype, {}).get(opponent, 0)
        if games == 0:
            return 0.0
        wins = self._wins.get(archetype, {}).get(opponent, 0)
        return wins / games

    def get_popularity(self, archetype: str) -> float:
        """Get archetype popularity (proportion of matches it appears in).

        Args:
            archetype: Archetype name

        Returns:
            Popularity between 0.0 and 1.0, or 0.0 if no matches
        """
        if self.total_matches == 0:
            return 0.0
        appearances = sum(
            1 for m in self._matches
            if m["a"] == archetype or m["b"] == archetype
        )
        return appearances / self.total_matches

    def get_counters(self, archetype: str, threshold: float = 0.55) -> List[str]:
        """Find archetypes that counter the given one.

        Args:
            archetype: Archetype to find counters for
            threshold: Minimum opponent win rate to count as counter

        Returns:
            List of archetype names that beat the given one above threshold
        """
        counters = []
        for opponent in self._games.get(archetype, {}):
            opp_wr = self.get_matchup_win_rate(opponent, archetype)
            if opp_wr >= threshold:
                counters.append(opponent)
        return counters

    def get_favorable_matchups(self, archetype: str, threshold: float = 0.55) -> List[str]:
        """Find archetypes that the given one beats.

        Args:
            archetype: Archetype to find favorable matchups for
            threshold: Minimum win rate to count as favorable

        Returns:
            List of archetype names the given one beats above threshold
        """
        favorable = []
        for opponent in self._games.get(archetype, {}):
            wr = self.get_matchup_win_rate(archetype, opponent)
            if wr >= threshold:
                favorable.append(opponent)
        return favorable

    def get_meta_summary(self) -> List[Dict[str, Any]]:
        """Get a summary of all archetypes sorted by win rate.

        Returns:
            List of dicts with name, win_rate, popularity, matches
        """
        summary = []
        for name in self.archetypes:
            total_games = sum(self._games.get(name, {}).values())
            summary.append({
                "name": name,
                "win_rate": self.get_win_rate(name),
                "popularity": self.get_popularity(name),
                "matches": total_games,
            })
        summary.sort(key=lambda x: x["win_rate"], reverse=True)
        return summary
