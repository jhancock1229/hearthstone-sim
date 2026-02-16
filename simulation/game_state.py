"""Immutable game state snapshots for simulation and replay.

GameState is an immutable, serializable snapshot of complete game state.
Used for:
- MCTS rollouts (efficient state copying)
- Game replay (deterministic state reconstruction)
- State hashing (transposition tables)
- Agent observation (with optional information hiding)

Design principles:
- Immutable (frozen dataclass)
- Hashable and comparable
- Serializable to/from dict
- Captures complete game state
- Efficient to copy
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class MinionState:
    """Immutable snapshot of a minion's state."""
    name: str
    mana_cost: int
    attack: int
    health: int
    mechanics: Tuple[str, ...] = field(default_factory=tuple)
    # Additional fields for minion state (has_attacked, etc.) can be added


@dataclass(frozen=True)
class PlayerState:
    """Immutable snapshot of a player's state."""
    health: int
    mana: int
    max_mana: int
    fatigue_counter: int
    hand_size: int
    deck_size: int
    board: Tuple[MinionState, ...] = field(default_factory=tuple)
    # For full state, could include actual hand/deck cards
    # For agent view, only sizes are exposed


@dataclass(frozen=True)
class GameState:
    """Immutable snapshot of complete game state.

    Attributes:
        player1: State of player 1
        player2: State of player 2
        active_player_index: Which player is active (0 or 1)
        turn_number: Current turn number (0-indexed)
    """
    player1: PlayerState
    player2: PlayerState
    active_player_index: int
    turn_number: int

    @classmethod
    def from_game(cls, game: Any) -> "GameState":
        """Create a GameState snapshot from a Game instance.

        Args:
            game: The Game instance to snapshot

        Returns:
            Immutable GameState snapshot
        """
        raise NotImplementedError("GameState.from_game not yet implemented")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize GameState to dictionary.

        Returns:
            Dictionary representation suitable for JSON serialization
        """
        raise NotImplementedError("GameState.to_dict not yet implemented")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GameState":
        """Deserialize GameState from dictionary.

        Args:
            data: Dictionary representation

        Returns:
            Reconstructed GameState
        """
        raise NotImplementedError("GameState.from_dict not yet implemented")

    def is_game_over(self) -> bool:
        """Check if the game is over (at least one player dead).

        Returns:
            True if game is over
        """
        raise NotImplementedError("is_game_over not yet implemented")

    def get_winner(self) -> Optional[int]:
        """Get the winning player index.

        Returns:
            0 or 1 if there's a winner, None if game not over
        """
        raise NotImplementedError("get_winner not yet implemented")

    def get_active_player_index(self) -> int:
        """Get the index of the active player.

        Returns:
            0 or 1
        """
        return self.active_player_index

    def clone(self) -> "GameState":
        """Create a deep copy of this GameState.

        Since GameState is immutable, this could just return self.
        Provided for clarity in simulation code.

        Returns:
            Copy of this state (or self, since immutable)
        """
        return self

    def apply_action(self, action: Any) -> "GameState":
        """Apply an action and return the resulting new state.

        Args:
            action: The action to apply

        Returns:
            New GameState after applying action
        """
        raise NotImplementedError("apply_action not yet implemented")
