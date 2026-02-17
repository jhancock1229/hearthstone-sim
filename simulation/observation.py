"""Agent-facing observations with information hiding.

Observation provides a view of game state from a specific player's perspective,
hiding opponent's private information (hand contents, deck order).

Used for:
- Agent decision-making (RL policies, MCTS, search algorithms)
- Self-play training (symmetric observations)
- Enforcing Hearthstone's information asymmetry rules

Design principles:
- Perspective-based (self vs opponent)
- Information hiding (hand contents, deck order)
- Public information visible (boards, health, mana)
- Supports perspective switching for self-play
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class Observation:
    """Agent-facing observation of game state from one player's perspective.

    Attributes:
        # Self (agent's own state - full information)
        self_health: int
        self_mana: int
        self_max_mana: int
        self_fatigue_counter: int
        self_hand: List[Any]  # Full hand contents visible
        self_deck_size: int  # Deck size only (not order)
        self_board: List[Any]  # Full board state

        # Opponent (hidden information)
        opponent_health: int
        opponent_mana: int
        opponent_max_mana: int
        opponent_fatigue_counter: int
        opponent_hand_size: int  # Count only, not contents
        opponent_deck_size: int  # Count only, not order
        opponent_board: List[Any]  # Full board state (public)

        # Game state
        turn_number: int
        is_my_turn: bool
        is_game_over: bool
        winner: Optional[int]  # 0, 1, or None
        player_index: int  # Which player this observation is for (0 or 1)
    """
    # Self state
    self_health: int
    self_mana: int
    self_max_mana: int
    self_fatigue_counter: int
    self_hand: List[Any] = field(default_factory=list)
    self_deck_size: int = 0
    self_board: List[Any] = field(default_factory=list)

    # Opponent state
    opponent_health: int = 30
    opponent_mana: int = 0
    opponent_max_mana: int = 0
    opponent_fatigue_counter: int = 0
    opponent_hand_size: int = 0
    opponent_deck_size: int = 0
    opponent_board: List[Any] = field(default_factory=list)

    # Weapon / armor / class state
    self_weapon_attack: int = 0
    self_weapon_durability: int = 0
    self_armor: int = 0
    self_hero_class: str = "NEUTRAL"
    opponent_armor: int = 0

    # Game state
    turn_number: int = 0
    is_my_turn: bool = False
    is_game_over: bool = False
    winner: Optional[int] = None
    player_index: int = 0
    legal_actions: List[Any] = field(default_factory=list)

    @classmethod
    def from_game(cls, game: Any, player_index: int) -> "Observation":
        """Create an Observation from a Game instance for a specific player.

        Args:
            game: The Game instance
            player_index: Which player's perspective (0 or 1)

        Returns:
            Observation from that player's perspective

        Raises:
            ValueError: If player_index is not 0 or 1
        """
        raise NotImplementedError("Observation.from_game not yet implemented")

    @classmethod
    def from_game_state(cls, state: Any, player_index: int) -> "Observation":
        """Create an Observation from a GameState for a specific player.

        Args:
            state: The GameState snapshot
            player_index: Which player's perspective (0 or 1)

        Returns:
            Observation from that player's perspective

        Raises:
            ValueError: If player_index is not 0 or 1
        """
        raise NotImplementedError("Observation.from_game_state not yet implemented")

    def switch_perspective(self) -> "Observation":
        """Switch to opponent's perspective (swap self/opponent views).

        Returns:
            New Observation from the opposite perspective
        """
        raise NotImplementedError("switch_perspective not yet implemented")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize Observation to dictionary.

        Returns:
            Dictionary representation
        """
        raise NotImplementedError("Observation.to_dict not yet implemented")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Observation":
        """Deserialize Observation from dictionary.

        Args:
            data: Dictionary representation

        Returns:
            Reconstructed Observation
        """
        raise NotImplementedError("Observation.from_dict not yet implemented")

    def to_feature_vector(self) -> List[float]:
        """Convert observation to fixed-size feature vector for ML models.

        Returns:
            List of float features with consistent size
        """
        raise NotImplementedError("to_feature_vector not yet implemented")
