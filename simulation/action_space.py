"""Legal action enumeration for game states.

ActionSpace enumerates all legal actions available to the active player
from a given game state. Used by agents to determine valid moves.

Actions include:
- Play card from hand (with board position for minions)
- Attack with minion (choose target: hero or enemy minion)
- Use hero power
- End turn (always available)

Design principles:
- Complete enumeration (all legal moves)
- Consistent ordering
- Efficient generation
- Rule-aware (mana, board limits, attack restrictions)
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional


class ActionType(Enum):
    """Types of actions a player can take."""
    PLAY_CARD = "play_card"
    ATTACK = "attack"
    HERO_POWER = "hero_power"
    HERO_ATTACK = "hero_attack"
    END_TURN = "end_turn"


@dataclass(frozen=True)
class Action:
    """Immutable representation of a game action.

    Attributes:
        type: The type of action
        card_index: For PLAY_CARD, index in hand
        position: For PLAY_CARD minion, board position (0-7)
        attacker_index: For ATTACK, index of attacking minion
        defender_index: For ATTACK, index of defending minion (None = hero)
        target: For targeted abilities
    """
    type: ActionType
    card_index: Optional[int] = None
    position: Optional[int] = None
    attacker_index: Optional[int] = None
    defender_index: Optional[int] = None
    target: Optional[Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize action to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            "type": self.type.value,
            "card_index": self.card_index,
            "position": self.position,
            "attacker_index": self.attacker_index,
            "defender_index": self.defender_index,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Action":
        """Deserialize action from dictionary.

        Args:
            data: Dictionary representation

        Returns:
            Reconstructed Action
        """
        return cls(
            type=ActionType(data["type"]),
            card_index=data.get("card_index"),
            position=data.get("position"),
            attacker_index=data.get("attacker_index"),
            defender_index=data.get("defender_index"),
        )


class ActionSpace:
    """Enumerates legal actions from a game state."""

    @staticmethod
    def get_legal_actions(game: Any) -> List[Action]:
        """Get all legal actions for the active player.

        Args:
            game: The Game instance

        Returns:
            List of legal Action objects
        """
        actions = []
        player = game.active_player
        opponent = game.player2 if player == game.player1 else game.player1

        # 1. Play card actions
        for card_index, card in enumerate(player.hand):
            # Check if we have enough mana
            if player.mana >= card.mana_cost:
                # Check if it's a weapon (has durability, not health)
                if hasattr(card, 'durability') and not hasattr(card, 'health'):
                    actions.append(Action(
                        type=ActionType.PLAY_CARD,
                        card_index=card_index
                    ))
                # Check if it's a minion
                elif hasattr(card, 'attack') and hasattr(card, 'health'):
                    # Check board limit
                    if len(player.board) < 7:
                        # Can play at any position
                        for position in range(len(player.board) + 1):
                            actions.append(Action(
                                type=ActionType.PLAY_CARD,
                                card_index=card_index,
                                position=position
                            ))
                else:
                    # Spell or other card type
                    actions.append(Action(
                        type=ActionType.PLAY_CARD,
                        card_index=card_index
                    ))

        # 2. Attack actions
        # First, check if opponent has any TAUNT minions
        taunt_indices = []
        for idx, minion in enumerate(opponent.board):
            if hasattr(minion, 'mechanics') and 'TAUNT' in minion.mechanics:
                taunt_indices.append(idx)

        for attacker_index, minion in enumerate(player.board):
            # Check if minion can attack
            if (hasattr(minion, 'attack') and minion.attack > 0 and
                not getattr(minion, 'exhausted', False) and
                not getattr(minion, 'summoning_sick', False)):

                if taunt_indices:
                    # TAUNT present: can only attack TAUNT minions
                    for defender_index in taunt_indices:
                        actions.append(Action(
                            type=ActionType.ATTACK,
                            attacker_index=attacker_index,
                            defender_index=defender_index
                        ))
                else:
                    # No TAUNT: can attack any enemy minion or face
                    for defender_index in range(len(opponent.board)):
                        actions.append(Action(
                            type=ActionType.ATTACK,
                            attacker_index=attacker_index,
                            defender_index=defender_index
                        ))

                    # Attack enemy hero (face)
                    actions.append(Action(
                        type=ActionType.ATTACK,
                        attacker_index=attacker_index,
                        defender_index=None
                    ))

        # 3. Hero power actions
        if player.mana >= 2 and not getattr(player, 'hero_power_used', False):
            hero_class = getattr(player, 'hero_class', 'NEUTRAL')
            # Targeted hero powers enumerate all valid targets
            if hero_class in ("MAGE", "PRIEST"):
                # Can target any minion (both sides) and both heroes
                for idx in range(len(player.board)):
                    actions.append(Action(
                        type=ActionType.HERO_POWER,
                        target=("self_minion", idx)
                    ))
                for idx in range(len(opponent.board)):
                    actions.append(Action(
                        type=ActionType.HERO_POWER,
                        target=("opponent_minion", idx)
                    ))
                actions.append(Action(
                    type=ActionType.HERO_POWER,
                    target=("self_hero",)
                ))
                actions.append(Action(
                    type=ActionType.HERO_POWER,
                    target=("opponent_hero",)
                ))
            else:
                # Untargeted hero powers produce a single action
                actions.append(Action(type=ActionType.HERO_POWER))

        # 4. Hero attack actions (weapon equipped and hasn't attacked)
        if getattr(player, 'weapon', None) is not None and not getattr(player, 'hero_attacked', False):
            if taunt_indices:
                # TAUNT present: can only attack TAUNT minions
                for defender_index in taunt_indices:
                    actions.append(Action(
                        type=ActionType.HERO_ATTACK,
                        defender_index=defender_index
                    ))
            else:
                # No TAUNT: can attack any enemy minion or face
                for defender_index in range(len(opponent.board)):
                    actions.append(Action(
                        type=ActionType.HERO_ATTACK,
                        defender_index=defender_index
                    ))
                # Attack enemy hero (face)
                actions.append(Action(
                    type=ActionType.HERO_ATTACK,
                    defender_index=None
                ))

        # 5. End turn (always available)
        actions.append(Action(type=ActionType.END_TURN))

        return actions

    @staticmethod
    def get_legal_actions_from_state(state: Any) -> List[Action]:
        """Get all legal actions from a GameState.

        Args:
            state: The GameState snapshot

        Returns:
            List of legal Action objects
        """
        raise NotImplementedError("get_legal_actions_from_state not yet implemented")

    @staticmethod
    def get_play_card_actions(game: Any) -> List[Action]:
        """Enumerate all legal play card actions.

        Args:
            game: The Game instance

        Returns:
            List of PLAY_CARD actions
        """
        raise NotImplementedError("get_play_card_actions not yet implemented")

    @staticmethod
    def get_attack_actions(game: Any) -> List[Action]:
        """Enumerate all legal attack actions.

        Args:
            game: The Game instance

        Returns:
            List of ATTACK actions
        """
        raise NotImplementedError("get_attack_actions not yet implemented")

    @staticmethod
    def get_hero_power_action(game: Any) -> Optional[Action]:
        """Get hero power action if available.

        Args:
            game: The Game instance

        Returns:
            HERO_POWER action or None if not available
        """
        raise NotImplementedError("get_hero_power_action not yet implemented")
