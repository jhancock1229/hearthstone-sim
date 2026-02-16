"""Game actions: PlayCard, Attack, HeroPower, EndTurn.

This module defines the core action interface for the game engine.
Actions validate game rules, modify game state, and emit events.

Each action:
- Validates preconditions (mana, board state, turn phase, etc.)
- Modifies game state atomically
- Emits appropriate events for listeners
- Raises IllegalActionError if action is invalid
"""

from typing import Optional, Any
from hearthstone.exceptions import IllegalActionError, GameOverError
from hearthstone.engine.combat import resolve_minion_attack, process_deaths


def play_card(
    player: Any,
    card_index: int,
    target: Optional[Any] = None,
    position: Optional[int] = None,
) -> None:
    """Play a card from the player's hand.

    Args:
        player: The player taking the action
        card_index: Index of card in player's hand
        target: Optional target for spells/battlecries
        position: Optional board position for minions (0-7)

    Raises:
        IllegalActionError: If action is invalid (not enough mana, board full, etc.)
        GameOverError: If game is already over
        IndexError: If card_index is out of bounds
    """
    # Validate card index
    if card_index < 0 or card_index >= len(player.hand):
        raise IndexError(f"Card index {card_index} out of bounds (hand size: {len(player.hand)})")

    card = player.hand[card_index]

    # Check mana
    if player.mana < card.mana_cost:
        raise IllegalActionError(f"Not enough mana to play {card.name} (need {card.mana_cost}, have {player.mana})")

    # Check if it's a minion card
    if hasattr(card, 'attack') and hasattr(card, 'health'):  # It's a minion
        # Check board limit
        if len(player.board) >= 7:
            raise IllegalActionError("Board is full (7 minions maximum)")

        # Remove from hand
        player.hand.pop(card_index)

        # Spend mana
        player.mana -= card.mana_cost

        # Add to board at specified position (or end)
        if position is not None and 0 <= position <= len(player.board):
            player.board.insert(position, card)
        else:
            player.board.append(card)

        # Mark as summoned this turn (summoning sickness)
        card.summoning_sick = True
        card.exhausted = False
        if not hasattr(card, '_can_attack'):
            card._can_attack = False
        if not hasattr(card, '_has_attacked'):
            card._has_attacked = False

        # CHARGE allows immediate attacks
        if "CHARGE" in card.mechanics:
            card.summoning_sick = False
            card._can_attack = True

        # RUSH allows attacking minions but not hero
        if "RUSH" in card.mechanics:
            card.summoning_sick = False
            card._can_attack = True
            card._rush_only = True

    else:  # It's a spell or other card
        # Remove from hand
        player.hand.pop(card_index)

        # Spend mana
        player.mana -= card.mana_cost

        # Spell execution would go here


def attack(
    attacker_player: Any,
    attacker_index: int,
    defender_player: Any,
    defender_index: Optional[int] = None,
) -> None:
    """Attack with a minion or hero.

    Args:
        attacker_player: The player whose minion/hero is attacking
        attacker_index: Index of attacking minion on board (or -1 for hero)
        defender_player: The defending player
        defender_index: Index of defending minion (or None to attack hero)

    Raises:
        IllegalActionError: If action is invalid (already attacked, no charge, etc.)
        GameOverError: If game is already over
        IndexError: If attacker/defender index is out of bounds
    """
    # Validate attacker index
    if attacker_index < 0 or attacker_index >= len(attacker_player.board):
        raise IndexError(f"Attacker index {attacker_index} out of bounds (board size: {len(attacker_player.board)})")

    attacker = attacker_player.board[attacker_index]

    # Check if minion can attack
    if getattr(attacker, 'summoning_sick', False):
        raise IllegalActionError(f"{attacker.name} cannot attack (summoning sickness)")

    if getattr(attacker, 'exhausted', False):
        # Check for WINDFURY (can attack twice)
        if "WINDFURY" in attacker.mechanics:
            if not hasattr(attacker, '_windfury_attacks'):
                attacker._windfury_attacks = 0
            if attacker._windfury_attacks >= 2:
                raise IllegalActionError(f"{attacker.name} has already attacked twice")
            # WINDFURY can attack again, don't raise error
        else:
            raise IllegalActionError(f"{attacker.name} has already attacked this turn")

    # Attacking hero
    if defender_index is None:
        # RUSH minions can't attack hero on first turn
        if hasattr(attacker, '_rush_only') and attacker._rush_only:
            raise IllegalActionError(f"{attacker.name} with RUSH cannot attack hero on summon turn")

        # Deal damage to hero
        defender_player.health -= attacker.attack

        # Mark as attacked
        if "WINDFURY" in attacker.mechanics:
            if not hasattr(attacker, '_windfury_attacks'):
                attacker._windfury_attacks = 0
            attacker._windfury_attacks += 1
            # Only exhaust after second attack
            if attacker._windfury_attacks >= 2:
                attacker.exhausted = True
        else:
            attacker.exhausted = True

    else:
        # Attacking minion
        # Validate defender index
        if defender_index < 0 or defender_index >= len(defender_player.board):
            raise IndexError(f"Defender index {defender_index} out of bounds (board size: {len(defender_player.board)})")

        defender = defender_player.board[defender_index]

        # Use combat system
        resolve_minion_attack(attacker_player, attacker_index, defender_player, defender_index)

        # Mark as attacked
        if "WINDFURY" in attacker.mechanics:
            if not hasattr(attacker, '_windfury_attacks'):
                attacker._windfury_attacks = 0
            attacker._windfury_attacks += 1
            # Only exhaust after second attack
            if attacker._windfury_attacks >= 2:
                attacker.exhausted = True
        else:
            attacker.exhausted = True

        # Process deaths
        process_deaths(attacker_player)
        process_deaths(defender_player)


def use_hero_power(
    player: Any,
    target: Optional[Any] = None,
) -> None:
    """Use the player's hero power.

    Args:
        player: The player using their hero power
        target: Optional target for targeted hero powers

    Raises:
        IllegalActionError: If action is invalid (not enough mana, already used, etc.)
        GameOverError: If game is already over
    """
    HERO_POWER_COST = 2

    # Check if already used
    if not hasattr(player, 'hero_power_used'):
        player.hero_power_used = False

    if player.hero_power_used:
        raise IllegalActionError("Hero power has already been used this turn")

    # Check mana
    if player.mana < HERO_POWER_COST:
        raise IllegalActionError(f"Not enough mana to use hero power (need {HERO_POWER_COST}, have {player.mana})")

    # Spend mana
    player.mana -= HERO_POWER_COST

    # Mark as used
    player.hero_power_used = True

    # Hero power effects would go here
    # For now, this is a placeholder


def end_turn(game: Any) -> None:
    """End the current player's turn and start the opponent's turn.

    Args:
        game: The game instance

    Raises:
        GameOverError: If game is already over
    """
    # Reset hero power usage
    game.active_player.hero_power_used = False

    # Reset minion attack status
    for minion in game.active_player.board:
        minion._has_attacked = False
        minion._can_attack = True  # Can attack next turn
        if hasattr(minion, '_windfury_attacks'):
            minion._windfury_attacks = 0
        if hasattr(minion, '_rush_only'):
            minion._rush_only = False  # RUSH restriction only for summon turn

    # Use game's built-in end_turn method
    game.end_turn()
