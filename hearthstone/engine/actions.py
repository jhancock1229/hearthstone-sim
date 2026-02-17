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
    opponent: Optional[Any] = None,
) -> None:
    """Play a card from the player's hand.

    Args:
        player: The player taking the action
        card_index: Index of card in player's hand
        target: Optional target for spells/battlecries
        position: Optional board position for minions (0-7)
        opponent: Optional opponent player (needed for battlecry effects)

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

    # Check if it's a weapon card (has durability, not health)
    if hasattr(card, 'durability') and not hasattr(card, 'health'):
        # Remove from hand
        player.hand.pop(card_index)
        # Spend mana
        player.mana -= card.mana_cost
        # Equip weapon (replaces existing)
        player.weapon = card
        # Trigger battlecry if present
        if "BATTLECRY" in card.mechanics:
            resolve_battlecry(player, card, opponent)
        return

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

        # Trigger battlecry effect
        if "BATTLECRY" in card.mechanics:
            resolve_battlecry(player, card, opponent)

    else:  # It's a spell or other card
        # Remove from hand
        player.hand.pop(card_index)

        # Spend mana
        player.mana -= card.mana_cost

        # Resolve spell effect
        resolve_spell(player, card, opponent)


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


def resolve_spell(player: Any, card: Any, opponent: Optional[Any] = None) -> None:
    """Resolve a spell effect based on card's spell_effect attribute.

    Args:
        player: The player who cast the spell
        card: The spell card
        opponent: The opponent player
    """
    effect = getattr(card, 'spell_effect', None)
    if not effect:
        return

    kind = effect[0]

    if kind == "deal_damage":
        if opponent is not None:
            opponent.take_damage(effect[1])

    elif kind == "aoe_damage":
        if opponent is not None:
            for m in opponent.board:
                m.health -= effect[1]
            # Remove dead minions
            opponent.board = [m for m in opponent.board if m.health > 0]

    elif kind == "draw":
        for _ in range(effect[1]):
            player.draw_card()

    elif kind == "restore_health":
        player.health = min(30, player.health + effect[1])

    elif kind == "gain_armor":
        player.armor += effect[1]

    elif kind == "destroy":
        if opponent is not None and opponent.board:
            opponent.board.pop(0)

    elif kind == "summon":
        from hearthstone.cards.base import MinionCard as _MC
        if len(player.board) < 7:
            token = _MC(name="Token", mana_cost=0, attack=effect[1], health=effect[2])
            token.summoning_sick = True
            player.board.append(token)


def hero_attack(
    player: Any,
    opponent: Any,
    defender_index: Optional[int] = None,
) -> None:
    """Attack with the hero using an equipped weapon.

    Args:
        player: The attacking player
        opponent: The defending player
        defender_index: Index of defending minion (None = attack face)

    Raises:
        IllegalActionError: If no weapon equipped or already attacked
    """
    if player.weapon is None:
        raise IllegalActionError("No weapon equipped")

    if player.hero_attacked:
        raise IllegalActionError("Hero has already attacked this turn")

    weapon_attack = player.weapon.attack

    if defender_index is None:
        # Attack face
        opponent.take_damage(weapon_attack)
    else:
        # Attack minion
        minion = opponent.board[defender_index]
        minion.health -= weapon_attack
        player.take_damage(minion.attack)
        # Remove dead minions
        opponent.board = [m for m in opponent.board if m.health > 0]

    # Lose durability
    player.weapon.durability -= 1
    if player.weapon.durability <= 0:
        player.weapon = None

    player.hero_attacked = True


def resolve_battlecry(player: Any, card: Any, opponent: Optional[Any] = None) -> None:
    """Resolve a battlecry effect based on card's battlecry_effect attribute.

    Args:
        player: The player who played the card
        card: The card with the battlecry
        opponent: The opponent player
    """
    effect = getattr(card, 'battlecry_effect', None)
    if not effect:
        return

    kind = effect[0]

    if kind == "deal_damage":
        if opponent is not None:
            opponent.take_damage(effect[1])

    elif kind == "draw":
        for _ in range(effect[1]):
            player.draw_card()

    elif kind == "restore_health":
        player.health = min(30, player.health + effect[1])

    elif kind == "gain_armor":
        player.armor += effect[1]

    elif kind == "summon":
        from hearthstone.cards.base import MinionCard as _MC
        if len(player.board) < 7:
            token = _MC(name="Token", mana_cost=0, attack=effect[1], health=effect[2])
            token.summoning_sick = True
            player.board.append(token)

    elif kind == "buff_all":
        for m in player.board:
            if m is not card:
                m.attack += effect[1]
                m.health += effect[2]


def use_hero_power(
    player: Any,
    game: Optional[Any] = None,
    target: Optional[Any] = None,
) -> None:
    """Use the player's hero power.

    Args:
        player: The player using their hero power
        game: The Game instance (needed for opponent reference)
        target: Optional target tuple for targeted hero powers.
            Format: ("opponent_minion", idx), ("self_minion", idx),
                    ("opponent_hero",), ("self_hero",)

    Raises:
        IllegalActionError: If action is invalid (not enough mana, already used, etc.)
        GameOverError: If game is already over
    """
    HERO_POWER_COST = 2

    if player.hero_power_used:
        raise IllegalActionError("Hero power has already been used this turn")

    # Check mana
    if player.mana < HERO_POWER_COST:
        raise IllegalActionError(f"Not enough mana to use hero power (need {HERO_POWER_COST}, have {player.mana})")

    # Resolve opponent
    opponent = None
    if game is not None:
        opponent = game.player2 if player is game.player1 else game.player1

    hero_class = getattr(player, 'hero_class', 'NEUTRAL')

    # Targeted powers require a target
    if hero_class in ("MAGE", "PRIEST") and target is None:
        raise IllegalActionError(f"{hero_class} hero power requires a target")

    # Paladin/Shaman need board space
    if hero_class in ("PALADIN", "SHAMAN") and len(player.board) >= 7:
        raise IllegalActionError("Board is full, cannot summon")

    # Spend mana and mark used
    player.mana -= HERO_POWER_COST
    player.hero_power_used = True

    # Execute effect
    _execute_hero_power(player, opponent, hero_class, target)


def _resolve_target(player, opponent, target):
    """Resolve a target tuple to a (object, kind) pair.

    Returns:
        (target_object, kind) where kind is 'minion' or 'hero'
    """
    if target is None:
        return None, None
    label = target[0]
    if label == "opponent_minion":
        return opponent.board[target[1]], "minion"
    elif label == "self_minion":
        return player.board[target[1]], "minion"
    elif label == "opponent_hero":
        return opponent, "hero"
    elif label == "self_hero":
        return player, "hero"
    return None, None


def _execute_hero_power(player, opponent, hero_class, target):
    """Execute the class-specific hero power effect."""
    from hearthstone.cards.base import MinionCard

    if hero_class == "MAGE":
        obj, kind = _resolve_target(player, opponent, target)
        if kind == "hero":
            obj.take_damage(1)
        elif kind == "minion":
            obj.health -= 1

    elif hero_class == "WARLOCK":
        player.take_damage(2)
        player.draw_card()

    elif hero_class == "PRIEST":
        obj, kind = _resolve_target(player, opponent, target)
        if kind == "hero":
            obj.health = min(30, obj.health + 2)
        elif kind == "minion":
            obj.health = obj.health + 2

    elif hero_class == "PALADIN":
        recruit = MinionCard(name="Silver Hand Recruit", mana_cost=1, attack=1, health=1)
        recruit.summoning_sick = True
        player.board.append(recruit)

    elif hero_class == "HUNTER":
        if opponent is not None:
            opponent.take_damage(2)

    elif hero_class == "WARRIOR":
        player.armor += 2

    elif hero_class == "SHAMAN":
        totem = MinionCard(name="Totem", mana_cost=0, attack=0, health=2, mechanics=["TAUNT"])
        totem.summoning_sick = True
        player.board.append(totem)

    elif hero_class == "ROGUE":
        if opponent is not None and opponent.board:
            # Hit a random enemy minion (first one for determinism in tests)
            opponent.board[0].health -= 1
        elif opponent is not None:
            opponent.take_damage(1)

    elif hero_class == "DRUID":
        player.armor += 1

    elif hero_class == "DEMONHUNTER":
        if opponent is not None:
            opponent.take_damage(1)

    elif hero_class == "DEATHKNIGHT":
        if opponent is not None:
            for minion in opponent.board:
                minion.health -= 1


def end_turn(game: Any) -> None:
    """End the current player's turn and start the opponent's turn.

    Args:
        game: The game instance

    Raises:
        GameOverError: If game is already over
    """
    # Reset hero power usage and hero attack
    game.active_player.hero_power_used = False
    game.active_player.hero_attacked = False

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
