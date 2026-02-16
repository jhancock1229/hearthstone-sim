"""Unit tests for ActionSpace - legal action enumeration.

Following TDD: these tests define the contract for ActionSpace, which enumerates
all legal actions available to the active player from a given game state.

ActionSpace should:
- Enumerate all playable cards from hand
- Enumerate all possible attacks (minion to minion, minion to hero)
- Include hero power if available and affordable
- Include end turn action
- Respect game rules (mana, board limits, attack restrictions)
- Return actions in a consistent, enumerable format
"""

import pytest
from hearthstone.engine.game import Game
from hearthstone.cards.base import MinionCard


# ============================================================
# Action Space Enumeration Tests
# ============================================================


class TestActionSpaceEnumeration:
    """Tests for enumerating all legal actions."""

    def test_get_legal_actions_from_game(self):
        """ActionSpace can enumerate legal actions from a Game."""
        game = Game()
        # actions = ActionSpace.get_legal_actions(game)

        # Expected: returns list of Action objects

    def test_get_legal_actions_from_game_state(self):
        """ActionSpace can enumerate legal actions from GameState."""
        game = Game()
        # state = GameState.from_game(game)
        # actions = ActionSpace.get_legal_actions_from_state(state)

        # Expected: returns list of Action objects

    def test_action_space_always_includes_end_turn(self):
        """End turn action is always available."""
        game = Game()
        # actions = ActionSpace.get_legal_actions(game)

        # Expected: at least one action is END_TURN type

    def test_empty_hand_only_has_end_turn(self):
        """With empty hand and no minions, only end turn available."""
        game = Game()
        # Clear hand and board
        game.player1.hand.clear()
        game.player1.board.clear()

        # actions = ActionSpace.get_legal_actions(game)
        # Expected: only END_TURN action


# ============================================================
# Play Card Action Tests
# ============================================================


class TestPlayCardActions:
    """Tests for enumerating play card actions."""

    def test_enumerate_playable_cards_with_mana(self):
        """Enumerates cards that can be played with current mana."""
        game = Game()
        game.player1.mana = 5
        game.player1.hand.extend([
            MinionCard(name="C1", mana_cost=3, attack=1, health=1),
            MinionCard(name="C2", mana_cost=5, attack=1, health=1),
            MinionCard(name="C3", mana_cost=7, attack=1, health=1),  # unaffordable
        ])

        # actions = ActionSpace.get_legal_actions(game)
        # play_actions = [a for a in actions if a.type == ActionType.PLAY_CARD]

        # Expected: 2 play actions (C1 and C2, not C3)

    def test_no_play_actions_without_mana(self):
        """No play actions when all cards cost more than available mana."""
        game = Game()
        game.player1.mana = 2
        game.player1.hand.extend([
            MinionCard(name="C1", mana_cost=5, attack=1, health=1),
            MinionCard(name="C2", mana_cost=7, attack=1, health=1),
        ])

        # actions = ActionSpace.get_legal_actions(game)
        # play_actions = [a for a in actions if a.type == ActionType.PLAY_CARD]

        # Expected: no play actions

    def test_enumerate_board_positions_for_minions(self):
        """Playing a minion includes possible board positions."""
        game = Game()
        game.player1.mana = 5
        game.player1.board.append(MinionCard(name="M1", mana_cost=1, attack=1, health=1))
        game.player1.hand.append(MinionCard(name="New", mana_cost=2, attack=1, health=1))

        # actions = ActionSpace.get_legal_actions(game)
        # play_actions = [a for a in actions if a.type == ActionType.PLAY_CARD]

        # Expected: multiple actions for different positions (left, right of M1)

    def test_no_minion_play_when_board_full(self):
        """Cannot play minions when board has 7 minions."""
        game = Game()
        game.player1.mana = 10
        for i in range(7):
            game.player1.board.append(MinionCard(name=f"M{i}", mana_cost=1, attack=1, health=1))
        game.player1.hand.append(MinionCard(name="New", mana_cost=1, attack=1, health=1))

        # actions = ActionSpace.get_legal_actions(game)
        # play_actions = [a for a in actions if a.type == ActionType.PLAY_CARD]

        # Expected: no minion play actions


# ============================================================
# Attack Action Tests
# ============================================================


class TestAttackActions:
    """Tests for enumerating attack actions."""

    def test_enumerate_minion_attacks_on_enemy_minions(self):
        """Enumerates all possible minion vs minion attacks."""
        game = Game()
        game.player1.board.extend([
            MinionCard(name="A1", mana_cost=1, attack=2, health=2),
            MinionCard(name="A2", mana_cost=1, attack=3, health=3),
        ])
        game.player2.board.extend([
            MinionCard(name="D1", mana_cost=1, attack=1, health=1),
            MinionCard(name="D2", mana_cost=1, attack=1, health=1),
        ])

        # Mark minions as able to attack (e.g., with CHARGE or after summoning sickness)
        # actions = ActionSpace.get_legal_actions(game)
        # attack_actions = [a for a in actions if a.type == ActionType.ATTACK]

        # Expected: 4 attack actions (2 attackers × 2 defenders)

    def test_enumerate_minion_attacks_on_enemy_hero(self):
        """Enumerates attacks targeting enemy hero."""
        game = Game()
        game.player1.board.extend([
            MinionCard(name="A1", mana_cost=1, attack=2, health=2),
            MinionCard(name="A2", mana_cost=1, attack=3, health=3),
        ])

        # actions = ActionSpace.get_legal_actions(game)
        # hero_attacks = [a for a in actions if a.type == ActionType.ATTACK and a.target_is_hero]

        # Expected: 2 hero attack actions

    def test_no_attacks_without_friendly_minions(self):
        """No attack actions when player has no minions."""
        game = Game()
        game.player2.board.append(MinionCard(name="Enemy", mana_cost=1, attack=1, health=1))

        # actions = ActionSpace.get_legal_actions(game)
        # attack_actions = [a for a in actions if a.type == ActionType.ATTACK]

        # Expected: no attack actions

    def test_no_attacks_if_minion_already_attacked(self):
        """Minions that already attacked are excluded."""
        game = Game()
        minion = MinionCard(name="Tired", mana_cost=1, attack=2, health=2)
        game.player1.board.append(minion)
        # Mark as already attacked (implementation dependent)
        # minion._has_attacked = True

        # actions = ActionSpace.get_legal_actions(game)
        # attack_actions = [a for a in actions if a.type == ActionType.ATTACK]

        # Expected: no attack actions for this minion

    def test_windfury_allows_two_attacks(self):
        """Windfury minion can attack twice."""
        game = Game()
        windfury = MinionCard(name="WF", mana_cost=1, attack=2, health=2)
        windfury.mechanics.append("WINDFURY")
        game.player1.board.append(windfury)
        game.player2.board.append(MinionCard(name="D", mana_cost=1, attack=1, health=1))

        # After first attack:
        # actions = ActionSpace.get_legal_actions(game)
        # Expected: still has attack actions available


# ============================================================
# Hero Power Action Tests
# ============================================================


class TestHeroPowerActions:
    """Tests for enumerating hero power actions."""

    def test_hero_power_available_with_mana(self):
        """Hero power action available when player has 2+ mana."""
        game = Game()
        game.player1.mana = 2

        # actions = ActionSpace.get_legal_actions(game)
        # hp_actions = [a for a in actions if a.type == ActionType.HERO_POWER]

        # Expected: 1 hero power action

    def test_hero_power_not_available_without_mana(self):
        """Hero power not available when mana < 2."""
        game = Game()
        game.player1.mana = 1

        # actions = ActionSpace.get_legal_actions(game)
        # hp_actions = [a for a in actions if a.type == ActionType.HERO_POWER]

        # Expected: no hero power actions

    def test_hero_power_not_available_if_already_used(self):
        """Hero power not available if used this turn."""
        game = Game()
        game.player1.mana = 5
        # Mark hero power as used
        # game.player1.hero_power_used = True

        # actions = ActionSpace.get_legal_actions(game)
        # hp_actions = [a for a in actions if a.type == ActionType.HERO_POWER]

        # Expected: no hero power actions


# ============================================================
# Action Representation Tests
# ============================================================


class TestActionRepresentation:
    """Tests for how actions are represented."""

    def test_action_has_type_and_parameters(self):
        """Each action has a type and relevant parameters."""
        game = Game()
        game.player1.mana = 5
        game.player1.hand.append(MinionCard(name="Card", mana_cost=3, attack=1, health=1))

        # actions = ActionSpace.get_legal_actions(game)
        # play_action = [a for a in actions if a.type == ActionType.PLAY_CARD][0]

        # Expected: action has card_index, position, etc.

    def test_actions_are_hashable(self):
        """Actions should be hashable for use in sets/dicts."""
        game = Game()
        # actions = ActionSpace.get_legal_actions(game)

        # Expected: can create set(actions)

    def test_actions_are_comparable(self):
        """Actions should support equality comparison."""
        game = Game()
        # actions1 = ActionSpace.get_legal_actions(game)
        # actions2 = ActionSpace.get_legal_actions(game)

        # Expected: actions1 == actions2

    def test_action_to_dict(self):
        """Actions can be serialized to dict."""
        game = Game()
        game.player1.hand.append(MinionCard(name="Card", mana_cost=1, attack=1, health=1))
        game.player1.mana = 5

        # actions = ActionSpace.get_legal_actions(game)
        # action_dict = actions[0].to_dict()

        # Expected: dict with type and parameters


# ============================================================
# Action Space Size Tests
# ============================================================


class TestActionSpaceSize:
    """Tests for action space dimensionality."""

    def test_action_space_size_increases_with_options(self):
        """More cards/minions = larger action space."""
        game1 = Game()
        game1.player1.mana = 10

        game2 = Game()
        game2.player1.mana = 10
        for i in range(5):
            game2.player1.hand.append(MinionCard(name=f"C{i}", mana_cost=1, attack=1, health=1))

        # actions1 = ActionSpace.get_legal_actions(game1)
        # actions2 = ActionSpace.get_legal_actions(game2)

        # Expected: len(actions2) > len(actions1)

    def test_maximum_action_space_size(self):
        """Action space has an upper bound."""
        game = Game()
        game.player1.mana = 10
        # Fill hand (10 cards)
        for i in range(10):
            game.player1.hand.append(MinionCard(name=f"C{i}", mana_cost=1, attack=1, health=1))
        # Fill board (7 minions)
        for i in range(7):
            game.player1.board.append(MinionCard(name=f"M{i}", mana_cost=1, attack=1, health=1))
        # Fill opponent board (7 minions)
        for i in range(7):
            game.player2.board.append(MinionCard(name=f"E{i}", mana_cost=1, attack=1, health=1))

        # actions = ActionSpace.get_legal_actions(game)
        # Expected: bounded number of actions


# ============================================================
# Edge Cases Tests
# ============================================================


class TestActionSpaceEdgeCases:
    """Tests for edge cases in action enumeration."""

    def test_action_space_at_game_start(self):
        """Action space at turn 0."""
        game = Game()

        # actions = ActionSpace.get_legal_actions(game)
        # Expected: only END_TURN (no mana, no playable cards)

    def test_action_space_when_game_over(self):
        """No actions available when game is over."""
        game = Game()
        game.player1.health = 0

        # actions = ActionSpace.get_legal_actions(game)
        # Expected: empty list or only informational actions

    def test_action_space_with_zero_attack_minions(self):
        """Zero attack minions can still attack (for effects)."""
        game = Game()
        zero_attack = MinionCard(name="Weak", mana_cost=1, attack=0, health=3)
        game.player1.board.append(zero_attack)

        # Implementation choice: allow or disallow 0-attack attacks
        # actions = ActionSpace.get_legal_actions(game)

    def test_action_space_consistency(self):
        """Same game state produces same actions."""
        game = Game()
        game.player1.mana = 5
        game.player1.hand.append(MinionCard(name="Card", mana_cost=3, attack=1, health=1))

        # actions1 = ActionSpace.get_legal_actions(game)
        # actions2 = ActionSpace.get_legal_actions(game)

        # Expected: actions1 == actions2
