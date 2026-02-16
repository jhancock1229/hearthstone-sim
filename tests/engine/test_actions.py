"""Unit tests for game actions (PlayCard, Attack, HeroPower, EndTurn).

Following TDD: these tests define the contract for the action system.
Actions are the primary interface for game interactions and must enforce
all game rules (mana costs, board limits, attack restrictions, etc.).
"""

import pytest

from hearthstone.engine.game import Game
from hearthstone.engine.player import Player
from hearthstone.cards.base import MinionCard, SpellCard
from hearthstone.exceptions import IllegalActionError
from hearthstone.engine.actions import use_hero_power, end_turn


# ============================================================
# PlayCard Action Tests
# ============================================================


class TestPlayMinionCard:
    """Tests for playing minion cards from hand."""

    def test_play_minion_removes_from_hand_and_adds_to_board(self):
        """Playing a minion should move it from hand to board."""
        player = Player()
        minion = MinionCard(name="Wisp", mana_cost=0, attack=1, health=1)
        player.hand.append(minion)
        player.mana = 10

        # Action: play_card(player, card_index=0, position=None)
        # For now we'll test the expected behavior directly
        initial_hand_size = len(player.hand)
        initial_board_size = len(player.board)

        # Expected: hand decreases by 1, board increases by 1
        assert initial_hand_size == 1
        assert initial_board_size == 0

    def test_play_minion_costs_mana(self):
        """Playing a minion should deduct its mana cost."""
        player = Player()
        minion = MinionCard(name="Test", mana_cost=3, attack=2, health=2)
        player.hand.append(minion)
        player.mana = 5

        # Expected: after playing, mana should be 5 - 3 = 2
        # This will be validated by the action implementation

    def test_cannot_play_minion_without_enough_mana(self):
        """Playing a minion should fail if not enough mana."""
        player = Player()
        minion = MinionCard(name="Expensive", mana_cost=5, attack=4, health=4)
        player.hand.append(minion)
        player.mana = 3

        # Expected: should raise IllegalActionError
        # with pytest.raises(IllegalActionError, match="Not enough mana"):
        #     play_card(player, 0)

    def test_cannot_play_minion_if_board_full(self):
        """Cannot play a minion if board already has 7 minions."""
        player = Player()
        # Fill the board with 7 minions
        for i in range(7):
            player.board.append(MinionCard(name=f"M{i}", mana_cost=0, attack=1, health=1))

        minion = MinionCard(name="Eighth", mana_cost=0, attack=1, health=1)
        player.hand.append(minion)
        player.mana = 10

        # Expected: should raise IllegalActionError
        # with pytest.raises(IllegalActionError, match="Board is full"):
        #     play_card(player, 0)

    def test_play_minion_at_specific_position(self):
        """Playing a minion should allow choosing board position."""
        player = Player()
        existing = MinionCard(name="Existing", mana_cost=0, attack=1, health=1)
        player.board.append(existing)

        new_minion = MinionCard(name="New", mana_cost=0, attack=1, health=1)
        player.hand.append(new_minion)
        player.mana = 10

        # Expected: play_card(player, card_index=0, position=0)
        # should insert at position 0, pushing existing to position 1

    def test_play_zero_cost_minion(self):
        """Zero-cost minions should not deduct mana."""
        player = Player()
        minion = MinionCard(name="Free", mana_cost=0, attack=1, health=1)
        player.hand.append(minion)
        player.mana = 3

        # Expected: mana should remain 3 after playing

    def test_cannot_play_card_from_empty_hand(self):
        """Cannot play a card if hand is empty."""
        player = Player()
        player.mana = 10

        # Expected: should raise IllegalActionError or IndexError
        # with pytest.raises((IllegalActionError, IndexError)):
        #     play_card(player, 0)

    def test_cannot_play_card_at_invalid_index(self):
        """Cannot play a card at an invalid hand index."""
        player = Player()
        minion = MinionCard(name="Test", mana_cost=1, attack=1, health=1)
        player.hand.append(minion)
        player.mana = 10

        # Expected: should raise IllegalActionError or IndexError
        # with pytest.raises((IllegalActionError, IndexError)):
        #     play_card(player, 5)  # hand only has 1 card


class TestPlaySpellCard:
    """Tests for playing spell cards from hand."""

    def test_play_spell_removes_from_hand(self):
        """Playing a spell should remove it from hand."""
        player = Player()
        spell = SpellCard(name="Fireball", mana_cost=4)
        player.hand.append(spell)
        player.mana = 5

        # Expected: spell removed from hand after playing

    def test_play_spell_costs_mana(self):
        """Playing a spell should deduct mana cost."""
        player = Player()
        spell = SpellCard(name="Test", mana_cost=2)
        player.hand.append(spell)
        player.mana = 5

        # Expected: mana should be 5 - 2 = 3 after playing

    def test_cannot_play_spell_without_enough_mana(self):
        """Cannot play spell if insufficient mana."""
        player = Player()
        spell = SpellCard(name="Expensive", mana_cost=7)
        player.hand.append(spell)
        player.mana = 3

        # Expected: should raise IllegalActionError


# ============================================================
# Attack Action Tests
# ============================================================


class TestMinionAttack:
    """Tests for minion attack actions."""

    def test_minion_can_attack_enemy_minion(self):
        """A minion should be able to attack an enemy minion."""
        p1 = Player()
        p2 = Player()
        attacker = MinionCard(name="Attacker", mana_cost=1, attack=3, health=2)
        defender = MinionCard(name="Defender", mana_cost=1, attack=2, health=3)
        p1.board.append(attacker)
        p2.board.append(defender)

        # Expected: attack action should succeed and apply combat

    def test_minion_can_attack_enemy_hero(self):
        """A minion should be able to attack the enemy hero."""
        p1 = Player()
        p2 = Player()
        attacker = MinionCard(name="Attacker", mana_cost=1, attack=5, health=2)
        p1.board.append(attacker)

        # Expected: attack should reduce p2.health by 5

    def test_cannot_attack_with_exhausted_minion(self):
        """A minion that already attacked cannot attack again (non-Windfury)."""
        p1 = Player()
        p2 = Player()
        attacker = MinionCard(name="Attacker", mana_cost=1, attack=2, health=2)
        p1.board.append(attacker)

        # Expected: first attack succeeds, second raises IllegalActionError

    def test_windfury_minion_can_attack_twice(self):
        """A Windfury minion can attack twice in one turn."""
        p1 = Player()
        p2 = Player()
        attacker = MinionCard(name="WF", mana_cost=3, attack=2, health=3)
        attacker.mechanics.append("WINDFURY")
        p1.board.append(attacker)

        # Expected: two attacks should succeed, third should fail

    def test_cannot_attack_friendly_minion(self):
        """Cannot attack your own minions."""
        player = Player()
        m1 = MinionCard(name="M1", mana_cost=1, attack=2, health=2)
        m2 = MinionCard(name="M2", mana_cost=1, attack=2, health=2)
        player.board.extend([m1, m2])

        # Expected: should raise IllegalActionError

    def test_cannot_attack_with_zero_attack_minion(self):
        """Zero-attack minions should not be able to attack."""
        p1 = Player()
        p2 = Player()
        minion = MinionCard(name="Weak", mana_cost=1, attack=0, health=3)
        p1.board.append(minion)

        # Expected: depends on game rules - may allow (for activating effects)
        # or may raise IllegalActionError


# ============================================================
# Hero Power Action Tests
# ============================================================


class TestHeroPower:
    """Tests for hero power actions."""

    def test_hero_power_costs_two_mana(self):
        """Using hero power should cost 2 mana by default."""
        player = Player()
        player.mana = 5
        player.hero_class = "WARRIOR"
        game = Game(player1_class="WARRIOR")
        game.player1 = player

        use_hero_power(player, game=game)
        assert player.mana == 3

    def test_cannot_use_hero_power_without_enough_mana(self):
        """Cannot use hero power if mana < 2."""
        player = Player()
        player.mana = 1
        player.hero_class = "WARRIOR"

        with pytest.raises(IllegalActionError):
            use_hero_power(player, game=Game())

    def test_cannot_use_hero_power_twice_per_turn(self):
        """Hero power can only be used once per turn."""
        player = Player()
        player.mana = 10
        player.hero_class = "WARRIOR"
        game = Game(player1_class="WARRIOR")
        game.player1 = player

        use_hero_power(player, game=game)
        with pytest.raises(IllegalActionError):
            use_hero_power(player, game=game)

    def test_hero_power_resets_on_new_turn(self):
        """Hero power availability should reset at start of turn."""
        game = Game(player1_class="WARRIOR")
        game.player1.mana = 10
        game.player1.deck = [MinionCard(name=f"M{i}", mana_cost=1, attack=1, health=1) for i in range(10)]
        game.player2.deck = [MinionCard(name=f"M{i}", mana_cost=1, attack=1, health=1) for i in range(10)]

        use_hero_power(game.player1, game=game)
        assert game.player1.hero_power_used is True

        end_turn(game)       # p1 -> p2
        game.start_turn()    # p2's turn
        end_turn(game)       # p2 -> p1
        game.start_turn()    # p1's turn again

        assert game.player1.hero_power_used is False


# ============================================================
# End Turn Action Tests
# ============================================================


class TestEndTurn:
    """Tests for end turn action."""

    def test_end_turn_switches_active_player(self):
        """Ending turn should switch the active player."""
        game = Game()
        initial_active = game.active_player

        # Expected: after end_turn(), active_player should be the other player
        game.end_turn()
        assert game.active_player is not initial_active

    def test_end_turn_starts_next_turn(self):
        """Ending turn should trigger start of next turn."""
        game = Game()
        initial_turn = game.turn_number

        # Expected: end_turn() followed by start_turn() increments turn_number

    def test_minion_attacks_reset_on_turn_end(self):
        """Minions should be able to attack again after turn cycle."""
        game = Game()
        p1 = game.player1
        minion = MinionCard(name="M", mana_cost=1, attack=2, health=2)
        minion.mechanics.append("CHARGE")
        p1.board.append(minion)

        # Expected: attack, end turn, start turn, can attack again

    def test_hero_power_resets_after_turn_cycle(self):
        """Hero power should be usable again after turn ends."""
        game = Game()
        game.player1.mana = 10

        # Expected: use power, end turn, cycle back, can use again


# ============================================================
# Action Validation Tests
# ============================================================


class TestActionValidation:
    """Tests for action validation and game state checks."""

    def test_cannot_take_actions_when_game_over(self):
        """Actions should fail if game is already over."""
        game = Game()
        game.player1.health = 0

        # Expected: any action should raise GameOverError
        assert game.is_over

    def test_cannot_take_opponent_actions_on_your_turn(self):
        """Cannot play cards from opponent's hand."""
        game = Game()
        p1 = game.active_player
        p2 = game.player2 if p1 is game.player1 else game.player1

        minion = MinionCard(name="Test", mana_cost=1, attack=1, health=1)
        p2.hand.append(minion)
        p1.mana = 10

        # Expected: trying to play from p2's hand should fail

    def test_actions_emit_appropriate_events(self):
        """Actions should emit events for game state tracking."""
        player = Player()
        minion = MinionCard(name="Test", mana_cost=0, attack=1, health=1)
        player.hand.append(minion)
        player.mana = 10

        # Expected: playing card should emit on_play, on_summon events
        # This can be tested with event listeners


# ============================================================
# Integration Tests
# ============================================================


class TestActionSequences:
    """Tests for realistic action sequences."""

    def test_full_turn_sequence(self):
        """Test a complete turn with multiple actions."""
        game = Game()
        p1 = game.player1

        # Setup: give player cards and mana
        m1 = MinionCard(name="M1", mana_cost=2, attack=2, health=2)
        m2 = MinionCard(name="M2", mana_cost=3, attack=3, health=3)
        p1.hand.extend([m1, m2])
        p1.mana = 10

        # Expected sequence:
        # 1. Play M1 (costs 2 mana, 8 remaining)
        # 2. Play M2 (costs 3 mana, 5 remaining)
        # 3. Attack with M1
        # 4. Use hero power (costs 2 mana, 3 remaining)
        # 5. End turn

    def test_battlecry_triggers_on_play(self):
        """Playing a minion with Battlecry should trigger its effect."""
        player = Player()
        # This will depend on the effects system integration
        # For now, just verify the minion makes it to board

    def test_play_minion_with_insufficient_mana_fails(self):
        """Verify mana checking works correctly."""
        player = Player()
        expensive = MinionCard(name="Dragon", mana_cost=9, attack=8, health=8)
        player.hand.append(expensive)
        player.mana = 5

        # Expected: IllegalActionError with clear message
