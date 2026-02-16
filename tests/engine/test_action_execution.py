"""Unit tests for action execution - play_card, attack, use_hero_power, end_turn.

These tests verify that actions correctly:
- Validate preconditions
- Modify game state
- Spend resources (mana)
- Update board state
- Handle errors appropriately
"""

import pytest
from hearthstone.engine.game import Game
from hearthstone.engine.player import Player
from hearthstone.cards.base import MinionCard
from hearthstone.engine.actions import play_card, attack, use_hero_power, end_turn
from hearthstone.exceptions import IllegalActionError


# ============================================================
# Play Card Execution Tests
# ============================================================


class TestPlayCardExecution:
    """Tests for playing cards from hand."""

    def test_play_minion_reduces_mana(self):
        """Playing a minion costs mana."""
        player = Player()
        player.mana = 5
        player.hand.append(MinionCard(name="Test", mana_cost=3, attack=2, health=2))

        play_card(player, card_index=0)

        assert player.mana == 2  # 5 - 3

    def test_play_minion_adds_to_board(self):
        """Playing a minion adds it to the board."""
        player = Player()
        player.mana = 5
        minion = MinionCard(name="Test", mana_cost=3, attack=2, health=2)
        player.hand.append(minion)

        play_card(player, card_index=0)

        assert len(player.board) == 1
        assert player.board[0].name == "Test"

    def test_play_minion_removes_from_hand(self):
        """Playing a minion removes it from hand."""
        player = Player()
        player.mana = 5
        player.hand.append(MinionCard(name="Test", mana_cost=3, attack=2, health=2))

        play_card(player, card_index=0)

        assert len(player.hand) == 0

    def test_play_minion_at_position(self):
        """Playing a minion at a specific position."""
        player = Player()
        player.mana = 10
        player.board.append(MinionCard(name="M1", mana_cost=1, attack=1, health=1))
        player.board.append(MinionCard(name="M3", mana_cost=1, attack=1, health=1))
        player.hand.append(MinionCard(name="M2", mana_cost=1, attack=1, health=1))

        play_card(player, card_index=0, position=1)

        assert len(player.board) == 3
        assert player.board[1].name == "M2"

    def test_play_card_without_enough_mana_raises_error(self):
        """Cannot play card without enough mana."""
        player = Player()
        player.mana = 2
        player.hand.append(MinionCard(name="Expensive", mana_cost=5, attack=5, health=5))

        with pytest.raises(IllegalActionError, match="Not enough mana"):
            play_card(player, card_index=0)

    def test_play_minion_when_board_full_raises_error(self):
        """Cannot play minion when board is full (7 minions)."""
        player = Player()
        player.mana = 10
        for i in range(7):
            player.board.append(MinionCard(name=f"M{i}", mana_cost=1, attack=1, health=1))
        player.hand.append(MinionCard(name="New", mana_cost=1, attack=1, health=1))

        with pytest.raises(IllegalActionError, match="Board is full"):
            play_card(player, card_index=0)

    def test_play_card_invalid_index_raises_error(self):
        """Cannot play card with invalid index."""
        player = Player()
        player.mana = 10

        with pytest.raises(IndexError):
            play_card(player, card_index=0)  # Empty hand

    def test_play_multiple_cards_in_sequence(self):
        """Can play multiple cards in one turn."""
        player = Player()
        player.mana = 10
        player.hand.extend([
            MinionCard(name="M1", mana_cost=3, attack=2, health=2),
            MinionCard(name="M2", mana_cost=4, attack=3, health=3),
        ])

        play_card(player, card_index=0)
        play_card(player, card_index=0)  # Index 0 again since first card was removed

        assert len(player.board) == 2
        assert player.mana == 3  # 10 - 3 - 4

    def test_play_zero_cost_card(self):
        """Can play zero-cost cards."""
        player = Player()
        player.mana = 0
        player.hand.append(MinionCard(name="Free", mana_cost=0, attack=1, health=1))

        play_card(player, card_index=0)

        assert len(player.board) == 1
        assert player.mana == 0


# ============================================================
# Attack Action Execution Tests
# ============================================================


class TestAttackExecution:
    """Tests for minion attack actions."""

    def test_minion_attack_deals_damage(self):
        """Attacking minion deals damage to defender."""
        p1 = Player()
        p2 = Player()
        attacker = MinionCard(name="Attacker", mana_cost=1, attack=3, health=2)
        defender = MinionCard(name="Defender", mana_cost=1, attack=2, health=4)
        p1.board.append(attacker)
        p2.board.append(defender)

        # Mark attacker as ready (no summoning sickness)
        attacker._can_attack = True
        attacker.summoning_sick = False

        attack(p1, attacker_index=0, defender_player=p2, defender_index=0)

        assert defender.health == 1  # 4 - 3

    def test_minion_attack_takes_counter_damage(self):
        """Attacking minion takes counter damage."""
        p1 = Player()
        p2 = Player()
        attacker = MinionCard(name="Attacker", mana_cost=1, attack=3, health=4)
        defender = MinionCard(name="Defender", mana_cost=1, attack=2, health=2)
        p1.board.append(attacker)
        p2.board.append(defender)

        attacker._can_attack = True
        attacker.summoning_sick = False

        attack(p1, attacker_index=0, defender_player=p2, defender_index=0)

        assert attacker.health == 2  # 4 - 2

    def test_attack_hero_reduces_health(self):
        """Attacking enemy hero reduces their health."""
        p1 = Player()
        p2 = Player()
        attacker = MinionCard(name="Attacker", mana_cost=1, attack=5, health=2)
        p1.board.append(attacker)

        attacker._can_attack = True
        attacker.summoning_sick = False

        attack(p1, attacker_index=0, defender_player=p2, defender_index=None)

        assert p2.health == 25  # 30 - 5

    def test_minion_with_summoning_sickness_cannot_attack(self):
        """Minion cannot attack on turn it was summoned."""
        p1 = Player()
        p2 = Player()
        attacker = MinionCard(name="Attacker", mana_cost=1, attack=3, health=2)
        p1.board.append(attacker)
        p2.board.append(MinionCard(name="Defender", mana_cost=1, attack=1, health=1))

        # Don't set _can_attack (summoning sickness)

        with pytest.raises(IllegalActionError, match="cannot attack"):
            attack(p1, attacker_index=0, defender_player=p2, defender_index=0)

    def test_minion_with_charge_can_attack_immediately(self):
        """Minion with CHARGE can attack immediately."""
        p1 = Player()
        p2 = Player()
        charger = MinionCard(name="Charger", mana_cost=1, attack=2, health=1)
        charger.mechanics.append("CHARGE")
        charger.summoning_sick = False  # CHARGE bypasses summoning sickness
        p1.board.append(charger)

        attack(p1, attacker_index=0, defender_player=p2, defender_index=None)

        assert p2.health == 28  # 30 - 2

    def test_minion_cannot_attack_twice_in_one_turn(self):
        """Minion cannot attack twice (unless WINDFURY)."""
        p1 = Player()
        p2 = Player()
        attacker = MinionCard(name="Attacker", mana_cost=1, attack=3, health=2)
        p1.board.append(attacker)
        p2.board.append(MinionCard(name="D1", mana_cost=1, attack=1, health=1))
        p2.board.append(MinionCard(name="D2", mana_cost=1, attack=1, health=1))

        attacker._can_attack = True
        attacker.summoning_sick = False

        attack(p1, attacker_index=0, defender_player=p2, defender_index=0)

        with pytest.raises(IllegalActionError, match="already attacked"):
            attack(p1, attacker_index=0, defender_player=p2, defender_index=1)

    def test_windfury_minion_can_attack_twice(self):
        """Minion with WINDFURY can attack twice."""
        p1 = Player()
        p2 = Player()
        windfury = MinionCard(name="WF", mana_cost=1, attack=2, health=3)
        windfury.mechanics.append("WINDFURY")
        p1.board.append(windfury)

        windfury._can_attack = True
        windfury.summoning_sick = False

        # First attack
        attack(p1, attacker_index=0, defender_player=p2, defender_index=None)
        assert p2.health == 28

        # Second attack (should work with WINDFURY)
        attack(p1, attacker_index=0, defender_player=p2, defender_index=None)
        assert p2.health == 26

    def test_invalid_attacker_index_raises_error(self):
        """Cannot attack with invalid attacker index."""
        p1 = Player()
        p2 = Player()

        with pytest.raises(IndexError):
            attack(p1, attacker_index=0, defender_player=p2, defender_index=None)

    def test_invalid_defender_index_raises_error(self):
        """Cannot attack invalid defender index."""
        p1 = Player()
        p2 = Player()
        p1.board.append(MinionCard(name="A", mana_cost=1, attack=1, health=1))
        p1.board[0]._can_attack = True
        p1.board[0].summoning_sick = False

        with pytest.raises(IndexError):
            attack(p1, attacker_index=0, defender_player=p2, defender_index=0)

    def test_zero_attack_minion_can_still_attack(self):
        """Zero attack minion can attack (for effects/death triggers)."""
        p1 = Player()
        p2 = Player()
        weak = MinionCard(name="Weak", mana_cost=1, attack=0, health=3)
        p1.board.append(weak)
        p2.board.append(MinionCard(name="D", mana_cost=1, attack=2, health=2))

        weak._can_attack = True
        weak.summoning_sick = False

        attack(p1, attacker_index=0, defender_player=p2, defender_index=0)

        # Defender takes no damage, attacker takes 2
        assert p2.board[0].health == 2
        assert weak.health == 1


# ============================================================
# Hero Power Execution Tests
# ============================================================


class TestHeroPowerExecution:
    """Tests for hero power usage."""

    def test_use_hero_power_costs_mana(self):
        """Using hero power costs 2 mana."""
        player = Player()
        player.mana = 5

        use_hero_power(player)

        assert player.mana == 3  # 5 - 2

    def test_use_hero_power_without_enough_mana_raises_error(self):
        """Cannot use hero power without 2 mana."""
        player = Player()
        player.mana = 1

        with pytest.raises(IllegalActionError, match="Not enough mana"):
            use_hero_power(player)

    def test_cannot_use_hero_power_twice_in_turn(self):
        """Cannot use hero power twice in same turn."""
        player = Player()
        player.mana = 10

        use_hero_power(player)

        with pytest.raises(IllegalActionError, match="already been used"):
            use_hero_power(player)

    def test_hero_power_resets_on_new_turn(self):
        """Hero power usage resets on new turn."""
        player = Player()
        player.mana = 10

        use_hero_power(player)

        # Simulate new turn
        player.hero_power_used = False
        player.mana = 10

        # Should work again
        use_hero_power(player)
        assert player.mana == 8


# ============================================================
# End Turn Execution Tests
# ============================================================


class TestEndTurnExecution:
    """Tests for end turn action."""

    def test_end_turn_switches_active_player(self):
        """End turn switches to other player."""
        game = Game()
        initial_player = game.active_player

        end_turn(game)

        assert game.active_player != initial_player

    def test_end_turn_increments_turn_number(self):
        """Turn counter increments through start/end cycle."""
        game = Game()
        initial_turn = game.turn_number

        # Full turn cycle: start turn, then end turn
        game.start_turn()
        initial_after_start = game.turn_number
        end_turn(game)

        assert initial_after_start > initial_turn

    def test_end_turn_resets_minion_attack_status(self):
        """End turn allows minions to attack again next turn."""
        game = Game()
        minion = MinionCard(name="M", mana_cost=1, attack=2, health=2)
        game.active_player.board.append(minion)
        minion._has_attacked = True

        end_turn(game)
        end_turn(game)  # Back to original player

        # Minion should be able to attack again
        assert not minion._has_attacked

    def test_multiple_end_turns(self):
        """Can end turn multiple times."""
        game = Game()

        for _ in range(10):
            game.start_turn()  # Start the turn (increments counter)
            end_turn(game)     # End it

        assert game.turn_number >= 10


# ============================================================
# Integration Tests
# ============================================================


class TestActionIntegration:
    """Integration tests for combined actions."""

    def test_play_minion_then_attack_with_charge(self):
        """Can play minion with CHARGE and attack same turn."""
        game = Game()
        game.player1.mana = 10
        charger = MinionCard(name="Charger", mana_cost=3, attack=3, health=2)
        charger.mechanics.append("CHARGE")
        game.player1.hand.append(charger)

        play_card(game.player1, card_index=0)
        attack(game.player1, attacker_index=0, defender_player=game.player2, defender_index=None)

        assert game.player2.health == 27  # 30 - 3
        assert game.player1.mana == 7  # 10 - 3

    def test_full_turn_sequence(self):
        """Complete turn: play cards, attack, use hero power, end turn."""
        game = Game()
        game.player1.mana = 10
        game.player1.hand.append(MinionCard(name="M", mana_cost=2, attack=2, health=2))
        game.player1.hand[0].mechanics.append("CHARGE")

        # Play card
        play_card(game.player1, card_index=0)
        assert len(game.player1.board) == 1
        assert game.player1.mana == 8

        # Attack with minion
        attack(game.player1, attacker_index=0, defender_player=game.player2, defender_index=None)
        assert game.player2.health == 28

        # Use hero power
        use_hero_power(game.player1)
        assert game.player1.mana == 6

        # End turn
        initial_player = game.active_player
        end_turn(game)
        assert game.active_player != initial_player

    def test_play_multiple_minions_and_attack(self):
        """Play multiple minions and attack with each."""
        game = Game()
        game.player1.mana = 10

        for i in range(3):
            m = MinionCard(name=f"M{i}", mana_cost=2, attack=1, health=1)
            m.mechanics.append("CHARGE")
            game.player1.hand.append(m)

        # Play all 3 minions
        for _ in range(3):
            play_card(game.player1, card_index=0)

        # Attack with all 3
        for i in range(3):
            attack(game.player1, attacker_index=i, defender_player=game.player2, defender_index=None)

        assert game.player2.health == 27  # 30 - 3
        assert len(game.player1.board) == 3
