"""
Tests for Hearthstone core game mechanics.

Following TDD: these tests are written FIRST, before any implementation.
They define the contract that our game engine must fulfill.

We start with the most fundamental mechanics:
- Game initialization
- Mana system
- Hero health
- Card drawing & fatigue
"""

import pytest
from hearthstone.engine.game import Game
from hearthstone.engine.player import Player
from hearthstone.cards.base import MinionCard


# ============================================================
# 1. Game Initialization
# ============================================================

class TestGameInitialization:
    """A new game should set up two players with correct starting state."""

    def test_new_game_has_two_players(self):
        game = Game()
        assert game.player1 is not None
        assert game.player2 is not None
        assert game.player1 is not game.player2

    def test_players_start_with_30_health(self):
        game = Game()
        assert game.player1.health == 30
        assert game.player2.health == 30


# ============================================================
# 2. Mana System
# ============================================================

class TestManaSystem:
    """
    Each turn, a player gains one mana crystal (up to 10 max).
    At the start of each turn, current mana refills to max.
    """

    def test_player_starts_with_zero_mana(self):
        player = Player()
        assert player.mana == 0
        assert player.max_mana == 0

    def test_gaining_mana_crystal_increases_max_mana(self):
        player = Player()
        player.gain_mana_crystal()
        assert player.max_mana == 1

    def test_max_mana_capped_at_ten(self):
        player = Player()
        for _ in range(15):
            player.gain_mana_crystal()
        assert player.max_mana == 10

    def test_refill_mana_restores_to_max(self):
        player = Player()
        player.gain_mana_crystal()
        player.gain_mana_crystal()
        player.refill_mana()
        assert player.mana == 2

    def test_spending_mana_reduces_current_mana(self):
        player = Player()
        for _ in range(5):
            player.gain_mana_crystal()
        player.refill_mana()

        player.spend_mana(3)
        assert player.mana == 2

    def test_cannot_spend_more_mana_than_available(self):
        player = Player()
        player.gain_mana_crystal()
        player.refill_mana()

        with pytest.raises(ValueError, match="Not enough mana"):
            player.spend_mana(5)


# ============================================================
# 3. Card Drawing & Fatigue
# ============================================================

class TestCardDrawing:
    """
    Players draw from their deck into their hand.
    Drawing from an empty deck causes fatigue damage (1, then 2, then 3...).
    Hand size is limited to 10 cards.
    """

    def test_drawing_a_card_moves_it_from_deck_to_hand(self):
        player = Player()
        card = MinionCard(name="Wisp", mana_cost=0, attack=1, health=1)
        player.deck.append(card)

        player.draw_card()

        assert len(player.hand) == 1
        assert len(player.deck) == 0
        assert player.hand[0] is card

    def test_fatigue_damage_increases_each_empty_draw(self):
        player = Player()
        # Deck is empty — drawing should cause escalating fatigue

        player.draw_card()  # 1 fatigue damage
        assert player.health == 29

        player.draw_card()  # 2 fatigue damage
        assert player.health == 27

        player.draw_card()  # 3 fatigue damage
        assert player.health == 24
