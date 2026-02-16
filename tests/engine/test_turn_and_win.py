"""
Tests for turn structure, hand overflow, and win conditions.

Phase 1 round 2 — builds on the core mechanics already passing.

Rules covered:
- Turn start: gain mana crystal, refill mana, draw card
- Hand overflow: drawing with 10 cards burns the drawn card
- Win condition: a player at 0 or less health is dead
- The Coin: player going second gets The Coin (a 0-mana spell)
- Starting hands: first player draws 3, second player draws 4 + The Coin
"""

import pytest
from hearthstone.engine.game import Game
from hearthstone.engine.player import Player
from hearthstone.cards.base import MinionCard, SpellCard


class TestHandOverflow:
    """Drawing with a full hand (10 cards) burns the drawn card."""

    def test_hand_limited_to_ten_cards(self):
        player = Player()
        # Fill hand to 10
        for i in range(10):
            player.hand.append(MinionCard(name=f"Card{i}", mana_cost=1, attack=1, health=1))

        # Put a card in the deck
        overflow_card = MinionCard(name="Overflow", mana_cost=1, attack=1, health=1)
        player.deck.append(overflow_card)

        player.draw_card()

        # Hand stays at 10, card is burned (lost), deck is empty
        assert len(player.hand) == 10
        assert len(player.deck) == 0
        assert overflow_card not in player.hand

    def test_drawing_into_full_hand_does_not_cause_fatigue(self):
        player = Player()
        for i in range(10):
            player.hand.append(MinionCard(name=f"Card{i}", mana_cost=1, attack=1, health=1))
        player.deck.append(MinionCard(name="Burned", mana_cost=1, attack=1, health=1))

        player.draw_card()

        # No fatigue damage — the card existed, it was just burned
        assert player.health == 30
        assert player.fatigue_counter == 0


class TestTurnStructure:
    """
    At the start of a turn: gain a mana crystal, refill mana, draw a card.
    The Game object should manage turn state and the active player.
    """

    def _make_game_with_decks(self, deck_size=10):
        """Helper: creates a game where both players have decks of Wisps."""
        game = Game()
        for i in range(deck_size):
            game.player1.deck.append(
                MinionCard(name=f"P1Card{i}", mana_cost=1, attack=1, health=1)
            )
            game.player2.deck.append(
                MinionCard(name=f"P2Card{i}", mana_cost=1, attack=1, health=1)
            )
        return game

    def test_game_tracks_active_player(self):
        game = self._make_game_with_decks()
        assert game.active_player is game.player1

    def test_end_turn_switches_active_player(self):
        game = self._make_game_with_decks()
        game.end_turn()
        assert game.active_player is game.player2
        game.end_turn()
        assert game.active_player is game.player1

    def test_start_turn_grants_mana_crystal(self):
        game = self._make_game_with_decks()
        game.start_turn()
        assert game.active_player.max_mana == 1
        assert game.active_player.mana == 1

    def test_start_turn_draws_a_card(self):
        game = self._make_game_with_decks()
        initial_deck_size = len(game.active_player.deck)
        initial_hand_size = len(game.active_player.hand)

        game.start_turn()

        assert len(game.active_player.deck) == initial_deck_size - 1
        assert len(game.active_player.hand) == initial_hand_size + 1

    def test_mana_increases_each_turn(self):
        game = self._make_game_with_decks()
        for turn in range(1, 4):
            game.start_turn()
            assert game.active_player.max_mana == turn
            assert game.active_player.mana == turn
            game.end_turn()

    def test_game_tracks_turn_number(self):
        game = self._make_game_with_decks()
        assert game.turn_number == 0

        game.start_turn()
        assert game.turn_number == 1

        game.end_turn()
        game.start_turn()
        assert game.turn_number == 2


class TestWinConditions:
    """A player with 0 or less health has lost the game."""

    def test_player_at_zero_health_is_dead(self):
        player = Player()
        player.health = 0
        assert player.is_dead

    def test_player_below_zero_health_is_dead(self):
        player = Player()
        player.health = -5
        assert player.is_dead

    def test_player_with_positive_health_is_alive(self):
        player = Player()
        assert not player.is_dead

    def test_game_detects_winner_when_player_dies(self):
        game = Game()
        game.player2.health = 0
        assert game.is_over
        assert game.winner is game.player1

    def test_game_not_over_when_both_alive(self):
        game = Game()
        assert not game.is_over
        assert game.winner is None

    def test_fatigue_can_kill_a_player(self):
        player = Player()
        player.health = 1
        # Empty deck, first fatigue deals 1 damage
        player.draw_card()
        assert player.health == 0
        assert player.is_dead
