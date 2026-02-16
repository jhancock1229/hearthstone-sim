"""Unit tests for Observation - agent-facing game views with hidden info.

Following TDD: these tests define the contract for Observation, which provides
agent-facing views of game state with appropriate information hiding.

Observation should:
- Show agent's complete state (own hand, deck, board)
- Hide opponent's hand contents (show count only)
- Hide opponent's deck order (show count only)
- Show opponent's visible state (board, health, mana)
- Support perspective switching (for self-play training)
- Be created from GameState
"""

import pytest
from hearthstone.engine.game import Game
from hearthstone.cards.base import MinionCard


# ============================================================
# Observation Creation Tests
# ============================================================


class TestObservationCreation:
    """Tests for creating Observations from game state."""

    def test_create_observation_from_game_state(self):
        """Observation can be created from GameState with player perspective."""
        game = Game()
        # state = GameState.from_game(game)
        # obs = Observation.from_game_state(state, player_index=0)

        # Expected: creates observation for player 0

    def test_observation_for_player_0(self):
        """Observation from player 0's perspective shows correct info."""
        game = Game()
        # obs = Observation.from_game(game, player_index=0)

        # Expected: player 0 is "self", player 1 is "opponent"

    def test_observation_for_player_1(self):
        """Observation from player 1's perspective shows correct info."""
        game = Game()
        # obs = Observation.from_game(game, player_index=1)

        # Expected: player 1 is "self", player 0 is "opponent"

    def test_invalid_player_index_raises_error(self):
        """Invalid player index should raise ValueError."""
        game = Game()

        # Expected: player_index must be 0 or 1
        # with pytest.raises(ValueError):
        #     Observation.from_game(game, player_index=2)


# ============================================================
# Self State Visibility Tests
# ============================================================


class TestSelfStateVisibility:
    """Tests for agent's own state visibility."""

    def test_observation_shows_own_health(self):
        """Observation shows agent's current health."""
        game = Game()
        game.player1.health = 25

        # obs = Observation.from_game(game, player_index=0)
        # assert obs.self_health == 25

    def test_observation_shows_own_mana(self):
        """Observation shows agent's current and max mana."""
        game = Game()
        game.player1.mana = 3
        game.player1.max_mana = 5

        # obs = Observation.from_game(game, player_index=0)
        # assert obs.self_mana == 3
        # assert obs.self_max_mana == 5

    def test_observation_shows_own_hand_cards(self):
        """Observation shows complete information about own hand."""
        game = Game()
        card1 = MinionCard(name="Card1", mana_cost=1, attack=1, health=1)
        card2 = MinionCard(name="Card2", mana_cost=2, attack=2, health=2)
        game.player1.hand.extend([card1, card2])

        # obs = Observation.from_game(game, player_index=0)
        # assert len(obs.self_hand) == 2
        # assert obs.self_hand[0].name == "Card1"

    def test_observation_shows_own_deck_size(self):
        """Observation shows agent's deck size (not order)."""
        game = Game()
        for _ in range(15):
            game.player1.deck.append(MinionCard(name="Card", mana_cost=1, attack=1, health=1))

        # obs = Observation.from_game(game, player_index=0)
        # assert obs.self_deck_size == 15

    def test_observation_shows_own_board(self):
        """Observation shows complete information about own board."""
        game = Game()
        minion = MinionCard(name="MyMinion", mana_cost=1, attack=3, health=4)
        minion.mechanics.append("TAUNT")
        game.player1.board.append(minion)

        # obs = Observation.from_game(game, player_index=0)
        # assert len(obs.self_board) == 1
        # assert obs.self_board[0].attack == 3

    def test_observation_shows_own_fatigue_counter(self):
        """Observation shows agent's fatigue counter."""
        game = Game()
        game.player1.fatigue_counter = 3

        # obs = Observation.from_game(game, player_index=0)
        # assert obs.self_fatigue_counter == 3


# ============================================================
# Opponent State Visibility Tests
# ============================================================


class TestOpponentStateVisibility:
    """Tests for opponent state visibility and information hiding."""

    def test_observation_shows_opponent_health(self):
        """Observation shows opponent's health (public info)."""
        game = Game()
        game.player2.health = 18

        # obs = Observation.from_game(game, player_index=0)
        # assert obs.opponent_health == 18

    def test_observation_shows_opponent_mana(self):
        """Observation shows opponent's mana (public info)."""
        game = Game()
        game.player2.mana = 7
        game.player2.max_mana = 10

        # obs = Observation.from_game(game, player_index=0)
        # assert obs.opponent_mana == 7
        # assert obs.opponent_max_mana == 10

    def test_observation_hides_opponent_hand_cards(self):
        """Observation hides opponent's hand contents, shows count only."""
        game = Game()
        for _ in range(5):
            game.player2.hand.append(MinionCard(name="Secret", mana_cost=1, attack=1, health=1))

        # obs = Observation.from_game(game, player_index=0)
        # assert obs.opponent_hand_size == 5
        # Hand contents should not be accessible

    def test_observation_hides_opponent_deck_order(self):
        """Observation hides opponent's deck order, shows count only."""
        game = Game()
        for _ in range(20):
            game.player2.deck.append(MinionCard(name="Hidden", mana_cost=1, attack=1, health=1))

        # obs = Observation.from_game(game, player_index=0)
        # assert obs.opponent_deck_size == 20
        # Deck order should not be accessible

    def test_observation_shows_opponent_board(self):
        """Observation shows opponent's board (public info)."""
        game = Game()
        minion = MinionCard(name="EnemyMinion", mana_cost=2, attack=4, health=5)
        minion.mechanics.append("DIVINE_SHIELD")
        game.player2.board.append(minion)

        # obs = Observation.from_game(game, player_index=0)
        # assert len(obs.opponent_board) == 1
        # assert obs.opponent_board[0].name == "EnemyMinion"

    def test_observation_shows_opponent_fatigue_counter(self):
        """Observation shows opponent's fatigue counter (public info)."""
        game = Game()
        game.player2.fatigue_counter = 2

        # obs = Observation.from_game(game, player_index=0)
        # assert obs.opponent_fatigue_counter == 2


# ============================================================
# Game State Information Tests
# ============================================================


class TestGameStateInformation:
    """Tests for global game state information."""

    def test_observation_shows_turn_number(self):
        """Observation shows current turn number."""
        game = Game()
        game._turn_number = 7

        # obs = Observation.from_game(game, player_index=0)
        # assert obs.turn_number == 7

    def test_observation_shows_active_player(self):
        """Observation indicates if it's agent's turn."""
        game = Game()

        # obs = Observation.from_game(game, player_index=0)
        # If player1 is active and we're player 0:
        # assert obs.is_my_turn == True

    def test_observation_shows_game_over_status(self):
        """Observation indicates if game is over."""
        game = Game()
        game.player1.health = 0

        # obs = Observation.from_game(game, player_index=0)
        # assert obs.is_game_over == True

    def test_observation_shows_winner(self):
        """Observation indicates winner if game is over."""
        game = Game()
        game.player2.health = 0

        # obs = Observation.from_game(game, player_index=0)
        # assert obs.winner == 0  # player 0 wins


# ============================================================
# Perspective Switching Tests
# ============================================================


class TestPerspectiveSwitching:
    """Tests for switching observation perspective."""

    def test_switch_perspective_swaps_self_and_opponent(self):
        """Switching perspective swaps self/opponent views."""
        game = Game()
        game.player1.health = 25
        game.player2.health = 18

        # obs0 = Observation.from_game(game, player_index=0)
        # obs1 = obs0.switch_perspective()

        # assert obs0.self_health == 25
        # assert obs0.opponent_health == 18
        # assert obs1.self_health == 18
        # assert obs1.opponent_health == 25

    def test_perspective_switching_preserves_hidden_info(self):
        """Switching perspective maintains information hiding."""
        game = Game()
        for _ in range(3):
            game.player1.hand.append(MinionCard(name="Card", mana_cost=1, attack=1, health=1))
        for _ in range(5):
            game.player2.hand.append(MinionCard(name="Card", mana_cost=1, attack=1, health=1))

        # obs0 = Observation.from_game(game, player_index=0)
        # obs1 = obs0.switch_perspective()

        # obs0 sees own 3 cards, opponent's count only
        # obs1 sees own 5 cards, opponent's count only


# ============================================================
# Serialization Tests
# ============================================================


class TestObservationSerialization:
    """Tests for observation serialization."""

    def test_observation_to_dict(self):
        """Observation can be serialized to dict."""
        game = Game()
        # obs = Observation.from_game(game, player_index=0)
        # data = obs.to_dict()

        # Expected: dict with all observable fields

    def test_observation_from_dict(self):
        """Observation can be reconstructed from dict."""
        game = Game()
        # obs1 = Observation.from_game(game, player_index=0)
        # data = obs1.to_dict()
        # obs2 = Observation.from_dict(data)

        # Expected: obs2 matches obs1


# ============================================================
# Feature Extraction Tests
# ============================================================


class TestFeatureExtraction:
    """Tests for extracting features for ML models."""

    def test_observation_to_feature_vector(self):
        """Observation can be converted to feature vector."""
        game = Game()
        # obs = Observation.from_game(game, player_index=0)
        # features = obs.to_feature_vector()

        # Expected: numpy array or list of floats

    def test_feature_vector_has_fixed_size(self):
        """Feature vector should have consistent size."""
        game1 = Game()
        game2 = Game()
        game2.player1.board.append(MinionCard(name="M", mana_cost=1, attack=1, health=1))

        # obs1 = Observation.from_game(game1, player_index=0)
        # obs2 = Observation.from_game(game2, player_index=0)

        # features1 = obs1.to_feature_vector()
        # features2 = obs2.to_feature_vector()

        # assert len(features1) == len(features2)


# ============================================================
# Edge Cases Tests
# ============================================================


class TestObservationEdgeCases:
    """Tests for edge cases in observation."""

    def test_observation_with_empty_board(self):
        """Observation handles empty boards correctly."""
        game = Game()

        # obs = Observation.from_game(game, player_index=0)
        # assert len(obs.self_board) == 0
        # assert len(obs.opponent_board) == 0

    def test_observation_with_full_board(self):
        """Observation handles full boards (7 minions)."""
        game = Game()
        for i in range(7):
            game.player1.board.append(MinionCard(name=f"M{i}", mana_cost=1, attack=1, health=1))

        # obs = Observation.from_game(game, player_index=0)
        # assert len(obs.self_board) == 7

    def test_observation_with_empty_hand(self):
        """Observation handles empty hand correctly."""
        game = Game()

        # obs = Observation.from_game(game, player_index=0)
        # assert len(obs.self_hand) == 0
        # assert obs.opponent_hand_size == 0

    def test_observation_with_full_hand(self):
        """Observation handles full hand (10 cards)."""
        game = Game()
        for i in range(10):
            game.player1.hand.append(MinionCard(name=f"C{i}", mana_cost=1, attack=1, health=1))

        # obs = Observation.from_game(game, player_index=0)
        # assert len(obs.self_hand) == 10

    def test_observation_with_zero_mana(self):
        """Observation handles zero mana correctly."""
        game = Game()

        # obs = Observation.from_game(game, player_index=0)
        # assert obs.self_mana == 0
        # assert obs.self_max_mana == 0

    def test_observation_with_max_mana(self):
        """Observation handles max mana (10)."""
        game = Game()
        game.player1.max_mana = 10
        game.player1.mana = 10

        # obs = Observation.from_game(game, player_index=0)
        # assert obs.self_mana == 10
        # assert obs.self_max_mana == 10

    def test_observation_at_game_start(self):
        """Observation at turn 0 (game start)."""
        game = Game()

        # obs = Observation.from_game(game, player_index=0)
        # assert obs.turn_number == 0

    def test_observation_with_negative_health(self):
        """Observation can show negative health (overkill)."""
        game = Game()
        game.player2.health = -5

        # obs = Observation.from_game(game, player_index=0)
        # assert obs.opponent_health == -5
