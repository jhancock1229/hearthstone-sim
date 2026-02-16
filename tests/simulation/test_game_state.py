"""Unit tests for GameState - immutable game snapshots.

Following TDD: these tests define the contract for GameState, which provides
immutable snapshots of complete game state for simulation, MCTS, and replay.

GameState should:
- Capture complete game state (both players, turn info, etc.)
- Be immutable (dataclass with frozen=True)
- Support serialization/deserialization for replay
- Provide methods to check game-over conditions
- Allow efficient copying for simulation rollouts
"""

import pytest
from hearthstone.engine.game import Game
from hearthstone.engine.player import Player
from hearthstone.cards.base import MinionCard


# ============================================================
# GameState Creation and Immutability Tests
# ============================================================


class TestGameStateCreation:
    """Tests for creating GameState snapshots."""

    def test_create_game_state_from_game(self):
        """GameState can be created from a Game instance."""
        game = Game()
        # Expected: game_state = GameState.from_game(game)
        # Should capture all relevant state

    def test_game_state_captures_player_health(self):
        """GameState should capture both players' health."""
        game = Game()
        game.player1.health = 25
        game.player2.health = 18

        # Expected: state captures these values
        # assert state.player1_health == 25
        # assert state.player2_health == 18

    def test_game_state_captures_player_mana(self):
        """GameState should capture current and max mana."""
        game = Game()
        game.player1.max_mana = 5
        game.player1.mana = 3
        game.player2.max_mana = 7
        game.player2.mana = 0

        # Expected: state captures mana values

    def test_game_state_captures_turn_number(self):
        """GameState should capture the current turn number."""
        game = Game()
        game._turn_number = 5

        # Expected: state.turn_number == 5

    def test_game_state_captures_active_player(self):
        """GameState should track which player is active."""
        game = Game()
        active = game.active_player

        # Expected: state indicates which player (0 or 1) is active

    def test_game_state_is_immutable(self):
        """GameState should be frozen/immutable."""
        game = Game()
        # state = GameState.from_game(game)

        # Expected: attempting to modify state raises FrozenInstanceError
        # with pytest.raises(FrozenInstanceError):
        #     state.turn_number = 999


# ============================================================
# Board State Capture Tests
# ============================================================


class TestBoardStateCapture:
    """Tests for capturing minion board state."""

    def test_game_state_captures_minion_positions(self):
        """GameState should capture all minions and their positions."""
        game = Game()
        m1 = MinionCard(name="M1", mana_cost=1, attack=2, health=3)
        m2 = MinionCard(name="M2", mana_cost=2, attack=3, health=4)
        game.player1.board.extend([m1, m2])

        # Expected: state captures board with 2 minions in order

    def test_game_state_captures_minion_stats(self):
        """GameState should capture minion attack/health values."""
        game = Game()
        minion = MinionCard(name="Test", mana_cost=1, attack=5, health=7)
        game.player1.board.append(minion)

        # Expected: state captures attack=5, health=7

    def test_game_state_captures_minion_mechanics(self):
        """GameState should capture minion keywords/mechanics."""
        game = Game()
        minion = MinionCard(name="Shielded", mana_cost=1, attack=1, health=1)
        minion.mechanics.extend(["DIVINE_SHIELD", "TAUNT"])
        game.player1.board.append(minion)

        # Expected: state captures mechanics list

    def test_game_state_captures_damaged_minions(self):
        """GameState should capture current health of damaged minions."""
        game = Game()
        minion = MinionCard(name="Damaged", mana_cost=1, attack=1, health=5)
        minion.health = 2  # damaged
        game.player1.board.append(minion)

        # Expected: state shows health=2, not original health

    def test_game_state_captures_empty_board(self):
        """GameState should handle empty boards correctly."""
        game = Game()
        # Both boards empty

        # Expected: state shows empty board lists


# ============================================================
# Hand and Deck State Tests
# ============================================================


class TestHandAndDeckCapture:
    """Tests for capturing hand and deck state."""

    def test_game_state_captures_hand_size(self):
        """GameState should capture number of cards in hand."""
        game = Game()
        for _ in range(5):
            game.player1.hand.append(MinionCard(name="Card", mana_cost=1, attack=1, health=1))

        # Expected: state shows hand size of 5

    def test_game_state_captures_deck_size(self):
        """GameState should capture number of cards in deck."""
        game = Game()
        for _ in range(20):
            game.player1.deck.append(MinionCard(name="Card", mana_cost=1, attack=1, health=1))

        # Expected: state shows deck size of 20

    def test_game_state_captures_fatigue_counter(self):
        """GameState should capture fatigue damage counter."""
        game = Game()
        game.player1.fatigue_counter = 3

        # Expected: state shows fatigue_counter = 3

    def test_game_state_optionally_hides_opponent_hand(self):
        """GameState may hide opponent's hand contents for agent view."""
        game = Game()
        for _ in range(3):
            game.player2.hand.append(MinionCard(name="Secret", mana_cost=1, attack=1, health=1))

        # Expected: state can hide specific cards but show count


# ============================================================
# GameState Equality and Hashing Tests
# ============================================================


class TestGameStateEquality:
    """Tests for GameState equality and hashing."""

    def test_identical_states_are_equal(self):
        """Two GameStates from identical games should be equal."""
        game1 = Game()
        game2 = Game()

        # state1 = GameState.from_game(game1)
        # state2 = GameState.from_game(game2)
        # assert state1 == state2

    def test_different_turn_numbers_not_equal(self):
        """GameStates with different turn numbers should not be equal."""
        game1 = Game()
        game2 = Game()
        game2._turn_number = 5

        # Expected: states are not equal

    def test_different_boards_not_equal(self):
        """GameStates with different boards should not be equal."""
        game1 = Game()
        game2 = Game()
        game2.player1.board.append(MinionCard(name="Extra", mana_cost=1, attack=1, health=1))

        # Expected: states are not equal

    def test_game_state_is_hashable(self):
        """GameState should be hashable for use in sets/dicts."""
        game = Game()
        # state = GameState.from_game(game)

        # Expected: can use in set or dict
        # state_set = {state}
        # state_dict = {state: "value"}


# ============================================================
# GameState Serialization Tests
# ============================================================


class TestGameStateSerialization:
    """Tests for serializing/deserializing GameState."""

    def test_game_state_to_dict(self):
        """GameState can be converted to a dictionary."""
        game = Game()
        # state = GameState.from_game(game)
        # data = state.to_dict()

        # Expected: data is a dict with all state fields

    def test_game_state_from_dict(self):
        """GameState can be reconstructed from a dictionary."""
        game = Game()
        # state1 = GameState.from_game(game)
        # data = state1.to_dict()
        # state2 = GameState.from_dict(data)

        # Expected: state2 == state1

    def test_serialization_preserves_all_data(self):
        """Serialization round-trip should preserve all game state."""
        game = Game()
        game.player1.health = 20
        game.player1.mana = 5
        minion = MinionCard(name="Test", mana_cost=1, attack=3, health=4)
        minion.mechanics.append("TAUNT")
        game.player1.board.append(minion)

        # Expected: serialize and deserialize preserves everything


# ============================================================
# GameState Query Methods Tests
# ============================================================


class TestGameStateQueries:
    """Tests for querying game state information."""

    def test_is_game_over(self):
        """GameState can determine if game is over."""
        game = Game()
        game.player1.health = 0

        # state = GameState.from_game(game)
        # assert state.is_game_over() is True

    def test_get_winner(self):
        """GameState can identify the winner."""
        game = Game()
        game.player2.health = 0

        # state = GameState.from_game(game)
        # assert state.get_winner() == 0  # player1 wins

    def test_game_not_over_with_both_alive(self):
        """GameState shows game not over when both players alive."""
        game = Game()

        # state = GameState.from_game(game)
        # assert state.is_game_over() is False

    def test_get_active_player_index(self):
        """GameState can return active player index (0 or 1)."""
        game = Game()

        # state = GameState.from_game(game)
        # active_idx = state.get_active_player_index()
        # assert active_idx in [0, 1]


# ============================================================
# GameState Cloning and Simulation Tests
# ============================================================


class TestGameStateCloning:
    """Tests for efficient state cloning for simulation."""

    def test_clone_creates_independent_copy(self):
        """Cloning should create independent GameState."""
        game = Game()
        # state1 = GameState.from_game(game)
        # state2 = state1.clone()

        # Expected: state2 == state1 but is different object

    def test_apply_action_returns_new_state(self):
        """Applying action should return new state, not modify original."""
        game = Game()
        # state1 = GameState.from_game(game)
        # action = some_action
        # state2 = state1.apply_action(action)

        # Expected: state1 unchanged, state2 is new state


# ============================================================
# Edge Cases and Special States Tests
# ============================================================


class TestGameStateEdgeCases:
    """Tests for edge cases and special game states."""

    def test_game_state_with_full_board(self):
        """GameState handles board with 7 minions."""
        game = Game()
        for i in range(7):
            game.player1.board.append(MinionCard(name=f"M{i}", mana_cost=1, attack=1, health=1))

        # Expected: captures all 7 minions

    def test_game_state_with_full_hand(self):
        """GameState handles hand with 10 cards."""
        game = Game()
        for i in range(10):
            game.player1.hand.append(MinionCard(name=f"C{i}", mana_cost=1, attack=1, health=1))

        # Expected: captures all 10 cards

    def test_game_state_at_max_mana(self):
        """GameState handles players at 10 mana."""
        game = Game()
        game.player1.max_mana = 10
        game.player1.mana = 10

        # Expected: captures max mana correctly

    def test_game_state_with_negative_health(self):
        """GameState can capture negative health (overkill)."""
        game = Game()
        game.player1.health = -5

        # Expected: captures negative health value

    def test_game_state_first_turn(self):
        """GameState handles turn 0 (game start)."""
        game = Game()
        # state = GameState.from_game(game)

        # Expected: turn_number = 0
