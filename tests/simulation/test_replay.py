"""Unit tests for Replay - game reconstruction from action logs.

Following TDD: these tests define the contract for Replay, which enables
reconstructing games from seed + action log for debugging and analysis.

Replay should:
- Record game seed and action sequence
- Reconstruct exact game states from log
- Support save/load of replay files
- Enable step-by-step replay
- Verify determinism
- Support replay analysis
"""

import pytest
from simulation.action_space import Action, ActionType
from agents.random_agent import RandomAgent


# ============================================================
# Replay Recording Tests
# ============================================================


class TestReplayRecording:
    """Tests for recording game replays."""

    def test_replay_recorder_can_be_created(self):
        """ReplayRecorder can be instantiated."""
        from simulation.replay import ReplayRecorder

        recorder = ReplayRecorder()
        assert recorder is not None

    def test_recorder_captures_seed(self):
        """Recorder captures game seed."""
        from simulation.replay import ReplayRecorder

        recorder = ReplayRecorder(seed=12345)
        assert recorder.seed == 12345

    def test_recorder_tracks_actions(self):
        """Recorder tracks action sequence."""
        from simulation.replay import ReplayRecorder

        recorder = ReplayRecorder()
        action = Action(type=ActionType.END_TURN)

        recorder.record_action(0, action)  # Player 0, action
        assert len(recorder.actions) == 1

    def test_recorder_tracks_multiple_actions(self):
        """Recorder tracks sequence of actions."""
        from simulation.replay import ReplayRecorder

        recorder = ReplayRecorder()
        actions = [
            Action(type=ActionType.PLAY_CARD, card_index=0),
            Action(type=ActionType.ATTACK, attacker_index=0, defender_index=0),
            Action(type=ActionType.END_TURN),
        ]

        for i, action in enumerate(actions):
            recorder.record_action(0, action)

        assert len(recorder.actions) == 3

    def test_recorder_tracks_player_index(self):
        """Recorder tracks which player took each action."""
        from simulation.replay import ReplayRecorder

        recorder = ReplayRecorder()
        recorder.record_action(0, Action(type=ActionType.END_TURN))
        recorder.record_action(1, Action(type=ActionType.HERO_POWER))

        assert recorder.actions[0][0] == 0  # Player index
        assert recorder.actions[1][0] == 1

    def test_recorder_includes_turn_number(self):
        """Recorder tracks turn number for each action."""
        from simulation.replay import ReplayRecorder

        recorder = ReplayRecorder()
        recorder.record_action(0, Action(type=ActionType.END_TURN), turn=1)
        recorder.record_action(1, Action(type=ActionType.END_TURN), turn=2)

        # Turn info should be preserved
        assert recorder.actions[0][2] == 1  # Turn number
        assert recorder.actions[1][2] == 2


# ============================================================
# Replay Serialization Tests
# ============================================================


class TestReplaySerialization:
    """Tests for saving and loading replays."""

    def test_replay_can_be_serialized_to_dict(self):
        """Replay can be converted to dictionary."""
        from simulation.replay import ReplayRecorder

        recorder = ReplayRecorder(seed=42)
        recorder.record_action(0, Action(type=ActionType.END_TURN))

        replay_dict = recorder.to_dict()
        assert isinstance(replay_dict, dict)
        assert 'seed' in replay_dict
        assert 'actions' in replay_dict

    def test_replay_can_be_saved_to_file(self):
        """Replay can be saved to JSON file."""
        from simulation.replay import ReplayRecorder
        import tempfile
        import os

        recorder = ReplayRecorder(seed=42)
        recorder.record_action(0, Action(type=ActionType.END_TURN))

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            filepath = f.name

        try:
            recorder.save(filepath)
            assert os.path.exists(filepath)
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)

    def test_replay_can_be_loaded_from_file(self):
        """Replay can be loaded from JSON file."""
        from simulation.replay import ReplayRecorder, load_replay
        import tempfile
        import os

        recorder = ReplayRecorder(seed=42)
        recorder.record_action(0, Action(type=ActionType.END_TURN))

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            filepath = f.name

        try:
            recorder.save(filepath)
            loaded = load_replay(filepath)

            assert loaded.seed == 42
            assert len(loaded.actions) == 1
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)

    def test_loaded_replay_matches_original(self):
        """Loaded replay exactly matches saved replay."""
        from simulation.replay import ReplayRecorder, load_replay
        import tempfile
        import os

        recorder = ReplayRecorder(seed=99)
        recorder.record_action(0, Action(type=ActionType.PLAY_CARD, card_index=1))
        recorder.record_action(1, Action(type=ActionType.HERO_POWER))
        recorder.record_action(0, Action(type=ActionType.END_TURN))

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            filepath = f.name

        try:
            recorder.save(filepath)
            loaded = load_replay(filepath)

            assert loaded.seed == recorder.seed
            assert len(loaded.actions) == len(recorder.actions)
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)


# ============================================================
# Replay Playback Tests
# ============================================================


class TestReplayPlayback:
    """Tests for replaying games from logs."""

    def test_replay_player_can_be_created(self):
        """ReplayPlayer can be instantiated."""
        from simulation.replay import ReplayPlayer, ReplayRecorder

        recorder = ReplayRecorder(seed=42)
        player = ReplayPlayer(recorder)

        assert player is not None

    def test_replay_player_recreates_game_state(self):
        """ReplayPlayer can reconstruct game from log."""
        from simulation.replay import ReplayPlayer, ReplayRecorder

        recorder = ReplayRecorder(seed=42)
        recorder.record_action(0, Action(type=ActionType.END_TURN))

        player = ReplayPlayer(recorder)
        # Should be able to initialize game from seed
        assert player.seed == 42

    def test_replay_player_steps_through_actions(self):
        """ReplayPlayer can step through action sequence."""
        from simulation.replay import ReplayPlayer, ReplayRecorder

        recorder = ReplayRecorder(seed=42)
        recorder.record_action(0, Action(type=ActionType.END_TURN))
        recorder.record_action(1, Action(type=ActionType.END_TURN))

        player = ReplayPlayer(recorder)
        # Step forward
        player.step()
        assert player.current_action_index == 1

        player.step()
        assert player.current_action_index == 2

    def test_replay_player_can_reset(self):
        """ReplayPlayer can reset to beginning."""
        from simulation.replay import ReplayPlayer, ReplayRecorder

        recorder = ReplayRecorder(seed=42)
        recorder.record_action(0, Action(type=ActionType.END_TURN))
        recorder.record_action(1, Action(type=ActionType.END_TURN))

        player = ReplayPlayer(recorder)
        player.step()
        player.step()

        player.reset()
        assert player.current_action_index == 0

    def test_replay_player_can_play_to_end(self):
        """ReplayPlayer can play entire game."""
        from simulation.replay import ReplayPlayer, ReplayRecorder

        recorder = ReplayRecorder(seed=42)
        for i in range(10):
            recorder.record_action(i % 2, Action(type=ActionType.END_TURN))

        player = ReplayPlayer(recorder)
        player.play_all()

        assert player.current_action_index == 10


# ============================================================
# Determinism Tests
# ============================================================


class TestReplayDeterminism:
    """Tests for replay determinism verification."""

    def test_same_seed_produces_same_game(self):
        """Same seed + actions produces identical game."""
        from simulation.replay import ReplayRecorder
        from simulation.simulator import Simulator

        # Run a game with seeded agents
        sim = Simulator(seed=42)
        agent1 = RandomAgent(seed=1)
        agent2 = RandomAgent(seed=2)

        result1 = sim.run_games(agent1, agent2, num_games=1)

        # Run again with same seeds
        sim2 = Simulator(seed=42)
        agent1_2 = RandomAgent(seed=1)
        agent2_2 = RandomAgent(seed=2)

        result2 = sim2.run_games(agent1_2, agent2_2, num_games=1)

        # Results should be identical
        assert result1.games[0].winner == result2.games[0].winner
        assert result1.games[0].turns == result2.games[0].turns

    def test_replay_matches_original_game(self):
        """Replayed game matches original execution."""
        from simulation.replay import ReplayRecorder
        # This test verifies that replaying a recorded game
        # produces the exact same game state
        # Implementation depends on full action execution

        recorder = ReplayRecorder(seed=123)
        # Record a game...
        # Replay it...
        # Verify states match
        pass  # Placeholder for full implementation


# ============================================================
# Replay Analysis Tests
# ============================================================


class TestReplayAnalysis:
    """Tests for replay analysis features."""

    def test_replay_provides_action_count(self):
        """Replay can report total actions."""
        from simulation.replay import ReplayRecorder

        recorder = ReplayRecorder()
        for i in range(15):
            recorder.record_action(i % 2, Action(type=ActionType.END_TURN))

        assert recorder.action_count() == 15

    def test_replay_provides_action_breakdown(self):
        """Replay can analyze action type distribution."""
        from simulation.replay import ReplayRecorder

        recorder = ReplayRecorder()
        recorder.record_action(0, Action(type=ActionType.PLAY_CARD, card_index=0))
        recorder.record_action(0, Action(type=ActionType.ATTACK, attacker_index=0))
        recorder.record_action(0, Action(type=ActionType.END_TURN))
        recorder.record_action(1, Action(type=ActionType.END_TURN))

        breakdown = recorder.action_breakdown()
        assert breakdown[ActionType.END_TURN] == 2
        assert breakdown[ActionType.PLAY_CARD] == 1
        assert breakdown[ActionType.ATTACK] == 1

    def test_replay_metadata(self):
        """Replay can store metadata."""
        from simulation.replay import ReplayRecorder

        recorder = ReplayRecorder(seed=42)
        recorder.set_metadata('player1_agent', 'RandomAgent')
        recorder.set_metadata('player2_agent', 'GreedyAgent')
        recorder.set_metadata('date', '2024-01-01')

        assert recorder.metadata['player1_agent'] == 'RandomAgent'
        assert recorder.metadata['player2_agent'] == 'GreedyAgent'

    def test_replay_includes_final_outcome(self):
        """Replay records final game outcome."""
        from simulation.replay import ReplayRecorder

        recorder = ReplayRecorder(seed=42)
        recorder.record_action(0, Action(type=ActionType.END_TURN))

        recorder.set_outcome(winner=0, turns=50)
        assert recorder.outcome['winner'] == 0
        assert recorder.outcome['turns'] == 50


# ============================================================
# Edge Cases Tests
# ============================================================


class TestReplayEdgeCases:
    """Tests for replay edge cases."""

    def test_empty_replay(self):
        """Replay with no actions."""
        from simulation.replay import ReplayRecorder

        recorder = ReplayRecorder(seed=42)
        assert recorder.action_count() == 0
        assert len(recorder.actions) == 0

    def test_replay_with_invalid_actions(self):
        """Replay handles invalid action indices gracefully."""
        from simulation.replay import ReplayPlayer, ReplayRecorder

        recorder = ReplayRecorder(seed=42)
        recorder.record_action(0, Action(type=ActionType.END_TURN))

        player = ReplayPlayer(recorder)
        # Try to step beyond available actions
        player.step()
        result = player.step()  # Beyond end
        # Should handle gracefully (return False or None)
        assert result is None or result is False

    def test_replay_from_dict(self):
        """Replay can be reconstructed from dict."""
        from simulation.replay import ReplayRecorder

        original = ReplayRecorder(seed=42)
        original.record_action(0, Action(type=ActionType.END_TURN))

        replay_dict = original.to_dict()
        restored = ReplayRecorder.from_dict(replay_dict)

        assert restored.seed == original.seed
        assert len(restored.actions) == len(original.actions)

    def test_replay_with_large_action_sequence(self):
        """Replay handles large action sequences."""
        from simulation.replay import ReplayRecorder

        recorder = ReplayRecorder(seed=42)
        for i in range(1000):
            recorder.record_action(i % 2, Action(type=ActionType.END_TURN))

        assert recorder.action_count() == 1000

    def test_replay_string_representation(self):
        """Replay has readable string representation."""
        from simulation.replay import ReplayRecorder

        recorder = ReplayRecorder(seed=42)
        recorder.record_action(0, Action(type=ActionType.END_TURN))

        replay_str = str(recorder)
        assert "42" in replay_str  # Seed
        assert "1" in replay_str or "action" in replay_str.lower()
