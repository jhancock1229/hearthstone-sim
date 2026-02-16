"""Unit tests for Simulator - fast N-game rollout engine.

Following TDD: these tests define the contract for Simulator, which runs
multiple games between agents and collects statistics.

Simulator should:
- Run N games between two agents
- Collect game outcomes (wins, losses, ties)
- Track game statistics (turns, final health)
- Support parallel execution for speed
- Return comprehensive results
- Handle edge cases (agent crashes, infinite loops)
"""

import pytest
from agents.base import Agent
from agents.random_agent import RandomAgent
from simulation.observation import Observation
from simulation.action_space import Action, ActionType


# ============================================================
# Simulator Basic Tests
# ============================================================


class TestSimulatorBasic:
    """Tests for basic Simulator functionality."""

    def test_simulator_can_be_instantiated(self):
        """Simulator can be created."""
        from simulation.simulator import Simulator

        sim = Simulator()
        assert sim is not None

    def test_simulator_has_run_games_method(self):
        """Simulator has run_games method."""
        from simulation.simulator import Simulator

        sim = Simulator()
        assert hasattr(sim, 'run_games')
        assert callable(sim.run_games)

    def test_run_games_accepts_agents_and_count(self):
        """run_games accepts two agents and game count."""
        from simulation.simulator import Simulator

        sim = Simulator()
        agent1 = RandomAgent()
        agent2 = RandomAgent()

        # Should not raise
        result = sim.run_games(agent1, agent2, num_games=1)
        assert result is not None

    def test_run_games_returns_result_object(self):
        """run_games returns SimulationResult."""
        from simulation.simulator import Simulator, SimulationResult

        sim = Simulator()
        agent1 = RandomAgent()
        agent2 = RandomAgent()

        result = sim.run_games(agent1, agent2, num_games=5)
        assert isinstance(result, SimulationResult)

    def test_simulation_result_has_statistics(self):
        """SimulationResult contains game statistics."""
        from simulation.simulator import Simulator

        sim = Simulator()
        agent1 = RandomAgent()
        agent2 = RandomAgent()

        result = sim.run_games(agent1, agent2, num_games=10)
        assert hasattr(result, 'player1_wins')
        assert hasattr(result, 'player2_wins')
        assert hasattr(result, 'total_games')


# ============================================================
# Game Execution Tests
# ============================================================


class TestGameExecution:
    """Tests for game execution and outcome collection."""

    def test_simulator_runs_correct_number_of_games(self):
        """Simulator runs exactly N games."""
        from simulation.simulator import Simulator

        sim = Simulator()
        agent1 = RandomAgent(seed=42)
        agent2 = RandomAgent(seed=99)

        result = sim.run_games(agent1, agent2, num_games=15)
        assert result.total_games == 15

    def test_simulator_tracks_wins(self):
        """Simulator tracks wins for each player."""
        from simulation.simulator import Simulator

        sim = Simulator()
        agent1 = RandomAgent(seed=42)
        agent2 = RandomAgent(seed=99)

        result = sim.run_games(agent1, agent2, num_games=20)
        # Win counts should sum to total games
        assert result.player1_wins + result.player2_wins == result.total_games

    def test_simulator_calculates_win_rates(self):
        """Simulator calculates win rates."""
        from simulation.simulator import Simulator

        sim = Simulator()
        agent1 = RandomAgent(seed=42)
        agent2 = RandomAgent(seed=99)

        result = sim.run_games(agent1, agent2, num_games=50)
        assert 0.0 <= result.player1_win_rate <= 1.0
        assert 0.0 <= result.player2_win_rate <= 1.0
        assert abs(result.player1_win_rate + result.player2_win_rate - 1.0) < 0.01

    def test_simulator_tracks_game_lengths(self):
        """Simulator tracks number of turns per game."""
        from simulation.simulator import Simulator

        sim = Simulator()
        agent1 = RandomAgent(seed=42)
        agent2 = RandomAgent(seed=99)

        result = sim.run_games(agent1, agent2, num_games=10)
        assert hasattr(result, 'average_game_length')
        assert result.average_game_length > 0

    def test_simulator_resets_agents_between_games(self):
        """Agents are reset between games."""
        from simulation.simulator import Simulator

        sim = Simulator()

        class CountingAgent(Agent):
            def __init__(self):
                super().__init__()
                self.action_count = 0
                self.reset_count = 0

            def choose_action(self, observation: Observation) -> Action:
                self.action_count += 1
                return Action(type=ActionType.END_TURN)

            def reset(self) -> None:
                self.reset_count += 1
                self.action_count = 0

        agent1 = CountingAgent()
        agent2 = RandomAgent()

        result = sim.run_games(agent1, agent2, num_games=5)
        # Agent should be reset 5 times (once per game)
        assert agent1.reset_count == 5


# ============================================================
# Statistics Collection Tests
# ============================================================


class TestStatisticsCollection:
    """Tests for comprehensive statistics collection."""

    def test_result_contains_individual_game_outcomes(self):
        """Result includes list of individual game outcomes."""
        from simulation.simulator import Simulator

        sim = Simulator()
        agent1 = RandomAgent(seed=42)
        agent2 = RandomAgent(seed=99)

        result = sim.run_games(agent1, agent2, num_games=10)
        assert hasattr(result, 'games')
        assert len(result.games) == 10

    def test_game_outcome_includes_winner(self):
        """Each game outcome includes winner."""
        from simulation.simulator import Simulator

        sim = Simulator()
        agent1 = RandomAgent(seed=42)
        agent2 = RandomAgent(seed=99)

        result = sim.run_games(agent1, agent2, num_games=5)
        for game in result.games:
            assert hasattr(game, 'winner')
            assert game.winner in [0, 1]

    def test_game_outcome_includes_turns(self):
        """Each game outcome includes turn count."""
        from simulation.simulator import Simulator

        sim = Simulator()
        agent1 = RandomAgent(seed=42)
        agent2 = RandomAgent(seed=99)

        result = sim.run_games(agent1, agent2, num_games=5)
        for game in result.games:
            assert hasattr(game, 'turns')
            assert game.turns > 0

    def test_result_calculates_statistics(self):
        """Result calculates aggregate statistics."""
        from simulation.simulator import Simulator

        sim = Simulator()
        agent1 = RandomAgent(seed=42)
        agent2 = RandomAgent(seed=99)

        result = sim.run_games(agent1, agent2, num_games=30)
        assert result.player1_wins + result.player2_wins == 30
        assert result.average_game_length > 0
        assert 0.0 <= result.player1_win_rate <= 1.0


# ============================================================
# Configuration Tests
# ============================================================


class TestSimulatorConfiguration:
    """Tests for simulator configuration options."""

    def test_simulator_accepts_seed(self):
        """Simulator can be seeded for reproducibility."""
        from simulation.simulator import Simulator

        sim = Simulator(seed=42)
        assert sim is not None

    def test_seeded_simulator_is_deterministic(self):
        """Same seed produces same results."""
        from simulation.simulator import Simulator

        agent1_config = RandomAgent(seed=1)
        agent2_config = RandomAgent(seed=2)

        sim1 = Simulator(seed=42)
        result1 = sim1.run_games(agent1_config, agent2_config, num_games=20)

        agent1_config2 = RandomAgent(seed=1)
        agent2_config2 = RandomAgent(seed=2)

        sim2 = Simulator(seed=42)
        result2 = sim2.run_games(agent1_config2, agent2_config2, num_games=20)

        # Results should be identical
        assert result1.player1_wins == result2.player1_wins
        assert result1.player2_wins == result2.player2_wins

    def test_simulator_accepts_max_turns_limit(self):
        """Simulator can limit max turns per game."""
        from simulation.simulator import Simulator

        sim = Simulator(max_turns=50)
        agent1 = RandomAgent()
        agent2 = RandomAgent()

        result = sim.run_games(agent1, agent2, num_games=10)
        # All games should end within max_turns
        for game in result.games:
            assert game.turns <= 50

    def test_simulator_handles_verbose_mode(self):
        """Simulator supports verbose output."""
        from simulation.simulator import Simulator

        sim = Simulator(verbose=True)
        agent1 = RandomAgent()
        agent2 = RandomAgent()

        # Should not crash
        result = sim.run_games(agent1, agent2, num_games=2)
        assert result is not None


# ============================================================
# Edge Cases Tests
# ============================================================


class TestSimulatorEdgeCases:
    """Tests for edge cases and error handling."""

    def test_simulator_handles_single_game(self):
        """Simulator can run just one game."""
        from simulation.simulator import Simulator

        sim = Simulator()
        agent1 = RandomAgent()
        agent2 = RandomAgent()

        result = sim.run_games(agent1, agent2, num_games=1)
        assert result.total_games == 1
        assert len(result.games) == 1

    def test_simulator_handles_many_games(self):
        """Simulator can run many games."""
        from simulation.simulator import Simulator

        sim = Simulator()
        agent1 = RandomAgent(seed=42)
        agent2 = RandomAgent(seed=99)

        result = sim.run_games(agent1, agent2, num_games=100)
        assert result.total_games == 100
        assert len(result.games) == 100

    def test_simulator_with_same_agent_instances(self):
        """Simulator can run with same agent for both players."""
        from simulation.simulator import Simulator

        sim = Simulator()
        agent = RandomAgent(seed=42)

        # Using same agent instance for both players
        result = sim.run_games(agent, agent, num_games=10)
        assert result.total_games == 10

    def test_result_string_representation(self):
        """SimulationResult has readable string representation."""
        from simulation.simulator import Simulator

        sim = Simulator()
        agent1 = RandomAgent()
        agent2 = RandomAgent()

        result = sim.run_games(agent1, agent2, num_games=10)
        result_str = str(result)
        assert "wins" in result_str.lower()
        assert "10" in result_str or "games" in result_str.lower()

    def test_result_to_dict(self):
        """SimulationResult can be serialized to dict."""
        from simulation.simulator import Simulator

        sim = Simulator()
        agent1 = RandomAgent()
        agent2 = RandomAgent()

        result = sim.run_games(agent1, agent2, num_games=5)
        result_dict = result.to_dict()

        assert isinstance(result_dict, dict)
        assert 'player1_wins' in result_dict
        assert 'player2_wins' in result_dict
        assert 'total_games' in result_dict


# ============================================================
# Performance Tests
# ============================================================


class TestSimulatorPerformance:
    """Tests for simulator performance features."""

    def test_simulator_completes_in_reasonable_time(self):
        """Simulator completes games in reasonable time."""
        from simulation.simulator import Simulator
        import time

        sim = Simulator()
        agent1 = RandomAgent(seed=42)
        agent2 = RandomAgent(seed=99)

        start = time.time()
        result = sim.run_games(agent1, agent2, num_games=10)
        elapsed = time.time() - start

        # Should complete 10 games in under 10 seconds
        assert elapsed < 10.0
        assert result.total_games == 10

    def test_simulator_progress_tracking(self):
        """Simulator can track progress during execution."""
        from simulation.simulator import Simulator

        progress_calls = []

        def progress_callback(completed, total):
            progress_calls.append((completed, total))

        sim = Simulator(progress_callback=progress_callback)
        agent1 = RandomAgent()
        agent2 = RandomAgent()

        result = sim.run_games(agent1, agent2, num_games=5)
        # Progress should be tracked
        assert len(progress_calls) > 0


# ============================================================
# Integration Tests
# ============================================================


class TestSimulatorIntegration:
    """Integration tests for Simulator with real game engine."""

    def test_simulator_uses_real_game_engine(self):
        """Simulator actually runs games using the game engine."""
        from simulation.simulator import Simulator

        sim = Simulator()
        agent1 = RandomAgent(seed=42)
        agent2 = RandomAgent(seed=99)

        result = sim.run_games(agent1, agent2, num_games=5)
        # Games should have reasonable turn counts
        assert all(1 <= game.turns <= 100 for game in result.games)

    def test_simulator_with_different_agent_types(self):
        """Simulator works with different agent types."""
        from simulation.simulator import Simulator

        class AlwaysEndTurn(Agent):
            def choose_action(self, observation: Observation) -> Action:
                return Action(type=ActionType.END_TURN)

        sim = Simulator()
        agent1 = AlwaysEndTurn()
        agent2 = RandomAgent()

        result = sim.run_games(agent1, agent2, num_games=3)
        assert result.total_games == 3


# ============================================================
# Custom Deck Support Tests
# ============================================================


class TestSimulatorCustomDecks:
    """Tests for running simulations with specific deck genotypes."""

    def test_custom_genotypes_accepted(self):
        """run_games accepts custom deck genotypes."""
        from simulation.simulator import Simulator
        from deckbuilding.deck import CARD_POOL

        sim = Simulator(seed=42)
        a1 = RandomAgent(seed=1)
        a2 = RandomAgent(seed=2)
        genotype = ["Brute"] * 15 + ["Wasp"] * 15
        result = sim.run_games(a1, a2, num_games=2,
                               deck1_genotype=genotype, deck2_genotype=genotype)
        assert result.total_games == 2

    def test_one_custom_one_random(self):
        """Can provide one custom genotype and leave the other random."""
        from simulation.simulator import Simulator

        sim = Simulator(seed=42)
        a1 = RandomAgent(seed=1)
        a2 = RandomAgent(seed=2)
        genotype = ["Adept"] * 30
        result = sim.run_games(a1, a2, num_games=2, deck1_genotype=genotype)
        assert result.total_games == 2

    def test_custom_pool_parameter(self):
        """run_games accepts a custom card pool."""
        from simulation.simulator import Simulator
        from deckbuilding.deck import CardSpec

        custom_pool = {
            "BigGuy": CardSpec("BigGuy", 5, 6, 6),
            "SmallGuy": CardSpec("SmallGuy", 1, 1, 1),
        }
        genotype = ["BigGuy"] * 15 + ["SmallGuy"] * 15
        sim = Simulator(seed=42)
        a1 = RandomAgent(seed=1)
        a2 = RandomAgent(seed=2)
        result = sim.run_games(a1, a2, num_games=2,
                               deck1_genotype=genotype, deck2_genotype=genotype,
                               pool=custom_pool)
        assert result.total_games == 2

    def test_no_args_still_works(self):
        """Calling without deck args still works (backward compat)."""
        from simulation.simulator import Simulator

        sim = Simulator(seed=42)
        a1 = RandomAgent(seed=1)
        a2 = RandomAgent(seed=2)
        result = sim.run_games(a1, a2, num_games=2)
        assert result.total_games == 2

    def test_registry_pool_works(self):
        """Can use a registry-derived pool for custom decks."""
        from simulation.simulator import Simulator
        from hearthstone.cards.registry import CardRegistry
        from deckbuilding.deck import build_pool_from_registry, random_deck
        from pathlib import Path

        registry = CardRegistry.from_json(Path("hearthstone/data/cards_collectible.json"))
        pool = build_pool_from_registry(registry)
        genotype = random_deck(pool, size=30)
        sim = Simulator(seed=42)
        a1 = RandomAgent(seed=1)
        a2 = RandomAgent(seed=2)
        result = sim.run_games(a1, a2, num_games=2,
                               deck1_genotype=genotype, deck2_genotype=genotype,
                               pool=pool)
        assert result.total_games == 2
        assert result.player1_wins + result.player2_wins == 2
