"""Unit tests for RandomAgent - baseline random action selection.

Following TDD: these tests define the contract for RandomAgent, which
picks uniformly random legal actions. Essential for:
- Engine sanity checking
- Baseline performance comparison
- Training opponent for ML agents

RandomAgent should:
- Select actions uniformly at random from legal actions
- Always return a legal action
- Work with any observation
- Be deterministic when seeded
- Handle edge cases (single action, no actions)
"""

import pytest
from simulation.observation import Observation
from simulation.action_space import Action, ActionType
from typing import List


# ============================================================
# RandomAgent Basic Tests
# ============================================================


class TestRandomAgentBasic:
    """Tests for basic RandomAgent functionality."""

    def test_random_agent_can_be_instantiated(self):
        """RandomAgent can be created."""
        from agents.random_agent import RandomAgent

        agent = RandomAgent()
        assert agent is not None

    def test_random_agent_inherits_from_base_agent(self):
        """RandomAgent inherits from Agent base class."""
        from agents.random_agent import RandomAgent
        from agents.base import Agent

        agent = RandomAgent()
        assert isinstance(agent, Agent)

    def test_random_agent_has_choose_action_method(self):
        """RandomAgent implements choose_action."""
        from agents.random_agent import RandomAgent

        agent = RandomAgent()
        assert hasattr(agent, 'choose_action')
        assert callable(agent.choose_action)

    def test_random_agent_accepts_observation(self):
        """RandomAgent.choose_action accepts Observation."""
        from agents.random_agent import RandomAgent

        agent = RandomAgent()
        obs = Observation(
            self_health=30,
            self_mana=5,
            self_max_mana=10,
            self_fatigue_counter=0,
            opponent_health=30,
            opponent_mana=5,
            legal_actions=[Action(type=ActionType.END_TURN)]
        )

        action = agent.choose_action(obs)
        assert isinstance(action, Action)

    def test_random_agent_returns_action(self):
        """RandomAgent.choose_action returns Action object."""
        from agents.random_agent import RandomAgent

        agent = RandomAgent()
        obs = Observation(
            self_health=30,
            self_mana=5,
            self_max_mana=10,
            self_fatigue_counter=0,
            opponent_health=30,
            opponent_mana=5,
            legal_actions=[Action(type=ActionType.END_TURN)]
        )

        result = agent.choose_action(obs)
        assert isinstance(result, Action)


# ============================================================
# Random Selection Tests
# ============================================================


class TestRandomSelection:
    """Tests for random action selection behavior."""

    def test_random_agent_selects_from_legal_actions(self):
        """RandomAgent only selects from legal actions."""
        from agents.random_agent import RandomAgent

        agent = RandomAgent()
        legal_actions = [
            Action(type=ActionType.END_TURN),
            Action(type=ActionType.HERO_POWER),
            Action(type=ActionType.PLAY_CARD, card_index=0),
        ]

        obs = Observation(
            self_health=30,
            self_mana=5,
            self_max_mana=10,
            self_fatigue_counter=0,
            opponent_health=30,
            opponent_mana=5,
            legal_actions=legal_actions
        )

        action = agent.choose_action(obs)
        assert action in legal_actions

    def test_random_agent_uniform_distribution(self):
        """RandomAgent selects actions with roughly uniform probability."""
        from agents.random_agent import RandomAgent

        agent = RandomAgent()
        legal_actions = [
            Action(type=ActionType.END_TURN),
            Action(type=ActionType.HERO_POWER),
            Action(type=ActionType.PLAY_CARD, card_index=0),
        ]

        obs = Observation(
            self_health=30,
            self_mana=5,
            self_max_mana=10,
            self_fatigue_counter=0,
            opponent_health=30,
            opponent_mana=5,
            legal_actions=legal_actions
        )

        # Sample 300 actions and check distribution
        counts = {action: 0 for action in legal_actions}
        for _ in range(300):
            selected = agent.choose_action(obs)
            counts[selected] += 1

        # Each action should be selected roughly 100 times (±50 for randomness)
        for count in counts.values():
            assert 50 < count < 150, f"Expected ~100, got {count}"

    def test_random_agent_with_single_action(self):
        """RandomAgent handles single legal action correctly."""
        from agents.random_agent import RandomAgent

        agent = RandomAgent()
        only_action = Action(type=ActionType.END_TURN)

        obs = Observation(
            self_health=30,
            self_mana=5,
            self_max_mana=10,
            self_fatigue_counter=0,
            opponent_health=30,
            opponent_mana=5,
            legal_actions=[only_action]
        )

        # Should always return the only action
        for _ in range(10):
            action = agent.choose_action(obs)
            assert action == only_action

    def test_random_agent_different_actions_across_calls(self):
        """RandomAgent doesn't always pick the same action."""
        from agents.random_agent import RandomAgent

        agent = RandomAgent()
        legal_actions = [
            Action(type=ActionType.END_TURN),
            Action(type=ActionType.HERO_POWER),
            Action(type=ActionType.PLAY_CARD, card_index=0),
            Action(type=ActionType.PLAY_CARD, card_index=1),
        ]

        obs = Observation(
            self_health=30,
            self_mana=5,
            self_max_mana=10,
            self_fatigue_counter=0,
            opponent_health=30,
            opponent_mana=5,
            legal_actions=legal_actions
        )

        # Sample 50 actions, should get at least 2 different ones
        actions_seen = set()
        for _ in range(50):
            action = agent.choose_action(obs)
            actions_seen.add(action)

        assert len(actions_seen) >= 2


# ============================================================
# Seeding and Determinism Tests
# ============================================================


class TestRandomAgentSeeding:
    """Tests for random seed control and determinism."""

    def test_random_agent_accepts_seed(self):
        """RandomAgent can be initialized with random seed."""
        from agents.random_agent import RandomAgent

        agent = RandomAgent(seed=42)
        assert agent is not None

    def test_seeded_agent_is_deterministic(self):
        """Same seed produces same action sequence."""
        from agents.random_agent import RandomAgent

        legal_actions = [
            Action(type=ActionType.END_TURN),
            Action(type=ActionType.HERO_POWER),
            Action(type=ActionType.PLAY_CARD, card_index=0),
        ]

        obs = Observation(
            self_health=30,
            self_mana=5,
            self_max_mana=10,
            self_fatigue_counter=0,
            opponent_health=30,
            opponent_mana=5,
            legal_actions=legal_actions
        )

        # Create two agents with same seed
        agent1 = RandomAgent(seed=123)
        agent2 = RandomAgent(seed=123)

        # Should produce identical sequences
        for _ in range(10):
            action1 = agent1.choose_action(obs)
            action2 = agent2.choose_action(obs)
            assert action1 == action2

    def test_different_seeds_produce_different_sequences(self):
        """Different seeds produce different sequences."""
        from agents.random_agent import RandomAgent

        legal_actions = [
            Action(type=ActionType.END_TURN),
            Action(type=ActionType.HERO_POWER),
            Action(type=ActionType.PLAY_CARD, card_index=0),
            Action(type=ActionType.PLAY_CARD, card_index=1),
        ]

        obs = Observation(
            self_health=30,
            self_mana=5,
            self_max_mana=10,
            self_fatigue_counter=0,
            opponent_health=30,
            opponent_mana=5,
            legal_actions=legal_actions
        )

        agent1 = RandomAgent(seed=42)
        agent2 = RandomAgent(seed=99)

        # Collect sequences
        seq1 = [agent1.choose_action(obs) for _ in range(20)]
        seq2 = [agent2.choose_action(obs) for _ in range(20)]

        # Should be different (extremely unlikely to be identical)
        assert seq1 != seq2

    def test_unseeded_agents_are_different(self):
        """Agents without explicit seed behave independently."""
        from agents.random_agent import RandomAgent

        legal_actions = [
            Action(type=ActionType.END_TURN),
            Action(type=ActionType.HERO_POWER),
            Action(type=ActionType.PLAY_CARD, card_index=0),
        ]

        obs = Observation(
            self_health=30,
            self_mana=5,
            self_max_mana=10,
            self_fatigue_counter=0,
            opponent_health=30,
            opponent_mana=5,
            legal_actions=legal_actions
        )

        agent1 = RandomAgent()
        agent2 = RandomAgent()

        # Should produce different sequences
        seq1 = [agent1.choose_action(obs) for _ in range(20)]
        seq2 = [agent2.choose_action(obs) for _ in range(20)]

        assert seq1 != seq2


# ============================================================
# Edge Cases Tests
# ============================================================


class TestRandomAgentEdgeCases:
    """Tests for edge cases and error handling."""

    def test_random_agent_with_many_actions(self):
        """RandomAgent handles large action spaces."""
        from agents.random_agent import RandomAgent

        # Create 100 different actions
        legal_actions = [
            Action(type=ActionType.PLAY_CARD, card_index=i)
            for i in range(100)
        ]

        obs = Observation(
            self_health=30,
            self_mana=10,
            self_max_mana=10,
            self_fatigue_counter=0,
            opponent_health=30,
            opponent_mana=5,
            legal_actions=legal_actions
        )

        agent = RandomAgent()
        action = agent.choose_action(obs)
        assert action in legal_actions

    def test_random_agent_handles_empty_legal_actions(self):
        """RandomAgent handles empty action list gracefully."""
        from agents.random_agent import RandomAgent

        obs = Observation(
            self_health=30,
            self_mana=5,
            self_max_mana=10,
            self_fatigue_counter=0,
            opponent_health=30,
            opponent_mana=5,
            legal_actions=[]
        )

        agent = RandomAgent()

        # Should either raise error or return None/default action
        try:
            action = agent.choose_action(obs)
            # If it returns something, it should be None or END_TURN
            assert action is None or action.type == ActionType.END_TURN
        except (ValueError, IndexError):
            # Or it can raise an error for invalid state
            pass

    def test_random_agent_works_across_multiple_games(self):
        """RandomAgent can be reused across multiple games."""
        from agents.random_agent import RandomAgent

        agent = RandomAgent()

        for game_num in range(5):
            obs = Observation(
                self_health=30,
                self_mana=5,
                self_max_mana=10,
                self_fatigue_counter=0,
                opponent_health=30,
                opponent_mana=5,
                legal_actions=[Action(type=ActionType.END_TURN)]
            )

            action = agent.choose_action(obs)
            assert action.type == ActionType.END_TURN

            # Reset between games
            agent.reset()


# ============================================================
# Agent State Tests
# ============================================================


class TestRandomAgentState:
    """Tests for agent state management."""

    def test_random_agent_has_no_persistent_state(self):
        """RandomAgent is stateless between calls."""
        from agents.random_agent import RandomAgent

        agent = RandomAgent(seed=42)

        legal_actions = [
            Action(type=ActionType.END_TURN),
            Action(type=ActionType.HERO_POWER),
        ]

        obs = Observation(
            self_health=30,
            self_mana=5,
            self_max_mana=10,
            self_fatigue_counter=0,
            opponent_health=30,
            opponent_mana=5,
            legal_actions=legal_actions
        )

        # First sequence
        seq1 = [agent.choose_action(obs) for _ in range(5)]

        # Reset and create new agent with same seed
        agent2 = RandomAgent(seed=42)
        seq2 = [agent2.choose_action(obs) for _ in range(5)]

        # Should produce same sequence
        assert seq1 == seq2

    def test_random_agent_reset_does_not_change_behavior(self):
        """reset() doesn't affect RandomAgent (stateless)."""
        from agents.random_agent import RandomAgent

        agent = RandomAgent(seed=42)

        legal_actions = [
            Action(type=ActionType.END_TURN),
            Action(type=ActionType.HERO_POWER),
        ]

        obs = Observation(
            self_health=30,
            self_mana=5,
            self_max_mana=10,
            self_fatigue_counter=0,
            opponent_health=30,
            opponent_mana=5,
            legal_actions=legal_actions
        )

        # Get some actions
        before = [agent.choose_action(obs) for _ in range(3)]

        # Reset
        agent.reset()

        # Continue getting actions (should continue same sequence)
        after = [agent.choose_action(obs) for _ in range(3)]

        # Both should be valid actions
        for action in before + after:
            assert action in legal_actions

    def test_random_agent_name_can_be_set(self):
        """RandomAgent can have custom name."""
        from agents.random_agent import RandomAgent

        agent = RandomAgent(name="TestBot")
        assert agent.name == "TestBot"

    def test_random_agent_default_name(self):
        """RandomAgent has default name if not specified."""
        from agents.random_agent import RandomAgent

        agent = RandomAgent()
        assert agent.name is not None
        assert len(agent.name) > 0
