"""Unit tests for base Agent interface.

Following TDD: these tests define the contract for all agents in the system.
All agents must implement choose_action(observation) -> Action.

The base agent interface should:
- Define abstract choose_action method
- Accept Observation as input
- Return Action as output
- Support agent initialization with optional configuration
- Allow agents to maintain internal state
- Provide mechanism for agent reset between games
"""

import pytest
from abc import ABC, abstractmethod
from typing import Any, Optional, Dict
from simulation.observation import Observation
from simulation.action_space import Action, ActionType


# ============================================================
# Base Agent Contract Tests
# ============================================================


class TestBaseAgentContract:
    """Tests for the abstract Agent base class."""

    def test_agent_is_abstract(self):
        """Agent base class cannot be instantiated directly."""
        from agents.base import Agent

        with pytest.raises(TypeError):
            Agent()

    def test_agent_has_choose_action_method(self):
        """Agent must define choose_action method."""
        from agents.base import Agent

        assert hasattr(Agent, 'choose_action')

    def test_choose_action_is_abstract(self):
        """choose_action must be abstract (enforced by ABC)."""
        from agents.base import Agent

        # Creating subclass without implementing choose_action should fail
        with pytest.raises(TypeError):
            class IncompleteAgent(Agent):
                pass
            IncompleteAgent()

    def test_agent_subclass_must_implement_choose_action(self):
        """Concrete agent must implement choose_action."""
        from agents.base import Agent

        class ConcreteAgent(Agent):
            def choose_action(self, observation: Observation) -> Action:
                return Action(type=ActionType.END_TURN)

        agent = ConcreteAgent()
        assert callable(agent.choose_action)

    def test_choose_action_accepts_observation(self):
        """choose_action accepts Observation parameter."""
        from agents.base import Agent

        class TestAgent(Agent):
            def choose_action(self, observation: Observation) -> Action:
                return Action(type=ActionType.END_TURN)

        agent = TestAgent()
        obs = Observation(
            self_health=30,
            self_mana=5,
            self_max_mana=10,
            self_fatigue_counter=0,
            opponent_health=30,
            opponent_mana=5
        )
        action = agent.choose_action(obs)
        assert isinstance(action, Action)

    def test_choose_action_returns_action(self):
        """choose_action returns Action object."""
        from agents.base import Agent

        class TestAgent(Agent):
            def choose_action(self, observation: Observation) -> Action:
                return Action(type=ActionType.END_TURN)

        agent = TestAgent()
        obs = Observation(
            self_health=30,
            self_mana=5,
            self_max_mana=10,
            self_fatigue_counter=0,
            opponent_health=30,
            opponent_mana=5
        )
        result = agent.choose_action(obs)
        assert isinstance(result, Action)


# ============================================================
# Agent Initialization Tests
# ============================================================


class TestAgentInitialization:
    """Tests for agent initialization and configuration."""

    def test_agent_can_be_initialized_without_config(self):
        """Agents can be created with default configuration."""
        from agents.base import Agent

        class SimpleAgent(Agent):
            def choose_action(self, observation: Observation) -> Action:
                return Action(type=ActionType.END_TURN)

        agent = SimpleAgent()
        assert agent is not None

    def test_agent_can_accept_optional_config(self):
        """Agents can accept configuration parameters."""
        from agents.base import Agent

        class ConfigurableAgent(Agent):
            def __init__(self, name: Optional[str] = None):
                self.name = name or "DefaultAgent"

            def choose_action(self, observation: Observation) -> Action:
                return Action(type=ActionType.END_TURN)

        agent = ConfigurableAgent(name="TestAgent")
        assert agent.name == "TestAgent"

    def test_agent_has_optional_name_property(self):
        """Agents can have a name for identification."""
        from agents.base import Agent

        class NamedAgent(Agent):
            def __init__(self, name: str = "Agent"):
                self.name = name

            def choose_action(self, observation: Observation) -> Action:
                return Action(type=ActionType.END_TURN)

        agent = NamedAgent("MyAgent")
        assert hasattr(agent, 'name')
        assert agent.name == "MyAgent"

    def test_agent_can_store_configuration(self):
        """Agents can store and access configuration."""
        from agents.base import Agent

        class ConfigAgent(Agent):
            def __init__(self, config: Optional[Dict[str, Any]] = None):
                self.config = config or {}

            def choose_action(self, observation: Observation) -> Action:
                return Action(type=ActionType.END_TURN)

        config = {"temperature": 0.8, "depth": 5}
        agent = ConfigAgent(config=config)
        assert agent.config["temperature"] == 0.8
        assert agent.config["depth"] == 5


# ============================================================
# Agent State Management Tests
# ============================================================


class TestAgentStateManagement:
    """Tests for agent state and reset functionality."""

    def test_agent_can_maintain_internal_state(self):
        """Agents can maintain state between action calls."""
        from agents.base import Agent

        class StatefulAgent(Agent):
            def __init__(self):
                self.action_count = 0

            def choose_action(self, observation: Observation) -> Action:
                self.action_count += 1
                return Action(type=ActionType.END_TURN)

        agent = StatefulAgent()
        obs = Observation(
            self_health=30,
            self_mana=5,
            self_max_mana=10,
            self_fatigue_counter=0,
            opponent_health=30,
            opponent_mana=5
        )

        agent.choose_action(obs)
        agent.choose_action(obs)
        assert agent.action_count == 2

    def test_agent_has_reset_method(self):
        """Agents have reset() method for state cleanup."""
        from agents.base import Agent

        class ResettableAgent(Agent):
            def __init__(self):
                self.action_count = 0

            def choose_action(self, observation: Observation) -> Action:
                self.action_count += 1
                return Action(type=ActionType.END_TURN)

            def reset(self) -> None:
                self.action_count = 0

        agent = ResettableAgent()
        obs = Observation(
            self_health=30,
            self_mana=5,
            self_max_mana=10,
            self_fatigue_counter=0,
            opponent_health=30,
            opponent_mana=5
        )

        agent.choose_action(obs)
        agent.choose_action(obs)
        agent.reset()
        assert agent.action_count == 0

    def test_reset_prepares_agent_for_new_game(self):
        """reset() clears game-specific state."""
        from agents.base import Agent

        class GameAgent(Agent):
            def __init__(self):
                self.game_history = []

            def choose_action(self, observation: Observation) -> Action:
                self.game_history.append(observation)
                return Action(type=ActionType.END_TURN)

            def reset(self) -> None:
                self.game_history.clear()

        agent = GameAgent()
        obs = Observation(
            self_health=30,
            self_mana=5,
            self_max_mana=10,
            self_fatigue_counter=0,
            opponent_health=30,
            opponent_mana=5
        )

        agent.choose_action(obs)
        agent.choose_action(obs)
        assert len(agent.game_history) == 2

        agent.reset()
        assert len(agent.game_history) == 0


# ============================================================
# Agent Behavior Tests
# ============================================================


class TestAgentBehavior:
    """Tests for agent decision-making behavior."""

    def test_agent_receives_full_observation(self):
        """Agent receives complete observation with all visible state."""
        from agents.base import Agent

        received_obs = None

        class InspectAgent(Agent):
            def choose_action(self, observation: Observation) -> Action:
                nonlocal received_obs
                received_obs = observation
                return Action(type=ActionType.END_TURN)

        agent = InspectAgent()
        obs = Observation(
            self_health=25,
            self_mana=7,
            self_max_mana=10,
            self_fatigue_counter=0,
            opponent_health=18,
            opponent_mana=5,
            opponent_hand_size=4
        )

        agent.choose_action(obs)
        assert received_obs == obs
        assert received_obs.self_health == 25
        assert received_obs.opponent_hand_size == 4

    def test_agent_can_return_different_action_types(self):
        """Agent can return various action types."""
        from agents.base import Agent

        class FlexibleAgent(Agent):
            def __init__(self):
                self.next_action = Action(type=ActionType.END_TURN)

            def choose_action(self, observation: Observation) -> Action:
                return self.next_action

        agent = FlexibleAgent()
        obs = Observation(
            self_health=30,
            self_mana=5,
            self_max_mana=10,
            self_fatigue_counter=0,
            opponent_health=30,
            opponent_mana=5
        )

        # Test END_TURN
        agent.next_action = Action(type=ActionType.END_TURN)
        action = agent.choose_action(obs)
        assert action.type == ActionType.END_TURN

        # Test PLAY_CARD
        agent.next_action = Action(type=ActionType.PLAY_CARD, card_index=0)
        action = agent.choose_action(obs)
        assert action.type == ActionType.PLAY_CARD
        assert action.card_index == 0

    def test_agent_decisions_based_on_observation(self):
        """Agent can make decisions based on observation data."""
        from agents.base import Agent

        class ConditionalAgent(Agent):
            def choose_action(self, observation: Observation) -> Action:
                if observation.self_health < 15:
                    # Low health - prioritize hero power
                    return Action(type=ActionType.HERO_POWER)
                else:
                    return Action(type=ActionType.END_TURN)

        agent = ConditionalAgent()

        # High health
        obs_high = Observation(
            self_health=28,
            self_mana=5,
            self_max_mana=10,
            self_fatigue_counter=0,
            opponent_health=30,
            opponent_mana=5
        )
        action = agent.choose_action(obs_high)
        assert action.type == ActionType.END_TURN

        # Low health
        obs_low = Observation(
            self_health=10,
            self_mana=5,
            self_max_mana=10,
            self_fatigue_counter=0,
            opponent_health=30,
            opponent_mana=5
        )
        action = agent.choose_action(obs_low)
        assert action.type == ActionType.HERO_POWER


# ============================================================
# Agent Interface Compliance Tests
# ============================================================


class TestAgentInterfaceCompliance:
    """Tests ensuring agents follow the interface contract."""

    def test_agent_method_signature_is_correct(self):
        """choose_action has correct signature."""
        from agents.base import Agent
        import inspect

        # Get the abstract method signature
        sig = inspect.signature(Agent.choose_action)
        params = list(sig.parameters.keys())

        assert 'observation' in params

    def test_multiple_agents_can_coexist(self):
        """Multiple agent instances can exist independently."""
        from agents.base import Agent

        class Agent1(Agent):
            def __init__(self):
                self.id = 1

            def choose_action(self, observation: Observation) -> Action:
                return Action(type=ActionType.END_TURN)

        class Agent2(Agent):
            def __init__(self):
                self.id = 2

            def choose_action(self, observation: Observation) -> Action:
                return Action(type=ActionType.HERO_POWER)

        a1 = Agent1()
        a2 = Agent2()

        assert a1.id == 1
        assert a2.id == 2

    def test_agent_is_reusable_across_games(self):
        """Same agent instance can be used for multiple games."""
        from agents.base import Agent

        class ReusableAgent(Agent):
            def __init__(self):
                self.games_played = 0

            def choose_action(self, observation: Observation) -> Action:
                return Action(type=ActionType.END_TURN)

            def reset(self) -> None:
                self.games_played += 1

        agent = ReusableAgent()
        obs = Observation(
            self_health=30,
            self_mana=5,
            self_max_mana=10,
            self_fatigue_counter=0,
            opponent_health=30,
            opponent_mana=5
        )

        # Play multiple "games"
        for _ in range(3):
            agent.choose_action(obs)
            agent.reset()

        assert agent.games_played == 3

    def test_agent_type_hints_are_enforced(self):
        """Agent uses proper type hints."""
        from agents.base import Agent
        import inspect

        # Check that choose_action has type hints
        sig = inspect.signature(Agent.choose_action)
        assert sig.return_annotation != inspect.Parameter.empty
