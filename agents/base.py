"""Abstract Agent interface for Hearthstone simulator.

All agents must implement the choose_action method which takes an
Observation and returns an Action. This contract enables:
- Uniform agent interface for simulation
- Agent swapping and comparison
- Consistent state management across games

Agents can:
- Maintain internal state (e.g., statistics, learning parameters)
- Be configured with optional parameters
- Reset state between games via reset()
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from simulation.observation import Observation
from simulation.action_space import Action


class Agent(ABC):
    """Abstract base class for all game agents.

    All agents must implement choose_action to decide which action
    to take given the current game observation.
    """

    def __init__(self, name: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        """Initialize the agent.

        Args:
            name: Optional name for the agent (for logging/identification)
            config: Optional configuration dictionary
        """
        self.name = name or self.__class__.__name__
        self.config = config or {}

    @abstractmethod
    def choose_action(self, observation: Observation) -> Action:
        """Choose an action based on the current observation.

        Args:
            observation: The current game state from this agent's perspective

        Returns:
            The action to take
        """
        raise NotImplementedError("Subclasses must implement choose_action")

    def reset(self) -> None:
        """Reset agent state between games.

        Override this method if your agent maintains game-specific state
        that should be cleared between games.
        """
        pass
