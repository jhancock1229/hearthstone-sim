"""RandomAgent - baseline agent that picks uniformly random legal actions.

Essential for:
- Engine sanity checking
- Baseline performance metrics
- Training opponent for ML agents
- Testing simulation infrastructure

The RandomAgent:
- Selects actions uniformly at random from legal actions
- Is stateless (no learning or adaptation)
- Can be seeded for reproducible behavior
- Always returns a valid legal action
"""

import random
from typing import Optional
from agents.base import Agent
from simulation.observation import Observation
from simulation.action_space import Action, ActionType


class RandomAgent(Agent):
    """Agent that selects uniformly random legal actions.

    This is the simplest possible agent and serves as a baseline
    for comparing more sophisticated strategies.
    """

    def __init__(self, seed: Optional[int] = None, name: Optional[str] = None):
        """Initialize the RandomAgent.

        Args:
            seed: Optional random seed for reproducibility
            name: Optional name for the agent
        """
        super().__init__(name=name or "RandomAgent")
        self.seed = seed
        self.rng = random.Random(seed)

    def choose_action(self, observation: Observation) -> Action:
        """Choose a uniformly random action from legal actions.

        Args:
            observation: Current game state observation

        Returns:
            Randomly selected legal action

        Raises:
            ValueError: If no legal actions available
        """
        if not observation.legal_actions:
            # Edge case: no legal actions available
            # This shouldn't happen in a valid game, but handle gracefully
            return Action(type=ActionType.END_TURN)

        # Select uniformly at random
        return self.rng.choice(observation.legal_actions)

    def reset(self) -> None:
        """Reset agent state (no-op for stateless RandomAgent)."""
        # RandomAgent is stateless, so reset doesn't need to do anything
        # But we keep the method for interface compliance
        pass
