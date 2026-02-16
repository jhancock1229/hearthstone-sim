"""GreedyAgent - heuristic-based agent using greedy value maximization.

The GreedyAgent makes decisions by:
1. Evaluating each legal action with a heuristic value function
2. Selecting the action with the highest immediate value
3. Prioritizing: cards > attacks > hero power > end turn

Heuristics consider:
- Mana efficiency (stats per mana cost)
- Board control (favorable trades)
- Tempo (playing cards early)
- Value trading (killing enemies while surviving)

This agent serves as:
- A strong baseline for comparison
- An opponent for training RL agents
- A sanity check that heuristics work
"""

from typing import List, Optional
from agents.base import Agent
from simulation.observation import Observation
from simulation.action_space import Action, ActionType


class GreedyAgent(Agent):
    """Agent that selects actions by maximizing a greedy heuristic value function.

    The agent evaluates each legal action and picks the one with the highest
    immediate value, without considering future game states.
    """

    def __init__(self, name: Optional[str] = None):
        """Initialize the GreedyAgent.

        Args:
            name: Optional name for the agent
        """
        super().__init__(name=name or "GreedyAgent")

    def choose_action(self, observation: Observation) -> Action:
        """Choose the action with highest greedy value.

        Args:
            observation: Current game state observation

        Returns:
            Action with highest heuristic value

        Strategy:
        1. Evaluate each legal action with value function
        2. Return action with maximum value
        3. Break ties consistently (prefer earlier actions)
        """
        if not observation.legal_actions:
            # Fallback: no legal actions (shouldn't happen)
            return Action(type=ActionType.END_TURN)

        # Evaluate all actions and pick best
        best_action = None
        best_value = float('-inf')

        for action in observation.legal_actions:
            value = self._evaluate_action(action, observation)
            if value > best_value:
                best_value = value
                best_action = action

        return best_action

    def _evaluate_action(self, action: Action, observation: Observation) -> float:
        """Evaluate the value of an action.

        Args:
            action: Action to evaluate
            observation: Current game state

        Returns:
            Heuristic value score (higher is better)
        """
        if action.type == ActionType.PLAY_CARD:
            return self._evaluate_play_card(action, observation)
        elif action.type == ActionType.ATTACK:
            return self._evaluate_attack(action, observation)
        elif action.type == ActionType.HERO_POWER:
            return self._evaluate_hero_power(action, observation)
        elif action.type == ActionType.END_TURN:
            return self._evaluate_end_turn(action, observation)
        else:
            return 0.0

    def _evaluate_play_card(self, action: Action, observation: Observation) -> float:
        """Evaluate playing a card.

        High priority - we want to build board presence.

        Factors:
        - Mana efficiency (stats per mana)
        - Board position
        - Tempo (playing cards is good)

        Args:
            action: PLAY_CARD action
            observation: Current game state

        Returns:
            Value score for playing this card
        """
        card_index = action.card_index
        if card_index is None or card_index >= len(observation.self_hand):
            return 0.0

        card = observation.self_hand[card_index]

        # Base value: prioritize playing cards (high base score)
        value = 100.0

        # Mana efficiency: stats per mana cost
        if hasattr(card, 'attack') and hasattr(card, 'health'):
            mana_cost = max(card.mana_cost, 1)  # Avoid division by zero
            stats = card.attack + card.health
            stats_per_mana = stats / mana_cost

            # Reward efficient cards (2.0+ stats per mana is good)
            value += stats_per_mana * 10.0

        # Prefer playing cards that use mana (tempo)
        mana_used = card.mana_cost
        value += mana_used * 2.0

        return value

    def _evaluate_attack(self, action: Action, observation: Observation) -> float:
        """Evaluate an attack action.

        Medium-high priority - we want to control board and deal damage.

        Factors:
        - Favorable trades (kill enemy, survive)
        - Value trades (kill expensive with cheap)
        - Face damage (when ahead)

        Args:
            action: ATTACK action
            observation: Current game state

        Returns:
            Value score for this attack
        """
        attacker_index = action.attacker_index
        if attacker_index is None or attacker_index >= len(observation.self_board):
            return 0.0

        attacker = observation.self_board[attacker_index]

        # Base value: attacking is generally good (medium priority)
        value = 50.0

        # Attacking a minion (trading)
        if action.defender_index is not None:
            if action.defender_index >= len(observation.opponent_board):
                return 0.0

            defender = observation.opponent_board[action.defender_index]

            # Can we kill the defender?
            if attacker.attack >= defender.health:
                value += 30.0  # Killing is valuable

                # Will we survive?
                if defender.attack < attacker.health:
                    value += 20.0  # Favorable trade (kill and survive)
                else:
                    value += 5.0  # Even trade (both die)

            # Value trade: killing expensive cards is good
            if hasattr(defender, 'mana_cost'):
                value += defender.mana_cost * 2.0

        # Attacking face (hero)
        else:
            # Face damage is good, especially with small minions
            value += attacker.attack * 3.0

        return value

    def _evaluate_hero_power(self, action: Action, observation: Observation) -> float:
        """Evaluate using hero power.

        Low-medium priority - use when nothing better to do.

        Args:
            action: HERO_POWER action
            observation: Current game state

        Returns:
            Value score for hero power
        """
        # Hero power is decent value (uses mana efficiently usually)
        # But lower priority than playing cards or attacking
        value = 20.0

        # If we have extra mana, hero power is more valuable
        if observation.self_mana >= 2:
            value += 10.0

        return value

    def _evaluate_end_turn(self, action: Action, observation: Observation) -> float:
        """Evaluate ending turn.

        Lowest priority - only do when nothing else to do.

        Args:
            action: END_TURN action
            observation: Current game state

        Returns:
            Value score for ending turn (always low)
        """
        # End turn has lowest priority
        # Only choose this if no better options
        return 0.0

    def reset(self) -> None:
        """Reset agent state (no-op for stateless GreedyAgent)."""
        # GreedyAgent is stateless, no need to reset anything
        pass
