"""MCTSAgent - Monte Carlo Tree Search agent.

Uses MCTS algorithm to select actions through:
1. Selection: Traverse tree using UCB1 to select promising nodes
2. Expansion: Add new child node to tree
3. Simulation: Random rollout from new node to terminal state
4. Backpropagation: Update statistics up the tree

MCTS balances exploration (trying new actions) and exploitation (choosing
known good actions) through the UCB1 formula.

This agent is:
- Stronger than random/greedy (looks ahead)
- Slower than greedy (needs simulations)
- A baseline for RL agent comparison
"""

import math
import random
from typing import Optional, List
from agents.base import Agent
from simulation.observation import Observation
from simulation.action_space import Action, ActionType


class MCTSNode:
    """Node in the MCTS search tree.

    Each node represents a game state after taking an action.
    Tracks visit count and total reward for UCB1 selection.
    """

    def __init__(self, action: Action, parent: Optional['MCTSNode'] = None):
        """Initialize a tree node.

        Args:
            action: The action that led to this node
            parent: Parent node (None for root)
        """
        self.action = action
        self.parent = parent
        self.children: List['MCTSNode'] = []
        self.visits = 0
        self.total_reward = 0.0

    def get_average_reward(self) -> float:
        """Calculate average reward (Q-value).

        Returns:
            Average reward, or 0 if never visited
        """
        if self.visits == 0:
            return 0.0
        return self.total_reward / self.visits

    def calculate_ucb1(self, exploration_constant: float = 1.41) -> float:
        """Calculate UCB1 value for this node.

        UCB1 = Q + C * sqrt(ln(N) / n)
        Where:
        - Q = average reward (exploitation)
        - C = exploration constant
        - N = parent visits
        - n = node visits

        Args:
            exploration_constant: Controls exploration vs exploitation trade-off

        Returns:
            UCB1 value (higher = more promising)
        """
        if self.visits == 0:
            return float('inf')  # Unvisited nodes have infinite priority

        if self.parent is None or self.parent.visits == 0:
            return self.get_average_reward()

        exploitation = self.get_average_reward()
        exploration = exploration_constant * math.sqrt(
            math.log(self.parent.visits) / self.visits
        )

        return exploitation + exploration

    def select_best_child(self, exploration_constant: float = 1.41) -> 'MCTSNode':
        """Select child with highest UCB1 value.

        Args:
            exploration_constant: UCB1 exploration constant

        Returns:
            Child node with highest UCB1
        """
        best_child = max(
            self.children,
            key=lambda child: child.calculate_ucb1(exploration_constant)
        )
        return best_child

    def is_fully_expanded(self, num_legal_actions: int) -> bool:
        """Check if all legal actions have been expanded.

        Args:
            num_legal_actions: Number of legal actions from this state

        Returns:
            True if all actions have child nodes
        """
        return len(self.children) >= num_legal_actions


class MCTSAgent(Agent):
    """Agent that uses Monte Carlo Tree Search to select actions.

    MCTS builds a search tree by running simulations. Each simulation:
    1. Selects a path through the tree using UCB1
    2. Expands the tree with a new node
    3. Simulates a random game from that node
    4. Backpropagates the result up the tree

    After many simulations, selects the action with most visits.
    """

    def __init__(
        self,
        name: Optional[str] = None,
        num_iterations: int = 100,
        exploration_constant: float = 1.41,
        seed: Optional[int] = None
    ):
        """Initialize the MCTSAgent.

        Args:
            name: Optional name for the agent
            num_iterations: Number of MCTS iterations per action
            exploration_constant: UCB1 exploration parameter (√2 ≈ 1.41 is common)
            seed: Random seed for deterministic rollouts
        """
        super().__init__(name=name or "MCTSAgent")
        self.num_iterations = num_iterations
        self.exploration_constant = exploration_constant
        self.rng = random.Random(seed)

    def choose_action(self, observation: Observation) -> Action:
        """Choose action using MCTS.

        Args:
            observation: Current game state observation

        Returns:
            Action selected by MCTS
        """
        legal_actions = observation.legal_actions

        if not legal_actions:
            # No legal actions - return END_TURN as fallback
            return Action(type=ActionType.END_TURN)

        if len(legal_actions) == 1:
            # Only one action - no need for search
            return legal_actions[0]

        # Create root node (no action)
        root = MCTSNode(action=Action(type=ActionType.END_TURN), parent=None)

        # Run MCTS iterations
        for _ in range(self.num_iterations):
            # 1. Selection: Select a leaf node
            node = self._select(root, legal_actions)

            # 2. Expansion: Expand if not terminal
            if node.visits > 0 and not node.is_fully_expanded(len(legal_actions)):
                node = self._expand(node, legal_actions)

            # 3. Simulation: Random rollout
            reward = self._simulate(observation, node.action)

            # 4. Backpropagation: Update statistics
            self._backpropagate(node, reward)

        # Select best action (most visited)
        return self._select_best_action(root)

    def _select(self, root: MCTSNode, legal_actions: List[Action]) -> MCTSNode:
        """Selection phase: Traverse tree using UCB1.

        Args:
            root: Root node
            legal_actions: Legal actions from root state

        Returns:
            Selected leaf node
        """
        node = root

        # If root has no children yet, expand it first
        if len(node.children) == 0:
            self._expand_node(node, legal_actions)
            # Return first child for initial exploration
            return node.children[0] if node.children else node

        # Traverse tree while node has children
        while len(node.children) > 0:
            if not node.is_fully_expanded(len(legal_actions)):
                # Not all children expanded - this is a frontier node
                return node

            # Select best child using UCB1
            node = node.select_best_child(self.exploration_constant)

        return node

    def _expand(self, node: MCTSNode, legal_actions: List[Action]) -> MCTSNode:
        """Expansion phase: Add a new child node.

        Args:
            node: Node to expand
            legal_actions: Legal actions

        Returns:
            The newly created child node
        """
        # If no children yet, expand all actions
        if len(node.children) == 0:
            self._expand_node(node, legal_actions)

        # Return least visited child (or first unvisited)
        unvisited = [child for child in node.children if child.visits == 0]
        if unvisited:
            return unvisited[0]

        # All visited - return least visited
        return min(node.children, key=lambda c: c.visits)

    def _expand_node(self, node: MCTSNode, legal_actions: List[Action]) -> None:
        """Create child nodes for all legal actions.

        Args:
            node: Parent node
            legal_actions: Legal actions to expand
        """
        for action in legal_actions:
            child = MCTSNode(action=action, parent=node)
            node.children.append(child)

    def _simulate(self, observation: Observation, action: Action) -> float:
        """Simulation phase: Random rollout to estimate value.

        Note: This is a simplified simulation that doesn't actually play out
        the game. For a full implementation, would need to simulate the game
        forward from this state.

        Args:
            observation: Current observation
            action: Action to simulate from

        Returns:
            Estimated reward (0.0 to 1.0)
        """
        return self._simulate_rollout(observation)

    def _simulate_rollout(self, observation: Observation) -> float:
        """Perform random rollout simulation.

        Simplified version: Returns heuristic evaluation based on board state.
        Full version would simulate game to completion.

        Args:
            observation: Current observation

        Returns:
            Reward estimate (0.0 to 1.0)
        """
        # Simplified heuristic evaluation
        # In a full implementation, would simulate to game end

        # Check if game is over
        if observation.is_game_over:
            if observation.winner == observation.player_index:
                return 1.0  # We won
            else:
                return 0.0  # We lost

        # Heuristic evaluation based on health advantage
        health_diff = observation.self_health - observation.opponent_health

        # Normalize to [0, 1]
        # Assuming max health difference is 30 (one player at 30, other at 0)
        reward = 0.5 + (health_diff / 60.0)
        reward = max(0.0, min(1.0, reward))  # Clamp to [0, 1]

        return reward

    def _backpropagate(self, node: MCTSNode, reward: float) -> None:
        """Backpropagation phase: Update node statistics.

        Updates visit count and total reward for this node and all ancestors.

        Args:
            node: Leaf node to backpropagate from
            reward: Reward to backpropagate
        """
        current = node
        while current is not None:
            current.visits += 1
            current.total_reward += reward
            current = current.parent

    def _select_best_action(self, root: MCTSNode) -> Action:
        """Select best action based on visit counts.

        After MCTS completes, select the most-visited child action.

        Args:
            root: Root node with statistics from MCTS

        Returns:
            Action with most visits
        """
        if not root.children:
            return Action(type=ActionType.END_TURN)

        # Select child with most visits
        best_child = max(root.children, key=lambda child: child.visits)
        return best_child.action

    def reset(self) -> None:
        """Reset agent state.

        MCTS is stateless - tree is rebuilt each turn.
        """
        pass
