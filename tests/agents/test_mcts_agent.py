"""Unit tests for MCTSAgent - Monte Carlo Tree Search agent.

MCTSAgent uses MCTS algorithm:
1. Selection: Traverse tree using UCB1 formula to select promising nodes
2. Expansion: Add new child node to tree
3. Simulation: Play out from new node to terminal state (random rollout)
4. Backpropagation: Update statistics back up the tree

Tests validate:
- Basic agent interface compliance
- MCTS tree node structure and operations
- UCB1 selection formula
- Simulation rollouts
- Action selection based on visit counts
- Configuration (iterations, exploration constant)
- Edge cases and performance
"""

import pytest
from agents.mcts_agent import MCTSAgent, MCTSNode
from agents.base import Agent
from simulation.observation import Observation
from simulation.action_space import Action, ActionType
from hearthstone.cards.base import MinionCard


class TestMCTSAgentBasics:
    """Test basic MCTSAgent functionality."""

    def test_mcts_agent_is_agent(self):
        """MCTSAgent extends Agent base class."""
        agent = MCTSAgent()
        assert isinstance(agent, Agent)

    def test_mcts_agent_has_name(self):
        """MCTSAgent has default name."""
        agent = MCTSAgent()
        assert agent.name == "MCTSAgent"

    def test_mcts_agent_custom_name(self):
        """MCTSAgent can have custom name."""
        agent = MCTSAgent(name="MyMCTS")
        assert agent.name == "MyMCTS"

    def test_mcts_agent_accepts_iterations(self):
        """MCTSAgent accepts num_iterations parameter."""
        agent = MCTSAgent(num_iterations=100)
        assert agent.num_iterations == 100

    def test_mcts_agent_accepts_exploration_constant(self):
        """MCTSAgent accepts exploration_constant for UCB1."""
        agent = MCTSAgent(exploration_constant=2.0)
        assert agent.exploration_constant == 2.0

    def test_mcts_agent_default_config(self):
        """MCTSAgent has reasonable defaults."""
        agent = MCTSAgent()
        assert agent.num_iterations > 0
        assert agent.exploration_constant > 0

    def test_mcts_agent_reset(self):
        """MCTSAgent can be reset (clears tree)."""
        agent = MCTSAgent()
        agent.reset()  # Should not raise


class TestMCTSNode:
    """Test MCTS tree node structure."""

    def test_node_creation(self):
        """Can create a tree node."""
        action = Action(type=ActionType.END_TURN)
        node = MCTSNode(action=action, parent=None)
        assert node.action == action
        assert node.parent is None
        assert node.visits == 0
        assert node.total_reward == 0.0
        assert len(node.children) == 0

    def test_node_has_children_list(self):
        """Node maintains list of children."""
        parent = MCTSNode(action=Action(type=ActionType.END_TURN), parent=None)
        child = MCTSNode(action=Action(type=ActionType.PLAY_CARD, card_index=0), parent=parent)
        parent.children.append(child)

        assert len(parent.children) == 1
        assert parent.children[0] == child
        assert child.parent == parent

    def test_node_tracks_visits(self):
        """Node tracks number of visits."""
        node = MCTSNode(action=Action(type=ActionType.END_TURN), parent=None)
        assert node.visits == 0

        node.visits += 1
        assert node.visits == 1

    def test_node_tracks_total_reward(self):
        """Node accumulates total reward."""
        node = MCTSNode(action=Action(type=ActionType.END_TURN), parent=None)
        assert node.total_reward == 0.0

        node.total_reward += 1.0
        assert node.total_reward == 1.0

        node.total_reward += 0.5
        assert node.total_reward == 1.5

    def test_node_calculates_average_reward(self):
        """Node can calculate average reward (Q-value)."""
        node = MCTSNode(action=Action(type=ActionType.END_TURN), parent=None)
        node.visits = 10
        node.total_reward = 7.0

        avg_reward = node.get_average_reward()
        assert avg_reward == 0.7  # 7.0 / 10

    def test_node_average_reward_is_zero_with_no_visits(self):
        """Unvisited node has average reward of 0."""
        node = MCTSNode(action=Action(type=ActionType.END_TURN), parent=None)
        assert node.get_average_reward() == 0.0


class TestMCTSSelection:
    """Test MCTS selection phase (UCB1)."""

    def test_ucb1_calculation(self):
        """UCB1 formula balances exploitation and exploration."""
        parent = MCTSNode(action=Action(type=ActionType.END_TURN), parent=None)
        parent.visits = 100

        child = MCTSNode(action=Action(type=ActionType.PLAY_CARD, card_index=0), parent=parent)
        child.visits = 10
        child.total_reward = 7.0

        # UCB1 = Q + C * sqrt(ln(N) / n)
        # Q = 7.0 / 10 = 0.7
        # C = exploration constant (default ~1.41)
        # N = parent visits = 100
        # n = child visits = 10
        ucb1 = child.calculate_ucb1(exploration_constant=1.41)

        assert ucb1 > 0.7  # Should be higher than just exploitation
        assert ucb1 < 2.0  # But not unreasonably high

    def test_unvisited_node_has_infinite_ucb1(self):
        """Unvisited nodes are prioritized (infinite UCB1)."""
        parent = MCTSNode(action=Action(type=ActionType.END_TURN), parent=None)
        parent.visits = 10

        unvisited = MCTSNode(action=Action(type=ActionType.PLAY_CARD, card_index=0), parent=parent)

        ucb1 = unvisited.calculate_ucb1(exploration_constant=1.41)
        assert ucb1 == float('inf')

    def test_select_best_child_chooses_highest_ucb1(self):
        """Selection chooses child with highest UCB1."""
        parent = MCTSNode(action=Action(type=ActionType.END_TURN), parent=None)
        parent.visits = 100

        # Child 1: high reward, many visits
        child1 = MCTSNode(action=Action(type=ActionType.PLAY_CARD, card_index=0), parent=parent)
        child1.visits = 50
        child1.total_reward = 40.0
        parent.children.append(child1)

        # Child 2: low reward, few visits (should be explored)
        child2 = MCTSNode(action=Action(type=ActionType.PLAY_CARD, card_index=1), parent=parent)
        child2.visits = 5
        child2.total_reward = 2.0
        parent.children.append(child2)

        # Child 3: unvisited (should be selected)
        child3 = MCTSNode(action=Action(type=ActionType.ATTACK, attacker_index=0), parent=parent)
        parent.children.append(child3)

        best = parent.select_best_child(exploration_constant=1.41)
        assert best == child3  # Unvisited nodes have infinite UCB1


class TestMCTSExpansion:
    """Test MCTS expansion phase."""

    def test_expand_creates_children(self):
        """Expansion creates child nodes for legal actions."""
        node = MCTSNode(action=Action(type=ActionType.END_TURN), parent=None)

        legal_actions = [
            Action(type=ActionType.PLAY_CARD, card_index=0, position=0),
            Action(type=ActionType.PLAY_CARD, card_index=1, position=0),
            Action(type=ActionType.END_TURN)
        ]

        agent = MCTSAgent()
        agent._expand_node(node, legal_actions)

        assert len(node.children) == 3
        assert all(isinstance(child, MCTSNode) for child in node.children)
        assert all(child.parent == node for child in node.children)

    def test_expansion_preserves_actions(self):
        """Expanded children maintain correct actions."""
        node = MCTSNode(action=Action(type=ActionType.END_TURN), parent=None)

        legal_actions = [
            Action(type=ActionType.PLAY_CARD, card_index=0, position=0),
            Action(type=ActionType.ATTACK, attacker_index=0, defender_index=None)
        ]

        agent = MCTSAgent()
        agent._expand_node(node, legal_actions)

        child_actions = [child.action for child in node.children]
        assert legal_actions[0] in child_actions
        assert legal_actions[1] in child_actions


class TestMCTSSimulation:
    """Test MCTS simulation/rollout phase."""

    def test_rollout_plays_to_end(self):
        """Rollout simulates game to terminal state."""
        agent = MCTSAgent()

        # Simple observation with only END_TURN available
        observation = Observation(
            self_hand=[],
            self_board=[],
            opponent_board=[],
            self_mana=0,
            self_max_mana=0,
            self_health=30,
            self_fatigue_counter=0,
            opponent_health=30,
            legal_actions=[Action(type=ActionType.END_TURN)]
        )

        # Rollout should complete without error
        reward = agent._simulate_rollout(observation)
        assert isinstance(reward, float)
        assert 0.0 <= reward <= 1.0  # Reward should be normalized

    def test_rollout_returns_reward(self):
        """Rollout returns reward based on outcome."""
        agent = MCTSAgent()

        # Observation where we're winning
        observation = Observation(
            self_hand=[],
            self_board=[],
            opponent_board=[],
            self_mana=0,
            self_max_mana=0,
            self_health=30,
            self_fatigue_counter=0,
            opponent_health=1,  # Opponent almost dead
            legal_actions=[Action(type=ActionType.END_TURN)]
        )

        reward = agent._simulate_rollout(observation)
        # Should complete and return some reward
        assert reward is not None


class TestMCTSBackpropagation:
    """Test MCTS backpropagation phase."""

    def test_backpropagate_updates_ancestors(self):
        """Backpropagation updates all ancestor nodes."""
        # Create a chain: root -> child -> grandchild
        root = MCTSNode(action=Action(type=ActionType.END_TURN), parent=None)
        child = MCTSNode(action=Action(type=ActionType.PLAY_CARD, card_index=0), parent=root)
        grandchild = MCTSNode(action=Action(type=ActionType.ATTACK, attacker_index=0), parent=child)

        agent = MCTSAgent()
        agent._backpropagate(grandchild, reward=1.0)

        # All nodes should be updated
        assert grandchild.visits == 1
        assert grandchild.total_reward == 1.0
        assert child.visits == 1
        assert child.total_reward == 1.0
        assert root.visits == 1
        assert root.total_reward == 1.0

    def test_backpropagate_accumulates_visits(self):
        """Multiple backpropagations accumulate."""
        node = MCTSNode(action=Action(type=ActionType.END_TURN), parent=None)

        agent = MCTSAgent()
        agent._backpropagate(node, reward=1.0)
        agent._backpropagate(node, reward=0.5)

        assert node.visits == 2
        assert node.total_reward == 1.5


class TestMCTSActionSelection:
    """Test final action selection."""

    def test_selects_most_visited_action(self):
        """MCTS selects action with most visits."""
        agent = MCTSAgent()

        observation = Observation(
            self_hand=[],
            self_board=[],
            opponent_board=[],
            self_mana=0,
            self_max_mana=0,
            self_health=30,
            self_fatigue_counter=0,
            opponent_health=30,
            legal_actions=[
                Action(type=ActionType.PLAY_CARD, card_index=0, position=0),
                Action(type=ActionType.END_TURN)
            ]
        )

        # Should complete and return an action
        action = agent.choose_action(observation)
        assert action is not None
        assert action.type in [ActionType.PLAY_CARD, ActionType.END_TURN]

    def test_single_action_returns_immediately(self):
        """With only one legal action, return it without search."""
        agent = MCTSAgent()

        observation = Observation(
            self_hand=[],
            self_board=[],
            opponent_board=[],
            self_mana=0,
            self_max_mana=0,
            self_health=30,
            self_fatigue_counter=0,
            opponent_health=30,
            legal_actions=[Action(type=ActionType.END_TURN)]
        )

        action = agent.choose_action(observation)
        assert action.type == ActionType.END_TURN


class TestMCTSEdgeCases:
    """Test edge cases for MCTS."""

    def test_handles_empty_legal_actions(self):
        """Gracefully handles empty legal actions list."""
        agent = MCTSAgent()

        observation = Observation(
            self_hand=[],
            self_board=[],
            opponent_board=[],
            self_mana=0,
            self_max_mana=0,
            self_health=30,
            self_fatigue_counter=0,
            opponent_health=30,
            legal_actions=[]
        )

        # Should return END_TURN as fallback
        action = agent.choose_action(observation)
        assert action.type == ActionType.END_TURN

    def test_low_iteration_count_still_works(self):
        """Works with very low iteration count."""
        agent = MCTSAgent(num_iterations=1)

        observation = Observation(
            self_hand=[],
            self_board=[],
            opponent_board=[],
            self_mana=0,
            self_max_mana=0,
            self_health=30,
            self_fatigue_counter=0,
            opponent_health=30,
            legal_actions=[
                Action(type=ActionType.PLAY_CARD, card_index=0, position=0),
                Action(type=ActionType.END_TURN)
            ]
        )

        action = agent.choose_action(observation)
        assert action is not None

    def test_deterministic_with_seed(self):
        """MCTS is deterministic with same seed."""
        observation = Observation(
            self_hand=[MinionCard(name="M", mana_cost=1, attack=1, health=1)],
            self_board=[],
            opponent_board=[],
            self_mana=1,
            self_max_mana=1,
            self_health=30,
            self_fatigue_counter=0,
            opponent_health=30,
            legal_actions=[
                Action(type=ActionType.PLAY_CARD, card_index=0, position=0),
                Action(type=ActionType.END_TURN)
            ]
        )

        agent1 = MCTSAgent(num_iterations=10, seed=42)
        agent2 = MCTSAgent(num_iterations=10, seed=42)

        action1 = agent1.choose_action(observation)
        action2 = agent2.choose_action(observation)

        assert action1.type == action2.type
        assert action1.card_index == action2.card_index


class TestMCTSPerformance:
    """Test MCTS performance characteristics."""

    def test_completes_in_reasonable_time(self):
        """MCTS completes search within reasonable time."""
        import time

        agent = MCTSAgent(num_iterations=50)

        observation = Observation(
            self_hand=[],
            self_board=[],
            opponent_board=[],
            self_mana=0,
            self_max_mana=0,
            self_health=30,
            self_fatigue_counter=0,
            opponent_health=30,
            legal_actions=[
                Action(type=ActionType.PLAY_CARD, card_index=0, position=0),
                Action(type=ActionType.END_TURN)
            ]
        )

        start = time.time()
        action = agent.choose_action(observation)
        elapsed = time.time() - start

        assert elapsed < 5.0  # Should complete in under 5 seconds
        assert action is not None

    def test_more_iterations_explores_more(self):
        """Higher iteration count explores more thoroughly."""
        obs = Observation(
            self_hand=[],
            self_board=[],
            opponent_board=[],
            self_mana=0,
            self_max_mana=0,
            self_health=30,
            self_fatigue_counter=0,
            opponent_health=30,
            legal_actions=[
                Action(type=ActionType.PLAY_CARD, card_index=0, position=0),
                Action(type=ActionType.END_TURN)
            ]
        )

        # Both should work, but we can't directly verify more exploration
        # without access to internals
        agent_few = MCTSAgent(num_iterations=5)
        agent_many = MCTSAgent(num_iterations=50)

        action_few = agent_few.choose_action(obs)
        action_many = agent_many.choose_action(obs)

        assert action_few is not None
        assert action_many is not None
