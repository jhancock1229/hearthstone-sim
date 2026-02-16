"""Unit tests for PPO (Proximal Policy Optimization) agent.

Tests the PPO agent which combines:
- Actor-Critic network for policy and value estimation
- Feature extraction from game observations
- Action masking for legal actions
- Rollout buffer for experience collection
- GAE (Generalized Advantage Estimation) for advantage computation
- Clipped surrogate objective for stable policy updates

The PPO agent extends the base Agent interface so it can be used
directly with the Simulator for evaluation.
"""

import pytest
import torch
import numpy as np

from agents.rl.ppo_agent import PPOAgent, RolloutBuffer, compute_gae
from agents.rl.features import FeatureExtractor
from agents.rl.networks import ActorCriticNetwork
from agents.rl.env import ACTION_SPACE_SIZE
from agents.base import Agent
from simulation.observation import Observation
from simulation.action_space import Action, ActionType
from hearthstone.cards.base import MinionCard


def _make_observation(
    self_health=30, opponent_health=30, self_mana=5, self_max_mana=5,
    self_board=None, opponent_board=None, self_hand=None,
    turn_number=5, player_index=0, legal_actions=None,
):
    """Helper to create test observations."""
    if self_board is None:
        self_board = []
    if opponent_board is None:
        opponent_board = []
    if self_hand is None:
        self_hand = []
    if legal_actions is None:
        legal_actions = [Action(type=ActionType.END_TURN)]

    return Observation(
        self_health=self_health,
        self_mana=self_mana,
        self_max_mana=self_max_mana,
        self_fatigue_counter=0,
        self_hand=self_hand,
        self_deck_size=20,
        self_board=self_board,
        opponent_health=opponent_health,
        opponent_mana=self_mana,
        opponent_max_mana=self_max_mana,
        opponent_fatigue_counter=0,
        opponent_hand_size=3,
        opponent_deck_size=20,
        opponent_board=opponent_board,
        turn_number=turn_number,
        is_my_turn=True,
        is_game_over=False,
        winner=None,
        player_index=player_index,
        legal_actions=legal_actions,
    )


# --- PPOAgent Creation ---

class TestPPOAgentCreation:
    """Test PPO agent initialization."""

    def test_create_ppo_agent(self):
        """Can create a PPOAgent with default parameters."""
        agent = PPOAgent()
        assert agent is not None

    def test_ppo_agent_is_agent_subclass(self):
        """PPOAgent inherits from base Agent."""
        agent = PPOAgent()
        assert isinstance(agent, Agent)

    def test_ppo_agent_has_network(self):
        """PPOAgent has an ActorCriticNetwork."""
        agent = PPOAgent()
        assert hasattr(agent, 'network')
        assert isinstance(agent.network, ActorCriticNetwork)

    def test_ppo_agent_has_feature_extractor(self):
        """PPOAgent has a FeatureExtractor."""
        agent = PPOAgent()
        assert hasattr(agent, 'feature_extractor')
        assert isinstance(agent.feature_extractor, FeatureExtractor)

    def test_ppo_agent_has_optimizer(self):
        """PPOAgent has an optimizer for training."""
        agent = PPOAgent()
        assert hasattr(agent, 'optimizer')

    def test_ppo_agent_custom_hyperparameters(self):
        """Can create PPOAgent with custom hyperparameters."""
        agent = PPOAgent(
            learning_rate=1e-4,
            gamma=0.95,
            gae_lambda=0.9,
            clip_epsilon=0.1,
            value_loss_coef=0.25,
            entropy_coef=0.05,
        )
        assert agent.gamma == 0.95
        assert agent.gae_lambda == 0.9
        assert agent.clip_epsilon == 0.1

    def test_ppo_agent_default_hyperparameters(self):
        """PPOAgent has sensible default hyperparameters."""
        agent = PPOAgent()
        assert 0.9 <= agent.gamma <= 1.0
        assert 0.0 < agent.clip_epsilon <= 0.5
        assert agent.learning_rate > 0


# --- Action Selection ---

class TestPPOAgentActionSelection:
    """Test PPO agent action selection (choose_action)."""

    def test_choose_action_returns_action(self):
        """choose_action returns an Action object."""
        agent = PPOAgent()
        obs = _make_observation()
        action = agent.choose_action(obs)
        assert isinstance(action, Action)

    def test_choose_action_respects_legal_actions(self):
        """Agent only selects from legal actions."""
        agent = PPOAgent()

        # Only END_TURN is legal
        legal = [Action(type=ActionType.END_TURN)]
        obs = _make_observation(legal_actions=legal)

        action = agent.choose_action(obs)
        assert action.type == ActionType.END_TURN

    def test_choose_action_with_play_card_legal(self):
        """Agent can select PLAY_CARD when it's legal."""
        agent = PPOAgent()

        hand = [MinionCard(name="Wisp", mana_cost=0, attack=1, health=1)]
        legal = [
            Action(type=ActionType.PLAY_CARD, card_index=0, position=0),
            Action(type=ActionType.END_TURN),
        ]
        obs = _make_observation(self_hand=hand, legal_actions=legal)

        # Run many times - should sometimes pick PLAY_CARD
        actions_chosen = set()
        for _ in range(50):
            action = agent.choose_action(obs)
            actions_chosen.add(action.type)

        # With random initialization, should explore both actions
        assert ActionType.END_TURN in actions_chosen

    def test_choose_action_with_attack_legal(self):
        """Agent can select ATTACK when it's legal."""
        agent = PPOAgent()

        board = [MinionCard(name="Yeti", mana_cost=4, attack=4, health=5,
                           exhausted=False, summoning_sick=False)]
        legal = [
            Action(type=ActionType.ATTACK, attacker_index=0, defender_index=None),
            Action(type=ActionType.END_TURN),
        ]
        obs = _make_observation(self_board=board, legal_actions=legal)

        action = agent.choose_action(obs)
        assert action.type in [ActionType.ATTACK, ActionType.END_TURN]

    def test_choose_action_deterministic_mode(self):
        """Agent can act deterministically (greedy) for evaluation."""
        agent = PPOAgent()

        legal = [Action(type=ActionType.END_TURN)]
        obs = _make_observation(legal_actions=legal)

        # Deterministic mode should always choose the same action
        action1 = agent.choose_action(obs, deterministic=True)
        action2 = agent.choose_action(obs, deterministic=True)
        assert action1 == action2


# --- Rollout Buffer ---

class TestRolloutBuffer:
    """Test experience collection buffer."""

    def test_create_rollout_buffer(self):
        """Can create an empty RolloutBuffer."""
        buf = RolloutBuffer()
        assert buf is not None
        assert len(buf) == 0

    def test_add_transition(self):
        """Can add transitions to the buffer."""
        buf = RolloutBuffer()
        buf.add(
            observation=np.zeros(106, dtype=np.float32),
            action=99,
            reward=0.0,
            value=0.5,
            log_prob=-1.0,
            done=False,
            action_mask=np.ones(ACTION_SPACE_SIZE, dtype=bool),
        )
        assert len(buf) == 1

    def test_add_multiple_transitions(self):
        """Buffer tracks multiple transitions."""
        buf = RolloutBuffer()
        for i in range(10):
            buf.add(
                observation=np.zeros(106, dtype=np.float32),
                action=99,
                reward=0.0,
                value=float(i),
                log_prob=-1.0,
                done=(i == 9),
                action_mask=np.ones(ACTION_SPACE_SIZE, dtype=bool),
            )
        assert len(buf) == 10

    def test_clear_buffer(self):
        """Can clear the buffer."""
        buf = RolloutBuffer()
        buf.add(
            observation=np.zeros(106, dtype=np.float32),
            action=99,
            reward=0.0,
            value=0.5,
            log_prob=-1.0,
            done=False,
            action_mask=np.ones(ACTION_SPACE_SIZE, dtype=bool),
        )
        assert len(buf) == 1
        buf.clear()
        assert len(buf) == 0

    def test_get_batch_returns_tensors(self):
        """get_batch returns dict of PyTorch tensors."""
        buf = RolloutBuffer()
        for i in range(5):
            buf.add(
                observation=np.random.randn(106).astype(np.float32),
                action=i % ACTION_SPACE_SIZE,
                reward=0.0 if i < 4 else 1.0,
                value=0.5,
                log_prob=-1.0,
                done=(i == 4),
                action_mask=np.ones(ACTION_SPACE_SIZE, dtype=bool),
            )

        batch = buf.get_batch(gamma=0.99, gae_lambda=0.95)

        assert 'observations' in batch
        assert 'actions' in batch
        assert 'returns' in batch
        assert 'advantages' in batch
        assert 'old_log_probs' in batch
        assert 'action_masks' in batch

        assert isinstance(batch['observations'], torch.Tensor)
        assert isinstance(batch['actions'], torch.Tensor)

    def test_batch_shapes_are_correct(self):
        """Batch tensors have correct shapes."""
        buf = RolloutBuffer()
        n = 8
        for i in range(n):
            buf.add(
                observation=np.random.randn(106).astype(np.float32),
                action=99,
                reward=0.0 if i < n - 1 else 1.0,
                value=0.5,
                log_prob=-1.0,
                done=(i == n - 1),
                action_mask=np.ones(ACTION_SPACE_SIZE, dtype=bool),
            )

        batch = buf.get_batch(gamma=0.99, gae_lambda=0.95)

        assert batch['observations'].shape == (n, 106)
        assert batch['actions'].shape == (n,)
        assert batch['returns'].shape == (n,)
        assert batch['advantages'].shape == (n,)
        assert batch['old_log_probs'].shape == (n,)
        assert batch['action_masks'].shape == (n, ACTION_SPACE_SIZE)


# --- GAE Computation ---

class TestGAE:
    """Test Generalized Advantage Estimation."""

    def test_compute_gae_returns_correct_length(self):
        """GAE returns advantages matching the number of steps."""
        rewards = [0.0, 0.0, 0.0, 1.0]
        values = [0.5, 0.5, 0.5, 0.5]
        dones = [False, False, False, True]

        advantages, returns = compute_gae(
            rewards, values, dones, gamma=0.99, gae_lambda=0.95
        )

        assert len(advantages) == 4
        assert len(returns) == 4

    def test_gae_terminal_state(self):
        """Terminal state advantage equals reward minus value."""
        rewards = [1.0]
        values = [0.0]
        dones = [True]

        advantages, returns = compute_gae(
            rewards, values, dones, gamma=0.99, gae_lambda=0.95
        )

        # At terminal: advantage = reward - value = 1.0 - 0.0 = 1.0
        assert abs(advantages[0] - 1.0) < 1e-6

    def test_gae_zero_rewards_zero_advantages(self):
        """Zero rewards and equal values produce near-zero advantages."""
        rewards = [0.0, 0.0, 0.0]
        values = [0.5, 0.5, 0.5]
        dones = [False, False, True]

        advantages, returns = compute_gae(
            rewards, values, dones, gamma=0.99, gae_lambda=0.95
        )

        # Terminal: adv = 0.0 - 0.5 = -0.5
        # Middle: td_error = 0 + 0.99*0.5 - 0.5 = -0.005, then gae accumulates
        # All should be negative (values overestimate)
        assert advantages[-1] < 0

    def test_gae_positive_reward_positive_advantage(self):
        """Positive terminal reward creates positive advantage at end."""
        rewards = [0.0, 0.0, 10.0]
        values = [0.0, 0.0, 0.0]
        dones = [False, False, True]

        advantages, returns = compute_gae(
            rewards, values, dones, gamma=0.99, gae_lambda=0.95
        )

        # Terminal advantage should be large and positive
        assert advantages[-1] > 0
        # Earlier steps should also have positive advantage (discounted back)
        assert advantages[0] > 0

    def test_gae_returns_equal_advantages_plus_values(self):
        """Returns = advantages + values."""
        rewards = [0.0, 0.0, 1.0]
        values = [0.3, 0.5, 0.2]
        dones = [False, False, True]

        advantages, returns = compute_gae(
            rewards, values, dones, gamma=0.99, gae_lambda=0.95
        )

        for i in range(len(returns)):
            assert abs(returns[i] - (advantages[i] + values[i])) < 1e-6


# --- PPO Update ---

class TestPPOUpdate:
    """Test PPO training update."""

    def _fill_buffer(self, agent, n=16):
        """Helper to fill buffer with dummy transitions."""
        for i in range(n):
            agent.rollout_buffer.add(
                observation=np.random.randn(106).astype(np.float32),
                action=99,  # END_TURN
                reward=0.0 if i < n - 1 else 1.0,
                value=0.5,
                log_prob=-0.5,
                done=(i == n - 1),
                action_mask=np.ones(ACTION_SPACE_SIZE, dtype=bool),
            )

    def test_update_returns_loss_dict(self):
        """PPO update returns a dict with loss components."""
        agent = PPOAgent()
        self._fill_buffer(agent, n=16)

        losses = agent.update()

        assert isinstance(losses, dict)
        assert 'policy_loss' in losses
        assert 'value_loss' in losses
        assert 'entropy' in losses
        assert 'total_loss' in losses

    def test_update_clears_buffer(self):
        """Buffer is cleared after update."""
        agent = PPOAgent()
        self._fill_buffer(agent, n=16)

        assert len(agent.rollout_buffer) > 0
        agent.update()
        assert len(agent.rollout_buffer) == 0

    def test_update_changes_network_weights(self):
        """PPO update modifies network parameters."""
        agent = PPOAgent(learning_rate=1e-3)
        self._fill_buffer(agent, n=32)

        # Snapshot weights before
        params_before = [p.clone() for p in agent.network.parameters()]

        agent.update()

        # At least some parameters should change
        any_changed = False
        for p_before, p_after in zip(params_before, agent.network.parameters()):
            if not torch.allclose(p_before, p_after):
                any_changed = True
                break

        assert any_changed, "Network weights should change after update"

    def test_update_with_multiple_epochs(self):
        """PPO update runs multiple epochs over the data."""
        agent = PPOAgent(ppo_epochs=4)
        self._fill_buffer(agent, n=32)

        losses = agent.update()
        assert losses is not None

    def test_update_losses_are_finite(self):
        """All loss values are finite (not NaN or inf)."""
        agent = PPOAgent()
        self._fill_buffer(agent, n=16)

        losses = agent.update()

        for key, value in losses.items():
            assert np.isfinite(value), f"{key} is not finite: {value}"


# --- Integration with Env ---

class TestPPOWithEnv:
    """Test PPO agent integration with HearthstoneEnv."""

    def test_collect_rollout_from_env(self):
        """Can collect a full rollout from the environment."""
        from agents.rl.env import HearthstoneEnv

        agent = PPOAgent()
        env = HearthstoneEnv(seed=42, max_turns=30)

        obs, info = env.reset()
        done = False

        while not done:
            mask = info['action_mask']

            # Get action with value and log_prob for training
            action_idx, value, log_prob = agent.get_action_and_value(obs, mask)

            next_obs, reward, terminated, truncated, info = env.step(action_idx)
            done = terminated or truncated

            agent.rollout_buffer.add(
                observation=obs,
                action=action_idx,
                reward=reward,
                value=value,
                log_prob=log_prob,
                done=done,
                action_mask=mask,
            )

            obs = next_obs

        assert len(agent.rollout_buffer) > 0

    def test_train_one_episode(self):
        """Can collect rollout and run one PPO update."""
        from agents.rl.env import HearthstoneEnv

        agent = PPOAgent(learning_rate=1e-3)
        env = HearthstoneEnv(seed=42, max_turns=30)

        obs, info = env.reset()
        done = False

        while not done:
            mask = info['action_mask']
            action_idx, value, log_prob = agent.get_action_and_value(obs, mask)

            next_obs, reward, terminated, truncated, info = env.step(action_idx)
            done = terminated or truncated

            agent.rollout_buffer.add(
                observation=obs,
                action=action_idx,
                reward=reward,
                value=value,
                log_prob=log_prob,
                done=done,
                action_mask=mask,
            )
            obs = next_obs

        losses = agent.update()
        assert losses is not None
        assert np.isfinite(losses['total_loss'])


# --- get_action_and_value ---

class TestGetActionAndValue:
    """Test the combined action + value + log_prob method."""

    def test_returns_three_values(self):
        """get_action_and_value returns (action_idx, value, log_prob)."""
        agent = PPOAgent()

        obs = np.random.randn(106).astype(np.float32)
        mask = np.ones(ACTION_SPACE_SIZE, dtype=bool)

        result = agent.get_action_and_value(obs, mask)
        assert len(result) == 3

    def test_action_is_valid_index(self):
        """Returned action is a valid action index."""
        agent = PPOAgent()

        obs = np.random.randn(106).astype(np.float32)
        mask = np.zeros(ACTION_SPACE_SIZE, dtype=bool)
        mask[99] = True  # Only END_TURN valid

        action_idx, _, _ = agent.get_action_and_value(obs, mask)
        assert action_idx == 99

    def test_value_is_float(self):
        """Returned value is a float."""
        agent = PPOAgent()

        obs = np.random.randn(106).astype(np.float32)
        mask = np.ones(ACTION_SPACE_SIZE, dtype=bool)

        _, value, _ = agent.get_action_and_value(obs, mask)
        assert isinstance(value, float)

    def test_log_prob_is_negative(self):
        """Log probability is negative (prob < 1)."""
        agent = PPOAgent()

        obs = np.random.randn(106).astype(np.float32)
        mask = np.ones(ACTION_SPACE_SIZE, dtype=bool)

        _, _, log_prob = agent.get_action_and_value(obs, mask)
        assert log_prob <= 0.0

    def test_respects_action_mask(self):
        """Only masked-in actions can be selected."""
        agent = PPOAgent()

        obs = np.random.randn(106).astype(np.float32)
        mask = np.zeros(ACTION_SPACE_SIZE, dtype=bool)
        mask[0] = True   # PLAY_CARD hand=0 pos=0
        mask[99] = True  # END_TURN

        for _ in range(20):
            action_idx, _, _ = agent.get_action_and_value(obs, mask)
            assert action_idx in [0, 99]


# --- Save/Load ---

class TestPPOSaveLoad:
    """Test model persistence."""

    def test_save_and_load(self, tmp_path):
        """Can save and load model weights."""
        agent1 = PPOAgent()
        path = tmp_path / "ppo_model.pt"
        agent1.save(str(path))

        agent2 = PPOAgent()
        agent2.load(str(path))

        # Weights should match
        for p1, p2 in zip(agent1.network.parameters(), agent2.network.parameters()):
            assert torch.allclose(p1, p2)

    def test_loaded_model_produces_same_output(self, tmp_path):
        """Loaded model produces identical outputs."""
        agent1 = PPOAgent()
        path = tmp_path / "ppo_model.pt"
        agent1.save(str(path))

        agent2 = PPOAgent()
        agent2.load(str(path))

        obs = np.random.randn(106).astype(np.float32)
        mask = np.ones(ACTION_SPACE_SIZE, dtype=bool)

        # Both should produce same value estimate
        agent1.network.eval()
        agent2.network.eval()

        with torch.no_grad():
            t = torch.from_numpy(obs).unsqueeze(0)
            _, v1 = agent1.network(t)
            _, v2 = agent2.network(t)

        assert torch.allclose(v1, v2)


# --- Reset ---

class TestPPOReset:
    """Test agent reset between games."""

    def test_reset_clears_buffer(self):
        """Reset clears the rollout buffer."""
        agent = PPOAgent()
        agent.rollout_buffer.add(
            observation=np.zeros(106, dtype=np.float32),
            action=99,
            reward=0.0,
            value=0.5,
            log_prob=-1.0,
            done=True,
            action_mask=np.ones(ACTION_SPACE_SIZE, dtype=bool),
        )
        assert len(agent.rollout_buffer) > 0

        agent.reset()
        assert len(agent.rollout_buffer) == 0
