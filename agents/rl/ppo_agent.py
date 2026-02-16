"""PPO (Proximal Policy Optimization) agent for Hearthstone.

Implements PPO-Clip with:
- Actor-Critic network (shared backbone)
- Feature extraction from Observations
- Action masking for legal moves
- Rollout buffer for experience collection
- GAE (Generalized Advantage Estimation)
- Clipped surrogate objective for stable updates
- Entropy bonus for exploration

Usage:
    agent = PPOAgent()
    obs, info = env.reset()
    action_idx, value, log_prob = agent.get_action_and_value(obs, info['action_mask'])
    next_obs, reward, done, truncated, info = env.step(action_idx)
    agent.rollout_buffer.add(obs, action_idx, reward, value, log_prob, done, mask)
    losses = agent.update()
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from agents.base import Agent
from agents.rl.env import ACTION_SPACE_SIZE
from agents.rl.features import FeatureExtractor
from agents.rl.networks import ActorCriticNetwork
from simulation.action_space import Action, ActionType
from simulation.observation import Observation


def compute_gae(
    rewards: List[float],
    values: List[float],
    dones: List[bool],
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> Tuple[List[float], List[float]]:
    """Compute Generalized Advantage Estimation.

    Args:
        rewards: List of rewards at each step
        values: List of value estimates at each step
        dones: List of done flags at each step
        gamma: Discount factor
        gae_lambda: GAE lambda for bias-variance tradeoff

    Returns:
        Tuple of (advantages, returns) as lists of floats
    """
    n = len(rewards)
    advantages = [0.0] * n
    last_gae = 0.0

    for t in reversed(range(n)):
        if dones[t]:
            next_value = 0.0
            last_gae = 0.0
        else:
            next_value = values[t + 1] if t + 1 < n else 0.0

        td_error = rewards[t] + gamma * next_value - values[t]
        last_gae = td_error + gamma * gae_lambda * last_gae
        advantages[t] = last_gae

    returns = [advantages[t] + values[t] for t in range(n)]
    return advantages, returns


class RolloutBuffer:
    """Collects experience tuples during rollout for PPO training.

    Stores (observation, action, reward, value, log_prob, done, action_mask)
    transitions and converts them to batched tensors for training.
    """

    def __init__(self):
        self.clear()

    def add(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        value: float,
        log_prob: float,
        done: bool,
        action_mask: np.ndarray,
    ):
        """Add a transition to the buffer."""
        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
        self.action_masks.append(action_mask)

    def clear(self):
        """Clear all stored transitions."""
        self.observations = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.action_masks = []

    def __len__(self):
        return len(self.observations)

    def get_batch(
        self, gamma: float = 0.99, gae_lambda: float = 0.95
    ) -> Dict[str, torch.Tensor]:
        """Convert buffer to batched tensors with computed advantages.

        Args:
            gamma: Discount factor
            gae_lambda: GAE lambda

        Returns:
            Dict with keys: observations, actions, returns, advantages,
            old_log_probs, action_masks
        """
        advantages, returns = compute_gae(
            self.rewards, self.values, self.dones,
            gamma=gamma, gae_lambda=gae_lambda,
        )

        return {
            'observations': torch.tensor(
                np.array(self.observations), dtype=torch.float32
            ),
            'actions': torch.tensor(self.actions, dtype=torch.long),
            'returns': torch.tensor(returns, dtype=torch.float32),
            'advantages': torch.tensor(advantages, dtype=torch.float32),
            'old_log_probs': torch.tensor(self.log_probs, dtype=torch.float32),
            'action_masks': torch.tensor(
                np.array(self.action_masks), dtype=torch.bool
            ),
        }


class PPOAgent(Agent):
    """PPO agent that extends the base Agent interface.

    Can be used both for training (with rollout collection and updates)
    and for evaluation (just choose_action).
    """

    def __init__(
        self,
        name: Optional[str] = None,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.01,
        ppo_epochs: int = 4,
        max_grad_norm: float = 0.5,
        hidden_sizes: Optional[List[int]] = None,
    ):
        super().__init__(name=name)

        self.learning_rate = learning_rate
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.ppo_epochs = ppo_epochs
        self.max_grad_norm = max_grad_norm

        # Feature extraction
        self.feature_extractor = FeatureExtractor()

        # Network
        self.network = ActorCriticNetwork(
            input_size=self.feature_extractor.feature_size,
            action_size=ACTION_SPACE_SIZE,
            hidden_sizes=hidden_sizes,
        )

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.network.parameters(), lr=learning_rate
        )

        # Rollout buffer
        self.rollout_buffer = RolloutBuffer()

    def choose_action(self, observation: Observation, deterministic: bool = False) -> Action:
        """Choose a game Action from an Observation (Agent interface).

        Extracts features, runs network, samples from masked policy.

        Args:
            observation: Game observation with legal_actions
            deterministic: If True, pick highest-probability action

        Returns:
            Game Action object
        """
        # Extract features
        features = self.feature_extractor.extract(observation)

        # Build action mask from legal actions
        mask = self._build_action_mask(observation.legal_actions)

        # Get action index
        with torch.no_grad():
            obs_tensor = torch.from_numpy(features).unsqueeze(0)
            mask_tensor = torch.from_numpy(mask).unsqueeze(0)

            logits, _ = self.network(obs_tensor)
            masked_logits = self.network.apply_action_mask(logits, mask_tensor)

            if deterministic:
                action_idx = masked_logits.argmax(dim=-1).item()
            else:
                dist = Categorical(logits=masked_logits)
                action_idx = dist.sample().item()

        # Convert action index back to game Action
        return self._index_to_action(action_idx, observation.legal_actions)

    def get_action_and_value(
        self, obs: np.ndarray, action_mask: np.ndarray
    ) -> Tuple[int, float, float]:
        """Get action, value estimate, and log probability.

        Used during rollout collection for training.

        Args:
            obs: Feature vector (numpy array)
            action_mask: Boolean mask over actions (numpy array)

        Returns:
            Tuple of (action_index, value, log_prob)
        """
        with torch.no_grad():
            obs_tensor = torch.from_numpy(obs).unsqueeze(0)
            mask_tensor = torch.from_numpy(action_mask).unsqueeze(0)

            logits, value = self.network(obs_tensor)
            masked_logits = self.network.apply_action_mask(logits, mask_tensor)

            dist = Categorical(logits=masked_logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        return action.item(), value.squeeze().item(), log_prob.item()

    def update(self) -> Dict[str, float]:
        """Run PPO update on collected rollout data.

        Returns:
            Dict with loss components (policy_loss, value_loss, entropy, total_loss)
        """
        batch = self.rollout_buffer.get_batch(
            gamma=self.gamma, gae_lambda=self.gae_lambda
        )

        # Normalize advantages
        advantages = batch['advantages']
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        observations = batch['observations']
        actions = batch['actions']
        returns = batch['returns']
        old_log_probs = batch['old_log_probs']
        action_masks = batch['action_masks']

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0

        for _ in range(self.ppo_epochs):
            # Forward pass
            logits, values = self.network(observations)
            masked_logits = self.network.apply_action_mask(logits, action_masks)

            dist = Categorical(logits=masked_logits)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            # Policy loss (clipped surrogate)
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            value_loss = F.mse_loss(values.squeeze(-1), returns)

            # Total loss
            loss = (
                policy_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy
            )

            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.network.parameters(), self.max_grad_norm
            )
            self.optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.item()

        n_epochs = self.ppo_epochs
        self.rollout_buffer.clear()

        return {
            'policy_loss': total_policy_loss / n_epochs,
            'value_loss': total_value_loss / n_epochs,
            'entropy': total_entropy / n_epochs,
            'total_loss': (total_policy_loss + total_value_loss) / n_epochs,
        }

    def save(self, path: str):
        """Save model weights to file.

        Args:
            path: File path for the checkpoint
        """
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)

    def load(self, path: str):
        """Load model weights from file.

        Args:
            path: File path to the checkpoint
        """
        checkpoint = torch.load(path, weights_only=True)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def reset(self):
        """Reset agent state between games."""
        self.rollout_buffer.clear()

    def _build_action_mask(self, legal_actions: List[Action]) -> np.ndarray:
        """Build boolean mask from list of legal Action objects.

        Args:
            legal_actions: List of legal game Actions

        Returns:
            Boolean numpy array of shape (ACTION_SPACE_SIZE,)
        """
        from agents.rl.env import (
            _PLAY_CARD_START, _ATTACK_START, _HERO_POWER_INDEX, _END_TURN_INDEX,
            BOARD_POSITIONS, _ATTACK_TARGETS_PER_ATTACKER, PLAY_CARD_SLOTS,
            NUM_ATTACKER_SLOTS,
        )

        mask = np.zeros(ACTION_SPACE_SIZE, dtype=bool)

        for action in legal_actions:
            if action.type == ActionType.END_TURN:
                mask[_END_TURN_INDEX] = True
            elif action.type == ActionType.HERO_POWER:
                mask[_HERO_POWER_INDEX] = True
            elif action.type == ActionType.PLAY_CARD:
                hand_idx = action.card_index or 0
                pos = action.position or 0
                if hand_idx < PLAY_CARD_SLOTS and pos < BOARD_POSITIONS:
                    idx = _PLAY_CARD_START + hand_idx * BOARD_POSITIONS + pos
                    mask[idx] = True
            elif action.type == ActionType.ATTACK:
                attacker = action.attacker_index or 0
                if attacker < NUM_ATTACKER_SLOTS:
                    if action.defender_index is None:
                        target_code = 0
                    else:
                        target_code = action.defender_index + 1
                    if target_code < _ATTACK_TARGETS_PER_ATTACKER:
                        idx = _ATTACK_START + attacker * _ATTACK_TARGETS_PER_ATTACKER + target_code
                        mask[idx] = True

        # END_TURN always valid as fallback
        mask[_END_TURN_INDEX] = True
        return mask

    def _index_to_action(self, action_idx: int, legal_actions: List[Action]) -> Action:
        """Convert integer action index to the closest matching legal Action.

        Args:
            action_idx: Integer action from the network
            legal_actions: List of legal game Actions

        Returns:
            Matching Action from legal_actions, or END_TURN as fallback
        """
        from agents.rl.env import (
            _PLAY_CARD_START, _PLAY_CARD_END, _ATTACK_START, _ATTACK_END,
            _HERO_POWER_INDEX, _END_TURN_INDEX,
            BOARD_POSITIONS, _ATTACK_TARGETS_PER_ATTACKER,
        )

        if action_idx == _END_TURN_INDEX:
            target_type = ActionType.END_TURN
            target_card = None
            target_pos = None
            target_attacker = None
            target_defender = None
        elif action_idx == _HERO_POWER_INDEX:
            target_type = ActionType.HERO_POWER
            target_card = None
            target_pos = None
            target_attacker = None
            target_defender = None
        elif _PLAY_CARD_START <= action_idx < _PLAY_CARD_END:
            idx = action_idx - _PLAY_CARD_START
            target_type = ActionType.PLAY_CARD
            target_card = idx // BOARD_POSITIONS
            target_pos = idx % BOARD_POSITIONS
            target_attacker = None
            target_defender = None
        elif _ATTACK_START <= action_idx < _ATTACK_END:
            idx = action_idx - _ATTACK_START
            target_type = ActionType.ATTACK
            target_card = None
            target_pos = None
            target_attacker = idx // _ATTACK_TARGETS_PER_ATTACKER
            target_code = idx % _ATTACK_TARGETS_PER_ATTACKER
            target_defender = None if target_code == 0 else (target_code - 1)
        else:
            target_type = ActionType.END_TURN
            target_card = None
            target_pos = None
            target_attacker = None
            target_defender = None

        # Find matching legal action
        for action in legal_actions:
            if action.type != target_type:
                continue
            if target_type == ActionType.END_TURN:
                return action
            if target_type == ActionType.HERO_POWER:
                return action
            if target_type == ActionType.PLAY_CARD:
                if action.card_index == target_card and action.position == target_pos:
                    return action
            if target_type == ActionType.ATTACK:
                if (action.attacker_index == target_attacker
                        and action.defender_index == target_defender):
                    return action

        # Fallback: return END_TURN
        for action in legal_actions:
            if action.type == ActionType.END_TURN:
                return action

        return Action(type=ActionType.END_TURN)
