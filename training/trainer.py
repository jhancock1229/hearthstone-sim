"""Training loop for PPO agent.

Orchestrates the training process:
1. Collect rollout episodes from the environment
2. Compute advantages (GAE)
3. Run PPO updates
4. Periodically evaluate against baseline opponents
5. Save checkpoints

Usage:
    from training.config import TrainingConfig
    from training.trainer import Trainer

    config = TrainingConfig(total_episodes=1000)
    trainer = Trainer(config)
    trainer.train()
"""

import os
import time
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from agents.rl.env import HearthstoneEnv
from agents.rl.ppo_agent import PPOAgent
from agents.random_agent import RandomAgent
from analysis.metrics import EpisodeMetrics, compute_win_rate
from training.config import TrainingConfig


class Trainer:
    """Manages the PPO training loop.

    Handles episode collection, PPO updates, evaluation, checkpointing,
    and logging.
    """

    def __init__(
        self,
        config: Optional[TrainingConfig] = None,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        """Initialize trainer.

        Args:
            config: Training configuration (uses defaults if None)
            progress_callback: Optional callback(info_dict) called after each episode
        """
        self.config = config or TrainingConfig()
        self.progress_callback = progress_callback

        # Create agent
        self.agent = PPOAgent(
            learning_rate=self.config.learning_rate,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
            clip_epsilon=self.config.clip_epsilon,
            value_loss_coef=self.config.value_loss_coef,
            entropy_coef=self.config.entropy_coef,
            ppo_epochs=self.config.ppo_epochs,
            max_grad_norm=self.config.max_grad_norm,
            hidden_sizes=self.config.hidden_sizes,
        )

        # Create opponent for environment
        self._opponent = self._create_opponent()

        # Create environment
        self.env = HearthstoneEnv(
            seed=self.config.seed,
            max_turns=self.config.max_turns,
            opponent_agent=self._opponent,
        )

        # Training state
        self.episode = 0
        self.total_steps = 0
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.eval_win_rates: List[float] = []
        self.losses_history: List[Dict[str, float]] = []

    def _create_opponent(self):
        """Create opponent agent based on config."""
        if self.config.opponent == "random":
            return RandomAgent(seed=self.config.seed + 1000)
        elif self.config.opponent == "greedy":
            from agents.greedy_agent import GreedyAgent
            return GreedyAgent()
        elif self.config.opponent == "self":
            return None  # Will use random for now; self-play is Phase 7
        return None

    def train(self) -> Dict[str, Any]:
        """Run the full training loop.

        Returns:
            Dict with training summary statistics
        """
        start_time = time.time()

        for ep in range(self.config.total_episodes):
            self.episode = ep + 1

            # Collect one episode
            episode_reward, episode_length = self._collect_episode()
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)

            # PPO update after each episode
            if len(self.agent.rollout_buffer) > 0:
                losses = self.agent.update()
                self.losses_history.append(losses)
            else:
                losses = {}

            # Evaluation
            if self.episode % self.config.eval_interval == 0:
                win_rate = self._evaluate()
                self.eval_win_rates.append(win_rate)

            # Checkpoint
            if self.config.save_interval > 0 and self.episode % self.config.save_interval == 0:
                self._save_checkpoint()

            # Progress callback
            if self.progress_callback:
                self.progress_callback({
                    'episode': self.episode,
                    'reward': episode_reward,
                    'length': episode_length,
                    'losses': losses,
                    'total_steps': self.total_steps,
                })

        elapsed = time.time() - start_time

        return {
            'total_episodes': self.config.total_episodes,
            'total_steps': self.total_steps,
            'elapsed_seconds': elapsed,
            'mean_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0.0,
            'mean_length': np.mean(self.episode_lengths) if self.episode_lengths else 0.0,
            'eval_win_rates': self.eval_win_rates,
            'final_win_rate': self.eval_win_rates[-1] if self.eval_win_rates else None,
        }

    def _collect_episode(self) -> tuple:
        """Collect one full episode of experience.

        Returns:
            (total_reward, episode_length)
        """
        obs, info = self.env.reset()
        done = False
        total_reward = 0.0
        steps = 0

        while not done:
            mask = info['action_mask']
            action_idx, value, log_prob = self.agent.get_action_and_value(obs, mask)

            next_obs, reward, terminated, truncated, info = self.env.step(action_idx)
            done = terminated or truncated

            self.agent.rollout_buffer.add(
                observation=obs,
                action=action_idx,
                reward=reward,
                value=value,
                log_prob=log_prob,
                done=done,
                action_mask=mask,
            )

            obs = next_obs
            total_reward += reward
            steps += 1
            self.total_steps += 1

        return total_reward, steps

    def _evaluate(self) -> float:
        """Evaluate agent against random opponent.

        Returns:
            Win rate (0.0 to 1.0)
        """
        wins = 0
        eval_env = HearthstoneEnv(
            seed=self.config.seed + 9999,
            max_turns=self.config.max_turns,
        )

        for game in range(self.config.eval_games):
            obs, info = eval_env.reset()
            done = False

            while not done:
                mask = info['action_mask']
                action_idx, _, _ = self.agent.get_action_and_value(obs, mask)
                obs, reward, terminated, truncated, info = eval_env.step(action_idx)
                done = terminated or truncated

            if reward > 0:
                wins += 1

        win_rate = wins / max(self.config.eval_games, 1)
        return win_rate

    def _save_checkpoint(self):
        """Save model checkpoint."""
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        path = os.path.join(
            self.config.checkpoint_dir,
            f"{self.config.experiment_name}_ep{self.episode}.pt",
        )
        self.agent.save(path)
