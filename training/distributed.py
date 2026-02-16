"""Distributed training: multi-process game generation with centralized learner.

Provides parallel episode collection and a centralized PPO update loop.
Uses a simple worker model where each worker generates complete episodes
which are fed into a shared experience buffer for training.

Components:
- ExperienceBuffer: Bounded replay buffer for collected transitions
- GameWorker: Generates game episodes using the RL environment
- DistributedTrainer: Orchestrates parallel collection + centralized updates

Usage:
    from training.distributed import DistributedTrainer
    from training.config import TrainingConfig

    config = TrainingConfig(total_episodes=1000)
    trainer = DistributedTrainer(num_workers=4, config=config)
    result = trainer.run()
"""

import random
import time
from collections import deque
from typing import Any, Dict, List, Optional

import numpy as np

from agents.rl.env import HearthstoneEnv
from agents.rl.ppo_agent import PPOAgent
from training.config import TrainingConfig


class ExperienceBuffer:
    """Bounded replay buffer for collected experience transitions.

    Stores experience dicts and evicts oldest entries when capacity
    is exceeded. Supports random sampling for training batches.
    """

    def __init__(self, capacity: int = 10000):
        """Initialize experience buffer.

        Args:
            capacity: Maximum number of transitions to store
        """
        self.capacity = capacity
        self._buffer: deque = deque(maxlen=capacity)

    def add(self, experience: Dict[str, Any]):
        """Add a single experience transition.

        Args:
            experience: Dict with observation, action, reward, etc.
        """
        self._buffer.append(experience)

    def add_batch(self, experiences: List[Dict[str, Any]]):
        """Add a batch of experiences.

        Args:
            experiences: List of experience dicts
        """
        for exp in experiences:
            self._buffer.append(exp)

    def sample(self, n: int) -> List[Dict[str, Any]]:
        """Sample n random experiences from the buffer.

        If n > len(buffer), returns all available experiences.

        Args:
            n: Number of experiences to sample

        Returns:
            List of experience dicts
        """
        if len(self._buffer) == 0:
            return []
        n = min(n, len(self._buffer))
        indices = random.sample(range(len(self._buffer)), n)
        return [self._buffer[i] for i in indices]

    def get_all(self) -> List[Dict[str, Any]]:
        """Get all experiences in order.

        Returns:
            List of all experience dicts
        """
        return list(self._buffer)

    def clear(self):
        """Remove all experiences."""
        self._buffer.clear()

    def __len__(self) -> int:
        return len(self._buffer)

    @property
    def is_empty(self) -> bool:
        """True if buffer has no experiences."""
        return len(self._buffer) == 0


class GameWorker:
    """Generates game episodes using the RL environment.

    Each worker has its own environment instance and seed for
    reproducible, independent episode generation.
    """

    def __init__(
        self,
        worker_id: int = 0,
        seed: int = 42,
        max_turns: int = 200,
        opponent: str = "random",
    ):
        """Initialize game worker.

        Args:
            worker_id: Unique worker identifier
            seed: Base random seed
            max_turns: Maximum turns per game
            opponent: Opponent type
        """
        self.worker_id = worker_id
        self.seed = seed
        self.effective_seed = seed + worker_id * 1000
        self.max_turns = max_turns
        self.opponent = opponent

        # Create environment with worker-specific seed
        self._env = HearthstoneEnv(
            seed=self.effective_seed,
            max_turns=max_turns,
        )

        # Lightweight agent for action selection during collection
        self._agent = PPOAgent()

    def generate_episode(self) -> List[Dict[str, Any]]:
        """Generate one complete episode.

        Returns:
            List of experience dicts, one per step.
            Each dict has: observation, action, reward, value,
            log_prob, done, action_mask
        """
        episode = []
        obs, info = self._env.reset()
        done = False

        while not done:
            mask = info["action_mask"]
            action_idx, value, log_prob = self._agent.get_action_and_value(obs, mask)

            next_obs, reward, terminated, truncated, info = self._env.step(action_idx)
            done = terminated or truncated

            episode.append({
                "observation": obs,
                "action": action_idx,
                "reward": reward if done else 0.0,
                "value": value,
                "log_prob": log_prob,
                "done": done,
                "action_mask": mask,
            })

            obs = next_obs

        return episode


class DistributedTrainer:
    """Orchestrates parallel episode collection with centralized PPO updates.

    Workers generate episodes in sequence (or parallel via multiprocessing),
    transitions are collected into a shared buffer, and the centralized
    agent performs PPO updates on the collected data.
    """

    def __init__(
        self,
        num_workers: int = 2,
        config: Optional[TrainingConfig] = None,
        buffer_capacity: int = 50000,
    ):
        """Initialize distributed trainer.

        Args:
            num_workers: Number of game workers
            config: Training configuration
            buffer_capacity: Experience buffer capacity
        """
        self.num_workers = num_workers
        self.config = config or TrainingConfig()
        self.buffer = ExperienceBuffer(capacity=buffer_capacity)

        # Centralized PPO agent
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

        # Create workers
        self.workers = [
            GameWorker(
                worker_id=i,
                seed=self.config.seed,
                max_turns=self.config.max_turns,
                opponent=self.config.opponent,
            )
            for i in range(num_workers)
        ]

    def collect_episodes(self, num_episodes: int) -> List[List[Dict[str, Any]]]:
        """Collect episodes from workers.

        Distributes episode generation across workers round-robin.

        Args:
            num_episodes: Total episodes to collect

        Returns:
            List of episodes (each episode is a list of experience dicts)
        """
        episodes = []
        for i in range(num_episodes):
            worker = self.workers[i % self.num_workers]
            # Sync worker agent weights with centralized agent
            worker._agent.network.load_state_dict(
                self.agent.network.state_dict()
            )
            episode = worker.generate_episode()
            episodes.append(episode)

            # Add to buffer
            self.buffer.add_batch(episode)

        return episodes

    def train_step(self) -> Dict[str, float]:
        """Perform a PPO update on collected buffer data.

        Loads buffer data into the agent's rollout buffer and runs update.
        Clears the buffer after training.

        Returns:
            Dict with loss components, or empty dict if buffer is empty
        """
        if self.buffer.is_empty:
            return {}

        # Transfer buffer data to agent's rollout buffer
        self.agent.rollout_buffer.clear()
        for exp in self.buffer.get_all():
            self.agent.rollout_buffer.add(
                observation=exp["observation"],
                action=exp["action"],
                reward=exp["reward"],
                value=exp["value"],
                log_prob=exp["log_prob"],
                done=exp["done"],
                action_mask=exp["action_mask"],
            )

        losses = self.agent.update()
        self.buffer.clear()
        return losses

    def run(self) -> Dict[str, Any]:
        """Run the full distributed training loop.

        Returns:
            Dict with training summary: total_episodes, elapsed_seconds,
            mean_reward, mean_length
        """
        start_time = time.time()
        total_episodes = self.config.total_episodes
        episodes_per_batch = max(1, self.num_workers)

        all_rewards = []
        all_lengths = []
        completed = 0

        while completed < total_episodes:
            batch_size = min(episodes_per_batch, total_episodes - completed)
            episodes = self.collect_episodes(batch_size)

            for episode in episodes:
                terminal_reward = episode[-1]["reward"] if episode else 0.0
                all_rewards.append(terminal_reward)
                all_lengths.append(len(episode))

            # Train on collected data
            if not self.buffer.is_empty:
                self.train_step()

            completed += batch_size

        elapsed = time.time() - start_time

        return {
            "total_episodes": completed,
            "elapsed_seconds": elapsed,
            "mean_reward": float(np.mean(all_rewards)) if all_rewards else 0.0,
            "mean_length": float(np.mean(all_lengths)) if all_lengths else 0.0,
        }
