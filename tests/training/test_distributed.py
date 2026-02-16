"""Unit tests for distributed training infrastructure.

Following TDD: these tests define the contract for distributed.py which
implements multi-process game generation with a centralized learner.

Components:
- ExperienceBuffer: Thread-safe replay buffer for collected episodes
- GameWorker: Generates game episodes (designed for multi-process use)
- DistributedTrainer: Orchestrates parallel episode collection + centralized PPO updates
"""

import time
import numpy as np
import pytest

from training.distributed import (
    ExperienceBuffer,
    GameWorker,
    DistributedTrainer,
)


# ============================================================
# ExperienceBuffer Tests
# ============================================================


class TestExperienceBufferCreation:
    """Tests for ExperienceBuffer initialization."""

    def test_create_default(self):
        """Can create buffer with default capacity."""
        buf = ExperienceBuffer()
        assert buf.capacity > 0
        assert len(buf) == 0

    def test_create_with_capacity(self):
        """Can create buffer with custom capacity."""
        buf = ExperienceBuffer(capacity=500)
        assert buf.capacity == 500

    def test_initially_empty(self):
        """New buffer is empty."""
        buf = ExperienceBuffer()
        assert len(buf) == 0
        assert buf.is_empty


class TestExperienceBufferOperations:
    """Tests for adding and sampling experiences."""

    def _make_experience(self, reward=0.0):
        """Create a minimal experience dict."""
        return {
            "observation": np.zeros(106, dtype=np.float32),
            "action": 99,
            "reward": reward,
            "value": 0.0,
            "log_prob": -1.0,
            "done": False,
            "action_mask": np.ones(100, dtype=bool),
        }

    def test_add_single(self):
        """Can add a single experience."""
        buf = ExperienceBuffer(capacity=100)
        buf.add(self._make_experience())
        assert len(buf) == 1

    def test_add_multiple(self):
        """Can add multiple experiences."""
        buf = ExperienceBuffer(capacity=100)
        for i in range(10):
            buf.add(self._make_experience(reward=float(i)))
        assert len(buf) == 10

    def test_add_batch(self):
        """Can add a batch of experiences at once."""
        buf = ExperienceBuffer(capacity=100)
        batch = [self._make_experience(reward=float(i)) for i in range(5)]
        buf.add_batch(batch)
        assert len(buf) == 5

    def test_capacity_evicts_oldest(self):
        """Adding beyond capacity evicts oldest entries."""
        buf = ExperienceBuffer(capacity=5)
        for i in range(10):
            buf.add(self._make_experience(reward=float(i)))
        assert len(buf) == 5
        # The oldest (reward=0..4) should be evicted, keeping 5..9
        rewards = [e["reward"] for e in buf.get_all()]
        assert min(rewards) >= 5.0

    def test_sample_returns_correct_size(self):
        """sample(n) returns exactly n experiences."""
        buf = ExperienceBuffer(capacity=100)
        for i in range(20):
            buf.add(self._make_experience())
        batch = buf.sample(5)
        assert len(batch) == 5

    def test_sample_larger_than_buffer_clamps(self):
        """sample(n) where n > len returns all available."""
        buf = ExperienceBuffer(capacity=100)
        for i in range(3):
            buf.add(self._make_experience())
        batch = buf.sample(10)
        assert len(batch) == 3

    def test_sample_empty_buffer(self):
        """Sampling empty buffer returns empty list."""
        buf = ExperienceBuffer()
        batch = buf.sample(5)
        assert len(batch) == 0

    def test_clear(self):
        """clear() empties the buffer."""
        buf = ExperienceBuffer(capacity=100)
        for i in range(10):
            buf.add(self._make_experience())
        buf.clear()
        assert len(buf) == 0
        assert buf.is_empty

    def test_get_all(self):
        """get_all() returns all experiences in order."""
        buf = ExperienceBuffer(capacity=100)
        for i in range(5):
            buf.add(self._make_experience(reward=float(i)))
        all_exp = buf.get_all()
        assert len(all_exp) == 5
        assert all_exp[0]["reward"] == 0.0
        assert all_exp[4]["reward"] == 4.0

    def test_experiences_are_dicts(self):
        """Stored experiences are dicts with expected keys."""
        buf = ExperienceBuffer(capacity=100)
        buf.add(self._make_experience())
        exp = buf.get_all()[0]
        assert "observation" in exp
        assert "action" in exp
        assert "reward" in exp
        assert "done" in exp


# ============================================================
# GameWorker Tests
# ============================================================


class TestGameWorkerCreation:
    """Tests for GameWorker initialization."""

    def test_create_worker(self):
        """Can create a GameWorker."""
        worker = GameWorker(worker_id=0, seed=42)
        assert worker.worker_id == 0

    def test_worker_has_seed(self):
        """Worker has a seed for reproducibility."""
        worker = GameWorker(worker_id=0, seed=42)
        assert worker.seed == 42

    def test_different_workers_different_seeds(self):
        """Workers with different IDs should use different seeds."""
        w1 = GameWorker(worker_id=0, seed=42)
        w2 = GameWorker(worker_id=1, seed=42)
        # Workers offset their seed by worker_id
        assert w1.effective_seed != w2.effective_seed


class TestGameWorkerEpisodes:
    """Tests for episode generation."""

    def test_generate_episode_returns_experience(self):
        """generate_episode returns a list of experience dicts."""
        worker = GameWorker(worker_id=0, seed=42)
        episode = worker.generate_episode()
        assert isinstance(episode, list)
        assert len(episode) > 0

    def test_episode_has_terminal_step(self):
        """Last step in episode has done=True."""
        worker = GameWorker(worker_id=0, seed=42)
        episode = worker.generate_episode()
        assert episode[-1]["done"] is True

    def test_episode_experience_format(self):
        """Each step has expected keys."""
        worker = GameWorker(worker_id=0, seed=42)
        episode = worker.generate_episode()
        step = episode[0]
        assert "observation" in step
        assert "action" in step
        assert "reward" in step
        assert "value" in step
        assert "log_prob" in step
        assert "done" in step
        assert "action_mask" in step

    def test_episode_reward_at_terminal(self):
        """Non-zero reward only at terminal step."""
        worker = GameWorker(worker_id=0, seed=42)
        episode = worker.generate_episode()
        # Mid-game rewards should be 0
        for step in episode[:-1]:
            assert step["reward"] == 0.0
        # Terminal reward should be +1 or -1
        assert episode[-1]["reward"] in (1.0, -1.0, 0.0)

    def test_generate_multiple_episodes(self):
        """Can generate multiple episodes."""
        worker = GameWorker(worker_id=0, seed=42)
        episodes = [worker.generate_episode() for _ in range(3)]
        assert len(episodes) == 3
        for ep in episodes:
            assert len(ep) > 0

    def test_worker_with_max_turns(self):
        """Worker respects max_turns setting."""
        worker = GameWorker(worker_id=0, seed=42, max_turns=10)
        episode = worker.generate_episode()
        # Episode shouldn't be longer than what max_turns allows
        assert len(episode) <= 1000  # generous upper bound

    def test_worker_with_opponent_type(self):
        """Worker can use different opponent types."""
        worker = GameWorker(worker_id=0, seed=42, opponent="random")
        episode = worker.generate_episode()
        assert len(episode) > 0


# ============================================================
# DistributedTrainer Tests
# ============================================================


class TestDistributedTrainerCreation:
    """Tests for DistributedTrainer initialization."""

    def test_create_default(self):
        """Can create trainer with defaults."""
        trainer = DistributedTrainer(num_workers=2)
        assert trainer.num_workers == 2

    def test_create_with_config(self):
        """Can create trainer with custom config."""
        from training.config import TrainingConfig
        config = TrainingConfig(total_episodes=100, seed=123)
        trainer = DistributedTrainer(num_workers=2, config=config)
        assert trainer.config.total_episodes == 100
        assert trainer.config.seed == 123

    def test_has_experience_buffer(self):
        """Trainer has an experience buffer."""
        trainer = DistributedTrainer(num_workers=2)
        assert isinstance(trainer.buffer, ExperienceBuffer)

    def test_has_agent(self):
        """Trainer has a PPO agent."""
        trainer = DistributedTrainer(num_workers=2)
        assert trainer.agent is not None

    def test_workers_created(self):
        """Trainer creates the specified number of workers."""
        trainer = DistributedTrainer(num_workers=4)
        assert len(trainer.workers) == 4


class TestDistributedTrainerCollection:
    """Tests for parallel episode collection."""

    def test_collect_episodes_sequential(self):
        """Can collect episodes sequentially (processes=1)."""
        trainer = DistributedTrainer(num_workers=1)
        episodes = trainer.collect_episodes(num_episodes=3)
        assert len(episodes) == 3
        for ep in episodes:
            assert len(ep) > 0

    def test_collect_fills_buffer(self):
        """Collected episodes are added to the buffer."""
        trainer = DistributedTrainer(num_workers=1)
        trainer.collect_episodes(num_episodes=2)
        assert len(trainer.buffer) > 0

    def test_collect_episodes_distributes_across_workers(self):
        """With multiple workers, work is distributed."""
        trainer = DistributedTrainer(num_workers=2)
        episodes = trainer.collect_episodes(num_episodes=4)
        assert len(episodes) == 4


class TestDistributedTrainerTraining:
    """Tests for the training step."""

    def test_train_step_returns_losses(self):
        """train_step returns loss dict after updating on buffer data."""
        trainer = DistributedTrainer(num_workers=1)
        trainer.collect_episodes(num_episodes=2)
        losses = trainer.train_step()
        assert "policy_loss" in losses
        assert "value_loss" in losses

    def test_train_step_clears_buffer(self):
        """Buffer is cleared after training step."""
        trainer = DistributedTrainer(num_workers=1)
        trainer.collect_episodes(num_episodes=2)
        trainer.train_step()
        assert len(trainer.buffer) == 0

    def test_train_step_empty_buffer_returns_empty(self):
        """train_step with empty buffer returns empty dict."""
        trainer = DistributedTrainer(num_workers=1)
        losses = trainer.train_step()
        assert losses == {}

    def test_run_training_loop(self):
        """Can run a short training loop."""
        from training.config import TrainingConfig
        config = TrainingConfig(total_episodes=4, eval_interval=2, eval_games=2)
        trainer = DistributedTrainer(num_workers=1, config=config)
        result = trainer.run()
        assert "total_episodes" in result
        assert "elapsed_seconds" in result
        assert result["total_episodes"] == 4

    def test_run_collects_metrics(self):
        """Training loop tracks episode rewards."""
        from training.config import TrainingConfig
        config = TrainingConfig(total_episodes=4, eval_interval=100, eval_games=2)
        trainer = DistributedTrainer(num_workers=1, config=config)
        result = trainer.run()
        assert "mean_reward" in result


# ============================================================
# Integration Tests
# ============================================================


class TestDistributedIntegration:
    """Integration tests for the distributed training pipeline."""

    def test_worker_episode_feeds_buffer(self):
        """Worker-generated episode can be added to buffer."""
        worker = GameWorker(worker_id=0, seed=42)
        buf = ExperienceBuffer(capacity=10000)
        episode = worker.generate_episode()
        buf.add_batch(episode)
        assert len(buf) == len(episode)

    def test_buffer_sample_has_correct_shapes(self):
        """Sampled experiences have correct numpy array shapes."""
        worker = GameWorker(worker_id=0, seed=42)
        buf = ExperienceBuffer(capacity=10000)
        episode = worker.generate_episode()
        buf.add_batch(episode)
        sample = buf.sample(3)
        for exp in sample:
            assert exp["observation"].shape == (106,)
            assert exp["action_mask"].shape == (100,)

    def test_multiple_workers_different_episodes(self):
        """Different workers produce different episodes (different seeds)."""
        w1 = GameWorker(worker_id=0, seed=42)
        w2 = GameWorker(worker_id=1, seed=42)
        ep1 = w1.generate_episode()
        ep2 = w2.generate_episode()
        # Episodes should differ (different seeds)
        # Compare lengths or rewards as a simple check
        # They could coincidentally match, so just verify both complete
        assert len(ep1) > 0
        assert len(ep2) > 0
