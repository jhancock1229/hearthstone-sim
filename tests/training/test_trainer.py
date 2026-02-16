"""Unit tests for training/trainer.py (Trainer class).

Tests cover:
- Trainer initialization with default and custom configs
- Opponent creation for different config types
- Training loop execution and return values
- Episode collection
- Evaluation
- Checkpoint saving
- Progress callback invocation
"""

import os
import tempfile

import pytest

from training.config import TrainingConfig
from training.trainer import Trainer


# ============================================================
# Trainer Creation
# ============================================================


class TestTrainerCreation:
    """Tests for Trainer initialization."""

    def test_default_config(self):
        """Trainer uses default config when none provided."""
        trainer = Trainer()
        assert trainer.config is not None
        assert isinstance(trainer.config, TrainingConfig)

    def test_custom_config(self):
        """Trainer uses provided config."""
        config = TrainingConfig(total_episodes=5, seed=123)
        trainer = Trainer(config=config)
        assert trainer.config.total_episodes == 5
        assert trainer.config.seed == 123

    def test_agent_created(self):
        """Trainer creates a PPOAgent."""
        trainer = Trainer()
        assert trainer.agent is not None

    def test_env_created(self):
        """Trainer creates an environment."""
        trainer = Trainer()
        assert trainer.env is not None

    def test_initial_state(self):
        """Trainer starts with clean state."""
        trainer = Trainer()
        assert trainer.episode == 0
        assert trainer.total_steps == 0
        assert trainer.episode_rewards == []
        assert trainer.episode_lengths == []
        assert trainer.eval_win_rates == []
        assert trainer.losses_history == []

    def test_progress_callback_stored(self):
        """Progress callback is stored."""
        cb = lambda info: None
        trainer = Trainer(progress_callback=cb)
        assert trainer.progress_callback is cb

    def test_no_callback_by_default(self):
        """No progress callback by default."""
        trainer = Trainer()
        assert trainer.progress_callback is None


# ============================================================
# Opponent Creation
# ============================================================


class TestTrainerOpponent:
    """Tests for opponent creation based on config."""

    def test_random_opponent(self):
        """Random opponent is created for 'random' config."""
        config = TrainingConfig(opponent="random")
        trainer = Trainer(config=config)
        from agents.random_agent import RandomAgent
        assert isinstance(trainer._opponent, RandomAgent)

    def test_greedy_opponent(self):
        """Greedy opponent is created for 'greedy' config."""
        config = TrainingConfig(opponent="greedy")
        trainer = Trainer(config=config)
        from agents.greedy_agent import GreedyAgent
        assert isinstance(trainer._opponent, GreedyAgent)

    def test_self_opponent_is_none(self):
        """'self' opponent returns None (placeholder for self-play)."""
        config = TrainingConfig(opponent="self")
        trainer = Trainer(config=config)
        assert trainer._opponent is None

    def test_unknown_opponent_is_none(self):
        """Unknown opponent type returns None."""
        config = TrainingConfig(opponent="unknown_type")
        trainer = Trainer(config=config)
        assert trainer._opponent is None


# ============================================================
# Training Loop
# ============================================================


class TestTrainerTrain:
    """Tests for the train() method."""

    def test_train_returns_dict(self):
        """train() returns a summary dictionary."""
        config = TrainingConfig(total_episodes=2, eval_interval=1, eval_games=1, save_interval=0)
        trainer = Trainer(config=config)
        result = trainer.train()
        assert isinstance(result, dict)

    def test_train_result_keys(self):
        """Result dict has expected keys."""
        config = TrainingConfig(total_episodes=2, eval_interval=1, eval_games=1, save_interval=0)
        trainer = Trainer(config=config)
        result = trainer.train()
        assert "total_episodes" in result
        assert "total_steps" in result
        assert "elapsed_seconds" in result
        assert "mean_reward" in result
        assert "mean_length" in result
        assert "eval_win_rates" in result
        assert "final_win_rate" in result

    def test_train_tracks_episodes(self):
        """Training records the correct number of episodes."""
        config = TrainingConfig(total_episodes=3, eval_interval=100, save_interval=0)
        trainer = Trainer(config=config)
        trainer.train()
        assert len(trainer.episode_rewards) == 3
        assert len(trainer.episode_lengths) == 3
        assert trainer.episode == 3

    def test_train_total_steps_positive(self):
        """Training accumulates positive total steps."""
        config = TrainingConfig(total_episodes=2, eval_interval=100, save_interval=0)
        trainer = Trainer(config=config)
        result = trainer.train()
        assert result["total_steps"] > 0
        assert trainer.total_steps > 0

    def test_train_elapsed_time(self):
        """Training reports non-negative elapsed time."""
        config = TrainingConfig(total_episodes=1, eval_interval=100, save_interval=0)
        trainer = Trainer(config=config)
        result = trainer.train()
        assert result["elapsed_seconds"] >= 0

    def test_train_mean_reward_is_float(self):
        """Mean reward is a numeric value."""
        config = TrainingConfig(total_episodes=2, eval_interval=100, save_interval=0)
        trainer = Trainer(config=config)
        result = trainer.train()
        assert isinstance(float(result["mean_reward"]), float)

    def test_train_mean_length_is_float(self):
        """Mean length is a numeric value."""
        config = TrainingConfig(total_episodes=2, eval_interval=100, save_interval=0)
        trainer = Trainer(config=config)
        result = trainer.train()
        assert isinstance(float(result["mean_length"]), float)

    def test_train_no_eval_when_interval_exceeds_episodes(self):
        """No evaluations when eval_interval > total_episodes."""
        config = TrainingConfig(total_episodes=3, eval_interval=100, save_interval=0)
        trainer = Trainer(config=config)
        result = trainer.train()
        assert result["eval_win_rates"] == []
        assert result["final_win_rate"] is None


# ============================================================
# Evaluation
# ============================================================


class TestTrainerEvaluation:
    """Tests for the _evaluate() method."""

    def test_evaluate_returns_float(self):
        """_evaluate() returns a float win rate."""
        config = TrainingConfig(eval_games=2)
        trainer = Trainer(config=config)
        win_rate = trainer._evaluate()
        assert isinstance(win_rate, float)

    def test_evaluate_in_range(self):
        """Win rate is between 0.0 and 1.0."""
        config = TrainingConfig(eval_games=5)
        trainer = Trainer(config=config)
        win_rate = trainer._evaluate()
        assert 0.0 <= win_rate <= 1.0

    def test_eval_during_training(self):
        """Evaluations are recorded during training."""
        config = TrainingConfig(total_episodes=4, eval_interval=2, eval_games=2, save_interval=0)
        trainer = Trainer(config=config)
        trainer.train()
        # Episodes 2 and 4 should trigger evaluation
        assert len(trainer.eval_win_rates) == 2


# ============================================================
# Checkpoint Saving
# ============================================================


class TestTrainerCheckpoint:
    """Tests for checkpoint saving."""

    def test_save_checkpoint_creates_file(self):
        """_save_checkpoint() creates a checkpoint file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainingConfig(
                checkpoint_dir=tmpdir,
                experiment_name="test_exp",
            )
            trainer = Trainer(config=config)
            trainer.episode = 10
            trainer._save_checkpoint()
            expected_path = os.path.join(tmpdir, "test_exp_ep10.pt")
            assert os.path.exists(expected_path)

    def test_save_during_training(self):
        """Checkpoints are saved at configured intervals."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainingConfig(
                total_episodes=4,
                save_interval=2,
                eval_interval=100,
                checkpoint_dir=tmpdir,
                experiment_name="test_save",
            )
            trainer = Trainer(config=config)
            trainer.train()
            # Episodes 2 and 4 should trigger saves
            assert os.path.exists(os.path.join(tmpdir, "test_save_ep2.pt"))
            assert os.path.exists(os.path.join(tmpdir, "test_save_ep4.pt"))

    def test_no_save_when_interval_zero(self):
        """No checkpoints when save_interval is 0."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TrainingConfig(
                total_episodes=3,
                save_interval=0,
                eval_interval=100,
                checkpoint_dir=tmpdir,
            )
            trainer = Trainer(config=config)
            trainer.train()
            files = os.listdir(tmpdir)
            assert len(files) == 0

    def test_checkpoint_dir_created(self):
        """Checkpoint directory is created if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nested = os.path.join(tmpdir, "sub", "dir")
            config = TrainingConfig(
                checkpoint_dir=nested,
                experiment_name="test_nested",
            )
            trainer = Trainer(config=config)
            trainer.episode = 1
            trainer._save_checkpoint()
            assert os.path.isdir(nested)


# ============================================================
# Progress Callback
# ============================================================


class TestTrainerCallback:
    """Tests for progress callback invocation."""

    def test_callback_called_each_episode(self):
        """Callback is called once per episode."""
        calls = []
        config = TrainingConfig(total_episodes=3, eval_interval=100, save_interval=0)
        trainer = Trainer(config=config, progress_callback=lambda info: calls.append(info))
        trainer.train()
        assert len(calls) == 3

    def test_callback_info_has_episode(self):
        """Callback info dict has episode number."""
        calls = []
        config = TrainingConfig(total_episodes=1, eval_interval=100, save_interval=0)
        trainer = Trainer(config=config, progress_callback=lambda info: calls.append(info))
        trainer.train()
        assert calls[0]["episode"] == 1

    def test_callback_info_has_reward(self):
        """Callback info dict has reward."""
        calls = []
        config = TrainingConfig(total_episodes=1, eval_interval=100, save_interval=0)
        trainer = Trainer(config=config, progress_callback=lambda info: calls.append(info))
        trainer.train()
        assert "reward" in calls[0]
        assert isinstance(calls[0]["reward"], float)

    def test_callback_info_has_length(self):
        """Callback info dict has episode length."""
        calls = []
        config = TrainingConfig(total_episodes=1, eval_interval=100, save_interval=0)
        trainer = Trainer(config=config, progress_callback=lambda info: calls.append(info))
        trainer.train()
        assert "length" in calls[0]
        assert isinstance(calls[0]["length"], int)

    def test_callback_info_has_total_steps(self):
        """Callback info dict has total_steps."""
        calls = []
        config = TrainingConfig(total_episodes=1, eval_interval=100, save_interval=0)
        trainer = Trainer(config=config, progress_callback=lambda info: calls.append(info))
        trainer.train()
        assert "total_steps" in calls[0]
        assert calls[0]["total_steps"] > 0

    def test_no_callback_no_error(self):
        """Training works fine without a callback."""
        config = TrainingConfig(total_episodes=1, eval_interval=100, save_interval=0)
        trainer = Trainer(config=config)
        result = trainer.train()
        assert result["total_episodes"] == 1


# ============================================================
# Episode Collection
# ============================================================


class TestTrainerEpisodeCollection:
    """Tests for _collect_episode()."""

    def test_collect_returns_tuple(self):
        """_collect_episode() returns (reward, length)."""
        trainer = Trainer()
        result = trainer._collect_episode()
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_collect_reward_is_float(self):
        """Episode reward is a float."""
        trainer = Trainer()
        reward, _ = trainer._collect_episode()
        assert isinstance(reward, float)

    def test_collect_length_is_positive(self):
        """Episode length is positive."""
        trainer = Trainer()
        _, length = trainer._collect_episode()
        assert length > 0

    def test_collect_increments_total_steps(self):
        """Collecting an episode increments total_steps."""
        trainer = Trainer()
        assert trainer.total_steps == 0
        _, length = trainer._collect_episode()
        assert trainer.total_steps == length

    def test_collect_populates_rollout_buffer(self):
        """Episode collection adds to rollout buffer."""
        trainer = Trainer()
        trainer._collect_episode()
        assert len(trainer.agent.rollout_buffer) > 0
