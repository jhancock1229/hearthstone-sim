"""Unit tests for training/config.py (TrainingConfig).

Tests cover:
- Default values
- Custom construction
- to_dict() serialization
- from_dict() deserialization
- Round-trip serialization
- Unknown keys handling
"""

import pytest

from training.config import TrainingConfig


# ============================================================
# Default Values
# ============================================================


class TestTrainingConfigDefaults:
    """Tests for default configuration values."""

    def test_default_learning_rate(self):
        config = TrainingConfig()
        assert config.learning_rate == 3e-4

    def test_default_gamma(self):
        config = TrainingConfig()
        assert config.gamma == 0.99

    def test_default_total_episodes(self):
        config = TrainingConfig()
        assert config.total_episodes == 10000

    def test_default_seed(self):
        config = TrainingConfig()
        assert config.seed == 42

    def test_default_opponent(self):
        config = TrainingConfig()
        assert config.opponent == "random"

    def test_default_hidden_sizes(self):
        config = TrainingConfig()
        assert config.hidden_sizes == [128, 64]

    def test_default_checkpoint_dir(self):
        config = TrainingConfig()
        assert config.checkpoint_dir == "checkpoints"

    def test_default_experiment_name(self):
        config = TrainingConfig()
        assert config.experiment_name == "ppo_default"


# ============================================================
# Custom Construction
# ============================================================


class TestTrainingConfigCustom:
    """Tests for custom configuration values."""

    def test_custom_learning_rate(self):
        config = TrainingConfig(learning_rate=1e-4)
        assert config.learning_rate == 1e-4

    def test_custom_total_episodes(self):
        config = TrainingConfig(total_episodes=500)
        assert config.total_episodes == 500

    def test_custom_hidden_sizes(self):
        config = TrainingConfig(hidden_sizes=[256, 128, 64])
        assert config.hidden_sizes == [256, 128, 64]

    def test_custom_opponent(self):
        config = TrainingConfig(opponent="greedy")
        assert config.opponent == "greedy"

    def test_multiple_overrides(self):
        config = TrainingConfig(
            learning_rate=1e-5,
            gamma=0.95,
            seed=99,
            experiment_name="test_run",
        )
        assert config.learning_rate == 1e-5
        assert config.gamma == 0.95
        assert config.seed == 99
        assert config.experiment_name == "test_run"


# ============================================================
# to_dict() Serialization
# ============================================================


class TestTrainingConfigToDict:
    """Tests for to_dict() serialization."""

    def test_to_dict_returns_dict(self):
        config = TrainingConfig()
        result = config.to_dict()
        assert isinstance(result, dict)

    def test_to_dict_has_all_fields(self):
        config = TrainingConfig()
        d = config.to_dict()
        assert "learning_rate" in d
        assert "gamma" in d
        assert "total_episodes" in d
        assert "seed" in d
        assert "opponent" in d
        assert "hidden_sizes" in d
        assert "checkpoint_dir" in d
        assert "experiment_name" in d

    def test_to_dict_preserves_values(self):
        config = TrainingConfig(learning_rate=1e-5, seed=99)
        d = config.to_dict()
        assert d["learning_rate"] == 1e-5
        assert d["seed"] == 99

    def test_to_dict_hidden_sizes_is_list(self):
        config = TrainingConfig(hidden_sizes=[256, 128])
        d = config.to_dict()
        assert d["hidden_sizes"] == [256, 128]

    def test_to_dict_all_ppo_params(self):
        config = TrainingConfig()
        d = config.to_dict()
        ppo_keys = [
            "learning_rate", "gamma", "gae_lambda", "clip_epsilon",
            "value_loss_coef", "entropy_coef", "ppo_epochs", "max_grad_norm",
        ]
        for key in ppo_keys:
            assert key in d


# ============================================================
# from_dict() Deserialization
# ============================================================


class TestTrainingConfigFromDict:
    """Tests for from_dict() deserialization."""

    def test_from_dict_creates_config(self):
        data = {"learning_rate": 1e-4, "seed": 99}
        config = TrainingConfig.from_dict(data)
        assert isinstance(config, TrainingConfig)
        assert config.learning_rate == 1e-4
        assert config.seed == 99

    def test_from_dict_uses_defaults_for_missing(self):
        data = {"seed": 77}
        config = TrainingConfig.from_dict(data)
        assert config.seed == 77
        assert config.learning_rate == 3e-4  # default
        assert config.total_episodes == 10000  # default

    def test_from_dict_ignores_unknown_keys(self):
        data = {
            "seed": 77,
            "unknown_field": "should_be_ignored",
            "another_extra": 42,
        }
        config = TrainingConfig.from_dict(data)
        assert config.seed == 77
        assert not hasattr(config, "unknown_field")

    def test_from_dict_empty_dict(self):
        config = TrainingConfig.from_dict({})
        # Should use all defaults
        assert config.learning_rate == 3e-4
        assert config.total_episodes == 10000

    def test_from_dict_all_fields(self):
        data = {
            "learning_rate": 1e-5,
            "gamma": 0.9,
            "gae_lambda": 0.8,
            "clip_epsilon": 0.1,
            "value_loss_coef": 0.25,
            "entropy_coef": 0.05,
            "ppo_epochs": 8,
            "max_grad_norm": 1.0,
            "hidden_sizes": [64, 32],
            "total_episodes": 500,
            "rollout_steps": 128,
            "eval_interval": 50,
            "eval_games": 10,
            "save_interval": 100,
            "max_turns": 100,
            "seed": 7,
            "opponent": "greedy",
            "checkpoint_dir": "/tmp/ckpts",
            "log_dir": "/tmp/logs",
            "experiment_name": "my_exp",
        }
        config = TrainingConfig.from_dict(data)
        for key, value in data.items():
            assert getattr(config, key) == value


# ============================================================
# Round-trip Serialization
# ============================================================


class TestTrainingConfigRoundTrip:
    """Tests for to_dict -> from_dict round-trip."""

    def test_round_trip_default(self):
        original = TrainingConfig()
        restored = TrainingConfig.from_dict(original.to_dict())
        assert original.to_dict() == restored.to_dict()

    def test_round_trip_custom(self):
        original = TrainingConfig(
            learning_rate=1e-5,
            hidden_sizes=[256, 128, 64],
            seed=99,
            experiment_name="round_trip_test",
        )
        restored = TrainingConfig.from_dict(original.to_dict())
        assert original.to_dict() == restored.to_dict()

    def test_round_trip_preserves_types(self):
        original = TrainingConfig()
        d = original.to_dict()
        restored = TrainingConfig.from_dict(d)
        assert type(restored.learning_rate) == float
        assert type(restored.total_episodes) == int
        assert type(restored.hidden_sizes) == list
        assert type(restored.opponent) == str
