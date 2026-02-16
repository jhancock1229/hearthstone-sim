"""Training configuration for RL experiments.

Centralizes all hyperparameters and settings for reproducible training runs.
Can be constructed programmatically or loaded from a dict/YAML.

Usage:
    config = TrainingConfig()                    # defaults
    config = TrainingConfig(learning_rate=1e-4)  # override
    config = TrainingConfig.from_dict(yaml_data) # from file
"""

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class TrainingConfig:
    """All settings for a training run.

    Attributes:
        # PPO hyperparameters
        learning_rate: Adam learning rate
        gamma: Discount factor
        gae_lambda: GAE lambda (bias-variance tradeoff)
        clip_epsilon: PPO clipping range
        value_loss_coef: Weight for value loss
        entropy_coef: Weight for entropy bonus
        ppo_epochs: Number of PPO epochs per update
        max_grad_norm: Gradient clipping threshold

        # Network architecture
        hidden_sizes: Hidden layer sizes for actor-critic backbone

        # Training loop
        total_episodes: Total episodes to train
        rollout_steps: Steps per rollout before update (0 = full episode)
        eval_interval: Evaluate every N episodes
        eval_games: Number of games per evaluation
        save_interval: Save checkpoint every N episodes
        max_turns: Max turns per game (truncation limit)

        # Environment
        seed: Random seed for reproducibility
        opponent: Opponent type ("random", "greedy", "self")

        # Paths
        checkpoint_dir: Directory for model checkpoints
        log_dir: Directory for training logs
        experiment_name: Name for this experiment run
    """
    # PPO hyperparameters
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    ppo_epochs: int = 4
    max_grad_norm: float = 0.5

    # Network architecture
    hidden_sizes: List[int] = field(default_factory=lambda: [128, 64])

    # Training loop
    total_episodes: int = 10000
    rollout_steps: int = 0  # 0 = full episode
    eval_interval: int = 100
    eval_games: int = 50
    save_interval: int = 500
    max_turns: int = 200

    # Environment
    seed: int = 42
    opponent: str = "random"

    # Paths
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    experiment_name: str = "ppo_default"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize config to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingConfig":
        """Create config from dictionary.

        Unknown keys are silently ignored so configs can include
        extra metadata fields.
        """
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)
