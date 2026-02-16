"""Launch a PPO training run.

Usage:
    python scripts/train.py                              # default config
    python scripts/train.py --episodes 5000 --lr 1e-4    # override params
    python scripts/train.py --opponent greedy             # train vs greedy
    python scripts/train.py --name my_experiment          # custom name
"""

import argparse
import sys

from training.config import TrainingConfig
from training.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description="Train a PPO Hearthstone agent")

    # Training loop
    parser.add_argument("--episodes", type=int, default=1000, help="Total training episodes")
    parser.add_argument("--eval-interval", type=int, default=100, help="Evaluate every N episodes")
    parser.add_argument("--eval-games", type=int, default=50, help="Games per evaluation")
    parser.add_argument("--save-interval", type=int, default=500, help="Save checkpoint every N episodes (0=disabled)")
    parser.add_argument("--max-turns", type=int, default=200, help="Max turns per game")

    # PPO hyperparameters
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--clip-epsilon", type=float, default=0.2, help="PPO clip range")
    parser.add_argument("--ppo-epochs", type=int, default=4, help="PPO epochs per update")
    parser.add_argument("--entropy-coef", type=float, default=0.01, help="Entropy bonus weight")

    # Environment
    parser.add_argument("--opponent", type=str, default="random", choices=["random", "greedy", "self"], help="Opponent type")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # Output
    parser.add_argument("--name", type=str, default="ppo_run", help="Experiment name")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Checkpoint directory")

    args = parser.parse_args()

    config = TrainingConfig(
        learning_rate=args.lr,
        gamma=args.gamma,
        clip_epsilon=args.clip_epsilon,
        ppo_epochs=args.ppo_epochs,
        entropy_coef=args.entropy_coef,
        total_episodes=args.episodes,
        eval_interval=args.eval_interval,
        eval_games=args.eval_games,
        save_interval=args.save_interval,
        max_turns=args.max_turns,
        seed=args.seed,
        opponent=args.opponent,
        checkpoint_dir=args.checkpoint_dir,
        experiment_name=args.name,
    )

    print(f"Starting training: {config.experiment_name}")
    print(f"  Episodes: {config.total_episodes}")
    print(f"  Opponent: {config.opponent}")
    print(f"  LR: {config.learning_rate}, Gamma: {config.gamma}, Clip: {config.clip_epsilon}")
    print()

    def on_progress(info):
        ep = info['episode']
        if ep % config.eval_interval == 0 or ep == 1:
            losses = info.get('losses', {})
            ploss = losses.get('policy_loss', 0)
            vloss = losses.get('value_loss', 0)
            print(
                f"  Episode {ep:>5d} | "
                f"Reward: {info['reward']:+.1f} | "
                f"Length: {info['length']:>4d} | "
                f"P.Loss: {ploss:.4f} | "
                f"V.Loss: {vloss:.4f}"
            )

    trainer = Trainer(config, progress_callback=on_progress)
    result = trainer.train()

    print()
    print("Training complete!")
    print(f"  Total episodes: {result['total_episodes']}")
    print(f"  Total steps: {result['total_steps']}")
    print(f"  Elapsed: {result['elapsed_seconds']:.1f}s")
    print(f"  Mean reward: {result['mean_reward']:.3f}")
    print(f"  Mean length: {result['mean_length']:.1f}")
    if result['final_win_rate'] is not None:
        print(f"  Final win rate: {result['final_win_rate']:.1%}")


if __name__ == "__main__":
    main()
