"""Evaluate agent performance over N games.

Usage:
    python scripts/evaluate.py                                    # random vs random
    python scripts/evaluate.py --agent1 greedy --agent2 random    # greedy vs random
    python scripts/evaluate.py --agent1 ppo --model checkpoint.pt # PPO vs random
    python scripts/evaluate.py --games 500 --seed 123             # more games
"""

import argparse

from agents.random_agent import RandomAgent
from agents.greedy_agent import GreedyAgent
from simulation.simulator import Simulator


def create_agent(agent_type: str, model_path: str = None, seed: int = 42):
    """Create an agent by type name.

    Args:
        agent_type: One of "random", "greedy", "mcts", "ppo"
        model_path: Path to model checkpoint (for ppo)
        seed: Random seed

    Returns:
        Agent instance
    """
    if agent_type == "random":
        return RandomAgent(seed=seed)
    elif agent_type == "greedy":
        return GreedyAgent()
    elif agent_type == "mcts":
        from agents.mcts_agent import MCTSAgent
        return MCTSAgent()
    elif agent_type == "ppo":
        from agents.rl.ppo_agent import PPOAgent
        agent = PPOAgent()
        if model_path:
            agent.load(model_path)
        return agent
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Hearthstone agents")

    parser.add_argument("--agent1", type=str, default="random", choices=["random", "greedy", "mcts", "ppo"], help="Player 1 agent type")
    parser.add_argument("--agent2", type=str, default="random", choices=["random", "greedy", "mcts", "ppo"], help="Player 2 agent type")
    parser.add_argument("--model1", type=str, default=None, help="Model checkpoint for agent 1 (ppo)")
    parser.add_argument("--model2", type=str, default=None, help="Model checkpoint for agent 2 (ppo)")
    parser.add_argument("--games", type=int, default=100, help="Number of games to play")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max-turns", type=int, default=200, help="Max turns per game")
    parser.add_argument("--verbose", action="store_true", help="Print progress")

    args = parser.parse_args()

    agent1 = create_agent(args.agent1, args.model1, seed=args.seed)
    agent2 = create_agent(args.agent2, args.model2, seed=args.seed + 1)

    print(f"Evaluating: {agent1.name} vs {agent2.name}")
    print(f"  Games: {args.games}")
    print(f"  Seed: {args.seed}")
    print()

    sim = Simulator(
        seed=args.seed,
        max_turns=args.max_turns,
        verbose=args.verbose,
    )

    result = sim.run_games(agent1, agent2, num_games=args.games)

    print(f"Results ({args.games} games):")
    print(f"  {agent1.name}: {result.player1_wins} wins ({result.player1_win_rate:.1%})")
    print(f"  {agent2.name}: {result.player2_wins} wins ({result.player2_win_rate:.1%})")
    print(f"  Avg game length: {result.average_game_length:.1f} turns")


if __name__ == "__main__":
    main()
