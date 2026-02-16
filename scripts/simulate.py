#!/usr/bin/env python3
"""Run N games between two agents and output statistics.

Usage:
    python scripts/simulate.py --p1 random --p2 random --games 100
    python scripts/simulate.py --p1 greedy --p2 random --games 1000 --seed 42
    python scripts/simulate.py --p1 greedy --p2 greedy --games 100 --verbose

Examples:
    # Run 100 games between two random agents
    python scripts/simulate.py --games 100

    # Run 1000 games: Greedy vs Random
    python scripts/simulate.py --p1 greedy --p2 random --games 1000

    # Run 100 games: Greedy vs Greedy
    python scripts/simulate.py --p1 greedy --p2 greedy --games 100

    # Run with verbose output
    python scripts/simulate.py --p1 greedy --p2 random --games 500 --verbose
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.random_agent import RandomAgent
from agents.greedy_agent import GreedyAgent
from agents.mcts_agent import MCTSAgent
from simulation.simulator import Simulator


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run N games between two agents and collect statistics"
    )

    parser.add_argument(
        '--p1',
        type=str,
        default='random',
        choices=['random', 'greedy', 'mcts'],
        help='Player 1 agent type (default: random)'
    )

    parser.add_argument(
        '--p2',
        type=str,
        default='random',
        choices=['random', 'greedy', 'mcts'],
        help='Player 2 agent type (default: random)'
    )

    parser.add_argument(
        '--games',
        type=int,
        default=100,
        help='Number of games to simulate (default: 100)'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility (default: None)'
    )

    parser.add_argument(
        '--max-turns',
        type=int,
        default=200,
        help='Maximum turns per game (default: 200)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print progress during simulation'
    )

    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Save results to JSON file (default: None)'
    )

    return parser.parse_args()


def create_agent(agent_type: str, player_num: int, seed: int = None):
    """Create an agent of the specified type.

    Args:
        agent_type: Type of agent to create
        player_num: Player number (1 or 2)
        seed: Optional seed for agent

    Returns:
        Agent instance
    """
    if agent_type == 'random':
        agent_seed = None if seed is None else seed + player_num
        return RandomAgent(
            seed=agent_seed,
            name=f"RandomAgent-P{player_num}"
        )
    elif agent_type == 'greedy':
        return GreedyAgent(name=f"GreedyAgent-P{player_num}")
    elif agent_type == 'mcts':
        agent_seed = None if seed is None else seed + player_num
        return MCTSAgent(
            name=f"MCTSAgent-P{player_num}",
            num_iterations=50,  # Reasonable default
            seed=agent_seed
        )
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


def main():
    """Main entry point."""
    args = parse_args()

    # Create agents
    agent1 = create_agent(args.p1, 1, args.seed)
    agent2 = create_agent(args.p2, 2, args.seed)

    # Create simulator
    simulator = Simulator(
        seed=args.seed,
        max_turns=args.max_turns,
        verbose=args.verbose
    )

    # Run simulation
    print(f"Running {args.games} games...")
    print(f"Player 1: {agent1.name}")
    print(f"Player 2: {agent2.name}")
    if args.seed is not None:
        print(f"Seed: {args.seed}")
    print()

    result = simulator.run_games(agent1, agent2, args.games)

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(result)
    print("=" * 60)

    # Save to file if requested
    if args.output:
        import json
        with open(args.output, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == '__main__':
    main()
