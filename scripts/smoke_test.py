#!/usr/bin/env python3
"""CI smoke test — runs a short simulation and asserts the engine is healthy.

Exits 0 if Greedy beats Random in >50% of 10 seeded games.
Exits 1 with a diagnostic message if something is wrong.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.random_agent import RandomAgent
from agents.greedy_agent import GreedyAgent
from simulation.simulator import Simulator


def main() -> int:
    sim = Simulator(seed=42, max_turns=200)
    greedy = GreedyAgent(name="Greedy")
    random = RandomAgent(seed=99, name="Random")

    result = sim.run_games(greedy, random, num_games=10)

    print(f"Greedy vs Random  ({result.total_games} games, seed=42)")
    print(f"  Greedy wins: {result.player1_wins}")
    print(f"  Random wins: {result.player2_wins}")
    print(f"  Greedy win rate: {result.player1_win_rate:.0%}")

    if result.player1_wins <= result.player2_wins:
        print("\nFAIL: Greedy should beat Random more often than not.")
        return 1

    print("\nPASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
