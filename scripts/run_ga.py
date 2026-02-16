"""Run a short GA experiment to verify the deckbuilding genetic algorithm.

This script is intended as a smoke test; adjust parameters for larger runs.
"""
import os
import sys
import random

# allow running from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from deckbuilding.genetic import run_ga
from deckbuilding.deck import CARD_POOL


if __name__ == "__main__":
    random.seed(42)
    result = run_ga(pop_size=12, generations=5, deck_size=15, games_per_eval=10, processes=1)
    best = result[0]
    print("Best deck score:", best[1])
    print("Deck sample:", best[0][:10])
