"""Co-evolutionary deck optimization.

Evolves a population of decks against each other using real game
simulations with intelligent agents. Tracks meta-game dynamics
via MetaTracker and MatchupMatrix.

Usage:
    from deckbuilding.coevolution import CoevolutionConfig, CoevolutionEngine
    from deckbuilding.deck import build_pool_from_registry
    from hearthstone.cards.registry import CardRegistry

    registry = CardRegistry.from_json(Path("hearthstone/data/cards_collectible.json"))
    pool = build_pool_from_registry(registry)
    config = CoevolutionConfig(pool=pool, population_size=50, generations=20)
    engine = CoevolutionEngine(config)
    results = engine.run()
    # results: List[(genotype, fitness)] sorted by fitness descending
"""

import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from agents.greedy_agent import GreedyAgent
from analysis.matchups import MatchupMatrix
from deckbuilding.deck import CardSpec, random_deck
from deckbuilding.genetic import crossover
from deckbuilding.meta import MetaTracker
from simulation.simulator import Simulator


@dataclass
class CoevolutionConfig:
    """Configuration for co-evolutionary deck optimization.

    Attributes:
        pool: Card pool mapping card ID/name to CardSpec
        population_size: Number of decks in the population
        generations: Number of evolutionary generations
        deck_size: Cards per deck
        opponents_per_eval: Number of opponents each deck plays per generation
        games_per_matchup: Games played per deck pair
        elite_fraction: Fraction of population preserved as elites
        mutation_rate: Per-card mutation probability
        crossover_rate: Probability of applying crossover to parent pair
        seed: Random seed for reproducibility
        max_turns_per_game: Max turns before game is declared a draw
    """
    pool: Dict[str, CardSpec] = field(default_factory=dict)
    population_size: int = 50
    generations: int = 20
    deck_size: int = 30
    opponents_per_eval: int = 7
    games_per_matchup: int = 5
    elite_fraction: float = 0.1
    mutation_rate: float = 0.05
    crossover_rate: float = 0.7
    seed: Optional[int] = None
    max_turns_per_game: int = 80


def classify_archetype(genotype: List[str], pool: Dict[str, CardSpec]) -> str:
    """Classify a deck genotype into an archetype by mana curve.

    Args:
        genotype: List of card pool keys
        pool: Card pool mapping keys to CardSpec

    Returns:
        Archetype label: "Aggro", "Midrange", "Control", "Tempo", or "Unknown"
    """
    costs = [pool[card_id].mana_cost for card_id in genotype if card_id in pool]
    if not costs:
        return "Unknown"
    avg_cost = sum(costs) / len(costs)
    if avg_cost <= 2.5:
        return "Aggro"
    elif avg_cost >= 5.0:
        return "Control"
    elif avg_cost < 4.0:
        return "Midrange"
    else:
        return "Tempo"


def constrained_mutate(
    genotype: List[str],
    pool_keys: List[str],
    rate: float = 0.05,
    max_copies: int = 2,
    rng: Optional[random.Random] = None,
) -> List[str]:
    """Mutate a genotype while respecting copy limits.

    Args:
        genotype: Deck genotype to mutate
        pool_keys: Available card keys to mutate into
        rate: Per-card mutation probability
        max_copies: Maximum copies of any single card
        rng: Random number generator

    Returns:
        New mutated genotype (original is not modified)
    """
    if rng is None:
        rng = random.Random()
    out = genotype[:]
    for i in range(len(out)):
        if rng.random() < rate:
            for _ in range(10):
                candidate = rng.choice(pool_keys)
                if out.count(candidate) < max_copies:
                    out[i] = candidate
                    break
    return out


class CoevolutionEngine:
    """Co-evolutionary deck optimization engine.

    Evolves a population of decks where fitness is measured by
    playing against other members of the population using the
    Simulator with GreedyAgent.
    """

    def __init__(self, config: CoevolutionConfig):
        self.config = config
        self.rng = random.Random(config.seed)
        self.meta_tracker = MetaTracker()
        self.matchup_matrix = MatchupMatrix()

    def run(self) -> List[Tuple[List[str], float]]:
        """Run the full co-evolutionary optimization.

        Returns:
            List of (genotype, fitness) sorted by fitness descending
        """
        pool_keys = list(self.config.pool.keys())
        population = [
            random_deck(self.config.pool, self.config.deck_size, self.rng)
            for _ in range(self.config.population_size)
        ]

        for gen in range(self.config.generations):
            fitness = self._evaluate_population(population, gen)
            self._record_meta(population, gen)
            population = self._next_generation(population, fitness, pool_keys)

        # Final evaluation
        final_fitness = self._evaluate_population(
            population, self.config.generations
        )
        scored = list(zip(population, final_fitness))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def _evaluate_population(
        self, population: List[List[str]], gen: int
    ) -> List[float]:
        """Evaluate each individual against K sampled opponents."""
        n = len(population)
        k = min(self.config.opponents_per_eval, n - 1)
        fitness = [0.0] * n

        for i in range(n):
            opponents = self.rng.sample(
                [j for j in range(n) if j != i], k
            )
            total_score = 0.0
            for j in opponents:
                sim = Simulator(
                    seed=self.rng.randint(0, 2**31 - 1),
                    max_turns=self.config.max_turns_per_game,
                )
                agent1 = GreedyAgent()
                agent2 = GreedyAgent()
                result = sim.run_games(
                    agent1, agent2, self.config.games_per_matchup,
                    deck1_genotype=population[i],
                    deck2_genotype=population[j],
                    pool=self.config.pool,
                )
                total_score += result.player1_win_rate

                # Record in matchup matrix
                label_i = f"g{gen}_d{i}"
                label_j = f"g{gen}_d{j}"
                self.matchup_matrix.record(
                    label_i, label_j,
                    wins=result.player1_wins,
                    losses=result.player2_wins,
                )

                # Record in meta tracker by archetype
                arch_i = classify_archetype(population[i], self.config.pool)
                arch_j = classify_archetype(population[j], self.config.pool)
                for _ in range(result.player1_wins):
                    self.meta_tracker.record_match(arch_i, arch_j, winner=arch_i)
                for _ in range(result.player2_wins):
                    self.meta_tracker.record_match(arch_i, arch_j, winner=arch_j)

            fitness[i] = total_score / k if k > 0 else 0.0

        return fitness

    def _record_meta(self, population: List[List[str]], gen: int):
        """Ensure all archetypes in the population are registered."""
        for genotype in population:
            archetype = classify_archetype(genotype, self.config.pool)
            self.meta_tracker._ensure_registered(archetype)

    def _next_generation(
        self,
        population: List[List[str]],
        fitness: List[float],
        pool_keys: List[str],
    ) -> List[List[str]]:
        """Create the next generation via selection, crossover, mutation."""
        scored = list(zip(population, fitness))
        scored.sort(key=lambda x: x[1], reverse=True)

        # Elite selection
        n_elites = max(1, int(self.config.population_size * self.config.elite_fraction))
        next_pop = [ind for ind, _ in scored[:n_elites]]

        # Build rest via crossover + mutation
        while len(next_pop) < self.config.population_size:
            # Tournament selection: pick 2 random parents weighted by fitness
            parents = self.rng.choices(scored, k=2)
            p1 = parents[0][0]
            p2 = parents[1][0]

            if self.rng.random() < self.config.crossover_rate:
                try:
                    c1, c2 = crossover(p1, p2)
                except ValueError:
                    c1, c2 = p1[:], p2[:]
            else:
                c1, c2 = p1[:], p2[:]

            c1 = constrained_mutate(
                c1, pool_keys, self.config.mutation_rate, rng=self.rng
            )
            c2 = constrained_mutate(
                c2, pool_keys, self.config.mutation_rate, rng=self.rng
            )

            next_pop.append(c1)
            if len(next_pop) < self.config.population_size:
                next_pop.append(c2)

        return next_pop
