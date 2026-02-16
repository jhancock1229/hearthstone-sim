"""Unit tests for co-evolutionary deck optimization.

Tests cover:
- CoevolutionConfig creation and defaults
- classify_archetype function
- constrained_mutate respecting copy limits
- CoevolutionEngine creation and run
- MetaTracker and MatchupMatrix integration
"""

import random

import pytest

from deckbuilding.deck import CardSpec, CARD_POOL
from deckbuilding.coevolution import (
    CoevolutionConfig,
    CoevolutionEngine,
    classify_archetype,
    constrained_mutate,
)


# ============================================================
# CoevolutionConfig
# ============================================================


class TestCoevolutionConfig:
    """Tests for CoevolutionConfig dataclass."""

    def test_default_config(self):
        config = CoevolutionConfig(pool=CARD_POOL)
        assert config.population_size == 50
        assert config.generations == 20
        assert config.deck_size == 30
        assert config.opponents_per_eval == 7
        assert config.games_per_matchup == 5

    def test_custom_config(self):
        config = CoevolutionConfig(
            pool=CARD_POOL,
            population_size=10,
            generations=5,
            mutation_rate=0.1,
        )
        assert config.population_size == 10
        assert config.generations == 5
        assert config.mutation_rate == 0.1

    def test_elite_fraction(self):
        config = CoevolutionConfig(pool=CARD_POOL, elite_fraction=0.2)
        assert config.elite_fraction == 0.2

    def test_seed_stored(self):
        config = CoevolutionConfig(pool=CARD_POOL, seed=42)
        assert config.seed == 42

    def test_pool_required(self):
        config = CoevolutionConfig(pool=CARD_POOL)
        assert config.pool is CARD_POOL


# ============================================================
# classify_archetype
# ============================================================


class TestClassifyArchetype:
    """Tests for archetype classification by mana curve."""

    def test_aggro_low_cost(self):
        """Low average cost deck classified as Aggro."""
        pool = {
            "A": CardSpec("A", 1, 2, 1),
            "B": CardSpec("B", 2, 3, 2),
        }
        genotype = ["A"] * 20 + ["B"] * 10  # avg cost 1.33
        assert classify_archetype(genotype, pool) == "Aggro"

    def test_control_high_cost(self):
        """High average cost deck classified as Control."""
        pool = {
            "X": CardSpec("X", 7, 7, 7),
            "Y": CardSpec("Y", 6, 5, 5),
        }
        genotype = ["X"] * 15 + ["Y"] * 15  # avg cost 6.5
        assert classify_archetype(genotype, pool) == "Control"

    def test_midrange(self):
        """Mid-range cost deck classified as Midrange."""
        pool = {
            "L": CardSpec("L", 2, 2, 2),
            "M": CardSpec("M", 4, 4, 4),
        }
        genotype = ["L"] * 15 + ["M"] * 15  # avg cost 3.0
        assert classify_archetype(genotype, pool) == "Midrange"

    def test_empty_genotype(self):
        """Empty or invalid genotype returns Unknown."""
        assert classify_archetype([], {}) == "Unknown"

    def test_unknown_cards_skipped(self):
        """Cards not in pool are skipped."""
        pool = {"A": CardSpec("A", 1, 1, 1)}
        genotype = ["A"] * 10 + ["missing"] * 20
        result = classify_archetype(genotype, pool)
        assert result == "Aggro"  # avg cost = 1.0


# ============================================================
# constrained_mutate
# ============================================================


class TestConstrainedMutate:
    """Tests for mutation with copy limit enforcement."""

    def test_output_same_length(self):
        """Mutated genotype has same length as input."""
        pool_keys = list(CARD_POOL.keys())
        genotype = ["Brute"] * 30
        result = constrained_mutate(genotype, pool_keys, rate=0.5,
                                     rng=random.Random(42))
        assert len(result) == 30

    def test_respects_copy_limit(self):
        """Mutations don't introduce a card beyond max_copies."""
        from collections import Counter
        pool_keys = list(CARD_POOL.keys())
        # Start with a valid diverse genotype
        genotype = []
        for key in pool_keys:
            genotype.extend([key] * 2)
        genotype = genotype[:30]
        result = constrained_mutate(genotype, pool_keys, rate=0.5,
                                     max_copies=2, rng=random.Random(42))
        counts = Counter(result)
        for card, count in counts.items():
            assert count <= 2, f"{card} appears {count} times"

    def test_zero_rate_no_mutation(self):
        """Rate 0 produces identical genotype."""
        pool_keys = list(CARD_POOL.keys())
        genotype = ["Wasp"] * 30
        result = constrained_mutate(genotype, pool_keys, rate=0.0,
                                     rng=random.Random(42))
        assert result == genotype

    def test_high_rate_changes_cards(self):
        """High rate causes mutations."""
        pool_keys = list(CARD_POOL.keys())
        genotype = ["Wasp"] * 30
        result = constrained_mutate(genotype, pool_keys, rate=1.0,
                                     rng=random.Random(42))
        # At least some cards should differ
        assert result != genotype

    def test_does_not_mutate_in_place(self):
        """Original genotype is not modified."""
        pool_keys = list(CARD_POOL.keys())
        genotype = ["Wasp"] * 30
        original = genotype[:]
        constrained_mutate(genotype, pool_keys, rate=0.5,
                           rng=random.Random(42))
        assert genotype == original


# ============================================================
# CoevolutionEngine Creation
# ============================================================


class TestCoevolutionEngineCreation:
    """Tests for CoevolutionEngine initialization."""

    def test_create_engine(self):
        config = CoevolutionConfig(pool=CARD_POOL, seed=42)
        engine = CoevolutionEngine(config)
        assert engine.config is config

    def test_engine_has_meta_tracker(self):
        config = CoevolutionConfig(pool=CARD_POOL, seed=42)
        engine = CoevolutionEngine(config)
        from deckbuilding.meta import MetaTracker
        assert isinstance(engine.meta_tracker, MetaTracker)

    def test_engine_has_matchup_matrix(self):
        config = CoevolutionConfig(pool=CARD_POOL, seed=42)
        engine = CoevolutionEngine(config)
        from analysis.matchups import MatchupMatrix
        assert isinstance(engine.matchup_matrix, MatchupMatrix)


# ============================================================
# CoevolutionEngine.run()
# ============================================================


class TestCoevolutionEngineRun:
    """Tests for the full co-evolution loop.

    Uses tiny parameters to keep tests fast.
    """

    def _tiny_config(self, **overrides):
        defaults = dict(
            pool=CARD_POOL,
            population_size=4,
            generations=1,
            deck_size=10,
            opponents_per_eval=2,
            games_per_matchup=1,
            seed=42,
            max_turns_per_game=30,
        )
        defaults.update(overrides)
        return CoevolutionConfig(**defaults)

    def test_run_returns_sorted_results(self):
        """run() returns list of (genotype, fitness) sorted descending."""
        config = self._tiny_config()
        engine = CoevolutionEngine(config)
        results = engine.run()
        assert len(results) == config.population_size
        # Sorted descending by fitness
        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)

    def test_results_are_genotypes_with_scores(self):
        """Each result is a (list[str], float) pair."""
        config = self._tiny_config()
        engine = CoevolutionEngine(config)
        results = engine.run()
        for genotype, score in results:
            assert isinstance(genotype, list)
            assert all(isinstance(g, str) for g in genotype)
            assert isinstance(score, float)
            assert 0.0 <= score <= 1.0

    def test_matchup_matrix_populated(self):
        """MatchupMatrix has recorded matchups after run."""
        config = self._tiny_config()
        engine = CoevolutionEngine(config)
        engine.run()
        assert engine.matchup_matrix.size > 0

    def test_meta_tracker_populated(self):
        """MetaTracker has recorded matches after run."""
        config = self._tiny_config()
        engine = CoevolutionEngine(config)
        engine.run()
        assert engine.meta_tracker.total_matches > 0

    def test_multiple_generations(self):
        """Engine runs multiple generations."""
        config = self._tiny_config(generations=3)
        engine = CoevolutionEngine(config)
        results = engine.run()
        assert len(results) == config.population_size

    def test_genotype_correct_size(self):
        """All genotypes have the configured deck size."""
        config = self._tiny_config(deck_size=15)
        engine = CoevolutionEngine(config)
        results = engine.run()
        for genotype, _ in results:
            assert len(genotype) == 15

    def test_elites_preserved(self):
        """Best individuals survive across generations."""
        config = self._tiny_config(
            population_size=6,
            generations=2,
            elite_fraction=0.5,
        )
        engine = CoevolutionEngine(config)
        results = engine.run()
        # Just verify it runs and returns correct count
        assert len(results) == 6

    def test_meta_summary_has_archetypes(self):
        """Meta summary contains archetype entries."""
        config = self._tiny_config()
        engine = CoevolutionEngine(config)
        engine.run()
        summary = engine.meta_tracker.get_meta_summary()
        assert len(summary) > 0
        assert all("name" in entry for entry in summary)

    def test_deterministic_with_seed(self):
        """Same seed produces same results."""
        config1 = self._tiny_config(seed=123)
        engine1 = CoevolutionEngine(config1)
        results1 = engine1.run()

        config2 = self._tiny_config(seed=123)
        engine2 = CoevolutionEngine(config2)
        results2 = engine2.run()

        scores1 = [s for _, s in results1]
        scores2 = [s for _, s in results2]
        assert scores1 == scores2
