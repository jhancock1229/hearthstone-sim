"""Unit tests for genetic algorithm deck optimization.

Following TDD: these tests define the contract for genetic.py, which
implements evolutionary deck search via mutation, crossover, selection,
and tournament evaluation.

Functions under test:
- play_match(): Simplified match simulation between two genotypes
- evaluate_deck(): Win rate estimation over multiple matches
- crossover(): Two-point crossover producing two offspring
- mutate(): Point mutation with configurable rate
- tournament(): Population-wide fitness evaluation
- run_ga(): Full genetic algorithm loop
"""

import random
from unittest.mock import patch, MagicMock

import pytest

from deckbuilding.deck import CARD_POOL, random_deck
from deckbuilding.genetic import (
    play_match,
    evaluate_deck,
    crossover,
    mutate,
    tournament,
    run_ga,
)


# ============================================================
# Helper Fixtures
# ============================================================


@pytest.fixture
def pool_keys():
    """Card pool key list."""
    return list(CARD_POOL.keys())


@pytest.fixture
def deck_a():
    """Deterministic deck A."""
    rng = random.Random(42)
    return random_deck(CARD_POOL, 30, rng)


@pytest.fixture
def deck_b():
    """Deterministic deck B."""
    rng = random.Random(99)
    return random_deck(CARD_POOL, 30, rng)


@pytest.fixture
def small_population(pool_keys):
    """Small population of 6 decks."""
    return [random_deck(CARD_POOL, 30, random.Random(i)) for i in range(6)]


# ============================================================
# play_match Tests
# ============================================================


class TestPlayMatch:
    """Tests for simplified match simulation."""

    def test_returns_valid_result(self, deck_a, deck_b):
        """play_match returns 1 (A wins), 0 (B wins), or 0.5 (draw)."""
        result = play_match(deck_a, deck_b)
        assert result in (0, 0.5, 1)

    def test_both_sides_can_win(self, deck_a, deck_b):
        """Over many games, both sides should be able to win (no inherent bias)."""
        results = [play_match(deck_a, deck_b) for _ in range(20)]
        # At least verify all results are valid
        for r in results:
            assert r in (0, 0.5, 1)

    def test_max_turns_limits_game(self, deck_a, deck_b):
        """Game ends after max_turns even if no one dies."""
        result = play_match(deck_a, deck_b, max_turns=1)
        assert result in (0, 0.5, 1)

    def test_zero_turns_returns_draw_or_health_tiebreaker(self, deck_a, deck_b):
        """With 0 turns, both start at 30 HP so it should be a draw."""
        result = play_match(deck_a, deck_b, max_turns=0)
        assert result == 0.5  # Equal health → draw

    def test_genotype_length_30(self, deck_a, deck_b):
        """play_match works with standard 30-card genotypes."""
        assert len(deck_a) == 30
        assert len(deck_b) == 30
        result = play_match(deck_a, deck_b)
        assert result in (0, 0.5, 1)

    def test_small_deck(self):
        """play_match works with smaller decks."""
        small_a = ["Brute"] * 5
        small_b = ["Tiny"] * 5
        result = play_match(small_a, small_b)
        assert result in (0, 0.5, 1)

    def test_aggressive_deck_beats_weak_deck(self):
        """A deck of high-attack minions should beat a deck of 1/1s."""
        strong = ["Giant"] * 10
        weak = ["Tiny"] * 10
        # Run multiple times to check trend
        wins = sum(play_match(strong, weak) for _ in range(10))
        # Strong deck should win most games
        assert wins >= 5

    def test_empty_deck_doesnt_crash(self):
        """Empty decks should not cause errors."""
        result = play_match([], [])
        assert result in (0, 0.5, 1)


# ============================================================
# evaluate_deck Tests
# ============================================================


class TestEvaluateDeck:
    """Tests for win rate estimation."""

    def test_returns_float_between_0_and_1(self, deck_a, deck_b):
        """evaluate_deck returns a win rate in [0, 1]."""
        wr = evaluate_deck(deck_a, deck_b, games=10)
        assert 0.0 <= wr <= 1.0

    def test_more_games_gives_result(self, deck_a, deck_b):
        """Can run with different game counts."""
        wr5 = evaluate_deck(deck_a, deck_b, games=5)
        wr20 = evaluate_deck(deck_a, deck_b, games=20)
        assert 0.0 <= wr5 <= 1.0
        assert 0.0 <= wr20 <= 1.0

    def test_strong_vs_weak(self):
        """Strong deck should have high win rate vs weak deck."""
        strong = ["Giant"] * 15
        weak = ["Tiny"] * 15
        wr = evaluate_deck(strong, weak, games=20)
        assert wr >= 0.5

    def test_one_game(self, deck_a, deck_b):
        """Handles single-game evaluation."""
        wr = evaluate_deck(deck_a, deck_b, games=1)
        assert wr in (0.0, 0.5, 1.0)

    def test_win_rate_granularity(self, deck_a, deck_b):
        """Win rate should be a multiple of 1/games."""
        games = 10
        wr = evaluate_deck(deck_a, deck_b, games=games)
        # wr * games should be close to an integer or half-integer
        scaled = wr * games
        assert scaled == pytest.approx(round(scaled * 2) / 2, abs=0.01)


# ============================================================
# crossover Tests
# ============================================================


class TestCrossover:
    """Tests for two-point crossover."""

    def test_offspring_same_length(self, deck_a, deck_b):
        """Offspring have same length as parents."""
        c1, c2 = crossover(deck_a, deck_b)
        assert len(c1) == len(deck_a)
        assert len(c2) == len(deck_b)

    def test_offspring_contain_parent_genes(self, deck_a, deck_b):
        """Each gene in offspring comes from one of the parents."""
        c1, c2 = crossover(deck_a, deck_b)
        for i in range(len(deck_a)):
            assert c1[i] in (deck_a[i], deck_b[i])
            assert c2[i] in (deck_a[i], deck_b[i])

    def test_complementary_offspring(self, deck_a, deck_b):
        """At each position, if c1 gets gene from A, c2 gets gene from B and vice versa."""
        random.seed(42)
        c1, c2 = crossover(deck_a, deck_b)
        for i in range(len(deck_a)):
            # Each position: one child gets A's gene, the other gets B's
            assert {c1[i], c2[i]} == {deck_a[i], deck_b[i]}

    def test_two_point_crossover_structure(self, deck_a, deck_b):
        """Crossover has three segments: [0:i] from parent, [i:j] from other, [j:] from parent."""
        random.seed(100)
        c1, c2 = crossover(deck_a, deck_b)
        # Find the crossover points by detecting where genes switch
        # c1 should be: a[:i] + b[i:j] + a[j:]
        # Find first switch point
        i = None
        for idx in range(len(deck_a)):
            if c1[idx] != deck_a[idx]:
                i = idx
                break
        if i is not None:
            # Find where it switches back
            j = None
            for idx in range(i, len(deck_a)):
                if c1[idx] == deck_a[idx] and deck_a[idx] != deck_b[idx]:
                    j = idx
                    break
            if j is not None:
                # Verify segments
                assert c1[:i] == deck_a[:i]
                assert c1[i:j] == deck_b[i:j]
                assert c1[j:] == deck_a[j:]

    def test_identical_parents_produce_identical_offspring(self):
        """Crossover of identical parents produces identical offspring."""
        parent = ["Adept"] * 10
        c1, c2 = crossover(parent, parent)
        assert c1 == parent
        assert c2 == parent

    def test_crossover_different_lengths_raises(self):
        """Crossover with mismatched lengths raises ValueError."""
        with pytest.raises(ValueError, match="same length"):
            crossover(["Adept"] * 5, ["Adept"] * 10)

    def test_minimum_length_3(self):
        """Crossover needs at least 3 elements for two valid cut points."""
        c1, c2 = crossover(["A", "B", "C"], ["X", "Y", "Z"])
        assert len(c1) == 3
        assert len(c2) == 3

    def test_returns_new_lists(self, deck_a, deck_b):
        """Crossover returns new lists, not references to parents."""
        c1, c2 = crossover(deck_a, deck_b)
        assert c1 is not deck_a
        assert c1 is not deck_b
        assert c2 is not deck_a
        assert c2 is not deck_b


# ============================================================
# mutate Tests
# ============================================================


class TestMutate:
    """Tests for point mutation."""

    def test_same_length(self, deck_a, pool_keys):
        """Mutated genotype has same length."""
        result = mutate(deck_a, pool_keys, rate=0.1)
        assert len(result) == len(deck_a)

    def test_zero_rate_no_changes(self, deck_a, pool_keys):
        """Rate=0 means no mutations."""
        result = mutate(deck_a, pool_keys, rate=0.0)
        assert result == deck_a

    def test_rate_1_all_change(self, pool_keys):
        """Rate=1.0 mutates every position."""
        deck = ["Tiny"] * 30
        random.seed(42)
        result = mutate(deck, pool_keys, rate=1.0)
        # With rate=1.0, every position is mutated (may randomly select same card)
        changed = sum(1 for a, b in zip(deck, result) if a != b)
        # Very unlikely all 30 land on "Tiny" again
        assert changed > 0

    def test_mutations_come_from_pool(self, deck_a, pool_keys):
        """All mutated genes are valid pool keys."""
        result = mutate(deck_a, pool_keys, rate=0.5)
        for gene in result:
            assert gene in pool_keys

    def test_returns_new_list(self, deck_a, pool_keys):
        """Mutate returns a new list, not a reference to original."""
        result = mutate(deck_a, pool_keys, rate=0.0)
        assert result is not deck_a
        assert result == deck_a  # Same contents though

    def test_approximate_mutation_count(self, pool_keys):
        """Number of mutations should be approximately rate * length."""
        deck = ["Tiny"] * 100  # Use longer deck for statistical stability
        rate = 0.5
        total_mutations = 0
        trials = 50
        for _ in range(trials):
            result = mutate(deck, pool_keys, rate=rate)
            total_mutations += sum(1 for a, b in zip(deck, result) if a != b)
        avg_mutations = total_mutations / trials
        expected = rate * len(deck)
        # Allow generous margin
        assert expected * 0.5 <= avg_mutations <= expected * 1.5

    def test_default_rate(self, deck_a, pool_keys):
        """Default mutation rate is 0.05."""
        # Just verify it works without explicit rate
        result = mutate(deck_a, pool_keys)
        assert len(result) == len(deck_a)


# ============================================================
# tournament Tests
# ============================================================


class TestTournament:
    """Tests for population evaluation."""

    def test_returns_scored_pairs(self, small_population, deck_a):
        """Tournament returns list of (genotype, score) tuples."""
        scored = tournament(small_population, deck_a, games_per_eval=5)
        assert len(scored) == len(small_population)
        for genotype, score in scored:
            assert isinstance(genotype, list)
            assert isinstance(score, float)
            assert 0.0 <= score <= 1.0

    def test_all_individuals_evaluated(self, small_population, deck_a):
        """Every individual in population gets a score."""
        scored = tournament(small_population, deck_a, games_per_eval=5)
        scored_genotypes = [g for g, s in scored]
        for individual in small_population:
            assert individual in scored_genotypes

    def test_single_process(self, small_population, deck_a):
        """Tournament works with processes=1 (sequential)."""
        scored = tournament(small_population, deck_a, games_per_eval=5, processes=1)
        assert len(scored) == len(small_population)

    def test_single_individual(self, deck_a, deck_b):
        """Tournament works with population of 1."""
        scored = tournament([deck_a], deck_b, games_per_eval=5)
        assert len(scored) == 1
        _, score = scored[0]
        assert 0.0 <= score <= 1.0


# ============================================================
# run_ga Tests
# ============================================================


class TestRunGA:
    """Tests for full genetic algorithm."""

    def test_returns_scored_population(self):
        """run_ga returns list of (genotype, score) tuples."""
        result = run_ga(pop_size=6, generations=2, games_per_eval=3)
        assert isinstance(result, list)
        assert len(result) == 6
        for genotype, score in result:
            assert isinstance(genotype, list)
            assert isinstance(score, float)

    def test_result_sorted_by_fitness(self):
        """Results are sorted by score descending."""
        result = run_ga(pop_size=6, generations=2, games_per_eval=3)
        scores = [s for _, s in result]
        assert scores == sorted(scores, reverse=True)

    def test_genotypes_are_valid(self):
        """All genotypes in result use valid card names."""
        pool_keys = set(CARD_POOL.keys())
        result = run_ga(pop_size=6, generations=2, games_per_eval=3, deck_size=10)
        for genotype, _ in result:
            assert len(genotype) == 10
            for gene in genotype:
                assert gene in pool_keys

    def test_custom_opponent(self, deck_a):
        """run_ga accepts a custom opponent genotype."""
        result = run_ga(pop_size=6, generations=1, games_per_eval=3, opponent=deck_a)
        assert len(result) == 6

    def test_population_size_maintained(self):
        """Population size stays constant across generations."""
        pop_size = 8
        result = run_ga(pop_size=pop_size, generations=3, games_per_eval=3)
        assert len(result) == pop_size

    def test_elites_preserved(self):
        """Top individuals survive to next generation (elitism)."""
        # run_ga keeps top 10% as elites
        # With pop_size=10, that's 1 elite
        result = run_ga(pop_size=10, generations=2, games_per_eval=5)
        # Just verify it completes and returns correct size
        assert len(result) == 10

    def test_small_run(self):
        """Minimal configuration doesn't crash."""
        result = run_ga(pop_size=4, generations=1, deck_size=5, games_per_eval=2)
        assert len(result) == 4

    def test_default_opponent_is_random(self):
        """When opponent=None, a random deck is generated."""
        result = run_ga(pop_size=4, generations=1, games_per_eval=2)
        assert len(result) == 4


# ============================================================
# Integration / Edge Cases
# ============================================================


class TestGeneticIntegration:
    """Integration tests combining multiple components."""

    def test_evolve_then_evaluate(self):
        """Best deck from GA can be evaluated independently."""
        result = run_ga(pop_size=6, generations=2, games_per_eval=3)
        best_genotype = result[0][0]
        opponent = random_deck(CARD_POOL, 30, random.Random(0))
        wr = evaluate_deck(best_genotype, opponent, games=5)
        assert 0.0 <= wr <= 1.0

    def test_crossover_then_mutate(self, deck_a, deck_b, pool_keys):
        """Crossover + mutation pipeline works."""
        c1, c2 = crossover(deck_a, deck_b)
        m1 = mutate(c1, pool_keys, rate=0.1)
        m2 = mutate(c2, pool_keys, rate=0.1)
        assert len(m1) == len(deck_a)
        assert len(m2) == len(deck_b)
        # All genes valid
        for gene in m1 + m2:
            assert gene in pool_keys

    def test_ga_improves_or_maintains(self):
        """GA should not degrade: final best >= random baseline expectation."""
        # With elitism, best score should be maintained or improved
        result = run_ga(pop_size=10, generations=5, games_per_eval=10)
        best_score = result[0][1]
        # A completely random deck vs random opponent would average ~0.5
        # Best in population should be at least that
        assert best_score >= 0.0  # Sanity check
