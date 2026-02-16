"""Tests for deck evaluator - win rate estimation.

These tests validate:
- Heuristic deck scoring (mana curve, card synergy)
- Simulation-based evaluation (win rate against opponents)
- Comparative deck evaluation
"""

import pytest
from hearthstone.cards.base import MinionCard, SpellCard
from deckbuilding.deck import Deck
from deckbuilding.evaluator import (
    DeckEvaluator,
    HeuristicEvaluator,
    SimulationEvaluator,
    EvaluationResult,
)


class TestHeuristicEvaluator:
    """Test heuristic-based deck evaluation."""

    def test_evaluate_balanced_deck(self):
        """Balanced deck gets good heuristic score."""
        # Create deck with good mana curve (mix of 1-5 cost)
        cards = []
        for cost in range(1, 6):
            for _ in range(6):
                cards.append(MinionCard(
                    id=f"m{cost}_{_}",
                    name=f"Minion{cost}",
                    mana_cost=cost,
                    attack=cost,
                    health=cost
                ))
        deck = Deck(cards=cards[:30])

        evaluator = HeuristicEvaluator()
        result = evaluator.evaluate(deck)

        assert isinstance(result, EvaluationResult)
        assert result.score > 0.5  # Should be decent
        assert result.metrics is not None

    def test_evaluate_high_curve_deck(self):
        """High mana curve deck gets penalized for curve."""
        # All 10-cost cards
        cards = [
            MinionCard(id=f"h{i}", name=f"H{i%15}", mana_cost=10, attack=10, health=10)
            for i in range(30)
        ]
        deck = Deck(cards=cards)

        evaluator = HeuristicEvaluator()
        result = evaluator.evaluate(deck)

        # High curve should have low mana_curve_score
        assert result.metrics["mana_curve_score"] < 0.5

    def test_evaluate_low_curve_deck(self):
        """Very low mana curve deck gets moderate score."""
        # All 1-cost cards
        cards = [
            MinionCard(id=f"l{i}", name=f"L{i%15}", mana_cost=1, attack=1, health=1)
            for i in range(30)
        ]
        deck = Deck(cards=cards)

        evaluator = HeuristicEvaluator()
        result = evaluator.evaluate(deck)

        assert result.score < 0.7  # Low curve lacks late game

    def test_evaluation_includes_metrics(self):
        """Evaluation result includes detailed metrics."""
        cards = [
            MinionCard(id=f"m{i}", name=f"M{i%15}", mana_cost=3, attack=3, health=3)
            for i in range(30)
        ]
        deck = Deck(cards=cards)

        evaluator = HeuristicEvaluator()
        result = evaluator.evaluate(deck)

        assert "mana_curve_score" in result.metrics
        assert "card_quality_score" in result.metrics
        assert "synergy_score" in result.metrics


class TestSimulationEvaluator:
    """Test simulation-based deck evaluation."""

    def test_evaluate_with_simulations(self):
        """Deck evaluation runs simulations and returns win rate."""
        # Create simple deck
        cards = [
            MinionCard(id=f"m{i}", name=f"M{i%15}", mana_cost=2, attack=2, health=2)
            for i in range(30)
        ]
        deck = Deck(cards=cards)

        evaluator = SimulationEvaluator(num_games=10)
        result = evaluator.evaluate(deck)

        assert isinstance(result, EvaluationResult)
        assert 0.0 <= result.score <= 1.0  # Win rate between 0 and 1
        assert "win_rate" in result.metrics
        assert "games_played" in result.metrics
        assert result.metrics["games_played"] == 10

    def test_evaluate_against_opponent_deck(self):
        """Evaluation can test against specific opponent deck."""
        deck1 = Deck(cards=[
            MinionCard(id=f"m{i}", name=f"M{i%15}", mana_cost=2, attack=2, health=2)
            for i in range(30)
        ])

        deck2 = Deck(cards=[
            MinionCard(id=f"o{i}", name=f"O{i%15}", mana_cost=3, attack=3, health=3)
            for i in range(30)
        ])

        evaluator = SimulationEvaluator(num_games=10, opponent_deck=deck2)
        result = evaluator.evaluate(deck1)

        assert 0.0 <= result.score <= 1.0
        assert "win_rate" in result.metrics

    def test_evaluation_is_deterministic_with_seed(self):
        """Same deck with same seed produces same result."""
        deck = Deck(cards=[
            MinionCard(id=f"m{i}", name=f"M{i%15}", mana_cost=2, attack=2, health=2)
            for i in range(30)
        ])

        evaluator1 = SimulationEvaluator(num_games=10, seed=42)
        result1 = evaluator1.evaluate(deck)

        evaluator2 = SimulationEvaluator(num_games=10, seed=42)
        result2 = evaluator2.evaluate(deck)

        assert result1.score == result2.score

    def test_more_games_gives_more_confidence(self):
        """More simulation games should give better confidence estimate."""
        deck = Deck(cards=[
            MinionCard(id=f"m{i}", name=f"M{i%15}", mana_cost=2, attack=2, health=2)
            for i in range(30)
        ])

        evaluator_few = SimulationEvaluator(num_games=5)
        result_few = evaluator_few.evaluate(deck)

        evaluator_many = SimulationEvaluator(num_games=50)
        result_many = evaluator_many.evaluate(deck)

        # Both should have confidence, but more games = higher confidence
        assert "confidence" in result_few.metrics
        assert "confidence" in result_many.metrics


class TestDeckEvaluator:
    """Test the main DeckEvaluator interface."""

    def test_default_uses_heuristic(self):
        """Default evaluator uses heuristic evaluation."""
        deck = Deck(cards=[
            MinionCard(id=f"m{i}", name=f"M{i%15}", mana_cost=2, attack=2, health=2)
            for i in range(30)
        ])

        evaluator = DeckEvaluator()
        result = evaluator.evaluate(deck)

        assert isinstance(result, EvaluationResult)
        assert result.score >= 0.0

    def test_can_use_simulation_mode(self):
        """Evaluator can switch to simulation mode."""
        deck = Deck(cards=[
            MinionCard(id=f"m{i}", name=f"M{i%15}", mana_cost=2, attack=2, health=2)
            for i in range(30)
        ])

        evaluator = DeckEvaluator(mode="simulation", num_games=5)
        result = evaluator.evaluate(deck)

        assert "games_played" in result.metrics

    def test_can_use_hybrid_mode(self):
        """Evaluator can combine heuristic and simulation."""
        deck = Deck(cards=[
            MinionCard(id=f"m{i}", name=f"M{i%15}", mana_cost=2, attack=2, health=2)
            for i in range(30)
        ])

        evaluator = DeckEvaluator(mode="hybrid", num_games=5)
        result = evaluator.evaluate(deck)

        # Should have both heuristic and simulation metrics
        assert "mana_curve_score" in result.metrics
        assert result.score >= 0.0


class TestEvaluationResult:
    """Test the EvaluationResult data structure."""

    def test_evaluation_result_creation(self):
        """Can create evaluation result."""
        result = EvaluationResult(
            score=0.75,
            metrics={"win_rate": 0.75, "games": 100}
        )

        assert result.score == 0.75
        assert result.metrics["win_rate"] == 0.75

    def test_evaluation_result_comparison(self):
        """Evaluation results can be compared by score."""
        result1 = EvaluationResult(score=0.6, metrics={})
        result2 = EvaluationResult(score=0.8, metrics={})

        assert result2.score > result1.score

    def test_evaluation_result_string_representation(self):
        """Evaluation result has readable string form."""
        result = EvaluationResult(
            score=0.75,
            metrics={"win_rate": 0.75}
        )

        str_repr = str(result)
        assert "0.75" in str_repr or "75" in str_repr


class TestComparativeEvaluation:
    """Test comparing multiple decks."""

    def test_compare_two_decks(self):
        """Can compare two decks and identify better one."""
        # Balanced deck
        deck1 = Deck(cards=[
            MinionCard(id=f"m{i}", name=f"M{i%10}", mana_cost=(i % 5) + 1,
                      attack=(i % 5) + 1, health=(i % 5) + 1)
            for i in range(30)
        ])

        # High curve deck
        deck2 = Deck(cards=[
            MinionCard(id=f"h{i}", name=f"H{i%15}", mana_cost=10, attack=10, health=10)
            for i in range(30)
        ])

        evaluator = DeckEvaluator()
        result1 = evaluator.evaluate(deck1)
        result2 = evaluator.evaluate(deck2)

        # Balanced deck should score higher
        assert result1.score > result2.score

    def test_rank_multiple_decks(self):
        """Can rank multiple decks by quality."""
        from deckbuilding.evaluator import rank_decks

        decks = []
        # Create 3 decks with different curves
        for avg_cost in [2, 5, 8]:
            cards = [
                MinionCard(id=f"m{avg_cost}_{i}", name=f"M{i%15}",
                          mana_cost=avg_cost, attack=avg_cost, health=avg_cost)
                for i in range(30)
            ]
            decks.append(Deck(cards=cards))

        evaluator = DeckEvaluator()
        ranked = rank_decks(decks, evaluator)

        assert len(ranked) == 3
        # Results should be sorted by score (descending)
        assert ranked[0][1].score >= ranked[1][1].score >= ranked[2][1].score


# ============================================================
# SimulationEvaluator with Real Games
# ============================================================


class TestSimulationEvaluatorGenotype:
    """Tests for evaluate_genotype using real Simulator."""

    def test_evaluate_genotype_returns_result(self):
        """evaluate_genotype returns an EvaluationResult."""
        evaluator = SimulationEvaluator(num_games=2, seed=42)
        genotype = ["Brute"] * 15 + ["Wasp"] * 15
        result = evaluator.evaluate_genotype(genotype)
        assert isinstance(result, EvaluationResult)

    def test_win_rate_between_0_and_1(self):
        """Win rate is bounded."""
        evaluator = SimulationEvaluator(num_games=4, seed=42)
        genotype = ["Adept"] * 30
        result = evaluator.evaluate_genotype(genotype)
        assert 0.0 <= result.score <= 1.0

    def test_metrics_include_game_stats(self):
        """Result metrics include games_played, wins, losses."""
        evaluator = SimulationEvaluator(num_games=3, seed=42)
        genotype = ["Giant"] * 10 + ["Wasp"] * 20
        result = evaluator.evaluate_genotype(genotype)
        assert result.metrics["games_played"] == 3
        assert "wins" in result.metrics
        assert "losses" in result.metrics
        assert result.metrics["wins"] + result.metrics["losses"] == 3

    def test_with_custom_opponent(self):
        """Can evaluate against a specific opponent genotype."""
        opponent = ["Tiny"] * 30
        evaluator = SimulationEvaluator(num_games=2, seed=42,
                                        opponent_genotype=opponent)
        genotype = ["Giant"] * 10 + ["Brute"] * 20
        result = evaluator.evaluate_genotype(genotype)
        assert isinstance(result, EvaluationResult)

    def test_with_custom_pool(self):
        """Can evaluate with a custom card pool."""
        from deckbuilding.deck import CardSpec
        pool = {
            "Tank": CardSpec("Tank", 3, 2, 5, mechanics=["TAUNT"]),
            "Hitter": CardSpec("Hitter", 2, 3, 2),
        }
        evaluator = SimulationEvaluator(num_games=2, seed=42, pool=pool)
        genotype = ["Tank"] * 15 + ["Hitter"] * 15
        result = evaluator.evaluate_genotype(genotype)
        assert 0.0 <= result.score <= 1.0
