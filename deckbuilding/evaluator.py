"""Deck evaluator: estimates deck quality and win rate.

Provides:
- Heuristic evaluation based on deck composition
- Simulation-based evaluation (win rate estimation)
- Hybrid evaluation combining both approaches
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple
import random


@dataclass
class EvaluationResult:
    """Result of deck evaluation."""
    score: float  # Overall score (0.0 to 1.0)
    metrics: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        """String representation."""
        return f"EvaluationResult(score={self.score:.3f}, metrics={self.metrics})"


class HeuristicEvaluator:
    """Evaluates decks using heuristic analysis of deck composition.

    Analyzes:
    - Mana curve distribution
    - Card quality (stats per cost)
    - Deck synergy (basic heuristics)
    """

    def evaluate(self, deck) -> EvaluationResult:
        """Evaluate deck using heuristics.

        Args:
            deck: Deck instance to evaluate

        Returns:
            EvaluationResult with score and detailed metrics
        """
        metrics = {}

        # Evaluate mana curve
        curve_score = self._evaluate_mana_curve(deck)
        metrics["mana_curve_score"] = curve_score

        # Evaluate card quality
        quality_score = self._evaluate_card_quality(deck)
        metrics["card_quality_score"] = quality_score

        # Evaluate synergy (placeholder - basic diversity check)
        synergy_score = self._evaluate_synergy(deck)
        metrics["synergy_score"] = synergy_score

        # Combine scores (weighted average)
        overall_score = (
            0.4 * curve_score +
            0.4 * quality_score +
            0.2 * synergy_score
        )

        return EvaluationResult(score=overall_score, metrics=metrics)

    def _evaluate_mana_curve(self, deck) -> float:
        """Evaluate mana curve quality.

        Ideal curve has good distribution across costs 1-5.

        Args:
            deck: Deck to evaluate

        Returns:
            Score from 0.0 to 1.0
        """
        avg_cost = deck.average_mana_cost()

        # Ideal average is around 3-4 mana
        if 3.0 <= avg_cost <= 4.0:
            return 1.0
        elif 2.0 <= avg_cost < 3.0 or 4.0 < avg_cost <= 5.0:
            return 0.8
        elif 1.5 <= avg_cost < 2.0 or 5.0 < avg_cost <= 6.0:
            return 0.5
        else:
            # Very low or very high curve
            return 0.3

    def _evaluate_card_quality(self, deck) -> float:
        """Evaluate average card quality (stats per mana cost).

        Args:
            deck: Deck to evaluate

        Returns:
            Score from 0.0 to 1.0
        """
        if not deck.cards:
            return 0.0

        total_stats = 0
        total_cost = 0

        for card in deck.cards:
            cost = max(card.mana_cost, 1)  # Avoid division by zero
            stats = 0

            # Count stats (attack + health for minions)
            if hasattr(card, 'attack') and hasattr(card, 'health'):
                stats = card.attack + card.health

            total_stats += stats
            total_cost += cost

        # Average stats per mana
        stats_per_mana = total_stats / total_cost if total_cost > 0 else 0

        # Normalize: ~2 stats per mana is average, ~2.5 is great
        if stats_per_mana >= 2.5:
            return 1.0
        elif stats_per_mana >= 2.0:
            return 0.8
        elif stats_per_mana >= 1.5:
            return 0.6
        else:
            return 0.4

    def _evaluate_synergy(self, deck) -> float:
        """Evaluate deck synergy (basic diversity check).

        Args:
            deck: Deck to evaluate

        Returns:
            Score from 0.0 to 1.0
        """
        unique_count = len(deck.unique_cards())

        # Good variety: 20-25 unique cards
        if 20 <= unique_count <= 25:
            return 1.0
        elif 15 <= unique_count < 20 or 25 < unique_count <= 30:
            return 0.7
        else:
            return 0.5


class SimulationEvaluator:
    """Evaluates decks by simulating games and measuring win rate."""

    def __init__(
        self,
        num_games: int = 100,
        opponent_deck: Optional[Any] = None,
        seed: Optional[int] = None,
        opponent_genotype: Optional[List[str]] = None,
        pool: Optional[Dict[str, Any]] = None,
        agent_class: Optional[type] = None,
    ):
        """Initialize simulation evaluator.

        Args:
            num_games: Number of games to simulate
            opponent_deck: Specific opponent deck (or None for default)
            seed: Random seed for reproducibility
            opponent_genotype: Optional genotype for opponent deck
            pool: Optional card pool for genotype-based evaluation
            agent_class: Agent class to use for simulations (default: GreedyAgent)
        """
        self.num_games = num_games
        self.opponent_deck = opponent_deck
        self.seed = seed
        self.opponent_genotype = opponent_genotype
        self.pool = pool
        self.agent_class = agent_class

    def evaluate(self, deck) -> EvaluationResult:
        """Evaluate deck by simulating games.

        Args:
            deck: Deck instance to evaluate

        Returns:
            EvaluationResult with win rate as score
        """
        # For now, use a simple heuristic-based simulation
        # In real implementation, would use simulation.simulator
        if self.seed is not None:
            random.seed(self.seed)

        # Simplified simulation: use heuristic + randomness
        heuristic_eval = HeuristicEvaluator()
        base_score = heuristic_eval.evaluate(deck).score

        # Add some variance based on number of games
        wins = 0
        for _ in range(self.num_games):
            # Simple win probability based on heuristic score
            win_chance = base_score * 0.8 + random.random() * 0.4
            if win_chance > 0.6:
                wins += 1

        win_rate = wins / self.num_games if self.num_games > 0 else 0.0

        # Calculate confidence (more games = higher confidence)
        # Using simple heuristic: confidence increases with games
        confidence = min(1.0, self.num_games / 100.0)

        metrics = {
            "win_rate": win_rate,
            "games_played": self.num_games,
            "wins": wins,
            "losses": self.num_games - wins,
            "confidence": confidence
        }

        return EvaluationResult(score=win_rate, metrics=metrics)

    def evaluate_genotype(self, genotype: List[str]) -> EvaluationResult:
        """Evaluate a deck genotype by running real simulated games.

        Uses the Simulator with the configured agent class to play actual games.

        Args:
            genotype: List of card pool keys representing the deck

        Returns:
            EvaluationResult with win rate and game statistics
        """
        from simulation.simulator import Simulator
        from agents.greedy_agent import GreedyAgent

        AgentCls = self.agent_class or GreedyAgent
        sim = Simulator(seed=self.seed, max_turns=80)
        agent1 = AgentCls()
        agent2 = AgentCls()
        result = sim.run_games(
            agent1, agent2, self.num_games,
            deck1_genotype=genotype,
            deck2_genotype=self.opponent_genotype,
            pool=self.pool,
        )
        return EvaluationResult(
            score=result.player1_win_rate,
            metrics={
                "win_rate": result.player1_win_rate,
                "games_played": result.total_games,
                "wins": result.player1_wins,
                "losses": result.player2_wins,
                "avg_game_length": result.average_game_length,
            },
        )


class DeckEvaluator:
    """Main deck evaluator interface with multiple evaluation modes."""

    def __init__(
        self,
        mode: str = "heuristic",
        num_games: int = 100,
        seed: Optional[int] = None
    ):
        """Initialize deck evaluator.

        Args:
            mode: Evaluation mode ("heuristic", "simulation", "hybrid")
            num_games: Number of games for simulation mode
            seed: Random seed for reproducibility
        """
        self.mode = mode
        self.num_games = num_games
        self.seed = seed

    def evaluate(self, deck) -> EvaluationResult:
        """Evaluate deck using configured mode.

        Args:
            deck: Deck instance to evaluate

        Returns:
            EvaluationResult with score and metrics
        """
        if self.mode == "heuristic":
            evaluator = HeuristicEvaluator()
            return evaluator.evaluate(deck)

        elif self.mode == "simulation":
            evaluator = SimulationEvaluator(
                num_games=self.num_games,
                seed=self.seed
            )
            return evaluator.evaluate(deck)

        elif self.mode == "hybrid":
            # Combine heuristic and simulation
            heuristic_eval = HeuristicEvaluator()
            heuristic_result = heuristic_eval.evaluate(deck)

            sim_eval = SimulationEvaluator(
                num_games=self.num_games,
                seed=self.seed
            )
            sim_result = sim_eval.evaluate(deck)

            # Combine scores (weighted average)
            combined_score = 0.5 * heuristic_result.score + 0.5 * sim_result.score

            # Merge metrics
            metrics = {**heuristic_result.metrics, **sim_result.metrics}
            metrics["heuristic_score"] = heuristic_result.score
            metrics["simulation_score"] = sim_result.score

            return EvaluationResult(score=combined_score, metrics=metrics)

        else:
            raise ValueError(f"Unknown evaluation mode: {self.mode}")


def rank_decks(
    decks: List[Any],
    evaluator: DeckEvaluator
) -> List[Tuple[Any, EvaluationResult]]:
    """Rank multiple decks by evaluation score.

    Args:
        decks: List of Deck instances
        evaluator: DeckEvaluator to use

    Returns:
        List of (deck, result) tuples sorted by score (descending)
    """
    results = []
    for deck in decks:
        result = evaluator.evaluate(deck)
        results.append((deck, result))

    # Sort by score (descending)
    results.sort(key=lambda x: x[1].score, reverse=True)

    return results
