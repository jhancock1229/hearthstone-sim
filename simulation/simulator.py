"""Fast game rollout engine for running multiple agent matchups.

Simulator runs N games between two agents and collects comprehensive
statistics. Designed for:
- Performance evaluation
- Agent comparison
- Win rate calculation
- Statistical analysis

Features:
- Batch game execution
- Statistics collection (wins, avg turns, etc.)
- Reproducible results (seeding)
- Progress tracking
- Timeout protection
"""

import random
from dataclasses import dataclass, field
from typing import List, Optional, Callable, Dict, Any
from agents.base import Agent
from hearthstone.engine.game import Game
from hearthstone.engine import actions as game_actions
from simulation.observation import Observation
from simulation.action_space import Action, ActionSpace, ActionType
from deckbuilding.deck import random_deck, build_concrete_deck


@dataclass
class GameOutcome:
    """Result of a single game."""
    winner: int  # 0 or 1
    turns: int
    player1_final_health: int = 0
    player2_final_health: int = 0


@dataclass
class SimulationResult:
    """Aggregated results from multiple games.

    Attributes:
        total_games: Number of games played
        player1_wins: Number of wins for player 1
        player2_wins: Number of wins for player 2
        player1_win_rate: Win rate for player 1 (0.0 to 1.0)
        player2_win_rate: Win rate for player 2 (0.0 to 1.0)
        average_game_length: Average number of turns per game
        games: List of individual game outcomes
    """
    total_games: int
    player1_wins: int
    player2_wins: int
    player1_win_rate: float
    player2_win_rate: float
    average_game_length: float
    games: List[GameOutcome] = field(default_factory=list)

    def __str__(self) -> str:
        """Human-readable summary of results."""
        return (
            f"SimulationResult({self.total_games} games):\n"
            f"  Player 1: {self.player1_wins} wins ({self.player1_win_rate:.1%})\n"
            f"  Player 2: {self.player2_wins} wins ({self.player2_win_rate:.1%})\n"
            f"  Avg game length: {self.average_game_length:.1f} turns"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary.

        Returns:
            Dictionary representation of results
        """
        return {
            'total_games': self.total_games,
            'player1_wins': self.player1_wins,
            'player2_wins': self.player2_wins,
            'player1_win_rate': self.player1_win_rate,
            'player2_win_rate': self.player2_win_rate,
            'average_game_length': self.average_game_length,
            'games': [
                {
                    'winner': g.winner,
                    'turns': g.turns,
                    'player1_final_health': g.player1_final_health,
                    'player2_final_health': g.player2_final_health,
                }
                for g in self.games
            ]
        }


class Simulator:
    """Runs multiple games and collects statistics.

    The Simulator executes N games between two agents, managing game
    initialization, agent interaction, and result collection.
    """

    def __init__(
        self,
        seed: Optional[int] = None,
        max_turns: int = 200,
        verbose: bool = False,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ):
        """Initialize the Simulator.

        Args:
            seed: Random seed for reproducibility
            max_turns: Maximum turns per game (prevents infinite loops)
            verbose: Whether to print progress information
            progress_callback: Optional callback(completed, total) for progress
        """
        self.seed = seed
        self.max_turns = max_turns
        self.verbose = verbose
        self.progress_callback = progress_callback
        self.rng = random.Random(seed)

    def run_games(
        self,
        agent1: Agent,
        agent2: Agent,
        num_games: int,
        deck1_genotype: Optional[List[str]] = None,
        deck2_genotype: Optional[List[str]] = None,
        pool: Optional[Dict] = None,
    ) -> SimulationResult:
        """Run N games between two agents and collect results.

        Args:
            agent1: Agent for player 1
            agent2: Agent for player 2
            num_games: Number of games to play
            deck1_genotype: Optional card-name list for player 1's deck
            deck2_genotype: Optional card-name list for player 2's deck
            pool: Optional card pool dict (defaults to CARD_POOL)

        Returns:
            SimulationResult with aggregated statistics
        """
        games: List[GameOutcome] = []
        player1_wins = 0
        player2_wins = 0

        for game_num in range(num_games):
            # Reset agents for new game
            agent1.reset()
            agent2.reset()

            # Run single game
            outcome = self._run_single_game(
                agent1, agent2,
                deck1_genotype=deck1_genotype,
                deck2_genotype=deck2_genotype,
                pool=pool,
            )
            games.append(outcome)

            # Track wins
            if outcome.winner == 0:
                player1_wins += 1
            else:
                player2_wins += 1

            # Progress tracking
            if self.progress_callback:
                self.progress_callback(game_num + 1, num_games)

            if self.verbose and (game_num + 1) % 10 == 0:
                print(f"Completed {game_num + 1}/{num_games} games...")

        # Calculate statistics
        average_turns = sum(g.turns for g in games) / len(games)
        player1_win_rate = player1_wins / num_games
        player2_win_rate = player2_wins / num_games

        return SimulationResult(
            total_games=num_games,
            player1_wins=player1_wins,
            player2_wins=player2_wins,
            player1_win_rate=player1_win_rate,
            player2_win_rate=player2_win_rate,
            average_game_length=average_turns,
            games=games
        )

    def _run_single_game(
        self,
        agent1: Agent,
        agent2: Agent,
        deck1_genotype: Optional[List[str]] = None,
        deck2_genotype: Optional[List[str]] = None,
        pool: Optional[Dict] = None,
    ) -> GameOutcome:
        """Run a single game between two agents.

        Args:
            agent1: Agent for player 1
            agent2: Agent for player 2
            deck1_genotype: Optional card-name list for player 1's deck
            deck2_genotype: Optional card-name list for player 2's deck
            pool: Optional card pool dict (defaults to CARD_POOL)

        Returns:
            GameOutcome with game result
        """
        game = Game()
        agents = [agent1, agent2]

        # Initialize decks for both players (use simulator's RNG for determinism)
        g1 = deck1_genotype if deck1_genotype is not None else random_deck(size=30, rng=self.rng)
        g2 = deck2_genotype if deck2_genotype is not None else random_deck(size=30, rng=self.rng)
        build_kwargs = {"rng": self.rng}
        if pool is not None:
            build_kwargs["pool"] = pool
        game.player1.deck = build_concrete_deck(g1, **build_kwargs)
        game.player2.deck = build_concrete_deck(g2, **build_kwargs)

        # Draw starting hands (3 cards for each player, or 4 for second player)
        for _ in range(3):
            if game.player1.deck:
                game.player1.hand.append(game.player1.deck.pop(0))
        for _ in range(4):  # Second player gets 4 cards (going second advantage)
            if game.player2.deck:
                game.player2.hand.append(game.player2.deck.pop(0))

        # Game loop
        for turn_num in range(self.max_turns):
            # Start turn
            game.start_turn()

            # Determine active agent
            active_player_index = 0 if game.active_player == game.player1 else 1
            active_agent = agents[active_player_index]

            # Get player and opponent references
            player = game.active_player
            opponent = game.player2 if player == game.player1 else game.player1

            # Agent takes actions until ending turn
            # Safety limit: max 100 actions per turn (prevents infinite loops)
            max_actions_per_turn = 100
            actions_taken = 0

            while not game.is_over and actions_taken < max_actions_per_turn:
                actions_taken += 1

                # Get observation for active agent
                obs = self._get_observation(game, active_player_index)

                # Agent chooses action
                action = active_agent.choose_action(obs)

                # Execute action
                try:
                    if action.type == ActionType.END_TURN:
                        break  # Exit action loop, will call game.end_turn() below
                    elif action.type == ActionType.PLAY_CARD:
                        game_actions.play_card(
                            player,
                            card_index=action.card_index,
                            position=action.position,
                            opponent=opponent
                        )
                    elif action.type == ActionType.ATTACK:
                        # Execute attack (pass player and index, not minion object)
                        game_actions.attack(
                            attacker_player=player,
                            attacker_index=action.attacker_index,
                            defender_player=opponent,
                            defender_index=action.defender_index
                        )
                    elif action.type == ActionType.HERO_POWER:
                        game_actions.use_hero_power(player, game=game, target=action.target)
                except Exception as e:
                    # If action fails, just continue (agent chose invalid action)
                    if self.verbose:
                        print(f"Action failed: {e}")
                    continue

            # End the turn
            game.end_turn()

            # Check if game is over
            if game.is_over:
                winner = 0 if game.winner == game.player1 else 1
                return GameOutcome(
                    winner=winner,
                    turns=game.turn_number,
                    player1_final_health=game.player1.health,
                    player2_final_health=game.player2.health
                )

        # Max turns reached - determine winner by health
        if game.player1.health > game.player2.health:
            winner = 0
        elif game.player2.health > game.player1.health:
            winner = 1
        else:
            # Tie - random winner
            winner = self.rng.choice([0, 1])

        return GameOutcome(
            winner=winner,
            turns=self.max_turns,
            player1_final_health=game.player1.health,
            player2_final_health=game.player2.health
        )

    def _get_observation(self, game: Game, player_index: int) -> Observation:
        """Create an observation for the specified player.

        Args:
            game: The current game
            player_index: Which player's perspective (0 or 1)

        Returns:
            Observation for that player
        """
        player = game.player1 if player_index == 0 else game.player2
        opponent = game.player2 if player_index == 0 else game.player1

        # Get legal actions
        legal_actions = ActionSpace.get_legal_actions(game)

        return Observation(
            self_health=player.health,
            self_mana=player.mana,
            self_max_mana=player.max_mana,
            self_fatigue_counter=player.fatigue_counter,
            self_hand=[card for card in player.hand],
            self_deck_size=len(player.deck),
            self_board=[minion for minion in player.board],
            opponent_health=opponent.health,
            opponent_mana=opponent.mana,
            opponent_max_mana=opponent.max_mana,
            opponent_fatigue_counter=opponent.fatigue_counter,
            opponent_hand_size=len(opponent.hand),
            opponent_deck_size=len(opponent.deck),
            opponent_board=[minion for minion in opponent.board],
            turn_number=game.turn_number,
            is_my_turn=(game.active_player == player),
            is_game_over=game.is_over,
            winner=None if not game.is_over else (0 if game.winner == game.player1 else 1),
            player_index=player_index,
            legal_actions=legal_actions
        )
