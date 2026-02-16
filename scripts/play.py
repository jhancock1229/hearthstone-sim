#!/usr/bin/env python3
"""Interactive game: human vs agent via command line.

Usage:
    python scripts/play.py
    python scripts/play.py --opponent random
    python scripts/play.py --opponent random --seed 42

This script allows you to play a Hearthstone game against an AI agent
through the command line interface.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hearthstone.engine.game import Game
from agents.random_agent import RandomAgent
from agents.base import Agent
from simulation.observation import Observation
from simulation.action_space import Action, ActionType


class HumanAgent(Agent):
    """Interactive agent for human players."""

    def __init__(self):
        """Initialize human agent."""
        super().__init__(name="Human")

    def choose_action(self, observation: Observation) -> Action:
        """Prompt human for action choice.

        Args:
            observation: Current game state

        Returns:
            Action chosen by human
        """
        print("\n" + "=" * 60)
        print("YOUR TURN")
        print("=" * 60)
        print(f"Your Health: {observation.self_health}")
        print(f"Your Mana: {observation.self_mana}/{observation.self_max_mana}")
        print(f"Opponent Health: {observation.opponent_health}")
        print(f"Opponent Mana: {observation.opponent_mana}/{observation.opponent_max_mana}")
        print()

        print(f"Hand ({len(observation.self_hand)} cards):")
        for i, card in enumerate(observation.self_hand):
            print(f"  {i}: {card.name} ({card.mana_cost} mana)")

        print(f"\nBoard ({len(observation.self_board)} minions):")
        for i, minion in enumerate(observation.self_board):
            print(f"  {i}: {minion.name} ({minion.attack}/{minion.health})")

        print(f"\nOpponent Board ({len(observation.opponent_board)} minions):")
        for i, minion in enumerate(observation.opponent_board):
            print(f"  {i}: {minion.name} ({minion.attack}/{minion.health})")

        print("\nAvailable Actions:")
        for i, action in enumerate(observation.legal_actions):
            action_desc = self._describe_action(action)
            print(f"  {i}: {action_desc}")

        while True:
            try:
                choice = input("\nChoose action (number): ").strip()
                action_idx = int(choice)

                if 0 <= action_idx < len(observation.legal_actions):
                    return observation.legal_actions[action_idx]
                else:
                    print(f"Invalid choice. Please enter 0-{len(observation.legal_actions) - 1}")
            except (ValueError, EOFError, KeyboardInterrupt):
                print("\nEnding turn...")
                return Action(type=ActionType.END_TURN)

    def _describe_action(self, action: Action) -> str:
        """Get human-readable description of action.

        Args:
            action: The action to describe

        Returns:
            String description
        """
        if action.type == ActionType.END_TURN:
            return "End Turn"
        elif action.type == ActionType.HERO_POWER:
            return "Use Hero Power"
        elif action.type == ActionType.PLAY_CARD:
            return f"Play Card {action.card_index}"
        elif action.type == ActionType.ATTACK:
            if action.defender_index is None:
                return f"Attack enemy hero with minion {action.attacker_index}"
            else:
                return f"Attack minion {action.defender_index} with minion {action.attacker_index}"
        return str(action)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Play Hearthstone against an AI agent"
    )

    parser.add_argument(
        '--opponent',
        type=str,
        default='random',
        choices=['random'],
        help='Opponent agent type (default: random)'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility (default: None)'
    )

    return parser.parse_args()


def create_opponent(agent_type: str, seed: int = None):
    """Create opponent agent.

    Args:
        agent_type: Type of opponent
        seed: Optional random seed

    Returns:
        Agent instance
    """
    if agent_type == 'random':
        return RandomAgent(seed=seed, name="RandomBot")
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


def print_game_state(game: Game):
    """Print current game state.

    Args:
        game: The game instance
    """
    print("\n" + "-" * 60)
    print(f"Turn {game.turn_number}")
    print(f"Active Player: {'You' if game.active_player == game.player1 else 'Opponent'}")
    print("-" * 60)


def print_game_result(game: Game):
    """Print final game result.

    Args:
        game: The finished game
    """
    print("\n" + "=" * 60)
    print("GAME OVER")
    print("=" * 60)

    if game.winner == game.player1:
        print("🎉 YOU WIN! 🎉")
    else:
        print("❌ YOU LOSE ❌")

    print(f"\nFinal Health:")
    print(f"  You: {game.player1.health}")
    print(f"  Opponent: {game.player2.health}")
    print(f"\nGame lasted {game.turn_number} turns")
    print("=" * 60)


def main():
    """Main entry point."""
    args = parse_args()

    print("=" * 60)
    print("HEARTHSTONE SIMULATOR")
    print("=" * 60)
    print()

    # Create agents
    human = HumanAgent()
    opponent = create_opponent(args.opponent, args.seed)

    print(f"You are playing against: {opponent.name}")
    if args.seed is not None:
        print(f"Game seed: {args.seed}")
    print()
    input("Press Enter to start...")

    # Create and run game
    game = Game()
    agents = [human, opponent]

    # Main game loop
    while not game.is_over:
        game.start_turn()
        print_game_state(game)

        # Determine active player
        active_player_index = 0 if game.active_player == game.player1 else 1
        active_agent = agents[active_player_index]

        # Agent turn (simplified - just ends turn for now)
        if active_player_index == 0:
            # Human turn - would get action from human
            print("\nYour turn (actions not fully implemented yet)")
            input("Press Enter to end turn...")
        else:
            # AI turn
            print(f"\n{opponent.name}'s turn...")
            # Would execute AI actions here
            input("Press Enter to continue...")

        game.end_turn()

    # Game over
    print_game_result(game)


if __name__ == '__main__':
    main()
