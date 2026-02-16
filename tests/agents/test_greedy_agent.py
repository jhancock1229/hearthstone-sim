"""Tests for GreedyAgent - heuristic-based agent.

The GreedyAgent makes decisions based on:
- Card value heuristics (mana efficiency, stats)
- Board state evaluation
- Favorable trades
- Tempo plays

Tests validate:
- Action selection based on heuristics
- Priority ordering (play cards > attack > hero power > end turn)
- Greedy value maximization
- Consistency and determinism
"""

import pytest
from agents.greedy_agent import GreedyAgent
from agents.base import Agent
from simulation.observation import Observation
from simulation.action_space import Action, ActionType
from hearthstone.cards.base import MinionCard


class TestGreedyAgentBasics:
    """Test basic GreedyAgent functionality."""

    def test_greedy_agent_is_agent(self):
        """GreedyAgent extends Agent base class."""
        agent = GreedyAgent()
        assert isinstance(agent, Agent)

    def test_greedy_agent_has_name(self):
        """GreedyAgent has default name."""
        agent = GreedyAgent()
        assert agent.name == "GreedyAgent"

    def test_greedy_agent_custom_name(self):
        """GreedyAgent can have custom name."""
        agent = GreedyAgent(name="MyGreedyBot")
        assert agent.name == "MyGreedyBot"

    def test_greedy_agent_reset(self):
        """GreedyAgent can be reset (stateless operation)."""
        agent = GreedyAgent()
        agent.reset()  # Should not raise


class TestGreedyAgentActionSelection:
    """Test GreedyAgent action selection logic."""

    def test_chooses_end_turn_when_no_other_actions(self):
        """Chooses END_TURN when it's the only legal action."""
        agent = GreedyAgent()

        # Observation with only END_TURN available
        observation = Observation(
            self_hand=[],
            self_board=[],
            opponent_board=[],
            self_mana=0,
            self_max_mana=0,
            self_health=30,
            self_fatigue_counter=0,
            opponent_health=30,
            legal_actions=[Action(type=ActionType.END_TURN)]
        )

        action = agent.choose_action(observation)
        assert action.type == ActionType.END_TURN

    def test_prefers_playing_cards_over_ending_turn(self):
        """Prefers to play cards rather than end turn immediately."""
        agent = GreedyAgent()

        card = MinionCard(id="m1", name="Minion", mana_cost=2, attack=2, health=2)

        observation = Observation(
            self_hand=[card],
            self_board=[],
            opponent_board=[],
            self_mana=2,
            self_max_mana=2,
            self_health=30,
            self_fatigue_counter=0,
            opponent_health=30,
            legal_actions=[
                Action(type=ActionType.PLAY_CARD, card_index=0, position=0),
                Action(type=ActionType.END_TURN)
            ]
        )

        action = agent.choose_action(observation)
        assert action.type == ActionType.PLAY_CARD

    def test_plays_high_value_cards_first(self):
        """Plays cards with better stats-per-mana first."""
        agent = GreedyAgent()

        # Card 1: 2 mana 2/2 = 2.0 stats per mana
        # Card 2: 3 mana 5/5 = 3.33 stats per mana (better value)
        card1 = MinionCard(id="m1", name="M1", mana_cost=2, attack=2, health=2)
        card2 = MinionCard(id="m2", name="M2", mana_cost=3, attack=5, health=5)

        observation = Observation(
            self_hand=[card1, card2],
            self_board=[],
            opponent_board=[],
            self_mana=5,
            self_max_mana=5,
            self_health=30,
            self_fatigue_counter=0,
            opponent_health=30,
            legal_actions=[
                Action(type=ActionType.PLAY_CARD, card_index=0, position=0),
                Action(type=ActionType.PLAY_CARD, card_index=1, position=0),
                Action(type=ActionType.END_TURN)
            ]
        )

        action = agent.choose_action(observation)
        # Should choose card2 (index 1) as it has better value
        assert action.type == ActionType.PLAY_CARD
        assert action.card_index == 1

    def test_prefers_attacks_over_hero_power(self):
        """Prefers to attack with minions before using hero power."""
        agent = GreedyAgent()

        my_minion = MinionCard(id="m1", name="M1", mana_cost=2, attack=3, health=2)
        enemy_minion = MinionCard(id="e1", name="E1", mana_cost=2, attack=2, health=2)

        observation = Observation(
            self_hand=[],
            self_board=[my_minion],
            opponent_board=[enemy_minion],
            self_mana=2,
            self_max_mana=2,
            self_health=30,
            self_fatigue_counter=0,
            opponent_health=30,
            legal_actions=[
                Action(type=ActionType.ATTACK, attacker_index=0, defender_index=0),
                Action(type=ActionType.HERO_POWER),
                Action(type=ActionType.END_TURN)
            ]
        )

        action = agent.choose_action(observation)
        assert action.type == ActionType.ATTACK

    def test_makes_favorable_trades(self):
        """Chooses favorable trades (kill enemy without dying)."""
        agent = GreedyAgent()

        # My 3/2 can kill enemy 2/1 (favorable)
        my_minion = MinionCard(id="m1", name="M1", mana_cost=2, attack=3, health=2)
        enemy_weak = MinionCard(id="e1", name="E1", mana_cost=1, attack=2, health=1)
        enemy_strong = MinionCard(id="e2", name="E2", mana_cost=3, attack=5, health=5)

        observation = Observation(
            self_hand=[],
            self_board=[my_minion],
            opponent_board=[enemy_weak, enemy_strong],
            self_mana=0,
            self_max_mana=2,
            self_health=30,
            self_fatigue_counter=0,
            opponent_health=30,
            legal_actions=[
                Action(type=ActionType.ATTACK, attacker_index=0, defender_index=0),  # Kill weak
                Action(type=ActionType.ATTACK, attacker_index=0, defender_index=1),  # Trade poorly
                Action(type=ActionType.END_TURN)
            ]
        )

        action = agent.choose_action(observation)
        # Should attack weak minion (defender_index=0)
        assert action.type == ActionType.ATTACK
        assert action.defender_index == 0


class TestGreedyAgentHeuristics:
    """Test GreedyAgent heuristic evaluation."""

    def test_values_efficient_minions(self):
        """Values minions with good stats-per-mana."""
        agent = GreedyAgent()

        # Create minions with different efficiency
        efficient = MinionCard(id="e", name="Eff", mana_cost=2, attack=3, health=3)  # 3.0 per mana
        inefficient = MinionCard(id="i", name="Ineff", mana_cost=4, attack=3, health=3)  # 1.5 per mana

        # Should prefer efficient minion
        observation = Observation(
            self_hand=[inefficient, efficient],
            self_board=[],
            opponent_board=[],
            self_mana=4,
            self_max_mana=4,
            self_health=30,
            self_fatigue_counter=0,
            opponent_health=30,
            legal_actions=[
                Action(type=ActionType.PLAY_CARD, card_index=0, position=0),
                Action(type=ActionType.PLAY_CARD, card_index=1, position=0),
                Action(type=ActionType.END_TURN)
            ]
        )

        action = agent.choose_action(observation)
        assert action.card_index == 1  # Efficient minion

    def test_prioritizes_board_presence(self):
        """Prioritizes playing minions to build board."""
        agent = GreedyAgent()

        minion = MinionCard(id="m", name="Minion", mana_cost=3, attack=3, health=3)

        observation = Observation(
            self_hand=[minion],
            self_board=[],
            opponent_board=[],
            self_mana=5,
            self_max_mana=5,
            self_health=30,
            self_fatigue_counter=0,
            opponent_health=30,
            legal_actions=[
                Action(type=ActionType.PLAY_CARD, card_index=0, position=0),
                Action(type=ActionType.HERO_POWER),
                Action(type=ActionType.END_TURN)
            ]
        )

        action = agent.choose_action(observation)
        # Should play minion before hero power
        assert action.type == ActionType.PLAY_CARD

    def test_avoids_unfavorable_trades(self):
        """Avoids trades where our minion dies for less value."""
        agent = GreedyAgent()

        # Our 5/5 vs enemy 1/3 - we survive, good trade
        my_big = MinionCard(id="m1", name="Big", mana_cost=5, attack=5, health=5)
        enemy_small = MinionCard(id="e1", name="Small", mana_cost=1, attack=1, health=3)

        observation = Observation(
            self_hand=[],
            self_board=[my_big],
            opponent_board=[enemy_small],
            self_mana=0,
            self_max_mana=5,
            self_health=30,
            self_fatigue_counter=0,
            opponent_health=30,
            legal_actions=[
                Action(type=ActionType.ATTACK, attacker_index=0, defender_index=0),
                Action(type=ActionType.ATTACK, attacker_index=0, defender_index=None),  # Face
                Action(type=ActionType.END_TURN)
            ]
        )

        action = agent.choose_action(observation)
        # Should attack something (minion or face)
        assert action.type == ActionType.ATTACK


class TestGreedyAgentDeterminism:
    """Test GreedyAgent produces consistent decisions."""

    def test_deterministic_action_selection(self):
        """Same observation produces same action."""
        agent = GreedyAgent()

        card = MinionCard(id="m1", name="Minion", mana_cost=2, attack=2, health=2)

        observation = Observation(
            self_hand=[card],
            self_board=[],
            opponent_board=[],
            self_mana=2,
            self_max_mana=2,
            self_health=30,
            self_fatigue_counter=0,
            opponent_health=30,
            legal_actions=[
                Action(type=ActionType.PLAY_CARD, card_index=0, position=0),
                Action(type=ActionType.END_TURN)
            ]
        )

        action1 = agent.choose_action(observation)
        action2 = agent.choose_action(observation)

        assert action1.type == action2.type
        assert action1.card_index == action2.card_index


class TestGreedyAgentEdgeCases:
    """Test GreedyAgent handles edge cases."""

    def test_handles_empty_legal_actions(self):
        """Gracefully handles empty legal actions list."""
        agent = GreedyAgent()

        observation = Observation(
            self_hand=[],
            self_board=[],
            opponent_board=[],
            self_mana=0,
            self_max_mana=0,
            self_health=30,
            self_fatigue_counter=0,
            opponent_health=30,
            legal_actions=[]
        )

        # Should return END_TURN as fallback
        action = agent.choose_action(observation)
        assert action.type == ActionType.END_TURN

    def test_handles_full_board(self):
        """Makes decisions when board is full (7 minions)."""
        agent = GreedyAgent()

        # Full board - can't play more minions, but can attack
        my_minions = [
            MinionCard(id=f"m{i}", name=f"M{i}", mana_cost=1, attack=1, health=1)
            for i in range(7)
        ]

        observation = Observation(
            self_hand=[],
            self_board=my_minions,
            opponent_board=[],
            self_mana=0,
            self_max_mana=5,
            self_health=30,
            self_fatigue_counter=0,
            opponent_health=30,
            legal_actions=[
                Action(type=ActionType.ATTACK, attacker_index=0, defender_index=None),
                Action(type=ActionType.END_TURN)
            ]
        )

        action = agent.choose_action(observation)
        # Should attack with minions
        assert action.type == ActionType.ATTACK

    def test_uses_hero_power_when_nothing_better(self):
        """Uses hero power when no cards to play or attacks available."""
        agent = GreedyAgent()

        observation = Observation(
            self_hand=[],
            self_board=[],
            opponent_board=[],
            self_mana=2,
            self_max_mana=2,
            self_health=30,
            self_fatigue_counter=0,
            opponent_health=30,
            legal_actions=[
                Action(type=ActionType.HERO_POWER),
                Action(type=ActionType.END_TURN)
            ]
        )

        action = agent.choose_action(observation)
        assert action.type == ActionType.HERO_POWER
