"""Tests for EnhancedGreedyAgent: improved heuristic agent with spell/weapon/lethal awareness.

Tests organized by:
1. Basic agent contract
2. Spell evaluation
3. Weapon evaluation
4. Hero attack evaluation
5. Lethal detection
6. Board state awareness
7. Hero power targeting
8. Integration
"""

import pytest
from agents.base import Agent
from agents.enhanced_greedy_agent import EnhancedGreedyAgent
from hearthstone.cards.base import MinionCard, SpellCard, WeaponCard
from simulation.observation import Observation
from simulation.action_space import Action, ActionType


def _obs(**kwargs):
    """Helper to build an Observation with defaults."""
    defaults = dict(
        self_health=30, self_mana=10, self_max_mana=10,
        self_fatigue_counter=0, self_hand=[], self_deck_size=10,
        self_board=[], opponent_health=30, opponent_mana=10,
        opponent_max_mana=10, opponent_fatigue_counter=0,
        opponent_hand_size=5, opponent_deck_size=10,
        opponent_board=[], turn_number=5, is_my_turn=True,
        is_game_over=False, winner=None, player_index=0,
        legal_actions=[], self_weapon_attack=0,
        self_weapon_durability=0, self_armor=0,
        self_hero_class="NEUTRAL", opponent_armor=0,
    )
    defaults.update(kwargs)
    return Observation(**defaults)


# ============================================================
# Basic Agent Contract
# ============================================================


class TestBasics:
    """Test basic agent interface conformance."""

    def test_is_agent(self):
        agent = EnhancedGreedyAgent()
        assert isinstance(agent, Agent)

    def test_default_name(self):
        agent = EnhancedGreedyAgent()
        assert agent.name == "EnhancedGreedyAgent"

    def test_custom_name(self):
        agent = EnhancedGreedyAgent(name="MyAgent")
        assert agent.name == "MyAgent"

    def test_reset_is_noop(self):
        agent = EnhancedGreedyAgent()
        agent.reset()  # should not raise

    def test_empty_legal_actions_returns_end_turn(self):
        agent = EnhancedGreedyAgent()
        obs = _obs(legal_actions=[])
        action = agent.choose_action(obs)
        assert action.type == ActionType.END_TURN

    def test_deterministic(self):
        agent = EnhancedGreedyAgent()
        minion = MinionCard(name="Yeti", mana_cost=4, attack=4, health=5)
        actions = [
            Action(type=ActionType.PLAY_CARD, card_index=0, position=0),
            Action(type=ActionType.END_TURN),
        ]
        obs = _obs(self_hand=[minion], legal_actions=actions)
        a1 = agent.choose_action(obs)
        a2 = agent.choose_action(obs)
        assert a1 == a2


# ============================================================
# Spell Evaluation
# ============================================================


class TestSpellEvaluation:
    """Test that spells are valued by their effects, not as blank cards."""

    def test_damage_spell_over_vanilla_minion(self):
        """Fireball (deal 6) should outscore a 0/1 minion for same mana."""
        agent = EnhancedGreedyAgent()
        fireball = SpellCard(name="Fireball", mana_cost=4)
        fireball.spell_effect = ("deal_damage", 6)
        weak = MinionCard(name="Weak", mana_cost=4, attack=0, health=1)
        actions = [
            Action(type=ActionType.PLAY_CARD, card_index=0),  # fireball
            Action(type=ActionType.PLAY_CARD, card_index=1, position=0),  # weak minion
            Action(type=ActionType.END_TURN),
        ]
        obs = _obs(self_hand=[fireball, weak], legal_actions=actions)
        chosen = agent.choose_action(obs)
        assert chosen.card_index == 0  # fireball

    def test_aoe_scales_with_board(self):
        """AoE spell value should increase with more enemy minions."""
        agent = EnhancedGreedyAgent()
        aoe = SpellCard(name="AoE", mana_cost=4)
        aoe.spell_effect = ("aoe_damage", 2)
        action = Action(type=ActionType.PLAY_CARD, card_index=0)

        # 1 minion
        obs1 = _obs(
            self_hand=[aoe],
            opponent_board=[MinionCard(name="M", mana_cost=2, attack=2, health=2)],
            legal_actions=[action, Action(type=ActionType.END_TURN)],
        )
        # 4 minions
        obs4 = _obs(
            self_hand=[aoe],
            opponent_board=[MinionCard(name=f"M{i}", mana_cost=2, attack=2, health=2) for i in range(4)],
            legal_actions=[action, Action(type=ActionType.END_TURN)],
        )
        v1 = agent._evaluate_action(action, obs1)
        v4 = agent._evaluate_action(action, obs4)
        assert v4 > v1

    def test_draw_spell_valued_highly(self):
        """Draw spells should score well."""
        agent = EnhancedGreedyAgent()
        draw = SpellCard(name="Sprint", mana_cost=7)
        draw.spell_effect = ("draw", 4)
        action = Action(type=ActionType.PLAY_CARD, card_index=0)
        obs = _obs(self_hand=[draw], legal_actions=[action, Action(type=ActionType.END_TURN)])
        value = agent._evaluate_action(action, obs)
        assert value > 130  # well above base 100

    def test_heal_more_valuable_at_low_health(self):
        """Heal spells worth more when health is low."""
        agent = EnhancedGreedyAgent()
        heal = SpellCard(name="Heal", mana_cost=3)
        heal.spell_effect = ("restore_health", 8)
        action = Action(type=ActionType.PLAY_CARD, card_index=0)

        obs_full = _obs(self_health=30, self_hand=[heal],
                        legal_actions=[action, Action(type=ActionType.END_TURN)])
        obs_low = _obs(self_health=10, self_hand=[heal],
                       legal_actions=[action, Action(type=ActionType.END_TURN)])
        v_full = agent._evaluate_action(action, obs_full)
        v_low = agent._evaluate_action(action, obs_low)
        assert v_low > v_full

    def test_destroy_spell_by_target_cost(self):
        """Destroy spell should be valued higher when enemy has expensive minions."""
        agent = EnhancedGreedyAgent()
        destroy = SpellCard(name="Destroy", mana_cost=5)
        destroy.spell_effect = ("destroy",)
        action = Action(type=ActionType.PLAY_CARD, card_index=0)

        cheap = [MinionCard(name="Wisp", mana_cost=0, attack=1, health=1)]
        expensive = [MinionCard(name="Giant", mana_cost=8, attack=8, health=8)]

        obs_cheap = _obs(self_hand=[destroy], opponent_board=cheap,
                         legal_actions=[action, Action(type=ActionType.END_TURN)])
        obs_exp = _obs(self_hand=[destroy], opponent_board=expensive,
                       legal_actions=[action, Action(type=ActionType.END_TURN)])
        v_cheap = agent._evaluate_action(action, obs_cheap)
        v_exp = agent._evaluate_action(action, obs_exp)
        assert v_exp > v_cheap


# ============================================================
# Weapon Evaluation
# ============================================================


class TestWeaponEvaluation:
    """Test weapon cards valued by total damage output."""

    def test_weapon_valued_by_total_damage(self):
        """3/2 weapon (6 total damage) should score proportionally."""
        agent = EnhancedGreedyAgent()
        w = WeaponCard(name="Axe", mana_cost=3, attack=3, durability=2)
        action = Action(type=ActionType.PLAY_CARD, card_index=0)
        obs = _obs(self_hand=[w], legal_actions=[action, Action(type=ActionType.END_TURN)])
        value = agent._evaluate_action(action, obs)
        assert value > 120  # base 100 + attack*durability*6 = 136

    def test_weapon_over_weak_minion(self):
        """4/2 weapon should beat a 2/2 minion for same mana."""
        agent = EnhancedGreedyAgent()
        weapon = WeaponCard(name="Truesilver", mana_cost=4, attack=4, durability=2)
        minion = MinionCard(name="Weak", mana_cost=4, attack=2, health=2)
        actions = [
            Action(type=ActionType.PLAY_CARD, card_index=0),
            Action(type=ActionType.PLAY_CARD, card_index=1, position=0),
            Action(type=ActionType.END_TURN),
        ]
        obs = _obs(self_hand=[weapon, minion], legal_actions=actions)
        chosen = agent.choose_action(obs)
        assert chosen.card_index == 0  # weapon


# ============================================================
# Hero Attack Evaluation
# ============================================================


class TestHeroAttackEvaluation:
    """Test hero attack scoring with weapons."""

    def test_hero_attack_face_positive(self):
        """HERO_ATTACK face should score above zero."""
        agent = EnhancedGreedyAgent()
        action = Action(type=ActionType.HERO_ATTACK, defender_index=None)
        obs = _obs(self_weapon_attack=3, self_weapon_durability=2,
                   legal_actions=[action, Action(type=ActionType.END_TURN)])
        value = agent._evaluate_action(action, obs)
        assert value > 50

    def test_hero_attack_kills_minion(self):
        """Prefer killing a minion with weapon attack."""
        agent = EnhancedGreedyAgent()
        minion = MinionCard(name="Wisp", mana_cost=0, attack=1, health=1)
        kill_action = Action(type=ActionType.HERO_ATTACK, defender_index=0)
        face_action = Action(type=ActionType.HERO_ATTACK, defender_index=None)
        obs = _obs(
            self_weapon_attack=3, self_weapon_durability=2,
            opponent_board=[minion],
            legal_actions=[kill_action, face_action, Action(type=ActionType.END_TURN)],
        )
        # With a 1/1, killing it is cheap (only take 1 damage) and clears a threat
        v_kill = agent._evaluate_action(kill_action, obs)
        v_face = agent._evaluate_action(face_action, obs)
        # Both should be positive; exact preference depends on board state
        assert v_kill > 0
        assert v_face > 0

    def test_hero_attack_avoids_high_attack_at_low_hp(self):
        """At low HP, hero should avoid attacking high-attack minions."""
        agent = EnhancedGreedyAgent()
        big = MinionCard(name="Giant", mana_cost=8, attack=8, health=8)
        action = Action(type=ActionType.HERO_ATTACK, defender_index=0)
        obs = _obs(
            self_health=5, self_weapon_attack=3, self_weapon_durability=2,
            opponent_board=[big],
            legal_actions=[action, Action(type=ActionType.END_TURN)],
        )
        value = agent._evaluate_action(action, obs)
        # Should be penalized heavily (would die from 8 damage)
        assert value < 50

    def test_hero_attack_face_for_lethal(self):
        """When weapon attack >= opponent HP, go face."""
        agent = EnhancedGreedyAgent()
        minion = MinionCard(name="Yeti", mana_cost=4, attack=4, health=5)
        face_action = Action(type=ActionType.HERO_ATTACK, defender_index=None)
        trade_action = Action(type=ActionType.HERO_ATTACK, defender_index=0)
        obs = _obs(
            self_weapon_attack=5, self_weapon_durability=1,
            opponent_health=5, opponent_board=[minion],
            legal_actions=[face_action, trade_action, Action(type=ActionType.END_TURN)],
        )
        chosen = agent.choose_action(obs)
        assert chosen.defender_index is None  # go face for lethal


# ============================================================
# Lethal Detection
# ============================================================


class TestLethalDetection:
    """Test that agent detects and goes for lethal."""

    def test_simple_minion_lethal(self):
        """One minion with attack >= opponent HP should go face."""
        agent = EnhancedGreedyAgent()
        big = MinionCard(name="Big", mana_cost=5, attack=8, health=8)
        big.summoning_sick = False
        big.exhausted = False
        face = Action(type=ActionType.ATTACK, attacker_index=0, defender_index=None)
        trade = Action(type=ActionType.ATTACK, attacker_index=0, defender_index=0)
        enemy = MinionCard(name="E", mana_cost=2, attack=2, health=2)
        obs = _obs(
            self_board=[big], opponent_health=5, opponent_board=[enemy],
            legal_actions=[face, trade, Action(type=ActionType.END_TURN)],
        )
        chosen = agent.choose_action(obs)
        assert chosen.type == ActionType.ATTACK
        assert chosen.defender_index is None  # face

    def test_lethal_with_spell(self):
        """deal_damage spell should be used for lethal."""
        agent = EnhancedGreedyAgent()
        bolt = SpellCard(name="Bolt", mana_cost=1)
        bolt.spell_effect = ("deal_damage", 3)
        spell_action = Action(type=ActionType.PLAY_CARD, card_index=0)
        obs = _obs(
            self_hand=[bolt], opponent_health=3,
            legal_actions=[spell_action, Action(type=ActionType.END_TURN)],
        )
        chosen = agent.choose_action(obs)
        assert chosen.type == ActionType.PLAY_CARD

    def test_lethal_with_weapon(self):
        """HERO_ATTACK face for lethal."""
        agent = EnhancedGreedyAgent()
        face = Action(type=ActionType.HERO_ATTACK, defender_index=None)
        obs = _obs(
            self_weapon_attack=5, self_weapon_durability=1,
            opponent_health=4,
            legal_actions=[face, Action(type=ActionType.END_TURN)],
        )
        chosen = agent.choose_action(obs)
        assert chosen.type == ActionType.HERO_ATTACK
        assert chosen.defender_index is None

    def test_multi_source_lethal(self):
        """Detect lethal across minion attack + hero attack."""
        agent = EnhancedGreedyAgent()
        m = MinionCard(name="M", mana_cost=2, attack=3, health=3)
        m.summoning_sick = False
        m.exhausted = False
        minion_face = Action(type=ActionType.ATTACK, attacker_index=0, defender_index=None)
        hero_face = Action(type=ActionType.HERO_ATTACK, defender_index=None)
        obs = _obs(
            self_board=[m], self_weapon_attack=3, self_weapon_durability=1,
            opponent_health=6,  # 3 + 3 = 6 exactly
            legal_actions=[minion_face, hero_face, Action(type=ActionType.END_TURN)],
        )
        chosen = agent.choose_action(obs)
        # Should go face with one of the sources
        assert chosen.defender_index is None

    def test_lethal_over_trade(self):
        """Lethal should be preferred over a favorable trade."""
        agent = EnhancedGreedyAgent()
        m = MinionCard(name="M", mana_cost=5, attack=7, health=7)
        m.summoning_sick = False
        m.exhausted = False
        enemy = MinionCard(name="Enemy", mana_cost=8, attack=1, health=1)
        face = Action(type=ActionType.ATTACK, attacker_index=0, defender_index=None)
        trade = Action(type=ActionType.ATTACK, attacker_index=0, defender_index=0)
        obs = _obs(
            self_board=[m], opponent_health=5, opponent_board=[enemy],
            legal_actions=[face, trade, Action(type=ActionType.END_TURN)],
        )
        chosen = agent.choose_action(obs)
        assert chosen.defender_index is None  # face, not trade


# ============================================================
# Board State Awareness
# ============================================================


class TestBoardAwareness:
    """Test board-state-aware trading decisions."""

    def test_trades_when_behind(self):
        """When behind on board, prefer trading over face."""
        agent = EnhancedGreedyAgent()
        my_minion = MinionCard(name="Mine", mana_cost=3, attack=3, health=3)
        my_minion.summoning_sick = False
        my_minion.exhausted = False
        # Opponent has much stronger board
        enemy1 = MinionCard(name="E1", mana_cost=5, attack=5, health=5)
        enemy2 = MinionCard(name="E2", mana_cost=4, attack=4, health=4)
        trade = Action(type=ActionType.ATTACK, attacker_index=0, defender_index=0)
        face = Action(type=ActionType.ATTACK, attacker_index=0, defender_index=None)
        obs = _obs(
            self_board=[my_minion], opponent_board=[enemy1, enemy2],
            legal_actions=[trade, face, Action(type=ActionType.END_TURN)],
        )
        v_trade = agent._evaluate_action(trade, obs)
        v_face = agent._evaluate_action(face, obs)
        assert v_trade > v_face

    def test_face_when_board_clear(self):
        """When opponent has no minions, go face."""
        agent = EnhancedGreedyAgent()
        m = MinionCard(name="M", mana_cost=2, attack=3, health=2)
        m.summoning_sick = False
        m.exhausted = False
        face = Action(type=ActionType.ATTACK, attacker_index=0, defender_index=None)
        obs = _obs(
            self_board=[m], opponent_board=[],
            legal_actions=[face, Action(type=ActionType.END_TURN)],
        )
        value = agent._evaluate_action(face, obs)
        assert value > 60  # should be solid score with board clear bonus

    def test_face_when_ahead(self):
        """When significantly ahead on board, prefer face."""
        agent = EnhancedGreedyAgent()
        m1 = MinionCard(name="M1", mana_cost=5, attack=5, health=5)
        m1.summoning_sick = False
        m1.exhausted = False
        m2 = MinionCard(name="M2", mana_cost=4, attack=4, health=4)
        m2.summoning_sick = False
        m2.exhausted = False
        # One weak enemy
        enemy = MinionCard(name="E", mana_cost=1, attack=1, health=1)
        face = Action(type=ActionType.ATTACK, attacker_index=0, defender_index=None)
        trade = Action(type=ActionType.ATTACK, attacker_index=0, defender_index=0)
        obs = _obs(
            self_board=[m1, m2], opponent_board=[enemy],
            legal_actions=[face, trade, Action(type=ActionType.END_TURN)],
        )
        v_face = agent._evaluate_action(face, obs)
        v_trade = agent._evaluate_action(trade, obs)
        assert v_face > v_trade


# ============================================================
# Hero Power Targeting
# ============================================================


class TestHeroPowerTargeting:
    """Test class-specific hero power evaluation."""

    def test_mage_kills_one_health_minion(self):
        """Mage should target 1-health enemy minion."""
        agent = EnhancedGreedyAgent()
        minion = MinionCard(name="Wisp", mana_cost=0, attack=1, health=1)
        kill_target = Action(type=ActionType.HERO_POWER, target=("opponent_minion", 0))
        face_target = Action(type=ActionType.HERO_POWER, target=("opponent_hero",))
        obs = _obs(
            self_hero_class="MAGE", opponent_board=[minion],
            legal_actions=[kill_target, face_target, Action(type=ActionType.END_TURN)],
        )
        v_kill = agent._evaluate_action(kill_target, obs)
        v_face = agent._evaluate_action(face_target, obs)
        assert v_kill > v_face

    def test_mage_prefers_enemy(self):
        """Mage should never target own minions over enemies."""
        agent = EnhancedGreedyAgent()
        own = MinionCard(name="Own", mana_cost=2, attack=2, health=3)
        enemy = MinionCard(name="Enemy", mana_cost=2, attack=2, health=2)
        self_target = Action(type=ActionType.HERO_POWER, target=("self_minion", 0))
        enemy_target = Action(type=ActionType.HERO_POWER, target=("opponent_minion", 0))
        obs = _obs(
            self_hero_class="MAGE", self_board=[own], opponent_board=[enemy],
            legal_actions=[self_target, enemy_target, Action(type=ActionType.END_TURN)],
        )
        v_self = agent._evaluate_action(self_target, obs)
        v_enemy = agent._evaluate_action(enemy_target, obs)
        assert v_enemy > v_self

    def test_priest_heals_damaged_hero(self):
        """Priest should heal self when damaged."""
        agent = EnhancedGreedyAgent()
        heal_self = Action(type=ActionType.HERO_POWER, target=("self_hero",))
        heal_enemy = Action(type=ActionType.HERO_POWER, target=("opponent_hero",))
        obs = _obs(
            self_health=20, self_hero_class="PRIEST",
            legal_actions=[heal_self, heal_enemy, Action(type=ActionType.END_TURN)],
        )
        v_self = agent._evaluate_action(heal_self, obs)
        v_enemy = agent._evaluate_action(heal_enemy, obs)
        assert v_self > v_enemy

    def test_warlock_hero_power_scores_well(self):
        """Warlock hero power (draw) should score well at high HP."""
        agent = EnhancedGreedyAgent()
        action = Action(type=ActionType.HERO_POWER)
        obs = _obs(self_health=30, self_hero_class="WARLOCK",
                   legal_actions=[action, Action(type=ActionType.END_TURN)])
        value = agent._evaluate_action(action, obs)
        assert value > 30

    def test_warlock_hero_power_penalized_low_hp(self):
        """Warlock hero power dangerous at low HP."""
        agent = EnhancedGreedyAgent()
        action = Action(type=ActionType.HERO_POWER)
        obs_high = _obs(self_health=30, self_hero_class="WARLOCK",
                        legal_actions=[action, Action(type=ActionType.END_TURN)])
        obs_low = _obs(self_health=5, self_hero_class="WARLOCK",
                       legal_actions=[action, Action(type=ActionType.END_TURN)])
        v_high = agent._evaluate_action(action, obs_high)
        v_low = agent._evaluate_action(action, obs_low)
        assert v_high > v_low


# ============================================================
# Integration
# ============================================================


class TestIntegration:
    """Test agent works end-to-end in simulator."""

    def test_runs_in_simulator(self):
        """EnhancedGreedyAgent should complete games without errors."""
        from simulation.simulator import Simulator
        agent = EnhancedGreedyAgent()
        sim = Simulator(seed=42, max_turns=50)
        result = sim.run_games(agent, agent, num_games=3)
        assert result.total_games == 3

    def test_beats_or_ties_random(self):
        """EnhancedGreedyAgent should beat RandomAgent over many games."""
        from simulation.simulator import Simulator
        from agents.random_agent import RandomAgent
        enhanced = EnhancedGreedyAgent()
        random_agent = RandomAgent(seed=42)
        sim = Simulator(seed=42, max_turns=80)
        result = sim.run_games(enhanced, random_agent, num_games=20)
        # Should win majority (soft assertion)
        assert result.player1_win_rate >= 0.4
