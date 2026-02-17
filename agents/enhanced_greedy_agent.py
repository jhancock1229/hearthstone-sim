"""EnhancedGreedyAgent - improved heuristic agent with spell/weapon/lethal awareness.

Improvements over GreedyAgent:
- Lethal detection: recognizes when it can kill the opponent this turn
- Card-type-aware scoring: spells valued by effect, weapons by total damage
- Hero attack evaluation: proper weapon swing scoring
- Board-state-aware trading: prefers trades when behind, face when ahead
- Class-specific hero power targeting: Mage pings 1-health minions, etc.

Remains stateless, deterministic, and fast (no tree search).
"""

from typing import Optional
from agents.base import Agent
from simulation.observation import Observation
from simulation.action_space import Action, ActionType


class EnhancedGreedyAgent(Agent):
    """Agent that selects actions using improved heuristics with lethal detection."""

    def __init__(self, name: Optional[str] = None):
        super().__init__(name=name or "EnhancedGreedyAgent")

    def choose_action(self, observation: Observation) -> Action:
        if not observation.legal_actions:
            return Action(type=ActionType.END_TURN)

        # 1. Check for lethal
        lethal = self._check_lethal(observation)
        if lethal is not None:
            return lethal

        # 2. Score all actions, pick max
        best_action = None
        best_value = float('-inf')
        for action in observation.legal_actions:
            value = self._evaluate_action(action, observation)
            if value > best_value:
                best_value = value
                best_action = action
        return best_action

    # ------------------------------------------------------------------
    # Lethal detection
    # ------------------------------------------------------------------

    def _check_lethal(self, obs: Observation) -> Optional[Action]:
        """If total available face damage >= opponent HP, return best face action."""
        opponent_hp = obs.opponent_health + obs.opponent_armor
        total_face = 0
        face_actions = []

        for action in obs.legal_actions:
            dmg = self._face_damage(action, obs)
            if dmg > 0:
                total_face += dmg
                face_actions.append((action, dmg))

        if total_face >= opponent_hp and face_actions:
            # Return highest-damage face action first
            face_actions.sort(key=lambda x: x[1], reverse=True)
            return face_actions[0][0]
        return None

    def _face_damage(self, action: Action, obs: Observation) -> int:
        """Return face damage this action would deal, or 0."""
        if action.type == ActionType.ATTACK and action.defender_index is None:
            idx = action.attacker_index
            if idx is not None and idx < len(obs.self_board):
                return obs.self_board[idx].attack
        elif action.type == ActionType.HERO_ATTACK and action.defender_index is None:
            return obs.self_weapon_attack
        elif action.type == ActionType.PLAY_CARD:
            card = self._get_card(action, obs)
            if card is None:
                return 0
            effect = getattr(card, 'spell_effect', None)
            if effect and effect[0] == 'deal_damage':
                return effect[1]
            # CHARGE minions can attack immediately
            if (hasattr(card, 'mechanics') and 'CHARGE' in card.mechanics
                    and hasattr(card, 'attack')):
                return card.attack
        return 0

    # ------------------------------------------------------------------
    # Action dispatch
    # ------------------------------------------------------------------

    def _evaluate_action(self, action: Action, obs: Observation) -> float:
        if action.type == ActionType.PLAY_CARD:
            return self._evaluate_play_card(action, obs)
        elif action.type == ActionType.ATTACK:
            return self._evaluate_attack(action, obs)
        elif action.type == ActionType.HERO_ATTACK:
            return self._evaluate_hero_attack(action, obs)
        elif action.type == ActionType.HERO_POWER:
            return self._evaluate_hero_power(action, obs)
        elif action.type == ActionType.END_TURN:
            return 0.0
        return 0.0

    # ------------------------------------------------------------------
    # Play card
    # ------------------------------------------------------------------

    def _get_card(self, action: Action, obs: Observation):
        idx = action.card_index
        if idx is None or idx >= len(obs.self_hand):
            return None
        return obs.self_hand[idx]

    def _evaluate_play_card(self, action: Action, obs: Observation) -> float:
        card = self._get_card(action, obs)
        if card is None:
            return 0.0

        # Weapon card (has durability, not health)
        if hasattr(card, 'durability') and not hasattr(card, 'health'):
            return self._evaluate_play_weapon(card, obs)

        # Minion card (has attack and health)
        if hasattr(card, 'attack') and hasattr(card, 'health'):
            return self._evaluate_play_minion(card, obs)

        # Spell card
        return self._evaluate_play_spell(card, obs)

    def _evaluate_play_minion(self, card, obs: Observation) -> float:
        value = 100.0
        mana = max(card.mana_cost, 1)
        stats = card.attack + card.health
        value += (stats / mana) * 10.0
        value += card.mana_cost * 2.0

        # Keyword bonuses
        mechs = getattr(card, 'mechanics', []) or []
        if "TAUNT" in mechs:
            value += 15.0
        if "CHARGE" in mechs:
            value += 10.0
        if "DIVINE_SHIELD" in mechs:
            value += 8.0
        if "LIFESTEAL" in mechs:
            value += 6.0
        if "RUSH" in mechs:
            value += 5.0
        if "WINDFURY" in mechs:
            value += 4.0

        # Battlecry value
        bc = getattr(card, 'battlecry_effect', None)
        if bc:
            value += self._evaluate_effect(bc, obs) * 0.8

        # Deathrattle value (discounted: delayed effect)
        dr = getattr(card, 'deathrattle_effect', None)
        if dr:
            value += self._evaluate_effect(dr, obs) * 0.5

        return value

    def _evaluate_play_spell(self, card, obs: Observation) -> float:
        value = 100.0
        value += card.mana_cost * 2.0

        effect = getattr(card, 'spell_effect', None)
        if effect:
            value += self._evaluate_effect(effect, obs)
        else:
            value += 5.0  # unknown spell, play for tempo

        return value

    def _evaluate_play_weapon(self, card, obs: Observation) -> float:
        value = 100.0
        value += card.attack * card.durability * 6.0
        value += card.mana_cost * 2.0

        # Battlecry value
        bc = getattr(card, 'battlecry_effect', None)
        if bc:
            value += self._evaluate_effect(bc, obs) * 0.8

        return value

    def _evaluate_effect(self, effect, obs: Observation) -> float:
        """Score an effect tuple (used for both spells and battlecries)."""
        kind = effect[0]
        if kind == "deal_damage":
            return effect[1] * 8.0
        elif kind == "aoe_damage":
            n_enemies = max(len(obs.opponent_board), 1)
            return effect[1] * 6.0 * n_enemies
        elif kind == "draw":
            return effect[1] * 12.0
        elif kind == "restore_health":
            missing = 30 - obs.self_health
            scale = 1.0 + missing / 30.0
            return effect[1] * 3.0 * scale
        elif kind == "gain_armor":
            return effect[1] * 2.5
        elif kind == "destroy":
            if obs.opponent_board:
                best_cost = max(getattr(m, 'mana_cost', 0) for m in obs.opponent_board)
                return best_cost * 8.0
            return 0.0
        elif kind == "summon":
            return (effect[1] + effect[2]) * 5.0
        elif kind == "buff_all":
            n = len(obs.self_board)
            return (effect[1] + effect[2]) * 4.0 * max(n, 1)
        return 5.0

    # ------------------------------------------------------------------
    # Attack (minion)
    # ------------------------------------------------------------------

    def _evaluate_attack(self, action: Action, obs: Observation) -> float:
        idx = action.attacker_index
        if idx is None or idx >= len(obs.self_board):
            return 0.0
        attacker = obs.self_board[idx]
        advantage = self._board_advantage(obs)

        if action.defender_index is not None:
            return self._score_trade(attacker, action.defender_index, obs, advantage)
        else:
            return self._score_face(attacker.attack, obs, advantage)

    def _score_trade(self, attacker, defender_index, obs, advantage):
        if defender_index >= len(obs.opponent_board):
            return 0.0
        defender = obs.opponent_board[defender_index]
        value = 50.0

        # Kill bonus
        if attacker.attack >= defender.health:
            value += 30.0
            if defender.attack < attacker.health:
                value += 20.0  # favorable trade
            else:
                value += 5.0  # even trade
        # Overkill penalty
        overkill = attacker.attack - defender.health
        if overkill > 0:
            value -= overkill * 2.0

        # Value trade
        value += getattr(defender, 'mana_cost', 0) * 3.0

        # Taunt removal bonus
        mechs = getattr(defender, 'mechanics', []) or []
        if "TAUNT" in mechs:
            value += 10.0

        # Board advantage modifier
        if advantage > 0.3:
            value -= 20.0  # ahead: prefer face
        elif advantage < -0.3:
            value += 10.0  # behind: prefer trading

        return value

    def _score_face(self, damage, obs, advantage):
        value = 50.0
        value += damage * 3.0

        # Board clear bonus
        if not obs.opponent_board:
            value += 15.0

        # Board advantage modifier
        if advantage > 0.3:
            value += 20.0  # ahead: go face
        elif advantage < -0.3:
            value -= 10.0  # behind: should trade

        return value

    # ------------------------------------------------------------------
    # Hero attack (weapon)
    # ------------------------------------------------------------------

    def _evaluate_hero_attack(self, action: Action, obs: Observation) -> float:
        weapon_attack = obs.self_weapon_attack
        if weapon_attack <= 0:
            return 0.0

        if action.defender_index is not None:
            return self._score_hero_trade(weapon_attack, action.defender_index, obs)
        else:
            return self._score_hero_face(weapon_attack, obs)

    def _score_hero_trade(self, weapon_attack, defender_index, obs):
        if defender_index >= len(obs.opponent_board):
            return 0.0
        defender = obs.opponent_board[defender_index]
        value = 55.0

        # Kill bonus
        if weapon_attack >= defender.health:
            value += 30.0

        # Self-damage penalty (hero takes damage from minion)
        value -= defender.attack * 2.0
        if obs.self_health <= 10:
            value -= defender.attack * 3.0

        # Value trade
        value += getattr(defender, 'mana_cost', 0) * 2.0

        return value

    def _score_hero_face(self, weapon_attack, obs):
        value = 55.0
        value += weapon_attack * 4.0
        if not obs.opponent_board:
            value += 10.0
        return value

    # ------------------------------------------------------------------
    # Hero power
    # ------------------------------------------------------------------

    def _evaluate_hero_power(self, action: Action, obs: Observation) -> float:
        hero_class = obs.self_hero_class
        target = action.target

        if hero_class == "MAGE":
            return self._eval_mage_hp(target, obs)
        elif hero_class == "PRIEST":
            return self._eval_priest_hp(target, obs)
        elif hero_class == "WARLOCK":
            return 35.0 if obs.self_health > 8 else 10.0
        elif hero_class == "HUNTER":
            return 30.0
        elif hero_class == "PALADIN":
            return 28.0 if len(obs.self_board) < 7 else 0.0
        elif hero_class == "SHAMAN":
            return 26.0 if len(obs.self_board) < 7 else 0.0
        elif hero_class == "WARRIOR":
            return 22.0
        elif hero_class == "DRUID":
            return 20.0
        elif hero_class == "ROGUE":
            return 22.0
        elif hero_class == "DEMONHUNTER":
            return 25.0
        elif hero_class == "DEATHKNIGHT":
            return 24.0 + 3.0 * len(obs.opponent_board)
        else:
            # Generic / NEUTRAL
            return 25.0 if obs.self_mana >= 2 else 15.0

    def _eval_mage_hp(self, target, obs):
        if target is None:
            return 25.0
        label = target[0]
        if label == "opponent_minion":
            idx = target[1]
            if idx < len(obs.opponent_board):
                minion = obs.opponent_board[idx]
                if minion.health == 1:
                    return 40.0  # secures a kill
                return 30.0
            return 20.0
        elif label == "opponent_hero":
            return 28.0
        elif label in ("self_minion", "self_hero"):
            return 5.0
        return 20.0

    def _eval_priest_hp(self, target, obs):
        if target is None:
            return 20.0
        label = target[0]
        if label == "self_hero":
            if obs.self_health < 28:
                return 25.0 + (30 - obs.self_health) * 0.5
            return 5.0  # full health, wasted
        elif label == "self_minion":
            return 22.0
        elif label in ("opponent_hero", "opponent_minion"):
            return 5.0
        return 20.0

    # ------------------------------------------------------------------
    # Board advantage
    # ------------------------------------------------------------------

    def _board_advantage(self, obs: Observation) -> float:
        """Return board advantage from -1.0 (losing) to +1.0 (winning)."""
        my_power = sum(m.attack for m in obs.self_board)
        my_health = sum(m.health for m in obs.self_board)
        opp_power = sum(m.attack for m in obs.opponent_board)
        opp_health = sum(m.health for m in obs.opponent_board)

        my_score = my_power + my_health * 0.5
        opp_score = opp_power + opp_health * 0.5
        total = my_score + opp_score

        if total == 0:
            return 0.0
        return (my_score - opp_score) / total

    def reset(self) -> None:
        pass
