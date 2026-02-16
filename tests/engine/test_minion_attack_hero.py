"""
Unit tests for minions attacking the enemy hero in Hearthstone.
Covers damage, lethal, invalid attacks, and board state.
"""

import pytest
from hearthstone.engine.player import Player
from hearthstone.cards.base import MinionCard

class TestMinionAttacksHero:
    def setup_method(self):
        self.player = Player()
        self.opponent = Player()
        self.player.board = []
        self.opponent.board = []

    def test_minion_attacks_hero_deals_damage(self):
        minion = MinionCard(name="Wisp", mana_cost=0, attack=2, health=1)
        self.player.board.append(minion)
        self.player.attack_hero(0, self.opponent)
        assert self.opponent.health == 28

    def test_minion_attack_reduces_hero_health_to_zero(self):
        minion = MinionCard(name="Big", mana_cost=10, attack=30, health=10)
        self.player.board.append(minion)
        self.player.attack_hero(0, self.opponent)
        assert self.opponent.health == 0
        assert self.opponent.is_dead

    def test_minion_attack_lethal_negative_health(self):
        minion = MinionCard(name="Big", mana_cost=10, attack=40, health=10)
        self.player.board.append(minion)
        self.player.attack_hero(0, self.opponent)
        assert self.opponent.health == -10
        assert self.opponent.is_dead

    def test_attack_hero_invalid_index_raises(self):
        with pytest.raises(IndexError):
            self.player.attack_hero(0, self.opponent)

    def test_multiple_minions_attack_hero(self):
        for atk in [1, 2, 3]:
            self.player.board.append(MinionCard(name=f"M{atk}", mana_cost=1, attack=atk, health=1))
        self.player.attack_hero(0, self.opponent)
        self.player.attack_hero(1, self.opponent)
        self.player.attack_hero(2, self.opponent)
        assert self.opponent.health == 30 - (1+2+3)

    def test_minion_attack_does_not_remove_minion(self):
        minion = MinionCard(name="Wisp", mana_cost=0, attack=2, health=1)
        self.player.board.append(minion)
        self.player.attack_hero(0, self.opponent)
        assert minion in self.player.board

    def test_minion_attack_hero_with_zero_attack(self):
        minion = MinionCard(name="Pacifist", mana_cost=1, attack=0, health=1)
        self.player.board.append(minion)
        self.player.attack_hero(0, self.opponent)
        assert self.opponent.health == 30

    def test_minion_attack_hero_with_negative_attack(self):
        minion = MinionCard(name="Weird", mana_cost=1, attack=-2, health=1)
        self.player.board.append(minion)
        self.player.attack_hero(0, self.opponent)
        assert self.opponent.health == 32

    def test_minion_attack_hero_does_not_affect_attacker_health(self):
        minion = MinionCard(name="Wisp", mana_cost=0, attack=2, health=1)
        self.player.board.append(minion)
        self.player.attack_hero(0, self.opponent)
        assert minion.health == 1

    def test_attack_hero_on_dead_hero_no_effect(self):
        minion = MinionCard(name="Wisp", mana_cost=0, attack=2, health=1)
        self.player.board.append(minion)
        self.opponent.health = 0
        self.player.attack_hero(0, self.opponent)
        assert self.opponent.health == 0
        assert self.opponent.is_dead
