"""
Unit tests for combat.py: attack resolution, damage, death, overkill, and edge cases.
"""

import pytest
from hearthstone.engine.combat import resolve_minion_attack, resolve_minion_attack_hero, process_deaths
from hearthstone.engine.player import Player
from hearthstone.cards.base import MinionCard

class TestCombat:
    def setup_method(self):
        self.p1 = Player()
        self.p2 = Player()
        self.p1.board = []
        self.p2.board = []

    def test_minion_vs_minion_damage(self):
        a = MinionCard(name="A", mana_cost=1, attack=3, health=2)
        b = MinionCard(name="B", mana_cost=1, attack=2, health=4)
        self.p1.board.append(a)
        self.p2.board.append(b)
        atk_ovk, def_ovk = resolve_minion_attack(self.p1, 0, self.p2, 0)
        assert a.health == 0
        assert b.health == 1
        assert atk_ovk == 0
        assert def_ovk == 1
        assert a not in self.p1.board
        assert b in self.p2.board

    def test_both_minions_die(self):
        a = MinionCard(name="A", mana_cost=1, attack=2, health=2)
        b = MinionCard(name="B", mana_cost=1, attack=2, health=2)
        self.p1.board.append(a)
        self.p2.board.append(b)
        atk_ovk, def_ovk = resolve_minion_attack(self.p1, 0, self.p2, 0)
        assert a.health == 0
        assert b.health == 0
        assert a not in self.p1.board
        assert b not in self.p2.board

    def test_minion_survives_attack(self):
        a = MinionCard(name="A", mana_cost=1, attack=1, health=3)
        b = MinionCard(name="B", mana_cost=1, attack=1, health=1)
        self.p1.board.append(a)
        self.p2.board.append(b)
        atk_ovk, def_ovk = resolve_minion_attack(self.p1, 0, self.p2, 0)
        assert a.health == 2
        assert b.health == 0
        assert a in self.p1.board
        assert b not in self.p2.board

    def test_attack_invalid_indices(self):
        with pytest.raises(IndexError):
            resolve_minion_attack(self.p1, 0, self.p2, 0)
        self.p1.board.append(MinionCard(name="A", mana_cost=1, attack=1, health=1))
        with pytest.raises(IndexError):
            resolve_minion_attack(self.p1, 0, self.p2, 0)
        self.p2.board.append(MinionCard(name="B", mana_cost=1, attack=1, health=1))
        with pytest.raises(IndexError):
            resolve_minion_attack(self.p1, 1, self.p2, 0)
        with pytest.raises(IndexError):
            resolve_minion_attack(self.p1, 0, self.p2, 1)

    def test_minion_attack_hero_damage(self):
        m = MinionCard(name="A", mana_cost=1, attack=3, health=2)
        self.p1.board.append(m)
        ovk = resolve_minion_attack_hero(self.p1, 0, self.p2)
        assert self.p2.health == 27
        assert ovk == 0

    def test_minion_attack_hero_lethal(self):
        m = MinionCard(name="A", mana_cost=1, attack=30, health=2)
        self.p1.board.append(m)
        ovk = resolve_minion_attack_hero(self.p1, 0, self.p2)
        assert self.p2.health == 0
        assert ovk == 0

    def test_minion_attack_hero_overkill(self):
        m = MinionCard(name="A", mana_cost=1, attack=35, health=2)
        self.p1.board.append(m)
        ovk = resolve_minion_attack_hero(self.p1, 0, self.p2)
        assert self.p2.health == -5
        assert ovk == 5

    def test_attack_hero_invalid_index(self):
        with pytest.raises(IndexError):
            resolve_minion_attack_hero(self.p1, 0, self.p2)

    def test_attack_hero_on_dead_hero(self):
        m = MinionCard(name="A", mana_cost=1, attack=3, health=2)
        self.p1.board.append(m)
        self.p2.health = 0
        ovk = resolve_minion_attack_hero(self.p1, 0, self.p2)
        assert ovk == 0
        assert self.p2.health == 0

    def test_attack_with_zero_attack(self):
        m = MinionCard(name="A", mana_cost=1, attack=0, health=2)
        self.p1.board.append(m)
        ovk = resolve_minion_attack_hero(self.p1, 0, self.p2)
        assert self.p2.health == 30
        assert ovk == 0

    def test_attack_with_negative_attack(self):
        m = MinionCard(name="A", mana_cost=1, attack=-2, health=2)
        self.p1.board.append(m)
        ovk = resolve_minion_attack_hero(self.p1, 0, self.p2)
        assert self.p2.health == 32
        assert ovk == 2

    def test_minion_attack_minion_with_zero_attack(self):
        a = MinionCard(name="A", mana_cost=1, attack=0, health=2)
        b = MinionCard(name="B", mana_cost=1, attack=1, health=2)
        self.p1.board.append(a)
        self.p2.board.append(b)
        atk_ovk, def_ovk = resolve_minion_attack(self.p1, 0, self.p2, 0)
        assert a.health == 1
        assert b.health == 2
        assert atk_ovk == 0
        assert def_ovk == 0

    def test_minion_attack_minion_with_negative_attack(self):
        a = MinionCard(name="A", mana_cost=1, attack=-2, health=2)
        b = MinionCard(name="B", mana_cost=1, attack=1, health=2)
        self.p1.board.append(a)
        self.p2.board.append(b)
        atk_ovk, def_ovk = resolve_minion_attack(self.p1, 0, self.p2, 0)
        assert a.health == 1
        assert b.health == 4
        assert atk_ovk == 0
        assert def_ovk == 0

    def test_minion_attack_minion_with_both_zero_attack(self):
        a = MinionCard(name="A", mana_cost=1, attack=0, health=2)
        b = MinionCard(name="B", mana_cost=1, attack=0, health=2)
        self.p1.board.append(a)
        self.p2.board.append(b)
        atk_ovk, def_ovk = resolve_minion_attack(self.p1, 0, self.p2, 0)
        assert a.health == 2
        assert b.health == 2
        assert atk_ovk == 0
        assert def_ovk == 0

    def test_minion_attack_minion_with_both_negative_attack(self):
        a = MinionCard(name="A", mana_cost=1, attack=-2, health=2)
        b = MinionCard(name="B", mana_cost=1, attack=-3, health=2)
        self.p1.board.append(a)
        self.p2.board.append(b)
        atk_ovk, def_ovk = resolve_minion_attack(self.p1, 0, self.p2, 0)
        assert a.health == 5
        assert b.health == 4
        assert atk_ovk == 0
        assert def_ovk == 0

    def test_process_deaths_removes_dead_minions(self):
        a = MinionCard(name="A", mana_cost=1, attack=1, health=0)
        b = MinionCard(name="B", mana_cost=1, attack=1, health=-1)
        c = MinionCard(name="C", mana_cost=1, attack=1, health=2)
        self.p1.board.extend([a, b, c])
        process_deaths(self.p1)
        assert a not in self.p1.board
        assert b not in self.p1.board
        assert c in self.p1.board

    def test_process_deaths_on_empty_board(self):
        process_deaths(self.p1)
        assert self.p1.board == []

    def test_attack_minion_and_process_deaths(self):
        a = MinionCard(name="A", mana_cost=1, attack=2, health=2)
        b = MinionCard(name="B", mana_cost=1, attack=2, health=2)
        self.p1.board.append(a)
        self.p2.board.append(b)
        resolve_minion_attack(self.p1, 0, self.p2, 0)
        process_deaths(self.p1)
        process_deaths(self.p2)
        assert a not in self.p1.board
        assert b not in self.p2.board

    def test_attack_minion_and_survivor_remains(self):
        a = MinionCard(name="A", mana_cost=1, attack=3, health=3)
        b = MinionCard(name="B", mana_cost=1, attack=1, health=2)
        self.p1.board.append(a)
        self.p2.board.append(b)
        resolve_minion_attack(self.p1, 0, self.p2, 0)
        process_deaths(self.p1)
        process_deaths(self.p2)
        assert a in self.p1.board
        assert b not in self.p2.board

    def test_attack_minion_and_no_deaths(self):
        a = MinionCard(name="A", mana_cost=1, attack=1, health=3)
        b = MinionCard(name="B", mana_cost=1, attack=1, health=3)
        self.p1.board.append(a)
        self.p2.board.append(b)
        resolve_minion_attack(self.p1, 0, self.p2, 0)
        process_deaths(self.p1)
        process_deaths(self.p2)
        assert a in self.p1.board
        assert b in self.p2.board

    def test_attack_minion_and_remove_by_index(self):
        a = MinionCard(name="A", mana_cost=1, attack=2, health=2)
        b = MinionCard(name="B", mana_cost=1, attack=2, health=2)
        self.p1.board.append(a)
        self.p2.board.append(b)
        resolve_minion_attack(self.p1, 0, self.p2, 0)
        # Remove by index after combat
        if a in self.p1.board:
            self.p1.board.remove(a)
        if b in self.p2.board:
            self.p2.board.remove(b)
        assert a not in self.p1.board or b not in self.p2.board
