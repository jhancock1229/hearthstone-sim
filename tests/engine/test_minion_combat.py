"""
Unit tests for minion vs minion combat: attacking, damage, and death.
"""

import pytest
from hearthstone.engine.player import Player
from hearthstone.cards.base import MinionCard

class TestMinionCombat:
    def setup_method(self):
        self.player = Player()
        self.opponent = Player()
        self.player.board = []
        self.opponent.board = []

    def test_minion_attack_minion_both_take_damage(self):
        minion1 = MinionCard(name="A", mana_cost=1, attack=3, health=2)
        minion2 = MinionCard(name="B", mana_cost=1, attack=2, health=4)
        self.player.board.append(minion1)
        self.opponent.board.append(minion2)
        # Minion1 attacks minion2
        self.player.attack_minion(0, self.opponent, 0)
        assert minion1.health == 2 - minion2.attack  # 2 - 2 = 0
        assert minion2.health == 4 - minion1.attack  # 4 - 3 = 1

    def test_minion_dies_if_health_zero_or_below(self):
        minion1 = MinionCard(name="A", mana_cost=1, attack=3, health=2)
        minion2 = MinionCard(name="B", mana_cost=1, attack=2, health=2)
        self.player.board.append(minion1)
        self.opponent.board.append(minion2)
        self.player.attack_minion(0, self.opponent, 0)
        # Both minions should die (health 0 or less)
        assert minion1 not in self.player.board
        assert minion2 not in self.opponent.board

    def test_minion_survives_if_health_above_zero(self):
        minion1 = MinionCard(name="A", mana_cost=1, attack=2, health=3)
        minion2 = MinionCard(name="B", mana_cost=1, attack=1, health=2)
        self.player.board.append(minion1)
        self.opponent.board.append(minion2)
        self.player.attack_minion(0, self.opponent, 0)
        assert minion1 in self.player.board
        assert minion2 not in self.opponent.board
        assert minion1.health == 2

    def test_attack_invalid_indices_raises(self):
        minion1 = MinionCard(name="A", mana_cost=1, attack=2, health=3)
        self.player.board.append(minion1)
        with pytest.raises(IndexError):
            self.player.attack_minion(0, self.opponent, 1)  # No minion at index 1
        with pytest.raises(IndexError):
            self.player.attack_minion(1, self.opponent, 0)  # No minion at index 1
