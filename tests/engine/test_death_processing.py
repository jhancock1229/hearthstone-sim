"""
Unit tests for death processing: removing minions at 0 or less health from the board.
"""

import pytest
from hearthstone.engine.player import Player
from hearthstone.cards.base import MinionCard

class TestDeathProcessing:
    def setup_method(self):
        self.player = Player()
        self.player.board = []

    def test_remove_dead_minion(self):
        minion = MinionCard(name="Wisp", mana_cost=0, attack=1, health=0)
        self.player.board.append(minion)
        self.player.process_deaths()
        assert minion not in self.player.board

    def test_remove_multiple_dead_minions(self):
        m1 = MinionCard(name="A", mana_cost=1, attack=1, health=0)
        m2 = MinionCard(name="B", mana_cost=1, attack=1, health=-2)
        m3 = MinionCard(name="C", mana_cost=1, attack=1, health=3)
        self.player.board.extend([m1, m2, m3])
        self.player.process_deaths()
        assert m1 not in self.player.board
        assert m2 not in self.player.board
        assert m3 in self.player.board

    def test_no_removal_if_all_alive(self):
        m1 = MinionCard(name="A", mana_cost=1, attack=1, health=2)
        m2 = MinionCard(name="B", mana_cost=1, attack=1, health=1)
        self.player.board.extend([m1, m2])
        self.player.process_deaths()
        assert m1 in self.player.board
        assert m2 in self.player.board

    def test_empty_board_no_error(self):
        self.player.process_deaths()
        assert self.player.board == []

    def test_remove_dead_minion_after_combat(self):
        m1 = MinionCard(name="A", mana_cost=1, attack=3, health=2)
        m2 = MinionCard(name="B", mana_cost=1, attack=2, health=4)
        self.player.board.append(m1)
        opponent = Player()
        opponent.board.append(m2)
        self.player.attack_minion(0, opponent, 0)
        self.player.process_deaths()
        opponent.process_deaths()
        assert m1 not in self.player.board
        assert m2 in opponent.board
        assert m2.health == 1

    def test_remove_all_dead_minions(self):
        minions = [MinionCard(name=f"M{i}", mana_cost=1, attack=1, health=0) for i in range(5)]
        self.player.board.extend(minions)
        self.player.process_deaths()
        assert self.player.board == []
