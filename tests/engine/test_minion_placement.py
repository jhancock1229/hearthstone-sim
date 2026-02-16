"""
Unit tests for placing minions on the board, enforcing the 7-slot limit.
"""

import pytest
from hearthstone.engine.player import Player
from hearthstone.cards.base import MinionCard

class TestMinionPlacement:
    def setup_method(self):
        self.player = Player()
        self.player.board = []  # Assume board is a list of minions

    def test_place_minion_on_empty_board(self):
        minion = MinionCard(name="Wisp", mana_cost=0, attack=1, health=1)
        assert len(self.player.board) == 0
        self.player.place_minion(minion)
        assert len(self.player.board) == 1
        assert self.player.board[0] == minion

    def test_place_minion_fills_up_to_seven(self):
        for i in range(7):
            minion = MinionCard(name=f"M{i}", mana_cost=1, attack=1, health=1)
            self.player.place_minion(minion)
        assert len(self.player.board) == 7

    def test_cannot_place_minion_if_board_full(self):
        for i in range(7):
            minion = MinionCard(name=f"M{i}", mana_cost=1, attack=1, health=1)
            self.player.place_minion(minion)
        extra = MinionCard(name="Extra", mana_cost=1, attack=1, health=1)
        with pytest.raises(Exception):
            self.player.place_minion(extra)

    def test_place_minion_returns_board_position(self):
        minion = MinionCard(name="Wisp", mana_cost=0, attack=1, health=1)
        pos = self.player.place_minion(minion)
        assert pos == 0
        minion2 = MinionCard(name="Wisp2", mana_cost=0, attack=1, health=1)
        pos2 = self.player.place_minion(minion2)
        assert pos2 == 1

    def test_place_minion_at_specific_position(self):
        for i in range(3):
            self.player.place_minion(MinionCard(name=f"M{i}", mana_cost=1, attack=1, health=1))
        new_minion = MinionCard(name="Insert", mana_cost=1, attack=1, health=1)
        pos = self.player.place_minion(new_minion, position=1)
        assert self.player.board[1] == new_minion
        assert pos == 1
        assert len(self.player.board) == 4
