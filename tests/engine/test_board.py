"""
Unit tests for Board: state tracking, 7-minion limit, and board operations.
"""

import pytest
from hearthstone.engine.board import Board
from hearthstone.cards.base import MinionCard

class TestBoard:
    def setup_method(self):
        self.board = Board()

    def test_board_starts_empty(self):
        assert len(self.board) == 0
        assert self.board.minions == []

    def test_add_minion_to_board(self):
        minion = MinionCard(name="Wisp", mana_cost=0, attack=1, health=1)
        pos = self.board.add_minion(minion)
        assert len(self.board) == 1
        assert self.board.minions[0] == minion
        assert pos == 0

    def test_add_minion_at_specific_position(self):
        m1 = MinionCard(name="A", mana_cost=1, attack=1, health=1)
        m2 = MinionCard(name="B", mana_cost=1, attack=1, health=1)
        self.board.add_minion(m1)
        self.board.add_minion(m2, position=0)
        assert self.board.minions[0] == m2
        assert self.board.minions[1] == m1

    def test_board_limit_is_seven(self):
        for i in range(7):
            self.board.add_minion(MinionCard(name=f"M{i}", mana_cost=1, attack=1, health=1))
        assert len(self.board) == 7
        with pytest.raises(Exception):
            self.board.add_minion(MinionCard(name="Extra", mana_cost=1, attack=1, health=1))

    def test_remove_minion_by_index(self):
        for i in range(3):
            self.board.add_minion(MinionCard(name=f"M{i}", mana_cost=1, attack=1, health=1))
        removed = self.board.remove_minion(1)
        assert removed.name == "M1"
        assert len(self.board) == 2
        assert [m.name for m in self.board.minions] == ["M0", "M2"]

    def test_remove_minion_invalid_index_raises(self):
        with pytest.raises(IndexError):
            self.board.remove_minion(0)
        self.board.add_minion(MinionCard(name="Wisp", mana_cost=0, attack=1, health=1))
        with pytest.raises(IndexError):
            self.board.remove_minion(1)

    def test_clear_board(self):
        for i in range(5):
            self.board.add_minion(MinionCard(name=f"M{i}", mana_cost=1, attack=1, health=1))
        self.board.clear()
        assert len(self.board) == 0
        assert self.board.minions == []

    def test_iterate_board(self):
        minions = [MinionCard(name=f"M{i}", mana_cost=1, attack=1, health=1) for i in range(3)]
        for m in minions:
            self.board.add_minion(m)
        assert list(self.board) == minions

    def test_board_str_and_repr(self):
        m = MinionCard(name="Wisp", mana_cost=0, attack=1, health=1)
        self.board.add_minion(m)
        s = str(self.board)
        r = repr(self.board)
        assert "Wisp" in s
        assert "Wisp" in r
