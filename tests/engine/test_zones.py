"""Unit tests for game zones: Deck, Hand, Board, Graveyard, Secrets.

These tests specify expected behavior and constraints for each zone:
- `Hand`: max 10 cards
- `Board`: max 7 minions, insert/remove semantics
- `Deck`: push/draw and empty behavior
- `Graveyard`: append and clear
- `Secrets`: max 5 secrets, add/remove

These are TDD-style tests; implementations should provide the named
classes and methods used here.
"""

import pytest

from hearthstone.cards.base import MinionCard

from hearthstone import zones


def make_card(name):
    return MinionCard(name=name, mana_cost=1, attack=1, health=1)


def test_hand_accepts_up_to_10_cards_and_raises_on_overflow():
    hand = zones.Hand()
    for i in range(10):
        hand.add(make_card(f"C{i}"))
    assert len(hand) == 10
    with pytest.raises(Exception):
        hand.add(make_card("Overflow"))


def test_board_accepts_up_to_7_minions_and_raises_on_overflow():
    board = zones.Board()
    for i in range(7):
        board.add(make_card(f"M{i}"))
    assert len(board) == 7
    with pytest.raises(Exception):
        board.add(make_card("TooMany"))


def test_board_insert_and_remove_maintain_order():
    board = zones.Board()
    a = make_card("A")
    b = make_card("B")
    c = make_card("C")
    board.add(a)
    board.add(c)
    board.add(b, position=1)
    # order should be A, B, C
    assert [m.name for m in board] == ["A", "B", "C"]
    removed = board.remove(1)
    assert removed.name == "B"
    assert [m.name for m in board] == ["A", "C"]


def test_deck_push_draw_and_empty_behavior():
    deck = zones.Deck()
    c1 = make_card("Top")
    c2 = make_card("Bottom")
    deck.push(c2)
    deck.push(c1)
    assert len(deck) == 2
    drawn = deck.draw()
    # expect LIFO-style draw (last pushed is drawn)
    assert drawn is c1
    assert len(deck) == 1
    deck.draw()
    assert len(deck) == 0
    with pytest.raises(Exception):
        deck.draw()


def test_deck_shuffle_preserves_cards():
    deck = zones.Deck()
    cards = [make_card(str(i)) for i in range(10)]
    for c in cards:
        deck.push(c)
    deck.shuffle()
    assert set(id(c) for c in deck) == set(id(c) for c in cards)


def test_graveyard_records_and_clears():
    gy = zones.Graveyard()
    a = make_card("A")
    b = make_card("B")
    gy.add(a)
    gy.add(b)
    assert list(gy) == [a, b]
    gy.clear()
    assert len(gy) == 0


def test_secrets_limit_and_remove():
    s = zones.Secrets()
    for i in range(5):
        s.add(f"S{i}")
    assert len(s) == 5
    with pytest.raises(Exception):
        s.add("TooMany")
    # remove one
    s.remove("S2")
    assert len(s) == 4
