"""Shared test fixtures: sample cards, decks, and game states."""

import pytest
from hearthstone.engine.player import Player
from hearthstone.cards.base import MinionCard


@pytest.fixture
def player():
    """A fresh player with default state."""
    return Player()


@pytest.fixture
def wisp():
    """A 0-mana 1/1 minion — the simplest possible card."""
    return MinionCard(name="Wisp", mana_cost=0, attack=1, health=1)
