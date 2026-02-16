"""
Tests for Battlecry registry, game-scoped handlers, and data-driven effect loading.
"""

from hearthstone.engine import events
from hearthstone.engine.player import Player
from hearthstone.engine.game import Game
from hearthstone.cards.base import MinionCard
from hearthstone.cards import effects


def test_battlecry_triggers_on_play_from_hand():
    p = Player()
    card = MinionCard(name="Bomber", mana_cost=2, attack=2, health=2)

    @effects.register_battlecry("Bomber")
    def bomber_battlecry(player, minion, **kwargs):
        token = MinionCard(name="Shrapnel", mana_cost=0, attack=1, health=1)
        player.place_minion(token)

    try:
        # Emulate playing from hand via event
        events.bus.emit("on_play_from_hand", source=p, minion=card)
        assert any(m.name == "Shrapnel" for m in p.board)
    finally:
        effects.unregister_battlecry("Bomber")
        effects.clear_registries()


def test_battlecry_scoped_to_game():
    g1 = Game()
    g2 = Game()
    p1 = g1.player1
    p2 = g2.player1
    card = MinionCard(name="Scoped", mana_cost=1, attack=1, health=1)

    @effects.register_battlecry("Scoped", game=g1)
    def scoped_battlecry(player, minion, **kwargs):
        token = MinionCard(name="G1Token", mana_cost=0, attack=1, health=1)
        player.place_minion(token)

    try:
        # Emit for game1 should trigger
        events.bus.emit("on_play_from_hand", source=p1, game=g1, minion=card)
        assert any(m.name == "G1Token" for m in p1.board)
        # Emit for game2 should not trigger
        events.bus.emit("on_play_from_hand", source=p2, game=g2, minion=card)
        assert not any(m.name == "G1Token" for m in p2.board)
    finally:
        effects.unregister_battlecry("Scoped")
        effects.clear_registries()


def test_load_effects_from_card_data_registers_battlecry():
    p = Player()
    card_data = {
        "name": "DataBomber",
        "effects": {
            "battlecry": {
                "action": "summon",
                "token": {"name": "DBToken", "mana_cost": 1, "attack": 1, "health": 1}
            }
        }
    }

    # Load effects (to be implemented)
    effects.load_effects_from_card_data(card_data)

    try:
        events.bus.emit("on_play_from_hand", source=p, minion=MinionCard(name="DataBomber", mana_cost=3, attack=3, health=3))
        assert any(m.name == "DBToken" for m in p.board)
    finally:
        effects.clear_registries()
