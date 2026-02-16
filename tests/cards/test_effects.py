"""Unit tests for card effect registries and loader.

Covers `register_deathrattle`, `register_battlecry` (global & scoped),
`load_effects_from_card_data`, and `clear_registries` behavior.
"""

from hearthstone.cards import effects
from hearthstone.cards.base import MinionCard
from hearthstone.engine.game import Game
from hearthstone.engine import events


def test_register_and_trigger_deathrattle(player):
    effects.clear_registries()
    calls = []

    @effects.register_deathrattle("Boom")
    def _boom(p, m):
        calls.append((p, m))

    m = MinionCard(name="Boom", health=1)
    player.place_minion(m)
    # kill the minion and process deaths
    m.health = 0
    player.process_deaths()

    assert len(calls) == 1
    assert calls[0][0] is player
    assert calls[0][1].name == "Boom"


def test_battlecry_global_handler_called_on_play(player):
    effects.clear_registries()
    called = []

    @effects.register_battlecry("TokenSummoner")
    def _bc(p, m, **kw):
        called.append((p, m, kw))

    m = MinionCard(name="TokenSummoner", health=1)
    player.place_minion(m)

    assert len(called) == 1
    assert called[0][0] is player
    assert called[0][1].name == "TokenSummoner"


def test_battlecry_scoped_triggers_only_with_matching_game(player):
    effects.clear_registries()
    called = []
    game = Game()

    @effects.register_battlecry("ScopedMinion", game=game)
    def _scoped(p, m, **kw):
        called.append((p, m, kw))

    m = MinionCard(name="ScopedMinion", health=1)
    # Emitting without game should not trigger
    events.bus.emit("on_play_from_hand", source=player, minion=m, position=0)
    assert called == []

    # Emitting with the matching game should trigger
    events.bus.emit("on_play_from_hand", source=player, minion=m, position=0, game=game)
    assert len(called) == 1


def test_load_effects_from_card_data_summons_token(player):
    effects.clear_registries()
    card_data = {
        "name": "TokenMaker",
        "effects": {
            "battlecry": {
                "action": "summon",
                "token": {"name": "TinyToken", "attack": 0, "health": 1, "mana_cost": 0},
            }
        },
    }
    effects.load_effects_from_card_data(card_data)

    m = MinionCard(name="TokenMaker", health=1)
    player.place_minion(m)

    # Original minion plus summoned token
    assert any(x.name == "TokenMaker" for x in player.board)
    assert any(x.name == "TinyToken" for x in player.board)


def test_clear_registries_unregisters_handlers(player):
    effects.clear_registries()
    called = []

    @effects.register_battlecry("Temp")
    def _t(p, m, **kw):
        called.append(True)

    effects.clear_registries()

    m = MinionCard(name="Temp", health=1)
    player.place_minion(m)
    assert called == []
"""PLACEHOLDER — Phase 2
Tests for Battlecry, Deathrattle, Aura, Triggered effects.
"""
