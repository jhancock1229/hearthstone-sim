"""
Integration tests for event emissions from combat and player actions.
"""

from hearthstone.engine.combat import resolve_minion_attack, resolve_minion_attack_hero, process_deaths
from hearthstone.engine.player import Player
from hearthstone.cards.base import MinionCard
from hearthstone.engine import events


def test_minion_attack_emits_damage_and_death_events():
    p1 = Player()
    p2 = Player()
    a = MinionCard(name="A", mana_cost=1, attack=3, health=2)
    b = MinionCard(name="B", mana_cost=1, attack=2, health=4)
    p1.board.append(a)
    p2.board.append(b)

    damage_calls = []
    death_calls = []

    def on_damage(evt):
        damage_calls.append((evt.source, evt.target, evt.data.get("amount")))

    def on_death(evt):
        death_calls.append((evt.source, evt.data.get("minion")))

    events.bus.on("on_damage", on_damage)
    events.bus.on("on_death", on_death)

    resolve_minion_attack(p1, 0, p2, 0)

    # Two damage events (defender and attacker)
    assert len(damage_calls) == 2
    # Death event for minion a
    assert len(death_calls) == 1
    assert death_calls[0][1].name == "A"

    # Cleanup
    events.bus.off("on_damage", on_damage)
    events.bus.off("on_death", on_death)


def test_negative_attack_on_hero_emits_heal():
    p1 = Player()
    p2 = Player()
    m = MinionCard(name="Healer", mana_cost=1, attack=-2, health=2)
    p1.board.append(m)

    heal_calls = []

    def on_heal(evt):
        heal_calls.append((evt.source, evt.target, evt.data.get("amount")))

    events.bus.on("on_heal", on_heal)

    resolve_minion_attack_hero(p1, 0, p2)

    assert len(heal_calls) == 1
    assert heal_calls[0][2] == 2

    events.bus.off("on_heal", on_heal)


def test_place_minion_emits_summon_and_play():
    p = Player()
    m = MinionCard(name="SummonMe", mana_cost=1, attack=1, health=1)

    calls = []

    def on_summon(evt):
        calls.append((evt.name, evt.data.get("minion"), evt.data.get("position")))

    events.bus.on("on_summon", on_summon)

    p.place_minion(m)

    assert calls and calls[0][1].name == "SummonMe"

    events.bus.off("on_summon", on_summon)
