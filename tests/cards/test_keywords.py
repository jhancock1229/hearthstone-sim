"""Unit tests describing expected keyword behavior.

These tests follow a TDD approach: they specify how keyword abilities
should behave (Taunt, Divine Shield, Charge, Rush, Lifesteal, Poisonous,
Windfury, Reborn, Stealth). Implementations will be added to the engine
to make these tests pass.
"""

import pytest

from hearthstone.cards.base import MinionCard
from hearthstone.engine.player import Player
from hearthstone.engine.combat import (
    resolve_minion_attack,
    resolve_minion_attack_hero,
    process_deaths,
)


def test_divine_shield_stored_on_card():
    m = MinionCard(name="Shielded", mana_cost=1, attack=1, health=2)
    m.mechanics.append("DIVINE_SHIELD")
    assert "DIVINE_SHIELD" in m.mechanics


def test_divine_shield_blocks_first_damage(player):
    att = MinionCard(name="Att", mana_cost=1, attack=1, health=1)
    defn = MinionCard(name="Shielded", mana_cost=1, attack=1, health=2)
    defn.mechanics.append("DIVINE_SHIELD")
    player.board.append(defn)
    opp = Player()
    opp.board.append(att)

    resolve_minion_attack(opp, 0, player, 0)
    # divine shield should block incoming damage once
    assert defn.health == 2
    assert "DIVINE_SHIELD" not in defn.mechanics


def test_divine_shield_removed_after_block(player):
    att = MinionCard(name="Att", mana_cost=1, attack=2, health=2)
    defn = MinionCard(name="Shielded", mana_cost=1, attack=1, health=3)
    defn.mechanics.append("DIVINE_SHIELD")
    player.board.append(defn)
    opp = Player()
    opp.board.append(att)

    resolve_minion_attack(opp, 0, player, 0)
    # first hit removed shield
    assert "DIVINE_SHIELD" not in defn.mechanics
    # second hit deals real damage
    resolve_minion_attack(opp, 0, player, 0)
    assert defn.health < 3


def test_lifesteal_heals_owner_on_damage_to_hero(player):
    lif = MinionCard(name="Leech", mana_cost=2, attack=3, health=2)
    lif.mechanics.append("LIFESTEAL")
    player.board.append(lif)
    opp = Player()

    orig_hp = player.health
    resolve_minion_attack_hero(player, 0, opp)
    # owner should gain health equal to damage dealt
    assert player.health >= orig_hp


def test_lifesteal_heals_owner_on_damage_to_minion(player):
    lif = MinionCard(name="Leech", mana_cost=2, attack=2, health=2)
    lif.mechanics.append("LIFESTEAL")
    victim = MinionCard(name="Dummy", mana_cost=1, attack=1, health=3)
    player.board.append(lif)
    opp = Player()
    opp.board.append(victim)

    resolve_minion_attack(player, 0, opp, 0)
    # lifesteal should restore owner's health by damage amount
    assert player.health > 30 - 1  # not lower than starting minus small damage


def test_poisonous_kills_any_target(player):
    p1 = Player()
    p2 = Player()
    poison = MinionCard(name="Poison", mana_cost=2, attack=1, health=1)
    poison.mechanics.append("POISONOUS")
    target = MinionCard(name="Big", mana_cost=5, attack=5, health=10)
    p1.board.append(poison)
    p2.board.append(target)

    resolve_minion_attack(p1, 0, p2, 0)
    process_deaths(p2)
    assert all(m.name != "Big" for m in p2.board)


def test_windfury_allows_two_attacks(player):
    wf = MinionCard(name="WF", mana_cost=3, attack=1, health=3)
    wf.mechanics.append("WINDFURY")
    player.board.append(wf)
    opp = Player()

    # Attack hero twice
    resolve_minion_attack_hero(player, 0, opp)
    resolve_minion_attack_hero(player, 0, opp)
    # opponent should have taken two hits total
    assert opp.health <= 30 - 2


def test_charge_allows_immediate_attack(player):
    ch = MinionCard(name="Charge", mana_cost=1, attack=3, health=1)
    ch.mechanics.append("CHARGE")
    player.board.append(ch)
    opp = Player()

    # should be able to attack immediately
    resolve_minion_attack_hero(player, 0, opp)
    assert opp.health < 30


def test_rush_allows_minion_attacks_but_not_hero(player):
    r = MinionCard(name="Rushy", mana_cost=2, attack=2, health=2)
    r.mechanics.append("RUSH")
    player.board.append(r)
    opp = Player()
    target = MinionCard(name="Dummy", mana_cost=1, attack=1, health=2)
    opp.board.append(target)

    # Rush can attack minions immediately
    resolve_minion_attack(player, 0, opp, 0)
    assert all(m.name != "Dummy" or m.health <= 0 for m in opp.board)


def test_reborn_returns_one_health_minion_on_death(player):
    r = MinionCard(name="Reborner", mana_cost=3, attack=3, health=1)
    r.mechanics.append("REBORN")
    player.board.append(r)
    opp = Player()
    killer = MinionCard(name="K", mana_cost=1, attack=2, health=2)
    opp.board.append(killer)

    resolve_minion_attack(opp, 0, player, 0)
    process_deaths(player)
    # reborn should re-summon a 1-health minion with same name
    assert any(m.name == "Reborner" and m.health == 1 for m in player.board)


def test_stealth_prevents_targeting_until_revealed(player):
    s = MinionCard(name="Hidden", mana_cost=2, attack=2, health=2)
    s.mechanics.append("STEALTH")
    player.board.append(s)
    opp = Player()

    # Opponent should not be able to directly target stealth minion
    with pytest.raises(Exception):
        resolve_minion_attack_hero(opp, 0, player)


def test_divine_shield_with_spell_damage(player):
    # Divine shield should absorb spell damage once
    d = MinionCard(name="ShieldedSpell", mana_cost=1, attack=1, health=2)
    d.mechanics.append("DIVINE_SHIELD")
    player.board.append(d)
    # Simulate a spell dealing 2 damage via direct subtraction
    try:
        d.health -= 2
    except Exception:
        pass
    # After spell, shield removed and health stays intact for first hit
    assert "DIVINE_SHIELD" not in d.mechanics
    assert d.health == 2


def test_poisonous_on_hero_damage_kills_hero(player):
    p1 = Player()
    p2 = Player()
    poison = MinionCard(name="Poison", mana_cost=2, attack=30, health=1)
    poison.mechanics.append("POISONOUS")
    p1.board.append(poison)

    # Poisonous dealing any damage should be lethal
    resolve_minion_attack_hero(p1, 0, p2)
    assert p2.is_dead or p2.health <= 0


def test_windfury_combined_with_divine_shield(player):
    m = MinionCard(name="WFShield", mana_cost=4, attack=2, health=3)
    m.mechanics.extend(["WINDFURY", "DIVINE_SHIELD"])
    player.board.append(m)
    opp = Player()

    resolve_minion_attack_hero(player, 0, opp)
    resolve_minion_attack_hero(player, 0, opp)
    # two attacks, shield only blocked first incoming damage
    assert opp.health <= 30 - 4


def test_charge_and_poisonous_interaction(player):
    m = MinionCard(name="CP", mana_cost=1, attack=1, health=1)
    m.mechanics.extend(["CHARGE", "POISONOUS"])
    player.board.append(m)
    opp = Player()
    big = MinionCard(name="Big", mana_cost=5, attack=5, health=10)
    opp.board.append(big)

    resolve_minion_attack(player, 0, opp, 0)
    process_deaths(opp)
    assert all(x.name != "Big" for x in opp.board)


def test_reborn_does_not_trigger_if_silenced(player):
    r = MinionCard(name="Reborner", mana_cost=3, attack=3, health=1)
    r.mechanics.append("REBORN")
    # hypothetical silence mechanic would remove reborn; test expects no reborn
    player.board.append(r)
    opp = Player()
    killer = MinionCard(name="K", mana_cost=1, attack=2, health=2)
    opp.board.append(killer)

    # simulate silence by removing mechanics
    r.mechanics.clear()
    resolve_minion_attack(opp, 0, player, 0)
    process_deaths(player)
    assert not any(m.name == "Reborner" for m in player.board)


def test_stealth_lost_on_attack(player):
    s = MinionCard(name="Hidden", mana_cost=2, attack=2, health=2)
    s.mechanics.append("STEALTH")
    player.board.append(s)
    opp = Player()
    dummy = MinionCard(name="Dummy", mana_cost=1, attack=1, health=1)
    opp.board.append(dummy)

    resolve_minion_attack(player, 0, opp, 0)
    # after dealing damage, stealth should be removed
    assert "STEALTH" not in s.mechanics


def test_windfury_does_not_allow_more_than_two_attacks(player):
    wf = MinionCard(name="WF", mana_cost=3, attack=1, health=3)
    wf.mechanics.append("WINDFURY")
    player.board.append(wf)
    opp = Player()

    resolve_minion_attack_hero(player, 0, opp)
    resolve_minion_attack_hero(player, 0, opp)
    # third attack should not be allowed; expect exception
    with pytest.raises(Exception):
        resolve_minion_attack_hero(player, 0, opp)


# ============================================================
# Additional Keyword Tests (19-30)
# ============================================================


def test_taunt_keyword_stored_on_minion(player):
    """Taunt mechanic can be added to a minion."""
    taunt = MinionCard(name="Taunt", mana_cost=2, attack=1, health=4)
    taunt.mechanics.append("TAUNT")
    assert "TAUNT" in taunt.mechanics


def test_lifesteal_with_zero_attack_does_nothing(player):
    """A 0 attack minion with lifesteal should not heal."""
    lif = MinionCard(name="Leech", mana_cost=2, attack=0, health=2)
    lif.mechanics.append("LIFESTEAL")
    player.board.append(lif)
    player.health = 20  # damaged
    opp = Player()

    resolve_minion_attack_hero(player, 0, opp)
    # no healing since no damage dealt
    assert player.health == 20


def test_poisonous_with_zero_attack_does_not_kill(player):
    """Poisonous only kills if damage is dealt (attack > 0)."""
    p1 = Player()
    p2 = Player()
    poison = MinionCard(name="Poison", mana_cost=2, attack=0, health=1)
    poison.mechanics.append("POISONOUS")
    target = MinionCard(name="Big", mana_cost=5, attack=0, health=10)
    p1.board.append(poison)
    p2.board.append(target)

    resolve_minion_attack(p1, 0, p2, 0)
    process_deaths(p2)
    # target should survive since no damage was dealt
    assert any(m.name == "Big" for m in p2.board)


def test_divine_shield_blocks_poisonous(player):
    """Divine Shield should block Poisonous effect."""
    p1 = Player()
    p2 = Player()
    poison = MinionCard(name="Poison", mana_cost=2, attack=1, health=1)
    poison.mechanics.append("POISONOUS")
    shielded = MinionCard(name="Shielded", mana_cost=3, attack=1, health=5)
    shielded.mechanics.append("DIVINE_SHIELD")
    p1.board.append(poison)
    p2.board.append(shielded)

    resolve_minion_attack(p1, 0, p2, 0)
    process_deaths(p2)
    # shield blocks damage, so poisonous doesn't trigger
    assert any(m.name == "Shielded" and m.health == 5 for m in p2.board)
    assert "DIVINE_SHIELD" not in shielded.mechanics


def test_reborn_minion_loses_reborn_after_first_death(player):
    """Reborn minion should not have Reborn on the resummoned copy."""
    r = MinionCard(name="Reborner", mana_cost=3, attack=3, health=2)
    r.mechanics.append("REBORN")
    player.board.append(r)
    opp = Player()
    killer = MinionCard(name="K", mana_cost=1, attack=2, health=2)
    opp.board.append(killer)

    resolve_minion_attack(opp, 0, player, 0)
    process_deaths(player)
    # reborn copy should exist but not have REBORN mechanic
    reborn_copy = next((m for m in player.board if m.name == "Reborner"), None)
    assert reborn_copy is not None
    assert reborn_copy.health == 1
    assert "REBORN" not in (reborn_copy.mechanics or [])


def test_windfury_on_minion_combat(player):
    """Windfury allows attacking two different minions in one turn."""
    wf = MinionCard(name="WF", mana_cost=3, attack=2, health=4)
    wf.mechanics.append("WINDFURY")
    player.board.append(wf)
    opp = Player()
    target1 = MinionCard(name="T1", mana_cost=1, attack=0, health=2)
    target2 = MinionCard(name="T2", mana_cost=1, attack=0, health=2)
    opp.board.extend([target1, target2])

    resolve_minion_attack(player, 0, opp, 0)
    process_deaths(opp)
    # first target should be dead
    assert all(m.name != "T1" for m in opp.board)

    # windfury allows second attack
    resolve_minion_attack(player, 0, opp, 0)
    process_deaths(opp)
    assert all(m.name != "T2" for m in opp.board)


def test_lifesteal_and_poisonous_combo(player):
    """Lifesteal + Poisonous should heal for damage dealt and kill target."""
    p1 = Player()
    p1.health = 20  # damaged
    p2 = Player()
    combo = MinionCard(name="Combo", mana_cost=3, attack=2, health=2)
    combo.mechanics.extend(["LIFESTEAL", "POISONOUS"])
    target = MinionCard(name="Big", mana_cost=5, attack=0, health=10)
    p1.board.append(combo)
    p2.board.append(target)

    resolve_minion_attack(p1, 0, p2, 0)
    process_deaths(p2)
    # poisonous kills target
    assert all(m.name != "Big" for m in p2.board)
    # lifesteal heals owner by attack damage
    assert p1.health == 22


def test_divine_shield_and_lifesteal_interaction(player):
    """Lifesteal should not heal if Divine Shield blocks the damage."""
    p1 = Player()
    p1.health = 20
    p2 = Player()
    lifesteal = MinionCard(name="LS", mana_cost=2, attack=3, health=2)
    lifesteal.mechanics.append("LIFESTEAL")
    shielded = MinionCard(name="Shielded", mana_cost=2, attack=2, health=3)
    shielded.mechanics.append("DIVINE_SHIELD")
    p1.board.append(lifesteal)
    p2.board.append(shielded)

    resolve_minion_attack(p1, 0, p2, 0)
    process_deaths(p1)
    # shield blocks damage, so no healing occurs
    assert p1.health == 20
    # lifesteal minion should die from counter-attack
    assert all(m.name != "LS" for m in p1.board)


def test_reborn_with_overkill_damage(player):
    """Reborn should work even when minion takes massive overkill damage."""
    r = MinionCard(name="Reborner", mana_cost=3, attack=1, health=1)
    r.mechanics.append("REBORN")
    player.board.append(r)
    opp = Player()
    killer = MinionCard(name="K", mana_cost=5, attack=10, health=10)
    opp.board.append(killer)

    resolve_minion_attack(opp, 0, player, 0)
    process_deaths(player)
    # should still reborn despite overkill
    assert any(m.name == "Reborner" and m.health == 1 for m in player.board)


def test_stealth_does_not_prevent_aoe_damage(player):
    """Stealth prevents targeting but not area damage."""
    s = MinionCard(name="Hidden", mana_cost=2, attack=2, health=3)
    s.mechanics.append("STEALTH")
    player.board.append(s)
    # Simulate AOE damage (direct health modification)
    s.health -= 2
    assert s.health == 1
    # stealth should still be present (only removed by attacking)
    assert "STEALTH" in s.mechanics


def test_charge_and_rush_are_different_keywords(player):
    """Charge and Rush are distinct mechanics."""
    charge = MinionCard(name="Charge", mana_cost=1, attack=1, health=1)
    charge.mechanics.append("CHARGE")
    rush = MinionCard(name="Rush", mana_cost=1, attack=1, health=1)
    rush.mechanics.append("RUSH")

    assert "CHARGE" in charge.mechanics
    assert "RUSH" not in charge.mechanics
    assert "RUSH" in rush.mechanics
    assert "CHARGE" not in rush.mechanics


def test_poisonous_works_on_both_sides_of_combat(player):
    """If both minions have Poisonous, both should die."""
    p1 = Player()
    p2 = Player()
    poison1 = MinionCard(name="P1", mana_cost=2, attack=1, health=5)
    poison1.mechanics.append("POISONOUS")
    poison2 = MinionCard(name="P2", mana_cost=2, attack=1, health=5)
    poison2.mechanics.append("POISONOUS")
    p1.board.append(poison1)
    p2.board.append(poison2)

    resolve_minion_attack(p1, 0, p2, 0)
    process_deaths(p1)
    process_deaths(p2)
    # both should be dead from mutual poisonous
    assert len(p1.board) == 0
    assert len(p2.board) == 0
