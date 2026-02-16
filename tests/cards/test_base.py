"""Unit tests for base card creation and basic properties.

These tests verify dataclass defaults and the `MinionCard.health`
property behavior (Divine Shield consumption and integer casting).
"""

import pytest

from hearthstone.cards.base import (
	Card,
	MinionCard,
	SpellCard,
	WeaponCard,
	HeroCard,
	LocationCard,
)


def test_card_defaults():
	c = Card()
	assert c.id == ""
	assert c.dbf_id == 0
	assert c.name == ""
	assert c.mana_cost == 0
	assert c.mechanics == []
	assert c.elite is False


def test_minion_initialization_moves_health_and_property_works():
	m = MinionCard(attack=3, health=5)
	assert getattr(m, "_health") == 5
	assert m.health == 5
	assert isinstance(m.attack, int)


def test_divine_shield_consumed_on_damage():
	m = MinionCard(attack=1, health=5, mechanics=["DIVINE_SHIELD"])
	# Applying damage (lower value) should consume Divine Shield and
	# leave health unchanged.
	m.health = 3
	assert m.health == 5
	assert "DIVINE_SHIELD" not in m.mechanics


def test_heal_updates_health_and_casts_to_int():
	m = MinionCard(attack=0, health=2)
	m.health = 4
	assert m.health == 4
	# Non-integer assignments are cast to int
	m.health = 3.7
	assert m.health == 3


def test_other_card_types_default_constructible():
	assert isinstance(SpellCard(), SpellCard)
	assert isinstance(WeaponCard(), WeaponCard)
	assert isinstance(HeroCard(), HeroCard)
	assert isinstance(LocationCard(), LocationCard)


def test_negative_health_assignment():
	m = MinionCard(attack=0, health=1)
	m.health = -2
	assert m.health == -2


def test_mechanics_none_handling():
	m = MinionCard(attack=0, health=3, mechanics=None)
	# When mechanics is None the setter should not attempt to remove shields
	m.health = 1
	assert m.health == 1


def test_divine_shield_multiple_consumed_once():
	m = MinionCard(attack=1, health=5, mechanics=["DIVINE_SHIELD", "DIVINE_SHIELD"])
	m.health = 4
	# First damage consumes one shield only and leaves health unchanged.
	assert m.health == 5
	assert m.mechanics.count("DIVINE_SHIELD") == 1


def test_mechanics_list_is_unique_per_instance():
	a = MinionCard(health=2)
	b = MinionCard(health=2)
	a.mechanics.append("SOME_MECH")
	assert "SOME_MECH" not in b.mechanics


def test_initial_health_is_int_cast():
	m = MinionCard(attack=0, health=3.9)
	assert m.health == 3


def test_health_set_with_string_and_float_casts_to_int():
	m = MinionCard(attack=0, health=2)
	with pytest.raises(TypeError):
		m.health = "4"
	m.health = 5.9
	assert m.health == 5
