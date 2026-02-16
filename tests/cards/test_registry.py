"""
Tests for the card registry — loading real Hearthstone cards from JSON data.

These tests validate that:
- We can load the full card pool from cards_collectible.json
- Cards can be looked up by id and dbfId
- Card types are correctly parsed into our domain objects
- Filtering by class, type, set, and mechanics works
- The data integrity is sound (no missing required fields)
"""

import pytest
from pathlib import Path
from hearthstone.cards.registry import CardRegistry
from hearthstone.cards.base import Card, MinionCard, SpellCard, WeaponCard, HeroCard, LocationCard
from hearthstone.enums import CardType, CardClass, Rarity


class TestRegistryLoading:
    """The registry loads all collectible cards from the JSON file."""

    @pytest.fixture(autouse=True)
    def setup_registry(self):
        self.registry = CardRegistry.from_json(
            Path(__file__).parent.parent.parent / "hearthstone" / "data" / "cards_collectible.json"
        )

    def test_loads_all_cards(self):
        assert len(self.registry) > 7000

    def test_lookup_by_id(self):
        """Leeroy Jenkins has id EX1_116."""
        card = self.registry.get_by_id("EX1_116")
        assert card is not None
        assert card.name == "Leeroy Jenkins"

    def test_lookup_by_dbf_id(self):
        """Leeroy Jenkins has dbfId 559."""
        card = self.registry.get_by_dbf_id(559)
        assert card is not None
        assert card.name == "Leeroy Jenkins"

    def test_lookup_missing_card_returns_none(self):
        assert self.registry.get_by_id("FAKE_CARD_999") is None
        assert self.registry.get_by_dbf_id(999999999) is None


class TestCardTypeParsing:
    """Cards are parsed into the correct domain type based on their JSON 'type' field."""

    @pytest.fixture(autouse=True)
    def setup_registry(self):
        self.registry = CardRegistry.from_json(
            Path(__file__).parent.parent.parent / "hearthstone" / "data" / "cards_collectible.json"
        )

    def test_minion_has_attack_and_health(self):
        """Fallen Hero (AT_003): 2-mana 3/2 Mage minion."""
        card = self.registry.get_by_id("AT_003")
        assert isinstance(card, MinionCard)
        assert card.attack == 3
        assert card.health == 2
        assert card.mana_cost == 2

    def test_spell_has_no_attack_or_health(self):
        """Flame Lance (AT_001): 5-mana Mage spell."""
        card = self.registry.get_by_id("AT_001")
        assert isinstance(card, SpellCard)
        assert card.mana_cost == 5

    def test_weapon_has_attack_and_durability(self):
        """Fiery War Axe: 3-mana 3/2 Warrior weapon."""
        card = self.registry.get_by_id("CS2_106")
        assert isinstance(card, WeaponCard)
        assert card.attack == 3
        assert card.durability == 2

    def test_hero_card_has_armor(self):
        """Beaststalker Tavish (AV_113): 6-mana hero card with 5 armor."""
        card = self.registry.get_by_id("AV_113")
        assert isinstance(card, HeroCard)
        assert card.armor == 5

    def test_location_has_health(self):
        """Cathedral of Atonement (CORE_REV_290): 3-mana location with 3 durability."""
        card = self.registry.get_by_id("CORE_REV_290")
        assert isinstance(card, LocationCard)
        assert card.health == 3


class TestCardFiltering:
    """The registry supports filtering cards by class, type, set, and mechanics."""

    @pytest.fixture(autouse=True)
    def setup_registry(self):
        self.registry = CardRegistry.from_json(
            Path(__file__).parent.parent.parent / "hearthstone" / "data" / "cards_collectible.json"
        )

    def test_filter_by_card_class(self):
        mage_cards = self.registry.filter(card_class=CardClass.MAGE)
        assert len(mage_cards) > 100
        assert all(c.card_class == CardClass.MAGE for c in mage_cards)

    def test_filter_by_card_type(self):
        minions = self.registry.filter(card_type=CardType.MINION)
        assert len(minions) > 4000
        assert all(isinstance(c, MinionCard) for c in minions)

    def test_filter_by_mechanic(self):
        taunt_cards = self.registry.filter(mechanic="TAUNT")
        assert len(taunt_cards) > 100
        assert all("TAUNT" in c.mechanics for c in taunt_cards)

    def test_combined_filters(self):
        """Paladin minions with DIVINE_SHIELD."""
        cards = self.registry.filter(
            card_class=CardClass.PALADIN,
            card_type=CardType.MINION,
            mechanic="DIVINE_SHIELD",
        )
        assert len(cards) > 5
        for card in cards:
            assert card.card_class == CardClass.PALADIN
            assert isinstance(card, MinionCard)
            assert "DIVINE_SHIELD" in card.mechanics


class TestCardDataIntegrity:
    """Every card in the registry has the minimum required fields."""

    @pytest.fixture(autouse=True)
    def setup_registry(self):
        self.registry = CardRegistry.from_json(
            Path(__file__).parent.parent.parent / "hearthstone" / "data" / "cards_collectible.json"
        )

    def test_all_cards_have_name_and_cost(self):
        for card in self.registry.all():
            assert card.name, f"Card {card.id} has no name"
            assert card.mana_cost is not None, f"Card {card.name} has no cost"

    def test_all_minions_have_attack_and_health(self):
        minions = self.registry.filter(card_type=CardType.MINION)
        for card in minions:
            assert card.attack is not None, f"Minion {card.name} has no attack"
            assert card.health is not None, f"Minion {card.name} has no health"

    def test_all_weapons_have_attack_and_durability(self):
        weapons = self.registry.filter(card_type=CardType.WEAPON)
        for card in weapons:
            assert card.attack is not None, f"Weapon {card.name} has no attack"
            assert card.durability is not None, f"Weapon {card.name} has no durability"
