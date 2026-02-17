"""Unit tests for Deck - deck representation and validation.

Following TDD: these tests define the contract for Deck, which represents
a Hearthstone deck with proper validation rules.

Deck should:
- Contain exactly 30 cards
- Enforce max 2 copies of non-legendary cards
- Enforce max 1 copy of legendary cards
- Support serialization/deserialization
- Provide deck statistics (mana curve, card types, etc.)
- Integrate with CardRegistry
"""

import pytest
from pathlib import Path
from hearthstone.cards.base import MinionCard, SpellCard
from hearthstone.cards.registry import CardRegistry
from hearthstone.enums import CardType
from deckbuilding.deck import Deck, CardSpec, build_pool_from_registry


# ============================================================
# Deck Creation Tests
# ============================================================


class TestDeckCreation:
    """Tests for creating decks."""

    def test_create_empty_deck(self):
        """Can create an empty deck."""
        deck = Deck()

        assert len(deck) == 0

    def test_create_deck_with_cards(self):
        """Can create deck with initial cards."""
        cards = [
            MinionCard(id="m1", name="M1", mana_cost=1, attack=1, health=1),
            MinionCard(id="m2", name="M2", mana_cost=2, attack=2, health=2),
        ]

        deck = Deck(cards=cards)

        assert len(deck) == 2

    def test_add_card_to_deck(self):
        """Can add cards to deck."""
        deck = Deck()
        card = MinionCard(id="m1", name="M1", mana_cost=1, attack=1, health=1)

        deck.add_card(card)

        assert len(deck) == 1

    def test_remove_card_from_deck(self):
        """Can remove cards from deck."""
        card = MinionCard(id="m1", name="M1", mana_cost=1, attack=1, health=1)
        deck = Deck(cards=[card])

        deck.remove_card(card.id)

        assert len(deck) == 0

    def test_clear_deck(self):
        """Can clear all cards from deck."""
        cards = [
            MinionCard(id=f"m{i}", name=f"M{i}", mana_cost=i, attack=i, health=i)
            for i in range(10)
        ]
        deck = Deck(cards=cards)

        deck.clear()

        assert len(deck) == 0


# ============================================================
# Deck Validation Tests
# ============================================================


class TestDeckValidation:
    """Tests for Hearthstone deck validation rules."""

    def test_valid_deck_has_exactly_30_cards(self):
        """A valid deck must have exactly 30 cards."""
        cards = [
            MinionCard(id=f"m{i}", name=f"M{i%15}", mana_cost=1, attack=1, health=1)
            for i in range(30)
        ]
        deck = Deck(cards=cards)

        assert deck.is_valid()

    def test_deck_with_less_than_30_cards_is_invalid(self):
        """Deck with < 30 cards is invalid."""
        cards = [
            MinionCard(id=f"m{i}", name=f"M{i}", mana_cost=1, attack=1, health=1)
            for i in range(29)
        ]
        deck = Deck(cards=cards)

        assert not deck.is_valid()

    def test_deck_with_more_than_30_cards_is_invalid(self):
        """Deck with > 30 cards is invalid."""
        cards = [
            MinionCard(id=f"m{i}", name=f"M{i%15}", mana_cost=1, attack=1, health=1)
            for i in range(31)
        ]
        deck = Deck(cards=cards)

        assert not deck.is_valid()

    def test_can_have_up_to_2_copies_of_same_card(self):
        """Can have up to 2 copies of non-legendary cards."""
        card_id = "m1"
        cards = [
            MinionCard(id=card_id, name="M1", mana_cost=1, attack=1, health=1),
            MinionCard(id=card_id, name="M1", mana_cost=1, attack=1, health=1),
        ] + [
            MinionCard(id=f"m{i}", name=f"M{i}", mana_cost=1, attack=1, health=1)
            for i in range(2, 30)
        ]
        deck = Deck(cards=cards)

        assert deck.is_valid()

    def test_cannot_have_more_than_2_copies_of_same_card(self):
        """Cannot have more than 2 copies of non-legendary cards."""
        card_id = "m1"
        cards = [
            MinionCard(id=card_id, name="M1", mana_cost=1, attack=1, health=1),
            MinionCard(id=card_id, name="M1", mana_cost=1, attack=1, health=1),
            MinionCard(id=card_id, name="M1", mana_cost=1, attack=1, health=1),
        ] + [
            MinionCard(id=f"m{i}", name=f"M{i}", mana_cost=1, attack=1, health=1)
            for i in range(3, 30)
        ]
        deck = Deck(cards=cards)

        assert not deck.is_valid()

    def test_can_have_only_1_copy_of_legendary(self):
        """Legendary cards limited to 1 copy."""
        legendary = MinionCard(id="leg", name="Legendary", mana_cost=5, attack=5, health=5, rarity="LEGENDARY")
        cards = [legendary] + [
            MinionCard(id=f"m{i}", name=f"M{i}", mana_cost=1, attack=1, health=1)
            for i in range(29)
        ]
        deck = Deck(cards=cards)

        assert deck.is_valid()

    def test_cannot_have_2_copies_of_legendary(self):
        """Cannot have 2 copies of legendary cards."""
        legendary = MinionCard(id="leg", name="Legendary", mana_cost=5, attack=5, health=5, rarity="LEGENDARY")
        cards = [legendary, legendary] + [
            MinionCard(id=f"m{i}", name=f"M{i}", mana_cost=1, attack=1, health=1)
            for i in range(28)
        ]
        deck = Deck(cards=cards)

        assert not deck.is_valid()

    def test_validation_errors_list(self):
        """Can get list of validation errors."""
        cards = [
            MinionCard(id="m1", name="M1", mana_cost=1, attack=1, health=1)
            for _ in range(29)  # Too few cards
        ]
        deck = Deck(cards=cards)

        errors = deck.validation_errors()

        assert len(errors) > 0
        assert any("30 cards" in error for error in errors)


# ============================================================
# Deck Statistics Tests
# ============================================================


class TestDeckStatistics:
    """Tests for deck statistics and analysis."""

    def test_mana_curve(self):
        """Can get mana curve distribution."""
        cards = [
            MinionCard(id=f"m{i}", name=f"M{i}", mana_cost=i%8, attack=1, health=1)
            for i in range(30)
        ]
        deck = Deck(cards=cards)

        curve = deck.mana_curve()

        assert isinstance(curve, dict)
        assert all(isinstance(k, int) for k in curve.keys())
        assert sum(curve.values()) == 30

    def test_average_mana_cost(self):
        """Can calculate average mana cost."""
        cards = [
            MinionCard(id="m1", name="M1", mana_cost=2, attack=1, health=1),
            MinionCard(id="m2", name="M2", mana_cost=4, attack=2, health=2),
            MinionCard(id="m3", name="M3", mana_cost=6, attack=3, health=3),
        ]
        deck = Deck(cards=cards)

        avg = deck.average_mana_cost()

        assert avg == 4.0  # (2+4+6)/3

    def test_card_type_distribution(self):
        """Can get card type distribution."""
        cards = [
            MinionCard(id="m1", name="M1", mana_cost=1, attack=1, health=1),
            MinionCard(id="m2", name="M2", mana_cost=2, attack=2, health=2),
            SpellCard(id="s1", name="S1", mana_cost=3),
        ]
        deck = Deck(cards=cards)

        distribution = deck.card_type_distribution()

        assert distribution['MINION'] == 2
        assert distribution['SPELL'] == 1

    def test_count_specific_card(self):
        """Can count copies of specific card."""
        card_id = "m1"
        cards = [
            MinionCard(id=card_id, name="M1", mana_cost=1, attack=1, health=1),
            MinionCard(id=card_id, name="M1", mana_cost=1, attack=1, health=1),
            MinionCard(id="m2", name="M2", mana_cost=2, attack=2, health=2),
        ]
        deck = Deck(cards=cards)

        count = deck.count_card(card_id)

        assert count == 2

    def test_get_all_unique_cards(self):
        """Can get list of unique cards."""
        cards = [
            MinionCard(id="m1", name="M1", mana_cost=1, attack=1, health=1),
            MinionCard(id="m1", name="M1", mana_cost=1, attack=1, health=1),
            MinionCard(id="m2", name="M2", mana_cost=2, attack=2, health=2),
        ]
        deck = Deck(cards=cards)

        unique = deck.unique_cards()

        assert len(unique) == 2


# ============================================================
# Serialization Tests
# ============================================================


class TestDeckSerialization:
    """Tests for deck serialization and deserialization."""

    def test_to_dict(self):
        """Can serialize deck to dictionary."""
        cards = [
            MinionCard(id="m1", name="M1", mana_cost=1, attack=1, health=1),
            MinionCard(id="m2", name="M2", mana_cost=2, attack=2, health=2),
        ]
        deck = Deck(cards=cards, name="Test Deck", hero_class="MAGE")

        deck_dict = deck.to_dict()

        assert isinstance(deck_dict, dict)
        assert deck_dict['name'] == "Test Deck"
        assert deck_dict['hero_class'] == "MAGE"
        assert len(deck_dict['cards']) == 2

    def test_from_dict(self):
        """Can deserialize deck from dictionary."""
        deck_data = {
            'name': "Test Deck",
            'hero_class': "WARRIOR",
            'cards': [
                {"id": "m1", "count": 2},
                {"id": "m2", "count": 1},
            ]
        }

        # This would require a registry to look up cards
        # For now, test the structure
        deck = Deck.from_dict(deck_data, card_registry=None)

        assert deck.name == "Test Deck"
        assert deck.hero_class == "WARRIOR"

    def test_to_deck_code(self):
        """Can generate Hearthstone deck code."""
        cards = [
            MinionCard(id="m1", dbf_id=1000, name="M1", mana_cost=1, attack=1, health=1),
            MinionCard(id="m1", dbf_id=1000, name="M1", mana_cost=1, attack=1, health=1),
        ]
        deck = Deck(cards=cards, hero_class="MAGE")

        deck_code = deck.to_deck_code()

        assert isinstance(deck_code, str)
        assert len(deck_code) > 0

    def test_from_deck_code(self):
        """Can parse Hearthstone deck code."""
        # Example deck code (would need real implementation)
        deck_code = "AAECAf0EBMABuwKVA/gHAA=="

        deck = Deck.from_deck_code(deck_code, card_registry=None)

        assert deck is not None


# ============================================================
# Deck Operations Tests
# ============================================================


class TestDeckOperations:
    """Tests for deck manipulation operations."""

    def test_shuffle_deck(self):
        """Can shuffle deck."""
        cards = [
            MinionCard(id=f"m{i}", name=f"M{i}", mana_cost=i, attack=i, health=i)
            for i in range(10)
        ]
        deck = Deck(cards=cards.copy())
        original_order = [c.id for c in deck.cards]

        deck.shuffle()
        shuffled_order = [c.id for c in deck.cards]

        # Should have same cards but likely different order
        assert sorted(original_order) == sorted(shuffled_order)
        # With 10 cards, very likely to be in different order
        assert original_order != shuffled_order or len(cards) < 3

    def test_draw_card(self):
        """Can draw a card from deck."""
        cards = [
            MinionCard(id="m1", name="M1", mana_cost=1, attack=1, health=1),
            MinionCard(id="m2", name="M2", mana_cost=2, attack=2, health=2),
        ]
        deck = Deck(cards=cards)

        drawn = deck.draw()

        assert drawn is not None
        assert len(deck) == 1

    def test_draw_from_empty_deck_returns_none(self):
        """Drawing from empty deck returns None."""
        deck = Deck()

        drawn = deck.draw()

        assert drawn is None

    def test_copy_deck(self):
        """Can create a copy of deck."""
        cards = [
            MinionCard(id="m1", name="M1", mana_cost=1, attack=1, health=1),
        ]
        deck = Deck(cards=cards, name="Original")

        deck_copy = deck.copy()

        assert deck_copy.name == "Original"
        assert len(deck_copy) == len(deck)
        assert deck_copy is not deck  # Different instance


# ============================================================
# Integration Tests
# ============================================================


class TestDeckIntegration:
    """Integration tests with CardRegistry."""

    def test_create_deck_from_card_ids(self):
        """Can create deck from list of card IDs using registry."""
        # Would use real registry
        card_ids = ["m1", "m1", "m2", "m2"]

        deck = Deck.from_card_ids(card_ids, card_registry=None)

        assert deck is not None

    def test_deck_with_hero_class(self):
        """Deck can have associated hero class."""
        deck = Deck(hero_class="PALADIN")

        assert deck.hero_class == "PALADIN"

    def test_deck_with_name(self):
        """Deck can have a name."""
        deck = Deck(name="Aggro Deck")

        assert deck.name == "Aggro Deck"


# ============================================================
# Card Pool from Registry Tests
# ============================================================


_CARDS_JSON = Path("hearthstone/data/cards_collectible.json")


@pytest.mark.skipif(not _CARDS_JSON.exists(), reason="cards_collectible.json not available (gitignored data file)")
class TestBuildPoolFromRegistry:
    """Tests for building GA card pool from real card registry."""

    @pytest.fixture
    def registry(self):
        return CardRegistry.from_json(_CARDS_JSON)

    def test_returns_dict_of_cardspec(self, registry):
        """Returns a dict mapping card IDs to CardSpec objects."""
        pool = build_pool_from_registry(registry)
        assert isinstance(pool, dict)
        assert len(pool) > 0
        for key, val in list(pool.items())[:5]:
            assert isinstance(key, str)
            assert isinstance(val, CardSpec)

    def test_uses_card_id_as_key(self, registry):
        """Pool keys are card IDs (e.g., 'CS2_121'), not card names."""
        pool = build_pool_from_registry(registry)
        # Card IDs contain underscores or alphanumeric patterns, not spaces
        for key in list(pool.keys())[:20]:
            assert "_" in key or key.isalnum()

    def test_pool_size_reasonable(self, registry):
        """Pool has substantial number of minions (>500)."""
        pool = build_pool_from_registry(registry)
        assert len(pool) > 500

    def test_excludes_unsupported_mechanics_on_minions(self, registry):
        """Minions with DISCOVER, SECRET etc. are excluded."""
        pool = build_pool_from_registry(registry)
        unsupported = {"DISCOVER", "SECRET"}
        for spec in pool.values():
            if spec.card_type == "MINION" and spec.mechanics:
                card_mechs = set(spec.mechanics)
                assert card_mechs.isdisjoint(unsupported), (
                    f"{spec.name} has unsupported mechanic: {card_mechs & unsupported}"
                )

    def test_includes_vanilla_minions(self, registry):
        """Pool includes minions with no mechanics."""
        pool = build_pool_from_registry(registry)
        vanilla = [s for s in pool.values() if not s.mechanics]
        assert len(vanilla) > 50

    def test_includes_taunt_minions(self, registry):
        """Pool includes TAUNT minions."""
        pool = build_pool_from_registry(registry)
        taunt = [s for s in pool.values() if s.mechanics and "TAUNT" in s.mechanics]
        assert len(taunt) > 10

    def test_includes_divine_shield_minions(self, registry):
        """Pool includes DIVINE_SHIELD minions."""
        pool = build_pool_from_registry(registry)
        ds = [s for s in pool.values() if s.mechanics and "DIVINE_SHIELD" in s.mechanics]
        assert len(ds) > 5

    def test_cardspec_has_rarity(self, registry):
        """CardSpec instances include rarity field."""
        pool = build_pool_from_registry(registry)
        # At least some cards should have rarity
        has_rarity = [s for s in pool.values() if s.rarity is not None]
        assert len(has_rarity) > 100

    def test_cardspec_has_valid_stats(self, registry):
        """All CardSpec instances have non-negative stats."""
        pool = build_pool_from_registry(registry)
        for spec in pool.values():
            assert spec.mana_cost >= 0
            assert spec.attack >= 0
            assert spec.health >= 0

    def test_build_concrete_deck_works_with_registry_pool(self, registry):
        """build_concrete_deck works with a registry-derived pool."""
        from deckbuilding.deck import build_concrete_deck, random_deck
        from hearthstone.cards.base import SpellCard
        pool = build_pool_from_registry(registry)
        genotype = random_deck(pool, size=30)
        deck = build_concrete_deck(genotype, pool=pool)
        assert len(deck) == 30
        assert all(isinstance(c, (MinionCard, SpellCard)) for c in deck)

    # -- Step 1: card_class on CardSpec --

    def test_cardspec_has_card_class(self, registry):
        """CardSpecs from registry have non-None card_class."""
        pool = build_pool_from_registry(registry)
        has_class = [s for s in pool.values() if s.card_class is not None]
        assert len(has_class) > 100

    def test_pool_contains_neutral_cards(self, registry):
        """Pool includes NEUTRAL cards."""
        pool = build_pool_from_registry(registry)
        neutrals = [s for s in pool.values() if s.card_class == "NEUTRAL"]
        assert len(neutrals) > 50

    def test_pool_contains_class_cards(self, registry):
        """Pool includes class-specific cards (e.g. MAGE)."""
        pool = build_pool_from_registry(registry)
        mage = [s for s in pool.values() if s.card_class == "MAGE"]
        assert len(mage) > 0

    # -- Step 2: class filtering in build_pool_from_registry --

    def test_build_pool_class_filter_mage(self, registry):
        """Filtering by MAGE returns only MAGE + NEUTRAL cards."""
        pool = build_pool_from_registry(registry, card_class="MAGE")
        for spec in pool.values():
            assert spec.card_class in ("MAGE", "NEUTRAL"), (
                f"{spec.name} is {spec.card_class}, expected MAGE or NEUTRAL"
            )

    def test_build_pool_class_filter_excludes_other(self, registry):
        """MAGE pool has no WARRIOR cards."""
        pool = build_pool_from_registry(registry, card_class="MAGE")
        warrior = [s for s in pool.values() if s.card_class == "WARRIOR"]
        assert len(warrior) == 0

    def test_build_pool_class_filter_none_returns_all(self, registry):
        """None card_class returns same as no filter (all classes)."""
        pool_all = build_pool_from_registry(registry)
        pool_none = build_pool_from_registry(registry, card_class=None)
        assert len(pool_all) == len(pool_none)

    def test_build_pool_class_filter_pool_size(self, registry):
        """Filtered pool is smaller than unfiltered."""
        pool_all = build_pool_from_registry(registry)
        pool_mage = build_pool_from_registry(registry, card_class="MAGE")
        assert len(pool_mage) < len(pool_all)
        assert len(pool_mage) > 0

    # -- card_set on CardSpec --

    def test_cardspec_has_card_set(self, registry):
        """CardSpecs from registry have non-None card_set."""
        pool = build_pool_from_registry(registry)
        has_set = [s for s in pool.values() if s.card_set is not None]
        assert len(has_set) > 100

    def test_pool_contains_multiple_sets(self, registry):
        """Pool has cards from several different sets."""
        pool = build_pool_from_registry(registry)
        sets = {s.card_set for s in pool.values() if s.card_set}
        assert len(sets) > 5

    # -- card_sets filtering --

    def test_build_pool_set_filter(self, registry):
        """Filtering by {'CORE'} only returns CORE cards."""
        pool = build_pool_from_registry(registry, card_sets={"CORE"})
        for spec in pool.values():
            assert spec.card_set == "CORE", (
                f"{spec.name} is set {spec.card_set}, expected CORE"
            )

    def test_build_pool_set_filter_excludes_other(self, registry):
        """CORE-only pool has no EXPERT1 cards."""
        pool = build_pool_from_registry(registry, card_sets={"CORE"})
        expert = [s for s in pool.values() if s.card_set == "EXPERT1"]
        assert len(expert) == 0

    def test_build_pool_set_filter_none_returns_all(self, registry):
        """None card_sets returns same as no filter."""
        pool_all = build_pool_from_registry(registry)
        pool_none = build_pool_from_registry(registry, card_sets=None)
        assert len(pool_all) == len(pool_none)

    def test_build_pool_set_filter_multiple_sets(self, registry):
        """Can filter by multiple sets."""
        pool = build_pool_from_registry(registry, card_sets={"CORE", "TITANS"})
        for spec in pool.values():
            assert spec.card_set in ("CORE", "TITANS"), (
                f"{spec.name} is set {spec.card_set}, expected CORE or TITANS"
            )
        sets = {s.card_set for s in pool.values()}
        assert "CORE" in sets
        assert "TITANS" in sets

    def test_build_pool_set_filter_pool_smaller(self, registry):
        """Filtered pool is smaller than unfiltered."""
        pool_all = build_pool_from_registry(registry)
        pool_core = build_pool_from_registry(registry, card_sets={"CORE"})
        assert len(pool_core) < len(pool_all)
        assert len(pool_core) > 0
