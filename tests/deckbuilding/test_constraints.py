"""Tests for deck constraints and validation rules.

These tests validate:
- Basic deck rules (30 cards, copy limits)
- Class restrictions (only neutral + class cards)
- Format restrictions (Standard, Wild, Classic)
- Banned/restricted cards
- Multi-class cards
"""

import pytest
from hearthstone.cards.base import MinionCard, SpellCard
from deckbuilding.deck import Deck
from deckbuilding.constraints import (
    DeckConstraints,
    validate_deck_size,
    validate_copy_limits,
    validate_class_restrictions,
    validate_format_legality,
    Format,
)


class TestBasicConstraints:
    """Test basic deck building rules."""

    def test_validate_deck_size_exactly_30(self):
        """Valid deck has exactly 30 cards."""
        cards = [MinionCard(id=f"m{i}", name=f"M{i%15}", mana_cost=1, attack=1, health=1) for i in range(30)]
        deck = Deck(cards=cards)

        result = validate_deck_size(deck)
        assert result.is_valid
        assert len(result.errors) == 0

    def test_validate_deck_size_too_few(self):
        """Deck with less than 30 cards is invalid."""
        cards = [MinionCard(id=f"m{i}", name=f"M{i}", mana_cost=1, attack=1, health=1) for i in range(25)]
        deck = Deck(cards=cards)

        result = validate_deck_size(deck)
        assert not result.is_valid
        assert len(result.errors) == 1
        assert "30 cards" in result.errors[0].lower()

    def test_validate_deck_size_too_many(self):
        """Deck with more than 30 cards is invalid."""
        cards = [MinionCard(id=f"m{i}", name=f"M{i%15}", mana_cost=1, attack=1, health=1) for i in range(35)]
        deck = Deck(cards=cards)

        result = validate_deck_size(deck)
        assert not result.is_valid
        assert len(result.errors) == 1
        assert "30 cards" in result.errors[0].lower()

    def test_validate_copy_limits_normal_cards(self):
        """Non-legendary cards can have up to 2 copies."""
        card_id = "m1"
        cards = [MinionCard(id=card_id, name="M1", mana_cost=1, attack=1, health=1)] * 2
        cards += [MinionCard(id=f"m{i}", name=f"M{i}", mana_cost=1, attack=1, health=1) for i in range(2, 30)]
        deck = Deck(cards=cards)

        result = validate_copy_limits(deck)
        assert result.is_valid
        assert len(result.errors) == 0

    def test_validate_copy_limits_too_many_copies(self):
        """More than 2 copies of non-legendary is invalid."""
        card_id = "m1"
        cards = [MinionCard(id=card_id, name="M1", mana_cost=1, attack=1, health=1)] * 3
        cards += [MinionCard(id=f"m{i}", name=f"M{i}", mana_cost=1, attack=1, health=1) for i in range(2, 29)]
        deck = Deck(cards=cards)

        result = validate_copy_limits(deck)
        assert not result.is_valid
        assert len(result.errors) == 1
        assert "2 copies" in result.errors[0].lower()

    def test_validate_copy_limits_legendary_single(self):
        """Legendary cards can have 1 copy."""
        legendary = MinionCard(id="leg1", name="Legendary", mana_cost=5, attack=5, health=5, rarity="LEGENDARY")
        cards = [legendary]
        cards += [MinionCard(id=f"m{i}", name=f"M{i}", mana_cost=1, attack=1, health=1) for i in range(29)]
        deck = Deck(cards=cards)

        result = validate_copy_limits(deck)
        assert result.is_valid
        assert len(result.errors) == 0

    def test_validate_copy_limits_legendary_duplicate(self):
        """Legendary cards cannot have 2 copies."""
        legendary = MinionCard(id="leg1", name="Legendary", mana_cost=5, attack=5, health=5, rarity="LEGENDARY")
        cards = [legendary, legendary]
        cards += [MinionCard(id=f"m{i}", name=f"M{i}", mana_cost=1, attack=1, health=1) for i in range(28)]
        deck = Deck(cards=cards)

        result = validate_copy_limits(deck)
        assert not result.is_valid
        assert len(result.errors) == 1
        assert "legendary" in result.errors[0].lower()


class TestClassRestrictions:
    """Test class card restrictions."""

    def test_validate_class_neutral_only(self):
        """Deck with only neutral cards is valid for any class."""
        cards = [MinionCard(id=f"n{i}", name=f"N{i%15}", mana_cost=1, attack=1, health=1, card_class="NEUTRAL") for i in range(30)]
        deck = Deck(cards=cards, hero_class="MAGE")

        result = validate_class_restrictions(deck)
        assert result.is_valid
        assert len(result.errors) == 0

    def test_validate_class_matching_class_cards(self):
        """Deck can include cards from its hero class."""
        cards = [MinionCard(id=f"m{i}", name=f"M{i%10}", mana_cost=1, attack=1, health=1, card_class="MAGE") for i in range(15)]
        cards += [MinionCard(id=f"n{i}", name=f"N{i}", mana_cost=1, attack=1, health=1, card_class="NEUTRAL") for i in range(15)]
        deck = Deck(cards=cards, hero_class="MAGE")

        result = validate_class_restrictions(deck)
        assert result.is_valid
        assert len(result.errors) == 0

    def test_validate_class_wrong_class_cards(self):
        """Deck cannot include cards from other classes."""
        cards = [MinionCard(id=f"w{i}", name=f"W{i}", mana_cost=1, attack=1, health=1, card_class="WARRIOR") for i in range(5)]
        cards += [MinionCard(id=f"m{i}", name=f"M{i}", mana_cost=1, attack=1, health=1, card_class="MAGE") for i in range(25)]
        deck = Deck(cards=cards, hero_class="MAGE")

        result = validate_class_restrictions(deck)
        assert not result.is_valid
        assert len(result.errors) > 0
        assert "warrior" in result.errors[0].lower()

    def test_validate_class_no_hero_class_specified(self):
        """Deck with no hero class can include any cards."""
        cards = [MinionCard(id=f"m{i}", name=f"M{i%15}", mana_cost=1, attack=1, health=1, card_class="MAGE") for i in range(30)]
        deck = Deck(cards=cards, hero_class="")

        # Should be valid when no class specified (for testing/flexibility)
        result = validate_class_restrictions(deck)
        assert result.is_valid


class TestFormatRestrictions:
    """Test format-specific restrictions (Standard, Wild, Classic)."""

    def test_validate_format_standard_recent_sets(self):
        """Standard format allows only recent sets."""
        cards = [
            MinionCard(id=f"c{i}", name=f"C{i%15}", mana_cost=1, attack=1, health=1, card_set="CORE")
            for i in range(30)
        ]
        deck = Deck(cards=cards)

        result = validate_format_legality(deck, Format.STANDARD)
        assert result.is_valid
        assert len(result.errors) == 0

    def test_validate_format_standard_rejects_old_sets(self):
        """Standard format rejects cards from old sets."""
        # Mix of CORE (legal) and NAXX (illegal in Standard)
        cards = [
            MinionCard(id=f"c{i}", name=f"C{i}", mana_cost=1, attack=1, health=1, card_set="CORE")
            for i in range(25)
        ]
        cards += [
            MinionCard(id=f"o{i}", name=f"O{i}", mana_cost=1, attack=1, health=1, card_set="NAXX")
            for i in range(5)
        ]
        deck = Deck(cards=cards)

        result = validate_format_legality(deck, Format.STANDARD)
        assert not result.is_valid
        assert len(result.errors) > 0
        assert "standard" in result.errors[0].lower()

    def test_validate_format_wild_allows_all(self):
        """Wild format allows cards from all sets."""
        cards = [
            MinionCard(id=f"o{i}", name=f"O{i%10}", mana_cost=1, attack=1, health=1, card_set="NAXX")
            for i in range(15)
        ]
        cards += [
            MinionCard(id=f"c{i}", name=f"C{i}", mana_cost=1, attack=1, health=1, card_set="CORE")
            for i in range(15)
        ]
        deck = Deck(cards=cards)

        result = validate_format_legality(deck, Format.WILD)
        assert result.is_valid
        assert len(result.errors) == 0

    def test_validate_format_classic_only_classic_sets(self):
        """Classic format allows only classic/basic sets."""
        cards = [
            MinionCard(id=f"b{i}", name=f"B{i%15}", mana_cost=1, attack=1, health=1, card_set="LEGACY")
            for i in range(30)
        ]
        deck = Deck(cards=cards)

        result = validate_format_legality(deck, Format.CLASSIC)
        assert result.is_valid


class TestDeckConstraintsComplete:
    """Test complete deck validation with all constraints."""

    def test_valid_standard_mage_deck(self):
        """Complete valid Standard Mage deck."""
        cards = [
            MinionCard(id=f"m{i}", name=f"M{i%10}", mana_cost=2, attack=2, health=2,
                      card_class="MAGE", card_set="CORE")
            for i in range(20)
        ]
        cards += [
            MinionCard(id=f"n{i}", name=f"N{i}", mana_cost=3, attack=3, health=3,
                      card_class="NEUTRAL", card_set="CORE")
            for i in range(10)
        ]
        deck = Deck(cards=cards, hero_class="MAGE", name="Standard Mage")

        constraints = DeckConstraints(format=Format.STANDARD)
        result = constraints.validate(deck)

        assert result.is_valid
        assert len(result.errors) == 0

    def test_invalid_deck_multiple_violations(self):
        """Deck with multiple constraint violations."""
        # Only 25 cards (size violation)
        # 3 copies of same card (copy limit violation)
        card_id = "m1"
        cards = [MinionCard(id=card_id, name="M1", mana_cost=1, attack=1, health=1)] * 3
        cards += [MinionCard(id=f"m{i}", name=f"M{i}", mana_cost=1, attack=1, health=1) for i in range(2, 24)]
        deck = Deck(cards=cards)

        constraints = DeckConstraints(format=Format.STANDARD)
        result = constraints.validate(deck)

        assert not result.is_valid
        assert len(result.errors) >= 2  # At least size + copy limit errors

    def test_validation_result_has_warnings(self):
        """Validation result can include warnings."""
        # Valid deck but with high mana curve (warning)
        cards = [
            MinionCard(id=f"h{i}", name=f"H{i%15}", mana_cost=10, attack=10, health=10)
            for i in range(30)
        ]
        deck = Deck(cards=cards, hero_class="WARRIOR")

        constraints = DeckConstraints(format=Format.WILD, enable_warnings=True)
        result = constraints.validate(deck)

        # Should be valid but with warnings about mana curve
        assert result.is_valid
        assert len(result.warnings) > 0

    def test_banned_cards_in_format(self):
        """Certain cards can be banned in specific formats."""
        # Create a deck with a banned card
        banned_card = MinionCard(id="banned", name="Banned", mana_cost=1, attack=1, health=1, card_set="CORE")
        cards = [banned_card] * 2
        cards += [MinionCard(id=f"m{i}", name=f"M{i}", mana_cost=1, attack=1, health=1) for i in range(28)]
        deck = Deck(cards=cards)

        # Create constraints with a banned list
        constraints = DeckConstraints(format=Format.STANDARD, banned_cards=["banned"])
        result = constraints.validate(deck)

        assert not result.is_valid
        assert any("banned" in error.lower() for error in result.errors)


class TestValidationResult:
    """Test the ValidationResult data structure."""

    def test_validation_result_success(self):
        """ValidationResult for successful validation."""
        from deckbuilding.constraints import ValidationResult

        result = ValidationResult(is_valid=True, errors=[], warnings=[])
        assert result.is_valid
        assert len(result.errors) == 0
        assert len(result.warnings) == 0

    def test_validation_result_with_errors(self):
        """ValidationResult with errors."""
        from deckbuilding.constraints import ValidationResult

        result = ValidationResult(
            is_valid=False,
            errors=["Error 1", "Error 2"],
            warnings=[]
        )
        assert not result.is_valid
        assert len(result.errors) == 2

    def test_validation_result_with_warnings(self):
        """ValidationResult can have warnings even when valid."""
        from deckbuilding.constraints import ValidationResult

        result = ValidationResult(
            is_valid=True,
            errors=[],
            warnings=["High mana curve", "No card draw"]
        )
        assert result.is_valid
        assert len(result.warnings) == 2
