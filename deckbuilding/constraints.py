"""Deck constraints and validation rules.

Provides validation for:
- Basic deck rules (30 cards, copy limits)
- Class card restrictions (neutral + class-specific)
- Format legality (Standard, Wild, Classic)
- Banned cards
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Set


class Format(Enum):
    """Hearthstone game formats."""
    STANDARD = "STANDARD"
    WILD = "WILD"
    CLASSIC = "CLASSIC"


@dataclass
class ValidationResult:
    """Result of deck validation."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


# Standard format legal sets (as of recent expansions)
STANDARD_SETS = {
    "CORE",  # Core set
    "TITANS",  # Titans
    "FESTIVAL_OF_LEGENDS",  # Festival of Legends
    "MARCH_OF_THE_LICH_KING",  # March of the Lich King
    "PATH_OF_ARTHAS",  # Path of Arthas
    "BATTLE_OF_THE_BANDS",  # Battle of the Bands
    "SHOWDOWN_IN_THE_BADLANDS",  # Showdown in the Badlands
    "WHIZBANGS_WORKSHOP",  # Whizbang's Workshop
    "PERILS_IN_PARADISE",  # Perils in Paradise
}

# Classic format sets
CLASSIC_SETS = {
    "LEGACY",  # Legacy/Basic
    "EXPERT1",  # Classic
    "CORE",  # Core set
}


def validate_deck_size(deck) -> ValidationResult:
    """Validate deck has exactly 30 cards.

    Args:
        deck: Deck instance to validate

    Returns:
        ValidationResult with size validation result
    """
    if len(deck) == 30:
        return ValidationResult(is_valid=True)

    return ValidationResult(
        is_valid=False,
        errors=[f"Deck must have exactly 30 cards (has {len(deck)})"]
    )


def validate_copy_limits(deck) -> ValidationResult:
    """Validate copy limits: 2 for normal cards, 1 for legendaries.

    Args:
        deck: Deck instance to validate

    Returns:
        ValidationResult with copy limit validation result
    """
    # Count copies of each card
    card_counts = {}
    for card in deck.cards:
        card_counts[card.id] = card_counts.get(card.id, 0) + 1

    errors = []
    for card_id, count in card_counts.items():
        # Find card to check rarity
        card = next(c for c in deck.cards if c.id == card_id)
        is_legendary = hasattr(card, 'rarity') and card.rarity == "LEGENDARY"

        if is_legendary and count > 1:
            errors.append(f"Legendary card {card.name} cannot have more than 1 copy (has {count})")
        elif not is_legendary and count > 2:
            errors.append(f"Card {card.name} cannot have more than 2 copies (has {count})")

    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors
    )


def validate_class_restrictions(deck) -> ValidationResult:
    """Validate cards match hero class (neutral + class-specific only).

    Args:
        deck: Deck instance to validate

    Returns:
        ValidationResult with class restriction validation result
    """
    # If no hero class specified, allow any cards (for testing/flexibility)
    if not deck.hero_class:
        return ValidationResult(is_valid=True)

    hero_class = deck.hero_class.upper()
    errors = []

    for card in deck.cards:
        card_class = getattr(card, 'card_class', 'NEUTRAL')
        if card_class is None:
            card_class = 'NEUTRAL'
        card_class = card_class.upper()

        # Cards must be neutral or match hero class
        if card_class != 'NEUTRAL' and card_class != hero_class:
            errors.append(
                f"Card {card.name} is {card_class.lower()} class, "
                f"cannot be used in {hero_class.lower()} deck"
            )
            break  # Only report first mismatch to avoid spam

    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors
    )


def validate_format_legality(deck, format: Format) -> ValidationResult:
    """Validate cards are legal in the specified format.

    Args:
        deck: Deck instance to validate
        format: Format to validate against

    Returns:
        ValidationResult with format legality validation result
    """
    if format == Format.WILD:
        # Wild allows all cards
        return ValidationResult(is_valid=True)

    errors = []
    illegal_cards = []

    for card in deck.cards:
        card_set = getattr(card, 'card_set', None)
        if card_set is None:
            card_set = getattr(card, 'set', 'UNKNOWN')
        if card_set is None:
            card_set = 'UNKNOWN'
        card_set = card_set.upper()

        if format == Format.STANDARD:
            if card_set not in STANDARD_SETS:
                illegal_cards.append(f"{card.name} ({card_set})")
        elif format == Format.CLASSIC:
            if card_set not in CLASSIC_SETS:
                illegal_cards.append(f"{card.name} ({card_set})")

    if illegal_cards:
        # Only show first few to avoid spam
        sample = illegal_cards[:3]
        error_msg = f"Cards not legal in {format.value.lower()} format: {', '.join(sample)}"
        if len(illegal_cards) > 3:
            error_msg += f" (and {len(illegal_cards) - 3} more)"
        errors.append(error_msg)

    return ValidationResult(
        is_valid=len(errors) == 0,
        errors=errors
    )


class DeckConstraints:
    """Complete deck validation with all constraints."""

    def __init__(
        self,
        format: Format = Format.WILD,
        banned_cards: List[str] = None,
        enable_warnings: bool = False
    ):
        """Initialize deck constraints.

        Args:
            format: Game format to validate against
            banned_cards: List of banned card IDs
            enable_warnings: Whether to generate warnings for suboptimal decks
        """
        self.format = format
        self.banned_cards = set(banned_cards or [])
        self.enable_warnings = enable_warnings

    def validate(self, deck) -> ValidationResult:
        """Validate deck against all constraints.

        Args:
            deck: Deck instance to validate

        Returns:
            ValidationResult with all validation results combined
        """
        all_errors = []
        all_warnings = []

        # Basic constraints
        size_result = validate_deck_size(deck)
        all_errors.extend(size_result.errors)

        copy_result = validate_copy_limits(deck)
        all_errors.extend(copy_result.errors)

        # Class restrictions
        class_result = validate_class_restrictions(deck)
        all_errors.extend(class_result.errors)

        # Format legality
        format_result = validate_format_legality(deck, self.format)
        all_errors.extend(format_result.errors)

        # Banned cards
        banned_result = self._validate_banned_cards(deck)
        all_errors.extend(banned_result.errors)

        # Warnings (if enabled)
        if self.enable_warnings:
            warnings = self._generate_warnings(deck)
            all_warnings.extend(warnings)

        return ValidationResult(
            is_valid=len(all_errors) == 0,
            errors=all_errors,
            warnings=all_warnings
        )

    def _validate_banned_cards(self, deck) -> ValidationResult:
        """Check for banned cards.

        Args:
            deck: Deck instance to validate

        Returns:
            ValidationResult for banned cards
        """
        if not self.banned_cards:
            return ValidationResult(is_valid=True)

        errors = []
        for card in deck.cards:
            if card.id in self.banned_cards:
                errors.append(f"Card {card.name} ({card.id}) is banned in this format")
                break  # Only report once

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors
        )

    def _generate_warnings(self, deck) -> List[str]:
        """Generate warnings for suboptimal deck construction.

        Args:
            deck: Deck instance to check

        Returns:
            List of warning messages
        """
        warnings = []

        # Check mana curve
        avg_cost = deck.average_mana_cost()
        if avg_cost > 5.0:
            warnings.append(f"High average mana cost ({avg_cost:.1f}), may struggle in early game")
        elif avg_cost < 2.0:
            warnings.append(f"Low average mana cost ({avg_cost:.1f}), may run out of cards")

        # Check for card draw (simple heuristic)
        # This is a placeholder - real implementation would check card text
        unique_count = len(deck.unique_cards())
        if unique_count < 15:
            warnings.append(f"Low variety ({unique_count} unique cards), consider more diverse cards")

        return warnings
