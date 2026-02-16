"""Deck representation and simple card pool for GA experiments.

Provides a small synthetic card pool (Minion templates) and utilities to
create concrete decks (lists of `MinionCard` instances) from a genotype
(list of card names / indices).
"""
from dataclasses import dataclass
import random
from typing import List, Dict, Any

from hearthstone.cards.base import MinionCard


@dataclass
class CardSpec:
    name: str
    mana_cost: int
    attack: int
    health: int
    mechanics: List[str] | None = None
    rarity: str | None = None
    card_class: str | None = None
    card_set: str | None = None
    text: str | None = None
    battlecry_effect: tuple | None = None


# Small synthetic pool used for initial experiments. Replace with
# real card data when available.
CARD_POOL: Dict[str, CardSpec] = {
    "Tiny": CardSpec("Tiny", 0, 1, 1),
    "Wasp": CardSpec("Wasp", 1, 2, 1),
    "Adept": CardSpec("Adept", 2, 2, 3),
    "Brute": CardSpec("Brute", 3, 4, 3),
    "Giant": CardSpec("Giant", 7, 8, 8),
    "Healer": CardSpec("Healer", 3, 1, 4, mechanics=["LIFESTEAL"]),
    "Poison": CardSpec("Poison", 2, 1, 1, mechanics=["POISONOUS"]),
    "Shield": CardSpec("Shield", 2, 2, 2, mechanics=["DIVINE_SHIELD"]),
}


def random_deck(pool: Dict[str, CardSpec] = CARD_POOL, size: int = 30, rng: random.Random = None) -> List[str]:
    if rng is None:
        rng = random
    names = list(pool.keys())
    return [rng.choice(names) for _ in range(size)]


def build_concrete_deck(genotype: List[str], pool: Dict[str, CardSpec] = CARD_POOL, rng: random.Random = None) -> List[MinionCard]:
    """Convert a genotype (list of card names) into MinionCard instances.

    Each card is instantiated fresh to avoid shared mutable state between
    deck copies used in simulations.
    """
    if rng is None:
        rng = random
    out: List[MinionCard] = []
    for name in genotype:
        spec = pool.get(name)
        if spec is None:
            # skip unknown entries
            continue
        m = MinionCard(name=spec.name, mana_cost=spec.mana_cost, attack=spec.attack, health=spec.health)
        if spec.mechanics:
            m.mechanics.extend(list(spec.mechanics))
        if spec.battlecry_effect is not None:
            m.battlecry_effect = spec.battlecry_effect
        out.append(m)
    # Shuffle deck top/bottom so draw order varies
    rng.shuffle(out)
    return out


def validate_deck(genotype: List[str], pool: Dict[str, CardSpec] = CARD_POOL, size: int = 30) -> bool:
    return len(genotype) == size and all(name in pool for name in genotype)


# Mechanics fully supported by the engine (combat.py / actions.py)
SUPPORTED_MECHANICS = frozenset({
    "TAUNT", "DIVINE_SHIELD", "LIFESTEAL", "CHARGE",
    "RUSH", "POISONOUS", "WINDFURY", "BATTLECRY",
})


def build_pool_from_registry(
    registry,
    card_class: str | None = None,
    card_sets: set[str] | None = None,
) -> Dict[str, CardSpec]:
    """Build a GA-compatible card pool from real minion cards in a registry.

    Only includes minions whose mechanics are all supported by the engine.
    Uses card.id as the dict key (unique across sets).

    Args:
        registry: CardRegistry instance loaded from card JSON
        card_class: If provided, only include cards from this class + NEUTRAL.
                    Use CardClass enum values like "MAGE", "WARRIOR", etc.
        card_sets: If provided, only include cards from these sets.
                   Use set names like {"CORE", "TITANS"}.

    Returns:
        Dict mapping card ID to CardSpec
    """
    from hearthstone.enums import CardType
    from hearthstone.cards.battlecries import parse_battlecry_text

    pool: Dict[str, CardSpec] = {}
    for card in registry.filter(card_type=CardType.MINION):
        card_mechs = set(card.mechanics) if card.mechanics else set()
        if card_mechs <= SUPPORTED_MECHANICS:
            cc = card.card_class or "NEUTRAL"
            if card_class is not None and cc not in (card_class, "NEUTRAL"):
                continue
            cs = card.card_set
            if card_sets is not None and cs not in card_sets:
                continue
            # Parse battlecry text if card has BATTLECRY mechanic
            bc_effect = None
            card_text = getattr(card, 'text', None)
            if "BATTLECRY" in card_mechs and card_text:
                bc_effect = parse_battlecry_text(card_text)
            pool[card.id] = CardSpec(
                name=card.name,
                mana_cost=card.mana_cost,
                attack=card.attack,
                health=card.health,
                mechanics=list(card_mechs) if card_mechs else None,
                rarity=card.rarity,
                card_class=cc,
                card_set=cs,
                text=card_text,
                battlecry_effect=bc_effect,
            )
    return pool


class Deck:
    """Represents a Hearthstone deck with validation and utility methods.

    A deck is a collection of cards with the following rules:
    - Must contain exactly 30 cards
    - Non-legendary cards limited to 2 copies
    - Legendary cards limited to 1 copy
    """

    def __init__(self, cards: List[Any] = None, name: str = "", hero_class: str = ""):
        """Initialize a deck.

        Args:
            cards: List of card instances
            name: Deck name
            hero_class: Hero class (e.g., "MAGE", "WARRIOR")
        """
        self.cards = list(cards) if cards else []
        self.name = name
        self.hero_class = hero_class

    def __len__(self) -> int:
        """Return number of cards in deck."""
        return len(self.cards)

    def add_card(self, card: Any) -> None:
        """Add a card to the deck.

        Args:
            card: Card instance to add
        """
        self.cards.append(card)

    def remove_card(self, card_id: str) -> None:
        """Remove first instance of card with given ID.

        Args:
            card_id: ID of card to remove
        """
        for i, card in enumerate(self.cards):
            if card.id == card_id:
                self.cards.pop(i)
                return

    def clear(self) -> None:
        """Remove all cards from deck."""
        self.cards.clear()

    def is_valid(self) -> bool:
        """Check if deck satisfies Hearthstone validation rules.

        Returns:
            True if deck is valid
        """
        # Must have exactly 30 cards
        if len(self.cards) != 30:
            return False

        # Count copies of each card
        card_counts: Dict[str, int] = {}
        for card in self.cards:
            card_id = card.id
            card_counts[card_id] = card_counts.get(card_id, 0) + 1

        # Check copy limits
        for card_id, count in card_counts.items():
            # Find the card to check rarity
            card = next(c for c in self.cards if c.id == card_id)

            # Check if legendary
            is_legendary = hasattr(card, 'rarity') and card.rarity == "LEGENDARY"

            if is_legendary and count > 1:
                return False
            elif not is_legendary and count > 2:
                return False

        return True

    def validation_errors(self) -> List[str]:
        """Get list of validation errors.

        Returns:
            List of error messages
        """
        errors = []

        # Check card count
        if len(self.cards) != 30:
            errors.append(f"Deck must have exactly 30 cards (has {len(self.cards)})")

        # Count copies of each card
        card_counts: Dict[str, int] = {}
        for card in self.cards:
            card_id = card.id
            card_counts[card_id] = card_counts.get(card_id, 0) + 1

        # Check copy limits
        for card_id, count in card_counts.items():
            card = next(c for c in self.cards if c.id == card_id)
            is_legendary = hasattr(card, 'rarity') and card.rarity == "LEGENDARY"

            if is_legendary and count > 1:
                errors.append(f"Legendary card {card.name} cannot have more than 1 copy (has {count})")
            elif not is_legendary and count > 2:
                errors.append(f"Card {card.name} cannot have more than 2 copies (has {count})")

        return errors

    def mana_curve(self) -> Dict[int, int]:
        """Get mana curve distribution.

        Returns:
            Dict mapping mana cost to count
        """
        curve: Dict[int, int] = {}
        for card in self.cards:
            cost = card.mana_cost
            curve[cost] = curve.get(cost, 0) + 1
        return curve

    def average_mana_cost(self) -> float:
        """Calculate average mana cost of cards.

        Returns:
            Average mana cost
        """
        if not self.cards:
            return 0.0
        return sum(card.mana_cost for card in self.cards) / len(self.cards)

    def card_type_distribution(self) -> Dict[str, int]:
        """Get distribution of card types.

        Returns:
            Dict mapping card type to count
        """
        distribution: Dict[str, int] = {}
        for card in self.cards:
            # Determine card type
            if hasattr(card, 'attack') and hasattr(card, 'health'):
                card_type = 'MINION'
            elif hasattr(card, 'card_type'):
                card_type = card.card_type
            else:
                card_type = 'SPELL'

            distribution[card_type] = distribution.get(card_type, 0) + 1
        return distribution

    def count_card(self, card_id: str) -> int:
        """Count copies of a specific card.

        Args:
            card_id: Card ID to count

        Returns:
            Number of copies
        """
        return sum(1 for card in self.cards if card.id == card_id)

    def unique_cards(self) -> List[Any]:
        """Get list of unique cards (one of each).

        Returns:
            List of unique card instances
        """
        seen = set()
        unique = []
        for card in self.cards:
            if card.id not in seen:
                seen.add(card.id)
                unique.append(card)
        return unique

    def shuffle(self) -> None:
        """Shuffle the deck in place."""
        random.shuffle(self.cards)

    def draw(self) -> Any:
        """Draw the top card from the deck.

        Returns:
            Card instance or None if deck is empty
        """
        if not self.cards:
            return None
        return self.cards.pop(0)

    def copy(self) -> 'Deck':
        """Create a deep copy of the deck.

        Returns:
            New Deck instance
        """
        return Deck(
            cards=list(self.cards),
            name=self.name,
            hero_class=self.hero_class
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize deck to dictionary.

        Returns:
            Dictionary representation
        """
        # Count cards
        card_counts: Dict[str, int] = {}
        for card in self.cards:
            card_counts[card.id] = card_counts.get(card.id, 0) + 1

        # Build card list
        cards_list = [
            {"id": card_id, "count": count}
            for card_id, count in card_counts.items()
        ]

        return {
            "name": self.name,
            "hero_class": self.hero_class,
            "cards": cards_list
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], card_registry: Any = None) -> 'Deck':
        """Deserialize deck from dictionary.

        Args:
            data: Dictionary representation
            card_registry: CardRegistry instance for card lookup

        Returns:
            Deck instance
        """
        deck = cls(
            name=data.get("name", ""),
            hero_class=data.get("hero_class", "")
        )

        # If we have a registry, populate cards
        if card_registry:
            for card_data in data.get("cards", []):
                card_id = card_data["id"]
                count = card_data["count"]

                # Look up card in registry
                card = card_registry.get_by_id(card_id)
                if card:
                    for _ in range(count):
                        deck.add_card(card)

        return deck

    def to_deck_code(self) -> str:
        """Generate Hearthstone deck code.

        Returns:
            Deck code string
        """
        # Simplified deck code generation
        # Real implementation would use proper encoding
        import base64

        # Count cards by DBF ID
        card_data = []
        card_counts: Dict[int, int] = {}

        for card in self.cards:
            dbf_id = getattr(card, 'dbf_id', 0)
            card_counts[dbf_id] = card_counts.get(dbf_id, 0) + 1

        # Create simple encoding
        data_str = f"{self.hero_class}:{','.join(f'{k}x{v}' for k, v in card_counts.items())}"
        return base64.b64encode(data_str.encode()).decode()

    @classmethod
    def from_deck_code(cls, deck_code: str, card_registry: Any = None) -> 'Deck':
        """Parse Hearthstone deck code.

        Args:
            deck_code: Deck code string
            card_registry: CardRegistry instance for card lookup

        Returns:
            Deck instance
        """
        # Simplified deck code parsing
        import base64

        try:
            data_str = base64.b64decode(deck_code).decode()
            parts = data_str.split(':')
            hero_class = parts[0] if parts else ""

            deck = cls(hero_class=hero_class)

            # Parse card data if available
            if len(parts) > 1 and card_registry:
                card_entries = parts[1].split(',')
                for entry in card_entries:
                    if 'x' in entry:
                        dbf_id_str, count_str = entry.split('x')
                        dbf_id = int(dbf_id_str)
                        count = int(count_str)

                        # Look up card by DBF ID
                        card = card_registry.get_by_dbf_id(dbf_id)
                        if card:
                            for _ in range(count):
                                deck.add_card(card)

            return deck
        except Exception:
            return cls()

    @classmethod
    def from_card_ids(cls, card_ids: List[str], card_registry: Any = None) -> 'Deck':
        """Create deck from list of card IDs.

        Args:
            card_ids: List of card IDs
            card_registry: CardRegistry instance for card lookup

        Returns:
            Deck instance
        """
        deck = cls()

        if card_registry:
            for card_id in card_ids:
                card = card_registry.get_by_id(card_id)
                if card:
                    deck.add_card(card)

        return deck
