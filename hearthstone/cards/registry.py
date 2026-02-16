"""Card registry: loads card definitions from JSON and provides lookup/filtering."""

import json
from pathlib import Path
from typing import Optional

from hearthstone.cards.base import (
    Card,
    MinionCard,
    SpellCard,
    WeaponCard,
    HeroCard,
    LocationCard,
)
from hearthstone.enums import CardType, CardClass


class CardRegistry:
    """Look up and filter cards loaded from the card data JSON."""

    def __init__(self, cards: list[Card]):
        """Initialize registry with a list of cards."""
        self._cards = cards
        self._by_id = {card.id: card for card in cards}
        self._by_dbf_id = {card.dbf_id: card for card in cards if card.dbf_id}

    @classmethod
    def from_json(cls, path: Path) -> "CardRegistry":
        """Load cards from a JSON file and parse them into Card objects."""
        with open(path, "r") as f:
            data = json.load(f)

        cards = []
        for card_data in data:
            card = cls._parse_card(card_data)
            if card:
                cards.append(card)

        return cls(cards)

    @staticmethod
    def _parse_card(data: dict) -> Optional[Card]:
        """Parse a single card from JSON data into the appropriate Card subclass."""
        card_type = data.get("type", "").upper()

        # Common fields for all cards
        common_fields = {
            "id": data.get("id", ""),
            "dbf_id": data.get("dbfId", 0),
            "name": data.get("name", ""),
            "mana_cost": data.get("cost", 0),
            "card_class": data.get("cardClass", "NEUTRAL"),
            "rarity": data.get("rarity"),
            "card_set": data.get("set"),
            "text": data.get("text"),
            "mechanics": data.get("mechanics", []),
            "elite": data.get("elite", False),
        }

        if card_type == "MINION":
            return MinionCard(
                **common_fields,
                attack=data.get("attack", 0),
                health=data.get("health", 0),
                race=data.get("race"),
            )
        elif card_type == "SPELL":
            return SpellCard(
                **common_fields,
                spell_school=data.get("spellSchool"),
                overload=data.get("overload", 0),
                spell_damage=data.get("spellDamage", 0),
            )
        elif card_type == "WEAPON":
            return WeaponCard(
                **common_fields,
                attack=data.get("attack", 0),
                durability=data.get("health", 0),  # health field maps to durability
                overload=data.get("overload", 0),
            )
        elif card_type == "HERO":
            return HeroCard(
                **common_fields,
                armor=data.get("armor", 0),
                health=data.get("health", 30),
            )
        elif card_type == "LOCATION":
            return LocationCard(
                **common_fields,
                health=data.get("health", 0),
            )

        # Unknown card type, skip it
        return None

    def __len__(self) -> int:
        """Return the number of cards in the registry."""
        return len(self._cards)

    def get_by_id(self, card_id: str) -> Optional[Card]:
        """Get a card by its string ID, or None if not found."""
        return self._by_id.get(card_id)

    def get_by_dbf_id(self, dbf_id: int) -> Optional[Card]:
        """Get a card by its numeric DBF ID, or None if not found."""
        return self._by_dbf_id.get(dbf_id)

    def all(self) -> list[Card]:
        """Return all cards in the registry."""
        return self._cards.copy()

    def filter(
        self,
        card_class: Optional[CardClass] = None,
        card_type: Optional[CardType] = None,
        mechanic: Optional[str] = None,
    ) -> list[Card]:
        """Filter cards by class, type, and/or mechanic.

        Args:
            card_class: Filter by card class (e.g., CardClass.MAGE)
            card_type: Filter by card type (e.g., CardType.MINION)
            mechanic: Filter by mechanic name (e.g., "TAUNT")

        Returns:
            List of cards matching all specified filters.
        """
        result = self._cards

        if card_class is not None:
            result = [c for c in result if c.card_class == card_class.value]

        if card_type is not None:
            result = self._filter_by_card_type(result, card_type)

        if mechanic is not None:
            result = [c for c in result if mechanic in c.mechanics]

        return result

    @staticmethod
    def _filter_by_card_type(cards: list[Card], card_type: CardType) -> list[Card]:
        """Filter cards by their type (MINION, SPELL, WEAPON, HERO, LOCATION)."""
        type_map = {
            CardType.MINION: MinionCard,
            CardType.SPELL: SpellCard,
            CardType.WEAPON: WeaponCard,
            CardType.HERO: HeroCard,
            CardType.LOCATION: LocationCard,
        }
        card_class = type_map.get(card_type)
        if card_class:
            return [c for c in cards if isinstance(c, card_class)]
        return []
