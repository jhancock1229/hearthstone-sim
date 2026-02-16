"""Base card definitions for Hearthstone simulator."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Card:
    """Base class for all cards."""
    id: str = ""
    dbf_id: int = 0
    name: str = ""
    mana_cost: int = 0
    card_class: Optional[str] = None
    rarity: Optional[str] = None
    card_set: Optional[str] = None
    text: Optional[str] = None
    mechanics: list[str] = field(default_factory=list)
    elite: bool = False


@dataclass
class MinionCard(Card):
    """A minion card with attack and health."""
    attack: int = 0
    health: int = 0
    race: Optional[str] = None
    exhausted: bool = False  # Can't attack if exhausted
    summoning_sick: bool = True  # Can't attack on turn summoned (unless CHARGE)

    def __post_init__(self):
        # Move the dataclass-initialized `health` into a private attribute
        # and expose a property so we can react to damage (e.g., Divine Shield).
        object.__setattr__(self, "_health", int(getattr(self, "health", 0)))
        # Remove the instance attribute so the property can take effect.
        try:
            delattr(self, "health")
        except Exception:
            pass

    @property
    def health(self) -> int:
        return getattr(self, "_health", 0)

    @health.setter
    def health(self, value: int) -> None:
        prev = getattr(self, "_health", 0)
        # If damage is being applied and the minion has Divine Shield,
        # consume the shield and ignore the damage.
        if value < prev and getattr(self, "mechanics", None) is not None:
            if "DIVINE_SHIELD" in self.mechanics:
                try:
                    self.mechanics.remove("DIVINE_SHIELD")
                except ValueError:
                    pass
                return
        object.__setattr__(self, "_health", int(value))


@dataclass
class SpellCard(Card):
    """A spell card."""
    spell_school: Optional[str] = None
    overload: int = 0
    spell_damage: int = 0


@dataclass
class WeaponCard(Card):
    """A weapon card with attack and durability."""
    attack: int = 0
    durability: int = 0
    overload: int = 0


@dataclass
class HeroCard(Card):
    """A hero card that replaces your hero."""
    armor: int = 0
    health: int = 0


@dataclass
class LocationCard(Card):
    """A location card with durability (stored as health in JSON)."""
    health: int = 0
