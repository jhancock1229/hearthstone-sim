"""Enumerations for Hearthstone game entities."""

from enum import Enum


class CardType(str, Enum):
    MINION = "MINION"
    SPELL = "SPELL"
    WEAPON = "WEAPON"
    HERO = "HERO"
    LOCATION = "LOCATION"


class CardClass(str, Enum):
    NEUTRAL = "NEUTRAL"
    DRUID = "DRUID"
    HUNTER = "HUNTER"
    MAGE = "MAGE"
    PALADIN = "PALADIN"
    PRIEST = "PRIEST"
    ROGUE = "ROGUE"
    SHAMAN = "SHAMAN"
    WARLOCK = "WARLOCK"
    WARRIOR = "WARRIOR"
    DEMONHUNTER = "DEMONHUNTER"
    DEATHKNIGHT = "DEATHKNIGHT"


class Rarity(str, Enum):
    FREE = "FREE"
    COMMON = "COMMON"
    RARE = "RARE"
    EPIC = "EPIC"
    LEGENDARY = "LEGENDARY"
