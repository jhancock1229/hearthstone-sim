"""Card text parsers for battlecries and spells.

Parses card text into structured effect tuples for the most common
patterns. Cards with unrecognized text return None and play as
vanilla stat-sticks (minions) or do nothing (spells).
"""

import re


def parse_battlecry_text(text):
    """Parse card text into a structured battlecry effect tuple.

    Args:
        text: Raw card text (may contain HTML tags)

    Returns:
        Effect tuple or None if unrecognized. Formats:
        - ("deal_damage", N)
        - ("draw", N)
        - ("restore_health", N)
        - ("gain_armor", N)
        - ("summon", attack, health)
        - ("buff_all", attack_buff, health_buff)
    """
    if not text:
        return None

    # Strip HTML tags and lowercase
    clean = re.sub(r'<[^>]+>', '', text).lower()

    # Deal damage
    m = re.search(r'deal (\d+) damage', clean)
    if m:
        return ("deal_damage", int(m.group(1)))

    # Draw cards
    m = re.search(r'draw (\d+|a) card', clean)
    if m:
        n = 1 if m.group(1) == 'a' else int(m.group(1))
        return ("draw", n)

    # Restore health
    m = re.search(r'restore (\d+) health', clean)
    if m:
        return ("restore_health", int(m.group(1)))

    # Gain armor
    m = re.search(r'gain (\d+) armor', clean)
    if m:
        return ("gain_armor", int(m.group(1)))

    # Buff all friendly minions
    m = re.search(r'give your minions \+(\d+)/\+(\d+)', clean)
    if m:
        return ("buff_all", int(m.group(1)), int(m.group(2)))

    # Summon (simplified: always 1/1 token)
    if 'summon' in clean:
        return ("summon", 1, 1)

    return None


def _clean_text(text):
    """Strip HTML tags, spell damage markers ($), and lowercase."""
    if not text:
        return ""
    clean = re.sub(r'<[^>]+>', '', text)
    clean = clean.replace('$', '')
    return clean.lower()


def parse_deathrattle_text(text):
    """Parse card text into a structured deathrattle effect tuple.

    Args:
        text: Raw card text (may contain HTML tags and $ markers)

    Returns:
        Effect tuple or None if unrecognized. Formats:
        - ("deal_damage", N)
        - ("aoe_damage", N)
        - ("draw", N)
        - ("restore_health", N)
        - ("gain_armor", N)
        - ("destroy", 1)
        - ("summon", attack, health)
        - ("buff_all", attack_buff, health_buff)
    """
    if not text:
        return None

    clean = _clean_text(text)

    # AoE damage (check BEFORE single-target)
    m = re.search(r'deal (\d+) damage to all', clean)
    if m:
        return ("aoe_damage", int(m.group(1)))

    # Deal damage (single target / face)
    m = re.search(r'deal (\d+) damage', clean)
    if m:
        return ("deal_damage", int(m.group(1)))

    # Draw cards
    m = re.search(r'draw (\d+|a) card', clean)
    if m:
        n = 1 if m.group(1) == 'a' else int(m.group(1))
        return ("draw", n)

    # Restore health
    m = re.search(r'restore (\d+) health', clean)
    if m:
        return ("restore_health", int(m.group(1)))

    # Gain armor
    m = re.search(r'gain (\d+) armor', clean)
    if m:
        return ("gain_armor", int(m.group(1)))

    # Destroy a minion
    if 'destroy a' in clean and 'minion' in clean:
        return ("destroy", 1)

    # Buff all friendly minions
    m = re.search(r'give your minions \+(\d+)/\+(\d+)', clean)
    if m:
        return ("buff_all", int(m.group(1)), int(m.group(2)))

    # Summon with explicit stats (e.g. "summon a 4/4")
    m = re.search(r'summon (?:a |an? )?(\d+)/(\d+)', clean)
    if m:
        return ("summon", int(m.group(1)), int(m.group(2)))

    # Summon without explicit stats (fallback 1/1)
    if 'summon' in clean:
        return ("summon", 1, 1)

    return None


def parse_spell_text(text):
    """Parse spell card text into a structured effect tuple.

    Args:
        text: Raw spell text (may contain HTML tags and $ markers)

    Returns:
        Effect tuple or None if unrecognized. Formats:
        - ("deal_damage", N)
        - ("aoe_damage", N)
        - ("draw", N)
        - ("restore_health", N)
        - ("gain_armor", N)
        - ("destroy",)
        - ("summon", attack, health)
    """
    if not text:
        return None

    clean = _clean_text(text)

    # AoE damage (check BEFORE single-target)
    m = re.search(r'deal (\d+) damage to all', clean)
    if m:
        return ("aoe_damage", int(m.group(1)))

    # Deal damage (single target / face)
    m = re.search(r'deal (\d+) damage', clean)
    if m:
        return ("deal_damage", int(m.group(1)))

    # Draw cards
    m = re.search(r'draw (\d+|a) card', clean)
    if m:
        n = 1 if m.group(1) == 'a' else int(m.group(1))
        return ("draw", n)

    # Restore health
    m = re.search(r'restore (\d+) health', clean)
    if m:
        return ("restore_health", int(m.group(1)))

    # Gain armor
    m = re.search(r'gain (\d+) armor', clean)
    if m:
        return ("gain_armor", int(m.group(1)))

    # Destroy a minion
    if 'destroy a minion' in clean or 'destroy an enemy minion' in clean:
        return ("destroy",)

    # Summon (simplified: always 1/1 token)
    if 'summon' in clean:
        return ("summon", 1, 1)

    return None
