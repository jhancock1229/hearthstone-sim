#!/usr/bin/env python3
"""Import card data from external sources.

This script fetches card data from HearthstoneJSON API and updates
the local card database files.

Usage:
    python scripts/import_cards.py [--output PATH] [--collectible-only]

Options:
    --output PATH          Output file path (default: hearthstone/data/cards.json)
    --collectible-only     Only import collectible cards
    --heroes              Also update heroes.json with basic heroes
    --help                Show this help message
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any


def fetch_cards_from_url(url: str = None) -> List[Dict[str, Any]]:
    """Fetch card data from HearthstoneJSON or other source.

    Args:
        url: API URL (defaults to HearthstoneJSON)

    Returns:
        List of card dictionaries

    Note:
        In production, this would use requests library to fetch from:
        https://api.hearthstonejson.com/v1/latest/enUS/cards.json
    """
    # Placeholder implementation
    # In real use, would do: requests.get(url).json()
    print("Note: This is a placeholder implementation.")
    print("To fetch real data, install requests and update this function.")
    print("Example: pip install requests")
    print("Then uncomment the requests code below.")

    # import requests
    # if url is None:
    #     url = "https://api.hearthstonejson.com/v1/latest/enUS/cards.json"
    # response = requests.get(url)
    # response.raise_for_status()
    # return response.json()

    return []


def filter_collectible(cards: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Filter to only collectible cards.

    Args:
        cards: List of all cards

    Returns:
        List of collectible cards only
    """
    return [card for card in cards if card.get("collectible", False)]


def extract_heroes(cards: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract basic hero cards.

    Args:
        cards: List of all cards

    Returns:
        List of basic hero cards (HERO_01 through HERO_11)
    """
    import re
    basic_pattern = re.compile(r'^HERO_(0[1-9]|1[01])$')
    heroes = [
        card for card in cards
        if card.get("type") == "HERO" and basic_pattern.match(card.get("id", ""))
    ]
    return sorted(heroes, key=lambda x: x.get("id", ""))


def save_cards(cards: List[Dict[str, Any]], output_path: Path) -> None:
    """Save cards to JSON file.

    Args:
        cards: List of card dictionaries
        output_path: Output file path
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(cards, f, indent=2)

    print(f"✅ Saved {len(cards)} cards to {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Import Hearthstone card data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Import all cards (placeholder - edit script to add requests)
  python scripts/import_cards.py

  # Import only collectible cards
  python scripts/import_cards.py --collectible-only

  # Also update heroes
  python scripts/import_cards.py --heroes
        """
    )

    parser.add_argument(
        '--output',
        type=Path,
        default=Path('hearthstone/data/cards.json'),
        help='Output file path'
    )

    parser.add_argument(
        '--collectible-only',
        action='store_true',
        help='Only import collectible cards'
    )

    parser.add_argument(
        '--heroes',
        action='store_true',
        help='Also update heroes.json'
    )

    parser.add_argument(
        '--url',
        type=str,
        help='Custom API URL'
    )

    args = parser.parse_args()

    try:
        # Fetch cards
        print("Fetching card data...")
        cards = fetch_cards_from_url(args.url)

        if not cards:
            print("\n⚠️  No cards fetched.")
            print("This script is a placeholder. To use it:")
            print("1. Install requests: pip install requests")
            print("2. Uncomment the requests code in fetch_cards_from_url()")
            print("\nFor now, the existing card data files are already populated.")
            return 0

        # Filter if needed
        if args.collectible_only:
            cards = filter_collectible(cards)
            print(f"Filtered to {len(cards)} collectible cards")

        # Save cards
        save_cards(cards, args.output)

        # Update heroes if requested
        if args.heroes:
            heroes = extract_heroes(cards)
            heroes_path = Path('hearthstone/data/heroes.json')
            save_cards(heroes, heroes_path)

        print("\n✅ Import complete!")
        return 0

    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
