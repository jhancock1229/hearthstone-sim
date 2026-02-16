"""Tests for battlecry text parsing and effect resolution.

Tests are organized by:
1. Text parser (parse_battlecry_text)
2. Effect resolver (resolve_battlecry)
3. Integration (play_card triggers battlecry, CardSpec/deck building)
"""

import pytest
from hearthstone.engine.player import Player
from hearthstone.engine.game import Game
from hearthstone.cards.base import MinionCard
from hearthstone.cards.battlecries import parse_battlecry_text
from hearthstone.engine.actions import resolve_battlecry, play_card


# ============================================================
# Text Parser Tests
# ============================================================


class TestParseBattlecryText:
    """Test parse_battlecry_text extracts structured effects from card text."""

    def test_parse_deal_damage(self):
        """Parse 'Deal N damage' pattern."""
        result = parse_battlecry_text("<b>Battlecry:</b> Deal 3 damage.")
        assert result == ("deal_damage", 3)

    def test_parse_deal_damage_to_target(self):
        """Parse 'Deal N damage to X' pattern."""
        result = parse_battlecry_text("<b>Battlecry:</b> Deal 2 damage to the enemy hero.")
        assert result == ("deal_damage", 2)

    def test_parse_draw_a_card(self):
        """Parse 'Draw a card' (singular)."""
        result = parse_battlecry_text("<b>Battlecry:</b> Draw a card.")
        assert result == ("draw", 1)

    def test_parse_draw_n_cards(self):
        """Parse 'Draw N cards'."""
        result = parse_battlecry_text("<b>Battlecry:</b> Draw 2 cards.")
        assert result == ("draw", 2)

    def test_parse_restore_health(self):
        """Parse 'Restore N Health' pattern."""
        result = parse_battlecry_text("<b>Battlecry:</b> Restore 8 Health.")
        assert result == ("restore_health", 8)

    def test_parse_gain_armor(self):
        """Parse 'Gain N Armor' pattern."""
        result = parse_battlecry_text("<b>Battlecry:</b> Gain 5 Armor.")
        assert result == ("gain_armor", 5)

    def test_parse_summon(self):
        """Parse 'Summon' pattern."""
        result = parse_battlecry_text("<b>Battlecry:</b> Summon a 1/1 Silver Hand Recruit.")
        assert result is not None
        assert result[0] == "summon"

    def test_parse_buff_all(self):
        """Parse 'Give your minions +X/+Y' pattern."""
        result = parse_battlecry_text("<b>Battlecry:</b> Give your minions +1/+1.")
        assert result == ("buff_all", 1, 1)

    def test_parse_unrecognized_returns_none(self):
        """Unrecognized text returns None."""
        result = parse_battlecry_text("<b>Battlecry:</b> Transform into a random Legendary minion.")
        assert result is None

    def test_parse_none_text_returns_none(self):
        """None input returns None."""
        result = parse_battlecry_text(None)
        assert result is None

    def test_parse_empty_text_returns_none(self):
        """Empty string returns None."""
        result = parse_battlecry_text("")
        assert result is None


# ============================================================
# Battlecry Resolver Tests
# ============================================================


class TestResolveBattlecry:
    """Test resolve_battlecry applies effects correctly."""

    def _make_card(self, effect):
        """Helper: create a MinionCard with a battlecry_effect."""
        card = MinionCard(name="Test", mana_cost=3, attack=3, health=3, mechanics=["BATTLECRY"])
        card.battlecry_effect = effect
        return card

    def test_resolve_deal_damage(self):
        """deal_damage should reduce opponent hero health."""
        player = Player()
        opponent = Player()
        card = self._make_card(("deal_damage", 3))

        resolve_battlecry(player, card, opponent)
        assert opponent.health == 27

    def test_resolve_deal_damage_with_armor(self):
        """deal_damage should use opponent's take_damage (armor absorbs)."""
        player = Player()
        opponent = Player()
        opponent.armor = 2
        card = self._make_card(("deal_damage", 3))

        resolve_battlecry(player, card, opponent)
        assert opponent.armor == 0
        assert opponent.health == 29

    def test_resolve_draw(self):
        """draw should draw N cards."""
        player = Player()
        opponent = Player()
        player.deck = [MinionCard(name=f"D{i}", mana_cost=1, attack=1, health=1) for i in range(5)]
        card = self._make_card(("draw", 2))

        resolve_battlecry(player, card, opponent)
        assert len(player.hand) == 2

    def test_resolve_restore_health(self):
        """restore_health should heal player hero."""
        player = Player()
        player.health = 20
        opponent = Player()
        card = self._make_card(("restore_health", 5))

        resolve_battlecry(player, card, opponent)
        assert player.health == 25

    def test_resolve_restore_health_capped(self):
        """restore_health should not exceed 30."""
        player = Player()
        player.health = 28
        opponent = Player()
        card = self._make_card(("restore_health", 8))

        resolve_battlecry(player, card, opponent)
        assert player.health == 30

    def test_resolve_gain_armor(self):
        """gain_armor should increase player armor."""
        player = Player()
        opponent = Player()
        card = self._make_card(("gain_armor", 5))

        resolve_battlecry(player, card, opponent)
        assert player.armor == 5

    def test_resolve_summon(self):
        """summon should add a token to board."""
        player = Player()
        opponent = Player()
        card = self._make_card(("summon", 1, 1))

        resolve_battlecry(player, card, opponent)
        assert len(player.board) == 1
        assert player.board[0].attack == 1
        assert player.board[0].health == 1

    def test_resolve_summon_full_board(self):
        """summon should not add token if board is full."""
        player = Player()
        player.board = [MinionCard(name=f"M{i}", mana_cost=1, attack=1, health=1) for i in range(7)]
        opponent = Player()
        card = self._make_card(("summon", 1, 1))

        resolve_battlecry(player, card, opponent)
        assert len(player.board) == 7  # unchanged

    def test_resolve_buff_all(self):
        """buff_all should give +X/+Y to all friendly minions."""
        player = Player()
        m1 = MinionCard(name="M1", mana_cost=1, attack=2, health=3)
        m2 = MinionCard(name="M2", mana_cost=2, attack=3, health=4)
        player.board = [m1, m2]
        opponent = Player()
        card = self._make_card(("buff_all", 1, 1))

        resolve_battlecry(player, card, opponent)
        assert m1.attack == 3
        assert m1.health == 4
        assert m2.attack == 4
        assert m2.health == 5

    def test_resolve_no_effect(self):
        """Card with no battlecry_effect does nothing."""
        player = Player()
        opponent = Player()
        card = MinionCard(name="Test", mana_cost=3, attack=3, health=3)

        resolve_battlecry(player, card, opponent)
        assert player.health == 30
        assert opponent.health == 30

    def test_resolve_no_opponent(self):
        """deal_damage with no opponent does nothing."""
        player = Player()
        card = self._make_card(("deal_damage", 3))

        resolve_battlecry(player, card, None)
        assert player.health == 30


# ============================================================
# Integration Tests
# ============================================================


class TestBattlecryIntegration:
    """Test battlecry effects trigger through play_card and deck building."""

    def test_play_card_triggers_battlecry(self):
        """Playing a BATTLECRY minion should trigger its effect."""
        game = Game()
        p1 = game.player1
        p2 = game.player2
        p1.mana = 10

        card = MinionCard(name="Healer", mana_cost=3, attack=2, health=2, mechanics=["BATTLECRY"])
        card.battlecry_effect = ("restore_health", 5)
        p1.hand.append(card)
        p1.health = 20

        play_card(p1, card_index=0, opponent=p2)
        assert p1.health == 25
        assert len(p1.board) == 1

    def test_play_card_no_battlecry_no_effect(self):
        """Playing a non-BATTLECRY minion should not trigger anything extra."""
        game = Game()
        p1 = game.player1
        p2 = game.player2
        p1.mana = 10

        card = MinionCard(name="Vanilla", mana_cost=2, attack=3, health=2)
        p1.hand.append(card)

        play_card(p1, card_index=0, opponent=p2)
        assert p2.health == 30
        assert len(p1.board) == 1

    def test_battlecry_deal_damage_via_play_card(self):
        """Deal damage battlecry should hit opponent hero through play_card."""
        game = Game()
        p1 = game.player1
        p2 = game.player2
        p1.mana = 10

        card = MinionCard(name="Damager", mana_cost=4, attack=4, health=2, mechanics=["BATTLECRY"])
        card.battlecry_effect = ("deal_damage", 3)
        p1.hand.append(card)

        play_card(p1, card_index=0, opponent=p2)
        assert p2.health == 27

    def test_battlecry_in_supported_mechanics(self):
        """BATTLECRY should be in SUPPORTED_MECHANICS."""
        from deckbuilding.deck import SUPPORTED_MECHANICS
        assert "BATTLECRY" in SUPPORTED_MECHANICS

    def test_cardspec_has_battlecry_effect(self):
        """CardSpec should support battlecry_effect field."""
        from deckbuilding.deck import CardSpec
        spec = CardSpec(
            name="Test", mana_cost=3, attack=3, health=3,
            mechanics=["BATTLECRY"],
            battlecry_effect=("deal_damage", 2),
        )
        assert spec.battlecry_effect == ("deal_damage", 2)

    def test_cardspec_has_text(self):
        """CardSpec should support text field."""
        from deckbuilding.deck import CardSpec
        spec = CardSpec(
            name="Test", mana_cost=3, attack=3, health=3,
            text="<b>Battlecry:</b> Deal 2 damage.",
        )
        assert spec.text is not None

    def test_build_concrete_deck_copies_battlecry_effect(self):
        """build_concrete_deck should copy battlecry_effect onto MinionCard."""
        from deckbuilding.deck import CardSpec, build_concrete_deck
        import random

        pool = {
            "Zapper": CardSpec(
                name="Zapper", mana_cost=3, attack=3, health=2,
                mechanics=["BATTLECRY"],
                battlecry_effect=("deal_damage", 2),
            ),
        }
        deck = build_concrete_deck(["Zapper"], pool=pool, rng=random.Random(42))
        assert len(deck) == 1
        assert getattr(deck[0], 'battlecry_effect', None) == ("deal_damage", 2)
