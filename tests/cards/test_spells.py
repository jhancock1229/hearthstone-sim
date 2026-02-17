"""Tests for spell text parsing, effect resolution, and integration.

Tests are organized by:
1. Spell text parser (parse_spell_text)
2. Spell effect resolver (resolve_spell)
3. Integration (play_card, deck building)
"""

import pytest
from hearthstone.engine.player import Player
from hearthstone.engine.game import Game
from hearthstone.cards.base import MinionCard, SpellCard
from hearthstone.cards.battlecries import parse_spell_text
from hearthstone.engine.actions import resolve_spell, play_card


# ============================================================
# Spell Text Parser Tests
# ============================================================


class TestParseSpellText:
    """Test parse_spell_text extracts structured effects from spell text."""

    def test_parse_deal_damage(self):
        """Parse 'Deal N damage' pattern."""
        result = parse_spell_text("Deal $3 damage.")
        assert result == ("deal_damage", 3)

    def test_parse_aoe_damage(self):
        """Parse 'Deal N damage to all' pattern."""
        result = parse_spell_text("Deal $2 damage to all minions.")
        assert result == ("aoe_damage", 2)

    def test_parse_aoe_before_single(self):
        """AoE should match before single-target damage."""
        result = parse_spell_text("Deal $4 damage to all enemy minions.")
        assert result[0] == "aoe_damage"

    def test_parse_draw_a_card(self):
        """Parse 'Draw a card'."""
        result = parse_spell_text("Draw a card.")
        assert result == ("draw", 1)

    def test_parse_draw_n_cards(self):
        """Parse 'Draw N cards'."""
        result = parse_spell_text("Draw 3 cards.")
        assert result == ("draw", 3)

    def test_parse_restore_health(self):
        """Parse 'Restore N Health'."""
        result = parse_spell_text("Restore $5 Health.")
        assert result == ("restore_health", 5)

    def test_parse_gain_armor(self):
        """Parse 'Gain N Armor'."""
        result = parse_spell_text("Gain 5 Armor.")
        assert result == ("gain_armor", 5)

    def test_parse_destroy(self):
        """Parse 'Destroy a minion'."""
        result = parse_spell_text("Destroy a minion.")
        assert result == ("destroy",)

    def test_parse_summon(self):
        """Parse 'Summon' pattern."""
        result = parse_spell_text("Summon two 1/1 Imps.")
        assert result is not None
        assert result[0] == "summon"

    def test_parse_unrecognized(self):
        """Unrecognized text returns None."""
        result = parse_spell_text("Transform a minion into a 1/1 Sheep.")
        assert result is None

    def test_parse_none(self):
        result = parse_spell_text(None)
        assert result is None

    def test_parse_empty(self):
        result = parse_spell_text("")
        assert result is None

    def test_parse_strips_dollar_signs(self):
        """Spell damage markers ($) should not interfere with parsing."""
        result = parse_spell_text("Deal $6 damage to a minion.")
        assert result == ("deal_damage", 6)


# ============================================================
# Spell Effect Resolver Tests
# ============================================================


class TestResolveSpell:
    """Test resolve_spell applies effects correctly."""

    def _make_spell(self, effect):
        """Helper: create a SpellCard with a spell_effect."""
        card = SpellCard(name="Test Spell", mana_cost=3)
        card.spell_effect = effect
        return card

    def test_resolve_deal_damage(self):
        """deal_damage should reduce opponent hero health."""
        player = Player()
        opponent = Player()
        card = self._make_spell(("deal_damage", 4))

        resolve_spell(player, card, opponent)
        assert opponent.health == 26

    def test_resolve_deal_damage_with_armor(self):
        """deal_damage uses take_damage (armor absorbs)."""
        player = Player()
        opponent = Player()
        opponent.armor = 2
        card = self._make_spell(("deal_damage", 5))

        resolve_spell(player, card, opponent)
        assert opponent.armor == 0
        assert opponent.health == 27

    def test_resolve_aoe_damage(self):
        """aoe_damage should hit all enemy minions."""
        player = Player()
        opponent = Player()
        m1 = MinionCard(name="M1", mana_cost=1, attack=2, health=4)
        m2 = MinionCard(name="M2", mana_cost=2, attack=3, health=3)
        opponent.board = [m1, m2]
        card = self._make_spell(("aoe_damage", 2))

        resolve_spell(player, card, opponent)
        assert m1.health == 2
        assert m2.health == 1

    def test_resolve_aoe_kills(self):
        """aoe_damage should remove dead minions."""
        player = Player()
        opponent = Player()
        m1 = MinionCard(name="M1", mana_cost=1, attack=2, health=1)
        m2 = MinionCard(name="M2", mana_cost=2, attack=3, health=5)
        opponent.board = [m1, m2]
        card = self._make_spell(("aoe_damage", 2))

        resolve_spell(player, card, opponent)
        assert len(opponent.board) == 1
        assert opponent.board[0].name == "M2"

    def test_resolve_draw(self):
        """draw should draw N cards."""
        player = Player()
        opponent = Player()
        player.deck = [MinionCard(name=f"D{i}", mana_cost=1, attack=1, health=1) for i in range(5)]
        card = self._make_spell(("draw", 2))

        resolve_spell(player, card, opponent)
        assert len(player.hand) == 2

    def test_resolve_restore_health(self):
        """restore_health should heal player."""
        player = Player()
        player.health = 20
        opponent = Player()
        card = self._make_spell(("restore_health", 5))

        resolve_spell(player, card, opponent)
        assert player.health == 25

    def test_resolve_restore_health_capped(self):
        """restore_health capped at 30."""
        player = Player()
        player.health = 28
        opponent = Player()
        card = self._make_spell(("restore_health", 8))

        resolve_spell(player, card, opponent)
        assert player.health == 30

    def test_resolve_gain_armor(self):
        """gain_armor should increase player armor."""
        player = Player()
        opponent = Player()
        card = self._make_spell(("gain_armor", 5))

        resolve_spell(player, card, opponent)
        assert player.armor == 5

    def test_resolve_destroy(self):
        """destroy should remove an enemy minion."""
        player = Player()
        opponent = Player()
        m1 = MinionCard(name="Target", mana_cost=5, attack=5, health=5)
        opponent.board = [m1]
        card = self._make_spell(("destroy",))

        resolve_spell(player, card, opponent)
        assert len(opponent.board) == 0

    def test_resolve_destroy_empty_board(self):
        """destroy on empty board does nothing."""
        player = Player()
        opponent = Player()
        card = self._make_spell(("destroy",))

        resolve_spell(player, card, opponent)
        assert len(opponent.board) == 0
        assert opponent.health == 30

    def test_resolve_summon(self):
        """summon should add token to board."""
        player = Player()
        opponent = Player()
        card = self._make_spell(("summon", 1, 1))

        resolve_spell(player, card, opponent)
        assert len(player.board) == 1
        assert player.board[0].attack == 1
        assert player.board[0].health == 1

    def test_resolve_no_effect(self):
        """Spell with no spell_effect does nothing."""
        player = Player()
        opponent = Player()
        card = SpellCard(name="Nothing", mana_cost=1)

        resolve_spell(player, card, opponent)
        assert player.health == 30
        assert opponent.health == 30


# ============================================================
# Integration Tests
# ============================================================


class TestSpellIntegration:
    """Test spells work through play_card and deck building."""

    def test_play_spell_triggers_effect(self):
        """Playing a spell should trigger its effect."""
        player = Player()
        opponent = Player()
        player.mana = 10

        spell = SpellCard(name="Fireball", mana_cost=4)
        spell.spell_effect = ("deal_damage", 6)
        player.hand.append(spell)

        play_card(player, card_index=0, opponent=opponent)
        assert opponent.health == 24

    def test_play_spell_removed_from_hand(self):
        """Playing a spell should remove it from hand."""
        player = Player()
        opponent = Player()
        player.mana = 10

        spell = SpellCard(name="Sprint", mana_cost=7)
        spell.spell_effect = ("draw", 4)
        player.deck = [MinionCard(name=f"D{i}", mana_cost=1, attack=1, health=1) for i in range(10)]
        player.hand.append(spell)

        play_card(player, card_index=0, opponent=opponent)
        # spell removed, 4 drawn
        assert not any(isinstance(c, SpellCard) for c in player.hand)

    def test_play_spell_spends_mana(self):
        """Playing a spell should spend mana."""
        player = Player()
        opponent = Player()
        player.mana = 10

        spell = SpellCard(name="Armor", mana_cost=2)
        spell.spell_effect = ("gain_armor", 5)
        player.hand.append(spell)

        play_card(player, card_index=0, opponent=opponent)
        assert player.mana == 8

    def test_cardspec_card_type_field(self):
        """CardSpec should support card_type field."""
        from deckbuilding.deck import CardSpec
        spec = CardSpec(
            name="Fireball", mana_cost=4, attack=0, health=0,
            card_type="SPELL",
            spell_effect=("deal_damage", 6),
        )
        assert spec.card_type == "SPELL"
        assert spec.spell_effect == ("deal_damage", 6)

    def test_cardspec_default_card_type_is_minion(self):
        """CardSpec card_type should default to MINION."""
        from deckbuilding.deck import CardSpec
        spec = CardSpec(name="Test", mana_cost=1, attack=1, health=1)
        assert spec.card_type == "MINION"

    def test_build_concrete_deck_creates_spellcard(self):
        """build_concrete_deck should create SpellCard for spell specs."""
        from deckbuilding.deck import CardSpec, build_concrete_deck
        import random

        pool = {
            "Fireball": CardSpec(
                name="Fireball", mana_cost=4, attack=0, health=0,
                card_type="SPELL",
                spell_effect=("deal_damage", 6),
            ),
        }
        deck = build_concrete_deck(["Fireball"], pool=pool, rng=random.Random(42))
        assert len(deck) == 1
        assert isinstance(deck[0], SpellCard)
        assert getattr(deck[0], 'spell_effect', None) == ("deal_damage", 6)

    def test_build_concrete_deck_mixed(self):
        """build_concrete_deck should handle mix of minions and spells."""
        from deckbuilding.deck import CardSpec, build_concrete_deck
        import random

        pool = {
            "Wisp": CardSpec(name="Wisp", mana_cost=0, attack=1, health=1),
            "Zap": CardSpec(
                name="Zap", mana_cost=1, attack=0, health=0,
                card_type="SPELL",
                spell_effect=("deal_damage", 2),
            ),
        }
        deck = build_concrete_deck(["Wisp", "Zap"], pool=pool, rng=random.Random(42))
        assert len(deck) == 2
        types = {type(c).__name__ for c in deck}
        assert "MinionCard" in types
        assert "SpellCard" in types
