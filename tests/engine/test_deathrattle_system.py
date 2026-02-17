"""Tests for deathrattle system: text parsing, effect resolution, integration.

Covers:
- parse_deathrattle_text: 8 patterns + edge cases
- resolve_deathrattle: each effect type
- process_deaths: triggers deathrattle effects
- Deck building: pool includes deathrattle cards, deck copies effect
- EnhancedGreedyAgent: values deathrattle minions higher
"""

import pytest
from hearthstone.cards.base import MinionCard
from hearthstone.engine.player import Player


# ── Helpers ──────────────────────────────────────────────────────

def _minion(name="Test", mana=1, attack=1, health=1, mechanics=None, **kwargs):
    m = MinionCard(name=name, mana_cost=mana, attack=attack, health=health,
                   mechanics=mechanics or [])
    for k, v in kwargs.items():
        setattr(m, k, v)
    m.summoning_sick = False
    m.exhausted = False
    return m


def _player_with_board(minions=None, health=30):
    p = Player()
    p.health = health
    p.mana = 10
    p.max_mana = 10
    if minions:
        p.board = list(minions)
    return p


# ══════════════════════════════════════════════════════════════════
# Parser tests
# ══════════════════════════════════════════════════════════════════

class TestParseDeathrattleText:
    """Tests for parse_deathrattle_text in battlecries.py."""

    def test_deal_damage(self):
        from hearthstone.cards.battlecries import parse_deathrattle_text
        assert parse_deathrattle_text("<b>Deathrattle:</b> Deal 2 damage to all minions.") is not None

    def test_deal_damage_single(self):
        from hearthstone.cards.battlecries import parse_deathrattle_text
        result = parse_deathrattle_text("<b>Deathrattle:</b> Deal 3 damage to the enemy hero.")
        assert result == ("deal_damage", 3)

    def test_aoe_damage_before_single(self):
        """AoE pattern matched before single-target deal damage."""
        from hearthstone.cards.battlecries import parse_deathrattle_text
        result = parse_deathrattle_text("<b>Deathrattle:</b> Deal 2 damage to all minions.")
        assert result == ("aoe_damage", 2)

    def test_draw_cards(self):
        from hearthstone.cards.battlecries import parse_deathrattle_text
        result = parse_deathrattle_text("<b>Deathrattle:</b> Draw a card.")
        assert result == ("draw", 1)

    def test_draw_multiple(self):
        from hearthstone.cards.battlecries import parse_deathrattle_text
        result = parse_deathrattle_text("<b>Deathrattle:</b> Draw 2 cards.")
        assert result == ("draw", 2)

    def test_restore_health(self):
        from hearthstone.cards.battlecries import parse_deathrattle_text
        result = parse_deathrattle_text("<b>Deathrattle:</b> Restore 5 health to your hero.")
        assert result == ("restore_health", 5)

    def test_gain_armor(self):
        from hearthstone.cards.battlecries import parse_deathrattle_text
        result = parse_deathrattle_text("<b>Deathrattle:</b> Gain 4 Armor.")
        assert result == ("gain_armor", 4)

    def test_destroy(self):
        from hearthstone.cards.battlecries import parse_deathrattle_text
        result = parse_deathrattle_text("<b>Deathrattle:</b> Destroy a random enemy minion.")
        assert result == ("destroy", 1)

    def test_summon(self):
        from hearthstone.cards.battlecries import parse_deathrattle_text
        result = parse_deathrattle_text("<b>Deathrattle:</b> Summon a 4/4 Nerubian.")
        assert result == ("summon", 4, 4)

    def test_summon_simple(self):
        """Summon without explicit stats falls back to 1/1."""
        from hearthstone.cards.battlecries import parse_deathrattle_text
        result = parse_deathrattle_text("<b>Deathrattle:</b> Summon a Leper Gnome.")
        assert result == ("summon", 1, 1)

    def test_buff_all(self):
        from hearthstone.cards.battlecries import parse_deathrattle_text
        result = parse_deathrattle_text("<b>Deathrattle:</b> Give your minions +1/+1.")
        assert result == ("buff_all", 1, 1)

    def test_none_text(self):
        from hearthstone.cards.battlecries import parse_deathrattle_text
        assert parse_deathrattle_text(None) is None

    def test_empty_text(self):
        from hearthstone.cards.battlecries import parse_deathrattle_text
        assert parse_deathrattle_text("") is None

    def test_unrecognized_text(self):
        from hearthstone.cards.battlecries import parse_deathrattle_text
        assert parse_deathrattle_text("<b>Deathrattle:</b> Add a random Legendary card to your hand.") is None

    def test_html_stripped(self):
        from hearthstone.cards.battlecries import parse_deathrattle_text
        result = parse_deathrattle_text("<b>Deathrattle:</b> Deal $3 damage to the enemy hero.")
        assert result == ("deal_damage", 3)


# ══════════════════════════════════════════════════════════════════
# Resolver tests
# ══════════════════════════════════════════════════════════════════

class TestResolveDeathrattle:
    """Tests for resolve_deathrattle in actions.py."""

    def test_no_effect_is_noop(self):
        from hearthstone.engine.actions import resolve_deathrattle
        player = _player_with_board()
        opponent = _player_with_board()
        minion = _minion()
        # Should not raise
        resolve_deathrattle(player, minion, opponent)

    def test_deal_damage(self):
        from hearthstone.engine.actions import resolve_deathrattle
        player = _player_with_board()
        opponent = _player_with_board(health=30)
        minion = _minion()
        minion.deathrattle_effect = ("deal_damage", 3)
        resolve_deathrattle(player, minion, opponent)
        assert opponent.health == 27

    def test_aoe_damage(self):
        from hearthstone.engine.actions import resolve_deathrattle
        player = _player_with_board()
        m1 = _minion(health=3)
        m2 = _minion(health=2)
        opponent = _player_with_board(minions=[m1, m2])
        minion = _minion()
        minion.deathrattle_effect = ("aoe_damage", 2)
        resolve_deathrattle(player, minion, opponent)
        assert m1.health == 1
        assert m2.health == 0
        # Dead minions removed
        assert len(opponent.board) == 1

    def test_draw(self):
        from hearthstone.engine.actions import resolve_deathrattle
        player = _player_with_board()
        player.deck = [_minion(name="A"), _minion(name="B")]
        opponent = _player_with_board()
        minion = _minion()
        minion.deathrattle_effect = ("draw", 1)
        resolve_deathrattle(player, minion, opponent)
        assert len(player.hand) == 1

    def test_restore_health(self):
        from hearthstone.engine.actions import resolve_deathrattle
        player = _player_with_board(health=20)
        opponent = _player_with_board()
        minion = _minion()
        minion.deathrattle_effect = ("restore_health", 5)
        resolve_deathrattle(player, minion, opponent)
        assert player.health == 25

    def test_restore_health_caps_at_30(self):
        from hearthstone.engine.actions import resolve_deathrattle
        player = _player_with_board(health=28)
        opponent = _player_with_board()
        minion = _minion()
        minion.deathrattle_effect = ("restore_health", 5)
        resolve_deathrattle(player, minion, opponent)
        assert player.health == 30

    def test_gain_armor(self):
        from hearthstone.engine.actions import resolve_deathrattle
        player = _player_with_board()
        opponent = _player_with_board()
        minion = _minion()
        minion.deathrattle_effect = ("gain_armor", 4)
        resolve_deathrattle(player, minion, opponent)
        assert player.armor == 4

    def test_destroy(self):
        from hearthstone.engine.actions import resolve_deathrattle
        player = _player_with_board()
        enemy = _minion(name="Target", health=5)
        opponent = _player_with_board(minions=[enemy])
        minion = _minion()
        minion.deathrattle_effect = ("destroy", 1)
        resolve_deathrattle(player, minion, opponent)
        assert len(opponent.board) == 0

    def test_destroy_empty_board(self):
        from hearthstone.engine.actions import resolve_deathrattle
        player = _player_with_board()
        opponent = _player_with_board()
        minion = _minion()
        minion.deathrattle_effect = ("destroy", 1)
        # Should not raise on empty board
        resolve_deathrattle(player, minion, opponent)

    def test_summon(self):
        from hearthstone.engine.actions import resolve_deathrattle
        player = _player_with_board()
        opponent = _player_with_board()
        minion = _minion()
        minion.deathrattle_effect = ("summon", 3, 3)
        resolve_deathrattle(player, minion, opponent)
        assert len(player.board) == 1
        assert player.board[0].attack == 3
        assert player.board[0].health == 3

    def test_buff_all(self):
        from hearthstone.engine.actions import resolve_deathrattle
        m1 = _minion(name="A", attack=2, health=3)
        m2 = _minion(name="B", attack=1, health=1)
        player = _player_with_board(minions=[m1, m2])
        opponent = _player_with_board()
        minion = _minion()  # The dying minion (not on board)
        minion.deathrattle_effect = ("buff_all", 1, 1)
        resolve_deathrattle(player, minion, opponent)
        assert m1.attack == 3
        assert m1.health == 4
        assert m2.attack == 2
        assert m2.health == 2


# ══════════════════════════════════════════════════════════════════
# Integration tests — process_deaths triggers deathrattle
# ══════════════════════════════════════════════════════════════════

class TestDeathrattleInCombat:
    """Tests that process_deaths triggers deathrattle resolution."""

    def test_process_deaths_triggers_deathrattle_effect(self):
        """A dying minion with deathrattle_effect should have it resolved."""
        from hearthstone.engine.combat import process_deaths
        minion = _minion(name="Bomber", health=0, mechanics=["DEATHRATTLE"])
        minion.deathrattle_effect = ("deal_damage", 2)
        player = _player_with_board(minions=[minion])
        opponent = _player_with_board(health=30)
        process_deaths(player, opponent)
        assert opponent.health == 28
        assert len(player.board) == 0

    def test_process_deaths_no_effect_still_dies(self):
        """Minion with DEATHRATTLE mechanic but no parsed effect still gets removed."""
        from hearthstone.engine.combat import process_deaths
        minion = _minion(name="Mystery", health=0, mechanics=["DEATHRATTLE"])
        player = _player_with_board(minions=[minion])
        process_deaths(player)
        assert len(player.board) == 0

    def test_deathrattle_summon_in_combat(self):
        """Deathrattle summon adds token after dying minion is removed."""
        from hearthstone.engine.combat import process_deaths
        dying = _minion(name="Egg", attack=0, health=0, mechanics=["DEATHRATTLE"])
        dying.deathrattle_effect = ("summon", 4, 4)
        player = _player_with_board(minions=[dying])
        opponent = _player_with_board()
        process_deaths(player, opponent)
        # Egg removed, token added
        assert len(player.board) == 1
        assert player.board[0].name == "Token"
        assert player.board[0].attack == 4

    def test_backward_compatible_no_opponent(self):
        """process_deaths still works when called without opponent arg."""
        from hearthstone.engine.combat import process_deaths
        dying = _minion(health=0)
        player = _player_with_board(minions=[dying])
        process_deaths(player)
        assert len(player.board) == 0


# ══════════════════════════════════════════════════════════════════
# Deck building integration
# ══════════════════════════════════════════════════════════════════

class TestDeathrattleDeckBuilding:
    """Tests that deathrattle cards enter the pool and deck builder copies effects."""

    def test_deathrattle_in_supported_mechanics(self):
        from deckbuilding.deck import SUPPORTED_MECHANICS
        assert "DEATHRATTLE" in SUPPORTED_MECHANICS

    def test_cardspec_has_deathrattle_effect(self):
        from deckbuilding.deck import CardSpec
        spec = CardSpec("Test", 3, 2, 4, deathrattle_effect=("draw", 1))
        assert spec.deathrattle_effect == ("draw", 1)

    def test_build_concrete_deck_copies_deathrattle_effect(self):
        from deckbuilding.deck import build_concrete_deck, CardSpec
        import random as rng_mod
        pool = {
            "DR_Card": CardSpec(
                "DR Card", 3, 2, 4,
                mechanics=["DEATHRATTLE"],
                deathrattle_effect=("deal_damage", 2),
            ),
        }
        deck = build_concrete_deck(["DR_Card"], pool=pool, rng=rng_mod.Random(42))
        assert len(deck) == 1
        assert hasattr(deck[0], 'deathrattle_effect')
        assert deck[0].deathrattle_effect == ("deal_damage", 2)

    def test_pool_builder_includes_deathrattle_minions(self):
        """build_pool_from_registry includes minions with DEATHRATTLE + parseable text."""
        from pathlib import Path
        pytest.importorskip("hearthstone.cards.registry")
        from hearthstone.cards.registry import CardRegistry
        from deckbuilding.deck import build_pool_from_registry

        cards_json = Path(__file__).parent.parent.parent / "hearthstone" / "data" / "cards_collectible.json"
        if not cards_json.exists():
            pytest.skip("Card data not available")
        registry = CardRegistry.from_json(cards_json)
        pool = build_pool_from_registry(registry)

        # Find at least one card with deathrattle_effect set
        dr_cards = [s for s in pool.values() if s.deathrattle_effect is not None]
        assert len(dr_cards) > 0, "Expected deathrattle cards in pool"


# ══════════════════════════════════════════════════════════════════
# Agent valuation
# ══════════════════════════════════════════════════════════════════

class TestDeathrattleAgentValuation:
    """EnhancedGreedyAgent should value deathrattle minions higher."""

    def test_deathrattle_minion_valued_higher(self):
        from agents.enhanced_greedy_agent import EnhancedGreedyAgent
        from simulation.observation import Observation
        from simulation.action_space import Action, ActionType

        agent = EnhancedGreedyAgent()

        # Create a vanilla minion and a deathrattle minion with same stats
        vanilla = _minion(name="Vanilla", mana=3, attack=3, health=3)
        dr_minion = _minion(name="DR", mana=3, attack=3, health=3, mechanics=["DEATHRATTLE"])
        dr_minion.deathrattle_effect = ("deal_damage", 3)

        obs_vanilla = Observation(
            self_health=30, self_mana=10, self_max_mana=10,
            self_fatigue_counter=0, self_hand=[vanilla],
            self_deck_size=20, self_board=[],
            opponent_health=30, opponent_mana=10, opponent_max_mana=10,
            opponent_fatigue_counter=0, opponent_hand_size=5,
            opponent_deck_size=20, opponent_board=[],
            turn_number=5, is_my_turn=True, is_game_over=False,
            winner=None, player_index=0,
            legal_actions=[
                Action(type=ActionType.PLAY_CARD, card_index=0),
                Action(type=ActionType.END_TURN),
            ],
        )

        obs_dr = Observation(
            self_health=30, self_mana=10, self_max_mana=10,
            self_fatigue_counter=0, self_hand=[dr_minion],
            self_deck_size=20, self_board=[],
            opponent_health=30, opponent_mana=10, opponent_max_mana=10,
            opponent_fatigue_counter=0, opponent_hand_size=5,
            opponent_deck_size=20, opponent_board=[],
            turn_number=5, is_my_turn=True, is_game_over=False,
            winner=None, player_index=0,
            legal_actions=[
                Action(type=ActionType.PLAY_CARD, card_index=0),
                Action(type=ActionType.END_TURN),
            ],
        )

        play_vanilla = Action(type=ActionType.PLAY_CARD, card_index=0)
        play_dr = Action(type=ActionType.PLAY_CARD, card_index=0)

        score_vanilla = agent._evaluate_action(play_vanilla, obs_vanilla)
        score_dr = agent._evaluate_action(play_dr, obs_dr)
        assert score_dr > score_vanilla
