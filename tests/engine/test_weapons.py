"""Tests for weapon system: equip, hero attack, durability, deck building.

Tests organized by:
1. Player weapon state
2. Weapon equipping via play_card
3. Hero attack mechanics
4. Action space hero attack enumeration
5. Deck building integration
"""

import pytest
from hearthstone.engine.player import Player
from hearthstone.engine.game import Game
from hearthstone.cards.base import MinionCard, WeaponCard
from hearthstone.engine.actions import play_card, hero_attack
from hearthstone.exceptions import IllegalActionError
from simulation.action_space import ActionSpace, ActionType


# ============================================================
# Player Weapon State
# ============================================================


class TestPlayerWeaponState:
    """Test weapon slot and hero_attacked flag on Player."""

    def test_player_has_weapon_slot(self):
        """Player should have weapon attribute defaulting to None."""
        p = Player()
        assert p.weapon is None

    def test_player_has_hero_attacked(self):
        """Player should have hero_attacked flag defaulting to False."""
        p = Player()
        assert p.hero_attacked is False

    def test_equip_weapon(self):
        """Equipping a weapon sets player.weapon."""
        p = Player()
        w = WeaponCard(name="Fiery War Axe", mana_cost=3, attack=3, durability=2)
        p.weapon = w
        assert p.weapon is w
        assert p.weapon.attack == 3
        assert p.weapon.durability == 2

    def test_equip_replaces_existing(self):
        """Equipping a new weapon replaces the old one."""
        p = Player()
        w1 = WeaponCard(name="Axe", mana_cost=2, attack=2, durability=2)
        w2 = WeaponCard(name="Sword", mana_cost=5, attack=5, durability=2)
        p.weapon = w1
        p.weapon = w2
        assert p.weapon is w2


# ============================================================
# Weapon Equipping via play_card
# ============================================================


class TestPlayWeapon:
    """Test playing weapon cards from hand."""

    def test_play_weapon_card(self):
        """Playing a weapon should equip it to the player."""
        p = Player()
        p.mana = 10
        w = WeaponCard(name="Fiery War Axe", mana_cost=3, attack=3, durability=2)
        p.hand.append(w)

        play_card(p, card_index=0)
        assert p.weapon is w
        assert len(p.hand) == 0

    def test_play_weapon_spends_mana(self):
        """Playing a weapon should spend mana."""
        p = Player()
        p.mana = 10
        w = WeaponCard(name="Axe", mana_cost=3, attack=3, durability=2)
        p.hand.append(w)

        play_card(p, card_index=0)
        assert p.mana == 7

    def test_play_weapon_replaces_existing(self):
        """Playing a new weapon should replace the old one."""
        p = Player()
        p.mana = 10
        w1 = WeaponCard(name="Axe", mana_cost=2, attack=2, durability=2)
        w2 = WeaponCard(name="Sword", mana_cost=5, attack=5, durability=2)
        p.weapon = w1
        p.hand.append(w2)

        play_card(p, card_index=0)
        assert p.weapon is w2

    def test_play_weapon_not_on_board(self):
        """Playing a weapon should NOT add it to board."""
        p = Player()
        p.mana = 10
        w = WeaponCard(name="Axe", mana_cost=2, attack=2, durability=2)
        p.hand.append(w)

        play_card(p, card_index=0)
        assert len(p.board) == 0

    def test_play_weapon_with_battlecry(self):
        """Playing a weapon with BATTLECRY should trigger the effect."""
        p = Player()
        opponent = Player()
        p.mana = 10
        w = WeaponCard(name="Perdition's Blade", mana_cost=3, attack=2, durability=2)
        w.mechanics.append("BATTLECRY")
        w.battlecry_effect = ("deal_damage", 1)
        p.hand.append(w)

        play_card(p, card_index=0, opponent=opponent)
        assert p.weapon is w
        assert opponent.health == 29


# ============================================================
# Hero Attack Mechanics
# ============================================================


class TestHeroAttack:
    """Test hero attacking with equipped weapon."""

    def test_hero_attack_face(self):
        """Hero should deal weapon damage to opponent hero."""
        p = Player()
        opponent = Player()
        p.weapon = WeaponCard(name="Axe", mana_cost=2, attack=3, durability=2)

        hero_attack(p, opponent, defender_index=None)
        assert opponent.health == 27

    def test_hero_attack_minion(self):
        """Hero should deal weapon damage to a minion."""
        p = Player()
        opponent = Player()
        p.weapon = WeaponCard(name="Axe", mana_cost=2, attack=3, durability=2)
        m = MinionCard(name="Yeti", mana_cost=4, attack=4, health=5)
        opponent.board.append(m)

        hero_attack(p, opponent, defender_index=0)
        assert m.health == 2

    def test_hero_takes_damage_from_minion(self):
        """Hero should take damage from minion when attacking it."""
        p = Player()
        opponent = Player()
        p.weapon = WeaponCard(name="Axe", mana_cost=2, attack=3, durability=2)
        m = MinionCard(name="Yeti", mana_cost=4, attack=4, health=5)
        opponent.board.append(m)

        hero_attack(p, opponent, defender_index=0)
        assert p.health == 26  # took 4 damage from Yeti

    def test_weapon_loses_durability(self):
        """Weapon should lose 1 durability per attack."""
        p = Player()
        opponent = Player()
        p.weapon = WeaponCard(name="Axe", mana_cost=2, attack=3, durability=2)

        hero_attack(p, opponent, defender_index=None)
        assert p.weapon.durability == 1

    def test_weapon_breaks_at_zero(self):
        """Weapon should be destroyed when durability reaches 0."""
        p = Player()
        opponent = Player()
        p.weapon = WeaponCard(name="Dagger", mana_cost=1, attack=1, durability=1)

        hero_attack(p, opponent, defender_index=None)
        assert p.weapon is None

    def test_cannot_attack_without_weapon(self):
        """Hero attack should fail without a weapon."""
        p = Player()
        opponent = Player()

        with pytest.raises(IllegalActionError, match="[Ww]eapon"):
            hero_attack(p, opponent, defender_index=None)

    def test_cannot_attack_twice(self):
        """Hero should not attack twice in one turn."""
        p = Player()
        opponent = Player()
        p.weapon = WeaponCard(name="Axe", mana_cost=2, attack=3, durability=3)

        hero_attack(p, opponent, defender_index=None)
        with pytest.raises(IllegalActionError, match="already attacked"):
            hero_attack(p, opponent, defender_index=None)

    def test_hero_attacked_resets_on_turn(self):
        """hero_attacked should reset at start of turn."""
        game = Game()
        game.player1.weapon = WeaponCard(name="Axe", mana_cost=2, attack=3, durability=3)
        game.player1.hero_attacked = True
        game.player1.deck = [MinionCard(name=f"M{i}", mana_cost=1, attack=1, health=1) for i in range(5)]
        game._active_player = game.player1

        game.start_turn()
        assert game.player1.hero_attacked is False

    def test_hero_kills_minion(self):
        """Hero attack should kill a minion with low enough health."""
        p = Player()
        opponent = Player()
        p.weapon = WeaponCard(name="Axe", mana_cost=2, attack=3, durability=2)
        m = MinionCard(name="Wisp", mana_cost=0, attack=1, health=1)
        opponent.board.append(m)

        hero_attack(p, opponent, defender_index=0)
        assert len(opponent.board) == 0
        assert p.health == 29  # took 1 damage from Wisp


# ============================================================
# Action Space Hero Attack
# ============================================================


class TestHeroAttackActionSpace:
    """Test hero attack appears in legal actions."""

    def test_hero_attack_in_legal_actions(self):
        """Hero attack should appear when player has a weapon."""
        game = Game()
        game.player1.weapon = WeaponCard(name="Axe", mana_cost=2, attack=3, durability=2)
        game.player1.mana = 5
        game._active_player = game.player1

        actions = ActionSpace.get_legal_actions(game)
        ha_actions = [a for a in actions if a.type == ActionType.HERO_ATTACK]
        # Should have at least face attack
        assert len(ha_actions) >= 1

    def test_no_hero_attack_without_weapon(self):
        """No hero attack when no weapon equipped."""
        game = Game()
        game.player1.mana = 5
        game._active_player = game.player1

        actions = ActionSpace.get_legal_actions(game)
        ha_actions = [a for a in actions if a.type == ActionType.HERO_ATTACK]
        assert len(ha_actions) == 0

    def test_no_hero_attack_when_already_attacked(self):
        """No hero attack when hero already attacked this turn."""
        game = Game()
        game.player1.weapon = WeaponCard(name="Axe", mana_cost=2, attack=3, durability=2)
        game.player1.hero_attacked = True
        game.player1.mana = 5
        game._active_player = game.player1

        actions = ActionSpace.get_legal_actions(game)
        ha_actions = [a for a in actions if a.type == ActionType.HERO_ATTACK]
        assert len(ha_actions) == 0

    def test_hero_attack_respects_taunt(self):
        """Hero attack should only target TAUNT minions when present."""
        game = Game()
        game.player1.weapon = WeaponCard(name="Axe", mana_cost=2, attack=3, durability=2)
        game.player1.mana = 5
        game._active_player = game.player1

        taunt = MinionCard(name="Taunt", mana_cost=2, attack=1, health=4, mechanics=["TAUNT"])
        normal = MinionCard(name="Normal", mana_cost=2, attack=2, health=3)
        game.player2.board = [taunt, normal]

        actions = ActionSpace.get_legal_actions(game)
        ha_actions = [a for a in actions if a.type == ActionType.HERO_ATTACK]
        # Should only be able to attack the TAUNT minion (index 0), not face or normal
        assert len(ha_actions) == 1
        assert ha_actions[0].defender_index == 0

    def test_hero_attack_targets_all_without_taunt(self):
        """Hero attack should target all minions + face without TAUNT."""
        game = Game()
        game.player1.weapon = WeaponCard(name="Axe", mana_cost=2, attack=3, durability=2)
        game.player1.mana = 5
        game._active_player = game.player1

        m1 = MinionCard(name="M1", mana_cost=1, attack=1, health=1)
        m2 = MinionCard(name="M2", mana_cost=2, attack=2, health=2)
        game.player2.board = [m1, m2]

        actions = ActionSpace.get_legal_actions(game)
        ha_actions = [a for a in actions if a.type == ActionType.HERO_ATTACK]
        # 2 minions + face = 3 targets
        assert len(ha_actions) == 3


# ============================================================
# Deck Building Integration
# ============================================================


class TestWeaponDeckBuilding:
    """Test weapons in deck pool and concrete deck building."""

    def test_cardspec_weapon_type(self):
        """CardSpec should support card_type=WEAPON."""
        from deckbuilding.deck import CardSpec
        spec = CardSpec(
            name="Axe", mana_cost=3, attack=3, health=2,
            card_type="WEAPON",
        )
        assert spec.card_type == "WEAPON"

    def test_build_concrete_deck_creates_weaponcard(self):
        """build_concrete_deck should create WeaponCard for weapon specs."""
        from deckbuilding.deck import CardSpec, build_concrete_deck
        import random

        pool = {
            "Axe": CardSpec(
                name="Fiery War Axe", mana_cost=3, attack=3, health=2,
                card_type="WEAPON",
            ),
        }
        deck = build_concrete_deck(["Axe"], pool=pool, rng=random.Random(42))
        assert len(deck) == 1
        assert isinstance(deck[0], WeaponCard)
        assert deck[0].attack == 3
        assert deck[0].durability == 2

    def test_build_concrete_deck_mixed_all_types(self):
        """build_concrete_deck should handle minions, spells, and weapons."""
        from deckbuilding.deck import CardSpec, build_concrete_deck
        from hearthstone.cards.base import SpellCard
        import random

        pool = {
            "Wisp": CardSpec(name="Wisp", mana_cost=0, attack=1, health=1),
            "Zap": CardSpec(name="Zap", mana_cost=1, attack=0, health=0, card_type="SPELL",
                            spell_effect=("deal_damage", 2)),
            "Axe": CardSpec(name="Axe", mana_cost=3, attack=3, health=2, card_type="WEAPON"),
        }
        deck = build_concrete_deck(["Wisp", "Zap", "Axe"], pool=pool, rng=random.Random(42))
        assert len(deck) == 3
        types = {type(c).__name__ for c in deck}
        assert types == {"MinionCard", "SpellCard", "WeaponCard"}
