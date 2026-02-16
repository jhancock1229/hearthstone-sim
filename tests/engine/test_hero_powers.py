"""Tests for hero power implementation.

Tests are organized by implementation step:
1. Player attributes (hero_class, armor, take_damage)
2. Game hero class integration
3. Individual hero power effects (all 11 classes)
4. Action space hero power enumeration
5. Simulator integration
"""

import pytest
from hearthstone.engine.player import Player
from hearthstone.engine.game import Game
from hearthstone.engine.actions import use_hero_power
from hearthstone.cards.base import MinionCard
from hearthstone.exceptions import IllegalActionError


# ============================================================
# Step 1: Player Attributes
# ============================================================


class TestPlayerAttributes:
    """Test hero_class, armor, hero_power_used, and take_damage on Player."""

    def test_player_has_hero_class(self):
        """Player should have a hero_class attribute defaulting to NEUTRAL."""
        p = Player()
        assert p.hero_class == "NEUTRAL"

    def test_player_has_armor(self):
        """Player should have armor attribute defaulting to 0."""
        p = Player()
        assert p.armor == 0

    def test_player_has_hero_power_used(self):
        """Player should have hero_power_used attribute defaulting to False."""
        p = Player()
        assert p.hero_power_used is False

    def test_take_damage_reduces_armor_first(self):
        """take_damage should consume armor before health."""
        p = Player()
        p.armor = 5
        p.health = 30
        p.take_damage(3)
        assert p.armor == 2
        assert p.health == 30

    def test_take_damage_excess_goes_to_health(self):
        """take_damage excess beyond armor should reduce health."""
        p = Player()
        p.armor = 2
        p.health = 30
        p.take_damage(5)
        assert p.armor == 0
        assert p.health == 27

    def test_take_damage_no_armor(self):
        """take_damage with no armor reduces health directly."""
        p = Player()
        p.health = 30
        p.take_damage(4)
        assert p.health == 26

    def test_take_damage_zero(self):
        """take_damage(0) changes nothing."""
        p = Player()
        p.armor = 3
        p.health = 30
        p.take_damage(0)
        assert p.armor == 3
        assert p.health == 30


# ============================================================
# Step 2: Game Hero Class Integration
# ============================================================


class TestGameHeroClass:
    """Test Game passes hero classes to players and resets hero power."""

    def test_game_assigns_hero_classes(self):
        """Game should assign hero classes to players."""
        game = Game(player1_class="MAGE", player2_class="WARRIOR")
        assert game.player1.hero_class == "MAGE"
        assert game.player2.hero_class == "WARRIOR"

    def test_game_default_hero_classes(self):
        """Game should default to MAGE for both players."""
        game = Game()
        assert game.player1.hero_class == "MAGE"
        assert game.player2.hero_class == "MAGE"

    def test_hero_power_resets_on_turn_start(self):
        """hero_power_used should reset to False on turn start."""
        game = Game()
        game.player1.deck = [MinionCard(name=f"M{i}", mana_cost=1, attack=1, health=1) for i in range(10)]
        game.player1.hero_power_used = True
        game._active_player = game.player1
        game.start_turn()
        assert game.player1.hero_power_used is False


# ============================================================
# Step 3: Hero Power Effects
# ============================================================


class TestMageHeroPower:
    """Mage: Fireblast - Deal 1 damage to a target."""

    def test_mage_fireblast_deals_1_damage_to_minion(self):
        """Fireblast should deal 1 damage to a target minion."""
        game = Game(player1_class="MAGE", player2_class="WARRIOR")
        p1 = game.player1
        p1.mana = 5
        target = MinionCard(name="Target", mana_cost=2, attack=2, health=3)
        game.player2.board.append(target)

        use_hero_power(p1, game=game, target=("opponent_minion", 0))
        assert target.health == 2
        assert p1.mana == 3

    def test_mage_fireblast_deals_1_damage_to_hero(self):
        """Fireblast should deal 1 damage to enemy hero."""
        game = Game(player1_class="MAGE", player2_class="WARRIOR")
        p1 = game.player1
        p1.mana = 5

        use_hero_power(p1, game=game, target=("opponent_hero",))
        assert game.player2.health == 29
        assert p1.mana == 3

    def test_mage_fireblast_requires_target(self):
        """Mage hero power requires a target."""
        game = Game(player1_class="MAGE", player2_class="WARRIOR")
        p1 = game.player1
        p1.mana = 5

        with pytest.raises(IllegalActionError, match="[Tt]arget"):
            use_hero_power(p1, game=game, target=None)


class TestWarlockHeroPower:
    """Warlock: Life Tap - Draw a card, take 2 damage."""

    def test_warlock_life_tap_draws_and_damages(self):
        """Life Tap should draw a card and deal 2 damage to self."""
        game = Game(player1_class="WARLOCK", player2_class="MAGE")
        p1 = game.player1
        p1.mana = 5
        p1.deck = [MinionCard(name="Drawn", mana_cost=1, attack=1, health=1)]
        initial_hand = len(p1.hand)

        use_hero_power(p1, game=game)
        assert p1.health == 28
        assert len(p1.hand) == initial_hand + 1
        assert p1.mana == 3


class TestPriestHeroPower:
    """Priest: Lesser Heal - Restore 2 health to a target."""

    def test_priest_lesser_heal_restores_health(self):
        """Lesser Heal should restore 2 health to a target."""
        game = Game(player1_class="PRIEST", player2_class="MAGE")
        p1 = game.player1
        p1.mana = 5
        p1.health = 25

        use_hero_power(p1, game=game, target=("self_hero",))
        assert p1.health == 27
        assert p1.mana == 3

    def test_priest_heal_does_not_exceed_max(self):
        """Healing should not exceed 30 health."""
        game = Game(player1_class="PRIEST", player2_class="MAGE")
        p1 = game.player1
        p1.mana = 5
        p1.health = 29

        use_hero_power(p1, game=game, target=("self_hero",))
        assert p1.health == 30

    def test_priest_heal_requires_target(self):
        """Priest hero power requires a target."""
        game = Game(player1_class="PRIEST", player2_class="MAGE")
        p1 = game.player1
        p1.mana = 5

        with pytest.raises(IllegalActionError, match="[Tt]arget"):
            use_hero_power(p1, game=game, target=None)


class TestPaladinHeroPower:
    """Paladin: Reinforce - Summon a 1/1 Silver Hand Recruit."""

    def test_paladin_summons_recruit(self):
        """Reinforce should summon a 1/1 minion."""
        game = Game(player1_class="PALADIN", player2_class="MAGE")
        p1 = game.player1
        p1.mana = 5

        use_hero_power(p1, game=game)
        assert len(p1.board) == 1
        recruit = p1.board[0]
        assert recruit.attack == 1
        assert recruit.health == 1
        assert p1.mana == 3

    def test_paladin_fails_on_full_board(self):
        """Reinforce should fail if board is full."""
        game = Game(player1_class="PALADIN", player2_class="MAGE")
        p1 = game.player1
        p1.mana = 5
        p1.board = [MinionCard(name=f"M{i}", mana_cost=1, attack=1, health=1) for i in range(7)]

        with pytest.raises(IllegalActionError, match="[Bb]oard.*full"):
            use_hero_power(p1, game=game)


class TestHunterHeroPower:
    """Hunter: Steady Shot - Deal 2 damage to enemy hero."""

    def test_hunter_deals_2_to_enemy_hero(self):
        """Steady Shot should deal 2 damage to enemy hero."""
        game = Game(player1_class="HUNTER", player2_class="MAGE")
        p1 = game.player1
        p1.mana = 5

        use_hero_power(p1, game=game)
        assert game.player2.health == 28
        assert p1.mana == 3


class TestWarriorHeroPower:
    """Warrior: Armor Up! - Gain 2 armor."""

    def test_warrior_gains_2_armor(self):
        """Armor Up! should give 2 armor."""
        game = Game(player1_class="WARRIOR", player2_class="MAGE")
        p1 = game.player1
        p1.mana = 5

        use_hero_power(p1, game=game)
        assert p1.armor == 2
        assert p1.mana == 3

    def test_warrior_armor_stacks(self):
        """Armor should stack across turns."""
        game = Game(player1_class="WARRIOR", player2_class="MAGE")
        p1 = game.player1
        p1.mana = 5
        p1.armor = 3

        use_hero_power(p1, game=game)
        assert p1.armor == 5


class TestShamanHeroPower:
    """Shaman: Totemic Call - Summon a 0/2 Taunt totem."""

    def test_shaman_summons_taunt_totem(self):
        """Totemic Call should summon a 0/2 with TAUNT."""
        game = Game(player1_class="SHAMAN", player2_class="MAGE")
        p1 = game.player1
        p1.mana = 5

        use_hero_power(p1, game=game)
        assert len(p1.board) == 1
        totem = p1.board[0]
        assert totem.attack == 0
        assert totem.health == 2
        assert "TAUNT" in totem.mechanics
        assert p1.mana == 3


class TestRogueHeroPower:
    """Rogue: Dagger Mastery - Deal 1 damage to a random enemy minion (simplified)."""

    def test_rogue_deals_1_damage_to_enemy_minion(self):
        """Dagger Mastery should deal 1 damage to a random enemy minion."""
        game = Game(player1_class="ROGUE", player2_class="MAGE")
        p1 = game.player1
        p1.mana = 5
        target = MinionCard(name="Target", mana_cost=2, attack=2, health=3)
        game.player2.board.append(target)

        use_hero_power(p1, game=game)
        assert target.health == 2
        assert p1.mana == 3

    def test_rogue_no_minions_hits_hero(self):
        """Rogue hero power hits enemy hero if no enemy minions."""
        game = Game(player1_class="ROGUE", player2_class="MAGE")
        p1 = game.player1
        p1.mana = 5

        use_hero_power(p1, game=game)
        assert game.player2.health == 29


class TestDruidHeroPower:
    """Druid: Shapeshift - Gain 1 armor (simplified)."""

    def test_druid_gains_1_armor(self):
        """Shapeshift should give 1 armor."""
        game = Game(player1_class="DRUID", player2_class="MAGE")
        p1 = game.player1
        p1.mana = 5

        use_hero_power(p1, game=game)
        assert p1.armor == 1
        assert p1.mana == 3


class TestDemonHunterHeroPower:
    """Demon Hunter: Demon Claws - Deal 1 damage to enemy hero."""

    def test_demonhunter_deals_1_to_enemy(self):
        """Demon Claws should deal 1 damage to enemy hero."""
        game = Game(player1_class="DEMONHUNTER", player2_class="MAGE")
        p1 = game.player1
        p1.mana = 5

        use_hero_power(p1, game=game)
        assert game.player2.health == 29
        assert p1.mana == 3


class TestDeathKnightHeroPower:
    """Death Knight: Ghoul Charge - Deal 1 damage to all enemy minions (simplified)."""

    def test_deathknight_damages_all_enemy_minions(self):
        """Ghoul Charge should deal 1 damage to all enemy minions."""
        game = Game(player1_class="DEATHKNIGHT", player2_class="MAGE")
        p1 = game.player1
        p1.mana = 5
        m1 = MinionCard(name="M1", mana_cost=2, attack=2, health=3)
        m2 = MinionCard(name="M2", mana_cost=3, attack=3, health=4)
        game.player2.board.extend([m1, m2])

        use_hero_power(p1, game=game)
        assert m1.health == 2
        assert m2.health == 3
        assert p1.mana == 3

    def test_deathknight_no_minions_is_noop(self):
        """Ghoul Charge with no enemy minions does nothing extra."""
        game = Game(player1_class="DEATHKNIGHT", player2_class="MAGE")
        p1 = game.player1
        p1.mana = 5

        use_hero_power(p1, game=game)
        assert p1.mana == 3
        assert game.player2.health == 30


# ============================================================
# Step 4: Action Space Hero Power Enumeration
# ============================================================


class TestHeroPowerActionSpace:
    """Test hero power appears correctly in legal actions."""

    def test_hero_power_available_with_mana(self):
        """Hero power should appear in legal actions when mana >= 2."""
        from simulation.action_space import ActionSpace, ActionType
        game = Game(player1_class="WARRIOR", player2_class="MAGE")
        game.player1.mana = 5
        game._active_player = game.player1

        actions = ActionSpace.get_legal_actions(game)
        hp_actions = [a for a in actions if a.type == ActionType.HERO_POWER]
        assert len(hp_actions) >= 1

    def test_hero_power_unavailable_without_mana(self):
        """Hero power should not appear when mana < 2."""
        from simulation.action_space import ActionSpace, ActionType
        game = Game(player1_class="WARRIOR", player2_class="MAGE")
        game.player1.mana = 1
        game._active_player = game.player1

        actions = ActionSpace.get_legal_actions(game)
        hp_actions = [a for a in actions if a.type == ActionType.HERO_POWER]
        assert len(hp_actions) == 0

    def test_hero_power_unavailable_when_used(self):
        """Hero power should not appear when already used this turn."""
        from simulation.action_space import ActionSpace, ActionType
        game = Game(player1_class="WARRIOR", player2_class="MAGE")
        game.player1.mana = 5
        game.player1.hero_power_used = True
        game._active_player = game.player1

        actions = ActionSpace.get_legal_actions(game)
        hp_actions = [a for a in actions if a.type == ActionType.HERO_POWER]
        assert len(hp_actions) == 0

    def test_mage_hero_power_has_targets(self):
        """Mage hero power should enumerate targets (minions + heroes)."""
        from simulation.action_space import ActionSpace, ActionType
        game = Game(player1_class="MAGE", player2_class="WARRIOR")
        game.player1.mana = 5
        game._active_player = game.player1
        # Add an enemy minion
        game.player2.board.append(MinionCard(name="M1", mana_cost=1, attack=1, health=2))

        actions = ActionSpace.get_legal_actions(game)
        hp_actions = [a for a in actions if a.type == ActionType.HERO_POWER]
        # Should have targets: enemy minion + enemy hero + friendly hero + friendly minions (0)
        # At minimum: enemy minion, enemy hero, self hero = 3 targets
        assert len(hp_actions) >= 3

    def test_untargeted_hero_power_single_action(self):
        """Untargeted hero powers (Warrior, Hunter, etc.) should produce 1 action."""
        from simulation.action_space import ActionSpace, ActionType
        game = Game(player1_class="HUNTER", player2_class="MAGE")
        game.player1.mana = 5
        game._active_player = game.player1

        actions = ActionSpace.get_legal_actions(game)
        hp_actions = [a for a in actions if a.type == ActionType.HERO_POWER]
        assert len(hp_actions) == 1
        assert hp_actions[0].target is None


# ============================================================
# Step 5: Simulator Integration
# ============================================================


class TestHeroPowerSimulatorIntegration:
    """Test hero powers work in full game simulation."""

    def test_simulator_hero_power_executes(self):
        """Smoke test: run a game and verify hero powers can execute."""
        from simulation.simulator import Simulator
        from agents.greedy_agent import GreedyAgent

        sim = Simulator(seed=42, max_turns=20)
        g1 = GreedyAgent(name="G1")
        g2 = GreedyAgent(name="G2")

        # Should not raise — hero powers execute during game
        result = sim.run_games(g1, g2, num_games=1)
        assert result.total_games == 1
