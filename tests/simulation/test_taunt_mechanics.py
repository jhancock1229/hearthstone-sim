"""Unit tests for TAUNT mechanic enforcement in ActionSpace.

TAUNT is a core Hearthstone mechanic:
- When opponent has minions with TAUNT, attackers MUST attack one of them
- Cannot attack face when enemy has TAUNT minions
- Cannot attack non-TAUNT minions when enemy has TAUNT minions
- Multiple TAUNT minions: can attack any of them
- When all TAUNT minions are removed, normal attack rules apply

These tests verify ActionSpace.get_legal_actions() correctly enforces TAUNT.
"""

import pytest
from hearthstone.engine.game import Game
from hearthstone.cards.base import MinionCard
from simulation.action_space import ActionSpace, ActionType


class TestTauntBasics:
    """Test basic TAUNT enforcement."""

    def test_cannot_attack_face_when_taunt_present(self):
        """Cannot attack enemy hero when they have a TAUNT minion."""
        game = Game()

        # P1 has an attacker
        attacker = MinionCard(name="Attacker", mana_cost=1, attack=2, health=2)
        attacker.summoning_sick = False
        game.player1.board.append(attacker)

        # P2 has a TAUNT minion
        taunt = MinionCard(name="Taunt", mana_cost=2, attack=1, health=4)
        taunt.mechanics.append("TAUNT")
        game.player2.board.append(taunt)

        actions = ActionSpace.get_legal_actions(game)
        attack_actions = [a for a in actions if a.type == ActionType.ATTACK]

        # Should have attack actions, but none targeting face
        assert len(attack_actions) > 0
        face_attacks = [a for a in attack_actions if a.defender_index is None]
        assert len(face_attacks) == 0, "Should not be able to attack face with TAUNT present"

    def test_cannot_attack_non_taunt_minions_when_taunt_present(self):
        """Cannot attack non-TAUNT minions when TAUNT is present."""
        game = Game()

        # P1 has an attacker
        attacker = MinionCard(name="Attacker", mana_cost=1, attack=2, health=2)
        attacker.summoning_sick = False
        game.player1.board.append(attacker)

        # P2 has TAUNT and non-TAUNT minions
        taunt = MinionCard(name="Taunt", mana_cost=2, attack=1, health=4)
        taunt.mechanics.append("TAUNT")
        normal = MinionCard(name="Normal", mana_cost=1, attack=1, health=1)
        game.player2.board.append(taunt)
        game.player2.board.append(normal)

        actions = ActionSpace.get_legal_actions(game)
        attack_actions = [a for a in actions if a.type == ActionType.ATTACK]

        # Should only be able to attack TAUNT minion (index 0)
        assert len(attack_actions) == 1
        assert attack_actions[0].defender_index == 0
        assert attack_actions[0].attacker_index == 0

    def test_must_attack_taunt_minions_only(self):
        """When TAUNT is present, all attacks must target TAUNT minions."""
        game = Game()

        # P1 has an attacker
        attacker = MinionCard(name="Attacker", mana_cost=1, attack=2, health=2)
        attacker.summoning_sick = False
        game.player1.board.append(attacker)

        # P2 has only TAUNT
        taunt = MinionCard(name="Taunt", mana_cost=2, attack=1, health=4)
        taunt.mechanics.append("TAUNT")
        game.player2.board.append(taunt)

        actions = ActionSpace.get_legal_actions(game)
        attack_actions = [a for a in actions if a.type == ActionType.ATTACK]

        # Should have exactly 1 attack action: attack the TAUNT
        assert len(attack_actions) == 1
        assert attack_actions[0].defender_index == 0


class TestTauntWithMultipleMinions:
    """Test TAUNT with multiple minions on board."""

    def test_can_attack_any_taunt_when_multiple_taunts(self):
        """Can attack any TAUNT minion when multiple are present."""
        game = Game()

        # P1 has an attacker
        attacker = MinionCard(name="Attacker", mana_cost=1, attack=2, health=2)
        attacker.summoning_sick = False
        game.player1.board.append(attacker)

        # P2 has two TAUNT minions
        taunt1 = MinionCard(name="Taunt1", mana_cost=2, attack=1, health=4)
        taunt1.mechanics.append("TAUNT")
        taunt2 = MinionCard(name="Taunt2", mana_cost=3, attack=2, health=5)
        taunt2.mechanics.append("TAUNT")
        game.player2.board.append(taunt1)
        game.player2.board.append(taunt2)

        actions = ActionSpace.get_legal_actions(game)
        attack_actions = [a for a in actions if a.type == ActionType.ATTACK]

        # Should be able to attack either TAUNT (indices 0 or 1)
        assert len(attack_actions) == 2
        defender_indices = [a.defender_index for a in attack_actions]
        assert 0 in defender_indices
        assert 1 in defender_indices

    def test_multiple_attackers_all_respect_taunt(self):
        """All friendly minions must respect TAUNT."""
        game = Game()

        # P1 has two attackers
        attacker1 = MinionCard(name="A1", mana_cost=1, attack=2, health=2)
        attacker1.summoning_sick = False
        attacker2 = MinionCard(name="A2", mana_cost=1, attack=1, health=1)
        attacker2.summoning_sick = False
        game.player1.board.append(attacker1)
        game.player1.board.append(attacker2)

        # P2 has TAUNT and normal minion
        taunt = MinionCard(name="Taunt", mana_cost=2, attack=1, health=4)
        taunt.mechanics.append("TAUNT")
        normal = MinionCard(name="Normal", mana_cost=1, attack=1, health=1)
        game.player2.board.append(taunt)
        game.player2.board.append(normal)

        actions = ActionSpace.get_legal_actions(game)
        attack_actions = [a for a in actions if a.type == ActionType.ATTACK]

        # Both attackers can only attack TAUNT (index 0)
        # Should have 2 attack actions: A1→Taunt, A2→Taunt
        assert len(attack_actions) == 2
        for action in attack_actions:
            assert action.defender_index == 0, "All attacks must target TAUNT"


class TestTauntWithNoTaunt:
    """Test normal attack rules when no TAUNT is present."""

    def test_can_attack_face_when_no_taunt(self):
        """Can attack face when opponent has no TAUNT minions."""
        game = Game()

        # P1 has an attacker
        attacker = MinionCard(name="Attacker", mana_cost=1, attack=2, health=2)
        attacker.summoning_sick = False
        game.player1.board.append(attacker)

        # P2 has normal minions (no TAUNT)
        normal = MinionCard(name="Normal", mana_cost=1, attack=1, health=1)
        game.player2.board.append(normal)

        actions = ActionSpace.get_legal_actions(game)
        attack_actions = [a for a in actions if a.type == ActionType.ATTACK]

        # Should be able to attack minion OR face
        assert len(attack_actions) == 2
        defender_indices = [a.defender_index for a in attack_actions]
        assert 0 in defender_indices  # Attack minion
        assert None in defender_indices  # Attack face

    def test_can_attack_any_minion_when_no_taunt(self):
        """Can attack any minion when no TAUNT is present."""
        game = Game()

        # P1 has an attacker
        attacker = MinionCard(name="Attacker", mana_cost=1, attack=2, health=2)
        attacker.summoning_sick = False
        game.player1.board.append(attacker)

        # P2 has multiple normal minions (no TAUNT)
        minion1 = MinionCard(name="M1", mana_cost=1, attack=1, health=1)
        minion2 = MinionCard(name="M2", mana_cost=2, attack=2, health=2)
        game.player2.board.append(minion1)
        game.player2.board.append(minion2)

        actions = ActionSpace.get_legal_actions(game)
        attack_actions = [a for a in actions if a.type == ActionType.ATTACK]

        # Can attack M1, M2, or face (3 options)
        assert len(attack_actions) == 3
        defender_indices = [a.defender_index for a in attack_actions]
        assert 0 in defender_indices
        assert 1 in defender_indices
        assert None in defender_indices


class TestTauntEdgeCases:
    """Test edge cases for TAUNT mechanic."""

    def test_can_attack_face_when_opponent_has_no_minions(self):
        """Can attack face when opponent board is empty."""
        game = Game()

        # P1 has an attacker
        attacker = MinionCard(name="Attacker", mana_cost=1, attack=2, health=2)
        attacker.summoning_sick = False
        game.player1.board.append(attacker)

        # P2 has no minions

        actions = ActionSpace.get_legal_actions(game)
        attack_actions = [a for a in actions if a.type == ActionType.ATTACK]

        # Should only be able to attack face
        assert len(attack_actions) == 1
        assert attack_actions[0].defender_index is None

    def test_all_taunt_board(self):
        """When all enemy minions have TAUNT, can attack any of them."""
        game = Game()

        # P1 has an attacker
        attacker = MinionCard(name="Attacker", mana_cost=1, attack=2, health=2)
        attacker.summoning_sick = False
        game.player1.board.append(attacker)

        # P2 has only TAUNT minions
        taunt1 = MinionCard(name="T1", mana_cost=1, attack=1, health=1)
        taunt1.mechanics.append("TAUNT")
        taunt2 = MinionCard(name="T2", mana_cost=2, attack=2, health=2)
        taunt2.mechanics.append("TAUNT")
        taunt3 = MinionCard(name="T3", mana_cost=3, attack=3, health=3)
        taunt3.mechanics.append("TAUNT")
        game.player2.board.append(taunt1)
        game.player2.board.append(taunt2)
        game.player2.board.append(taunt3)

        actions = ActionSpace.get_legal_actions(game)
        attack_actions = [a for a in actions if a.type == ActionType.ATTACK]

        # Can attack any of the 3 TAUNT minions (no face attack)
        assert len(attack_actions) == 3
        defender_indices = [a.defender_index for a in attack_actions]
        assert 0 in defender_indices
        assert 1 in defender_indices
        assert 2 in defender_indices
        assert None not in defender_indices  # Cannot attack face

    def test_no_attacks_when_no_friendly_minions(self):
        """No attack actions when we have no minions."""
        game = Game()

        # P1 has no minions

        # P2 has TAUNT
        taunt = MinionCard(name="Taunt", mana_cost=2, attack=1, health=4)
        taunt.mechanics.append("TAUNT")
        game.player2.board.append(taunt)

        actions = ActionSpace.get_legal_actions(game)
        attack_actions = [a for a in actions if a.type == ActionType.ATTACK]

        # No attack actions possible
        assert len(attack_actions) == 0
