"""
Tests for deathrattle registry and a sample deathrattle that summons a token.
"""

from hearthstone.engine.player import Player
from hearthstone.cards.base import MinionCard
from hearthstone.cards import effects


def test_deathrattle_summons_token_on_death():
    p = Player()
    # Define a sample deathrattle that summons a 1/1 token called 'Whelp'
    @effects.register_deathrattle("Exploder")
    def exploder_deathrattle(player, minion):
        token = MinionCard(name="Whelp", mana_cost=1, attack=1, health=1)
        # place at end
        player.place_minion(token)

    try:
        dead_minion = MinionCard(name="Exploder", mana_cost=2, attack=2, health=0)
        p.board.append(dead_minion)
        # process_deaths should trigger the registry and summon Whelp
        p.process_deaths()
        # There should be a Whelp on the board
        assert any(m.name == "Whelp" for m in p.board)
    finally:
        effects.unregister_deathrattle("Exploder")
        effects.clear_registries()