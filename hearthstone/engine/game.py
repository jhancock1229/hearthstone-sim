"""Game orchestration for Hearthstone simulator."""

from hearthstone.engine.player import Player
from hearthstone.engine import events


class Game:
    """Represents a Hearthstone game between two players."""

    def __init__(self, player1_class: str = "MAGE", player2_class: str = "MAGE"):
        self.player1 = Player()
        self.player2 = Player()
        self.player1.hero_class = player1_class
        self.player2.hero_class = player2_class
        self._active_player = self.player1
        self._turn_number = 0

    @property
    def active_player(self) -> Player:
        """Return the player whose turn it is."""
        return self._active_player

    @property
    def turn_number(self) -> int:
        """Return the current turn number (0-indexed)."""
        return self._turn_number

    def start_turn(self) -> None:
        """Start the turn.

        Both players gain a mana crystal (up to max 10).
        The active player refills mana, draws a card, and the turn number increments.
        """
        self.player1.gain_mana_crystal()
        self.player2.gain_mana_crystal()
        self.active_player.refill_mana()
        self.active_player.draw_card()
        self.active_player.hero_power_used = False
        self.active_player.hero_attacked = False
        self._turn_number += 1

        # Clear exhausted status and summoning sickness for active player's minions
        for minion in self.active_player.board:
            minion.exhausted = False
            minion.summoning_sick = False

        # Emit turn start event
        try:
            events.bus.emit("on_turn_start", source=self.active_player, turn=self._turn_number)
        except Exception:
            pass

    def end_turn(self) -> None:
        """End the current player's turn and switch to the other player."""
        # Emit turn end for the current active player
        try:
            events.bus.emit("on_turn_end", source=self.active_player, turn=self._turn_number)
        except Exception:
            pass
        if self._active_player is self.player1:
            self._active_player = self.player2
        else:
            self._active_player = self.player1

    @property
    def is_over(self) -> bool:
        """Return True if the game is over (one player is dead)."""
        return self.player1.is_dead or self.player2.is_dead

    @property
    def winner(self) -> Player | None:
        """Return the winning player if the game is over, None otherwise."""
        if not self.is_over:
            return None
        if self.player1.is_dead:
            return self.player2
        if self.player2.is_dead:
            return self.player1
        return None
