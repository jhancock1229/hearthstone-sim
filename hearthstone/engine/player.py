"""Player state for Hearthstone simulator."""



class Player:
    def process_deaths(self):
        from hearthstone.engine.combat import process_deaths
        process_deaths(self)


    def attack_hero(self, attacker_idx, opponent):
        from hearthstone.engine.combat import resolve_minion_attack_hero
        return resolve_minion_attack_hero(self, attacker_idx, opponent)

    def attack_minion(self, attacker_idx, opponent, defender_idx):
        from hearthstone.engine.combat import resolve_minion_attack
        return resolve_minion_attack(self, attacker_idx, opponent, defender_idx)

    def __init__(self):
        self.health: int = 30
        self.mana: int = 0
        self.max_mana: int = 0
        self.fatigue_counter: int = 0
        self.deck: list = []
        self.hand: list = []
        self.board: list = []  # List of minions on board
        self.hero_class: str = "NEUTRAL"
        self.armor: int = 0
        self.hero_power_used: bool = False
    BOARD_LIMIT = 7

    def place_minion(self, minion, position=None):
        """
        Place a minion on the board at the given position (default: rightmost).
        Raises Exception if the board is full.
        Returns the position where the minion was placed.
        """
        if len(self.board) >= self.BOARD_LIMIT:
            raise Exception("Board is full (7 minions max)")
        if position is None:
            self.board.append(minion)
            pos = len(self.board) - 1
        else:
            if not (0 <= position <= len(self.board)):
                raise ValueError(f"Invalid board position: {position}")
            self.board.insert(position, minion)
            pos = position
        # Emit summon/play events
        try:
            from hearthstone.engine import events
            events.bus.emit("on_play", source=self, minion=minion, position=pos)
            events.bus.emit("on_play_from_hand", source=self, minion=minion, position=pos)
            events.bus.emit("on_summon", source=self, minion=minion, position=pos)
            events.bus.emit("on_minion_summoned", source=self, minion=minion, position=pos)
        except Exception:
            pass
        return pos

    MAX_MANA = 10

    @property
    def is_dead(self) -> bool:
        """Return True if the player's health is <= 0."""
        return self.health <= 0

    def gain_mana_crystal(self) -> None:
        """Add one mana crystal, up to the maximum of 10."""
        if self.max_mana < self.MAX_MANA:
            self.max_mana += 1

    def refill_mana(self) -> None:
        """Restore current mana to max. Called at the start of each turn."""
        self.mana = self.max_mana

    def spend_mana(self, amount: int) -> None:
        """Spend mana to play a card or use hero power.

        Raises:
            ValueError: If the player doesn't have enough mana.
        """
        if amount > self.mana:
            raise ValueError(f"Not enough mana: have {self.mana}, need {amount}")
        self.mana -= amount

    def draw_card(self) -> None:
        """Draw the top card from the deck into the hand.

        If the hand is full (10 cards), the drawn card is burned instead.
        If the deck is empty, take escalating fatigue damage instead
        (1, then 2, then 3, ...).
        """
        if self.deck:
            card = self.deck.pop()
            # Only add to hand if it has less than 10 cards
            if len(self.hand) < 10:
                self.hand.append(card)
            # else: card is burned (discarded without effect)
        else:
            self.fatigue_counter += 1
            self.health -= self.fatigue_counter

    def take_damage(self, amount: int) -> None:
        """Take damage, absorbing with armor first.

        Args:
            amount: Amount of damage to take
        """
        if amount <= 0:
            return
        if self.armor >= amount:
            self.armor -= amount
        else:
            remainder = amount - self.armor
            self.armor = 0
            self.health -= remainder
