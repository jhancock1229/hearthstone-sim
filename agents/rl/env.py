"""Gymnasium-compatible Hearthstone RL environment.

Wraps the Hearthstone game engine as a Gymnasium environment for
training RL agents. Handles:
- Game lifecycle (reset, step, done)
- Observation: game state → feature vector (106 floats)
- Actions: integer → game Action mapping
- Rewards: sparse (+1 win, -1 loss, 0 mid-game)
- Action masking: boolean mask over 100 discrete actions

Action space encoding (100 total):
- 0-69:  PLAY_CARD (10 hand slots × 7 board positions)
- 70-98: ATTACK (7 attacker slots × (4 defender slots + face))
- 99:    END_TURN

The agent always plays as player 1. The opponent is controlled by
a provided agent (defaults to random).
"""

import random
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from agents.rl.features import FeatureExtractor
from hearthstone.engine.game import Game
from hearthstone.engine import actions as game_actions
from simulation.action_space import Action, ActionSpace, ActionType
from simulation.observation import Observation
from deckbuilding.deck import random_deck, build_concrete_deck


# Action space layout (100 total):
# 0-69:  PLAY_CARD (10 hand slots × 7 board positions)
# 70-97: ATTACK (7 attackers × 4 targets: face + minion 0-2)
# 98:    HERO_POWER
# 99:    END_TURN
PLAY_CARD_SLOTS = 10
BOARD_POSITIONS = 7
NUM_ATTACKER_SLOTS = 7

ACTION_SPACE_SIZE = 100

_PLAY_CARD_START = 0
_PLAY_CARD_END = 70      # exclusive
_ATTACK_START = 70
_ATTACK_END = 98          # exclusive (7 attackers × 4 targets = 28)
_HERO_POWER_INDEX = 98
_END_TURN_INDEX = 99

_ATTACK_TARGETS_PER_ATTACKER = 4  # 0=face, 1=minion0, 2=minion1, 3=minion2


class HearthstoneEnv(gym.Env):
    """Gymnasium environment for Hearthstone.

    The RL agent plays as player 1. An opponent agent controls player 2.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        seed: Optional[int] = None,
        max_turns: int = 200,
        opponent_agent: Optional[Any] = None,
    ):
        super().__init__()

        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)
        self.max_turns = max_turns
        self.opponent_agent = opponent_agent

        self.feature_extractor = FeatureExtractor()

        # Gymnasium spaces
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.feature_extractor.feature_size,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(ACTION_SPACE_SIZE)

        # Game state
        self.game: Optional[Game] = None
        self._turn_actions_taken = 0

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment for new episode.

        Returns:
            Tuple of (observation, info_dict)
        """
        if seed is not None:
            self.rng = random.Random(seed)
            self.np_rng = np.random.RandomState(seed)

        # Create new game
        self.game = Game()

        # Initialize decks
        deck1 = random_deck(size=30, rng=self.rng)
        deck2 = random_deck(size=30, rng=self.rng)
        self.game.player1.deck = build_concrete_deck(deck1, rng=self.rng)
        self.game.player2.deck = build_concrete_deck(deck2, rng=self.rng)

        # Draw starting hands
        for _ in range(3):
            if self.game.player1.deck:
                self.game.player1.hand.append(self.game.player1.deck.pop(0))
        for _ in range(4):
            if self.game.player2.deck:
                self.game.player2.hand.append(self.game.player2.deck.pop(0))

        # Start the first turn
        self.game.start_turn()
        self._turn_actions_taken = 0

        # If it's player 2's turn first, play opponent's turn
        self._play_opponent_turns()

        obs = self._get_obs()
        info = self._get_info()

        return obs, info

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Take an action in the environment.

        Args:
            action: Integer action index (0 to ACTION_SPACE_SIZE-1)

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        reward = 0.0
        terminated = False
        truncated = False

        # Decode and execute action
        game_action = self._decode_action(action)

        if game_action.type == ActionType.END_TURN:
            self.game.end_turn()
            self._turn_actions_taken = 0

            # Check game over after ending turn
            if self.game.is_over:
                terminated = True
                reward = self._compute_reward()
            elif self.game.turn_number >= self.max_turns:
                truncated = True
                reward = self._compute_reward()
            else:
                # Opponent's turn(s)
                self.game.start_turn()
                self._play_opponent_turns()

                # After opponent, start our new turn
                if not self.game.is_over:
                    self.game.end_turn()
                    if self.game.is_over:
                        terminated = True
                        reward = self._compute_reward()
                    elif self.game.turn_number >= self.max_turns:
                        truncated = True
                        reward = self._compute_reward()
                    else:
                        self.game.start_turn()
                        self._turn_actions_taken = 0
                else:
                    terminated = True
                    reward = self._compute_reward()
        else:
            # Execute non-END_TURN action
            self._execute_action(game_action)
            self._turn_actions_taken += 1

            if self.game.is_over:
                terminated = True
                reward = self._compute_reward()

        obs = self._get_obs()
        info = self._get_info()

        return obs, reward, terminated, truncated, info

    def _play_opponent_turns(self):
        """Play the opponent's turn using the opponent agent."""
        if self.game.is_over:
            return

        # Check if it's player 2's turn
        if self.game.active_player != self.game.player2:
            return

        # Simple random opponent if no agent provided
        max_actions = 100
        actions_taken = 0

        while not self.game.is_over and actions_taken < max_actions:
            actions_taken += 1

            legal_actions = ActionSpace.get_legal_actions(self.game)

            if self.opponent_agent is not None:
                obs = self._make_observation(player_index=1)
                obs.legal_actions = legal_actions
                chosen = self.opponent_agent.choose_action(obs)
            else:
                # Random opponent
                chosen = self.rng.choice(legal_actions)

            if chosen.type == ActionType.END_TURN:
                break
            else:
                self._execute_game_action(chosen)

    def _execute_action(self, game_action: Action):
        """Execute a game action for player 1."""
        player = self.game.player1
        opponent = self.game.player2

        try:
            if game_action.type == ActionType.PLAY_CARD:
                game_actions.play_card(
                    player,
                    card_index=game_action.card_index,
                    position=game_action.position,
                )
            elif game_action.type == ActionType.ATTACK:
                game_actions.attack(
                    attacker_player=player,
                    attacker_index=game_action.attacker_index,
                    defender_player=opponent,
                    defender_index=game_action.defender_index,
                )
            elif game_action.type == ActionType.HERO_POWER:
                game_actions.use_hero_power(player)
        except Exception:
            pass  # Invalid action, silently skip

    def _execute_game_action(self, game_action: Action):
        """Execute a game action for the active player."""
        player = self.game.active_player
        opponent = self.game.player2 if player == self.game.player1 else self.game.player1

        try:
            if game_action.type == ActionType.PLAY_CARD:
                game_actions.play_card(
                    player,
                    card_index=game_action.card_index,
                    position=game_action.position,
                )
            elif game_action.type == ActionType.ATTACK:
                game_actions.attack(
                    attacker_player=player,
                    attacker_index=game_action.attacker_index,
                    defender_player=opponent,
                    defender_index=game_action.defender_index,
                )
            elif game_action.type == ActionType.HERO_POWER:
                game_actions.use_hero_power(player)
        except Exception:
            pass

    def _decode_action(self, action_id: int) -> Action:
        """Convert integer action to game Action.

        Args:
            action_id: Integer action index

        Returns:
            Game Action object
        """
        if action_id == _END_TURN_INDEX:
            return Action(type=ActionType.END_TURN)

        if action_id == _HERO_POWER_INDEX:
            return Action(type=ActionType.HERO_POWER)

        if _PLAY_CARD_START <= action_id < _PLAY_CARD_END:
            idx = action_id - _PLAY_CARD_START
            hand_index = idx // BOARD_POSITIONS
            position = idx % BOARD_POSITIONS
            return Action(
                type=ActionType.PLAY_CARD,
                card_index=hand_index,
                position=position,
            )

        if _ATTACK_START <= action_id < _ATTACK_END:
            idx = action_id - _ATTACK_START
            attacker_index = idx // _ATTACK_TARGETS_PER_ATTACKER
            target_code = idx % _ATTACK_TARGETS_PER_ATTACKER
            # target_code 0 = face (defender_index=None)
            # target_code 1-3 = minion index 0-2
            defender_index = None if target_code == 0 else (target_code - 1)
            return Action(
                type=ActionType.ATTACK,
                attacker_index=attacker_index,
                defender_index=defender_index,
            )

        # Fallback: END_TURN
        return Action(type=ActionType.END_TURN)

    def _get_action_mask(self) -> np.ndarray:
        """Build boolean action mask from legal game actions.

        Returns:
            Boolean array of shape (ACTION_SPACE_SIZE,)
        """
        mask = np.zeros(ACTION_SPACE_SIZE, dtype=bool)

        if self.game is None or self.game.is_over:
            mask[_END_TURN_INDEX] = True
            return mask

        legal_actions = ActionSpace.get_legal_actions(self.game)

        for action in legal_actions:
            idx = self._encode_action(action)
            if idx is not None and 0 <= idx < ACTION_SPACE_SIZE:
                mask[idx] = True

        # END_TURN always valid
        mask[_END_TURN_INDEX] = True

        return mask

    def _encode_action(self, action: Action) -> Optional[int]:
        """Convert game Action to integer action index.

        Args:
            action: Game Action object

        Returns:
            Integer action index, or None if can't be encoded
        """
        if action.type == ActionType.END_TURN:
            return _END_TURN_INDEX

        if action.type == ActionType.HERO_POWER:
            return _HERO_POWER_INDEX

        if action.type == ActionType.PLAY_CARD:
            hand_index = action.card_index or 0
            position = action.position or 0
            if hand_index < PLAY_CARD_SLOTS and position < BOARD_POSITIONS:
                return _PLAY_CARD_START + hand_index * BOARD_POSITIONS + position
            return None

        if action.type == ActionType.ATTACK:
            attacker = action.attacker_index or 0
            if attacker >= NUM_ATTACKER_SLOTS:
                return None
            if action.defender_index is None:
                target_code = 0  # face
            else:
                target_code = action.defender_index + 1
                if target_code >= _ATTACK_TARGETS_PER_ATTACKER:
                    return None  # Can't encode defender index > 2
            return _ATTACK_START + attacker * _ATTACK_TARGETS_PER_ATTACKER + target_code

        return None

    def _get_obs(self) -> np.ndarray:
        """Get current observation as feature vector."""
        obs = self._make_observation(player_index=0)
        return self.feature_extractor.extract(obs)

    def _make_observation(self, player_index: int) -> Observation:
        """Create Observation for specified player."""
        player = self.game.player1 if player_index == 0 else self.game.player2
        opponent = self.game.player2 if player_index == 0 else self.game.player1

        return Observation(
            self_health=player.health,
            self_mana=player.mana,
            self_max_mana=player.max_mana,
            self_fatigue_counter=player.fatigue_counter,
            self_hand=list(player.hand),
            self_deck_size=len(player.deck),
            self_board=list(player.board),
            opponent_health=opponent.health,
            opponent_mana=opponent.mana,
            opponent_max_mana=opponent.max_mana,
            opponent_fatigue_counter=opponent.fatigue_counter,
            opponent_hand_size=len(opponent.hand),
            opponent_deck_size=len(opponent.deck),
            opponent_board=list(opponent.board),
            turn_number=self.game.turn_number,
            is_my_turn=(self.game.active_player == player),
            is_game_over=self.game.is_over,
            winner=None,
            player_index=player_index,
        )

    def _get_info(self) -> Dict[str, Any]:
        """Get info dict with action mask."""
        return {
            'action_mask': self._get_action_mask(),
        }

    def _compute_reward(self) -> float:
        """Compute reward: +1 win, -1 loss."""
        if not self.game.is_over:
            # Max turns: use health comparison
            if self.game.player1.health > self.game.player2.health:
                return 1.0
            elif self.game.player1.health < self.game.player2.health:
                return -1.0
            else:
                return 0.0  # Draw

        if self.game.winner == self.game.player1:
            return 1.0
        else:
            return -1.0
