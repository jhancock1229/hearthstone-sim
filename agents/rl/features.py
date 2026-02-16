"""Feature extraction for RL agents.

Converts rich Observation objects into fixed-size numerical vectors suitable
for neural network input.

Feature extraction pipeline:
1. Extract scalar features (health, mana, deck/hand sizes)
2. Encode board state (minion stats for 7 positions × 2 boards)
3. Encode hand as histogram by mana cost
4. Add misc features (turn number, player index)
5. Normalize all features to [0, 1] range

Total features: ~106
- 10 scalar features
- 84 board features (7 slots × 6 features × 2 boards)
- 10 hand features (mana cost histogram 0-9)
- 2 misc features

All features are normalized to [0, 1] for stable neural network training.
"""

import numpy as np
from typing import List
from simulation.observation import Observation


# Normalization constants
MAX_HEALTH = 30
MAX_MANA = 10
MAX_DECK_SIZE = 30
MAX_HAND_SIZE = 10
MAX_FATIGUE = 20
MAX_TURN = 50
MAX_MINION_ATTACK = 20
MAX_MINION_HEALTH = 20
BOARD_SIZE = 7
MANA_BINS = 10  # 0-9 mana cost


class FeatureExtractor:
    """Extracts fixed-size feature vectors from Observations.

    Features are organized as:
    1. Scalar features (10): health, mana, deck/hand sizes, fatigue
    2. Self board (42): 7 slots × 6 features (present, attack, health, taunt, divine_shield, poisonous)
    3. Opponent board (42): same as self board
    4. Hand histogram (10): count of cards by mana cost 0-9
    5. Misc (2): turn number, player index

    Total: 106 features
    """

    def __init__(self):
        """Initialize the feature extractor."""
        # Calculate feature size
        self.num_scalar_features = 10
        self.num_board_features_per_slot = 6  # present, attack, health, taunt, divine_shield, poisonous
        self.num_board_features = BOARD_SIZE * self.num_board_features_per_slot * 2  # × 2 for both boards
        self.num_hand_features = MANA_BINS
        self.num_misc_features = 2

        self.feature_size = (
            self.num_scalar_features +
            self.num_board_features +
            self.num_hand_features +
            self.num_misc_features
        )

    def extract(self, observation: Observation) -> np.ndarray:
        """Extract feature vector from observation.

        Args:
            observation: Game state observation

        Returns:
            1D numpy array of normalized features
        """
        features = []

        # 1. Scalar features (10)
        features.extend(self._extract_scalar_features(observation))

        # 2. Self board features (42)
        features.extend(self._extract_board_features(observation.self_board))

        # 3. Opponent board features (42)
        features.extend(self._extract_board_features(observation.opponent_board))

        # 4. Hand features (10)
        features.extend(self._extract_hand_features(observation.self_hand))

        # 5. Misc features (2)
        features.extend(self._extract_misc_features(observation))

        return np.array(features, dtype=np.float32)

    def _extract_scalar_features(self, obs: Observation) -> List[float]:
        """Extract scalar features: health, mana, deck/hand sizes, fatigue.

        Args:
            obs: Observation

        Returns:
            List of 10 normalized scalar features
        """
        return [
            obs.self_health / MAX_HEALTH,
            obs.opponent_health / MAX_HEALTH,
            obs.self_mana / MAX_MANA,
            obs.self_max_mana / MAX_MANA,
            len(obs.self_deck) / MAX_DECK_SIZE if hasattr(obs, 'self_deck') else 0.0,
            len(obs.self_hand) / MAX_HAND_SIZE,
            len(obs.opponent_deck) / MAX_DECK_SIZE if hasattr(obs, 'opponent_deck') else 0.0,
            len(obs.self_hand) / MAX_HAND_SIZE,  # Note: opponent hand size not visible, use self as proxy
            obs.self_fatigue_counter / MAX_FATIGUE,
            0.0,  # opponent fatigue not visible in observation
        ]

    def _extract_board_features(self, board: List) -> List[float]:
        """Extract board features for 7 minion slots.

        For each slot (7 total):
        - present (0/1)
        - attack (normalized)
        - health (normalized)
        - taunt (0/1)
        - divine_shield (0/1)
        - poisonous (0/1)

        Args:
            board: List of minions on board

        Returns:
            List of 42 features (7 slots × 6 features)
        """
        features = []

        for i in range(BOARD_SIZE):
            if i < len(board):
                minion = board[i]
                features.append(1.0)  # present
                features.append(min(minion.attack / MAX_MINION_ATTACK, 1.0))
                features.append(min(minion.health / MAX_MINION_HEALTH, 1.0))

                # Mechanics
                mechanics = getattr(minion, 'mechanics', [])
                features.append(1.0 if 'TAUNT' in mechanics else 0.0)
                features.append(1.0 if 'DIVINE_SHIELD' in mechanics else 0.0)
                features.append(1.0 if 'POISONOUS' in mechanics else 0.0)
            else:
                # Empty slot
                features.extend([0.0] * self.num_board_features_per_slot)

        return features

    def _extract_hand_features(self, hand: List) -> List[float]:
        """Extract hand features as histogram by mana cost.

        Creates a histogram of card counts by mana cost (0-9).
        Cards with mana cost > 9 are clamped to bin 9.

        Args:
            hand: List of cards in hand

        Returns:
            List of 10 features (mana cost histogram)
        """
        histogram = [0] * MANA_BINS

        for card in hand:
            mana_cost = min(card.mana_cost, MANA_BINS - 1)  # Clamp to 0-9
            histogram[mana_cost] += 1

        # Normalize by max hand size
        return [count / MAX_HAND_SIZE for count in histogram]

    def _extract_misc_features(self, obs: Observation) -> List[float]:
        """Extract miscellaneous features: turn number, player index.

        Args:
            obs: Observation

        Returns:
            List of 2 features
        """
        turn_number = getattr(obs, 'turn_number', 1)
        player_index = getattr(obs, 'player_index', 0)

        return [
            min(turn_number / MAX_TURN, 1.0),
            float(player_index),  # 0 or 1
        ]


# Global extractor instance
_extractor = FeatureExtractor()


def extract_features(observation: Observation) -> np.ndarray:
    """Extract feature vector from observation.

    Convenience function that uses a global FeatureExtractor instance.

    Args:
        observation: Game state observation

    Returns:
        1D numpy array of normalized features (size 106)
    """
    return _extractor.extract(observation)
