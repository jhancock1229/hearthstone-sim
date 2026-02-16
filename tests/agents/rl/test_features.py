"""Unit tests for feature extraction - Observation → numerical vector.

Feature extraction converts rich Observation objects into fixed-size
numerical vectors suitable for neural network input.

Features include:
- Scalar features: health, mana, deck/hand sizes
- Board features: minion stats for 7 positions (self + opponent)
- Hand features: histogram by mana cost
- Misc: turn number, first player flag

All features normalized to [0, 1] range.

Tests validate:
- Correct output size (fixed dimension)
- Proper normalization
- Board encoding
- Hand encoding
- Edge cases (empty boards, full boards)
"""

import pytest
import numpy as np
from agents.rl.features import FeatureExtractor, extract_features
from simulation.observation import Observation
from hearthstone.cards.base import MinionCard


class TestFeatureExtractorBasics:
    """Test basic feature extraction functionality."""

    def test_feature_extractor_creation(self):
        """Can create a FeatureExtractor."""
        extractor = FeatureExtractor()
        assert extractor is not None

    def test_feature_size_is_fixed(self):
        """Feature extractor has fixed output size."""
        extractor = FeatureExtractor()
        assert extractor.feature_size > 0
        assert isinstance(extractor.feature_size, int)

    def test_extract_features_returns_numpy_array(self):
        """extract_features returns numpy array."""
        obs = Observation(
            self_hand=[],
            self_board=[],
            opponent_board=[],
            self_mana=0,
            self_max_mana=0,
            self_health=30,
            self_fatigue_counter=0,
            opponent_health=30,
            legal_actions=[]
        )

        features = extract_features(obs)
        assert isinstance(features, np.ndarray)

    def test_feature_vector_has_correct_size(self):
        """Feature vector has expected fixed size."""
        obs = Observation(
            self_hand=[],
            self_board=[],
            opponent_board=[],
            self_mana=0,
            self_max_mana=0,
            self_health=30,
            self_fatigue_counter=0,
            opponent_health=30,
            legal_actions=[]
        )

        extractor = FeatureExtractor()
        features = extract_features(obs)

        assert len(features) == extractor.feature_size

    def test_feature_vector_is_1d(self):
        """Feature vector is 1-dimensional."""
        obs = Observation(
            self_hand=[],
            self_board=[],
            opponent_board=[],
            self_mana=0,
            self_max_mana=0,
            self_health=30,
            self_fatigue_counter=0,
            opponent_health=30,
            legal_actions=[]
        )

        features = extract_features(obs)
        assert features.ndim == 1


class TestScalarFeatures:
    """Test extraction of scalar features."""

    def test_health_features_normalized(self):
        """Health features are normalized to [0, 1]."""
        obs = Observation(
            self_hand=[],
            self_board=[],
            opponent_board=[],
            self_mana=0,
            self_max_mana=0,
            self_health=15,  # Half health
            self_fatigue_counter=0,
            opponent_health=30,  # Full health
            legal_actions=[]
        )

        features = extract_features(obs)

        # Health features should be in [0, 1]
        # All features should be normalized
        assert np.all(features >= 0.0)
        assert np.all(features <= 1.0)

    def test_mana_features_normalized(self):
        """Mana features are normalized."""
        obs = Observation(
            self_hand=[],
            self_board=[],
            opponent_board=[],
            self_mana=5,
            self_max_mana=10,
            self_health=30,
            self_fatigue_counter=0,
            opponent_health=30,
            legal_actions=[]
        )

        features = extract_features(obs)

        # All features in valid range
        assert np.all(features >= 0.0)
        assert np.all(features <= 1.0)

    def test_fatigue_counter_encoded(self):
        """Fatigue counter is included in features."""
        obs = Observation(
            self_hand=[],
            self_board=[],
            opponent_board=[],
            self_mana=0,
            self_max_mana=0,
            self_health=20,
            self_fatigue_counter=5,
            opponent_health=30,
            legal_actions=[]
        )

        features = extract_features(obs)

        # Should complete without error
        assert len(features) > 0


class TestBoardFeatures:
    """Test board state encoding."""

    def test_empty_board_encoded(self):
        """Empty board is properly encoded (all zeros for minion features)."""
        obs = Observation(
            self_hand=[],
            self_board=[],  # Empty
            opponent_board=[],  # Empty
            self_mana=0,
            self_max_mana=0,
            self_health=30,
            self_fatigue_counter=0,
            opponent_health=30,
            legal_actions=[]
        )

        features = extract_features(obs)

        # Should work with empty boards
        assert len(features) > 0
        assert np.all(features >= 0.0)
        assert np.all(features <= 1.0)

    def test_single_minion_on_board(self):
        """Single minion is properly encoded."""
        minion = MinionCard(name="Test", mana_cost=2, attack=3, health=2)

        obs = Observation(
            self_hand=[],
            self_board=[minion],
            opponent_board=[],
            self_mana=0,
            self_max_mana=0,
            self_health=30,
            self_fatigue_counter=0,
            opponent_health=30,
            legal_actions=[]
        )

        features = extract_features(obs)

        assert len(features) > 0
        assert np.all(features >= 0.0)
        assert np.all(features <= 1.0)

    def test_full_board_encoded(self):
        """Full board (7 minions) is properly encoded."""
        minions = [
            MinionCard(name=f"M{i}", mana_cost=1, attack=1, health=1)
            for i in range(7)
        ]

        obs = Observation(
            self_hand=[],
            self_board=minions,
            opponent_board=[],
            self_mana=0,
            self_max_mana=0,
            self_health=30,
            self_fatigue_counter=0,
            opponent_health=30,
            legal_actions=[]
        )

        features = extract_features(obs)

        assert len(features) > 0
        assert np.all(features >= 0.0)
        assert np.all(features <= 1.0)

    def test_minion_stats_normalized(self):
        """Minion attack/health are normalized."""
        # Large minion to test normalization
        minion = MinionCard(name="Giant", mana_cost=8, attack=8, health=8)

        obs = Observation(
            self_hand=[],
            self_board=[minion],
            opponent_board=[],
            self_mana=0,
            self_max_mana=0,
            self_health=30,
            self_fatigue_counter=0,
            opponent_health=30,
            legal_actions=[]
        )

        features = extract_features(obs)

        # All features should be normalized
        assert np.all(features >= 0.0)
        assert np.all(features <= 1.0)

    def test_opponent_board_encoded(self):
        """Opponent board is encoded separately."""
        my_minion = MinionCard(name="Mine", mana_cost=2, attack=2, health=2)
        enemy_minion = MinionCard(name="Enemy", mana_cost=3, attack=3, health=3)

        obs = Observation(
            self_hand=[],
            self_board=[my_minion],
            opponent_board=[enemy_minion],
            self_mana=0,
            self_max_mana=0,
            self_health=30,
            self_fatigue_counter=0,
            opponent_health=30,
            legal_actions=[]
        )

        features = extract_features(obs)

        # Should encode both boards
        assert len(features) > 0
        assert np.all(features >= 0.0)
        assert np.all(features <= 1.0)


class TestHandFeatures:
    """Test hand encoding as histogram."""

    def test_empty_hand_encoded(self):
        """Empty hand is properly encoded."""
        obs = Observation(
            self_hand=[],
            self_board=[],
            opponent_board=[],
            self_mana=0,
            self_max_mana=0,
            self_health=30,
            self_fatigue_counter=0,
            opponent_health=30,
            legal_actions=[]
        )

        features = extract_features(obs)

        assert len(features) > 0

    def test_hand_with_cards_encoded(self):
        """Hand with cards creates histogram."""
        cards = [
            MinionCard(name="C1", mana_cost=1, attack=1, health=1),
            MinionCard(name="C2", mana_cost=2, attack=2, health=2),
            MinionCard(name="C3", mana_cost=2, attack=2, health=3),
        ]

        obs = Observation(
            self_hand=cards,
            self_board=[],
            opponent_board=[],
            self_mana=5,
            self_max_mana=5,
            self_health=30,
            self_fatigue_counter=0,
            opponent_health=30,
            legal_actions=[]
        )

        features = extract_features(obs)

        # Should encode hand as histogram
        assert len(features) > 0
        assert np.all(features >= 0.0)
        assert np.all(features <= 1.0)


class TestMiscFeatures:
    """Test miscellaneous features (turn number, etc.)."""

    def test_turn_number_encoded(self):
        """Turn number is encoded and normalized."""
        obs = Observation(
            self_hand=[],
            self_board=[],
            opponent_board=[],
            self_mana=0,
            self_max_mana=0,
            self_health=30,
            self_fatigue_counter=0,
            opponent_health=30,
            turn_number=10,
            legal_actions=[]
        )

        features = extract_features(obs)

        assert len(features) > 0
        assert np.all(features >= 0.0)
        assert np.all(features <= 1.0)

    def test_player_index_encoded(self):
        """Player index (first/second player) is encoded."""
        obs = Observation(
            self_hand=[],
            self_board=[],
            opponent_board=[],
            self_mana=0,
            self_max_mana=0,
            self_health=30,
            self_fatigue_counter=0,
            opponent_health=30,
            player_index=0,  # First player
            legal_actions=[]
        )

        features = extract_features(obs)

        assert len(features) > 0


class TestFeatureConsistency:
    """Test feature extraction consistency."""

    def test_same_observation_produces_same_features(self):
        """Same observation always produces same features."""
        obs = Observation(
            self_hand=[],
            self_board=[],
            opponent_board=[],
            self_mana=5,
            self_max_mana=10,
            self_health=25,
            self_fatigue_counter=0,
            opponent_health=20,
            legal_actions=[]
        )

        features1 = extract_features(obs)
        features2 = extract_features(obs)

        np.testing.assert_array_equal(features1, features2)

    def test_different_observations_produce_different_features(self):
        """Different observations produce different features."""
        obs1 = Observation(
            self_hand=[],
            self_board=[],
            opponent_board=[],
            self_mana=5,
            self_max_mana=10,
            self_health=30,
            self_fatigue_counter=0,
            opponent_health=30,
            legal_actions=[]
        )

        obs2 = Observation(
            self_hand=[],
            self_board=[],
            opponent_board=[],
            self_mana=5,
            self_max_mana=10,
            self_health=15,  # Different health
            self_fatigue_counter=0,
            opponent_health=30,
            legal_actions=[]
        )

        features1 = extract_features(obs1)
        features2 = extract_features(obs2)

        # Should be different (at least one element)
        assert not np.array_equal(features1, features2)


class TestFeatureNormalization:
    """Test all features are properly normalized."""

    def test_all_features_in_valid_range(self):
        """All features are in [0, 1] range."""
        # Create complex observation with many features
        minions = [
            MinionCard(name=f"M{i}", mana_cost=i % 8, attack=i % 10, health=i % 10)
            for i in range(5)
        ]

        cards = [
            MinionCard(name=f"C{i}", mana_cost=i, attack=i, health=i)
            for i in range(8)
        ]

        obs = Observation(
            self_hand=cards,
            self_board=minions[:3],
            opponent_board=minions[3:],
            self_mana=7,
            self_max_mana=10,
            self_health=22,
            self_fatigue_counter=3,
            opponent_health=18,
            turn_number=15,
            player_index=1,
            legal_actions=[]
        )

        features = extract_features(obs)

        # All features must be in [0, 1]
        assert np.all(features >= 0.0), f"Some features < 0: {features[features < 0]}"
        assert np.all(features <= 1.0), f"Some features > 1: {features[features > 1]}"

    def test_extreme_values_normalized(self):
        """Extreme values are clamped to [0, 1]."""
        # Edge case: damaged minion (health = 0 means dead, but test boundary)
        obs = Observation(
            self_hand=[],
            self_board=[],
            opponent_board=[],
            self_mana=10,
            self_max_mana=10,
            self_health=1,  # Almost dead
            self_fatigue_counter=15,  # High fatigue
            opponent_health=30,
            turn_number=50,  # Long game
            legal_actions=[]
        )

        features = extract_features(obs)

        # Should handle extremes gracefully
        assert np.all(features >= 0.0)
        assert np.all(features <= 1.0)
