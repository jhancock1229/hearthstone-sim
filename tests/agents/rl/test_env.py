"""Unit tests for HearthstoneEnv - Gymnasium-compatible RL environment.

Tests the RL environment wrapper that:
- Provides Gymnasium interface (reset, step)
- Converts game state → feature vectors (observations)
- Maps integer actions → game Actions
- Provides sparse rewards (+1 win, -1 loss)
- Supports action masking
- Handles game lifecycle (reset, play, done)

The environment uses a fixed action space of 100 discrete actions:
- 0-69: PLAY_CARD (10 hand slots × 7 board positions)
- 70-98: ATTACK (7 attacker slots × 4 targets + face)
- 99: END_TURN (special, always last)
- Note: invalid actions are masked; only valid ones can be taken

Observation space: Box(106,) - feature vector from FeatureExtractor
Action space: Discrete(100) with action masking
"""

import pytest
import numpy as np

from agents.rl.env import HearthstoneEnv, ACTION_SPACE_SIZE
from agents.rl.features import FeatureExtractor


class TestEnvCreation:
    """Test environment creation and configuration."""

    def test_env_creation(self):
        """Can create a HearthstoneEnv."""
        env = HearthstoneEnv()
        assert env is not None

    def test_env_has_observation_space(self):
        """Environment has an observation_space attribute."""
        env = HearthstoneEnv()
        assert hasattr(env, 'observation_space')

    def test_env_has_action_space(self):
        """Environment has an action_space attribute."""
        env = HearthstoneEnv()
        assert hasattr(env, 'action_space')

    def test_action_space_is_discrete(self):
        """Action space is Discrete(100)."""
        import gymnasium as gym
        env = HearthstoneEnv()
        assert isinstance(env.action_space, gym.spaces.Discrete)
        assert env.action_space.n == ACTION_SPACE_SIZE

    def test_observation_space_is_box(self):
        """Observation space is a Box with correct size."""
        import gymnasium as gym
        env = HearthstoneEnv()
        assert isinstance(env.observation_space, gym.spaces.Box)
        extractor = FeatureExtractor()
        assert env.observation_space.shape == (extractor.feature_size,)

    def test_env_with_seed(self):
        """Can create environment with a seed."""
        env = HearthstoneEnv(seed=42)
        assert env is not None


class TestEnvReset:
    """Test environment reset."""

    def test_reset_returns_observation_and_info(self):
        """Reset returns (observation, info) tuple."""
        env = HearthstoneEnv(seed=42)
        result = env.reset()
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_reset_observation_is_numpy_array(self):
        """Reset observation is a numpy array."""
        env = HearthstoneEnv(seed=42)
        obs, info = env.reset()
        assert isinstance(obs, np.ndarray)

    def test_reset_observation_has_correct_shape(self):
        """Reset observation has shape matching feature extractor."""
        env = HearthstoneEnv(seed=42)
        obs, info = env.reset()
        extractor = FeatureExtractor()
        assert obs.shape == (extractor.feature_size,)

    def test_reset_info_contains_action_mask(self):
        """Reset info contains action_mask."""
        env = HearthstoneEnv(seed=42)
        obs, info = env.reset()
        assert 'action_mask' in info

    def test_reset_action_mask_is_boolean_array(self):
        """Action mask is boolean array of correct size."""
        env = HearthstoneEnv(seed=42)
        obs, info = env.reset()
        mask = info['action_mask']
        assert isinstance(mask, np.ndarray)
        assert mask.dtype == bool
        assert mask.shape == (ACTION_SPACE_SIZE,)

    def test_reset_action_mask_has_valid_actions(self):
        """Action mask has at least one valid action (END_TURN always valid)."""
        env = HearthstoneEnv(seed=42)
        obs, info = env.reset()
        mask = info['action_mask']
        assert mask.any(), "At least one action must be valid"

    def test_reset_end_turn_always_valid(self):
        """END_TURN action (index 99) is always valid after reset."""
        env = HearthstoneEnv(seed=42)
        obs, info = env.reset()
        mask = info['action_mask']
        # END_TURN is the last action
        assert mask[ACTION_SPACE_SIZE - 1], "END_TURN should always be valid"

    def test_reset_with_seed_is_deterministic(self):
        """Same seed produces same initial observation."""
        env1 = HearthstoneEnv(seed=42)
        obs1, _ = env1.reset()

        env2 = HearthstoneEnv(seed=42)
        obs2, _ = env2.reset()

        np.testing.assert_array_equal(obs1, obs2)


class TestEnvStep:
    """Test environment step."""

    def test_step_returns_five_values(self):
        """Step returns (obs, reward, terminated, truncated, info)."""
        env = HearthstoneEnv(seed=42)
        obs, info = env.reset()

        # Take END_TURN action (always valid)
        result = env.step(ACTION_SPACE_SIZE - 1)  # END_TURN
        assert isinstance(result, tuple)
        assert len(result) == 5

    def test_step_observation_is_numpy_array(self):
        """Step returns numpy array observation."""
        env = HearthstoneEnv(seed=42)
        env.reset()

        obs, reward, terminated, truncated, info = env.step(ACTION_SPACE_SIZE - 1)
        assert isinstance(obs, np.ndarray)

    def test_step_reward_is_float(self):
        """Step reward is a float."""
        env = HearthstoneEnv(seed=42)
        env.reset()

        obs, reward, terminated, truncated, info = env.step(ACTION_SPACE_SIZE - 1)
        assert isinstance(reward, (int, float))

    def test_step_terminated_is_bool(self):
        """Step terminated flag is boolean."""
        env = HearthstoneEnv(seed=42)
        env.reset()

        obs, reward, terminated, truncated, info = env.step(ACTION_SPACE_SIZE - 1)
        assert isinstance(terminated, bool)

    def test_step_truncated_is_bool(self):
        """Step truncated flag is boolean."""
        env = HearthstoneEnv(seed=42)
        env.reset()

        obs, reward, terminated, truncated, info = env.step(ACTION_SPACE_SIZE - 1)
        assert isinstance(truncated, bool)

    def test_step_info_contains_action_mask(self):
        """Step info contains action_mask for next step."""
        env = HearthstoneEnv(seed=42)
        env.reset()

        obs, reward, terminated, truncated, info = env.step(ACTION_SPACE_SIZE - 1)
        if not terminated and not truncated:
            assert 'action_mask' in info

    def test_step_mid_game_reward_is_zero(self):
        """Mid-game reward is 0 (sparse rewards)."""
        env = HearthstoneEnv(seed=42)
        env.reset()

        obs, reward, terminated, truncated, info = env.step(ACTION_SPACE_SIZE - 1)
        if not terminated:
            assert reward == 0.0


class TestEnvGameplay:
    """Test full gameplay episodes."""

    def test_game_eventually_ends(self):
        """Game eventually terminates (fatigue, combat, or max turns)."""
        env = HearthstoneEnv(seed=42, max_turns=100)
        obs, info = env.reset()

        done = False
        steps = 0
        max_steps = 5000  # Safety limit

        while not done and steps < max_steps:
            mask = info['action_mask']
            # Pick random valid action
            valid_actions = np.where(mask)[0]
            action = np.random.choice(valid_actions)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1

        assert done, f"Game should end eventually (ran {steps} steps)"

    def test_terminal_reward_is_nonzero(self):
        """Terminal reward is +1 (win) or -1 (loss)."""
        env = HearthstoneEnv(seed=42, max_turns=50)
        obs, info = env.reset()

        done = False
        final_reward = 0.0

        while not done:
            mask = info['action_mask']
            valid_actions = np.where(mask)[0]
            action = np.random.choice(valid_actions)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            final_reward = reward

        assert final_reward in [1.0, -1.0], f"Terminal reward should be +/-1, got {final_reward}"

    def test_multiple_resets(self):
        """Can reset and play multiple episodes."""
        env = HearthstoneEnv(seed=42, max_turns=30)

        for episode in range(3):
            obs, info = env.reset()
            assert obs is not None

            done = False
            while not done:
                mask = info['action_mask']
                valid_actions = np.where(mask)[0]
                action = np.random.choice(valid_actions)

                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated


class TestActionMapping:
    """Test integer action → game Action mapping."""

    def test_end_turn_action_index(self):
        """END_TURN is mapped to the last action index."""
        env = HearthstoneEnv(seed=42)
        env.reset()

        # END_TURN should be at index ACTION_SPACE_SIZE - 1
        game_action = env._decode_action(ACTION_SPACE_SIZE - 1)
        from simulation.action_space import ActionType
        assert game_action.type == ActionType.END_TURN

    def test_play_card_action_indices(self):
        """PLAY_CARD actions map to indices 0-69."""
        env = HearthstoneEnv(seed=42)
        env.reset()

        from simulation.action_space import ActionType

        # First play_card action: hand slot 0, board position 0
        action = env._decode_action(0)
        assert action.type == ActionType.PLAY_CARD
        assert action.card_index == 0
        assert action.position == 0

    def test_attack_action_indices(self):
        """ATTACK actions map to indices 70-98."""
        env = HearthstoneEnv(seed=42)
        env.reset()

        from simulation.action_space import ActionType

        # First attack action: attacker 0, target face (None)
        action = env._decode_action(70)
        assert action.type == ActionType.ATTACK

    def test_action_mask_matches_legal_actions(self):
        """Action mask correctly marks legal actions as True."""
        env = HearthstoneEnv(seed=42)
        obs, info = env.reset()
        mask = info['action_mask']

        # END_TURN should always be valid
        assert mask[ACTION_SPACE_SIZE - 1] == True

        # At least one action should be valid
        assert mask.sum() >= 1


class TestEnvMaxTurns:
    """Test max turns truncation."""

    def test_game_truncates_at_max_turns(self):
        """Game is truncated when max_turns is reached."""
        env = HearthstoneEnv(seed=42, max_turns=5)
        obs, info = env.reset()

        done = False
        steps = 0

        while not done and steps < 10000:
            # Always end turn immediately to cycle through turns fast
            obs, reward, terminated, truncated, info = env.step(ACTION_SPACE_SIZE - 1)
            done = terminated or truncated
            steps += 1

        assert done, "Game should eventually end"
