"""Unit tests for curriculum learning.

Following TDD: these tests define the contract for curriculum.py which
implements staged training with gradually increasing complexity.

Components:
- Stage: Defines a curriculum stage (name, config overrides, unlock criteria)
- CurriculumManager: Manages progression through stages based on metrics
"""

import pytest

from training.curriculum import Stage, CurriculumManager


# ============================================================
# Stage Tests
# ============================================================


class TestStageCreation:
    """Tests for Stage dataclass."""

    def test_create_stage(self):
        """Can create a stage with name and description."""
        stage = Stage(
            name="basics",
            description="Learn basic card play",
        )
        assert stage.name == "basics"
        assert stage.description == "Learn basic card play"

    def test_stage_with_env_overrides(self):
        """Stage can carry environment config overrides."""
        stage = Stage(
            name="combat",
            description="Add combat mechanics",
            env_config={"max_hand_size": 3, "max_board_size": 3},
        )
        assert stage.env_config["max_hand_size"] == 3
        assert stage.env_config["max_board_size"] == 3

    def test_stage_with_unlock_criteria(self):
        """Stage has unlock criteria (win rate threshold, min episodes)."""
        stage = Stage(
            name="advanced",
            description="Full game",
            win_rate_threshold=0.55,
            min_episodes=100,
        )
        assert stage.win_rate_threshold == 0.55
        assert stage.min_episodes == 100

    def test_stage_defaults(self):
        """Default stage has reasonable defaults."""
        stage = Stage(name="test", description="test stage")
        assert stage.env_config == {}
        assert stage.win_rate_threshold == 0.0
        assert stage.min_episodes == 0

    def test_stage_with_opponent_type(self):
        """Stage can specify opponent type."""
        stage = Stage(
            name="vs_greedy",
            description="Train against greedy",
            opponent="greedy",
        )
        assert stage.opponent == "greedy"

    def test_stage_default_opponent_is_random(self):
        """Default opponent is random."""
        stage = Stage(name="test", description="test")
        assert stage.opponent == "random"


# ============================================================
# CurriculumManager Creation Tests
# ============================================================


class TestCurriculumManagerCreation:
    """Tests for CurriculumManager initialization."""

    def test_create_with_stages(self):
        """Can create manager with a list of stages."""
        stages = [
            Stage(name="stage1", description="First"),
            Stage(name="stage2", description="Second"),
        ]
        cm = CurriculumManager(stages)
        assert len(cm.stages) == 2

    def test_starts_at_first_stage(self):
        """Manager starts at stage index 0."""
        stages = [
            Stage(name="stage1", description="First"),
            Stage(name="stage2", description="Second"),
        ]
        cm = CurriculumManager(stages)
        assert cm.current_stage_index == 0
        assert cm.current_stage.name == "stage1"

    def test_empty_stages_raises(self):
        """Must have at least one stage."""
        with pytest.raises(ValueError, match="at least one"):
            CurriculumManager([])

    def test_tracks_episodes_per_stage(self):
        """Manager tracks how many episodes have been run in current stage."""
        stages = [Stage(name="s1", description="d")]
        cm = CurriculumManager(stages)
        assert cm.episodes_in_stage == 0


# ============================================================
# CurriculumManager Progression Tests
# ============================================================


class TestCurriculumProgression:
    """Tests for advancing through curriculum stages."""

    def test_should_advance_win_rate(self):
        """Advances when win rate exceeds stage threshold."""
        stages = [
            Stage(name="s1", description="d", win_rate_threshold=0.55, min_episodes=0),
            Stage(name="s2", description="d"),
        ]
        cm = CurriculumManager(stages)
        assert cm.should_advance(win_rate=0.6) is True
        assert cm.should_advance(win_rate=0.4) is False

    def test_should_advance_min_episodes(self):
        """Won't advance until min_episodes reached even if win rate is high."""
        stages = [
            Stage(name="s1", description="d", win_rate_threshold=0.55, min_episodes=10),
            Stage(name="s2", description="d"),
        ]
        cm = CurriculumManager(stages)
        cm.episodes_in_stage = 5
        assert cm.should_advance(win_rate=0.8) is False
        cm.episodes_in_stage = 10
        assert cm.should_advance(win_rate=0.8) is True

    def test_advance_moves_to_next_stage(self):
        """advance() moves to the next stage."""
        stages = [
            Stage(name="s1", description="d"),
            Stage(name="s2", description="d"),
        ]
        cm = CurriculumManager(stages)
        cm.advance()
        assert cm.current_stage_index == 1
        assert cm.current_stage.name == "s2"

    def test_advance_resets_episode_counter(self):
        """Advancing resets the per-stage episode counter."""
        stages = [
            Stage(name="s1", description="d"),
            Stage(name="s2", description="d"),
        ]
        cm = CurriculumManager(stages)
        cm.episodes_in_stage = 50
        cm.advance()
        assert cm.episodes_in_stage == 0

    def test_cannot_advance_past_last_stage(self):
        """advance() at the last stage does nothing."""
        stages = [Stage(name="s1", description="d")]
        cm = CurriculumManager(stages)
        cm.advance()
        assert cm.current_stage_index == 0  # Still at 0

    def test_should_advance_false_at_last_stage(self):
        """should_advance returns False at the last stage."""
        stages = [Stage(name="s1", description="d", win_rate_threshold=0.5)]
        cm = CurriculumManager(stages)
        assert cm.should_advance(win_rate=0.9) is False

    def test_is_complete(self):
        """is_complete is True only at last stage."""
        stages = [
            Stage(name="s1", description="d"),
            Stage(name="s2", description="d"),
        ]
        cm = CurriculumManager(stages)
        assert cm.is_complete is False
        cm.advance()
        assert cm.is_complete is True

    def test_record_episode_increments_counter(self):
        """record_episode increments the episode counter."""
        stages = [Stage(name="s1", description="d")]
        cm = CurriculumManager(stages)
        cm.record_episode()
        cm.record_episode()
        assert cm.episodes_in_stage == 2

    def test_multi_stage_progression(self):
        """Can advance through multiple stages sequentially."""
        stages = [
            Stage(name="s1", description="d", win_rate_threshold=0.5, min_episodes=0),
            Stage(name="s2", description="d", win_rate_threshold=0.6, min_episodes=0),
            Stage(name="s3", description="d"),
        ]
        cm = CurriculumManager(stages)

        # Advance past s1
        assert cm.should_advance(win_rate=0.55) is True
        cm.advance()
        assert cm.current_stage.name == "s2"

        # Not enough for s2
        assert cm.should_advance(win_rate=0.55) is False

        # Advance past s2
        assert cm.should_advance(win_rate=0.65) is True
        cm.advance()
        assert cm.current_stage.name == "s3"
        assert cm.is_complete is True


# ============================================================
# CurriculumManager Env Config Tests
# ============================================================


class TestCurriculumEnvConfig:
    """Tests for environment configuration at each stage."""

    def test_get_env_config(self):
        """get_env_config returns current stage's env_config."""
        stages = [
            Stage(name="s1", description="d", env_config={"max_turns": 50}),
            Stage(name="s2", description="d", env_config={"max_turns": 200}),
        ]
        cm = CurriculumManager(stages)
        assert cm.get_env_config()["max_turns"] == 50
        cm.advance()
        assert cm.get_env_config()["max_turns"] == 200

    def test_empty_env_config_returns_empty_dict(self):
        """Stage with no env_config overrides returns empty dict."""
        stages = [Stage(name="s1", description="d")]
        cm = CurriculumManager(stages)
        assert cm.get_env_config() == {}

    def test_get_opponent_type(self):
        """get_opponent returns current stage's opponent."""
        stages = [
            Stage(name="s1", description="d", opponent="random"),
            Stage(name="s2", description="d", opponent="greedy"),
        ]
        cm = CurriculumManager(stages)
        assert cm.get_opponent() == "random"
        cm.advance()
        assert cm.get_opponent() == "greedy"


# ============================================================
# CurriculumManager Stats / Logging Tests
# ============================================================


class TestCurriculumStats:
    """Tests for curriculum status reporting."""

    def test_get_status(self):
        """get_status returns current stage info."""
        stages = [
            Stage(name="basics", description="Learn basics"),
            Stage(name="combat", description="Learn combat"),
        ]
        cm = CurriculumManager(stages)
        status = cm.get_status()
        assert status["current_stage"] == "basics"
        assert status["stage_index"] == 0
        assert status["total_stages"] == 2
        assert status["episodes_in_stage"] == 0
        assert status["is_complete"] is False

    def test_stage_history_tracked(self):
        """Advancement history is recorded."""
        stages = [
            Stage(name="s1", description="d"),
            Stage(name="s2", description="d"),
            Stage(name="s3", description="d"),
        ]
        cm = CurriculumManager(stages)
        cm.episodes_in_stage = 50
        cm.advance()
        cm.episodes_in_stage = 30
        cm.advance()
        assert len(cm.stage_history) == 2
        assert cm.stage_history[0]["stage"] == "s1"
        assert cm.stage_history[0]["episodes"] == 50
        assert cm.stage_history[1]["stage"] == "s2"
        assert cm.stage_history[1]["episodes"] == 30


# ============================================================
# Default Curriculum Tests
# ============================================================


class TestDefaultCurriculum:
    """Tests for the pre-built default curriculum."""

    def test_create_default_curriculum(self):
        """Can create a default Hearthstone curriculum."""
        cm = CurriculumManager.create_default()
        assert len(cm.stages) >= 3
        assert cm.current_stage_index == 0

    def test_default_stages_have_increasing_complexity(self):
        """Default stages progress from simple to complex."""
        cm = CurriculumManager.create_default()
        # First stage should be simpler (fewer turns, random opponent)
        first = cm.stages[0]
        last = cm.stages[-1]
        assert first.opponent == "random"
