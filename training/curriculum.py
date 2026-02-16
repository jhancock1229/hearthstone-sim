"""Curriculum learning: staged training with gradually increasing complexity.

Starts with simplified rules (e.g., small hands, random opponents),
gradually adds complexity (larger boards, greedy opponents, full rules).

Components:
- Stage: Defines a curriculum stage with config overrides and unlock criteria
- CurriculumManager: Manages progression through stages based on metrics

Usage:
    from training.curriculum import Stage, CurriculumManager

    stages = [
        Stage(name="basics", description="Random opponent, short games",
              env_config={"max_turns": 50}, opponent="random",
              win_rate_threshold=0.55, min_episodes=100),
        Stage(name="intermediate", description="Greedy opponent",
              opponent="greedy", win_rate_threshold=0.55, min_episodes=200),
        Stage(name="advanced", description="Full complexity"),
    ]
    cm = CurriculumManager(stages)

    if cm.should_advance(win_rate=0.6):
        cm.advance()
    config = cm.get_env_config()
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Stage:
    """A single curriculum stage.

    Attributes:
        name: Stage identifier
        description: Human-readable description
        env_config: Dict of environment config overrides for this stage
        opponent: Opponent type ("random", "greedy", "self")
        win_rate_threshold: Win rate needed to advance past this stage
        min_episodes: Minimum episodes before advancement is allowed
    """
    name: str
    description: str
    env_config: Dict[str, Any] = field(default_factory=dict)
    opponent: str = "random"
    win_rate_threshold: float = 0.0
    min_episodes: int = 0


class CurriculumManager:
    """Manages progression through curriculum stages.

    Tracks the current stage, episode counts, and decides when
    to advance based on win rate and episode thresholds.
    """

    def __init__(self, stages: List[Stage]):
        """Initialize curriculum manager.

        Args:
            stages: Ordered list of stages (first = simplest)

        Raises:
            ValueError: If stages list is empty
        """
        if not stages:
            raise ValueError("Curriculum must have at least one stage")

        self.stages = stages
        self.current_stage_index = 0
        self.episodes_in_stage = 0
        self.stage_history: List[Dict[str, Any]] = []

    @property
    def current_stage(self) -> Stage:
        """Get the current stage."""
        return self.stages[self.current_stage_index]

    @property
    def is_complete(self) -> bool:
        """True if we're at the last stage."""
        return self.current_stage_index == len(self.stages) - 1

    def should_advance(self, win_rate: float = 0.0) -> bool:
        """Check whether the agent should advance to the next stage.

        Args:
            win_rate: Current win rate in the current stage

        Returns:
            True if criteria are met and there's a next stage
        """
        if self.is_complete:
            return False

        stage = self.current_stage
        meets_episodes = self.episodes_in_stage >= stage.min_episodes
        meets_win_rate = win_rate >= stage.win_rate_threshold

        return meets_episodes and meets_win_rate

    def advance(self):
        """Advance to the next stage.

        Records the completed stage in history and resets counters.
        Does nothing if already at the last stage.
        """
        if self.is_complete:
            return

        # Record completed stage
        self.stage_history.append({
            "stage": self.current_stage.name,
            "episodes": self.episodes_in_stage,
        })

        self.current_stage_index += 1
        self.episodes_in_stage = 0

    def record_episode(self):
        """Record that one episode was completed in the current stage."""
        self.episodes_in_stage += 1

    def get_env_config(self) -> Dict[str, Any]:
        """Get environment config overrides for the current stage.

        Returns:
            Dict of config overrides
        """
        return dict(self.current_stage.env_config)

    def get_opponent(self) -> str:
        """Get the opponent type for the current stage.

        Returns:
            Opponent type string
        """
        return self.current_stage.opponent

    def get_status(self) -> Dict[str, Any]:
        """Get current curriculum status.

        Returns:
            Dict with current_stage, stage_index, total_stages,
            episodes_in_stage, is_complete
        """
        return {
            "current_stage": self.current_stage.name,
            "stage_index": self.current_stage_index,
            "total_stages": len(self.stages),
            "episodes_in_stage": self.episodes_in_stage,
            "is_complete": self.is_complete,
        }

    @classmethod
    def create_default(cls) -> "CurriculumManager":
        """Create a default Hearthstone training curriculum.

        Stages:
        1. Basics: Random opponent, short games
        2. Intermediate: Random opponent, longer games
        3. Advanced: Greedy opponent, full games
        4. Expert: Self-play, full complexity

        Returns:
            CurriculumManager with default stages
        """
        stages = [
            Stage(
                name="basics",
                description="Random opponent, short games",
                env_config={"max_turns": 50},
                opponent="random",
                win_rate_threshold=0.55,
                min_episodes=100,
            ),
            Stage(
                name="intermediate",
                description="Random opponent, standard games",
                env_config={"max_turns": 100},
                opponent="random",
                win_rate_threshold=0.60,
                min_episodes=200,
            ),
            Stage(
                name="advanced",
                description="Greedy opponent, full games",
                env_config={"max_turns": 200},
                opponent="greedy",
                win_rate_threshold=0.55,
                min_episodes=500,
            ),
            Stage(
                name="expert",
                description="Self-play, full complexity",
                env_config={"max_turns": 200},
                opponent="self",
            ),
        ]
        return cls(stages)
