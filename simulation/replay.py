"""Record and replay games from random seed + action log.

Replay system enables:
- Perfect game reconstruction
- Debugging
- Analysis of agent behavior
- Tournament archiving
- Bug reproduction

Features:
- Seed + action log recording
- JSON serialization
- Step-by-step playback
- Metadata storage
- Action analysis
"""

import json
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
from collections import Counter
from simulation.action_space import Action, ActionType
from hearthstone.engine.game import Game


class ReplayRecorder:
    """Records game seed and action sequence for perfect replay.

    Stores:
    - Random seed for initialization
    - Sequence of (player_index, action, turn) tuples
    - Metadata (agents, date, etc.)
    - Final outcome
    """

    def __init__(self, seed: Optional[int] = None):
        """Initialize the recorder.

        Args:
            seed: Game random seed
        """
        self.seed = seed
        self.actions: List[Tuple[int, Action, int]] = []
        self.metadata: Dict[str, Any] = {}
        self.outcome: Dict[str, Any] = {}

    def record_action(self, player_index: int, action: Action, turn: int = 0) -> None:
        """Record an action taken by a player.

        Args:
            player_index: Which player (0 or 1)
            action: The action taken
            turn: Turn number when action was taken
        """
        self.actions.append((player_index, action, turn))

    def set_metadata(self, key: str, value: Any) -> None:
        """Store metadata about the game.

        Args:
            key: Metadata key
            value: Metadata value
        """
        self.metadata[key] = value

    def set_outcome(self, winner: int, turns: int, **kwargs) -> None:
        """Record final game outcome.

        Args:
            winner: Winning player (0 or 1)
            turns: Total number of turns
            **kwargs: Additional outcome data
        """
        self.outcome = {
            'winner': winner,
            'turns': turns,
            **kwargs
        }

    def action_count(self) -> int:
        """Get total number of recorded actions.

        Returns:
            Number of actions
        """
        return len(self.actions)

    def action_breakdown(self) -> Dict[ActionType, int]:
        """Analyze action type distribution.

        Returns:
            Dictionary mapping ActionType to count
        """
        action_types = [action.type for _, action, _ in self.actions]
        return dict(Counter(action_types))

    def to_dict(self) -> Dict[str, Any]:
        """Serialize replay to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            'seed': self.seed,
            'actions': [
                {
                    'player': player_idx,
                    'action': action.to_dict(),
                    'turn': turn
                }
                for player_idx, action, turn in self.actions
            ],
            'metadata': self.metadata,
            'outcome': self.outcome
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReplayRecorder":
        """Deserialize replay from dictionary.

        Args:
            data: Dictionary representation

        Returns:
            Reconstructed ReplayRecorder
        """
        recorder = cls(seed=data['seed'])

        for action_data in data['actions']:
            action = Action.from_dict(action_data['action'])
            recorder.actions.append((
                action_data['player'],
                action,
                action_data['turn']
            ))

        recorder.metadata = data.get('metadata', {})
        recorder.outcome = data.get('outcome', {})

        return recorder

    def save(self, filepath: str) -> None:
        """Save replay to JSON file.

        Args:
            filepath: Path to save file
        """
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    def __str__(self) -> str:
        """Human-readable representation."""
        return (
            f"ReplayRecorder(seed={self.seed}, "
            f"actions={len(self.actions)}, "
            f"outcome={self.outcome.get('winner', 'N/A')})"
        )


def load_replay(filepath: str) -> ReplayRecorder:
    """Load replay from JSON file.

    Args:
        filepath: Path to replay file

    Returns:
        Loaded ReplayRecorder
    """
    with open(filepath, 'r') as f:
        data = json.load(f)

    return ReplayRecorder.from_dict(data)


class ReplayPlayer:
    """Plays back a recorded game step-by-step.

    Enables:
    - Step-through debugging
    - Visual replay
    - State inspection at any point
    """

    def __init__(self, recorder: ReplayRecorder):
        """Initialize the replay player.

        Args:
            recorder: The recorded game to replay
        """
        self.recorder = recorder
        self.seed = recorder.seed
        self.current_action_index = 0
        self.game: Optional[Game] = None

    def reset(self) -> None:
        """Reset replay to beginning."""
        self.current_action_index = 0
        self.game = None

    def step(self) -> Optional[bool]:
        """Execute next action in replay.

        Returns:
            True if step successful, False/None if at end
        """
        if self.current_action_index >= len(self.recorder.actions):
            return None

        # Execute the action
        # (Full implementation requires action execution system)
        self.current_action_index += 1
        return True

    def play_all(self) -> None:
        """Play entire replay to completion."""
        while self.current_action_index < len(self.recorder.actions):
            result = self.step()
            if result is None or result is False:
                break
