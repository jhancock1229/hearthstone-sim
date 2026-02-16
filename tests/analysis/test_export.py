"""Unit tests for data export.

Following TDD: these tests define the contract for export.py which
exports training results, matchup data, and meta summaries to
CSV and JSON formats.

Components:
- export_training_csv: Export training metrics to CSV
- export_matchups_csv: Export matchup matrix to CSV
- export_meta_json: Export meta summary to JSON
- export_training_json: Export training metrics to JSON
"""

import json
import os
import tempfile

import pytest

from analysis.export import (
    export_training_csv,
    export_matchups_csv,
    export_meta_json,
    export_training_json,
)
from analysis.metrics import MetricsTracker
from analysis.matchups import MatchupMatrix
from deckbuilding.meta import MetaTracker, Archetype


# ============================================================
# CSV Export Tests
# ============================================================


class TestExportTrainingCSV:
    """Tests for training metrics CSV export."""

    def test_export_creates_file(self):
        """Export creates a CSV file."""
        tracker = MetricsTracker()
        tracker.record_episode(reward=1.0, length=20)
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        try:
            export_training_csv(tracker, path)
            assert os.path.exists(path)
        finally:
            os.unlink(path)

    def test_csv_has_header(self):
        """CSV file has a header row."""
        tracker = MetricsTracker()
        tracker.record_episode(reward=1.0, length=20)
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            path = f.name
        try:
            export_training_csv(tracker, path)
            with open(path) as f:
                header = f.readline().strip()
            assert "episode" in header
            assert "reward" in header
            assert "length" in header
        finally:
            os.unlink(path)

    def test_csv_has_correct_rows(self):
        """CSV has one row per episode plus header."""
        tracker = MetricsTracker()
        for i in range(5):
            tracker.record_episode(reward=1.0 if i % 2 == 0 else -1.0, length=20 + i)
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        try:
            export_training_csv(tracker, path)
            with open(path) as f:
                lines = f.readlines()
            assert len(lines) == 6  # header + 5 data rows
        finally:
            os.unlink(path)

    def test_csv_empty_tracker(self):
        """Exporting empty tracker creates file with just header."""
        tracker = MetricsTracker()
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        try:
            export_training_csv(tracker, path)
            with open(path) as f:
                lines = f.readlines()
            assert len(lines) == 1  # header only
        finally:
            os.unlink(path)

    def test_csv_includes_losses(self):
        """CSV includes loss columns."""
        tracker = MetricsTracker()
        tracker.record_episode(
            reward=1.0, length=20,
            losses={"policy_loss": 0.5, "value_loss": 0.3, "entropy": 0.1},
        )
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        try:
            export_training_csv(tracker, path)
            with open(path) as f:
                header = f.readline().strip()
                data_line = f.readline().strip()
            assert "policy_loss" in header
            assert "0.5" in data_line
        finally:
            os.unlink(path)


class TestExportMatchupsCSV:
    """Tests for matchup matrix CSV export."""

    def test_export_creates_file(self):
        """Export creates a CSV file."""
        matrix = MatchupMatrix()
        matrix.record("A", "B", wins=7, losses=3)
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        try:
            export_matchups_csv(matrix, path)
            assert os.path.exists(path)
        finally:
            os.unlink(path)

    def test_csv_is_square_matrix(self):
        """CSV represents a square matrix with labels."""
        matrix = MatchupMatrix()
        matrix.record("A", "B", wins=7, losses=3)
        matrix.record("A", "C", wins=6, losses=4)
        matrix.record("B", "C", wins=5, losses=5)
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        try:
            export_matchups_csv(matrix, path)
            with open(path) as f:
                lines = f.readlines()
            # Header + 3 rows
            assert len(lines) == 4
            # Each row has label + 3 values
            header_cols = lines[0].strip().split(",")
            assert len(header_cols) == 4  # empty corner + A, B, C
        finally:
            os.unlink(path)

    def test_empty_matrix_creates_file(self):
        """Empty matrix still creates a file."""
        matrix = MatchupMatrix()
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        try:
            export_matchups_csv(matrix, path)
            assert os.path.exists(path)
        finally:
            os.unlink(path)


# ============================================================
# JSON Export Tests
# ============================================================


class TestExportMetaJSON:
    """Tests for meta summary JSON export."""

    def test_export_creates_file(self):
        """Export creates a JSON file."""
        tracker = MetaTracker()
        tracker.register_archetype(Archetype(name="Aggro", description="Fast"))
        tracker.record_match("Aggro", "Control", winner="Aggro")
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            export_meta_json(tracker, path)
            assert os.path.exists(path)
        finally:
            os.unlink(path)

    def test_json_is_valid(self):
        """Output is valid JSON."""
        tracker = MetaTracker()
        tracker.record_match("Aggro", "Control", winner="Aggro")
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            export_meta_json(tracker, path)
            with open(path) as f:
                data = json.load(f)
            assert isinstance(data, dict)
        finally:
            os.unlink(path)

    def test_json_contains_archetypes(self):
        """JSON has archetype data."""
        tracker = MetaTracker()
        tracker.record_match("Aggro", "Control", winner="Aggro")
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            export_meta_json(tracker, path)
            with open(path) as f:
                data = json.load(f)
            assert "archetypes" in data
            assert "total_matches" in data
        finally:
            os.unlink(path)

    def test_empty_tracker_exports(self):
        """Empty tracker still produces valid JSON."""
        tracker = MetaTracker()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            export_meta_json(tracker, path)
            with open(path) as f:
                data = json.load(f)
            assert data["total_matches"] == 0
        finally:
            os.unlink(path)


class TestExportTrainingJSON:
    """Tests for training metrics JSON export."""

    def test_export_creates_file(self):
        """Export creates a JSON file."""
        tracker = MetricsTracker()
        tracker.record_episode(reward=1.0, length=20)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            export_training_json(tracker, path)
            assert os.path.exists(path)
        finally:
            os.unlink(path)

    def test_json_contains_episodes(self):
        """JSON has episode data."""
        tracker = MetricsTracker()
        tracker.record_episode(reward=1.0, length=20)
        tracker.record_episode(reward=-1.0, length=30)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            export_training_json(tracker, path)
            with open(path) as f:
                data = json.load(f)
            assert "episodes" in data
            assert len(data["episodes"]) == 2
        finally:
            os.unlink(path)

    def test_json_contains_summary(self):
        """JSON includes summary statistics."""
        tracker = MetricsTracker()
        tracker.record_episode(reward=1.0, length=20)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            export_training_json(tracker, path)
            with open(path) as f:
                data = json.load(f)
            assert "summary" in data
            assert "mean_reward" in data["summary"]
        finally:
            os.unlink(path)

    def test_empty_tracker_exports(self):
        """Empty tracker still produces valid JSON."""
        tracker = MetricsTracker()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            export_training_json(tracker, path)
            with open(path) as f:
                data = json.load(f)
            assert data["episodes"] == []
        finally:
            os.unlink(path)
