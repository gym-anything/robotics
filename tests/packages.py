"""Paths the tests use to read this project's files."""

from pathlib import Path

from gym_anything_robotics.paths import worker_path

# The repository root, for pyproject.toml and the benchmark splits.
REPO = Path(__file__).resolve().parents[1]
# The project's top-level packages side by side, for paths such as PACKAGES / "welding" / "benchmark.json".
PACKAGES = Path(worker_path())
