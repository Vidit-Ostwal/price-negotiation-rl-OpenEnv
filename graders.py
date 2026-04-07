"""Task-specific graders for OpenEnv task configuration."""

from __future__ import annotations

from dataclasses import dataclass

from price_negotiation.reward import reward_breakdown, score_trajectory
from price_negotiation.trajectory_types import TrajectoryResult


@dataclass
class DifficultyGrader:
    """Grade a trajectory and ensure it matches the expected difficulty."""

    difficulty: str

    def _trajectory_difficulty(self, trajectory: TrajectoryResult) -> str | None:
        """Read the configured difficulty from the final environment state."""
        product_info = trajectory.final_state.product_info or {}
        return product_info.get("difficulty") or product_info.get("valuations", {}).get(
            "difficulty"
        )

    def validate(self, trajectory: TrajectoryResult) -> None:
        """Raise if the trajectory does not match the grader's difficulty."""
        actual = self._trajectory_difficulty(trajectory)
        if actual != self.difficulty:
            raise ValueError(
                f"Expected difficulty '{self.difficulty}', got '{actual}'"
            )

    def breakdown(self, trajectory: TrajectoryResult) -> dict[str, float]:
        """Return the raw reward-component breakdown for the trajectory."""
        self.validate(trajectory)
        return reward_breakdown(trajectory)

    def score(self, trajectory: TrajectoryResult) -> float:
        """Return the normalized continuous score in [0, 1]."""
        self.validate(trajectory)
        return score_trajectory(trajectory)

    def __call__(self, trajectory: TrajectoryResult) -> float:
        """Allow the grader instance to be used as a callable scorer."""
        return self.score(trajectory)


class EasyGrader(DifficultyGrader):
    """Grader for easy negotiation tasks."""

    def __init__(self) -> None:
        super().__init__(difficulty="easy")


class MediumGrader(DifficultyGrader):
    """Grader for medium negotiation tasks."""

    def __init__(self) -> None:
        super().__init__(difficulty="medium")


class HardGrader(DifficultyGrader):
    """Grader for hard negotiation tasks."""

    def __init__(self) -> None:
        super().__init__(difficulty="hard")
