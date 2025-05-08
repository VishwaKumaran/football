from enum import Enum
from typing import List, Tuple

from pydantic_settings import BaseSettings


class PitchType(str, Enum):
    DEFAULT = "default"
    FIVE = "five"
    KINGS_LEAGUE = "KINGS_LEAGUE"


class PitchConfig(BaseSettings):
    WIDTH: int
    LENGTH: int
    CENTRE_CIRCLE_RADIUS: int
    GOAL_BOX_LENGTH: int
    GOAL_BOX_WIDTH: int
    PENALTY_SPOT_DISTANCE: int
    PENALTY_BOX_LENGTH: int
    PENALTY_BOX_WIDTH: int
    EDGES: List[Tuple[int, int]]
    LABELS: List[str]
    COLORS: List[str]

    @property
    def vertices(self) -> List[Tuple[int, int]]:
        w, l = self.WIDTH, self.LENGTH
        pbw, pbl, gbl, gbw = self.PENALTY_BOX_WIDTH, self.PENALTY_BOX_LENGTH, self.GOAL_BOX_LENGTH, self.GOAL_BOX_WIDTH
        center = self.CENTRE_CIRCLE_RADIUS
        spot = self.PENALTY_SPOT_DISTANCE
        return [
            (0, 0),  # 1
            (0, (self.WIDTH - self.PENALTY_BOX_WIDTH) / 2),  # 2
            (0, (self.WIDTH - self.GOAL_BOX_WIDTH) / 2),  # 3
            (0, (self.WIDTH + self.GOAL_BOX_WIDTH) / 2),  # 4
            (0, (self.WIDTH + self.PENALTY_BOX_WIDTH) / 2),  # 5
            (0, self.WIDTH),  # 6
            (self.GOAL_BOX_LENGTH, (self.WIDTH - self.GOAL_BOX_WIDTH) / 2),  # 7
            (self.GOAL_BOX_LENGTH, (self.WIDTH + self.GOAL_BOX_WIDTH) / 2),  # 8
            (self.PENALTY_SPOT_DISTANCE, self.WIDTH / 2),  # 9
            (self.PENALTY_BOX_LENGTH, (self.WIDTH - self.PENALTY_BOX_WIDTH) / 2),  # 10
            (self.PENALTY_BOX_LENGTH, (self.WIDTH - self.GOAL_BOX_WIDTH) / 2),  # 11
            (self.PENALTY_BOX_LENGTH, (self.WIDTH + self.GOAL_BOX_WIDTH) / 2),  # 12
            (self.PENALTY_BOX_LENGTH, (self.WIDTH + self.PENALTY_BOX_WIDTH) / 2),  # 13
            (self.LENGTH / 2, 0),  # 14
            (self.LENGTH / 2, self.WIDTH / 2 - self.CENTRE_CIRCLE_RADIUS),  # 15
            (self.LENGTH / 2, self.WIDTH / 2 + self.CENTRE_CIRCLE_RADIUS),  # 16
            (self.LENGTH / 2, self.WIDTH),  # 17
            (
                self.LENGTH - self.PENALTY_BOX_LENGTH,
                (self.WIDTH - self.PENALTY_BOX_WIDTH) / 2
            ),  # 18
            (
                self.LENGTH - self.PENALTY_BOX_LENGTH,
                (self.WIDTH - self.GOAL_BOX_WIDTH) / 2
            ),  # 19
            (
                self.LENGTH - self.PENALTY_BOX_LENGTH,
                (self.WIDTH + self.GOAL_BOX_WIDTH) / 2
            ),  # 20
            (
                self.LENGTH - self.PENALTY_BOX_LENGTH,
                (self.WIDTH + self.PENALTY_BOX_WIDTH) / 2
            ),  # 21
            (self.LENGTH - self.PENALTY_SPOT_DISTANCE, self.WIDTH / 2),  # 22
            (
                self.LENGTH - self.GOAL_BOX_LENGTH,
                (self.WIDTH - self.GOAL_BOX_WIDTH) / 2
            ),  # 23
            (
                self.LENGTH - self.GOAL_BOX_LENGTH,
                (self.WIDTH + self.GOAL_BOX_WIDTH) / 2
            ),  # 24
            (self.LENGTH, 0),  # 25
            (self.LENGTH, (self.WIDTH - self.PENALTY_BOX_WIDTH) / 2),  # 26
            (self.LENGTH, (self.WIDTH - self.GOAL_BOX_WIDTH) / 2),  # 27
            (self.LENGTH, (self.WIDTH + self.GOAL_BOX_WIDTH) / 2),  # 28
            (self.LENGTH, (self.WIDTH + self.PENALTY_BOX_WIDTH) / 2),  # 29
            (self.LENGTH, self.WIDTH),  # 30
            (self.LENGTH / 2 - self.CENTRE_CIRCLE_RADIUS, self.WIDTH / 2),  # 31
            (self.LENGTH / 2 + self.CENTRE_CIRCLE_RADIUS, self.WIDTH / 2),  # 32
        ]

    @classmethod
    def from_type(cls, pitch_type: PitchType) -> "PitchConfig":
        if pitch_type == PitchType.DEFAULT:
            return cls(
                WIDTH=7000,
                LENGTH=12000,
                CENTRE_CIRCLE_RADIUS=915,
                GOAL_BOX_LENGTH=550,
                GOAL_BOX_WIDTH=1832,
                PENALTY_SPOT_DISTANCE=1100,
                PENALTY_BOX_LENGTH=2015,
                PENALTY_BOX_WIDTH=4100,
                EDGES=[
                    (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (7, 8),
                    (10, 11), (11, 12), (12, 13), (14, 15), (15, 16),
                    (16, 17), (18, 19), (19, 20), (20, 21), (23, 24),
                    (25, 26), (26, 27), (27, 28), (28, 29), (29, 30),
                    (1, 14), (2, 10), (3, 7), (4, 8), (5, 13), (6, 17),
                    (14, 25), (18, 26), (23, 27), (24, 28), (21, 29), (17, 30),
                ],
                LABELS=[
                    "01", "02", "03", "04", "05", "06", "07", "08", "09", "10",
                    "11", "12", "13", "15", "16", "17", "18", "20", "21", "22",
                    "23", "24", "25", "26", "27", "28", "29", "30", "31", "32",
                    "14", "19"
                ],
                COLORS=[
                    "#FF1493", "#FF1493", "#FF1493", "#FF1493", "#FF1493", "#FF1493",
                    "#FF1493", "#FF1493", "#FF1493", "#FF1493", "#FF1493", "#FF1493",
                    "#FF1493", "#00BFFF", "#00BFFF", "#00BFFF", "#00BFFF", "#FF6347",
                    "#FF6347", "#FF6347", "#FF6347", "#FF6347", "#FF6347", "#FF6347",
                    "#FF6347", "#FF6347", "#FF6347", "#FF6347", "#FF6347", "#FF6347",
                    "#00BFFF", "#00BFFF"
                ],
            )

        elif pitch_type == PitchType.FIVE:
            return cls(
                WIDTH=3000,
                LENGTH=5000,
                CENTRE_CIRCLE_RADIUS=500,
                GOAL_BOX_LENGTH=400,
                GOAL_BOX_WIDTH=1500,
                PENALTY_SPOT_DISTANCE=800,
                PENALTY_BOX_LENGTH=1000,
                PENALTY_BOX_WIDTH=3000,
                EDGES=[(1, 6), (6, 30), (30, 25), (25, 24), (24, 13), (13, 14), (14, 15), (15, 16),
                       (16, 29), (29, 28), (28, 27), (27, 26), (26, 1)],
                LABELS=[],
                COLORS=[],
            )

        elif pitch_type == PitchType.KINGS_LEAGUE:
            return cls(
                WIDTH=2000,
                LENGTH=4000,
                CENTRE_CIRCLE_RADIUS=400,
                GOAL_BOX_LENGTH=300,
                GOAL_BOX_WIDTH=1000,
                PENALTY_SPOT_DISTANCE=600,
                PENALTY_BOX_LENGTH=800,
                PENALTY_BOX_WIDTH=2000,
                EDGES=[(1, 6), (6, 30), (30, 25), (25, 24), (24, 13), (13, 14), (14, 15), (15, 16),
                       (16, 29), (29, 28), (28, 27), (27, 26), (26, 1)],
                LABELS=[],
                COLORS=[],
            )

        raise ValueError(f"Unknown pitch type: {pitch_type}")
