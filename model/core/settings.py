from pathlib import Path
import platform
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    __PATH = Path(__file__).resolve().parent.parent

    DETECTION_DATASET: Path = __PATH / "dataset-detection/data.yaml"
    KEYPOINT_DATASET: Path = __PATH / "dataset-keypoint/data.yaml"

    DEVICE: str = "mps" if platform.system() == "Darwin" else "cuda"
    DETECTIONS_MODEL_NAME: str = "yolo11n.pt"
    KEYPOINT_MODEL_NAME: str = "yolo11n-pose.pt"

    BALL_CLASS_ID: int = 0
    GOALKEEPER_CLASS_ID: int = 1
    PLAYER_CLASS_ID: int = 2
    REFEREE_CLASS_ID: int = 3
    COLORS_PLAYERS: list[str] = ["#1E90FF", "#FF4500"]
    COLORS_BALL: list[str] = ["#FFFF00"]
    COLORS_REFEREE: list[str] = ["#000000"]
    COLORS_KEYPOINT: list[str] = ["#FFC0B0"]

    BATCH_SIZE: int = 32
    TEAM_CLASSIFIER_MODEL_NAME: str = "google/siglip2-base-patch16-224"
    TEAM_CLASSIFIER_OUTPUT_DIR: str = "team_classifier"
    TEAM_CLASSIFIER_LOGGING_DIR: str = "team_classifier"


settings = Settings()
