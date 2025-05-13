import os
from typing import List

from ultralytics import YOLO
from ultralytics.data.annotator import auto_annotate

from core import settings
from core.tools import convert_seg_to_bbox, visualize_image


class Trainer:
    def __init__(self, model_name: str, task: str = None) -> None:
        self.__model = YOLO(model_name, task)

    @staticmethod
    def annotate_dataset(data: str, output_dir: str, model_name: str, sam_model: str, classes: List[str] = None,
                         confidence: float = 0.25, ):
        auto_annotate(
            data, model_name, device=settings.DEVICE, output_dir=output_dir, classes=classes, conf=confidence,
            sam_model=sam_model,
        )
        for file in os.listdir(output_dir):
            if file.endswith(".txt"):
                convert_seg_to_bbox(os.path.join(output_dir, file))

    def train(self, data: str, epochs: int = 100, imgsz: int = 640, batch_size: int = 16) -> None:
        return self.__model.train(data=data, epochs=epochs, imgsz=imgsz, batch=batch_size, device=settings.DEVICE)


if __name__ == "__main__":
    # trainer = Trainer(settings.DETECTIONS_MODEL_NAME)
    # print("Training started...")
    # # Start training
    # print(trainer.train(data=settings.DETECTION_DATASET, epochs=1, imgsz=640, batch_size=8))
    # print("Training finished.")

    # trainer_keypoint = Trainer(settings.KEYPOINT_MODEL_NAME)
    # print("Training started...")
    # print(trainer_keypoint.train(data=settings.KEYPOINT_DATASET, epochs=100, imgsz=640, batch_size=32))
    # print("Training finished.")

    # print(Trainer.annotate_dataset(
    #     "/Users/vishwa/Documents/Projects/football/model/tests/images",
    #     "/Users/vishwa/Documents/Projects/football/model/tests/output",
    #     "/Users/vishwa/Documents/Projects/football/model/detection.pt",
    #     "sam2_t.pt"
    # ))

    visualize_image(
        {0: "ball", 1: "goalkeeper", 2: "player", 3: "referee"},
        "/Users/vishwa/Documents/Projects/football/model/tests/images/2e57b9_1_8_png.rf.eddb5d52cda524e90e21bc091077ef2d.jpg",
        "/Users/vishwa/Documents/Projects/football/model/tests/output/2e57b9_1_8_png.rf.eddb5d52cda524e90e21bc091077ef2d.txt",
    )
