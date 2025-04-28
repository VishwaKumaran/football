from ultralytics import YOLO

from core import settings


class Trainer:
    def __init__(self, model_name: str, task: str = None) -> None:
        self.__model = YOLO(model_name, task)

    def train(self, data: str, epochs: int = 100, imgsz: int = 640, batch_size: int = 16) -> None:
        """
        Train the YOLO model.

        Args:
            data (str): Path to the dataset YAML file.
            epochs (int): Number of training epochs. Default is 100.
        """
        return self.__model.train(data=data, epochs=epochs, imgsz=imgsz, batch=batch_size, device=settings.DEVICE)


if __name__ == "__main__":
    # trainer = Trainer(settings.DETECTIONS_MODEL_NAME)
    # print("Training started...")
    # # Start training
    # print(trainer.train(data=settings.DETECTION_DATASET, epochs=1, imgsz=640, batch_size=8))
    # print("Training finished.")

    trainer_keypoint = Trainer(settings.KEYPOINT_MODEL_NAME)
    print("Training started...")
    print(trainer_keypoint.train(data=settings.KEYPOINT_DATASET, epochs=100, imgsz=640, batch_size=32))
    print("Training finished.")
