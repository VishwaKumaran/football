from random import shuffle
import numpy as np
import supervision as sv
from ultralytics import YOLO

from core import settings
from model.team_classifier import TeamClassifier


class Inference:
    def __init__(self, detection_model_name: str = settings.DETECTIONS_MODEL_NAME,
                 keypoint_model_name: str = settings.KEYPOINT_MODEL_NAME):
        self.__detection_model = YOLO(detection_model_name).to(settings.DEVICE)
        self.__keypoint_model = YOLO(keypoint_model_name).to(settings.DEVICE)

    def __goalkeeper_classification(self, player_detections, goalkeeper_detections):
        players_xy = player_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        first_team = players_xy[player_detections.class_id == 0].mean(axis=0)
        second_team = players_xy[player_detections.class_id == 1].mean(axis=0)

        goalkeepers_teams = []
        for xy in goalkeeper_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER):
            distance_first_team = np.linalg.norm(xy - first_team)
            distance_second_team = np.linalg.norm(xy - second_team)
            goalkeepers_teams.append(0 if distance_first_team < distance_second_team else 1)

        return np.array(goalkeepers_teams)

    def detection(self, source_path: str, stride: int = 1):
        tracker = sv.ByteTrack()
        player_annotator = sv.EllipseAnnotator(sv.ColorPalette.from_hex(settings.COLORS_PLAYERS))
        referee_annotator = sv.EllipseAnnotator(sv.ColorPalette.from_hex(settings.COLORS_REFEREE))
        triangle_annotator = sv.TriangleAnnotator(sv.ColorPalette.from_hex(settings.COLORS_BALL))
        label_annotator = sv.LabelAnnotator(color=sv.ColorPalette.from_hex(settings.COLORS_PLAYERS),
                                            text_position=sv.Position.BOTTOM_CENTER, text_thickness=2)

        vertex_annotator = sv.VertexAnnotator(sv.ColorPalette.from_hex(settings.COLORS_KEYPOINT))

        crops = self.crop(source_path, stride)
        shuffle(crops)
        team_classifier = TeamClassifier()
        team_classifier.fit(crops)

        def callback(frame: np.ndarray, _: int) -> np.ndarray:
            results = self.__detection_model.predict(source=frame)[0]
            keypoint_results = self.__keypoint_model.predict(source=frame)[0]
            key_points = sv.KeyPoints.from_ultralytics(keypoint_results)
            detections = sv.Detections.from_ultralytics(results)
            detections = tracker.update_with_detections(detections)

            ball_detections = detections[detections.class_id == settings.BALL_CLASS_ID]
            referee_detections = detections[detections.class_id == settings.REFEREE_CLASS_ID]
            player_detections = detections[detections.class_id == settings.PLAYER_CLASS_ID]
            player_crops = [sv.crop_image(frame, xyxy) for xyxy in player_detections.xyxy]
            player_detections.class_id = team_classifier.predict(player_crops)
            goalkeeper_detections = detections[detections.class_id == settings.GOALKEEPER_CLASS_ID]
            goalkeeper_detections.class_id = self.__goalkeeper_classification(player_detections, goalkeeper_detections)

            all_player_detections = sv.Detections.merge([player_detections, goalkeeper_detections])

            labels = [str(tracker_id) for tracker_id in all_player_detections.tracker_id]
            annotated_frame = player_annotator.annotate(frame.copy(), detections=all_player_detections)
            annotated_frame = referee_annotator.annotate(annotated_frame, detections=referee_detections)
            annotated_frame = label_annotator.annotate(annotated_frame, detections=all_player_detections, labels=labels)
            annotated_frame = triangle_annotator.annotate(annotated_frame, detections=ball_detections)
            return vertex_annotator.annotate(annotated_frame, key_points=key_points)

        return callback

    def crop(self, source_path: str, stride: int = 1):
        crops = []
        for frame in sv.get_video_frames_generator(source_path, stride):
            results = self.__detection_model.predict(source=frame)[0]
            detections = sv.Detections.from_ultralytics(results)
            detections = detections[detections.class_id == settings.PLAYER_CLASS_ID]

            crops += [sv.crop_image(frame, xyxy) for xyxy in detections.xyxy]

        return crops

    def __callback(self, type: str, **kwargs):
        if type == "detection":
            return self.detection(**kwargs)
        raise ValueError(f"Invalid callback type: {type}. Supported types are: detection")

    def process(self, source_path: str, target_path: str, callback: str, **kwargs):
        return sv.process_video(source_path=source_path, target_path=target_path,
                                callback=self.__callback(callback, source_path=source_path, **kwargs))

    def process_one_frame(self, source_path: str, target_path: str, callback: str, **kwargs):
        for frame in sv.get_video_frames_generator(source_path):
            cb = self.__callback(callback, source_path=source_path, **kwargs)
            return sv.plot_image(cb(frame, 0))


if __name__ == "__main__":
    inference = Inference("detection.pt")
    inference.process_one_frame(
        source_path="/Users/vishwa/Documents/Projects/football/model/tests/test.mp4",
        target_path="output.mp4",
        callback="detection",
        stride=10
    )
    # a = inference.crop(
    #     source_path="/Users/vishwa/Documents/Projects/football/model/tests/test.mp4",
    #     stride=60
    # )
    # print(len(a))
