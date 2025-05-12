import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

from core import settings
from model.pitch import Pitch
from model.team_classifier import TeamClassifier


class Inference:
    def __init__(self, detection_model_name: str = settings.DETECTIONS_MODEL_NAME,
                 keypoint_model_name: str = settings.KEYPOINT_MODEL_NAME):
        self.__detection_model = YOLO(detection_model_name).to(settings.DEVICE)
        self.__keypoint_model = YOLO(keypoint_model_name).to(settings.DEVICE)

    def __predict(self, model: YOLO, frame: np.ndarray):
        return model.predict(source=frame, verbose=False)[0]

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

    def __classify_players(self, frame: np.ndarray, detections, team_classifier):
        player_detections = detections[detections.class_id == settings.PLAYER_CLASS_ID]
        player_crops = [sv.crop_image(frame, xyxy) for xyxy in player_detections.xyxy]
        player_detections.class_id = team_classifier.predict(player_crops)
        return player_detections

    def __process_frame_with_detections(self, frame: np.ndarray, tracker, team_classifier, pitch):
        results = self.__predict(self.__detection_model, frame)
        keypoint_results = self.__predict(self.__keypoint_model, frame)
        detections = sv.Detections.from_ultralytics(results)
        detections = tracker.update_with_detections(detections)
        key_points = sv.KeyPoints.from_ultralytics(keypoint_results)

        player_detections = self.__classify_players(frame, detections, team_classifier)
        goalkeeper_detections = detections[detections.class_id == settings.GOALKEEPER_CLASS_ID]
        goalkeeper_detections.class_id = self.__goalkeeper_classification(player_detections, goalkeeper_detections)
        player_goalkeeper_detections = sv.Detections.merge([player_detections, goalkeeper_detections])
        ball_detections = detections[detections.class_id == settings.BALL_CLASS_ID]
        referee_detections = detections[detections.class_id == settings.REFEREE_CLASS_ID]

        mask = (key_points.xy[0][:, 0] > 1) & (key_points.xy[0][:, 1] > 1)
        transformer = pitch.get_transform(key_points.xy[0], mask=mask)

        return {
            "key_points": key_points,
            "transformer": transformer,
            "player_goalkeeper_detections": player_goalkeeper_detections,
            "ball_detections": ball_detections,
            "referee_detections": referee_detections
        }

    def detection(self, source_path: str, stride: int = 1):
        pitch = Pitch()
        tracker = sv.ByteTrack()
        player_annotator = sv.EllipseAnnotator(sv.ColorPalette.from_hex(settings.COLORS_PLAYERS))
        referee_annotator = sv.EllipseAnnotator(sv.ColorPalette.from_hex(settings.COLORS_REFEREE))
        triangle_annotator = sv.TriangleAnnotator(sv.ColorPalette.from_hex(settings.COLORS_BALL))
        label_annotator = sv.LabelAnnotator(color=sv.ColorPalette.from_hex(settings.COLORS_PLAYERS),
                                            text_position=sv.Position.BOTTOM_CENTER, text_thickness=2)

        vertex_annotator = sv.VertexAnnotator(sv.Color.from_hex(settings.COLORS_KEYPOINT))

        edge_annotator = sv.EdgeAnnotator(sv.Color.from_hex("00BFFF"), thickness=2, edges=pitch.config.EDGES)

        crops = self.crop(source_path, stride)
        team_classifier = TeamClassifier()
        team_classifier.fit(crops)

        def callback(frame: np.ndarray, _: int) -> np.ndarray:
            data = self.__process_frame_with_detections(frame, tracker, team_classifier, pitch)

            all_player_detections = data["player_goalkeeper_detections"]
            all_player_detections.class_id = all_player_detections.class_id.astype(int)

            labels = [str(tid) for tid in all_player_detections.tracker_id]

            conf_filter = data["key_points"].confidence[0] > 0.5
            frame_points = data["key_points"].xy[0][conf_filter]
            pitch_points = np.array(pitch.config.vertices)[conf_filter]

            transformer = pitch.get_transform(pitch_points, frame_points)
            transformed_keypoints = sv.KeyPoints(xy=transformer(np.array(pitch.config.vertices))[np.newaxis])

            annotated = player_annotator.annotate(frame, detections=all_player_detections)
            annotated = edge_annotator.annotate(annotated, key_points=transformed_keypoints)
            annotated = referee_annotator.annotate(annotated, detections=data["referee_detections"])
            annotated = label_annotator.annotate(annotated, detections=all_player_detections, labels=labels)
            annotated = triangle_annotator.annotate(annotated, detections=data["ball_detections"])
            return vertex_annotator.annotate(annotated, key_points=data["key_points"])

        return callback

    def homography(self, source_path: str, stride: int = 1, only_homography: bool = True):
        pitch = Pitch()
        tracker = sv.ByteTrack()

        crops = self.crop(source_path, stride)
        team_classifier = TeamClassifier()
        team_classifier.fit(crops)

        def callback(frame: np.ndarray, _: int) -> np.ndarray:
            data = self.__process_frame_with_detections(frame, tracker, team_classifier, pitch)

            def transform_and_draw(detections, color_hex, radius=16, image=None):
                coords = data["transformer"](detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER))
                return pitch.draw_points(coords, face_color=sv.Color.from_hex(color_hex),
                                         edge_color=sv.Color.BLACK, radius=radius, image=image)

            image = pitch.draw_points(
                data["transformer"](data["ball_detections"].get_anchors_coordinates(sv.Position.BOTTOM_CENTER)),
                face_color=sv.Color.WHITE, edge_color=sv.Color.BLACK
            )
            image = transform_and_draw(
                data["player_goalkeeper_detections"][data["player_goalkeeper_detections"].class_id == 0],
                '00BFFF', image=image)
            image = transform_and_draw(
                data["player_goalkeeper_detections"][data["player_goalkeeper_detections"].class_id == 1],
                'FF1493', image=image)
            image = transform_and_draw(data["referee_detections"], 'FFD700', image=image)

            h, w, _ = frame.shape
            if only_homography:
                return sv.resize_image(image, (w, h))

            image = sv.resize_image(image, (w // 2, h // 2))
            radar_h, radar_w, _ = image.shape
            rect = sv.Rect(x=w // 2 - radar_w // 2, y=h - radar_h, width=radar_w, height=radar_h)
            return sv.draw_image(frame.copy(), image, 1, rect)

        return callback

    def crop(self, source_path: str, stride: int = 1):
        crops = []
        for frame in sv.get_video_frames_generator(source_path, stride):
            results = self.__predict(self.__detection_model, frame)
            detections = sv.Detections.from_ultralytics(results)
            detections = detections[detections.class_id == settings.PLAYER_CLASS_ID]

            crops += [sv.crop_image(frame, xyxy) for xyxy in detections.xyxy]

        return crops

    def __callback(self, type: str, **kwargs):
        if type == "detection":
            return self.detection(**kwargs)
        elif type == "homography":
            return self.homography(**kwargs)
        raise ValueError(f"Invalid callback type: {type}. Supported types are: detection")

    def process(self, source_path: str, target_path: str, callback: str, **kwargs):
        return sv.process_video(source_path=source_path, target_path=target_path,
                                callback=self.__callback(callback, source_path=source_path, **kwargs))

    def process_one_frame(self, source_path: str, target_path: str, callback: str, **kwargs):
        for frame in sv.get_video_frames_generator(source_path):
            cb = self.__callback(callback, source_path=source_path, **kwargs)
            sv.plot_image(cb(frame, 0))

    def process_real_time(self, source: str, callback: str, stride: int = 1, **kwargs):
        cap = cv2.VideoCapture(source)
        cb = self.__callback(callback, source_path=source, stride=stride, **kwargs)

        frame_count = 0
        skip_frame = 2

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % skip_frame != 0:
                continue

            annotated_frame = cb(frame, frame_count)
            cv2.imshow("Football", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    inference = Inference("/Users/vishwa/Documents/Projects/football/model/detection.pt",
                          "/Users/vishwa/Documents/Projects/football/model/keypoint.pt")

    inference.process_real_time("/Users/vishwa/Documents/Projects/football/model/tests/test.mp4", "detection", 60)

    # inference.process_one_frame(
    #     source_path="/Users/vishwa/Documents/Projects/football/model/tests/test.mp4",
    #     target_path="output.mp4",
    #     callback="detection",
    #     stride=60
    # )

    # inference.process(
    #     source_path="/Users/vishwa/Documents/Projects/football/model/tests/t.mp4",
    #     target_path="output.mp4",
    #     callback="homography",
    #     stride=60,
    #     only_homography=False
    # )

    # a = inference.crop(
    #     source_path="/Users/vishwa/Documents/Projects/football/model/tests/test.mp4",
    #     stride=60
    # )
    # print(len(a))
