from typing import List, Tuple

import cv2
import numpy as np
import supervision as sv

from core.pitch_config import PitchType, PitchConfig


class Pitch:
    def __init__(
            self,
            pitch_type: PitchType = PitchType.DEFAULT,
            scale: float = 0.1,
            padding: int = 50,
            line_thickness: int = 4,
            point_radius: int = 8
    ):
        self.config = PitchConfig.from_type(pitch_type)
        self.scale = scale
        self.padding = padding
        self.line_thickness = line_thickness
        self.point_radius = point_radius
        self.vertices_scaled = [
            (int(x * scale) + padding, int(y * scale) + padding)
            for x, y in self.config.vertices
        ]
        self.__pitch_img = self.__draw_base_pitch()

    def __draw_base_pitch(self):
        w, l = int(self.config.WIDTH * self.scale), int(self.config.LENGTH * self.scale)
        img = np.ones((w + 2 * self.padding, l + 2 * self.padding, 3), dtype=np.uint8)
        img *= np.array(sv.Color(34, 139, 34).as_bgr(), dtype=np.uint8)

        for start, end in self.config.EDGES:
            pt1, pt2 = self.vertices_scaled[start - 1], self.vertices_scaled[end - 1]
            cv2.line(img, pt1, pt2, color=sv.Color.WHITE.as_bgr(), thickness=self.line_thickness)

        center = (l // 2 + self.padding, w // 2 + self.padding)
        radius = int(self.config.CENTRE_CIRCLE_RADIUS * self.scale)
        cv2.circle(img, center, radius, sv.Color.WHITE.as_bgr(), thickness=self.line_thickness)

        for x in [
            self.config.PENALTY_SPOT_DISTANCE,
            self.config.LENGTH - self.config.PENALTY_SPOT_DISTANCE
        ]:
            pt = (
                int(x * self.scale) + self.padding,
                w // 2 + self.padding
            )
            cv2.circle(img, pt, self.point_radius, sv.Color.WHITE.as_bgr(), thickness=-1)

        return img

    @property
    def image(self):
        return self.__pitch_img.copy()

    def draw(self):
        return self.image

    def draw_points(
            self,
            points: List[Tuple[float, float]],
            face_color=sv.Color.RED,
            edge_color=sv.Color.BLACK,
            radius: int = 10,
            thickness: int = 4,
            image=None
    ):
        img = self.__pitch_img.copy() if image is None else image.copy()

        for x, y in points:
            pt = (int(x * self.scale) + self.padding, int(y * self.scale) + self.padding)
            cv2.circle(img, pt, radius, face_color.as_bgr(), thickness=-1)
            cv2.circle(img, pt, radius, edge_color.as_bgr(), thickness=thickness)

        return img

    def draw_voronoi(
            self,
            team_1_xy: np.ndarray,
            team_2_xy: np.ndarray,
            team_1_color=sv.Color.RED,
            team_2_color=sv.Color.BLUE,
            opacity=0.5,
            image=None,
    ):
        img = self.__pitch_img.copy() if image is None else image.copy()

        scaled_width = int(self.scale * self.config.WIDTH)
        scaled_length = int(self.scale * self.config.LENGTH)

        voronoi = np.zeros_like(img, dtype=np.uint8)

        team_1_color_bg = np.array(team_1_color.as_bgr(), dtype=np.uint8)
        team_2_color_bg = np.array(team_2_color.as_bgr(), dtype=np.uint8)

        y_coordinates, x_coordinates = np.indices((
            scaled_width + 2 * self.padding,
            scaled_length + 2 * self.padding
        ))

        y_coordinates -= self.padding
        x_coordinates -= self.padding

        def calculate_distances(xy, x, y):
            return np.sqrt((xy[:, 0][:, None, None] * self.scale - x) ** 2 +
                           (xy[:, 1][:, None, None] * self.scale - y) ** 2)

        distances_team_1 = calculate_distances(team_1_xy, x_coordinates, y_coordinates)
        distances_team_2 = calculate_distances(team_2_xy, x_coordinates, y_coordinates)

        min_distances_team_1 = np.min(distances_team_1, axis=0)
        min_distances_team_2 = np.min(distances_team_2, axis=0)

        control_mask = min_distances_team_1 < min_distances_team_2

        voronoi[control_mask] = team_1_color_bg
        voronoi[~control_mask] = team_2_color_bg

        overlay = cv2.addWeighted(voronoi, opacity, img, 1 - opacity, 0)

        return overlay


    def get_transform(self, source: np.ndarray, target: np.ndarray = None, mask: np.ndarray = None) -> callable:
        if target is None:
            target = np.array(self.config.vertices)

        if mask is not None:
            target = target[mask]
            source = source[mask]

        target = target.astype(np.float32)
        source = source.astype(np.float32)

        print("Source", source.shape, "Target", target.shape)

        if source.shape != target.shape or source.shape[1] != 2:
            raise ValueError("Source and target must have the same shape and be 2D coordinates.")

        m, _ = cv2.findHomography(source, target)
        if m is None:
            raise ValueError("Homography matrix could not be calculated.")

        def transform(pts: np.ndarray) -> np.ndarray:
            if pts.size == 0:
                return pts
            return cv2.perspectiveTransform(pts.reshape(-1, 1, 2).astype(np.float32), m).reshape(-1, 2).astype(
                np.float32)

        return transform


class PitchTransformer:
    def __init__(self, source, target):
        if source.shape != target.shape:
            raise ValueError("Source and target must have the same shape.")
        if source.shape[1] != 2:
            raise ValueError("Source and target points must be 2D coordinates.")

        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m, _ = cv2.findHomography(source, target)
        if self.m is None:
            raise ValueError("Homography matrix could not be calculated.")

    def transform_points(self, points):
        if points.size == 0:
            return points

        if points.shape[1] != 2:
            raise ValueError("Points must be 2D coordinates.")

        return cv2.perspectiveTransform(points.reshape(-1, 1, 2).astype(np.float32), self.m).reshape(-1, 2).astype(
            np.float32)
