import math
import time
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import config


HAND_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),  # thumb
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),  # index
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),  # middle
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),  # ring
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),  # pinky
    (0, 17),
)


class HandTracker:
    def __init__(self) -> None:
        base_options = python.BaseOptions(model_asset_path=config.MODEL_PATH)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.7,
            min_tracking_confidence=0.7,
        )

        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._results: Optional[vision.HandLandmarkerResult] = None

        self._start_time = time.time()
        self._prev_pinch: Optional[tuple[float, float]] = None

    def close(self) -> None:
        if self._landmarker is not None:
            self._landmarker.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _timestamp_ms(self) -> int:
        return int((time.time() - self._start_time) * 1000.0)

    def _draw_hand(self, img: np.ndarray, hand_landmarks) -> None:
        h, w = img.shape[0], img.shape[1]
        pts: list[tuple[int, int]] = []
        for lm in hand_landmarks:
            pts.append((int(lm.x * w), int(lm.y * h)))

        for a, b in HAND_CONNECTIONS:
            if a < len(pts) and b < len(pts):
                cv2.line(img, pts[a], pts[b], (0, 255, 0), 2, cv2.LINE_AA)

        for x, y in pts:
            cv2.circle(img, (x, y), 4, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(img, (x, y), 4, (0, 0, 0), 1, cv2.LINE_AA)

    def find_hands(self, img: np.ndarray, draw: bool = True) -> np.ndarray:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

        self._results = self._landmarker.detect_for_video(mp_image, self._timestamp_ms())

        if draw and self._results and self._results.hand_landmarks:
            for hand_landmarks in self._results.hand_landmarks:
                self._draw_hand(img, hand_landmarks)

        return img

    def find_position(self, img: np.ndarray, hand_no: int = 0) -> list[list[int]]:
        lm_list: list[list[int]] = []
        if not self._results or not self._results.hand_landmarks:
            return lm_list

        if hand_no < 0 or hand_no >= len(self._results.hand_landmarks):
            return lm_list

        hand_landmarks = self._results.hand_landmarks[hand_no]
        h, w = img.shape[0], img.shape[1]
        for idx, lm in enumerate(hand_landmarks):
            cx = int(lm.x * w)
            cy = int(lm.y * h)
            lm_list.append([idx, cx, cy])
        return lm_list

    def _smooth_point(self, raw: tuple[float, float]) -> tuple[int, int]:
        alpha = float(config.SMOOTHING_FACTOR)
        if self._prev_pinch is None:
            self._prev_pinch = raw
            return int(raw[0]), int(raw[1])

        px, py = self._prev_pinch
        sx = alpha * raw[0] + (1.0 - alpha) * px
        sy = alpha * raw[1] + (1.0 - alpha) * py
        self._prev_pinch = (sx, sy)
        return int(sx), int(sy)

    def is_pinching(self, lm_list: list[list[int]]) -> tuple[bool, tuple[int, int]]:
        if len(lm_list) < 9:
            self._prev_pinch = None
            return False, (0, 0)

        x1, y1 = lm_list[4][1], lm_list[4][2]
        x2, y2 = lm_list[8][1], lm_list[8][2]

        length = math.hypot(x2 - x1, y2 - y1)
        if length >= float(config.PINCH_THRESHOLD):
            self._prev_pinch = None
            return False, (0, 0)

        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        return True, self._smooth_point((cx, cy))
