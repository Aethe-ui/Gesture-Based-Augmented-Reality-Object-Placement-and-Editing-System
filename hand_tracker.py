import cv2
import mediapipe as mp
import math
import config
import numpy as np

# Import Tasks API
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class HandTracker:
    def __init__(self, mode=False, max_hands=2, detection_confidence=0.7, track_confidence=0.7):
        # Initialize HandLandmarker
        base_options = python.BaseOptions(model_asset_path=config.MODEL_PATH)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_confidence,
            min_hand_presence_confidence=track_confidence,
            min_tracking_confidence=track_confidence)
        
        self.detector = vision.HandLandmarker.create_from_options(options)
        self.results = None
        self.timestamp_ms = 0

        # Connections for manual drawing
        self.connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),           # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),           # Index
            (0, 9), (9, 10), (10, 11), (11, 12),      # Middle
            (0, 13), (13, 14), (14, 15), (15, 16),    # Ring
            (0, 17), (17, 18), (18, 19), (19, 20),    # Pinky
            (5, 9), (9, 13), (13, 17)                 # Palm
        ]

    def find_hands(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        
        # Increase timestamp for video mode
        self.timestamp_ms += int(1000 / 30) # approx 30fps
        # Or meaningful timestamp from cap
        
        self.results = self.detector.detect_for_video(mp_image, self.timestamp_ms)

        if self.results.hand_landmarks:
            for hand_lms in self.results.hand_landmarks:
                if draw:
                    self.draw_custom_landmarks(img, hand_lms)
        return img

    def draw_custom_landmarks(self, img, landmarks):
        h, w, c = img.shape
        # Convert landmarks to pixel coordinates
        points = []
        for lm in landmarks:
            cx, cy = int(lm.x * w), int(lm.y * h)
            points.append((cx, cy))
            cv2.circle(img, (cx, cy), 5, config.COLOR_RED, cv2.FILLED)
            
        # Draw connections
        for start_idx, end_idx in self.connections:
            if start_idx < len(points) and end_idx < len(points):
                cv2.line(img, points[start_idx], points[end_idx], config.COLOR_WHITE, 2)

    def find_position(self, img, hand_no=0):
        lm_list = []
        if self.results and self.results.hand_landmarks:
            if len(self.results.hand_landmarks) > hand_no:
                my_hand = self.results.hand_landmarks[hand_no]
                h, w, c = img.shape
                for id, lm in enumerate(my_hand):
                    # NormalizedLandmark has x, y, z
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lm_list.append([id, cx, cy])
        return lm_list

    def is_pinching(self, lm_list):
        """
        Check if thumb (4) and index finger (8) are pinching.
        Returns: (True/False, (cx, cy) midpoint)
        """
        if len(lm_list) < 9:
            return False, (0, 0)

        x1, y1 = lm_list[4][1], lm_list[4][2] # Thumb tip
        x2, y2 = lm_list[8][1], lm_list[8][2] # Index tip
        
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        length = math.hypot(x2 - x1, y2 - y1)

        if length < config.PINCH_THRESHOLD:
            return True, (cx, cy)
        return False, (0, 0)
