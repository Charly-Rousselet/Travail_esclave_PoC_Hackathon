"""
Détection du geste de pincement (pouce + index) via MediaPipe HandLandmarker.
"""

import os
import time
import cv2
import numpy as np


class PinchDetector:
    """Detects pinch gesture using scale-invariant 3D ratio."""

    MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    MODEL_PATH = "hand_landmarker.task"
    WRIST = 0
    INDEX_MCP = 5
    THUMB_TIP = 4
    INDEX_TIP = 8
    PINCH_RATIO_THRESHOLD = 0.25
    PINCH_CONFIRM_FRAMES = 5
    PINCH_COOLDOWN = 1.5

    def __init__(self):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision
        self._mp = mp
        if not os.path.isfile(self.MODEL_PATH):
            print("[...] Downloading MediaPipe HandLandmarker model...")
            import urllib.request
            urllib.request.urlretrieve(self.MODEL_URL, self.MODEL_PATH)
            print(f"[OK] Model downloaded: {self.MODEL_PATH}")
        base_options = BaseOptions(model_asset_path=self.MODEL_PATH)
        options = vision.HandLandmarkerOptions(
            base_options=base_options, num_hands=1,
            min_hand_detection_confidence=0.5, min_hand_presence_confidence=0.5,
        )
        self._detector = vision.HandLandmarker.create_from_options(options)
        self._last_toggle_time = 0.0
        self._consecutive_pinch_frames = 0
        print("[OK] MediaPipe HandLandmarker (pinch) loaded.")

    def detect(self, frame: np.ndarray) -> dict:
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._detector.detect(mp_image)
        info = {"pinch_toggled": False, "pinch_pts": None, "is_pinching": False, "hand_landmarks": []}
        if not result.hand_landmarks:
            self._consecutive_pinch_frames = 0
            return info
        for hand_lm in result.hand_landmarks:
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lm]
            info["hand_landmarks"].append(pts)
        hand_lm = result.hand_landmarks[0]
        info["pinch_pts"] = (
            (int(hand_lm[self.THUMB_TIP].x * w), int(hand_lm[self.THUMB_TIP].y * h)),
            (int(hand_lm[self.INDEX_TIP].x * w), int(hand_lm[self.INDEX_TIP].y * h))
        )
        thumb = hand_lm[self.THUMB_TIP]
        index = hand_lm[self.INDEX_TIP]
        pinch_dist = np.sqrt((thumb.x-index.x)**2 + (thumb.y-index.y)**2 + (thumb.z-index.z)**2)
        wrist = hand_lm[self.WRIST]
        index_mcp = hand_lm[self.INDEX_MCP]
        palm_size = np.sqrt((wrist.x-index_mcp.x)**2 + (wrist.y-index_mcp.y)**2 + (wrist.z-index_mcp.z)**2)
        pinch_ratio = pinch_dist / (palm_size + 1e-6)
        is_pinching = pinch_ratio < self.PINCH_RATIO_THRESHOLD
        info["is_pinching"] = is_pinching
        if is_pinching:
            self._consecutive_pinch_frames += 1
        else:
            self._consecutive_pinch_frames = 0
        now = time.time()
        if self._consecutive_pinch_frames == self.PINCH_CONFIRM_FRAMES:
            if now - self._last_toggle_time > self.PINCH_COOLDOWN:
                info["pinch_toggled"] = True
                self._last_toggle_time = now
        return info

    def cleanup(self):
        self._detector.close()
