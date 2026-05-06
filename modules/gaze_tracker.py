"""
Suivi du regard via MediaPipe FaceLandmarker + solvePnP.
"""

import os

import cv2
import numpy as np

from .config import (
    CHIN, FACE_3D_MODEL, FACE_CONFIDENCE, LEFT_EYE_CORNER, LEFT_IRIS_CENTER,
    LEFT_MOUTH, NOSE_TIP, PITCH_THRESHOLD, RIGHT_EYE_CORNER,
    RIGHT_IRIS_CENTER, RIGHT_MOUTH, YAW_THRESHOLD,
)


class GazeTracker:
    """Tracks head pose + iris using MediaPipe FaceLandmarker + solvePnP."""

    MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    MODEL_PATH = "face_landmarker.task"

    def __init__(self, confidence: float = FACE_CONFIDENCE):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        self._mp = mp

        # Auto-download model if not present
        if not os.path.isfile(self.MODEL_PATH):
            print("[...] Downloading MediaPipe FaceLandmarker model...")
            import urllib.request
            urllib.request.urlretrieve(self.MODEL_URL, self.MODEL_PATH)
            print(f"[OK] Model downloaded: {self.MODEL_PATH}")

        base_options = BaseOptions(model_asset_path=self.MODEL_PATH)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1,
            min_face_detection_confidence=confidence,
            min_face_presence_confidence=confidence,
        )
        self._detector = vision.FaceLandmarker.create_from_options(options)
        print("[OK] MediaPipe FaceLandmarker loaded.")

    def _compute_head_pose(self, lm, h: int, w: int) -> tuple:
        """Compute yaw and pitch angles using solvePnP."""
        # 2D image points from face landmarks
        face_2d = np.array([
            [lm[NOSE_TIP].x * w, lm[NOSE_TIP].y * h],
            [lm[CHIN].x * w, lm[CHIN].y * h],
            [lm[LEFT_EYE_CORNER].x * w, lm[LEFT_EYE_CORNER].y * h],
            [lm[RIGHT_EYE_CORNER].x * w, lm[RIGHT_EYE_CORNER].y * h],
            [lm[LEFT_MOUTH].x * w, lm[LEFT_MOUTH].y * h],
            [lm[RIGHT_MOUTH].x * w, lm[RIGHT_MOUTH].y * h],
        ], dtype=np.float64)

        # Camera matrix (approximate with frame dimensions)
        focal_length = w
        center = (w / 2, h / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1],
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        success, rvec, tvec = cv2.solvePnP(
            FACE_3D_MODEL, face_2d, camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            return 0.0, 0.0

        # Convert rotation vector to rotation matrix, then to Euler angles
        rmat, _ = cv2.Rodrigues(rvec)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

        yaw = angles[1]    # horizontal turn (left/right)
        pitch = angles[0]  # vertical tilt (up/down)

        # Normalize angles: RQDecomp3x3 can wrap around ±180°
        # e.g. pitch=-170 when facing forward → should be +10
        if pitch > 90:
            pitch = pitch - 180
        elif pitch < -90:
            pitch = pitch + 180
        if yaw > 90:
            yaw = yaw - 180
        elif yaw < -90:
            yaw = yaw + 180

        return yaw, pitch

    def analyze(self, frame: np.ndarray) -> dict:
        """
        Analyze head pose and iris position.

        Returns dict with:
            - 'face_detected': bool
            - 'looking_at_screen': bool (based on head pose)
            - 'yaw': float (degrees, - = left, + = right)
            - 'pitch': float (degrees, - = down, + = up)
            - 'iris_pts': dict with left_iris/right_iris pixel coords
            - 'nose_pt': (x, y) nose tip pixel coords
            - 'face_bbox': (x, y, w, h) or None
        """
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._detector.detect(mp_image)

        info = {
            "face_detected": False,
            "looking_at_screen": False,
            "yaw": 0.0,
            "pitch": 0.0,
            "iris_pts": None,
            "nose_pt": None,
            "face_bbox": None,
        }

        if not result.face_landmarks:
            return info

        lm = result.face_landmarks[0]
        info["face_detected"] = True

        # Head pose estimation
        yaw, pitch = self._compute_head_pose(lm, h, w)
        info["yaw"] = yaw
        info["pitch"] = pitch

        # Looking at screen if head is roughly facing forward
        info["looking_at_screen"] = abs(yaw) < YAW_THRESHOLD and abs(pitch) < PITCH_THRESHOLD

        # Nose tip for drawing
        info["nose_pt"] = (int(lm[NOSE_TIP].x * w), int(lm[NOSE_TIP].y * h))

        # Iris positions for visual feedback
        if len(lm) > RIGHT_IRIS_CENTER:
            info["iris_pts"] = {
                "left": (int(lm[LEFT_IRIS_CENTER].x * w), int(lm[LEFT_IRIS_CENTER].y * h)),
                "right": (int(lm[RIGHT_IRIS_CENTER].x * w), int(lm[RIGHT_IRIS_CENTER].y * h)),
            }

        # Compute face bbox from all landmarks
        xs = [int(l.x * w) for l in lm]
        ys = [int(l.y * h) for l in lm]
        info["face_bbox"] = (min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))

        return info

    def cleanup(self):
        self._detector.close()
