#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════╗
║              🔥 EXTREME PRODUCTIVITY ENFORCER 🔥                ║
║                                                                  ║
║  Surveille l'utilisateur via webcam et déclenche une alarme      ║
║  sonore lorsqu'un téléphone est détecté (3s) ou que              ║
║  l'attention est perdue (20s).                                   ║
╚══════════════════════════════════════════════════════════════════╝

Usage:
    python main.py [--alarm ALARM_PATH] [--camera CAMERA_ID]

Appuyer sur 'q' pour quitter.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# ─────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────

PHONE_TRIGGER_DURATION = 3.0    # seconds of continuous phone detection
ATTENTION_TRIGGER_DURATION = 20.0  # seconds of continuous inattention
YOLO_CONFIDENCE = 0.45          # YOLO detection confidence threshold
FACE_CONFIDENCE = 0.5           # MediaPipe face detection confidence
CELL_PHONE_CLASS_ID = 67        # COCO class ID for "cell phone"

# Head pose thresholds (degrees)
YAW_THRESHOLD = 30.0            # max horizontal head turn before "distracted"
PITCH_THRESHOLD = 15.0          # max vertical head tilt before "distracted"

# Pinch detection (thumb + index)
PINCH_DISTANCE_THRESHOLD = 50   # max pixel distance between thumb & index tips
PINCH_COOLDOWN = 1.5            # seconds between toggle events
COLOR_HAND = (255, 180, 0)      # hand landmarks color

# Visual overlay colors (BGR)
COLOR_SAFE = (0, 200, 100)
COLOR_WARNING = (0, 180, 255)
COLOR_DANGER = (0, 0, 255)
COLOR_PHONE_BOX = (0, 100, 255)
COLOR_IRIS = (0, 255, 0)
COLOR_IRIS_BAD = (0, 0, 255)
COLOR_BG_PANEL = (30, 30, 30)

# Key face landmark indices for head pose (MediaPipe 478 landmarks)
NOSE_TIP = 1
CHIN = 152
LEFT_EYE_CORNER = 33
RIGHT_EYE_CORNER = 263
LEFT_MOUTH = 61
RIGHT_MOUTH = 291

# Iris landmark indices
LEFT_IRIS_CENTER = 468
RIGHT_IRIS_CENTER = 473

# 3D model points for a generic face (used by solvePnP)
FACE_3D_MODEL = np.array([
    [0.0, 0.0, 0.0],          # Nose tip
    [0.0, -330.0, -65.0],     # Chin
    [-225.0, 170.0, -135.0],  # Left eye corner
    [225.0, 170.0, -135.0],   # Right eye corner
    [-150.0, -150.0, -125.0], # Left mouth corner
    [150.0, -150.0, -125.0],  # Right mouth corner
], dtype=np.float64)


class VolumeController:
    """macOS system volume controller using osascript."""

    @staticmethod
    def set_max_volume():
        """Force system output volume to 100%."""
        try:
            subprocess.run(
                ["osascript", "-e", "set volume output volume 100"],
                check=True,
                capture_output=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"[WARN] Could not set volume: {e}")


class AlarmPlayer:
    """Handles alarm sound playback using pygame.mixer."""

    def __init__(self, alarm_path: str):
        import pygame

        self._pygame = pygame
        self._pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)

        if not os.path.isfile(alarm_path):
            print(f"[INFO] Alarm file not found at '{alarm_path}', generating default beep...")
            alarm_path = self._generate_default_alarm(alarm_path)

        self._sound = self._pygame.mixer.Sound(alarm_path)
        self._sound.set_volume(1.0)
        self._playing = False
        print(f"[OK] Alarm loaded: {alarm_path}")

    def _generate_default_alarm(self, path: str) -> str:
        """Generate a loud, aggressive alarm .wav file if no mp3 is provided."""
        import wave
        import struct
        import math

        sample_rate = 44100
        duration = 3.0
        wav_path = path.replace(".mp3", ".wav") if path.endswith(".mp3") else path + ".wav"

        samples = []
        num_samples = int(sample_rate * duration)

        for i in range(num_samples):
            t = i / sample_rate
            freq1 = 880 + 200 * math.sin(2 * math.pi * 3 * t)
            freq2 = 1200 + 300 * math.sin(2 * math.pi * 5 * t)
            sample = 0.5 * math.sin(2 * math.pi * freq1 * t) + \
                     0.3 * math.sin(2 * math.pi * freq2 * t) + \
                     0.2 * math.sin(2 * math.pi * 440 * t)
            envelope = 0.5 + 0.5 * math.sin(2 * math.pi * 8 * t)
            sample *= envelope
            samples.append(int(sample * 32767 * 0.9))

        with wave.open(wav_path, "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(struct.pack(f"<{len(samples)}h", *samples))

        print(f"[OK] Generated default alarm: {wav_path}")
        return wav_path

    def start_loop(self):
        """Start playing alarm in infinite loop."""
        if not self._playing:
            self._sound.play(loops=-1)
            self._playing = True

    def stop(self):
        """Stop the alarm sound."""
        if self._playing:
            self._sound.stop()
            self._playing = False

    @property
    def is_playing(self) -> bool:
        return self._playing

    def cleanup(self):
        """Clean up pygame mixer."""
        self.stop()
        self._pygame.mixer.quit()


class PhoneDetector:
    """Detects cell phones in frames using YOLOv8 nano."""

    def __init__(self, confidence: float = YOLO_CONFIDENCE):
        from ultralytics import YOLO

        print("[...] Loading YOLOv8 nano model...")
        self._model = YOLO("yolov8n.pt")
        self._confidence = confidence
        print("[OK] YOLOv8 nano loaded.")

    def detect(self, frame: np.ndarray) -> list:
        results = self._model(
            frame,
            conf=self._confidence,
            classes=[CELL_PHONE_CLASS_ID],
            verbose=False,
        )
        detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0])
                detections.append((x1, y1, x2, y2, conf))
        return detections


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


class PinchDetector:
    """Detects pinch gesture (thumb + index) using MediaPipe HandLandmarker."""

    MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    MODEL_PATH = "hand_landmarker.task"
    THUMB_TIP = 4
    INDEX_TIP = 8

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
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
        )
        self._detector = vision.HandLandmarker.create_from_options(options)
        self._last_toggle_time = 0.0
        self._prev_pinching = False
        print("[OK] MediaPipe HandLandmarker (pinch) loaded.")

    def detect(self, frame: np.ndarray) -> dict:
        """
        Detect a hand and check for pinch gesture (thumb + index touching).

        Returns dict with:
            - 'pinch_toggled': bool (True on the frame pinch is first detected)
            - 'pinch_pts': tuple ((thumb_x,thumb_y), (index_x,index_y)) or None
            - 'is_pinching': bool
            - 'hand_landmarks': list of landmark lists for drawing
        """
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._detector.detect(mp_image)

        info = {"pinch_toggled": False, "pinch_pts": None,
                "is_pinching": False, "hand_landmarks": []}

        if not result.hand_landmarks:
            self._prev_pinching = False
            return info

        # Store landmarks for drawing
        for hand_lm in result.hand_landmarks:
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lm]
            info["hand_landmarks"].append(pts)

        # Check pinch on the first detected hand
        hand_lm = result.hand_landmarks[0]
        thumb = (int(hand_lm[self.THUMB_TIP].x * w), int(hand_lm[self.THUMB_TIP].y * h))
        index = (int(hand_lm[self.INDEX_TIP].x * w), int(hand_lm[self.INDEX_TIP].y * h))
        info["pinch_pts"] = (thumb, index)

        dist = np.sqrt((thumb[0] - index[0])**2 + (thumb[1] - index[1])**2)
        is_pinching = dist < PINCH_DISTANCE_THRESHOLD
        info["is_pinching"] = is_pinching

        now = time.time()
        # Toggle on rising edge (fingers just came together)
        if is_pinching and not self._prev_pinching:
            if now - self._last_toggle_time > PINCH_COOLDOWN:
                info["pinch_toggled"] = True
                self._last_toggle_time = now

        self._prev_pinching = is_pinching
        return info

    def cleanup(self):
        self._detector.close()


class OverlayRenderer:
    """Renders debug overlays on the video frame."""

    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_BOLD = cv2.FONT_HERSHEY_DUPLEX

    @staticmethod
    def draw_hands(frame: np.ndarray, pinch_info: dict):
        """Draw hand landmarks and pinch indicator."""
        for hand_pts in pinch_info.get("hand_landmarks", []):
            # Draw connections between key landmarks
            connections = [
                (0, 1), (1, 2), (2, 3), (3, 4),    # thumb
                (0, 5), (5, 6), (6, 7), (7, 8),    # index
                (5, 9), (9, 10), (10, 11), (11, 12), # middle
                (9, 13), (13, 14), (14, 15), (15, 16), # ring
                (13, 17), (17, 18), (18, 19), (19, 20), # pinky
                (0, 17),  # palm
            ]
            for i, j in connections:
                if i < len(hand_pts) and j < len(hand_pts):
                    cv2.line(frame, hand_pts[i], hand_pts[j], COLOR_HAND, 1, cv2.LINE_AA)
            for pt in hand_pts:
                cv2.circle(frame, pt, 2, (255, 255, 255), -1)

        # Highlight thumb-index pinch points
        if pinch_info.get("pinch_pts"):
            thumb, index = pinch_info["pinch_pts"]
            color = COLOR_SAFE if pinch_info["is_pinching"] else COLOR_HAND
            cv2.circle(frame, thumb, 6, color, -1)
            cv2.circle(frame, index, 6, color, -1)
            cv2.line(frame, thumb, index, color, 2, cv2.LINE_AA)

    @staticmethod
    def draw_paused_overlay(frame: np.ndarray):
        """Draw a PAUSED banner across the screen."""
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h // 2 - 40), (w, h // 2 + 40), (50, 50, 50), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        text = "PAUSED  (pinch to resume)"
        (tw, th), _ = cv2.getTextSize(text, OverlayRenderer.FONT_BOLD, 1.2, 2)
        cv2.putText(frame, text, ((w - tw) // 2, h // 2 + th // 2),
                    OverlayRenderer.FONT_BOLD, 1.2, (100, 200, 255), 2, cv2.LINE_AA)

    @staticmethod
    def draw_phone_boxes(frame: np.ndarray, detections: list):
        for (x1, y1, x2, y2, conf) in detections:
            cv2.rectangle(frame, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), COLOR_DANGER, 3)
            cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_PHONE_BOX, 2)
            label = f"PHONE {conf:.0%}"
            (tw, th), _ = cv2.getTextSize(label, OverlayRenderer.FONT, 0.6, 1)
            cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 8, y1), COLOR_PHONE_BOX, -1)
            cv2.putText(frame, label, (x1 + 4, y1 - 5),
                        OverlayRenderer.FONT, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    @staticmethod
    def draw_gaze_info(frame: np.ndarray, gaze: dict):
        """Draw head pose info, iris dots, and face bbox."""
        if not gaze["face_detected"]:
            # No face — show warning
            cv2.putText(frame, "NO FACE DETECTED", (frame.shape[1] // 2 - 140, 50),
                        OverlayRenderer.FONT_BOLD, 0.7, COLOR_DANGER, 2, cv2.LINE_AA)
            return

        looking = gaze["looking_at_screen"]
        iris_color = COLOR_IRIS if looking else COLOR_IRIS_BAD

        # Draw iris dots
        if gaze["iris_pts"]:
            for key in ("left", "right"):
                pt = gaze["iris_pts"][key]
                cv2.circle(frame, pt, 4, iris_color, -1)
                cv2.circle(frame, pt, 6, (255, 255, 255), 1)

        # Nose dot
        if gaze["nose_pt"]:
            cv2.circle(frame, gaze["nose_pt"], 3, (255, 200, 0), -1)

        # Face bounding box
        if gaze["face_bbox"]:
            bx, by, bw, bh = gaze["face_bbox"]
            box_color = COLOR_SAFE if looking else COLOR_WARNING
            cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), box_color, 1)

        # Head pose label
        label = "FOCUSED" if looking else "DISTRACTED"
        color = COLOR_SAFE if looking else COLOR_DANGER
        yaw, pitch = gaze["yaw"], gaze["pitch"]
        txt = f"{label} (yaw:{yaw:+.0f} pitch:{pitch:+.0f})"
        if gaze["face_bbox"]:
            bx, by, _, _ = gaze["face_bbox"]
            cv2.putText(frame, txt, (bx, by - 8),
                        OverlayRenderer.FONT, 0.45, color, 1, cv2.LINE_AA)

    @staticmethod
    def draw_status_panel(frame, phone_elapsed, attention_elapsed, state, fps, alarm_reason=""):
        h, w = frame.shape[:2]
        panel_h = 140
        panel_w = 340

        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (10 + panel_w, 10 + panel_h), COLOR_BG_PANEL, -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        cv2.rectangle(frame, (10, 10), (10 + panel_w, 10 + panel_h), (80, 80, 80), 1)

        cv2.putText(frame, "PRODUCTIVITY ENFORCER", (20, 35),
                    OverlayRenderer.FONT_BOLD, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        # Phone timer bar
        phone_ratio = min(phone_elapsed / PHONE_TRIGGER_DURATION, 1.0)
        phone_color = COLOR_DANGER if phone_ratio >= 1.0 else (
            COLOR_WARNING if phone_ratio > 0.3 else COLOR_SAFE)
        cv2.putText(frame, f"Phone: {phone_elapsed:.1f}s / {PHONE_TRIGGER_DURATION:.0f}s",
                    (20, 58), OverlayRenderer.FONT, 0.45, phone_color, 1, cv2.LINE_AA)
        cv2.rectangle(frame, (20, 63), (320, 73), (60, 60, 60), -1)
        cv2.rectangle(frame, (20, 63), (20 + int(300 * phone_ratio), 73), phone_color, -1)

        # Attention timer bar
        att_ratio = min(attention_elapsed / ATTENTION_TRIGGER_DURATION, 1.0)
        att_color = COLOR_DANGER if att_ratio >= 1.0 else (
            COLOR_WARNING if att_ratio > 0.5 else COLOR_SAFE)
        cv2.putText(frame, f"Gaze: {attention_elapsed:.1f}s / {ATTENTION_TRIGGER_DURATION:.0f}s",
                    (20, 92), OverlayRenderer.FONT, 0.45, att_color, 1, cv2.LINE_AA)
        cv2.rectangle(frame, (20, 97), (320, 107), (60, 60, 60), -1)
        cv2.rectangle(frame, (20, 97), (20 + int(300 * att_ratio), 107), att_color, -1)

        # State + FPS
        state_color = COLOR_DANGER if state == "ALARM" else COLOR_SAFE
        cv2.putText(frame, f"[{state}]", (20, 125),
                    OverlayRenderer.FONT_BOLD, 0.5, state_color, 1, cv2.LINE_AA)
        cv2.putText(frame, f"FPS: {fps:.0f}", (280, 125),
                    OverlayRenderer.FONT, 0.4, (150, 150, 150), 1, cv2.LINE_AA)

        if alarm_reason:
            cv2.putText(frame, alarm_reason, (20, 145),
                        OverlayRenderer.FONT, 0.4, COLOR_DANGER, 1, cv2.LINE_AA)

    @staticmethod
    def draw_alarm_flash(frame: np.ndarray, intensity: float = 0.3):
        red_overlay = np.full_like(frame, (0, 0, 255), dtype=np.uint8)
        cv2.addWeighted(red_overlay, intensity, frame, 1.0 - intensity, 0, frame)
        h, w = frame.shape[:2]
        text = "!!! ALARM !!!"
        (tw, th), _ = cv2.getTextSize(text, OverlayRenderer.FONT_BOLD, 2.0, 3)
        cx = (w - tw) // 2
        cy = (h + th) // 2
        cv2.putText(frame, text, (cx, cy),
                    OverlayRenderer.FONT_BOLD, 2.0, (255, 255, 255), 3, cv2.LINE_AA)


class ProductivityEnforcer:
    """Main application controller with Frame Skipping optimization."""

    def __init__(self, alarm_path: str, camera_id: int = 0):
        self._camera_id = camera_id
        self._cap = None

        # Flag de contrôle externe (pour le GUI)
        self.running = False

        # Durées configurables (modifiables par le GUI avant run())
        self.phone_trigger_duration = PHONE_TRIGGER_DURATION
        self.attention_trigger_duration = ATTENTION_TRIGGER_DURATION

        # Sub-modules
        self._phone_detector = PhoneDetector()
        self._gaze_tracker = GazeTracker()
        self._pinch_detector = PinchDetector()
        self._alarm_player = AlarmPlayer(alarm_path)
        self._volume_ctrl = VolumeController()
        self._renderer = OverlayRenderer()

        # Timer state
        self._phone_start: float | None = None
        self._attention_start: float | None = None

        # Alarm state
        self._alarm_active = False
        self._alarm_reason = ""

        # Monitoring enabled state (toggled by pinch)
        self._monitoring_enabled = True

        # FPS & Frame Skipping tracking
        self._fps = 0.0
        self._frame_count = 0        # Compteur global pour les FPS
        self._fps_timer = time.time()
        self._skip_counter = 0       # Compteur pour le Frame Skipping

        # Cache pour mémoriser les dernières détections entre les frames analysées
        self._last_pinch_info = {"pinch_toggled": False, "pinch_pts": None, "is_pinching": False, "hand_landmarks": []}
        self._last_phone_detections = []
        self._last_gaze_info = {
            "face_detected": False, "looking_at_screen": False,
            "yaw": 0.0, "pitch": 0.0, "iris_pts": None,
            "nose_pt": None, "face_bbox": None,
        }

    def _open_camera(self):
        self._cap = cv2.VideoCapture(self._camera_id)
        if not self._cap.isOpened():
            print(f"[ERROR] Cannot open camera {self._camera_id}")
            sys.exit(1)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        print(f"[OK] Camera {self._camera_id} opened.")

    def _update_fps(self):
        self._frame_count += 1
        elapsed = time.time() - self._fps_timer
        if elapsed >= 1.0:
            self._fps = self._frame_count / elapsed
            self._frame_count = 0
            self._fps_timer = time.time()

    def _start_alarm(self, reason: str):
        if not self._alarm_active:
            print(f"\n🚨 ALARM TRIGGERED: {reason}")
            self._alarm_reason = reason
            self._alarm_active = True
            self._volume_ctrl.set_max_volume()
            self._alarm_player.start_loop()

    def _stop_alarm(self):
        if self._alarm_active:
            print("[OK] Condition resolved — alarm stopped.")
            self._alarm_active = False
            self._alarm_reason = ""
            self._alarm_player.stop()
            self._phone_start = None
            self._attention_start = None

    def _process_frame(self, frame: np.ndarray) -> str:
        now = time.time()
        self._skip_counter += 1

        # ── Module 0: Pinch Detection (1 image sur 2) ─────────────
        if self._skip_counter % 2 == 0:
            self._last_pinch_info = self._pinch_detector.detect(frame)
        
        # On utilise le cache pour l'affichage et la logique
        self._renderer.draw_hands(frame, self._last_pinch_info)

        if self._last_pinch_info["pinch_toggled"]:
            self._monitoring_enabled = not self._monitoring_enabled
            status = "ACTIVE" if self._monitoring_enabled else "PAUSED"
            print(f"\n🤏 PINCH DETECTED — Monitoring {status}")
            # Reset du toggle pour éviter qu'il ne se déclenche en boucle sur les frames skippées
            self._last_pinch_info["pinch_toggled"] = False 
            
            if not self._monitoring_enabled:
                self._stop_alarm()
                self._phone_start = None
                self._attention_start = None

        # ── If paused, show overlay and skip detection ────────────
        if not self._monitoring_enabled:
            self._renderer.draw_paused_overlay(frame)
            self._renderer.draw_status_panel(
                frame, 0.0, 0.0, "PAUSED", self._fps, "Pinch to resume"
            )
            return "PAUSED"

        # ── Module 1: Phone Detection YOLO (1 image sur 10) ───────
        if self._skip_counter % 10 == 0:
            self._last_phone_detections = self._phone_detector.detect(frame)
            
        self._renderer.draw_phone_boxes(frame, self._last_phone_detections)

        if self._last_phone_detections:
            if self._phone_start is None:
                self._phone_start = now
        else:
            self._phone_start = None

        phone_elapsed = (now - self._phone_start) if self._phone_start else 0.0

        # ── Module 2: Gaze Tracking (1 image sur 5) ───────────────
        if self._skip_counter % 5 == 0:
            self._last_gaze_info = self._gaze_tracker.analyze(frame)
            
        self._renderer.draw_gaze_info(frame, self._last_gaze_info)

        is_attentive = self._last_gaze_info["face_detected"] and self._last_gaze_info["looking_at_screen"]

        if is_attentive:
            self._attention_start = None
        else:
            if self._attention_start is None:
                self._attention_start = now

        attention_elapsed = (now - self._attention_start) if self._attention_start else 0.0

        # ── Alarm logic: start/stop based on conditions ───────────
        phone_triggered = phone_elapsed >= self.phone_trigger_duration
        attention_triggered = attention_elapsed >= self.attention_trigger_duration

        if phone_triggered:
            self._start_alarm(f"Phone detected for {phone_elapsed:.1f}s")
        elif attention_triggered:
            self._start_alarm(f"Inattention for {attention_elapsed:.1f}s")

        # Stop alarm only when ALL conditions are resolved
        if self._alarm_active and not phone_triggered and not attention_triggered:
            self._stop_alarm()

        state = "ALARM" if self._alarm_active else "MONITORING"

        # ── Draw HUD ──────────────────────────────────────────────
        self._renderer.draw_status_panel(
            frame, phone_elapsed, attention_elapsed, state, self._fps, self._alarm_reason
        )
        if self._alarm_active:
            self._renderer.draw_alarm_flash(frame)

        # Empêcher le compteur d'exploser vers l'infini
        if self._skip_counter >= 1000:
            self._skip_counter = 0

        return state

    def run(self):
        self._open_camera()

        print("\n" + "=" * 55)
        print("  🔥 EXTREME PRODUCTIVITY ENFORCER — ACTIVE 🔥")
        print("  ⚙️  Optimisation Frame Skipping : ACTIVÉE")
        print(f"  Phone trigger:     {PHONE_TRIGGER_DURATION}s")
        print(f"  Attention trigger: {ATTENTION_TRIGGER_DURATION}s")
        print("  Alarm: LOOPS until condition resolved")
        print("  🤏 Pinch (thumb+index) to PAUSE / RESUME")
        print("  Press 'q' to quit.")
        print("=" * 55 + "\n")

        try:
            self.running = True
            while self.running:
                ret, frame = self._cap.read()
                if not ret:
                    print("[ERROR] Failed to read frame from camera.")
                    break

                frame = cv2.flip(frame, 1)
                self._update_fps()
                self._process_frame(frame)

                cv2.imshow("Productivity Enforcer", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    self.running = False
                    break
        except KeyboardInterrupt:
            print("\n[INFO] Interrupted by user.")
        finally:
            self._cleanup()

    def _cleanup(self):
        print("[INFO] Cleaning up...")
        if self._cap:
            self._cap.release()
        cv2.destroyAllWindows()
        self._alarm_player.cleanup()
        self._gaze_tracker.cleanup()
        self._pinch_detector.cleanup()
        print("[OK] Shutdown complete.")

def main():
    parser = argparse.ArgumentParser(
        description="🔥 Extreme Productivity Enforcer — Webcam monitoring with alarm system.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--alarm", type=str, default="alarme.mp3",
                        help="Path to the alarm sound file (mp3 or wav). Default: alarme.mp3")
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera device ID. Default: 0")
    args = parser.parse_args()

    enforcer = ProductivityEnforcer(alarm_path=args.alarm, camera_id=args.camera)
    enforcer.run()


if __name__ == "__main__":
    main()
