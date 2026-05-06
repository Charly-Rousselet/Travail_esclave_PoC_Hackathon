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
COOLDOWN_DURATION = 5.0         # pause after alarm trigger
YOLO_CONFIDENCE = 0.45          # YOLO detection confidence threshold
FACE_CONFIDENCE = 0.5           # MediaPipe face detection confidence
CELL_PHONE_CLASS_ID = 67        # COCO class ID for "cell phone"

# Visual overlay colors (BGR)
COLOR_SAFE = (0, 200, 100)
COLOR_WARNING = (0, 180, 255)
COLOR_DANGER = (0, 0, 255)
COLOR_PHONE_BOX = (0, 100, 255)
COLOR_FACE_DOT = (255, 200, 0)
COLOR_BG_PANEL = (30, 30, 30)


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
        print(f"[OK] Alarm loaded: {alarm_path}")

    def _generate_default_alarm(self, path: str) -> str:
        """Generate a loud, aggressive alarm .wav file if no mp3 is provided."""
        import wave
        import struct
        import math

        sample_rate = 44100
        duration = 3.0  # seconds
        wav_path = path.replace(".mp3", ".wav") if path.endswith(".mp3") else path + ".wav"

        samples = []
        num_samples = int(sample_rate * duration)

        for i in range(num_samples):
            t = i / sample_rate
            # Multi-frequency aggressive alarm
            freq1 = 880 + 200 * math.sin(2 * math.pi * 3 * t)  # Siren sweep
            freq2 = 1200 + 300 * math.sin(2 * math.pi * 5 * t)  # Higher siren
            sample = 0.5 * math.sin(2 * math.pi * freq1 * t) + \
                     0.3 * math.sin(2 * math.pi * freq2 * t) + \
                     0.2 * math.sin(2 * math.pi * 440 * t)  # Bass tone
            # Pulsating envelope
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

    def play(self):
        """Play the alarm sound."""
        self._sound.play()

    def stop(self):
        """Stop the alarm sound."""
        self._sound.stop()

    def cleanup(self):
        """Clean up pygame mixer."""
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
        """
        Run YOLO inference on a frame.

        Returns:
            List of bounding boxes [(x1, y1, x2, y2, confidence)] for detected phones.
        """
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


class AttentionTracker:
    """Tracks user face presence and forward gaze using MediaPipe Tasks API."""

    MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
    MODEL_PATH = "blaze_face_short_range.tflite"

    def __init__(self, confidence: float = FACE_CONFIDENCE):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        self._mp = mp
        self._vision = vision

        # Auto-download model if not present
        if not os.path.isfile(self.MODEL_PATH):
            print(f"[...] Downloading MediaPipe face detection model...")
            import urllib.request
            urllib.request.urlretrieve(self.MODEL_URL, self.MODEL_PATH)
            print(f"[OK] Model downloaded: {self.MODEL_PATH}")

        base_options = BaseOptions(
            model_asset_path=self.MODEL_PATH
        )
        options = vision.FaceDetectorOptions(
            base_options=base_options,
            min_detection_confidence=confidence,
        )
        self._detector = vision.FaceDetector.create_from_options(options)
        print("[OK] MediaPipe Face Detection loaded.")

    def detect_face(self, frame: np.ndarray) -> list:
        """
        Detect faces in a frame.

        Returns:
            List of face landmarks [(nose_x, nose_y, bbox)] for detected faces.
            bbox = (x, y, w, h) in pixel coordinates.
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB, data=rgb_frame
        )
        results = self._detector.detect(mp_image)

        faces = []
        h, w, _ = frame.shape
        for detection in results.detections:
            bbox = detection.bounding_box
            bx = bbox.origin_x
            by = bbox.origin_y
            bw = bbox.width
            bh = bbox.height

            # Estimate nose position as center of bounding box (approximation)
            # The keypoints list contains: left_eye, right_eye, nose_tip, mouth, left_ear, right_ear
            keypoints = detection.keypoints
            if keypoints and len(keypoints) > 2:
                nose_kp = keypoints[2]  # nose_tip
                nx = int(nose_kp.x * w)
                ny = int(nose_kp.y * h)
            else:
                nx = bx + bw // 2
                ny = by + bh // 2

            faces.append((nx, ny, (bx, by, bw, bh)))

        return faces

    def cleanup(self):
        """Release MediaPipe resources."""
        self._detector.close()


class OverlayRenderer:
    """Renders debug overlays on the video frame."""

    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_BOLD = cv2.FONT_HERSHEY_DUPLEX

    @staticmethod
    def draw_phone_boxes(frame: np.ndarray, detections: list):
        """Draw bounding boxes around detected phones."""
        for (x1, y1, x2, y2, conf) in detections:
            # Glowing box effect
            cv2.rectangle(frame, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), COLOR_DANGER, 3)
            cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_PHONE_BOX, 2)

            label = f"PHONE {conf:.0%}"
            (tw, th), _ = cv2.getTextSize(label, OverlayRenderer.FONT, 0.6, 1)
            cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 8, y1), COLOR_PHONE_BOX, -1)
            cv2.putText(frame, label, (x1 + 4, y1 - 5),
                        OverlayRenderer.FONT, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    @staticmethod
    def draw_face_points(frame: np.ndarray, faces: list):
        """Draw face detection indicators."""
        for (nx, ny, (bx, by, bw, bh)) in faces:
            # Face bounding box
            cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), COLOR_SAFE, 1)
            # Nose point (gaze direction indicator)
            cv2.circle(frame, (nx, ny), 6, COLOR_FACE_DOT, -1)
            cv2.circle(frame, (nx, ny), 8, (255, 255, 255), 1)

    @staticmethod
    def draw_status_panel(
        frame: np.ndarray,
        phone_elapsed: float,
        attention_elapsed: float,
        state: str,
        fps: float,
    ):
        """Draw a translucent HUD panel with timer and status information."""
        h, w = frame.shape[:2]
        panel_h = 120
        panel_w = 340

        # Create translucent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (10 + panel_w, 10 + panel_h), COLOR_BG_PANEL, -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # Border
        cv2.rectangle(frame, (10, 10), (10 + panel_w, 10 + panel_h), (80, 80, 80), 1)

        # Title
        cv2.putText(frame, "PRODUCTIVITY ENFORCER", (20, 35),
                    OverlayRenderer.FONT_BOLD, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        # Phone timer bar
        phone_ratio = min(phone_elapsed / PHONE_TRIGGER_DURATION, 1.0)
        phone_color = COLOR_DANGER if phone_ratio >= 1.0 else (
            COLOR_WARNING if phone_ratio > 0.3 else COLOR_SAFE
        )
        cv2.putText(frame, f"Phone: {phone_elapsed:.1f}s / {PHONE_TRIGGER_DURATION:.0f}s",
                    (20, 58), OverlayRenderer.FONT, 0.45, phone_color, 1, cv2.LINE_AA)
        cv2.rectangle(frame, (20, 63), (20 + 300, 73), (60, 60, 60), -1)
        cv2.rectangle(frame, (20, 63), (20 + int(300 * phone_ratio), 73), phone_color, -1)

        # Attention timer bar
        att_ratio = min(attention_elapsed / ATTENTION_TRIGGER_DURATION, 1.0)
        att_color = COLOR_DANGER if att_ratio >= 1.0 else (
            COLOR_WARNING if att_ratio > 0.5 else COLOR_SAFE
        )
        cv2.putText(frame, f"Attention: {attention_elapsed:.1f}s / {ATTENTION_TRIGGER_DURATION:.0f}s",
                    (20, 92), OverlayRenderer.FONT, 0.45, att_color, 1, cv2.LINE_AA)
        cv2.rectangle(frame, (20, 97), (20 + 300, 107), (60, 60, 60), -1)
        cv2.rectangle(frame, (20, 97), (20 + int(300 * att_ratio), 107), att_color, -1)

        # State + FPS
        state_color = COLOR_DANGER if state == "ALARM" else (
            COLOR_WARNING if state == "COOLDOWN" else COLOR_SAFE
        )
        cv2.putText(frame, f"[{state}]", (20, 125),
                    OverlayRenderer.FONT_BOLD, 0.5, state_color, 1, cv2.LINE_AA)
        cv2.putText(frame, f"FPS: {fps:.0f}", (280, 125),
                    OverlayRenderer.FONT, 0.4, (150, 150, 150), 1, cv2.LINE_AA)

    @staticmethod
    def draw_alarm_flash(frame: np.ndarray, intensity: float = 0.3):
        """Flash the screen red during alarm."""
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
    """
    Main application controller.

    Orchestrates the video capture, detection modules, timer logic,
    and alarm system into a cohesive real-time monitoring loop.
    """

    def __init__(self, alarm_path: str, camera_id: int = 0):
        self._camera_id = camera_id
        self._cap = None

        # Sub-modules
        self._phone_detector = PhoneDetector()
        self._attention_tracker = AttentionTracker()
        self._alarm_player = AlarmPlayer(alarm_path)
        self._volume_ctrl = VolumeController()
        self._renderer = OverlayRenderer()

        # Timer state
        self._phone_start: float | None = None  # When phone was first seen
        self._attention_start: float | None = None  # When attention was first lost
        self._cooldown_until: float = 0.0  # Cooldown end timestamp

        # FPS tracking
        self._fps = 0.0
        self._frame_count = 0
        self._fps_timer = time.time()

    def _open_camera(self):
        """Open the webcam capture."""
        self._cap = cv2.VideoCapture(self._camera_id)
        if not self._cap.isOpened():
            print(f"[ERROR] Cannot open camera {self._camera_id}")
            sys.exit(1)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        print(f"[OK] Camera {self._camera_id} opened.")

    def _update_fps(self):
        """Update FPS counter."""
        self._frame_count += 1
        elapsed = time.time() - self._fps_timer
        if elapsed >= 1.0:
            self._fps = self._frame_count / elapsed
            self._frame_count = 0
            self._fps_timer = time.time()

    def _trigger_alarm(self, reason: str):
        """Execute the punishment sequence."""
        print(f"\n🚨 ALARM TRIGGERED: {reason}")
        self._volume_ctrl.set_max_volume()
        self._alarm_player.play()

        # Set cooldown
        self._cooldown_until = time.time() + COOLDOWN_DURATION

        # Reset timers
        self._phone_start = None
        self._attention_start = None

    def _process_frame(self, frame: np.ndarray) -> str:
        """
        Process a single frame through both detection modules.

        Returns the current application state: 'MONITORING', 'ALARM', or 'COOLDOWN'.
        """
        now = time.time()

        # During cooldown, skip detection
        if now < self._cooldown_until:
            remaining = self._cooldown_until - now
            self._renderer.draw_status_panel(frame, 0.0, 0.0, "COOLDOWN", self._fps)
            # Show cooldown countdown
            cv2.putText(frame, f"Cooldown: {remaining:.1f}s",
                        (frame.shape[1] // 2 - 100, frame.shape[0] - 30),
                        OverlayRenderer.FONT, 0.7, COLOR_WARNING, 2, cv2.LINE_AA)
            return "COOLDOWN"

        # ── Module 1: Phone Detection (YOLO) ──────────────────────
        phone_detections = self._phone_detector.detect(frame)
        self._renderer.draw_phone_boxes(frame, phone_detections)

        if phone_detections:
            if self._phone_start is None:
                self._phone_start = now
        else:
            self._phone_start = None

        phone_elapsed = (now - self._phone_start) if self._phone_start else 0.0

        # ── Module 2: Attention Tracking (MediaPipe) ──────────────
        faces = self._attention_tracker.detect_face(frame)
        self._renderer.draw_face_points(frame, faces)

        face_detected = len(faces) > 0

        if face_detected:
            self._attention_start = None
        else:
            if self._attention_start is None:
                self._attention_start = now

        attention_elapsed = (now - self._attention_start) if self._attention_start else 0.0

        # ── Check trigger conditions ──────────────────────────────
        state = "MONITORING"

        if phone_elapsed >= PHONE_TRIGGER_DURATION:
            self._trigger_alarm(f"Phone detected for {phone_elapsed:.1f}s")
            state = "ALARM"
        elif attention_elapsed >= ATTENTION_TRIGGER_DURATION:
            self._trigger_alarm(f"Inattention for {attention_elapsed:.1f}s")
            state = "ALARM"

        # ── Draw HUD ──────────────────────────────────────────────
        self._renderer.draw_status_panel(frame, phone_elapsed, attention_elapsed, state, self._fps)

        if state == "ALARM":
            self._renderer.draw_alarm_flash(frame)

        return state

    def run(self):
        """Main application loop."""
        self._open_camera()

        print("\n" + "=" * 55)
        print("  🔥 EXTREME PRODUCTIVITY ENFORCER — ACTIVE 🔥")
        print(f"  Phone trigger:     {PHONE_TRIGGER_DURATION}s")
        print(f"  Attention trigger:  {ATTENTION_TRIGGER_DURATION}s")
        print(f"  Cooldown:          {COOLDOWN_DURATION}s")
        print("  Press 'q' to quit.")
        print("=" * 55 + "\n")

        try:
            while True:
                ret, frame = self._cap.read()
                if not ret:
                    print("[ERROR] Failed to read frame from camera.")
                    break

                # Mirror the frame for a natural experience
                frame = cv2.flip(frame, 1)

                self._update_fps()
                self._process_frame(frame)

                cv2.imshow("Productivity Enforcer", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        except KeyboardInterrupt:
            print("\n[INFO] Interrupted by user.")
        finally:
            self._cleanup()

    def _cleanup(self):
        """Release all resources."""
        print("[INFO] Cleaning up...")
        if self._cap:
            self._cap.release()
        cv2.destroyAllWindows()
        self._alarm_player.stop()
        self._alarm_player.cleanup()
        self._attention_tracker.cleanup()
        print("[OK] Shutdown complete.")


def main():
    parser = argparse.ArgumentParser(
        description="🔥 Extreme Productivity Enforcer — Webcam monitoring with alarm system.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--alarm",
        type=str,
        default="alarme.mp3",
        help="Path to the alarm sound file (mp3 or wav). Default: alarme.mp3",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera device ID. Default: 0",
    )
    args = parser.parse_args()

    enforcer = ProductivityEnforcer(alarm_path=args.alarm, camera_id=args.camera)
    enforcer.run()


if __name__ == "__main__":
    main()
