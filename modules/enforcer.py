"""
Contrôleur principal du Productivity Enforcer.
Orchestre tous les sous-modules (détection, alarme, overlay).
"""

import sys
import time

import cv2
import numpy as np

from .config import PHONE_TRIGGER_DURATION, ATTENTION_TRIGGER_DURATION
from .alarm import AlarmPlayer
from .gaze_tracker import GazeTracker
from .overlay import OverlayRenderer
from .phone_detector import PhoneDetector
from .pinch_detector import PinchDetector
from .volume import VolumeController


class ProductivityEnforcer:
    """Main application controller with Frame Skipping optimization."""

    def __init__(self, alarm_path: str, camera_id: int = 0, headless: bool = False):
        self._camera_id = camera_id
        self._cap = None
        self._headless = headless

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
        self._phone_start = None
        self._attention_start = None

        # Alarm state
        self._alarm_active = False
        self._alarm_reason = ""

        # Monitoring enabled state (toggled by pinch)
        self._monitoring_enabled = True

        # FPS & Frame Skipping
        self._fps = 0.0
        self._frame_count = 0
        self._fps_timer = time.time()
        self._skip_counter = 0

        # Cache pour les frames skippées
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
        if not self._alarm_active and self.running:
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

        # ── Pinch Detection (1 image sur 2) ───────────────────────
        if self._skip_counter % 2 == 0:
            self._last_pinch_info = self._pinch_detector.detect(frame)
        self._renderer.draw_hands(frame, self._last_pinch_info)

        if self._last_pinch_info["pinch_toggled"]:
            self._monitoring_enabled = not self._monitoring_enabled
            status = "ACTIVE" if self._monitoring_enabled else "PAUSED"
            print(f"\n🤏 PINCH DETECTED — Monitoring {status}")
            self._last_pinch_info["pinch_toggled"] = False
            if not self._monitoring_enabled:
                self._stop_alarm()
                self._phone_start = None
                self._attention_start = None

        if not self._monitoring_enabled:
            self._renderer.draw_paused_overlay(frame)
            self._renderer.draw_status_panel(frame, 0.0, 0.0, "PAUSED", self._fps, "Pinch to resume")
            return "PAUSED"

        # ── Phone Detection YOLO (1 image sur 10) ─────────────────
        if self._skip_counter % 10 == 0:
            self._last_phone_detections = self._phone_detector.detect(frame)
        self._renderer.draw_phone_boxes(frame, self._last_phone_detections)

        if self._last_phone_detections:
            if self._phone_start is None:
                self._phone_start = now
        else:
            self._phone_start = None
        phone_elapsed = (now - self._phone_start) if self._phone_start else 0.0

        # ── Gaze Tracking (1 image sur 5) ─────────────────────────
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

        # ── Alarm logic ───────────────────────────────────────────
        phone_triggered = phone_elapsed >= self.phone_trigger_duration
        attention_triggered = attention_elapsed >= self.attention_trigger_duration

        if phone_triggered:
            self._start_alarm(f"Phone detected for {phone_elapsed:.1f}s")
        elif attention_triggered:
            self._start_alarm(f"Inattention for {attention_elapsed:.1f}s")

        if self._alarm_active and not phone_triggered and not attention_triggered:
            self._stop_alarm()

        state = "ALARM" if self._alarm_active else "MONITORING"

        # ── HUD ───────────────────────────────────────────────────
        self._renderer.draw_status_panel(frame, phone_elapsed, attention_elapsed, state, self._fps, self._alarm_reason)
        if self._alarm_active:
            self._renderer.draw_alarm_flash(frame)

        if self._skip_counter >= 1000:
            self._skip_counter = 0

        return state

    def run(self):
        self._open_camera()
        print("\n" + "=" * 55)
        print("  🔥 EXTREME PRODUCTIVITY ENFORCER — ACTIVE 🔥")
        print("  ⚙️  Optimisation Frame Skipping : ACTIVÉE")
        print(f"  Phone trigger:     {self.phone_trigger_duration}s")
        print(f"  Attention trigger: {self.attention_trigger_duration}s")
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

                if self._headless:
                    time.sleep(0.01)
                else:
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
        if not self._headless:
            cv2.destroyAllWindows()
        self._alarm_player.cleanup()
        self._gaze_tracker.cleanup()
        self._pinch_detector.cleanup()
        print("[OK] Shutdown complete.")
