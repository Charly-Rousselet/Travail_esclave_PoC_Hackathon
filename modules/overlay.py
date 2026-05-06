"""
Rendu des overlays visuels sur les frames OpenCV.
"""

import cv2
import numpy as np

from .config import (
    COLOR_BG_PANEL, COLOR_DANGER, COLOR_HAND, COLOR_IRIS, COLOR_IRIS_BAD,
    COLOR_PHONE_BOX, COLOR_SAFE, COLOR_WARNING,
    PHONE_TRIGGER_DURATION, ATTENTION_TRIGGER_DURATION,
)


class OverlayRenderer:
    """Renders debug overlays on the video frame."""

    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_BOLD = cv2.FONT_HERSHEY_DUPLEX

    @staticmethod
    def draw_hands(frame, pinch_info):
        for hand_pts in pinch_info.get("hand_landmarks", []):
            connections = [
                (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
                (5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),
                (13,17),(17,18),(18,19),(19,20),(0,17),
            ]
            for i, j in connections:
                if i < len(hand_pts) and j < len(hand_pts):
                    cv2.line(frame, hand_pts[i], hand_pts[j], COLOR_HAND, 1, cv2.LINE_AA)
            for pt in hand_pts:
                cv2.circle(frame, pt, 2, (255,255,255), -1)
        if pinch_info.get("pinch_pts"):
            thumb, index = pinch_info["pinch_pts"]
            color = COLOR_SAFE if pinch_info["is_pinching"] else COLOR_HAND
            cv2.circle(frame, thumb, 6, color, -1)
            cv2.circle(frame, index, 6, color, -1)
            cv2.line(frame, thumb, index, color, 2, cv2.LINE_AA)

    @staticmethod
    def draw_paused_overlay(frame):
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h//2-40), (w, h//2+40), (50,50,50), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        text = "PAUSED  (pinch to resume)"
        (tw, th), _ = cv2.getTextSize(text, OverlayRenderer.FONT_BOLD, 1.2, 2)
        cv2.putText(frame, text, ((w-tw)//2, h//2+th//2),
                    OverlayRenderer.FONT_BOLD, 1.2, (100,200,255), 2, cv2.LINE_AA)

    @staticmethod
    def draw_phone_boxes(frame, detections):
        for (x1, y1, x2, y2, conf) in detections:
            cv2.rectangle(frame, (x1-2,y1-2), (x2+2,y2+2), COLOR_DANGER, 3)
            cv2.rectangle(frame, (x1,y1), (x2,y2), COLOR_PHONE_BOX, 2)
            label = f"PHONE {conf:.0%}"
            (tw, th), _ = cv2.getTextSize(label, OverlayRenderer.FONT, 0.6, 1)
            cv2.rectangle(frame, (x1, y1-th-10), (x1+tw+8, y1), COLOR_PHONE_BOX, -1)
            cv2.putText(frame, label, (x1+4, y1-5),
                        OverlayRenderer.FONT, 0.6, (255,255,255), 1, cv2.LINE_AA)

    @staticmethod
    def draw_gaze_info(frame, gaze):
        if not gaze["face_detected"]:
            cv2.putText(frame, "NO FACE DETECTED", (frame.shape[1]//2-140, 50),
                        OverlayRenderer.FONT_BOLD, 0.7, COLOR_DANGER, 2, cv2.LINE_AA)
            return
        looking = gaze["looking_at_screen"]
        iris_color = COLOR_IRIS if looking else COLOR_IRIS_BAD
        if gaze["iris_pts"]:
            for key in ("left", "right"):
                pt = gaze["iris_pts"][key]
                cv2.circle(frame, pt, 4, iris_color, -1)
                cv2.circle(frame, pt, 6, (255,255,255), 1)
        if gaze["nose_pt"]:
            cv2.circle(frame, gaze["nose_pt"], 3, (255,200,0), -1)
        if gaze["face_bbox"]:
            bx, by, bw, bh = gaze["face_bbox"]
            cv2.rectangle(frame, (bx,by), (bx+bw,by+bh), COLOR_SAFE if looking else COLOR_WARNING, 1)
        label = "FOCUSED" if looking else "DISTRACTED"
        color = COLOR_SAFE if looking else COLOR_DANGER
        yaw, pitch = gaze["yaw"], gaze["pitch"]
        txt = f"{label} (yaw:{yaw:+.0f} pitch:{pitch:+.0f})"
        if gaze["face_bbox"]:
            bx, by, _, _ = gaze["face_bbox"]
            cv2.putText(frame, txt, (bx, by-8),
                        OverlayRenderer.FONT, 0.45, color, 1, cv2.LINE_AA)

    @staticmethod
    def draw_status_panel(frame, phone_elapsed, attention_elapsed, state, fps, alarm_reason=""):
        h, w = frame.shape[:2]
        panel_h, panel_w = 140, 340
        overlay = frame.copy()
        cv2.rectangle(overlay, (10,10), (10+panel_w, 10+panel_h), COLOR_BG_PANEL, -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        cv2.rectangle(frame, (10,10), (10+panel_w, 10+panel_h), (80,80,80), 1)
        cv2.putText(frame, "PRODUCTIVITY ENFORCER", (20,35),
                    OverlayRenderer.FONT_BOLD, 0.55, (255,255,255), 1, cv2.LINE_AA)
        # Phone bar
        pr = min(phone_elapsed / PHONE_TRIGGER_DURATION, 1.0)
        pc = COLOR_DANGER if pr >= 1.0 else (COLOR_WARNING if pr > 0.3 else COLOR_SAFE)
        cv2.putText(frame, f"Phone: {phone_elapsed:.1f}s / {PHONE_TRIGGER_DURATION:.0f}s",
                    (20,58), OverlayRenderer.FONT, 0.45, pc, 1, cv2.LINE_AA)
        cv2.rectangle(frame, (20,63), (320,73), (60,60,60), -1)
        cv2.rectangle(frame, (20,63), (20+int(300*pr),73), pc, -1)
        # Attention bar
        ar = min(attention_elapsed / ATTENTION_TRIGGER_DURATION, 1.0)
        ac = COLOR_DANGER if ar >= 1.0 else (COLOR_WARNING if ar > 0.5 else COLOR_SAFE)
        cv2.putText(frame, f"Gaze: {attention_elapsed:.1f}s / {ATTENTION_TRIGGER_DURATION:.0f}s",
                    (20,92), OverlayRenderer.FONT, 0.45, ac, 1, cv2.LINE_AA)
        cv2.rectangle(frame, (20,97), (320,107), (60,60,60), -1)
        cv2.rectangle(frame, (20,97), (20+int(300*ar),107), ac, -1)
        # State + FPS
        sc = COLOR_DANGER if state == "ALARM" else COLOR_SAFE
        cv2.putText(frame, f"[{state}]", (20,125), OverlayRenderer.FONT_BOLD, 0.5, sc, 1, cv2.LINE_AA)
        cv2.putText(frame, f"FPS: {fps:.0f}", (280,125), OverlayRenderer.FONT, 0.4, (150,150,150), 1, cv2.LINE_AA)
        if alarm_reason:
            cv2.putText(frame, alarm_reason, (20,145), OverlayRenderer.FONT, 0.4, COLOR_DANGER, 1, cv2.LINE_AA)

    @staticmethod
    def draw_alarm_flash(frame, intensity=0.3):
        red_overlay = np.full_like(frame, (0,0,255), dtype=np.uint8)
        cv2.addWeighted(red_overlay, intensity, frame, 1.0-intensity, 0, frame)
        h, w = frame.shape[:2]
        text = "!!! ALARM !!!"
        (tw, th), _ = cv2.getTextSize(text, OverlayRenderer.FONT_BOLD, 2.0, 3)
        cv2.putText(frame, text, ((w-tw)//2, (h+th)//2),
                    OverlayRenderer.FONT_BOLD, 2.0, (255,255,255), 3, cv2.LINE_AA)
