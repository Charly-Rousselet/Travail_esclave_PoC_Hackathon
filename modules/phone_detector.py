"""
Détection de téléphone portable via YOLOv8 nano.
"""

import numpy as np

from .config import CELL_PHONE_CLASS_ID, YOLO_CONFIDENCE


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
