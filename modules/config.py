"""
Configuration globale du Productivity Enforcer.
Toutes les constantes et seuils de détection.
"""

import numpy as np

# ─────────────────────────────────────────────────────────────────
# Seuils de déclenchement
# ─────────────────────────────────────────────────────────────────

PHONE_TRIGGER_DURATION = 3.0       # secondes de détection continue du téléphone
ATTENTION_TRIGGER_DURATION = 20.0  # secondes d'inattention continue
YOLO_CONFIDENCE = 0.45             # seuil de confiance YOLO
FACE_CONFIDENCE = 0.5              # seuil de confiance MediaPipe
CELL_PHONE_CLASS_ID = 67           # ID COCO pour "cell phone"

# ─────────────────────────────────────────────────────────────────
# Seuils de pose de la tête (degrés)
# ─────────────────────────────────────────────────────────────────

YAW_THRESHOLD = 30.0   # rotation horizontale max
PITCH_THRESHOLD = 15.0 # inclinaison verticale max

# ─────────────────────────────────────────────────────────────────
# Pinch (pouce + index)
# ─────────────────────────────────────────────────────────────────

PINCH_COOLDOWN = 1.5       # secondes entre deux toggles
COLOR_HAND = (255, 180, 0) # couleur des landmarks main

# ─────────────────────────────────────────────────────────────────
# Couleurs overlay (BGR)
# ─────────────────────────────────────────────────────────────────

COLOR_SAFE = (0, 200, 100)
COLOR_WARNING = (0, 180, 255)
COLOR_DANGER = (0, 0, 255)
COLOR_PHONE_BOX = (0, 100, 255)
COLOR_IRIS = (0, 255, 0)
COLOR_IRIS_BAD = (0, 0, 255)
COLOR_BG_PANEL = (30, 30, 30)

# ─────────────────────────────────────────────────────────────────
# Indices des landmarks du visage (MediaPipe 478 points)
# ─────────────────────────────────────────────────────────────────

NOSE_TIP = 1
CHIN = 152
LEFT_EYE_CORNER = 33
RIGHT_EYE_CORNER = 263
LEFT_MOUTH = 61
RIGHT_MOUTH = 291

LEFT_IRIS_CENTER = 468
RIGHT_IRIS_CENTER = 473

# ─────────────────────────────────────────────────────────────────
# Points 3D d'un visage générique (pour solvePnP)
# ─────────────────────────────────────────────────────────────────

FACE_3D_MODEL = np.array([
    [0.0, 0.0, 0.0],          # Nose tip
    [0.0, -330.0, -65.0],     # Chin
    [-225.0, 170.0, -135.0],  # Left eye corner
    [225.0, 170.0, -135.0],   # Right eye corner
    [-150.0, -150.0, -125.0], # Left mouth corner
    [150.0, -150.0, -125.0],  # Right mouth corner
], dtype=np.float64)
