#!/usr/bin/env python3
"""
🔥 EXTREME PRODUCTIVITY ENFORCER 🔥

Surveille l'utilisateur via webcam et déclenche une alarme sonore
lorsqu'un téléphone est détecté ou que l'attention est perdue.

Usage:
    uv run main.py [--alarm ALARM_PATH] [--camera CAMERA_ID]
"""

import argparse

from modules.enforcer import ProductivityEnforcer


def main():
    parser = argparse.ArgumentParser(
        description="🔥 Extreme Productivity Enforcer — Webcam monitoring with alarm system.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--alarm", type=str, default="alarme.wav",
                        help="Path to the alarm sound file (mp3 or wav). Default: alarme.wav")
    parser.add_argument("--camera", type=int, default=0,
                        help="Camera device ID. Default: 0")
    args = parser.parse_args()

    enforcer = ProductivityEnforcer(alarm_path=args.alarm, camera_id=args.camera)
    enforcer.run()


if __name__ == "__main__":
    main()
