"""
Contrôle du volume système macOS via osascript.
"""

import subprocess


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
