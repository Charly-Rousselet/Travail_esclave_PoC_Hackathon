"""
Gestion de l'alarme sonore via pygame.mixer.
"""

import math
import os
import struct
import wave


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
