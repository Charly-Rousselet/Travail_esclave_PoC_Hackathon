#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════╗
║          🔥 PRODUCTIVITY ENFORCER — Interface Graphique 🔥       ║
║                                                                  ║
║  Lance main.py dans un thread séparé avec des réglages           ║
║  personnalisables. Se minimise dans le Dock / System Tray.       ║
╚══════════════════════════════════════════════════════════════════╝

Usage:
    uv run app.py
"""

import platform
import threading
import sys

import customtkinter as ctk

# Import du moteur de surveillance depuis modules/
from modules.enforcer import ProductivityEnforcer


# ─────────────────────────────────────────────────────────────────
# Configuration de l'interface
# ─────────────────────────────────────────────────────────────────

APP_TITLE = "🔥 Productivity Enforcer"
APP_WIDTH = 460
APP_HEIGHT = 400

IS_MACOS = platform.system() == "Darwin"

# Thème customtkinter
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


# ─────────────────────────────────────────────────────────────────
# System Tray (optionnel — crash sur macOS si mal géré)
# ─────────────────────────────────────────────────────────────────

_TRAY_AVAILABLE = False
try:
    if not IS_MACOS:
        # Sur macOS, pystray nécessite le main thread → incompatible avec tkinter
        import pystray
        from PIL import Image, ImageDraw
        _TRAY_AVAILABLE = True
except ImportError:
    pass


def _create_tray_icon_image(size: int = 64) -> "Image.Image":
    """Génère une icône basique pour le System Tray."""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    margin = 4
    draw.ellipse(
        [margin, margin, size - margin, size - margin],
        fill="red", outline="white", width=2,
    )
    try:
        draw.text((size // 2 - 6, size // 2 - 10), "P", fill="white")
    except Exception:
        pass
    return img


class ProductivityApp(ctk.CTk):
    """Interface graphique principale."""

    def __init__(self):
        super().__init__()

        # ── Configuration de la fenêtre ───────────────────────────
        self.title(APP_TITLE)
        self.geometry(f"{APP_WIDTH}x{APP_HEIGHT}")
        self.resizable(False, False)

        # ── État interne ──────────────────────────────────────────
        self._enforcer = None         # Instance du ProductivityEnforcer
        self._enforcer_thread = None  # Thread daemon pour le moteur
        self._is_running = False      # État de la surveillance
        self._tray_icon = None        # Icône System Tray (si disponible)
        self._is_hidden = False       # Fenêtre masquée ?

        # ── Construction de l'interface ───────────────────────────
        self._build_ui()

        # ── Gestion de la fermeture ───────────────────────────────
        self.protocol("WM_DELETE_WINDOW", self._on_close_window)

        # ── System Tray (Linux/Windows uniquement) ────────────────
        if _TRAY_AVAILABLE:
            self._start_tray()

    # ══════════════════════════════════════════════════════════════
    # Construction de l'interface
    # ══════════════════════════════════════════════════════════════

    def _build_ui(self):
        """Construit tous les widgets de l'interface."""

        # ── Titre ─────────────────────────────────────────────────
        title_label = ctk.CTkLabel(
            self, text="🔥 Productivity Enforcer",
            font=ctk.CTkFont(size=24, weight="bold"),
        )
        title_label.pack(pady=(20, 5))

        subtitle = ctk.CTkLabel(
            self, text="Surveillance webcam avec alarme",
            font=ctk.CTkFont(size=13),
            text_color="gray",
        )
        subtitle.pack(pady=(0, 15))

        # ── Cadre des sliders ─────────────────────────────────────
        sliders_frame = ctk.CTkFrame(self, fg_color="transparent")
        sliders_frame.pack(padx=30, fill="x")

        # Slider : Tolérance Téléphone (1–10s)
        phone_label = ctk.CTkLabel(
            sliders_frame, text="📱 Tolérance Téléphone",
            font=ctk.CTkFont(size=14, weight="bold"),
        )
        phone_label.pack(anchor="w", pady=(5, 0))

        self._phone_value_label = ctk.CTkLabel(
            sliders_frame, text="3s",
            font=ctk.CTkFont(size=12), text_color="#4a9eff",
        )
        self._phone_value_label.pack(anchor="e")

        self._phone_slider = ctk.CTkSlider(
            sliders_frame, from_=1, to=10, number_of_steps=9,
            command=self._on_phone_slider_change,
        )
        self._phone_slider.set(3)
        self._phone_slider.pack(fill="x", pady=(0, 10))

        # Slider : Tolérance Inattention (5–60s)
        attention_label = ctk.CTkLabel(
            sliders_frame, text="👁️ Tolérance Inattention",
            font=ctk.CTkFont(size=14, weight="bold"),
        )
        attention_label.pack(anchor="w", pady=(5, 0))

        self._attention_value_label = ctk.CTkLabel(
            sliders_frame, text="20s",
            font=ctk.CTkFont(size=12), text_color="#4a9eff",
        )
        self._attention_value_label.pack(anchor="e")

        self._attention_slider = ctk.CTkSlider(
            sliders_frame, from_=5, to=60, number_of_steps=55,
            command=self._on_attention_slider_change,
        )
        self._attention_slider.set(20)
        self._attention_slider.pack(fill="x", pady=(0, 15))

        # ── Bouton Start / Stop ───────────────────────────────────
        self._start_button = ctk.CTkButton(
            self, text="▶  DÉMARRER LA SURVEILLANCE",
            font=ctk.CTkFont(size=16, weight="bold"),
            height=50, corner_radius=12,
            fg_color="#1a8f3c", hover_color="#15732f",
            command=self._toggle_surveillance,
        )
        self._start_button.pack(padx=30, pady=(5, 10), fill="x")

        # ── Label de statut ───────────────────────────────────────
        self._status_label = ctk.CTkLabel(
            self, text="⏸ En attente",
            font=ctk.CTkFont(size=12),
            text_color="gray",
        )
        self._status_label.pack(pady=(0, 5))

        # ── Bouton Restaurer (visible quand minimisé sur macOS) ───
        bottom_frame = ctk.CTkFrame(self, fg_color="transparent")
        bottom_frame.pack(fill="x", padx=30, pady=(0, 10))

        if IS_MACOS:
            hint_text = "Fermer → minimise dans le Dock"
        else:
            hint_text = "Fermer → minimise dans le System Tray"

        tray_info = ctk.CTkLabel(
            bottom_frame, text=hint_text,
            font=ctk.CTkFont(size=10), text_color="#666666",
        )
        tray_info.pack(side="left")

        # Bouton quitter proprement
        quit_btn = ctk.CTkButton(
            bottom_frame, text="Quitter",
            font=ctk.CTkFont(size=11),
            width=70, height=28, corner_radius=6,
            fg_color="#555555", hover_color="#c0392b",
            command=self._quit_app,
        )
        quit_btn.pack(side="right")

    # ══════════════════════════════════════════════════════════════
    # Callbacks des sliders
    # ══════════════════════════════════════════════════════════════

    def _on_phone_slider_change(self, value):
        """Met à jour le label du slider téléphone."""
        v = int(value)
        self._phone_value_label.configure(text=f"{v}s")

    def _on_attention_slider_change(self, value):
        """Met à jour le label du slider inattention."""
        v = int(value)
        self._attention_value_label.configure(text=f"{v}s")

    # ══════════════════════════════════════════════════════════════
    # Logique Start / Stop
    # ══════════════════════════════════════════════════════════════

    def _toggle_surveillance(self):
        """Démarre ou arrête la surveillance."""
        if self._is_running:
            self._stop_surveillance()
        else:
            self._start_surveillance()

    def _start_surveillance(self):
        """Démarre le ProductivityEnforcer dans un thread daemon."""
        phone_dur = int(self._phone_slider.get())
        attention_dur = int(self._attention_slider.get())

        # Créer une nouvelle instance avec les réglages (headless = pas de fenêtre OpenCV)
        self._enforcer = ProductivityEnforcer(alarm_path="alarme.wav", camera_id=0, headless=True)
        self._enforcer.phone_trigger_duration = float(phone_dur)
        self._enforcer.attention_trigger_duration = float(attention_dur)

        # Lancer dans un thread daemon (se termine avec le programme)
        self._enforcer_thread = threading.Thread(
            target=self._enforcer.run, daemon=True
        )
        self._enforcer_thread.start()

        # Mettre à jour l'interface
        self._is_running = True
        self._start_button.configure(
            text="⏹  ARRÊTER LA SURVEILLANCE",
            fg_color="#c0392b", hover_color="#962d22",
        )
        self._status_label.configure(
            text=f"🟢 Active — Tél: {phone_dur}s | Attention: {attention_dur}s",
            text_color="#1a8f3c",
        )
        # Désactiver les sliders pendant la surveillance
        self._phone_slider.configure(state="disabled")
        self._attention_slider.configure(state="disabled")

        print(f"[GUI] Surveillance démarrée (phone={phone_dur}s, attention={attention_dur}s)")

    def _stop_surveillance(self):
        """Arrête le ProductivityEnforcer proprement."""
        if self._enforcer:
            self._enforcer.running = False

        # Attendre la fin du thread (avec timeout)
        if self._enforcer_thread and self._enforcer_thread.is_alive():
            self._enforcer_thread.join(timeout=5.0)

        self._enforcer = None
        self._enforcer_thread = None

        # Mettre à jour l'interface
        self._is_running = False
        self._start_button.configure(
            text="▶  DÉMARRER LA SURVEILLANCE",
            fg_color="#1a8f3c", hover_color="#15732f",
        )
        self._status_label.configure(
            text="⏸ En attente", text_color="gray",
        )
        # Réactiver les sliders
        self._phone_slider.configure(state="normal")
        self._attention_slider.configure(state="normal")

        print("[GUI] Surveillance arrêtée.")

    # ══════════════════════════════════════════════════════════════
    # System Tray (Linux / Windows uniquement)
    # ══════════════════════════════════════════════════════════════

    def _start_tray(self):
        """Démarre l'icône du System Tray dans un thread séparé."""
        icon_image = _create_tray_icon_image()

        menu = pystray.Menu(
            pystray.MenuItem("Afficher l'application", self._tray_show_window, default=True),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Quitter", self._tray_quit),
        )

        self._tray_icon = pystray.Icon(
            "productivity_enforcer",
            icon_image,
            "Productivity Enforcer",
            menu,
        )

        tray_thread = threading.Thread(target=self._tray_icon.run, daemon=True)
        tray_thread.start()

    def _tray_show_window(self, icon=None, item=None):
        """Restaure la fenêtre depuis le System Tray."""
        self.after(0, self._restore_window)

    def _tray_quit(self, icon=None, item=None):
        """Quitte proprement l'application depuis le tray."""
        self.after(0, self._quit_app)

    # ══════════════════════════════════════════════════════════════
    # Gestion de la fermeture de la fenêtre
    # ══════════════════════════════════════════════════════════════

    def _on_close_window(self):
        """Masque la fenêtre au lieu de quitter."""
        if IS_MACOS:
            # Sur macOS : minimiser dans le Dock
            self.iconify()
            print("[GUI] Fenêtre minimisée dans le Dock.")
        else:
            # Sur Linux/Windows : masquer + tray
            self.withdraw()
            self._is_hidden = True
            print("[GUI] Fenêtre masquée → l'app tourne en arrière-plan.")

    def _restore_window(self):
        """Restaure la fenêtre et la met au premier plan."""
        self.deiconify()
        self._is_hidden = False
        self.lift()
        self.focus_force()

    def _quit_app(self):
        """Quitte proprement l'application."""
        print("[GUI] Fermeture de l'application...")

        # Arrêter la surveillance si active
        if self._is_running:
            self._stop_surveillance()

        # Arrêter le tray si actif
        if self._tray_icon:
            try:
                self._tray_icon.stop()
            except Exception:
                pass

        # Fermer tkinter
        self.destroy()
        sys.exit(0)


# ─────────────────────────────────────────────────────────────────
# Point d'entrée
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = ProductivityApp()
    app.mainloop()
