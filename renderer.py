"""
renderer.py — OpenCV Overlay Renderer
======================================
Isolates every ``cv2.putText`` / ``cv2.circle`` call from the capture loop,
keeping ``engine.py`` free of rendering concerns (Single Responsibility).

All colour constants use BGR ordering to match OpenCV's convention.

Author  : CHETTOUH Elalmi Automation Engineer
Version : 2.0.0
"""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np

from config import TelemetryConfig
from tracker import EyeMetrics

# ---------------------------------------------------------------------------
# Colour palette (BGR)
# ---------------------------------------------------------------------------
_GREEN = (0, 255, 0)
_RED = (0, 0, 255)
_CYAN = (255, 255, 0)
_WHITE = (255, 255, 255)
_FONT = cv2.FONT_HERSHEY_SIMPLEX


class OverlayRenderer:
    """Draws real-time telemetry annotations onto a BGR frame.

    Args:
        config: Telemetry display configuration.

    Example::

        renderer = OverlayRenderer(config=cfg.telemetry)
        renderer.draw(frame, metrics_list)
    """

    def __init__(self, config: TelemetryConfig) -> None:
        self._cfg = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def draw(
        self,
        frame: np.ndarray,
        metrics_list: Sequence[EyeMetrics],
    ) -> None:
        """Render all per-eye overlays onto *frame* in place.

        Calls :meth:`_draw_eye_markers`, :meth:`_draw_velocity_hud`, and
        :meth:`_draw_anomaly_alerts` for each ``EyeMetrics`` entry.

        Args:
            frame:        BGR image to annotate (modified in place).
            metrics_list: Ordered sequence of per-eye metrics (right, left).
        """
        frame_width = frame.shape[1]
        for metrics in metrics_list:
            self._draw_eye_markers(frame, metrics)
            self._draw_velocity_hud(frame, metrics, frame_width)
            if metrics.is_anomaly:
                self._draw_anomaly_alert(frame, metrics, frame_width)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _draw_eye_markers(
        self, frame: np.ndarray, metrics: EyeMetrics
    ) -> None:
        """Draw the iris dot and ocular-centre anchor dot.

        Args:
            frame:   BGR frame (mutated in place).
            metrics: Metrics snapshot for this eye.
        """
        cv2.circle(
            frame,
            metrics.iris_px,
            self._cfg.iris_dot_radius,
            _GREEN,
            -1,
        )
        cv2.circle(
            frame,
            metrics.anchor_px,
            self._cfg.anchor_dot_radius,
            _RED,
            -1,
        )

    def _draw_velocity_hud(
        self,
        frame: np.ndarray,
        metrics: EyeMetrics,
        frame_width: int,
    ) -> None:
        """Render the velocity readout in the top corner of the relevant side.

        Right Eye → top-left region.
        Left Eye  → top-right region.

        Args:
            frame:       BGR frame.
            metrics:     Metrics snapshot.
            frame_width: Full frame width, used to position left-eye text.
        """
        text = f"{metrics.eye_label}: {metrics.omega_deg_s:.2f} deg/s"
        colour = _GREEN if metrics.eye_label.startswith("Right") else _CYAN
        x = 30 if metrics.eye_label.startswith("Right") else frame_width - 300
        cv2.putText(
            frame,
            text,
            (x, 40),
            _FONT,
            self._cfg.overlay_font_scale,
            colour,
            self._cfg.overlay_thickness,
        )

    def _draw_anomaly_alert(
        self,
        frame: np.ndarray,
        metrics: EyeMetrics,
        frame_width: int,
    ) -> None:
        """Render a prominent anomaly banner for the flagged eye.

        Args:
            frame:       BGR frame.
            metrics:     Metrics snapshot (``is_anomaly`` must be ``True``).
            frame_width: Full frame width.
        """
        is_right = metrics.eye_label.startswith("Right")
        text = "RIGHT ANOMALY!" if is_right else "LEFT ANOMALY!"
        x = 30 if is_right else frame_width - 300
        cv2.putText(
            frame,
            text,
            (x, 70),
            _FONT,
            self._cfg.overlay_font_scale,
            _RED,
            self._cfg.overlay_thickness,
        )
