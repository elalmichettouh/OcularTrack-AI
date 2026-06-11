"""
tracker.py — Per-Eye Angular Velocity Tracker
==============================================
Encapsulates the state and mathematics required to compute the instantaneous
angular velocity of a single eye from consecutive MediaPipe landmark frames.

Mathematical background
-----------------------
Let **r** = (x, y) be the displacement vector from the ocular centre to the
iris centre at time *t*, and **r'** = (x', y') the same vector at *t − dt*.

The signed angular velocity is derived from the 2-D cross product:

    ω  =  (r × Δr) / (|r|² · dt)
       =  (x · Δy − y · Δx) / (r_sq · dt)        [rad/s]

where  Δr = r − r'  is the incremental displacement.

This avoids computing ``arctan2`` on every frame (O(1) trig) and remains
numerically stable provided ``r_sq > r_sq_min_threshold``.

Author  : Senior AI & CV Staff Engineer
Version : 2.0.0
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import mediapipe as mp

from config import LandmarkConfig

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public result type
# ---------------------------------------------------------------------------

@dataclass
class EyeMetrics:
    """Snapshot of a single eye's computed metrics for one frame.

    Attributes:
        eye_label:       Human-readable identifier (e.g. ``"Right Eye"``).
        iris_px:         Iris centre in pixel coordinates ``(x, y)``.
        anchor_px:       Ocular-midpoint anchor in pixel coordinates ``(x, y)``.
        displacement:    Displacement vector ``(dx, dy)`` from anchor to iris.
        omega_deg_s:     Signed angular velocity in degrees / second.
        is_anomaly:      ``True`` when ``|omega_deg_s| > velocity_threshold``.
    """

    eye_label: str
    iris_px: tuple[int, int]
    anchor_px: tuple[int, int]
    displacement: tuple[float, float]
    omega_deg_s: float
    is_anomaly: bool


# ---------------------------------------------------------------------------
# Single-eye tracker
# ---------------------------------------------------------------------------

class EyeTracker:
    """Stateful tracker that emits angular velocity for one eye per frame.

    Args:
        eye_label:           Display label, e.g. ``"Right Eye"``.
        iris_idx:            MediaPipe landmark index for the iris centre.
        canthus_inner_idx:   Landmark index for the medial canthus.
        canthus_outer_idx:   Landmark index for the lateral canthus.
        velocity_threshold:  Anomaly threshold in deg/s.
        r_sq_min:            Minimum squared displacement below which angular
                             velocity is clamped to zero (avoids divide-by-zero
                             when the iris sits exactly on the anchor).

    Example::

        tracker = EyeTracker(
            eye_label="Right Eye",
            iris_idx=468,
            canthus_inner_idx=33,
            canthus_outer_idx=133,
            velocity_threshold=500.0,
        )
        metrics = tracker.process(landmarks, frame_width, frame_height, dt)
    """

    def __init__(
        self,
        eye_label: str,
        iris_idx: int,
        canthus_inner_idx: int,
        canthus_outer_idx: int,
        velocity_threshold: float,
        r_sq_min: float = 1.0,
    ) -> None:
        self._label = eye_label
        self._iris_idx = iris_idx
        self._canthus_inner_idx = canthus_inner_idx
        self._canthus_outer_idx = canthus_outer_idx
        self._velocity_threshold = velocity_threshold
        self._r_sq_min = r_sq_min

        # State for numerical differentiation (previous frame displacement)
        self._prev_disp: Optional[np.ndarray] = None

        log.debug(
            "EyeTracker initialised — label=%r, iris=%d, canthus=[%d, %d], "
            "threshold=%.1f deg/s",
            eye_label, iris_idx, canthus_inner_idx, canthus_outer_idx,
            velocity_threshold,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear cached state (call when tracking is lost / re-acquired)."""
        self._prev_disp = None
        log.debug("EyeTracker[%s] state reset.", self._label)

    def process(
        self,
        face_landmarks: "mp.framework.formats.landmark_pb2.NormalizedLandmarkList",
        frame_width: int,
        frame_height: int,
        dt: float,
    ) -> EyeMetrics:
        """Compute angular velocity from the current landmark list.

        Args:
            face_landmarks: MediaPipe ``NormalizedLandmarkList`` for one face.
            frame_width:    Width of the source frame in pixels.
            frame_height:   Height of the source frame in pixels.
            dt:             Elapsed time since the previous call, in seconds.

        Returns:
            :class:`EyeMetrics` populated for this frame.

        Raises:
            IndexError: If a required landmark index is out of range (logged
                        and re-raised so the engine can skip the frame).
        """
        try:
            iris_lm = face_landmarks.landmark[self._iris_idx]
            inner_lm = face_landmarks.landmark[self._canthus_inner_idx]
            outer_lm = face_landmarks.landmark[self._canthus_outer_idx]
        except IndexError as exc:
            log.error(
                "EyeTracker[%s] landmark index out of range: %s", self._label, exc
            )
            raise

        # --- Pixel-space coordinates ----------------------------------------
        iris_px = (
            int(iris_lm.x * frame_width),
            int(iris_lm.y * frame_height),
        )
        anchor_px = (
            int(((inner_lm.x + outer_lm.x) * 0.5) * frame_width),
            int(((inner_lm.y + outer_lm.y) * 0.5) * frame_height),
        )

        # Displacement vector r = iris − anchor  (float for precision)
        disp = np.array(
            [iris_px[0] - anchor_px[0], iris_px[1] - anchor_px[1]],
            dtype=np.float64,
        )

        # --- Angular velocity (cross-product formulation) -------------------
        omega_deg_s = self._compute_omega(disp, dt)

        # Update cached displacement for next frame
        self._prev_disp = disp

        is_anomaly = abs(omega_deg_s) > self._velocity_threshold

        return EyeMetrics(
            eye_label=self._label,
            iris_px=iris_px,
            anchor_px=anchor_px,
            displacement=(float(disp[0]), float(disp[1])),
            omega_deg_s=omega_deg_s,
            is_anomaly=is_anomaly,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_omega(self, disp: np.ndarray, dt: float) -> float:
        """Compute signed angular velocity via the 2-D cross-product estimator.

        Formula::

            ω_rad = (r × Δr) / (|r|² · dt)
                  = (x·Δy − y·Δx) / (r_sq · dt)

        Args:
            disp: Current displacement vector ``[x, y]`` (pixel units).
            dt:   Elapsed time in seconds.  Must be > 0.

        Returns:
            Angular velocity in degrees per second.  Returns ``0.0`` if the
            previous frame state is unavailable or ``|r|²`` is below the
            minimum threshold.
        """
        if self._prev_disp is None or dt <= 0.0:
            return 0.0

        r_sq = float(np.dot(disp, disp))           # |r|²  (dot product)
        if r_sq < self._r_sq_min:
            return 0.0

        # Δr — incremental displacement (vectorised subtraction)
        delta = disp - self._prev_disp             # shape (2,)

        # 2-D cross product:  x·Δy − y·Δx
        cross = float(disp[0] * delta[1] - disp[1] * delta[0])

        omega_rad_s = cross / (r_sq * dt)
        return float(np.degrees(omega_rad_s))


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def build_eye_trackers(
    landmarks: LandmarkConfig,
    velocity_threshold: float,
) -> tuple[EyeTracker, EyeTracker]:
    """Construct a (right, left) pair of :class:`EyeTracker` instances.

    Args:
        landmarks:          Landmark index configuration.
        velocity_threshold: Shared anomaly threshold in deg/s.

    Returns:
        A ``(right_tracker, left_tracker)`` tuple.
    """
    right = EyeTracker(
        eye_label="Right Eye",
        iris_idx=landmarks.right_iris,
        canthus_inner_idx=landmarks.right_canthus_inner,
        canthus_outer_idx=landmarks.right_canthus_outer,
        velocity_threshold=velocity_threshold,
    )
    left = EyeTracker(
        eye_label="Left Eye",
        iris_idx=landmarks.left_iris,
        canthus_inner_idx=landmarks.left_canthus_inner,
        canthus_outer_idx=landmarks.left_canthus_outer,
        velocity_threshold=velocity_threshold,
    )
    return right, left
