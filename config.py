"""
config.py — Centralised Configuration for the Ocular Telemetry System
======================================================================
All tuneable parameters live here as frozen / mutable dataclasses.
Downstream modules import only what they need, keeping coupling minimal.

Design notes
------------
* ``LandmarkConfig``  — anatomical index constants, never mutated at runtime.
* ``FaceMeshConfig``  — MediaPipe FaceMesh construction kwargs.
* ``CameraConfig``    — OpenCV VideoCapture settings.
* ``EngineConfig``    — Top-level aggregate consumed by OcularTelemetryEngine.

Author  : Author  : CHETTOUH Elalmi Automation Engineer
Version : 2.0.0
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Landmark topology
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LandmarkConfig:
    """Immutable mapping of MediaPipe facial landmark indices.

    All indices are sourced from the canonical MediaPipe Face Mesh topology
    (478-point model with iris refinement enabled).

    Attributes:
        right_iris:   Refined iris centre for the right eye (index 468).
        right_canthus_inner: Medial canthus of the right eye (index 33).
        right_canthus_outer: Lateral canthus of the right eye (index 133).
        left_iris:    Refined iris centre for the left eye (index 473).
        left_canthus_inner:  Medial canthus of the left eye (index 362).
        left_canthus_outer:  Lateral canthus of the left eye (index 263).
    """

    right_iris: int = 468
    right_canthus_inner: int = 33
    right_canthus_outer: int = 133

    left_iris: int = 473
    left_canthus_inner: int = 362
    left_canthus_outer: int = 263


# ---------------------------------------------------------------------------
# MediaPipe face mesh
# ---------------------------------------------------------------------------

@dataclass
class FaceMeshConfig:
    """Construction parameters forwarded to ``mp.solutions.face_mesh.FaceMesh``.

    Attributes:
        max_num_faces:           Maximum faces tracked simultaneously.
        refine_landmarks:        Enable 478-point iris-refined model.
        min_detection_confidence: Minimum confidence to initialise tracking.
        min_tracking_confidence:  Minimum confidence to continue tracking.
    """

    max_num_faces: int = 1
    refine_landmarks: bool = True
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------

@dataclass
class CameraConfig:
    """OpenCV VideoCapture initialisation and stream settings.

    Attributes:
        device_index:  Camera device index (0 = default webcam).
        frame_width:   Requested capture width in pixels (0 = driver default).
        frame_height:  Requested capture height in pixels (0 = driver default).
        fps_target:    Target frames per second for ``cv2.waitKey`` timing.
        flip_code:     ``cv2.flip`` flipCode.  1 = horizontal mirror,
                       0 = vertical, -1 = both.  ``None`` disables flipping.
    """

    device_index: int = 0
    frame_width: int = 0
    frame_height: int = 0
    fps_target: int = 30
    flip_code: int | None = 1


# ---------------------------------------------------------------------------
# Telemetry thresholds & display
# ---------------------------------------------------------------------------

@dataclass
class TelemetryConfig:
    """Thresholds and UI parameters for the telemetry overlay.

    Attributes:
        velocity_threshold_deg_s: Angular velocity (deg/s) above which a
            saccade is flagged as an anomaly.
        overlay_font_scale:       OpenCV text scale factor.
        overlay_thickness:        OpenCV text stroke thickness (px).
        iris_dot_radius:          Radius of the iris centre marker (px).
        anchor_dot_radius:        Radius of the ocular-centre anchor marker (px).
        export_filename:          Base filename for the CSV diagnostic report.
    """

    velocity_threshold_deg_s: float = 500.0
    overlay_font_scale: float = 0.6
    overlay_thickness: int = 2
    iris_dot_radius: int = 4
    anchor_dot_radius: int = 3
    export_filename: str = "eye_movement_telemetry_report.csv"


# ---------------------------------------------------------------------------
# Top-level aggregate
# ---------------------------------------------------------------------------

@dataclass
class EngineConfig:
    """Aggregate configuration object consumed by ``OcularTelemetryEngine``.

    Compose all sub-configs here so callers only need to import one symbol.

    Attributes:
        landmarks:  Anatomical landmark index mapping.
        face_mesh:  MediaPipe FaceMesh construction kwargs.
        camera:     OpenCV capture settings.
        telemetry:  Thresholds and display options.
    """

    landmarks: LandmarkConfig = field(default_factory=LandmarkConfig)
    face_mesh: FaceMeshConfig = field(default_factory=FaceMeshConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)
