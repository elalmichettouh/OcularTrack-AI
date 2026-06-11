"""
engine.py — OcularTelemetryEngine
==================================
Top-level orchestrator that wires together the camera, MediaPipe face mesh,
bilateral eye trackers, overlay renderer, and data logger into a single,
production-grade capture loop.

Responsibilities
----------------
* Open and validate the camera stream.
* Drive the MediaPipe FaceMesh inference pipeline.
* Dispatch per-eye landmark data to both :class:`~tracker.EyeTracker` instances.
* Feed the rendered frame to the display window.
* Delegate logging to :class:`~logger.DataLogger`.
* Guarantee deterministic resource release via :meth:`shutdown`.

Author  : Senior AI & CV Staff Engineer
Version : 2.0.0
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

from config import EngineConfig
from logger import DataLogger
from renderer import OverlayRenderer
from tracker import EyeMetrics, EyeTracker, build_eye_trackers

log = logging.getLogger(__name__)

_WINDOW_TITLE = "Bi-Lateral Eye Tracking & Medical Telemetry  [press Q to quit]"


class OcularTelemetryEngine:
    """Main runtime engine for the Bi-Lateral Ocular Telemetry system.

    Encapsulates the full lifecycle: initialisation → capture loop → export
    → cleanup.  Resource acquisition/release follows the RAII pattern:
    :meth:`run` acquires, :meth:`shutdown` always releases (typically called
    from a ``finally`` block in ``main.py``).

    Args:
        config: Aggregate :class:`~config.EngineConfig` instance.

    Example::

        engine = OcularTelemetryEngine(config=EngineConfig())
        try:
            engine.run()
        finally:
            engine.shutdown()
    """

    def __init__(self, config: EngineConfig) -> None:
        self._cfg = config
        self._is_running = False

        # Sub-system handles (populated in _initialise)
        self._cap: Optional[cv2.VideoCapture] = None
        self._face_mesh: Optional[mp.solutions.face_mesh.FaceMesh] = None
        self._right_tracker: Optional[EyeTracker] = None
        self._left_tracker: Optional[EyeTracker] = None
        self._renderer: Optional[OverlayRenderer] = None
        self._data_logger: Optional[DataLogger] = None

    # ------------------------------------------------------------------
    # Public lifecycle API
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Initialise all sub-systems and enter the main capture loop.

        Exits cleanly on ``q`` key-press, stream exhaustion, or an unhandled
        exception (which propagates to the caller).
        """
        self._initialise()
        log.info("System active — position your face clearly in front of the camera.")
        log.info("Press 'Q' in the video window to terminate and export the telemetry log.")

        self._is_running = True
        prev_time: float = time.time()

        assert self._cap is not None
        assert self._face_mesh is not None

        while self._is_running and self._cap.isOpened():
            success, frame = self._cap.read()
            if not success:
                log.warning("Frame drop detected or camera stream disconnected.")
                break

            frame = self._preprocess_frame(frame)
            current_time = time.time()
            dt = current_time - prev_time
            prev_time = current_time

            metrics_list = self._process_frame(frame, dt, current_time)
            if metrics_list:
                assert self._renderer is not None
                self._renderer.draw(frame, metrics_list)

            cv2.imshow(_WINDOW_TITLE, frame)

            # ``waitKey`` drives OpenCV's event loop; 30 ms ≈ 33 fps ceiling
            wait_ms = max(1, 1000 // self._cfg.camera.fps_target)
            if cv2.waitKey(wait_ms) & 0xFF == ord("q"):
                log.info("Quit signal received from display window.")
                self._is_running = False

        log.info("Capture loop exited — total records: %d", len(self._data_logger))

    def shutdown(self) -> None:
        """Release all hardware and process resources, then export data.

        Safe to call multiple times (idempotent).
        """
        log.info("Shutting down OcularTelemetryEngine …")

        if self._cap is not None and self._cap.isOpened():
            self._cap.release()
            log.debug("Camera released.")

        if self._face_mesh is not None:
            self._face_mesh.close()
            log.debug("FaceMesh context closed.")

        cv2.destroyAllWindows()
        log.debug("OpenCV windows destroyed.")

        if self._data_logger is not None:
            self._data_logger.export()

        log.info("Shutdown complete.")

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _initialise(self) -> None:
        """Instantiate and validate all sub-systems before entering the loop.

        Raises:
            RuntimeError: If the camera cannot be opened.
        """
        log.info("Initialising sub-systems …")

        # Camera
        self._cap = cv2.VideoCapture(self._cfg.camera.device_index)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera at device index {self._cfg.camera.device_index}. "
                "Check that the device is connected and not in use."
            )

        cam = self._cfg.camera
        if cam.frame_width > 0:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam.frame_width)
        if cam.frame_height > 0:
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam.frame_height)
        if cam.fps_target > 0:
            self._cap.set(cv2.CAP_PROP_FPS, cam.fps_target)

        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self._cap.get(cv2.CAP_PROP_FPS)
        log.info("Camera opened — resolution: %dx%d @ %.1f fps", actual_w, actual_h, actual_fps)

        # MediaPipe FaceMesh
        fm_cfg = self._cfg.face_mesh
        self._face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=fm_cfg.max_num_faces,
            refine_landmarks=fm_cfg.refine_landmarks,
            min_detection_confidence=fm_cfg.min_detection_confidence,
            min_tracking_confidence=fm_cfg.min_tracking_confidence,
        )
        log.info("FaceMesh initialised (refine_landmarks=%s).", fm_cfg.refine_landmarks)

        # Eye trackers
        self._right_tracker, self._left_tracker = build_eye_trackers(
            landmarks=self._cfg.landmarks,
            velocity_threshold=self._cfg.telemetry.velocity_threshold_deg_s,
        )

        # Renderer & logger
        self._renderer = OverlayRenderer(config=self._cfg.telemetry)
        self._data_logger = DataLogger(
            export_filename=self._cfg.telemetry.export_filename
        )

        log.info("All sub-systems ready.")

    # ------------------------------------------------------------------
    # Per-frame processing
    # ------------------------------------------------------------------

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Apply camera-level transforms before inference.

        Currently performs optional horizontal flipping for mirror mode.

        Args:
            frame: Raw BGR frame from ``VideoCapture.read()``.

        Returns:
            Transformed BGR frame (new array if flipped, same reference otherwise).
        """
        if self._cfg.camera.flip_code is not None:
            return cv2.flip(frame, self._cfg.camera.flip_code)
        return frame

    def _process_frame(
        self,
        frame: np.ndarray,
        dt: float,
        wall_time: float,
    ) -> list[EyeMetrics]:
        """Run face-mesh inference and compute bilateral eye metrics.

        Args:
            frame:     Pre-processed BGR frame.
            dt:        Elapsed time since last frame (seconds).
            wall_time: Current ``time.time()`` for logging timestamps.

        Returns:
            List of :class:`~tracker.EyeMetrics` (one per tracked eye),
            or an empty list if no face was detected or ``dt`` is invalid.
        """
        assert self._face_mesh is not None
        assert self._right_tracker is not None
        assert self._left_tracker is not None
        assert self._data_logger is not None

        if dt <= 0.0:
            log.debug("Non-positive dt=%.6f skipped.", dt)
            return []

        # MediaPipe expects RGB; convert without allocating a persistent buffer
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._face_mesh.process(rgb_frame)

        if not results.multi_face_landmarks:
            # Reset trackers to avoid stale prev-frame state after re-detection
            self._right_tracker.reset()
            self._left_tracker.reset()
            return []

        h, w = frame.shape[:2]
        metrics_list: list[EyeMetrics] = []

        # Process only the primary face (index 0 when max_num_faces=1)
        face_landmarks = results.multi_face_landmarks[0]

        for tracker in (self._right_tracker, self._left_tracker):
            try:
                metrics = tracker.process(face_landmarks, w, h, dt)
            except IndexError:
                # Landmark index error already logged inside EyeTracker.process
                continue

            metrics_list.append(metrics)
            self._data_logger.record(metrics, wall_time)

            if metrics.is_anomaly:
                log.warning(
                    "Anomaly — %s | ω=%.2f deg/s (threshold=%.1f)",
                    metrics.eye_label,
                    metrics.omega_deg_s,
                    self._cfg.telemetry.velocity_threshold_deg_s,
                )

        return metrics_list
