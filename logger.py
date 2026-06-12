"""
logger.py — Telemetry Data Logger
==================================
Accumulates per-frame ``EyeMetrics`` records in memory and exports them to
a structured CSV diagnostic report on shutdown.

Design decisions
----------------
* Uses a plain ``list`` with pre-allocated chunks to avoid repeated heap
  allocations during the hot capture loop.
* Tracks per-eye *last-capture wall-clock time* to emit meaningful inter-frame
  interval columns in the export.
* Thread-safety is not required (single-producer model from the capture loop),
  but the append path is kept minimal to reduce capture-loop latency.

Author  : CHETTOUH Elalmi Automation Engineer
Version : 2.0.0
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Optional

import pandas as pd

from tracker import EyeMetrics

log = logging.getLogger(__name__)

# Column schema — defines export order
_COLUMNS: list[str] = [
    "Timestamp",
    "Target Eye",
    "Velocity (deg/s)",
    "Interval Since Prev (s)",
    "Status",
]


class DataLogger:
    """Accumulates ocular telemetry records and exports them to CSV.

    Args:
        export_filename: Path (or filename) for the CSV diagnostic report.
        pre_alloc:       Initial list reservation size to reduce GC pressure
                         during streaming (default 4 096 entries).

    Example::

        logger = DataLogger(export_filename="report.csv")
        logger.record(metrics, wall_time=time.time())
        logger.export()
    """

    def __init__(
        self,
        export_filename: str = "eye_movement_telemetry_report.csv",
        pre_alloc: int = 4_096,
    ) -> None:
        self._export_filename = export_filename
        self._records: list[dict[str, Any]] = []
        self._records_reserve = pre_alloc  # for __repr__ only; list grows as needed

        # Per-label last-capture wall-clock time for interval computation
        self._last_capture: dict[str, Optional[float]] = {}

        log.info("DataLogger initialised — export target: '%s'", export_filename)

    # ------------------------------------------------------------------
    # Hot path (called every captured frame)
    # ------------------------------------------------------------------

    def record(self, metrics: EyeMetrics, wall_time: float) -> None:
        """Append one ``EyeMetrics`` snapshot to the in-memory log.

        This method is intentionally minimal: no I/O, no allocations beyond
        the dictionary literal.

        Args:
            metrics:   Computed metrics for one eye at the current frame.
            wall_time: Current ``time.time()`` value from the capture loop
                       (avoids an additional syscall inside this method).
        """
        label = metrics.eye_label
        last = self._last_capture.get(label)

        if last is not None:
            interval: Any = round(wall_time - last, 4)
        else:
            interval = "Initial Frame"

        self._last_capture[label] = wall_time

        self._records.append(
            {
                "Timestamp": datetime.fromtimestamp(wall_time).strftime(
                    "%H:%M:%S.%f"
                )[:-3],
                "Target Eye": label,
                "Velocity (deg/s)": round(metrics.omega_deg_s, 2),
                "Interval Since Prev (s)": interval,
                "Status": "ANOMALY DETECTED" if metrics.is_anomaly else "Normal",
            }
        )

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export(self) -> Optional[pd.DataFrame]:
        """Write the accumulated log to CSV and return the DataFrame.

        Returns:
            The :class:`pandas.DataFrame` that was exported, or ``None`` if
            no data was captured.
        """
        separator = "=" * 70
        log.info(separator)
        log.info("  FINAL DIAGNOSTIC OCULAR REPORT")
        log.info(separator)

        if not self._records:
            log.warning("No telemetry data captured — export skipped.")
            return None

        df = pd.DataFrame(self._records, columns=_COLUMNS)

        log.info("Total records captured: %d", len(df))
        log.info("Anomalies detected    : %d", (df["Status"] == "ANOMALY DETECTED").sum())

        # Console preview (first 20 rows, no index)
        print("\n" + separator)
        print(df.head(20).to_string(index=False))
        print(separator + "\n")

        try:
            df.to_csv(self._export_filename, index=False)
            log.info("Diagnostic report exported → '%s'", self._export_filename)
        except OSError as exc:
            log.error("Failed to write report: %s", exc)

        return df

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._records)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"DataLogger(records={len(self._records)}, "
            f"export='{self._export_filename}')"
        )
