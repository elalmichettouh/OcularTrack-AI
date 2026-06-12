"""
Bi-Lateral Ocular Telemetry System — Entry Point
=================================================
Launches the OcularTelemetryEngine with configuration loaded from
``config.py``.  All runtime behaviour is governed by ``EngineConfig``;
adjust that dataclass rather than touching this file.

Usage
-----
    python main.py
    python main.py --config custom_config.json   # (future extension hook)

Author  : CHETTOUH Elalmi Automation Engineer
Version : 2.0.0
"""

import logging
import sys

from config import EngineConfig
from engine import OcularTelemetryEngine


def _bootstrap_root_logger(level: int = logging.DEBUG) -> None:
    """Configure the root logger with a structured, timestamped formatter.

    Args:
        level: Minimum log level for the root handler (default ``DEBUG``).
    """
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    logging.basicConfig(
        level=level,
        format=fmt,
        datefmt=datefmt,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("ocular_telemetry.log", mode="w", encoding="utf-8"),
        ],
    )


def main() -> None:
    """Application entry point."""
    _bootstrap_root_logger()
    log = logging.getLogger(__name__)

    log.info("=" * 60)
    log.info("  Bi-Lateral Ocular Telemetry System  v2.0.0")
    log.info("=" * 60)

    cfg = EngineConfig()
    log.debug("Runtime configuration: %s", cfg)

    engine = OcularTelemetryEngine(config=cfg)
    try:
        engine.run()
    except KeyboardInterrupt:
        log.warning("Keyboard interrupt received — shutting down gracefully.")
    finally:
        engine.shutdown()


if __name__ == "__main__":
    main()
