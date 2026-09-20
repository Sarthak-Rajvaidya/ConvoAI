"""
utils/logging.py
===============================================================================
Central logging configuration + a `log_stage` helper that emits a
structured, greppable line for every pipeline stage:

    STAGE=TRANSCRIPTION provider=groq model=whisper-large-v3-turbo
    duration=14.2s status=success meeting_id=ab12cd34

This is the "observability" layer referenced in the README: it makes
performance debugging and demoing possible without a full tracing stack.
===============================================================================
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

_CONFIGURED = False


def configure_logging(level: str = "INFO") -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        handlers=[
            logging.FileHandler(logs_dir / "app.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    configure_logging(os.getenv("LOG_LEVEL", "INFO"))
    return logging.getLogger(name)


def log_stage(
    logger: logging.Logger,
    stage: str,
    *,
    duration: float | None = None,
    status: str = "success",
    **fields: object,
) -> None:
    """Emit one structured, single-line log entry for a pipeline stage."""
    parts = [f"STAGE={stage.upper()}"]
    for key, value in fields.items():
        if value is None or value == "":
            continue
        parts.append(f"{key}={value}")
    if duration is not None:
        parts.append(f"duration={duration:.2f}s")
    parts.append(f"status={status}")

    line = " ".join(parts)
    if status == "success":
        logger.info(line)
    else:
        logger.warning(line)
