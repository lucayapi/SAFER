"""Logging fichier sous resultats/<method>/logs/."""

from __future__ import annotations

import logging
from pathlib import Path

from safer_core.paths import method_logs_dir


def setup_method_logger(method_name: str, name: str = "safer") -> logging.Logger:
    log_dir = method_logs_dir(method_name)
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"{name}.{method_name}")
    if not logger.handlers:
        handler = logging.FileHandler(log_dir / f"{name}.log", encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
