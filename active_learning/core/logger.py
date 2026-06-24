"""Shared logging helpers."""

from __future__ import annotations

import logging
import sys


def configure_logging() -> None:
    """Configure package logging to stdout."""
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
        force=True,
    )
    logging.getLogger("active_learning").setLevel(logging.INFO)
    logging.getLogger("active_learning.seed").setLevel(logging.INFO)
    logging.getLogger("interface").setLevel(logging.INFO)
    # Keep high-level interface progress, but suppress per-image export/backup spam.
    logging.getLogger("interface.client.azure.client").setLevel(logging.WARNING)
    logging.getLogger("interface.client.elasticseach.client").setLevel(logging.WARNING)
