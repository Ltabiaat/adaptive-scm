"""Structured logging configuration.

Single entrypoint ``get_logger`` returns a configured ``structlog`` logger. All
modules in the package use this instead of ``print``. Configuration is process-
global and applied lazily on first call.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog

_CONFIGURED = False


def _configure(level: int = logging.INFO) -> None:
    """Configure ``structlog`` once per process.

    Applies a JSON-friendly key-value renderer for production runs and wires
    structlog into the stdlib logging system so third-party libraries (xgboost,
    pytorch-lightning) also flow through the same handler. Called lazily by
    ``get_logger``; safe to call multiple times.

    Args:
        level: Stdlib logging level for the root logger.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=level,
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        cache_logger_on_first_use=True,
    )
    _CONFIGURED = True


def get_logger(name: str | None = None, **initial_context: Any) -> structlog.BoundLogger:
    """Return a configured structlog logger.

    Lazily initializes process-wide logging on first call, then returns a bound
    logger with optional initial context (e.g. ``run_id``, ``forecaster``) that
    will be attached to every log record from that logger.

    Args:
        name: Optional logger name (typically ``__name__`` of the caller).
        **initial_context: Key-value pairs bound into the returned logger.

    Returns:
        A ``structlog.BoundLogger`` ready for use.
    """
    _configure()
    logger = structlog.get_logger(name)
    if initial_context:
        logger = logger.bind(**initial_context)
    return logger
