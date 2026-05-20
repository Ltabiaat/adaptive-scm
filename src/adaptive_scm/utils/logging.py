"""Structured logging configuration using structlog.

Configures structlog with a JSON renderer for production and a colored
console renderer for development. All modules import `get_logger` from
here; no `print()` calls are allowed anywhere in `src/` (PRD §2.3).

Usage:
    from adaptive_scm.utils.logging import get_logger
    log = get_logger(__name__)
    log.info("training_started", forecaster="arima", n_epochs=50)
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog


def configure_logging(level: str = "INFO", json_output: bool = False) -> None:
    """Configure the global structlog pipeline.

    Idempotent — safe to call multiple times. Called once at the start of
    every CLI entry point in `scripts/`. JSON output is intended for
    file-based logs during long experiment runs; colored console output
    is for interactive development.

    Args:
        level: Standard logging level name (DEBUG, INFO, WARNING, ERROR).
        json_output: If True, emit JSON lines; otherwise emit colored text.
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
    )

    processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if json_output:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer(colors=True))

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a bound structlog logger for the given module.

    Wraps structlog.get_logger so call sites have a single, typed import
    path. Pass `__name__` so log records carry the module name.

    Args:
        name: Module name, typically `__name__`.

    Returns:
        A structlog BoundLogger ready to emit structured events.
    """
    return structlog.get_logger(name)
