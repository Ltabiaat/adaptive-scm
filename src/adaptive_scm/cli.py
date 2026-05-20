"""CLI entry point for the adaptive-scm package.

Registers sub-commands for preprocessing, training, and experiment running.
Each sub-command is currently a stub that prints a not-yet-implemented
message; concrete implementations land in `scripts/` modules during the
respective phases (see PRD §6).

Invoked via the `adaptive-scm` console script declared in pyproject.toml,
or directly with `python -m adaptive_scm.cli`.
"""

from __future__ import annotations

import click

from adaptive_scm.utils.logging import configure_logging, get_logger

log = get_logger(__name__)


@click.group()
@click.option("--log-level", default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR).")
@click.option("--json-logs", is_flag=True, help="Emit JSON log lines instead of colored text.")
def main(log_level: str, json_logs: bool) -> None:
    """Adaptive supply chain optimization CLI."""
    configure_logging(level=log_level, json_output=json_logs)


@main.command()
def version() -> None:
    """Print the installed package version."""
    from adaptive_scm import __version__

    click.echo(f"adaptive-scm {__version__}")


@main.command()
def status() -> None:
    """Show scaffold status — useful as an install sanity check."""
    log.info("scaffold_ok", phase="0_scaffold_complete")
    click.echo("Scaffold installed. Next step: implement Feature 1 (data pipeline).")


if __name__ == "__main__":
    main()
