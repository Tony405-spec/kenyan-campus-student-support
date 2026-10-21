"""Application logging configuration."""

import logging


def configure_logging(level: int = logging.INFO) -> None:
    """Configure readable timestamped logs for the command-line application."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
