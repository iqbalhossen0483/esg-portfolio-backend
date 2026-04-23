"""Application logging.

Plain text in development, single-line key=value in production.
Use `get_logger(__name__)` everywhere instead of `print`.
"""

import logging
import logging.config

from config import settings


_CONFIGURED = False


def configure_logging() -> None:
    """Idempotent global logging configuration."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    level = "DEBUG" if settings.IS_DEV else "INFO"
    fmt = (
        "%(asctime)s %(levelname)s %(name)s :: %(message)s"
        if settings.IS_DEV
        else "ts=%(asctime)s level=%(levelname)s logger=%(name)s msg=%(message)s"
    )

    logging.config.dictConfig({
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {"format": fmt},
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "default",
            },
        },
        "loggers": {
            "uvicorn": {"level": level},
            "uvicorn.access": {"level": "INFO"},
            "sqlalchemy.engine": {"level": "WARNING"},
            "celery": {"level": level},
        },
        "root": {"level": level, "handlers": ["console"]},
    })

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    if not _CONFIGURED:
        configure_logging()
    return logging.getLogger(name)
