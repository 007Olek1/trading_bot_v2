import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional


DEFAULT_LOG_DIR = Path(os.getenv("LOG_DIR", "logs"))
DEFAULT_LOG_DIR.mkdir(parents=True, exist_ok=True)


def setup_logging(name: str = "goldtrigger", filename: str = "system.log") -> logging.Logger:
    """
    Configure structured logging for the swing bot.
    All modules should call this once and reuse the returned logger.
    """
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    log_path = DEFAULT_LOG_DIR / filename

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    handlers = []

    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=int(os.getenv("LOG_MAX_BYTES", 5 * 1024 * 1024)),
        backupCount=int(os.getenv("LOG_BACKUP_COUNT", 5)),
    )
    file_handler.setFormatter(formatter)
    handlers.append(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    handlers.append(stream_handler)

    logging.basicConfig(level=log_level, handlers=handlers, force=True)
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    logger.debug("Logging initialized at %s (file: %s)", log_level, log_path)
    return logger


def get_child_logger(parent: Optional[logging.Logger], child_name: str) -> logging.Logger:
    if parent:
        return parent.getChild(child_name)
    return logging.getLogger(child_name)
