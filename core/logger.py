import logging
import sys
from pathlib import Path
from typing import Optional


class LoggerSetup:
    """
    Just logger.
    """

    _instance = None

    @staticmethod
    def setup(name: str = "MGRE",
              log_file: Optional[Path] = None,
              level: int = logging.INFO) -> logging.Logger:

        logger = logging.getLogger(name)

        if logger.handlers:
            return logger

        logger.setLevel(level)

        fmt = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(fmt)
        logger.addHandler(console)

        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(fmt)
            logger.addHandler(file_handler)

        return logger

    @staticmethod
    def get_logger(name: str = "MGRE") -> logging.Logger:
        return logging.getLogger(name)


logger = LoggerSetup.setup()


def log_section(title: str, width: int = 70) -> None:
    logger.info("\n" + "=" * width)
    logger.info(title.center(width))
    logger.info("=" * width + "\n")


def log_subsection(title: str, width: int = 70) -> None:
    logger.info("\n" + "-" * width)
    logger.info(title)
    logger.info("-" * width)


def log_success(message: str) -> None:
    logger.info(message)


def log_warning(message: str) -> None:
    logger.warning(message)


def log_error(message: str) -> None:
    logger.error(message)


def log_progress(current: int, total: int, prefix: str = "") -> None:
    pct = (current / total) * 100
    logger.info(f"{prefix} [{current}/{total}] {pct:.1f}%")
