import logging
from argparse import Namespace
from typing import TYPE_CHECKING, Any

from lightning.pytorch.loggers.logger import Logger
from lightning.pytorch.utilities import rank_zero_only
from loguru import logger

if TYPE_CHECKING:
    from types import FrameType


class LoguruLightningLogger(Logger):
    def __init__(self, name: str, version: int = 1) -> None:
        self._name = name
        self._version = version
        # Keep an in-memory history of all metric dicts Lightning passes us.
        # Each entry is a flat dict that should contain an 'epoch' key when on_epoch=True.
        self._history: list[dict[str, float]] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def version(self) -> int:
        return self._version

    @property
    def history(self) -> list[dict[str, float]]:
        """Return a reference of the collected metric history for this run."""
        return self._history

    @rank_zero_only
    def log_hyperparams(
        self, params: dict[str, Any] | Namespace, *args: Any, **kwargs: Any
    ) -> None:
        logger.info("{}", params)

    @rank_zero_only
    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        # NOTE: step != metrics["epoch"]
        summary = ", ".join(f"{k}: {v:.5f}" for k, v in metrics.items() if k != "epoch")
        logger.info("Epoch {}: {}", metrics.get("epoch", "?"), summary)

        # Persist a numeric-only snapshot for later export (skip other values)
        snapshot: dict[str, float] = {}
        for k, v in metrics.items():
            try:
                snapshot[k] = float(v)
            except (TypeError, ValueError):
                continue
        if snapshot:
            self._history.append(snapshot)


class InterceptHandler(logging.Handler):
    """Intercepts standard logging and routes it to Loguru."""

    def emit(self, record: logging.LogRecord) -> None:
        """Route a record to loguru."""
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = str(record.levelno)

        # Traverse the stack to find the actual caller
        frame: FrameType | None
        frame, depth = logging.currentframe(), 1
        while frame:
            frame = frame.f_back
            if frame and frame.f_globals.get("__name__") not in {
                "logging",
                "loguru._handler",
                "loguru._logger",
            }:
                break
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())
