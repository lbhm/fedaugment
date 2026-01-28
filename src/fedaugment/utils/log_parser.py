import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar, TypedDict

import polars as pl


class ExperimentLog(TypedDict):
    records: pl.DataFrame
    epoch_metrics: pl.DataFrame
    test_metrics: pl.DataFrame
    config: dict[str, Any]


class LogParser:
    LOG_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"^(?P<date>[^|]+)\s*\|\s*"
        r"(?P<severity>[^\s|]+)\s*\|\s*"
        r"(?P<module>[A-Za-z0-9_./-]+):(?P<line>\d+)\s*\|\s*"
        r"(?P<message>.+)$"
    )
    EPOCH_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"^Epoch\s+(?P<epoch>\d+):\s*(?P<metrics>.*)$"
    )
    METRIC_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"\s*(?P<key>[A-Za-z0-9_\-/]+)\s*[:=]\s*"
        r"(?P<value>[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?"  # normal floats (e.g., 1.23, 1e-4)
        r"|[-+]?inf|nan)"  # inf, +inf, -inf, nan
        r"(?:\s*,\s*|$)",
        re.IGNORECASE,
    )

    def parse(self, path: str | Path) -> list[ExperimentLog]:
        """Parse a log file and extract structured data for every experiment run it contains."""
        experiments: list[ExperimentLog] = []
        records: list[dict[str, Any]] = []
        epoch_metrics: dict[int, dict[str, float]] = defaultdict(dict)
        test_metrics: dict[str, float] = {}
        config: dict[str, Any] = {}

        def flush_current_experiment() -> None:
            """Finalize the current experiment block before starting the next one."""
            nonlocal records, epoch_metrics, test_metrics, config
            if not (records or epoch_metrics or test_metrics or config):
                return

            flattened_metrics = [
                {"epoch": epoch, **metrics} for epoch, metrics in epoch_metrics.items()
            ]
            experiments.append(
                ExperimentLog(
                    records=pl.DataFrame(records),
                    epoch_metrics=pl.DataFrame(flattened_metrics),
                    test_metrics=pl.DataFrame(test_metrics),
                    config=config,
                )
            )
            records = []
            epoch_metrics = defaultdict(dict)
            test_metrics = {}
            config = {}

        with Path(path).open("r", encoding="utf-8") as f:
            for line in f:
                match = self.LOG_PATTERN.match(line)
                if not match:
                    continue

                entry = match.groupdict()
                message = entry["message"]

                # Check for experiment configuration
                if message.startswith("Experiment configuration: "):
                    flush_current_experiment()
                    config_str = message[len("Experiment configuration: ") :]
                    config = json.loads(config_str)
                    continue

                # Check for epoch metrics
                epoch_match = self.EPOCH_PATTERN.match(message)
                if epoch_match:
                    epoch = int(epoch_match.group("epoch"))
                    for m in self.METRIC_PATTERN.finditer(epoch_match.group("metrics")):
                        if m.group("key").startswith("test_"):
                            test_metrics[m.group("key")[5:]] = float(m.group("value"))
                        else:
                            epoch_metrics[epoch][m.group("key")] = float(m.group("value"))
                else:
                    entry["date"] = self._parse_date(entry["date"])
                    entry["line"] = int(entry["line"])
                    records.append(entry)

        flush_current_experiment()
        return experiments

    @staticmethod
    def _parse_date(date_str: str) -> datetime | str:
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S,%f"):
            try:
                return datetime.strptime(date_str.strip(), fmt).astimezone()
            except ValueError:
                continue
        return date_str.strip()
