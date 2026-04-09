from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from marlin_ad.alerting.formatter import Alert, format_alert


@dataclass(frozen=True)
class FileAlertSink:
    """Append formatted alerts to a file."""

    path: str = "alerts.log"

    def send(self, alert: Alert) -> None:
        output_path = Path(self.path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("a", encoding="utf-8") as handle:
            handle.write(format_alert(alert) + "\n")

    def send_many(self, alerts: list[Alert]) -> None:
        for alert in alerts:
            self.send(alert)


def send(alert: Alert, path: str = "alerts.log") -> None:
    """Backward-compatible functional sink API."""

    FileAlertSink(path=path).send(alert)
