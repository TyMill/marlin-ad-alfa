from __future__ import annotations

from dataclasses import dataclass

from marlin_ad.alerting.formatter import Alert, format_alert


@dataclass(frozen=True)
class StdoutAlertSink:
    """Write formatted alerts to stdout."""

    def send(self, alert: Alert) -> None:
        print(format_alert(alert))

    def send_many(self, alerts: list[Alert]) -> None:
        for alert in alerts:
            self.send(alert)


def send(alert: Alert) -> None:
    """Backward-compatible functional sink API."""

    StdoutAlertSink().send(alert)
