from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Mapping


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class Alert:
    """Normalized alert envelope used by alert rules and sinks."""

    name: str
    payload: Mapping[str, object]
    severity: str = "warning"
    source: str = "marlin_ad"
    timestamp: str | None = None

    def with_timestamp(self) -> "Alert":
        if self.timestamp is not None:
            return self
        return Alert(
            name=self.name,
            payload=self.payload,
            severity=self.severity,
            source=self.source,
            timestamp=_utc_now_iso(),
        )


def _format_payload(payload: Mapping[str, object]) -> str:
    ordered_items = sorted(payload.items(), key=lambda item: item[0])
    body = ", ".join(f"{key}={value!r}" for key, value in ordered_items)
    return "{" + body + "}"


def format_alert(alert: Alert) -> str:
    """Render an alert into a deterministic single-line string."""

    stamped = alert.with_timestamp()
    return (
        f"{stamped.timestamp} | severity={stamped.severity.upper()} | "
        f"source={stamped.source} | name={stamped.name} | payload={_format_payload(stamped.payload)}"
    )


def format_alerts(alerts: list[Alert]) -> list[str]:
    """Format a list of alerts using consistent rendering semantics."""

    return [format_alert(alert) for alert in alerts]
