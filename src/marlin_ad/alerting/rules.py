from __future__ import annotations

from dataclasses import dataclass
from operator import eq, ge, gt, le, lt, ne
from typing import Callable, Iterable, Mapping

from marlin_ad.alerting.formatter import Alert

_COMPARATORS: Mapping[str, Callable[[float, float], bool]] = {
    "gte": ge,
    "gt": gt,
    "lte": le,
    "lt": lt,
    "eq": eq,
    "ne": ne,
}


@dataclass(frozen=True)
class AlertRule:
    """Threshold-based rule for turning metrics into structured alerts."""

    name: str
    threshold: float
    metric_key: str
    comparator: str = "gte"
    severity: str = "warning"
    source: str = "rule_engine"

    def evaluate(self, metrics: Mapping[str, float]) -> bool:
        metric_value = float(metrics.get(self.metric_key, 0.0))
        op = _COMPARATORS.get(self.comparator)
        if op is None:
            valid = ", ".join(sorted(_COMPARATORS))
            raise ValueError(f"Unsupported comparator '{self.comparator}'. Expected one of: {valid}.")
        return op(metric_value, self.threshold)

    def build_alert(self, metrics: Mapping[str, float]) -> Alert:
        payload = {
            "metric_key": self.metric_key,
            "metric_value": float(metrics.get(self.metric_key, 0.0)),
            "threshold": self.threshold,
            "comparator": self.comparator,
        }
        return Alert(name=self.name, payload=payload, severity=self.severity, source=self.source)


def evaluate_rules(rules: Iterable[AlertRule], metrics: Mapping[str, float]) -> list[Alert]:
    """Evaluate a list of rules against a metrics mapping."""

    alerts: list[Alert] = []
    for rule in rules:
        if rule.evaluate(metrics):
            alerts.append(rule.build_alert(metrics))
    return alerts
