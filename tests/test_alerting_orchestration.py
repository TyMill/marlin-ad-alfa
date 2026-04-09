from __future__ import annotations

from marlin_ad.alerting.formatter import Alert, format_alert
from marlin_ad.alerting.rules import AlertRule, evaluate_rules


def test_alert_format_is_consistent() -> None:
    alert = Alert(
        name="drift:psi",
        severity="critical",
        source="pipeline",
        payload={"threshold": 0.2, "metric_value": 0.4},
        timestamp="2026-01-01T00:00:00+00:00",
    )
    rendered = format_alert(alert)
    assert "severity=CRITICAL" in rendered
    assert "source=pipeline" in rendered
    assert "name=drift:psi" in rendered


def test_alert_rules_support_comparators() -> None:
    metrics = {"summary.alert_count": 2.0}
    rules = [
        AlertRule(name="high_alert_count", metric_key="summary.alert_count", threshold=1.0),
        AlertRule(
            name="too_few_alerts",
            metric_key="summary.alert_count",
            threshold=4.0,
            comparator="lte",
        ),
    ]
    alerts = evaluate_rules(rules, metrics)
    names = {alert.name for alert in alerts}
    assert names == {"high_alert_count", "too_few_alerts"}
