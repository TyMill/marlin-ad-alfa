from __future__ import annotations

from marlin_ad.alerting.sinks.stdout import StdoutAlertSink
from marlin_ad.detection.classical.isolation_forest import IsolationForestDetector
from marlin_ad.plugins.registry import PluginSpec, register


def register_builtin_plugins() -> None:
    """Register built-in plugin examples."""

    register(
        PluginSpec(
            name="isolation_forest",
            kind="detector",
            factory=IsolationForestDetector,
        ),
        replace=True,
    )
    register(
        PluginSpec(
            name="stdout_sink",
            kind="alert_sink",
            factory=StdoutAlertSink,
        ),
        replace=True,
    )
