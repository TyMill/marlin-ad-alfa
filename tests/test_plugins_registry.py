from __future__ import annotations

import pytest

from marlin_ad.detection.classical.isolation_forest import IsolationForestDetector
from marlin_ad.plugins import create_plugin, list_plugins
from marlin_ad.plugins.registry import PluginRegistry, PluginSpec


def test_plugin_registry_create_and_list() -> None:
    registry = PluginRegistry()
    registry.register(PluginSpec(name="demo", kind="detector", factory=IsolationForestDetector))

    assert registry.list_plugins() == ["demo"]
    assert registry.list_plugins(kind="detector") == ["demo"]
    created = registry.create("demo", n_estimators=10)
    assert isinstance(created, IsolationForestDetector)
    assert created.n_estimators == 10


def test_plugin_registry_rejects_duplicates() -> None:
    registry = PluginRegistry()
    spec = PluginSpec(name="demo", kind="detector", factory=IsolationForestDetector)
    registry.register(spec)
    with pytest.raises(ValueError):
        registry.register(spec)


def test_builtin_plugin_registered() -> None:
    assert "isolation_forest" in list_plugins(kind="detector")
    created = create_plugin("isolation_forest", n_estimators=20)
    assert isinstance(created, IsolationForestDetector)
