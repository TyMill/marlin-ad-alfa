from __future__ import annotations

from importlib import metadata
from typing import Iterable

from marlin_ad.plugins.registry import PluginSpec, register


def load_entrypoint_plugins(group: str = "marlin_ad.plugins") -> list[str]:
    """Load plugins from Python entry points."""

    loaded: list[str] = []
    entry_points = metadata.entry_points()
    if hasattr(entry_points, "select"):
        selected: Iterable[metadata.EntryPoint] = entry_points.select(group=group)
    elif isinstance(entry_points, dict):
        selected = entry_points.get(group, [])
    else:
        selected = [ep for ep in entry_points if ep.group == group]

    for entry_point in selected:
        factory = entry_point.load()
        register(
            PluginSpec(name=entry_point.name, kind="external", factory=factory),
            replace=True,
        )
        loaded.append(entry_point.name)
    return loaded
