from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class PluginSpec:
    """Definition for lightweight plugin registration."""

    name: str
    kind: str
    factory: Callable[..., Any]


class PluginRegistry:
    """In-memory registry for detector/monitor/sink plugin factories."""

    def __init__(self) -> None:
        self._registry: dict[str, PluginSpec] = {}

    def register(self, spec: PluginSpec, *, replace: bool = False) -> None:
        if not replace and spec.name in self._registry:
            raise ValueError(f"Plugin '{spec.name}' is already registered.")
        self._registry[spec.name] = spec

    def get(self, name: str) -> PluginSpec:
        if name not in self._registry:
            raise KeyError(f"Plugin '{name}' is not registered.")
        return self._registry[name]

    def create(self, name: str, **kwargs: Any) -> Any:
        return self.get(name).factory(**kwargs)

    def list_plugins(self, kind: str | None = None) -> list[str]:
        if kind is None:
            names: list[str] = list(self._registry.keys())
        else:
            names = [name for name, spec in self._registry.items() if spec.kind == kind]
        return sorted(names)

    def clear(self) -> None:
        self._registry.clear()


_DEFAULT_REGISTRY = PluginRegistry()


def register(spec: PluginSpec, *, replace: bool = False) -> None:
    _DEFAULT_REGISTRY.register(spec, replace=replace)


def get_plugin(name: str) -> PluginSpec:
    return _DEFAULT_REGISTRY.get(name)


def create_plugin(name: str, **kwargs: Any) -> Any:
    return _DEFAULT_REGISTRY.create(name, **kwargs)


def list_plugins(kind: str | None = None) -> list[str]:
    return _DEFAULT_REGISTRY.list_plugins(kind=kind)


def clear_plugins() -> None:
    _DEFAULT_REGISTRY.clear()
