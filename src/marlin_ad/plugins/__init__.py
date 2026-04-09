from marlin_ad.plugins.builtin import register_builtin_plugins
from marlin_ad.plugins.registry import (
    PluginRegistry,
    PluginSpec,
    clear_plugins,
    create_plugin,
    get_plugin,
    list_plugins,
    register,
)

register_builtin_plugins()

__all__ = [
    "PluginRegistry",
    "PluginSpec",
    "register",
    "get_plugin",
    "create_plugin",
    "list_plugins",
    "clear_plugins",
]
