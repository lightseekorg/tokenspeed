# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Out-of-tree runtime plugins.

An installed distribution extends an unmodified runtime by declaring an entry
point in the ``tokenspeed.plugins`` group whose target is a no-argument
``register()`` that calls the functions in
:mod:`tokenspeed.runtime.plugins.registry`. Plugins activate by installation;
``TOKENSPEED_DISABLE_PLUGINS=a,b`` skips entry points by name.

Every process that builds a ``ModelConfig`` calls :func:`ensure_loaded` first,
so the frontend, the engine ranks and the encode ranks all resolve plugin
architectures the same way. Loading also runs the kernel package's own plugin
discovery, which registers plugin kernels before any model selects one.

At worker startup, ``global_server_args_dict_update`` publishes the resolved
``disaggregation_mode`` before model construction and warmup. Model plugins
use that role to prepare topology-specific resources; a prefill worker must
not infer a decode role from graph settings or missing metadata.
"""

from __future__ import annotations

import importlib.metadata as importlib_metadata
import os
import threading
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module

from tokenspeed.runtime.utils import get_colorful_logger

logger = get_colorful_logger(__name__)

ENTRY_POINT_GROUP = "tokenspeed.plugins"
DISABLE_ENV_VAR = "TOKENSPEED_DISABLE_PLUGINS"
# Bumped whenever a class or protocol on the plugin contract changes.
PLUGIN_API_VERSION = 4

__all__ = [
    "DISABLE_ENV_VAR",
    "ENTRY_POINT_GROUP",
    "PLUGIN_API_VERSION",
    "PluginInfo",
    "ensure_loaded",
    "finish_headless_plugins",
    "list_plugins",
]


@dataclass(frozen=True)
class PluginInfo:
    """One loaded runtime plugin.

    Attributes:
        name: Entry-point name.
        package: Distribution that declared the entry point.
        version: Distribution version.
        registrations: ``(registry kind, registered names)`` pairs, in the
            order the plugin registered them.
        entry_point: Import target of the successful registration.
    """

    name: str
    package: str
    version: str
    registrations: tuple[tuple[str, tuple[str, ...]], ...]
    entry_point: str | None = None

    def describe(self) -> str:
        """Render the registrations for the startup log."""
        return ", ".join(
            f"{kind}=[{', '.join(names)}]" for kind, names in self.registrations
        )


_loaded_plugins: dict[tuple[str, str, str, str], PluginInfo] = {}
_loaded = False
_deferred_plugins: (
    tuple[tuple[importlib_metadata.EntryPoint, ...], list[Callable[[], None]]] | None
) = None
# Reentrant: a plugin that builds a ModelConfig while registering re-enters
# ensure_loaded on the same thread and must observe the _loaded flag instead
# of deadlocking on the loader lock.
_lock = threading.RLock()


def _disabled_from_env() -> set[str]:
    raw = os.environ.get(DISABLE_ENV_VAR, "")
    return {part.strip() for part in raw.split(",") if part.strip()}


def _distribution(ep: importlib_metadata.EntryPoint) -> tuple[str, str]:
    dist = ep.dist
    if dist is None:
        return ep.value.split(":", 1)[0].split(".", 1)[0], ""
    return dist.metadata["Name"], dist.version


def _discover_kernel_plugins() -> None:
    """Load the kernel plugin registry before regular runtime activation."""
    from tokenspeed_kernel.plugins import discover_plugins

    discover_plugins()


def _activate_plugin(ep: importlib_metadata.EntryPoint) -> bool:
    """Activate one runtime entry point and record its successful registrations."""
    from tokenspeed.runtime.plugins.registry import recording

    try:
        with recording() as record:
            ep.load()()
    except Exception as exc:
        warnings.warn(f"Failed to load plugin {ep.name!r}: {exc!r}", stacklevel=2)
        logger.warning(f"Plugin {ep.name!r} failed to load", exc_info=exc)
        return False
    package, version = _distribution(ep)
    info = PluginInfo(
        name=ep.name,
        package=package,
        version=version,
        registrations=tuple(
            (kind, tuple(names)) for kind, names in record.names.items()
        ),
        entry_point=ep.value,
    )
    _loaded_plugins[ep.name, package, version, ep.value] = info
    logger.info(f"Loaded plugin {ep.name!r} ({package} {version}): {info.describe()}")
    return True


def _prepare_headless_configuration(
    entries: list[importlib_metadata.EntryPoint],
) -> tuple[tuple[importlib_metadata.EntryPoint, ...], list[Callable[[], None]]] | None:
    """Prepare all enabled plugins only when each opts into lightweight startup.

    A plugin's hook must register its complete argument policy without relying
    on any runtime or kernel plugin's execution registrations. If any plugin
    lacks that contract, preserve eager kernel-before-runtime activation for
    the entire set, including plugins that might depend on each other.
    """
    if not entries:
        return None

    from tokenspeed.runtime.plugins.registry import recording

    try:
        hooks = [
            getattr(import_module(ep.module), "register_extra_server_args", None)
            for ep in entries
        ]
        if any(hook is None for hook in hooks):
            return None
        with recording() as record:
            for hook in hooks:
                hook()
    except Exception:
        logger.warning(
            "Plugin server argument registration failed; loading plugins eagerly",
            exc_info=True,
        )
        return None
    return tuple(entries), record.undo


def finish_headless_plugins() -> None:
    """Complete the supervisor's registration after independently loading ranks."""
    global _deferred_plugins
    with _lock:
        if _deferred_plugins is None:
            return
        entries, undo = _deferred_plugins
        _deferred_plugins = None
        for unregister in reversed(undo):
            unregister()
        _discover_kernel_plugins()
        for ep in entries:
            if not _activate_plugin(ep):
                raise RuntimeError(f"Deferred plugin {ep.name!r} failed to load")


def ensure_loaded(*, headless_config: bool = False) -> list[PluginInfo]:
    """Load kernel and runtime plugins once per process.

    Imports the kernel package (built-in kernels register on import), runs
    its plugin discovery, then calls each ``tokenspeed.plugins`` entry point
    in name order. A plugin whose loading or ``register()`` raises is
    reported with a ``UserWarning``, its partial registrations are undone,
    and startup continues. Each loaded plugin is logged once with what it
    registered.

    Args:
        headless_config: Allow CLI preparation to defer a compatible plugin's GPU
            execution registration until ``finish_headless_plugins``.

    Returns:
        The runtime plugins loaded in this process.
    """
    global _loaded, _deferred_plugins
    # No unlocked fast path: _loaded is set when loading STARTS, so another
    # thread reading it outside the lock would return with registries still
    # being populated. Other threads block here until the load completes.
    with _lock:
        if _loaded:
            return list_plugins()
        # Set first: a plugin that builds a ModelConfig while registering
        # re-enters through the reentrant lock and returns on the flag.
        _loaded = True

        disabled = _disabled_from_env()
        entry_points = sorted(
            importlib_metadata.entry_points(group=ENTRY_POINT_GROUP),
            key=lambda ep: ep.name,
        )
        if headless_config:
            _deferred_plugins = _prepare_headless_configuration(
                [ep for ep in entry_points if ep.name not in disabled]
            )
        if _deferred_plugins is not None:
            return list_plugins()
        _discover_kernel_plugins()
        for ep in entry_points:
            if ep.name in disabled:
                logger.info(f"Skipping disabled plugin {ep.name!r}")
                continue
            _activate_plugin(ep)
    return list_plugins()


def list_plugins() -> list[PluginInfo]:
    """Return the runtime plugins loaded in this process."""
    return sorted(_loaded_plugins.values(), key=lambda info: info.name)
