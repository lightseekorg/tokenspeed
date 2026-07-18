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

"""Out-of-tree kernel plugin system.

Third-party packages register kernel implementations via Python entry points
in the ``tokenspeed_kernel.plugins`` group. Loading is explicit: the host
application must call :func:`discover_plugins` (typically once at startup,
after built-in kernels have been imported) for installed plugins to take
effect. See ``README.md`` for details.
"""

from __future__ import annotations

import importlib.metadata as importlib_metadata
import logging
import os
import warnings
from dataclasses import dataclass, field

from tokenspeed_kernel.registry import KernelRegistry

logger = logging.getLogger(__name__)

ENTRY_POINT_GROUP = "tokenspeed_kernel.plugins"
DISABLE_ENV_VAR = "TOKENSPEED_KERNEL_DISABLE_PLUGINS"

__all__ = [
    "PluginInfo",
    "ENTRY_POINT_GROUP",
    "DISABLE_ENV_VAR",
    "discover_plugins",
    "list_plugins",
    "disable_plugin",
    "enable_plugin",
    "reset_plugins",
]


@dataclass(frozen=True)
class PluginInfo:
    """Metadata describing a discovered plugin."""

    name: str
    package: str
    version: str
    num_kernels: int
    kernel_names: tuple[str, ...] = field(default_factory=tuple)


_loaded_plugins: dict[str, PluginInfo] = {}
_disabled_plugins: set[str] = set()


def _disabled_from_env() -> set[str]:
    raw = os.environ.get(DISABLE_ENV_VAR, "")
    return {part.strip() for part in raw.split(",") if part.strip()}


def disable_plugin(name: str) -> None:
    """Mark a plugin as disabled. Future ``discover_plugins`` calls skip it."""
    _disabled_plugins.add(name)


def enable_plugin(name: str) -> None:
    """Remove a plugin from the disabled set (does not auto-load it)."""
    _disabled_plugins.discard(name)


def reset_plugins() -> None:
    """Clear all plugin tracking state (for testing)."""
    _loaded_plugins.clear()
    _disabled_plugins.clear()


def list_plugins() -> list[PluginInfo]:
    """Return metadata for all plugins loaded in this process."""
    return list(_loaded_plugins.values())


def _entry_points(group: str):
    """Compatibility shim across importlib.metadata API variants."""
    eps = importlib_metadata.entry_points()
    # Python 3.10+ exposes a SelectableGroups API.
    selector = getattr(eps, "select", None)
    if callable(selector):
        return list(selector(group=group))
    return list(eps.get(group, []))  # pragma: no cover - legacy path


def _resolve_package_and_version(ep) -> tuple[str, str]:
    """Best-effort distribution name and version for an entry point."""
    dist = getattr(ep, "dist", None)
    if dist is not None:
        meta = getattr(dist, "metadata", None)
        if meta is not None:
            name = meta.get("Name") or ""
        else:
            name = getattr(dist, "name", "") or ""
        version = getattr(dist, "version", "") or ""
        if name:
            return name, version
    # Fallback: derive from the entry point's module path.
    package = (ep.value or "").split(":", 1)[0].split(".", 1)[0]
    version = ""
    if package:
        try:
            version = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            version = ""
    return package, version


def _new_kernel_names(before: set[str], after: set[str]) -> tuple[str, ...]:
    return tuple(sorted(after - before))


def _check_priority_collisions(
    registry: KernelRegistry, new_names: tuple[str, ...]
) -> None:
    for name in new_names:
        spec = registry.get_by_name(name)
        if spec is None:
            continue
        peers = registry.get_for_operator(spec.family, spec.mode)
        same = [s for s in peers if s.priority == spec.priority and s.name != name]
        for other in same:
            warnings.warn(
                f"Plugin kernel '{name}' has equal priority ({spec.priority}) with "
                f"existing kernel '{other.name}' for {spec.family}.{spec.mode}. "
                "Set explicit priorities to make selection deterministic.",
                stacklevel=3,
            )


def discover_plugins(*, force: bool = False) -> list[PluginInfo]:
    """Discover and load installed kernel plugins via entry points.

    Plugins disabled via :func:`disable_plugin` or the
    ``TOKENSPEED_KERNEL_DISABLE_PLUGINS`` env var are skipped.

    Already-loaded plugins are skipped unless ``force=True``, in which case
    their ``register()`` is invoked again (useful for tests that reset the
    registry).
    """
    disabled = _disabled_plugins | _disabled_from_env()
    registry = KernelRegistry.get()
    loaded: list[PluginInfo] = []

    # Sort entry points alphabetically for deterministic load order. Plugins
    # registering with the same priority will then have a stable winner.
    eps = sorted(_entry_points(ENTRY_POINT_GROUP), key=lambda ep: ep.name)
    for ep in eps:
        if ep.name in disabled:
            logger.info("Skipping disabled kernel plugin %r", ep.name)
            continue
        if ep.name in _loaded_plugins and not force:
            continue

        before = set(registry._by_name.keys())
        try:
            register_fn = ep.load()
            register_fn()
        except Exception as exc:
            warnings.warn(
                f"Failed to load kernel plugin {ep.name!r}: {exc}",
                stacklevel=2,
            )
            continue

        new_names = _new_kernel_names(before, set(registry._by_name.keys()))
        _check_priority_collisions(registry, new_names)
        package, version = _resolve_package_and_version(ep)
        info = PluginInfo(
            name=ep.name,
            package=package,
            version=version,
            num_kernels=len(new_names),
            kernel_names=new_names,
        )
        _loaded_plugins[ep.name] = info
        loaded.append(info)
        logger.info(
            "Loaded kernel plugin %r (%s %s) registering %d kernels",
            ep.name,
            package,
            version,
            len(new_names),
        )

    return loaded
