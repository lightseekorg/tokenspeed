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
so the frontend, the scheduler and the encode loop all resolve plugin
architectures the same way. Loading also runs the kernel package's own plugin
discovery, which registers plugin kernels before any model selects one.
"""

from __future__ import annotations

import importlib.metadata as importlib_metadata
import os
import threading
import warnings
from dataclasses import dataclass

from tokenspeed.runtime.utils import get_colorful_logger

logger = get_colorful_logger(__name__)

ENTRY_POINT_GROUP = "tokenspeed.plugins"
DISABLE_ENV_VAR = "TOKENSPEED_DISABLE_PLUGINS"
# Bumped whenever a class or protocol on the plugin contract changes.
PLUGIN_API_VERSION = 1

__all__ = [
    "DISABLE_ENV_VAR",
    "ENTRY_POINT_GROUP",
    "PLUGIN_API_VERSION",
    "PluginInfo",
    "ensure_loaded",
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
    """

    name: str
    package: str
    version: str
    registrations: tuple[tuple[str, tuple[str, ...]], ...]

    def describe(self) -> str:
        """Render the registrations for the startup log."""
        return ", ".join(
            f"{kind}=[{', '.join(names)}]" for kind, names in self.registrations
        )


_loaded_plugins: dict[str, PluginInfo] = {}
_loaded = False
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


def ensure_loaded() -> list[PluginInfo]:
    """Load kernel and runtime plugins once per process.

    Imports the kernel package (built-in kernels register on import), runs
    its plugin discovery, then calls each ``tokenspeed.plugins`` entry point
    in name order. A plugin whose loading or ``register()`` raises is
    reported with a ``UserWarning``, its partial registrations are undone,
    and startup continues. Each loaded plugin is logged once with what it
    registered.

    Returns:
        The runtime plugins loaded in this process.
    """
    global _loaded
    if _loaded:
        return list_plugins()
    with _lock:
        if _loaded:
            return list_plugins()
        # Set first: a plugin that builds a ModelConfig while registering
        # re-enters through the reentrant lock and returns on the flag.
        _loaded = True

        from tokenspeed_kernel.plugins import discover_plugins

        from tokenspeed.runtime.plugins.registry import recording

        discover_plugins()
        disabled = _disabled_from_env()
        entry_points = sorted(
            importlib_metadata.entry_points(group=ENTRY_POINT_GROUP),
            key=lambda ep: ep.name,
        )
        for ep in entry_points:
            if ep.name in disabled:
                logger.info(f"Skipping disabled plugin {ep.name!r}")
                continue
            try:
                with recording() as record:
                    ep.load()()
            except Exception as exc:
                warnings.warn(
                    f"Failed to load plugin {ep.name!r}: {exc!r}",
                    stacklevel=2,
                )
                # The warning is the contract; the traceback is how an
                # operator finds the failing frame deep in the plugin.
                logger.warning(f"Plugin {ep.name!r} failed to load", exc_info=exc)
                continue
            package, version = _distribution(ep)
            info = PluginInfo(
                name=ep.name,
                package=package,
                version=version,
                registrations=tuple(
                    (kind, tuple(names)) for kind, names in record.names.items()
                ),
            )
            _loaded_plugins[ep.name] = info
            logger.info(
                f"Loaded plugin {ep.name!r} ({package} {version}): {info.describe()}"
            )
    return list_plugins()


def list_plugins() -> list[PluginInfo]:
    """Return the runtime plugins loaded in this process."""
    return list(_loaded_plugins.values())
