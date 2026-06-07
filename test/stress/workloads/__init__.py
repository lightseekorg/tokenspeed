"""Workload registry.

A workload is any async iterator that yields ChatRequest objects. Register
new ones with @register so the CLI can address them by name.
"""

from __future__ import annotations

from typing import AsyncIterator, Callable, Dict

from ..client import ChatRequest

WorkloadFactory = Callable[..., AsyncIterator[ChatRequest]]
_REGISTRY: Dict[str, WorkloadFactory] = {}


def register(name: str) -> Callable[[WorkloadFactory], WorkloadFactory]:
    def deco(fn: WorkloadFactory) -> WorkloadFactory:
        _REGISTRY[name] = fn
        return fn

    return deco


def get(name: str) -> WorkloadFactory:
    if name not in _REGISTRY:
        raise KeyError(f"unknown workload: {name}; known: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def known() -> list[str]:
    return sorted(_REGISTRY)


# Import side-effects register the built-in workloads.
from . import cancel_mix  # noqa: F401,E402
from . import grammar_burst_drain  # noqa: F401,E402
from . import grammar_cancel_mix  # noqa: F401,E402
from . import grammar_invalid_schema  # noqa: F401,E402
from . import grammar_schema  # noqa: F401,E402
from . import grammar_schema_diverse  # noqa: F401,E402
from . import long_context  # noqa: F401,E402
from . import reality_mix  # noqa: F401,E402
from . import retract_stress  # noqa: F401,E402
from . import shared_prefix  # noqa: F401,E402
