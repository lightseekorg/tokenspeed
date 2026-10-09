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

"""Registry mapping model families to their decision adapter.

Names are explicit: an unknown family is an error, not a silent fallback
to the generic adapter — callers that want generic behavior ask for it by
name. Family aliases resolve to an adapter instance configured with the
family name (used in its error messages); a family that outgrows the
generic scaffold gets its own adapter class here.
"""

from __future__ import annotations

from collections.abc import Callable

from tokenspeed.runtime.decision.adapter import (
    DecisionAdapter,
    GenericDecisionAdapter,
)

# Family name (as used by ``ts serve`` model defaults) → adapter name.
# v1: every family shares the generic scaffold; the indirection exists so a
# family-specific adapter can replace one entry without touching callers.
_FAMILY_TO_ADAPTER: dict[str, str] = {
    "deepseek_v4": "generic",
    "deepseek_v41": "generic",
    "glm": "generic",
    "inkling": "generic",
    "kimi_k25": "generic",
    "kimi_k3": "generic",
    "minimax": "generic",
    "qwen3": "generic",
    "qwen3_5": "generic",
}

# Adapter name → factory taking the resolved family name. Adapter classes
# must accept a ``family: str`` constructor argument (used in their error
# messages); a plain class object is itself a valid factory.
AdapterFactory = Callable[[str], DecisionAdapter]
_ADAPTERS: dict[str, AdapterFactory] = {}


def register_decision_adapter(name: str, factory: AdapterFactory) -> None:
    """Register ``factory`` under ``name``, replacing any existing entry."""
    _ADAPTERS[name] = factory


def get_decision_adapter(name: str) -> DecisionAdapter:
    """Resolve an adapter by registry name or family alias.

    Raises ``ValueError`` for unknown names — a misspelled family must fail
    at request time, not silently pick up a mismatched scaffold.
    """
    adapter_name = name if name in _ADAPTERS else _FAMILY_TO_ADAPTER.get(name, name)
    factory = _ADAPTERS.get(adapter_name)
    if factory is None:
        available = sorted(set(_ADAPTERS) | set(_FAMILY_TO_ADAPTER))
        raise ValueError(f"Unknown decision adapter {name!r}; available: {available}.")
    # Aliases keep the family name for the adapter's messages; a bare
    # adapter name resolves with its own name.
    return factory(name if name in _FAMILY_TO_ADAPTER else adapter_name)


register_decision_adapter("generic", GenericDecisionAdapter)
