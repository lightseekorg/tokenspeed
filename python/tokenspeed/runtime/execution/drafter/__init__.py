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

"""Drafter registry: resolve a speculative algorithm to its drafter class.

The in-tree algorithms stay lazily imported (drafter modules pull in kernel
ops and model code, and this package init must stay importable from
lightweight contexts). Plugins extend the table through
:func:`register_drafter`: either a new algorithm name, or a draft-model-class
scoped entry that specializes an existing algorithm the way the in-tree
Inkling/DFlash2/DSpark cases do.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

    from tokenspeed.runtime.execution.drafter.base import BaseDrafter

__all__ = [
    "get_drafter_impl",
    "register_drafter",
    "registered_drafter_algorithms",
    "validate_drafter_algorithm",
]

#: Algorithms the in-tree resolution below serves.
_IN_TREE_ALGORITHMS = frozenset({"EAGLE3", "MTP", "DFLASH", "DSPARK"})


@dataclass
class _AlgorithmEntries:
    """Plugin entries for one algorithm: class-scoped first, then a default."""

    scoped: list[tuple[type, type]] = field(default_factory=list)
    default: type | None = None
    defaults_to_base_checkpoint: bool = False


_PLUGIN_DRAFTERS: dict[str, _AlgorithmEntries] = {}


def register_drafter(
    algorithm: str,
    drafter_cls: type,
    *,
    model_cls: type | None = None,
    defaults_to_base_checkpoint: bool = False,
    override: bool = False,
) -> None:
    """Register a drafter class for a speculative algorithm.

    Args:
        algorithm: The ``--speculative-algorithm`` name. A new name adds an
            algorithm; an in-tree name may only be extended with a
            ``model_cls``-scoped entry (or replaced with ``override=True``).
        drafter_cls: The ``BaseDrafter`` subclass to instantiate.
        model_cls: When given, the entry applies only to draft models that
            are instances of this class; scoped entries win over the
            algorithm default, most recently registered first.
        defaults_to_base_checkpoint: Whether launches of this algorithm
            without ``--speculative-draft-model-path`` should read the draft
            weights from the base checkpoint. Recorded for introspection;
            server-args resolution runs before plugins load, so launches of a
            plugin algorithm still pass ``--draft-model-path-use-base``.
        override: Allow replacing an existing default entry (in-tree or
            plugin). Overrides are logged by the plugin loader.

    Raises:
        ValueError: An unscoped registration collides without ``override``.
    """
    existing = _PLUGIN_DRAFTERS.get(algorithm)
    if model_cls is None:
        taken = (existing is not None and existing.default is not None) or (
            algorithm in _IN_TREE_ALGORITHMS
        )
        if taken and not override:
            raise ValueError(
                f"drafter algorithm {algorithm!r} is already registered; pass "
                "override=True to replace it, or scope the entry with "
                "model_cls"
            )
    entries = _PLUGIN_DRAFTERS.get(algorithm)
    created = entries is None
    if entries is None:
        entries = _PLUGIN_DRAFTERS[algorithm] = _AlgorithmEntries()
    # Snapshot for the rollback: undoing must RESTORE what an override
    # replaced (an earlier plugin's default, the base-checkpoint flag), not
    # blank it — undos run in reverse registration order, so each one puts
    # back exactly the state it saw.
    prior_default = entries.default
    prior_defaults_to_base = entries.defaults_to_base_checkpoint
    if model_cls is not None:
        entries.scoped.insert(0, (model_cls, drafter_cls))
    else:
        entries.default = drafter_cls
    if defaults_to_base_checkpoint:
        entries.defaults_to_base_checkpoint = True

    def undo() -> None:
        if model_cls is not None:
            try:
                entries.scoped.remove((model_cls, drafter_cls))
            except ValueError:
                pass
        else:
            entries.default = prior_default
        entries.defaults_to_base_checkpoint = prior_defaults_to_base
        if created and entries.default is None and not entries.scoped:
            _PLUGIN_DRAFTERS.pop(algorithm, None)

    from tokenspeed.runtime.plugins import registry as plugin_registry

    plugin_registry.record_external(
        "drafter",
        algorithm if model_cls is None else f"{algorithm}[{model_cls.__name__}]",
        undo,
    )


def registered_drafter_algorithms() -> frozenset[str]:
    """Every speculative algorithm name, in-tree and plugin-registered."""
    return _IN_TREE_ALGORITHMS | frozenset(_PLUGIN_DRAFTERS)


def validate_drafter_algorithm(name: str | None) -> None:
    """Reject an unknown ``--speculative-algorithm`` after plugin discovery.

    Args:
        name: The launch's algorithm name, or None when speculation is off.

    Raises:
        ValueError: The name matches no registered algorithm.
    """
    if name is None:
        return
    known = registered_drafter_algorithms()
    if name not in known:
        raise ValueError(
            f"unknown --speculative-algorithm {name!r}; available: " f"{sorted(known)}"
        )


def get_drafter_impl(spec_algo: str, model: torch.nn.Module) -> type[BaseDrafter]:
    """Resolve the drafter class for ``spec_algo`` and a loaded draft model.

    Args:
        spec_algo: The speculative algorithm name from server args.
        model: The loaded draft model; some algorithms route on its class.

    Returns:
        The ``BaseDrafter`` subclass to instantiate (not an instance).
    """
    plugin = _PLUGIN_DRAFTERS.get(spec_algo)
    if plugin is not None:
        for model_cls, drafter_cls in plugin.scoped:
            if isinstance(model, model_cls):
                return drafter_cls
        if plugin.default is not None:
            return plugin.default

    validate_drafter_algorithm(spec_algo)

    # Imports are local: drafter modules pull in kernel ops and model code,
    # and this package init must stay importable from lightweight contexts.
    from tokenspeed.runtime.execution.drafter.dflash import DFlash
    from tokenspeed.runtime.execution.drafter.dspark import DSpark
    from tokenspeed.runtime.execution.drafter.eagle import Eagle
    from tokenspeed.runtime.models.inkling_nextn import (
        InklingForConditionalGenerationNextN,
    )

    DRAFTER_MAPPING = {
        "EAGLE3": Eagle,
        "MTP": Eagle,
        "DFLASH": DFlash,
        "DSPARK": DSpark,
    }

    if spec_algo == "DFLASH":
        from tokenspeed.runtime.execution.drafter.dflash2 import DFlash2
        from tokenspeed.runtime.models.dflash2 import DFlash2DraftModel

        if isinstance(model, DFlash2DraftModel):
            return DFlash2

    # "MTP" covers two algorithms:
    # (1) Eagle-like MTP (e.g. DeepSeek) stays on Eagle in eagle.py;
    # (2) Vanilla MTP (e.g. Inkling) with multi-layer weights stays on Mtp in mtp.py.
    if spec_algo == "DSPARK":
        from tokenspeed.runtime.execution.drafter.deepseek_v4_dspark import (
            DeepseekV4DSpark,
        )
        from tokenspeed.runtime.execution.drafter.deepseek_v41_dspark import (
            DeepseekV41DSpark,
        )
        from tokenspeed.runtime.models.deepseek_v4_dspark import (
            DeepseekV4ForCausalLMDSpark,
        )
        from tokenspeed.runtime.models.deepseek_v41_dspark import (
            DeepseekV41ForCausalLMDSpark,
        )

        if isinstance(model, DeepseekV41ForCausalLMDSpark):
            return DeepseekV41DSpark
        if isinstance(model, DeepseekV4ForCausalLMDSpark):
            return DeepseekV4DSpark
    if spec_algo == "MTP" and isinstance(model, InklingForConditionalGenerationNextN):
        from tokenspeed.runtime.execution.drafter.mtp import Mtp

        return Mtp
    return DRAFTER_MAPPING[spec_algo]
