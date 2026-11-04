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

"""Decode monokernel layers: the part of a runtime seam that every model
shares (``docs/design/monokernel.md``).

With ``--decode-monokernel``, a model builds each decoder layer from its
monokernel class: a subclass of the regular layer class with a
:class:`MonokernelLayer` mixin. Such a layer runs a decode step that its
kernels serve as one launch, and every other forward with the regular
layer's code. After loading, the model's seam packs every layer, raising
:class:`NoMonokernel` for a layer that it cannot pack, and gives all layers
one :class:`MonokernelState`. The model calls
:meth:`MonokernelState.begin_forward` before its layers and
:meth:`MonokernelState.end_forward` after them.
"""

from __future__ import annotations

import torch

from tokenspeed.runtime.execution.context import ForwardContext


class MonokernelState:
    """The state that all monokernel launches of one backbone share: the
    mailbox scratch, the step counter, and the values of the current
    forward.

    Args:
        device: The device of the launches.
        scratch_pairs: The int64 mailbox pairs of the scratch buffer, the
            most that one layer's launch needs.
    """

    def __init__(self, device: torch.device, scratch_pairs: int) -> None:
        self.scratch = torch.zeros(scratch_pairs, dtype=torch.int64, device=device)
        # The launches' tag and a count of a launch's finished CTAs (0
        # between launches).
        self.step = torch.zeros(2, dtype=torch.int32, device=device)
        # The current forward's values. begin_forward resets them.
        # Whether a layer launched.
        self.launched = False
        # The input of the LM head, when the forward's last launch also ran
        # the model's final norm or mixer and moved the step counter.
        self.lm_head_input: torch.Tensor | None = None

    def begin_forward(self) -> None:
        """Start a forward, before its first layer."""
        self.launched = False
        self.lm_head_input = None

    def end_forward(self) -> torch.Tensor | None:
        """End a forward, after its last layer. Returns ``lm_head_input``.

        The next forward's launches need a new tag. A launch that sets
        ``lm_head_input`` moves the step counter as it ends. After other
        launches, this moves it (graph safe). A forward without launches adds
        no kernel.
        """
        if self.launched and self.lm_head_input is None:
            self.step[:1].add_(1)
        lm_head_input, self.lm_head_input = self.lm_head_input, None
        return lm_head_input


class NoMonokernel(Exception):
    """No monokernel can serve a layer. ``reasons`` says why."""

    def __init__(self, *reasons: str) -> None:
        super().__init__("; ".join(reasons))
        self.reasons = list(reasons)


# The reason when no kernel of this platform serves a layer (its prepare op
# returns None).
NO_KERNEL = "no monokernel for this platform"


def unserved_report(unserved: dict[int, list[str]]) -> str:
    """``unserved`` (layer index -> reasons) grouped by reason:
    ``"reason: layers 0, 2; other reason: layer 5"``."""
    layers_by_reason: dict[str, list[int]] = {}
    for index, reasons in sorted(unserved.items()):
        for reason in reasons:
            layers_by_reason.setdefault(reason, []).append(index)
    return "; ".join(
        f"{reason}: layer{'s' if len(layers) > 1 else ''} "
        f"{', '.join(str(i) for i in layers)}"
        for reason, layers in layers_by_reason.items()
    )


def unserved_error(name: str, unserved: dict[int, list[str]]) -> str:
    """The startup error of a backbone ``name`` whose layers ``unserved``
    (layer index -> reasons) have no monokernel."""
    return (
        f"--decode-monokernel: {len(unserved)} {name} layers have no monokernel "
        f"({unserved_report(unserved)}). Serve without --decode-monokernel."
    )


class MonokernelLayer:
    """A decoder layer that runs a decode step that its kernels serve as one
    monokernel launch, and every other forward with its regular code.

    Put a model's subclass of it before the regular layer class in the
    bases. The subclass gives :meth:`launch_chain` and :meth:`_decode`. A
    launch takes and returns the regular layer's boundary, so launches and
    regular forwards follow each other freely.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # The state that all layers share, set when the model's seam packs
        # the layers after loading.
        self.monokernel_state: MonokernelState | None = None

    def launch_chain(self, rows: int, ctx: ForwardContext) -> int | None:
        """The tokens per request (``Plan.chain``) of the launch for a
        forward of ``rows`` token rows, or None for the regular code. Fixed
        per CUDA graph."""
        raise NotImplementedError

    def forward(self, *, hidden_states: torch.Tensor, ctx: ForwardContext, **kwargs):
        chain = self.launch_chain(hidden_states.shape[0], ctx)
        if chain is None:
            return super().forward(hidden_states=hidden_states, ctx=ctx, **kwargs)
        if self.monokernel_state is None:
            raise RuntimeError(
                f"{type(self).__name__} {self.layer_id}: a decode step before "
                "post_quant_warmup packed the layer's monokernel"
            )
        self.monokernel_state.launched = True
        return self._decode(chain, hidden_states=hidden_states, ctx=ctx, **kwargs)

    def _decode(self, chain: int, **kwargs):
        """Run the launch for ``chain`` tokens per request. Takes the regular
        layer's forward arguments and returns its outputs."""
        raise NotImplementedError


__all__ = [
    "NO_KERNEL",
    "MonokernelLayer",
    "MonokernelState",
    "NoMonokernel",
    "unserved_error",
]
