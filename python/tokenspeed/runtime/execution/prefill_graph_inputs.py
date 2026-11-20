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

"""Model-owned preparation and runner-owned storage for prefill graph inputs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import torch

    from tokenspeed.runtime.execution.context import ForwardContext
    from tokenspeed.runtime.multimodal.inputs import MultimodalForwardContext


def copy_prefill_graph_input(destination: torch.Tensor, source: torch.Tensor) -> None:
    """Refresh a captured input, preserving overlapping source views.

    Identical views need no copy. Other views of the same storage must be
    read before a write can overwrite their remaining source elements.
    """
    if (
        destination.is_set_to(source)
        and destination.dtype == source.dtype
        and destination.is_conj() == source.is_conj()
        and destination.is_neg() == source.is_neg()
    ):
        return
    if (
        destination.device == source.device
        and destination.untyped_storage().data_ptr()
        == source.untyped_storage().data_ptr()
    ):
        source = source.clone()
    destination.copy_(source)


@runtime_checkable
class PrefillGraphInputModel(Protocol):
    """Prepare all modality-dependent inputs before the token-only graph.

    Implementations cover every supported modality with one captured token
    path. Text-only batches must produce the same input structure. The runner
    allocates the returned token-leading buffers in a CPU-backed named region
    and refreshes their contents before replay.
    """

    def allocate_prefill_graph_inputs(self, num_tokens: int) -> dict[str, torch.Tensor]:
        """Return initialized token-leading buffers with capacity ``num_tokens``."""
        ...

    def prepare_prefill_graph_inputs(
        self,
        input_ids: torch.Tensor,
        ctx: ForwardContext,
        *,
        model_kwargs: dict[str, object],
        multimodal_context: MultimodalForwardContext | None,
    ) -> dict[str, object]:
        """Return live embeddings and model inputs without retaining batch state."""
        ...
