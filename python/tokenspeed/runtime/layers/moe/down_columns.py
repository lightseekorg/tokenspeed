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

"""A second copy of MXFP4 expert down projections, split by output columns.

Expert tensor parallelism normally splits the down projection by its input
(intermediate) columns, so each rank produces a partial sum of every route
that ranks must add. With this copy, rank r holds output columns
``[r * H / P, (r + 1) * H / P)`` over the full intermediate: once every rank
has the complete intermediate rows, each produces complete route values for
its columns. The copy costs one more down projection of memory per rank.
"""

import torch
from tokenspeed_kernel.ops.moe import moe_sorted_down, prepare_moe_sorted_down_weight
from torch import nn

_MXFP4_BLOCK = 32


class MoEDownColumns(nn.Module):
    """One rank's output-column slice of every local expert's down projection.

    Args:
        num_experts: Local experts.
        hidden_size: Down projection output width H.
        intermediate_size: Full (unsharded) intermediate width I.
        tp_rank: This rank's index in the expert tensor-parallel group.
        tp_size: Expert tensor-parallel group size P; H must divide by it.
    """

    def __init__(
        self,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        tp_rank: int,
        tp_size: int,
    ) -> None:
        super().__init__()
        if hidden_size % tp_size:
            raise ValueError("the down projection's columns must divide across ranks")
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.width = hidden_size // tp_size
        self.weight = nn.Parameter(
            torch.zeros(
                num_experts, self.width, intermediate_size // 2, dtype=torch.uint8
            ),
            requires_grad=False,
        )
        self.weight_scale = nn.Parameter(
            torch.zeros(
                num_experts,
                self.width,
                intermediate_size // _MXFP4_BLOCK,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        self.prepared: torch.Tensor | None = None

    def shard(self, loaded: torch.Tensor) -> torch.Tensor:
        """Select this rank's output rows of one expert's full down weight or scales."""
        return loaded.narrow(-2, self.tp_rank * self.width, self.width)

    def process_weights_after_loading(self, module: nn.Module) -> None:
        """Lay out the loaded copy for the kernel and release the checkpoint layout."""
        del module  # The model loader passes this module to its own hook.
        self.prepared = prepare_moe_sorted_down_weight(
            self.weight.data, self.weight_scale.data
        )
        # The kernel layout is a separate copy; keep empty loaded parameters.
        self.weight.data = self.weight.data.new_empty((0,))
        self.weight_scale.data = self.weight_scale.data.new_empty((0,))

    def forward(
        self,
        intermediate: torch.Tensor,
        intermediate_scale: torch.Tensor,
        routes,
        output: torch.Tensor,
    ) -> torch.Tensor:
        """Project complete sorted intermediate rows into this rank's columns.

        Args:
            intermediate: Sorted MXFP4 rows over the full intermediate.
            intermediate_scale: Their block scales.
            routes: The sorted-route handle from ``moe_sorted_gate_up``.
            output: BF16 ``[tokens * top_k, H / P]`` destination.

        Returns:
            ``output`` with unweighted routes in token/slot order.
        """
        assert self.prepared is not None, "weights were not processed"
        return moe_sorted_down(
            intermediate, intermediate_scale, routes, self.prepared, output
        )
