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

"""Shared publication and completion lifecycle for fused shared-RS up GEMM."""

from abc import ABC, abstractmethod

import torch
from tokenspeed_kernel.ops.communication.mnnvl_cutedsl_symmetric_up_projection import (
    BoundSymmetricUpProjection,
    _rank_barrier_kernel,
)


class BoundFusedRsUpProjection(BoundSymmetricUpProjection, ABC):
    """One-stream raw symmetric partials → reduced/additive up GEMM → final AG.

    The input workspace and output must remain alive and non-overlapping until
    all captured graphs finish. Independent concurrent instances need independent
    workspaces. Sequential layers can reuse raw storage only after this plan's
    all-rank exit. There is no skip-publication option or standalone RS launch.
    """

    @abstractmethod
    def _compile(self) -> None:
        """Concrete bindings must compile a fused-RS kernel, not plain up GEMM."""
        raise NotImplementedError

    def stage(self, partial: torch.Tensor) -> torch.Tensor:
        """Copy BF16 [M,7168] into input_view; real producers may use out= instead."""
        if (
            partial.shape != self.input_view.shape
            or partial.dtype != self.input_view.dtype
            or partial.device != self.input_view.device
            or not partial.is_contiguous()
        ):
            raise ValueError("stage requires colocated contiguous BF16 [M,7168]")
        self.input_view.copy_(partial)
        return self.input_view

    def __call__(self) -> torch.Tensor:
        # The entry's release/acquire and alias fences publish raw partials,
        # including same-stream copies/producers issued before routed BT/HT.
        _rank_barrier_kernel[(1,)](
            self.workspace.state.symm_mem_hdl.signal_pad_ptrs_dev,
            RANK=self.output.rank,
            ENABLE_PDL=False,
            num_warps=4,
        )
        super().producer_only(multicast=True)
        # All CTAs have finished their raw reads before this complete-rank exit.
        # It protects both final output consumers and subsequent raw overwrites.
        self._barrier()
        return self.output.tensor

    def producer_only(self, *, multicast: bool) -> None:
        """Reject unsafe un-published standalone access to symmetric partials."""
        raise RuntimeError("use the complete fused plan; raw publication is mandatory")
