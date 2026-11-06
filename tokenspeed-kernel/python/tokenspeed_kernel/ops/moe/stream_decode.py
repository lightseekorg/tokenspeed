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


"""MoE gate/up and down projections of a few decode rows in one launch.

Some kernels replace the two-launch decode (gate/up, then down) at small row
counts by one launch that streams every touched expert's weights once.
:func:`prepare_moe_stream_decode` selects such a kernel for a layer geometry
and device and creates its launch state; the caller keeps the returned
:class:`MoeStreamDecode` for the device and calls it per launch with
the layer's processed experts. Its output is the unweighted per-route rows
the deferred-finalize MoE kernels return, so the caller's route reduction is
unchanged.
"""

from dataclasses import dataclass
from typing import Any

import torch
from tokenspeed_kernel.selection import (
    NoKernelFoundError,
    SelectedKernel,
    select_kernel,
)
from tokenspeed_kernel.signature import dense_format_signature

_SIGNATURE = dense_format_signature(("x",), (torch.uint8,))


@dataclass(frozen=True)
class MoeStreamDecode:
    """A selected one-launch MoE decode kernel and its state on one device.

    Launches through one object must run one after another (one stream, or
    graph replays captured on it); every MoE layer of the device may share
    it.
    """

    kernel: SelectedKernel
    state: Any

    def __call__(
        self,
        x_quantized: tuple[torch.Tensor, torch.Tensor],
        topk_ids: torch.Tensor,
        w: torch.nn.Module,
        deferred_out: torch.Tensor,
    ) -> torch.Tensor:
        """Write each route's unweighted down projection.

        Args:
            x_quantized: E2M1 rows ``[tokens, hidden / 2]`` and their
                CDNA4-swizzled E8M0 scales.
            topk_ids: Contiguous int32 expert ids ``[tokens, top_k]``; every
                expert is local.
            w: The processed MoE layer.
            deferred_out: Contiguous BF16 ``[tokens * top_k, hidden]``
                destination, in token/slot order.

        Returns:
            ``deferred_out``.
        """
        return self.kernel(self.state, x_quantized, topk_ids, w, deferred_out)

    def warmup(self, w: torch.nn.Module, rows: int) -> None:
        """Compile and run the kernel once for ``rows`` tokens on ``w``'s
        geometry, so serving and graph capture compile nothing."""
        self.kernel.impl._tokenspeed_warmup(self.state, w, rows)


def prepare_moe_stream_decode(
    device: torch.device,
    *,
    weight_dtype: str,
    hidden: int,
    ispp: int,
    num_experts: int,
    top_k: int,
    ep_size: int,
    num_cus: int,
) -> MoeStreamDecode | None:
    """Select a one-launch MoE decode kernel and create its launch state.

    Args:
        device: The GPU the launches run on (outside graph capture).
        weight_dtype: The experts' weight format (registry ``weight_dtype``
            trait, such as ``"mxfp4"``).
        hidden: The experts' hidden size.
        ispp: Intermediate size per partition (this rank's slice).
        num_experts: Local experts.
        top_k: Routes per token.
        ep_size: Expert parallel size.
        num_cus: Compute units of ``device``; a kernel may size its grid to
            them.

    Returns:
        The kernel with its state, or None when no kernel serves this
        geometry on this platform.
    """
    try:
        kernel = select_kernel(
            "moe",
            "stream_decode",
            _SIGNATURE,
            traits={
                "weight_dtype": weight_dtype,
                "hidden": hidden,
                "ispp": ispp,
                "num_experts": num_experts,
                "top_k": top_k,
                "ep_size": ep_size,
                "num_cus": num_cus,
            },
        )
    except NoKernelFoundError:
        return None
    return MoeStreamDecode(kernel, kernel.impl._tokenspeed_state(device))


__all__ = ["MoeStreamDecode", "prepare_moe_stream_decode"]
