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

"""FlashInfer NVLink one-sided MoE all-to-all for one expert-parallel group."""

import torch
from tokenspeed_kernel.platform import pdl_enabled


class FlashInferMoeAlltoAll:
    def __init__(
        self,
        group: torch.distributed.ProcessGroup,
        max_tokens: int,
        hidden_size: int,
        top_k: int,
        num_experts: int,
        dtype: torch.dtype,
    ) -> None:
        from flashinfer.moe_ep import (
            BootstrapConfig,
            MoEEpCommParams,
            NVLinkOneSidedConfig,
            create_communication,
        )

        ep_rank, ep_size = group.rank(), group.size()
        if num_experts % ep_size:
            raise ValueError(
                "MoE all-to-all requires an even contiguous expert partition"
            )
        # Rows are sized for unquantized activations, which also covers packed
        # NVFP4 plus its block scales. FlashInfer builds its host-side layout
        # tensors on the default device, which a model loader may set to CUDA.
        with torch.device("cpu"):
            self.communication = create_communication(
                BootstrapConfig(
                    world_size=ep_size,
                    rank=ep_rank,
                    device=torch.cuda.current_device(),
                    process_group=group,
                ),
                MoEEpCommParams(
                    num_experts=num_experts,
                    top_k=top_k,
                    max_tokens_per_rank=max_tokens,
                    hidden_size=hidden_size,
                    dtype=dtype,
                ),
                NVLinkOneSidedConfig(enable_pdl=pdl_enabled()),
            )

    def dispatch(
        self,
        hidden_states: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        max_tokens: int,
    ) -> tuple[
        torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        torch.Tensor,
        torch.Tensor,
    ]:
        """Send local tokens to the ranks owning their experts.

        Args:
            hidden_states: ``[local_tokens, width]`` activations, or an NVFP4
                ``(packed, scales)`` pair.
            topk_ids: ``[local_tokens, top_k]`` global expert ids.
            topk_weights: ``[local_tokens, top_k]`` routing weights.
            max_tokens: Largest local token count over all ranks this step;
                every rank passes the same value.

        Returns:
            Received activations (same form as ``hidden_states``), expert ids
            and routing weights, each with ``ep_size * max_tokens`` rows. They
            are views of the transport's buffers, valid until ``combine``.
            Unused expert slots carry -1 and must contribute zero.
        """
        prequantized = isinstance(hidden_states, tuple)
        received = self.communication.dispatch(
            hidden_states[0] if prequantized else hidden_states,
            topk_ids,
            topk_weights,
            hidden_states_scale=hidden_states[1] if prequantized else None,
            runtime_max_tokens_per_rank=max_tokens,
        )
        routed_input = received.hidden_states
        if prequantized:
            routed_input = (routed_input, received.hidden_states_scale)
        return routed_input, received.topk_ids, received.topk_weights

    def combine(self, routed_output: torch.Tensor) -> torch.Tensor:
        """Sum already-weighted rank outputs of the last dispatch.

        Args:
            routed_output: ``[ep_size * max_tokens, hidden_size]`` outputs for
                the rows ``dispatch`` returned.

        Returns:
            ``[local_tokens, hidden_size]`` outputs in the dispatched order.
        """
        return self.communication.combine(routed_output)
