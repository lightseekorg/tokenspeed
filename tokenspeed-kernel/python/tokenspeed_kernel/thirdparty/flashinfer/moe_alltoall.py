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

"""FlashInfer MoE all-to-all with caller-owned MNNVL workspace."""

from types import SimpleNamespace

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
        weights_dtype: torch.dtype,
    ) -> None:
        from flashinfer.comm import (
            moe_a2a_get_workspace_size_per_rank,
            moe_a2a_initialize,
        )
        from flashinfer.comm.mnnvl import MnnvlConfig, MnnvlMemory, TorchDistBackend

        self.ep_rank = group.rank()
        self.ep_size = group.size()
        self.max_tokens = max_tokens
        self.hidden_size = hidden_size
        self.top_k = top_k
        self.num_experts = num_experts
        if num_experts % self.ep_size:
            raise ValueError(
                "MoE all-to-all requires an even contiguous expert partition"
            )
        # The BF16 capacity also covers packed NVFP4 plus its block scales.
        payload_bytes = hidden_size * (torch.finfo(dtype).bits // 8)
        routing_bytes = top_k * (4 + torch.finfo(weights_dtype).bits // 8)
        size = moe_a2a_get_workspace_size_per_rank(
            ep_size=self.ep_size,
            max_num_tokens=max_tokens,
            total_dispatch_payload_size_per_token=payload_bytes + routing_bytes,
            combine_payload_size_per_token=payload_bytes,
            eplb_stats_num_experts=0,
        )
        mapping = SimpleNamespace(pp_rank=0, cp_rank=0, cp_size=1, tp_rank=self.ep_rank)
        MnnvlMemory.initialize()
        MnnvlMemory.set_comm_from_config(
            mapping,
            MnnvlConfig(
                comm_backend=TorchDistBackend(group=group),
                allocation_granularity=0,
                fabric_page_size=1 << 29,
            ),
        )
        self.memory = MnnvlMemory(mapping, size)
        self.workspace = self.memory.as_torch_strided_tensor(torch.uint8)
        self.metainfo = moe_a2a_initialize(
            workspace=self.workspace,
            ep_rank=self.ep_rank,
            ep_size=self.ep_size,
            max_num_tokens=max_tokens,
            eplb_stats_num_experts=0,
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
        int,
    ]:
        """Dispatch local tokens or an NVFP4 pair, routing, and return flattened receives."""
        from flashinfer.comm import moe_a2a_dispatch, moe_a2a_sanitize_expert_ids

        if not 0 < max_tokens <= self.max_tokens:
            raise ValueError(
                "MoE all-to-all token count exceeds its workspace capacity"
            )
        prequantized = isinstance(hidden_states, tuple)
        input_payloads = [topk_ids, topk_weights]
        if prequantized:
            input_payloads.extend(hidden_states)
        else:
            input_payloads.append(hidden_states)
        payloads, offset, _ = moe_a2a_dispatch(
            token_selected_experts=topk_ids,
            input_payloads=input_payloads,
            workspace=self.workspace,
            metainfo=self.metainfo,
            runtime_max_tokens_per_rank=max_tokens,
            ep_rank=self.ep_rank,
            ep_size=self.ep_size,
            top_k=self.top_k,
            num_experts=self.num_experts,
            enable_pdl=pdl_enabled(),
            eplb_local_stats=None,
            enable_rank_mask=False,
            active_rank_mask=None,
        )
        ids, weights, hidden = payloads[:3]
        # A valid nonlocal ID makes every expert backend ignore unused receive slots.
        invalid_id = ((self.ep_rank + 1) % self.ep_size) * (
            self.num_experts // self.ep_size
        )
        moe_a2a_sanitize_expert_ids(
            expert_ids=ids,
            workspace=self.workspace,
            metainfo=self.metainfo,
            ep_rank=self.ep_rank,
            invalid_expert_id=invalid_id,
            enable_pdl=pdl_enabled(),
        )
        routed_input = hidden.view(-1, input_payloads[2].shape[1])
        if prequantized:
            routed_input = (
                routed_input,
                payloads[3].view(-1, input_payloads[3].shape[1]),
            )
        return (
            routed_input,
            ids.view(-1, self.top_k),
            weights.view(-1, self.top_k),
            offset,
        )

    def combine(
        self,
        routed_output: torch.Tensor,
        local_tokens: int,
        max_tokens: int,
        combine_offset: int,
    ) -> torch.Tensor:
        """Sum already-weighted rank outputs and restore the original local token order."""
        from flashinfer.comm import moe_a2a_combine
        from flashinfer.tllm_enums import SfLayout

        return moe_a2a_combine(
            payload=routed_output.view(self.ep_size, max_tokens, self.hidden_size),
            local_num_tokens=local_tokens,
            workspace=self.workspace,
            metainfo=self.metainfo,
            runtime_max_tokens_per_rank=max_tokens,
            ep_rank=self.ep_rank,
            ep_size=self.ep_size,
            top_k=self.top_k,
            combine_payload_offset=combine_offset,
            payload_in_workspace=False,
            output_dtype=routed_output.dtype,
            output_scales=None,
            output_scalar_scale=1.0,
            sf_layout=SfLayout.layout_linear,
            output=None,
            use_low_precision=False,
            enable_pdl=pdl_enabled(),
            enable_rank_mask=False,
            active_rank_mask=None,
        )
