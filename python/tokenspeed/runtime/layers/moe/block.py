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

"""MoE block shared by models whose shared expert reads the routed experts' input."""

from __future__ import annotations

import enum
from collections.abc import Sequence

import torch
from tokenspeed_kernel.ops.moe.cuda import moe_finalize_fuse_shared
from torch import nn

from tokenspeed.runtime.execution.forward_step import (
    get_is_capture_mode,
    get_is_cuda_graph_phase,
)
from tokenspeed.runtime.layers.moe.expert import MoELayer
from tokenspeed.runtime.layers.moe.topk import TopK
from tokenspeed.runtime.utils.cuda_stream import StreamFork


class MoEBlockFusion(enum.Enum):
    """A kernel that fuses steps of the MoE block."""

    # One epilogue reduces the routed rows and adds the shared output.
    FINALIZE_ADD_SHARED = "finalize_add_shared"


class MoEBlock(nn.Module):
    """``experts(topk(router_logits(x))) + shared_experts(x)``.

    The caller builds the router, top-k, routed experts and optional shared
    expert, which fixes their parallel layout and quantization; the routed
    and shared outputs are partial sums the caller reduces. Under CUDA graphs
    the shared expert runs on ``alt_stream`` beside the router and routed
    experts. ``fusions`` lists the fused kernels validated for the model,
    since each rounds differently from the separate kernels.
    """

    def __init__(
        self,
        gate: nn.Module,
        topk: TopK,
        experts: MoELayer,
        shared_experts: nn.Module | None,
        alt_stream: torch.cuda.Stream | None,
        fusions: Sequence[MoEBlockFusion],
    ) -> None:
        super().__init__()
        self.gate = gate
        self.shared_experts = shared_experts
        self.experts = experts
        self.topk = topk
        self.stream_fork = StreamFork(alt_stream)
        self.finalize_add_shared = MoEBlockFusion.FINALIZE_ADD_SHARED in fusions

    def router_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """The ``[tokens, experts]`` router logits for ``hidden_states``."""
        return self.gate(hidden_states)

    def get_moe_routed_weights(self) -> list[torch.Tensor]:
        """The routed experts' slot tensors, ``[num_local, ...]`` each."""
        return [
            x.data
            for name, x in self.experts.named_parameters()
            if name not in ["correction_bias"] and "shared_experts" not in name
        ]

    def forward(
        self,
        hidden_states: torch.Tensor,
        num_global_tokens: int,
        max_num_tokens_per_gpu: int,
    ) -> torch.Tensor:
        num_tokens = hidden_states.shape[0]
        # Checked per call: a model may disable deferral after construction.
        deferred = (
            self.finalize_add_shared and self.experts.supports_deferred_cuda_finalize
        )

        # Warm the shared-expert branch serially on its capture stream, then
        # overlap it with routed experts during capture.
        with self.stream_fork.scope(
            enable=get_is_cuda_graph_phase(), overlap=get_is_capture_mode()
        ) as fork:
            router_logits = self.router_logits(hidden_states)
            if num_tokens > 0:
                topk_output = self.topk(hidden_states, router_logits)
            else:
                topk_output = self.topk.empty_topk_output(
                    hidden_states.device,
                    hidden_states=hidden_states,
                    router_logits=router_logits,
                )

            routed_output = self.experts(
                hidden_states=hidden_states,
                topk_output=topk_output,
                num_global_tokens=num_global_tokens,
                max_num_tokens_per_gpu=max_num_tokens_per_gpu,
                do_finalize=not deferred,
            )

            shared_output = None
            with fork.branch():
                if self.shared_experts is not None and num_tokens > 0:
                    shared_output = self.shared_experts(hidden_states)

        if deferred:
            gemm2_out, expert_weights, expanded_idx = routed_output
            return moe_finalize_fuse_shared(
                gemm2_out,
                expanded_idx,
                expert_weights,
                shared_output,
                top_k=self.topk.topk_config.top_k,
                hidden_dim=hidden_states.shape[-1],
            )
        if shared_output is None:
            return routed_output
        return routed_output + shared_output
