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

from __future__ import annotations

import functools
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from flashinfer.moe_ep import MoEEpMegaLayer, MoEEpMegaWorkspace


@functools.cache
def megamoe_unavailable_reason() -> str | None:
    """Check the optional Rubin API without allocating a workspace or compiling."""
    try:
        from flashinfer.moe_ep import (  # noqa: F401
            BootstrapConfig,
            FleetParams,
            MegaConfig,
            MoEEpLayer,
            MoEEpMegaWorkspace,
            MoEEpTensors,
            PrequantizedMoEWeights,
            Sm107_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig,
        )
        from flashinfer.moe_ep.kernel_src.sm107.next_cutedsl_megamoe import (  # noqa: F401
            preprocess_prequantized_block_scaled_weights,
        )
    except ImportError as exc:
        return f"FlashInfer Rubin MegaMoE is unavailable: {exc}"
    return None


@dataclass
class MegaMoeState:
    """Per-layer weights; FlashInfer pools scratch across stream-ordered layers."""

    layer: MoEEpMegaLayer
    hidden: int
    top_k: int
    workspace: MoEEpMegaWorkspace | None = None

    def validate_hidden(self, x: torch.Tensor) -> None:
        """Validate activations before routing or either expert backend."""
        if (
            x.ndim != 2
            or x.shape[1] != self.hidden
            or x.dtype != torch.bfloat16
            or not x.is_contiguous()
        ):
            raise ValueError("MegaMoE requires contiguous BF16 [tokens, hidden] input")

    def validate_inputs(
        self,
        x: torch.Tensor,
        ids: torch.Tensor,
        scores: torch.Tensor,
    ) -> None:
        """Validate activations and caller-provided routes before execution."""
        self.validate_hidden(x)
        shape = (x.shape[0], self.top_k)
        if (
            ids.shape != shape
            or scores.shape != shape
            or ids.dtype != torch.int32
            or scores.dtype != torch.float32
            or ids.device != x.device
            or scores.device != x.device
            or not ids.is_contiguous()
            or not scores.is_contiguous()
        ):
            raise ValueError(
                "MegaMoE requires contiguous int32 IDs and FP32 scores [tokens, top_k]"
            )

    def apply(
        self,
        x: torch.Tensor,
        ids: torch.Tensor,
        scores: torch.Tensor,
        *,
        capacity: int,
    ) -> torch.Tensor:
        from flashinfer.moe_ep import MoEEpTensors

        self.validate_inputs(x, ids, scores)
        if x.shape[0] == 0:
            return torch.empty_like(x)
        capacity = 1 << (max(capacity, x.shape[0]) - 1).bit_length()
        # Each returned output owns its storage: another layer may reuse scratch.
        tensors = MoEEpTensors(x, ids, scores, output=torch.empty_like(x))
        if self.workspace is None or capacity > self.workspace.max_tokens_per_rank:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "Warm MegaMoE at the required capacity before CUDA graph capture"
                )
            workspace = self.layer.create_workspace(capacity)
            self.layer.warmup(tensors, workspace=workspace)
            # Older handles stay owned by the layer for any captured graphs.
            self.workspace = workspace
        return self.layer.forward(tensors, workspace=self.workspace)


def prepare_megamoe(w: torch.nn.Module, *, fast_math: bool) -> MegaMoeState:
    """Repack NVFP4 weights in place and return their upstream execution state.

    The module's four weight/block-scale parameters share the transformed storage
    used by FlashInfer. The checkpoint layout is released; global scales and the
    RNE contract are preserved. This preparation must run before graph capture.
    """
    from flashinfer.moe_ep import (
        BootstrapConfig,
        FleetParams,
        MegaConfig,
        MoEEpLayer,
        PrequantizedMoEWeights,
        Sm107_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig,
    )
    from flashinfer.moe_ep.kernel_src.sm107.next_cutedsl_megamoe import (
        preprocess_prequantized_block_scaled_weights,
    )

    if w.tp_size != 1 or w.ep_size != 1:
        raise ValueError("FlashInfer MegaMoE NVFP4 requires MoE TP=EP=1")
    if os.environ.get("MEGA_NO_DIST") != "1":
        raise ValueError(
            "Set MEGA_NO_DIST=1 for the single-rank FlashInfer MegaMoE adapter"
        )
    if torch.distributed.is_initialized() and torch.distributed.get_world_size() != 1:
        raise ValueError(
            "FlashInfer MegaMoE adapter currently requires a single-rank process"
        )
    if not fast_math:
        raise ValueError("FlashInfer MegaMoE does not implement fast_math=False")
    experts, rows, packed_hidden = w.w13_weight.shape
    hidden, intermediate = packed_hidden * 2, rows // 2
    if rows % 2 or hidden % 128 or intermediate % 128:
        raise ValueError("FlashInfer MegaMoE NVFP4 requires H/I divisible by 128")
    gate_scale, up_scale = w.w13_weight_scale_2.unbind(dim=1)
    if not torch.equal(gate_scale, up_scale):
        raise ValueError("FlashInfer MegaMoE requires equal gate/up weight_scale_2")
    if w.w13_input_scale.numel() != 1:
        raise ValueError("FlashInfer MegaMoE requires a scalar FC1 input scale")
    for scale in (w.w13_input_scale, w.w2_input_scale, gate_scale, w.w2_weight_scale_2):
        if not bool((torch.isfinite(scale) & (scale > 0)).all()):
            raise ValueError("FlashInfer MegaMoE requires finite positive scales")
    # Match the measured Generic NVFP4 path: late FP32 scores, BF16 route
    # outputs, a separate deterministic reduce, and a fixed kernel configuration.
    config = Sm107_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig(
        intermediate_size=intermediate,
        top_k=w.top_k,
        kernel_variant="inference",
        activation="swiglu",
        fast_math=fast_math,
        apply_routing_weights_before_fc2=False,
        combine_dtype="bf16",
        in_kernel_fc2_reduce=False,
        knobs=None,
        mma_tiler_mnk=(256, 256, 256),
        cluster_shape_mn=(4, 1),
        fallback_cluster_shape_mn=(2, 1),
        schedule_policy=("phase_interleave", None),
        work_id_mode="atomic_counter",
        fc2_use_bulk=True,
        fc2_tma_stages=2,
        epi_flag_batches=(2, 4),
        token_in_flag_batch=1,
        token_back_mode="epi_warps",
        input_norm_const=w.w13_input_scale.float().reciprocal().item(),
        fc1_alpha=(w.w13_input_scale * gate_scale).float().contiguous(),
        fc2_alpha=(w.w2_input_scale * w.w2_weight_scale_2).float().contiguous(),
        fc1_norm_const=w.w2_input_scale.float().reciprocal().contiguous(),
    )
    pack = PrequantizedMoEWeights(
        w.w13_weight, w.w2_weight, w.w13_weight_scale, w.w2_weight_scale
    )
    transformed = preprocess_prequantized_block_scaled_weights(
        pack.w13,
        pack.w2,
        pack.w13_scale,
        pack.w2_scale,
        quant_kind="nvfp4",
        hidden_size=hidden,
        intermediate_size=intermediate,
    )
    layer = MoEEpLayer(
        bootstrap=BootstrapConfig(
            rank=0, world_size=1, device=w.w13_weight.device.index, auto_bootstrap=False
        ),
        # Weight preparation precedes resolution of the executor's token budget.
        # The explicit workspace is sized on the first eager/startup call.
        fleet_params=FleetParams(
            num_experts=experts, max_tokens_per_rank=1, token_hidden_size=hidden
        ),
        weights=pack,
        backend=MegaConfig(
            megakernel=config,
            quantize_input=True,
            preprocess_weights=False,
            transformed_weights=transformed,
        ),
    )
    # FlashInfer retains only transformed weights. Register aliases of that same
    # storage so model memory accounting sees the weights and the checkpoint
    # layout can be released. There is no second backend layout or data copy.
    w.w13_weight = torch.nn.Parameter(transformed[0][0], requires_grad=False)
    w.w13_weight_scale = torch.nn.Parameter(transformed[0][1], requires_grad=False)
    w.w2_weight = torch.nn.Parameter(transformed[1][0], requires_grad=False)
    w.w2_weight_scale = torch.nn.Parameter(transformed[1][1], requires_grad=False)
    return MegaMoeState(layer=layer, hidden=hidden, top_k=w.top_k)
