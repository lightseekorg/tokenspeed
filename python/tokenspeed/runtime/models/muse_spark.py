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

"""Muse Spark language and vision runtime using portable kernels."""

from __future__ import annotations

import re
from collections.abc import Iterable
from functools import partial
from types import SimpleNamespace
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
from tokenspeed_kernel.ops.gemm import (
    _use_mv,
    fp32_router_gemv_counters,
    fp32_router_gemv_execute,
    fp32_router_gemv_plan,
    fp32_router_gemv_process_weight,
    mv,
    mv_weightless_rmsnorm_gated_sigmoid,
    mv_weightless_rmsnorm_gated_sigmoid_supported,
)
from tokenspeed_kernel.ops.layernorm import (
    affine_rmsnorm,
    weightless_rmsnorm,
    weightless_rmsnorm_gated_sigmoid,
)
from tokenspeed_kernel.ops.moe import (
    FUSED_ROUTING_MAX_TOKENS,
    PreparedNvfp4Routing,
    bf16_moe_routes,
    bf16_moe_routes_warmup,
    gather_moe_routes,
    moe_sigmoid_bias_topk_routing,
    moe_topk,
    moe_warmup,
    prepare_nvfp4_routing,
    routed_input_scale_rmsnorm_sum,
    routed_input_scale_rmsnorm_sum_gathered,
)
from tokenspeed_kernel.ops.moe._cute_dsl.latent_down import KimiK3LatentDownOp
from tokenspeed_kernel.ops.quantization import rmsnorm_mxfp4_quant, rmsnorm_nvfp4_quant
from tokenspeed_kernel.ops.residual import (
    boundary_gathers_mailbox,
    post_norm_residual_pre_norm,
    post_norm_residual_pre_norm_bf16,
    residual_gate_coefficients,
)
from tokenspeed_kernel.platform import current_platform, pdl_enabled
from tokenspeed_kernel.selection import resolve_kernel_override
from torch import nn

from tokenspeed.runtime.configs.muse_spark_config import MuseSparkConfig
from tokenspeed.runtime.configs.numerics import BITWISE_ENVELOPES
from tokenspeed.runtime.distributed.comm_backend.registry import get_global_backend
from tokenspeed.runtime.distributed.comm_manager import CommManager
from tokenspeed.runtime.distributed.comm_ops import (
    acquire_all_reduce_outputs,
    all_gather,
    all_reduce,
    can_acquire_all_reduce_outputs,
)
from tokenspeed.runtime.distributed.mapping import Group, Mapping
from tokenspeed.runtime.distributed.process_group_manager import process_group_manager
from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.execution.forward_step import (
    get_is_capture_mode,
    get_is_cuda_graph_phase,
)
from tokenspeed.runtime.layers.dense import UnquantizedLinearMethod
from tokenspeed.runtime.layers.linear import (
    ColumnParallelLinear,
    LinearBase,
    RowParallelLinear,
)
from tokenspeed.runtime.layers.logits_processor import LogitsProcessor
from tokenspeed.runtime.layers.moe.expert import MoELayer
from tokenspeed.runtime.layers.moe.topk import StandardTopKOutput
from tokenspeed.runtime.layers.paged_attention import (
    PagedAttention,
    hf_sliding_window_to_window_left,
)
from tokenspeed.runtime.layers.quantization.base_config import QuantizationConfig
from tokenspeed.runtime.layers.quantization.utils import (
    should_exclude_quant_module,
    should_ignore_quant_layer,
)
from tokenspeed.runtime.layers.rotary_embedding import get_rope
from tokenspeed.runtime.model_loader.weight_utils import default_weight_loader
from tokenspeed.runtime.models.base import BaseCausalLM, BaseTransformerModel
from tokenspeed.runtime.models.muse_spark_audio import MuseSparkAudioTower
from tokenspeed.runtime.models.muse_spark_vision import MuseSparkVisionTower
from tokenspeed.runtime.moe.expert_location import ModelConfigForExpertLocation
from tokenspeed.runtime.multimodal.embedder import (
    EncoderSpec,
    MultimodalEmbedder,
    pad_input_tokens,
)
from tokenspeed.runtime.multimodal.inputs import (
    Modality,
    MultimodalDataItem,
    MultimodalForwardContext,
    MultimodalInputs,
)
from tokenspeed.runtime.utils import add_prefix
from tokenspeed.runtime.utils.common import make_layers
from tokenspeed.runtime.utils.cuda_stream import StreamFork
from tokenspeed.runtime.utils.env import global_server_args_dict

if TYPE_CHECKING:
    from tokenspeed_kernel.ops.communication._iris.muse_spark import MuseIrisState


_GATHER_WORKSPACE_ROWS = 64
_gather_workspaces: dict[tuple[torch.device, torch.dtype, int], torch.Tensor] = {}


def _acquire_all_reduce_output(
    shape: tuple[int, ...], like: torch.Tensor, group: Group
) -> torch.Tensor | None:
    shapes = (shape,)
    if not can_acquire_all_reduce_outputs(shapes, like, group):
        return None
    return acquire_all_reduce_outputs(shapes, like, group)[0]


def _muse_all_reduce(
    iris: MuseIrisState | None,
    tensors: tuple[torch.Tensor, ...],
    group: Group,
) -> tuple[torch.Tensor, ...]:
    """Use Muse's prepared payload policy, retaining ordinary backend fallback."""
    if iris is None:
        return all_reduce(tensors, group)
    if iris.all_reduce.owns_outputs(tensors):
        return iris.reduce_outputs(tensors)
    results = []
    for tensor in tensors:
        if (
            tensor.dtype == torch.bfloat16
            and tensor.device == iris.all_reduce.device
            and tensor.is_contiguous()
            and tensor.numel() > 0
            and tensor.numel() % (iris.world_size * 8) == 0
        ):
            results.append(iris.reduce(tensor))
        else:
            results.append(all_reduce(tensor, group))
    return tuple(results)


def _gemm_dispatch_required() -> bool:
    """Whether an explicit kernel or numerics policy owns dense projection."""
    return (
        global_server_args_dict["numerics"] in BITWISE_ENVELOPES
        or resolve_kernel_override("gemm", "mm", None) is not None
    )


class _MuseGatheredLinear(ColumnParallelLinear):
    """Muse pre-/post-expert projection with its checkpoint weight stored at .weight."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        prefix: str,
        tp_rank: int,
        tp_size: int,
        tp_group: Group,
        layer_id: int,
        layer_count: int,
        replicate_for_owned_rows: bool,
    ) -> None:
        # Replicate pre-/post-expert weights to avoid gathers and project owned rows.
        mapping = global_server_args_dict.get("mapping")
        replicated = (
            tp_size > 1
            and current_platform().is_cdna4
            and len(tp_group) in (4, 8)
            and torch.get_default_dtype() == torch.bfloat16
            and (input_size, output_size) in ((4096, 8192), (8192, 4096))
            and not global_server_args_dict.get("force_deterministic_rsag", False)
            and not global_server_args_dict.get("batch_invariant_collectives", False)
            and (
                mapping is None
                or not mapping.nprocs_per_node
                or len({rank // mapping.nprocs_per_node for rank in tp_group}) == 1
            )
        )
        super().__init__(
            input_size=input_size,
            output_size=output_size,
            bias=False,
            gather_output=False,
            skip_bias_add=False,
            params_dtype=None,
            quant_config=None,
            output_sizes=None,
            prefix=prefix,
            tp_rank=0 if replicated else tp_rank,
            tp_size=1 if replicated else tp_size,
            tp_group=None if replicated else tp_group,
            use_presharded_weights=False,
            override_kernel_name=None,
            interleave_linear_and_gate=False,
        )
        self.rank = tp_rank
        self.group = tp_group
        self.gather_column_view = replicated and replicate_for_owned_rows
        # Every rank holds the whole weight, so a rank can project any columns.
        self.replicated = replicated
        self.workspace = None
        self.multicast = None
        if tp_size > 1 and (not replicated or self.gather_column_view):
            key = (self.weight.device, self.weight.dtype, output_size)
            if key not in _gather_workspaces:
                _gather_workspaces[key] = self.weight.new_zeros(
                    _GATHER_WORKSPACE_ROWS, output_size
                )
            self.workspace = _gather_workspaces[key]
            # Mailboxes pool across layers, so their scope excludes the layer index.
            self.multicast = KimiK3LatentDownOp.initialize(
                group=process_group_manager.get_device_process_group(self.group),
                hidden_size=input_size,
                latent_size=output_size,
                device=self.weight.device,
                block_index=layer_id,
                layer_count=layer_count,
                model_scope=f"{prefix.split('.layers.')[0]}.{prefix.rsplit('.', 1)[-1]}",
                max_m=_GATHER_WORKSPACE_ROWS,
                # Takes this call's weight view; the mailbox publishes wider blocks.
                publish_gemm=partial(_block_gemm, self),
            )
        self.iris_state: MuseIrisState | None = None

    def _project(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        out: torch.Tensor | None,
    ) -> torch.Tensor:
        """Apply an unbiased projection, writing into ``out`` when supplied."""
        if (
            current_platform().is_cdna4
            and x.dtype == weight.dtype
            and not _gemm_dispatch_required()
            # Rows gemm.mm would route to gemm.mv go there directly, writing a
            # contiguous ``out`` in place.
            and (
                x.shape[0] == 1
                or ((out is None or out.is_contiguous()) and _use_mv(x, weight))
            )
        ) or (
            x.shape[0] == 1
            and (
                not current_platform().is_cdna4
                and resolve_kernel_override("gemm", "mm", None) is None
                and _use_mv(x, weight)
            )
        ):
            result = mv(x, weight, out=out)
        else:
            result = self.quant_method.apply(SimpleNamespace(weight=weight), x, None)
        if out is None:
            return result
        if result.data_ptr() != out.data_ptr():
            out.copy_(result)
        return out

    def _local_weight(self) -> torch.Tensor:
        if not self.gather_column_view:
            return self.weight
        width = self.output_size // len(self.group)
        return self.weight[self.rank * width : (self.rank + 1) * width]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the projection and return complete full-width rows."""
        weight = self._local_weight()
        if self.workspace is None:
            return (
                self._project(x, weight, None)
                if current_platform().is_cdna4 or self.gather_column_view
                else super().forward(x)[0]
            )
        tokens = x.shape[0]
        if self.multicast is not None and self.multicast.handles(tokens):
            return self.multicast(x, weight, output_scale=None)
        if tokens > self.workspace.shape[0]:
            return all_gather(
                (
                    self._project(x, weight, None)
                    if current_platform().is_cdna4
                    else super().forward(x)[0]
                ),
                self.group,
                dim=-1,
                backend=(
                    get_global_backend()
                    if current_platform().is_cdna4
                    else get_global_backend().nccl
                ),
            )
        local_width = weight.shape[0]
        probe = self.workspace.reshape(-1)[: tokens * local_width].view(
            tokens, local_width
        )
        iris = self.iris_state
        if iris is not None and iris.gather.can_run(probe):
            gathered = (
                _acquire_all_reduce_output(
                    (tokens, self.workspace.shape[1]),
                    x,
                    self.group,
                )
                if tokens == 1
                else None
            )
            local = (
                gathered[:, self.rank * local_width : (self.rank + 1) * local_width]
                if gathered is not None
                else x.new_empty((tokens, local_width))
            )
            self._project(x, weight, local)
            return iris.gather.run(local)

        rows = (
            _acquire_all_reduce_output((tokens, self.workspace.shape[1]), x, self.group)
            if current_platform().is_cdna4
            else None
        )
        direct = rows is not None
        if rows is None:
            rows = self.workspace[:tokens]
        else:
            rows.zero_()
        local = rows[:, self.rank * local_width : (self.rank + 1) * local_width]
        self._project(x, weight, local)
        if direct:
            return _muse_all_reduce(self.iris_state, (rows,), self.group)[0]

        # Allocate before the collective so a failure cannot strand peer blocks.
        owned = torch.empty_like(rows)
        gathered = (
            _muse_all_reduce(self.iris_state, (rows,), self.group)[0]
            if current_platform().is_cdna4
            else all_reduce(rows, self.group)
        )
        if gathered is rows:
            owned.copy_(rows)
            rows.zero_()
            return owned
        return gathered

    def forward_owned_rows(self, x: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        """Project complete owned rows through an already replicated weight."""
        assert self.weight.shape == (self.output_size, self.input_size)
        assert self.bias is None and isinstance(
            self.quant_method, UnquantizedLinearMethod
        )
        if _gemm_dispatch_required():
            return self._project(x, self.weight, out)
        return torch.mm(x, self.weight.T, out=out)

    def forward_with_mailbox(
        self,
        x: torch.Tensor,
        consumer_handles_mailbox: bool,
    ) -> tuple[torch.Tensor, int | None]:
        if (
            consumer_handles_mailbox
            and self.multicast is not None
            and self.multicast.handles(x.shape[0])
        ):
            return (
                self.multicast.publish(x, self._local_weight()),
                self.multicast.sentinel,
            )
        return self(x), None

    def forward_mxfp4(
        self,
        x: torch.Tensor,
        norm_weight: torch.Tensor,
        eps: float,
        weight_offset: float,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Project, gather, normalize, and dynamically quantize to MXFP4."""
        if self.workspace is not None and x.shape[0] <= self.workspace.shape[0]:
            tokens = x.shape[0]
            local_width = self._local_weight().shape[0]
            gathered = (
                _acquire_all_reduce_output(
                    (1, self.workspace.shape[1]),
                    x,
                    self.group,
                )
                if tokens == 1
                else None
            )
            local = (
                gathered[:, self.rank * local_width : (self.rank + 1) * local_width]
                if gathered is not None
                else x.new_empty((tokens, local_width))
            )
            iris = self.iris_state
            if iris is not None and iris.can_all_gather_rmsnorm_mxfp4_quant(
                local, norm_weight
            ):
                self._project(x, self._local_weight(), local)
                return self.workspace[:tokens], iris.all_gather_rmsnorm_mxfp4_quant(
                    local,
                    norm_weight,
                    eps,
                    weight_offset,
                )
        rows = self(x)
        return rows, rmsnorm_mxfp4_quant(rows, norm_weight, eps, weight_offset)


class MuseSparkRMSNorm(nn.Module):
    """RMSNorm weights with a checkpoint center; input-scale norms only lend them to fused kernels."""

    def __init__(
        self,
        hidden_size: int,
        eps: float,
        has_weight: bool,
        weight_offset: float,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.weight_offset = weight_offset
        if has_weight:
            self.weight = nn.Parameter(torch.empty(hidden_size))
        else:
            self.register_parameter("weight", None)

    def forward(self, x: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
        """Normalize ``x`` in FP32 and store the result in ``out_dtype``."""
        if self.weight is None:
            return weightless_rmsnorm(x, self.eps, out_dtype)
        return affine_rmsnorm(x, self.weight, self.eps, self.weight_offset, out_dtype)


class QKVOGateParallelLinear(ColumnParallelLinear):
    """Tensor-parallel Q/K/V projection with one extra query-sized output gate.

    Packing the gate into the QKV GEMM issues one wider projection per layer
    instead of two narrower ones. Q and the gate shard by rank; K and V shard
    by replicated KV head, as in ``QKVParallelLinear``.
    """

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int,
        quant_config: QuantizationConfig | None,
        prefix: str,
        tp_rank: int,
        tp_size: int,
        tp_group: tuple[int, ...] | None,
        replicate_for_owned_rows: bool,
    ) -> None:
        if total_num_heads % tp_size:
            raise ValueError("num_attention_heads must be divisible by attention TP")
        self.num_heads = total_num_heads // tp_size
        if tp_size >= total_num_kv_heads:
            if tp_size % total_num_kv_heads:
                raise ValueError("attention TP must divide replicated KV heads")
            self.num_kv_heads = 1
            self.num_kv_head_replicas = tp_size // total_num_kv_heads
        else:
            if total_num_kv_heads % tp_size:
                raise ValueError(
                    "num_key_value_heads must be divisible by attention TP"
                )
            self.num_kv_heads = total_num_kv_heads // tp_size
            self.num_kv_head_replicas = 1
        self.q_shard = self.num_heads * head_size
        self.kv_shard = self.num_kv_heads * head_size
        local = 2 * self.q_shard + 2 * self.kv_shard
        super().__init__(
            input_size=hidden_size,
            output_size=local * tp_size,
            bias=False,
            gather_output=False,
            skip_bias_add=False,
            params_dtype=None,
            quant_config=quant_config,
            output_sizes=None,
            prefix=prefix,
            tp_rank=tp_rank,
            tp_size=tp_size,
            tp_group=tp_group,
            use_presharded_weights=False,
            override_kernel_name=None,
            interleave_linear_and_gate=False,
        )
        self.replicate_for_owned_rows = (
            replicate_for_owned_rows
            and isinstance(self.quant_method, UnquantizedLinearMethod)
            and self.weight.dtype == torch.bfloat16
        )
        if self.replicate_for_owned_rows:
            # One Parameter owns the rank-major matrix; small batches use its
            # contiguous original TP view, including after an in-place reload.
            self.weight.data = torch.empty(
                (local * tp_size, hidden_size),
                dtype=self.weight.dtype,
                device=self.weight.device,
            )
        self.iris_state: MuseIrisState | None = None

    def _local_weight(self) -> torch.Tensor:
        if not self.replicate_for_owned_rows:
            return self.weight
        width = 2 * self.q_shard + 2 * self.kv_shard
        return self.weight[self.tp_rank * width : (self.tp_rank + 1) * width]

    def forward(self, input_, block_scale=None, output_dtype=None):
        if not self.replicate_for_owned_rows:
            return super().forward(input_, block_scale, output_dtype)
        assert block_scale is None and output_dtype is None
        return (
            self.quant_method.apply(
                SimpleNamespace(weight=self._local_weight()), input_, None
            ),
            None,
        )

    def project_owned_rows(
        self, inputs: torch.Tensor, global_rows: int
    ) -> torch.Tensor:
        """Borrow symmetric Q/K/V/gate output until all peer reads finish."""
        assert self.replicate_for_owned_rows
        assert inputs.dtype == self.weight.dtype == torch.bfloat16
        assert inputs.is_contiguous()
        local_rows = (global_rows + self.tp_size - 1) // self.tp_size
        assert inputs.shape == (local_rows, self.input_size)
        iris = self.iris_state
        assert iris is not None
        output = iris.acquire_owned_qkvg_output(global_rows)
        source_start = inputs.data_ptr()
        source_end = source_start + inputs.numel() * inputs.element_size()
        target_start = output.data_ptr()
        target_end = target_start + output.numel() * output.element_size()
        assert source_end <= target_start or target_end <= source_start
        if _gemm_dispatch_required():
            output.copy_(self.quant_method.apply(self, inputs, None))
        else:
            torch.mm(inputs, self.weight.T, out=output)
        return output

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: str,
    ) -> None:
        sizes = {
            "q": self.q_shard,
            "k": self.kv_shard,
            "v": self.kv_shard,
            "g": self.q_shard,
        }
        offsets = {
            "q": 0,
            "k": self.q_shard,
            "v": self.q_shard + self.kv_shard,
            "g": self.q_shard + 2 * self.kv_shard,
        }
        if loaded_shard_id not in sizes:
            raise ValueError(f"unsupported QKV gate shard {loaded_shard_id!r}")
        output_dim = param.output_dim
        shard_size = sizes[loaded_shard_id]
        ranks = (
            range(self.tp_size) if self.replicate_for_owned_rows else (self.tp_rank,)
        )
        local_width = 2 * self.q_shard + 2 * self.kv_shard
        for rank in ranks:
            if loaded_shard_id in ("q", "g"):
                start = rank * shard_size
            else:
                start = (rank // self.num_kv_head_replicas) * shard_size
            destination = offsets[loaded_shard_id]
            if self.replicate_for_owned_rows:
                destination += rank * local_width
            param.data.narrow(output_dim, destination, shard_size).copy_(
                loaded_weight.narrow(output_dim, start, shard_size)
            )


def _block_gemm(
    linear: LinearBase,
    x: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    """``x`` through this rank's ``weight`` block; one row is written into ``out``.

    The kernels are the ones the projection's own forward would pick, so the
    published rows carry the bits an unpublished projection would have.
    """
    if (
        x.shape[0] == 1
        and resolve_kernel_override("gemm", "mm", None) is None
        and _use_mv(x, weight)
    ):
        return mv(x, weight, out=out)
    return linear.quant_method.apply(SimpleNamespace(weight=weight), x, None)


def _bind_mailbox(
    linear: LinearBase,
    shard_group: dist.ProcessGroup,
    layer_id: int,
    config: MuseSparkConfig,
    prefix: str,
    ranks: int,
) -> KimiK3LatentDownOp | None:
    """Bind a pooled NVLS mailbox for one BF16 sharded projection, or None.

    A mailbox row is ``ranks`` blocks of ``linear.output_size`` columns: one
    block, and each rank publishes its column slice of a gathered projection;
    one block per rank, and each publishes its full partial for a consumer
    that reduces them. The publish runs the projection's own BF16 kernels, so
    a quantized projection is left to its forward.
    """
    from tokenspeed.runtime.distributed.process_group_manager import (
        process_group_manager,
    )

    if not isinstance(linear.quant_method, UnquantizedLinearMethod):
        return None
    # Mailboxes pool across layers, so the scope must exclude the layer index.
    model_scope = f"{prefix.split('.layers.')[0]}.{linear.prefix.rsplit('.', 1)[-1]}"
    return KimiK3LatentDownOp.initialize(
        group=process_group_manager.get_device_process_group(shard_group),
        hidden_size=linear.weight.shape[1],
        latent_size=linear.output_size * ranks,
        device=torch.device("cuda", torch.cuda.current_device()),
        block_index=layer_id,
        layer_count=config.num_hidden_layers,
        model_scope=model_scope,
        max_m=_GATHER_WORKSPACE_ROWS,
        # Pooled across layers: the weight must come from the call, never the closure.
        publish_gemm=lambda x, weight, out: _block_gemm(linear, x, weight, out),
    )


class _ScaledHeadNorm(nn.Module):
    """A weightless per-head RMSNorm whose heads scale by a constant, for the attention prologue.

    The fp32 weight carries the constant exactly, so ``x * rsqrt(mean(x^2) + eps) * scale``
    keeps the checkpoint's query scale unrounded.
    """

    def __init__(self, head_dim: int, scale: float, eps: float) -> None:
        super().__init__()
        self.register_buffer(
            "weight",
            torch.full((head_dim,), scale, dtype=torch.float32),
            persistent=False,
        )
        self.weight_offset = 0.0
        self.variance_epsilon = eps


class MuseSparkAttention(nn.Module):
    """GQA with weightless QK norm and per-head normalized output gates."""

    def __init__(
        self,
        config: MuseSparkConfig,
        mapping: Mapping,
        layer_id: int,
        quant_config: QuantizationConfig | None,
        prefix: str,
        reduce_in_boundary: bool,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.tp_size = mapping.attn.tp_size
        self.head_dim = config.head_dim
        self.q_scale = config.qk_scale_factor
        self.rms_norm_eps = config.rms_norm_eps
        self.qkv_proj = QKVOGateParallelLinear(
            hidden_size=config.hidden_size,
            head_size=config.head_dim,
            total_num_heads=config.num_attention_heads,
            total_num_kv_heads=config.num_key_value_heads,
            quant_config=quant_config,
            prefix=add_prefix("qkv_proj", prefix),
            tp_rank=mapping.attn.tp_rank,
            tp_size=mapping.attn.tp_size,
            tp_group=mapping.attn.tp_group,
            replicate_for_owned_rows=(
                layer_id > 0
                and current_platform().is_cdna4
                and mapping.pp_size == 1
                and mapping.attn.tp_size == mapping.moe.tp_ep_size
                and mapping.attn.tp_size in (4, 8)
                and (mapping.moe.tp_size == 1 or mapping.moe.ep_size == 1)
                and mapping.attn.tp_group == mapping.moe.tp_ep_group
                and config.hidden_size == 8192
                and config.num_attention_heads == 128
                and config.num_key_value_heads == 16
                and config.head_dim == 64
            ),
        )
        self.num_heads = self.qkv_proj.num_heads
        self.num_kv_heads = self.qkv_proj.num_kv_heads
        self.q_size = self.qkv_proj.q_shard
        self.kv_size = self.qkv_proj.kv_shard
        self.o_proj = RowParallelLinear(
            input_size=config.num_attention_heads * config.head_dim,
            output_size=config.hidden_size,
            bias=False,
            input_is_parallel=True,
            skip_bias_add=False,
            params_dtype=None,
            reduce_results=False,
            quant_config=quant_config,
            prefix=add_prefix("o_proj", prefix),
            tp_rank=mapping.attn.tp_rank,
            tp_size=mapping.attn.tp_size,
            tp_group=mapping.attn.tp_group,
            use_presharded_weights=False,
            override_kernel_name=None,
            interleave_linear_and_gate=False,
        )
        self.rotary_emb = (
            get_rope(
                config.head_dim,
                rotary_dim=config.head_dim,
                max_position=config.max_position_embeddings,
                base=config.layer_rope_theta[layer_id],
                is_neox_style=False,
                rope_scaling=None,
                dtype=torch.float32,
                partial_rotary_factor=1.0,
            )
            if config.uses_rope(layer_id)
            else None
        )
        # The fused owned-row receiver uses an empty table for NoPE layers.
        self.register_buffer(
            "nope_cos_sin_cache",
            (
                torch.empty(1, 0, dtype=torch.float32)
                if current_platform().is_cdna4
                else None
            ),
            persistent=False,
        )
        self.q_norm = _ScaledHeadNorm(config.head_dim, self.q_scale, self.rms_norm_eps)
        self.k_norm = _ScaledHeadNorm(config.head_dim, 1.0, self.rms_norm_eps)
        self.o_proj_multicast = (
            _bind_mailbox(
                self.o_proj,
                mapping.attn.tp_group,
                layer_id,
                config,
                prefix,
                self.tp_size,
            )
            if not current_platform().is_cdna4
            and reduce_in_boundary
            and self.tp_size > 1
            else None
        )
        window = config.sliding_window[layer_id]
        self.attn = PagedAttention(
            self.num_heads,
            self.head_dim,
            self.head_dim**-0.5,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            logit_cap=0.0,
            v_head_dim=-1,
            sliding_window_size=(
                hf_sliding_window_to_window_left(window) if window is not None else -1
            ),
            rotary_emb=self.rotary_emb,
            qk_norm=(self.q_norm, self.k_norm),
        )
        self.iris_state: MuseIrisState | None = None

    def _acquire_projection_output(self, x: torch.Tensor) -> torch.Tensor | None:
        """Borrow the all-reduce producer buffer for the projection, if compatible."""
        linear = self.o_proj
        return (
            _acquire_all_reduce_output(
                (x.shape[0], linear.output_size),
                x,
                linear.tp_group,
            )
            if linear.tp_size > 1
            and type(linear.quant_method) is UnquantizedLinearMethod
            and linear.bias is None
            and linear.input_is_parallel
            and not linear.reduce_results
            and x.dtype == linear.weight.dtype
            and not _gemm_dispatch_required()
            else None
        )

    def _project_output(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Project ``x`` and defer the TP reduction to the consuming operation."""
        linear = self.o_proj
        output = self._acquire_projection_output(x)
        return (
            RowParallelLinear.forward(linear, x)[0]
            if output is None
            else mv(x, linear.weight, out=output)
        )

    def forward(
        self,
        positions: torch.Tensor,
        normalized: torch.Tensor,
        ctx: ForwardContext,
        owned_global_rows: int | None,
    ) -> tuple[torch.Tensor, int | None, int]:
        """Attend over rows already passed through ``normalize``."""
        if owned_global_rows is not None:
            projected = self.qkv_proj.project_owned_rows(normalized, owned_global_rows)
            iris = self.iris_state
            assert iris is not None
        if (
            owned_global_rows is not None
            and owned_global_rows >= MUSE_OWNED_QKVG_MIN_TOKENS
            and ctx.forward_mode.is_extend()
            and ctx.token_to_kv_pool.dtype == torch.bfloat16
            and normalized.dtype == torch.bfloat16
            and (self.head_dim, self.q_size, self.kv_size)
            == (64, 8192 // self.tp_size, 1024 // self.tp_size)
        ):
            cos_sin_cache = (
                self.rotary_emb.cos_sin_cache
                if self.rotary_emb is not None
                else self.nope_cos_sin_cache
            )
            q, k, v, output_gate = iris.all_to_all_transpose_qkvg_norm_rope(
                projected,
                owned_global_rows,
                positions,
                cos_sin_cache,
                self.rms_norm_eps,
                self.q_scale,
            )
            prepared = self.attn.store_prepared(q, k, v, ctx)
            output = self.attn.attend(prepared.q, prepared.k, prepared.v, ctx)
        else:
            if owned_global_rows is not None:
                qkv_gate = iris.all_to_all_transpose_qkvg(projected, owned_global_rows)
            elif normalized.shape[0] == 0:
                return normalized.new_empty((0, self.hidden_size)), None, 1
            else:
                qkv_gate, _ = self.qkv_proj(normalized)
            q, k, v, output_gate = qkv_gate.split(
                (self.q_size, self.kv_size, self.kv_size, self.q_size), dim=-1
            )
            output = self.attn(q, k, v, positions, ctx)
        output = output.contiguous()
        # One row normalizes and gates inside the output projection's GEMV.
        projection_out = (
            self._acquire_projection_output(output)
            if output.shape[0] == 1
            and mv_weightless_rmsnorm_gated_sigmoid_supported(
                output, output_gate, self.o_proj.weight, self.num_heads, self.head_dim
            )
            else None
        )
        if projection_out is not None:
            projected = mv_weightless_rmsnorm_gated_sigmoid(
                output,
                output_gate,
                self.o_proj.weight,
                self.num_heads,
                self.rms_norm_eps,
                projection_out,
            )
            return projected, None, 1
        output = weightless_rmsnorm_gated_sigmoid(
            output,
            output_gate,
            self.num_heads,
            self.head_dim,
            self.rms_norm_eps,
            pdl_enabled(),
        )
        return self._project(output)

    def _project(self, output: torch.Tensor) -> tuple[torch.Tensor, int | None, int]:
        """Project attended rows; a batch the mailbox serves publishes its partial for the boundary to reduce."""
        if current_platform().is_cdna4:
            return self._project_output(output), None, 1
        multicast = self.o_proj_multicast
        tokens = output.shape[0]
        if (
            multicast is not None
            and multicast.handles(tokens)
            and boundary_gathers_mailbox(tokens, self.hidden_size, self.tp_size)
        ):
            rows = multicast.publish(output, self.o_proj.weight)
            return rows, multicast.sentinel, self.tp_size
        output, _ = self.o_proj(output)
        return output, None, 1


# Every decoder gain is stored as its deviation from unity; the output gain is not.
ZERO_CENTERED_GAIN_OFFSET = 1.0
OUTPUT_GAIN_OFFSET = 0.0
# Norm-preserving residual gate constants, fixed by the architecture.
RESIDUAL_GATE_TEMPERATURE = 0.3
RESIDUAL_GATE_EPS = 1e-3
MUSE_OWNED_ROWS_MIN_TOKENS = 192
MUSE_OWNED_QKVG_MIN_TOKENS = 2048


class PreparedFP32RouterGEMV(nn.Module):
    """Own the Muse router weight and its selected GEMV's persistent layout."""

    def __init__(
        self,
        hidden_size: int,
        output_size: int,
        multiplier: float,
        solution: str | None,
    ) -> None:
        super().__init__()
        self.multiplier = multiplier
        self.weight = nn.Parameter(
            torch.empty(hidden_size, output_size, dtype=torch.float32),
            requires_grad=False,
        )
        self.register_buffer(
            "counters",
            fp32_router_gemv_counters(output_size, self.weight.device),
            persistent=False,
        )
        self._plan = fp32_router_gemv_plan(
            hidden_size=hidden_size,
            output_size=output_size,
            dtype=torch.float32,
            solution=solution,
            batch_invariant=global_server_args_dict["numerics"] in BITWISE_ENVELOPES,
        )
        self.register_buffer("prepared_weight", None, persistent=False)

    def process_weights_after_loading(self, _module: nn.Module) -> None:
        """Prepare the loaded canonical weight for the selected backend."""
        self.prepared_weight = fp32_router_gemv_process_weight(
            self._plan,
            self.weight,
        )

    def forward(self, x: torch.Tensor, enable_pdl: bool) -> torch.Tensor:
        """Project contiguous FP32 rows to scaled expert logits."""
        if self.prepared_weight is None:
            raise RuntimeError("router weight was not prepared after loading")
        return fp32_router_gemv_execute(
            self._plan,
            x,
            self.prepared_weight,
            self.multiplier,
            self.counters,
            enable_pdl,
        )


class MuseSparkRouter(PreparedFP32RouterGEMV):
    """FP32 routing scores plus the bias that only steers the top-k selection."""

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        multiplier: float,
        solution: str | None,
    ) -> None:
        super().__init__(
            hidden_size=hidden_size,
            output_size=num_experts,
            multiplier=multiplier,
            solution=solution,
        )
        self.e_score_correction_bias = nn.Parameter(
            torch.empty(num_experts, dtype=torch.float32), requires_grad=False
        )


class MuseSparkMoE(nn.Module):
    """Sharded SwiGLU experts with per-route post normalization.

    The experts run as raw BF16 weights through the portable Triton route
    kernel, or as FP4 weights through ``MoELayer``'s planned kernel when the
    checkpoint quantizes this layer. Both arms finish with the same post-expert
    norm and weighted reduction: the BF16 arm over a materialized per-route
    tensor, the quantized one straight through the planned kernel's permute map.
    """

    def __init__(
        self,
        config: MuseSparkConfig,
        mapping: Mapping,
        quant_config: QuantizationConfig | None,
        layer_id: int,
        prefix: str,
        alt_stream: torch.cuda.Stream | None,
    ) -> None:
        super().__init__()
        self.stream_fork = StreamFork(alt_stream)
        if torch.get_default_dtype() != torch.bfloat16:
            raise ValueError("Muse Spark requires --dtype bfloat16")
        if mapping.moe.tp_size > 1 and mapping.moe.ep_size > 1:
            raise ValueError(
                "Muse Spark MoE requires pure tensor or expert parallelism"
            )
        if mapping.attn.tp_group != mapping.moe.tp_ep_group:
            raise ValueError(
                "Muse Spark requires the attention TP and MoE groups to match"
            )
        if config.num_local_experts % mapping.moe.ep_size:
            raise ValueError("the expert count must divide evenly across EP ranks")
        if config.moe_intermediate_size % mapping.moe.tp_size:
            raise ValueError("the expert intermediate size must divide across TP ranks")
        self.num_experts = config.num_local_experts
        self.topk = config.num_experts_per_tok
        self.route_scale = config.routed_scaling_factor
        self.ep_rank = mapping.moe.ep_rank
        self.ep_size = mapping.moe.ep_size
        self.experts_per_rank = config.num_local_experts // self.ep_size
        self.expert_start = self.ep_rank * self.experts_per_rank
        self.ep_group = mapping.moe.ep_group
        self.tp_rank = mapping.moe.tp_rank
        self.tp_size = mapping.moe.tp_size
        self.tp_group = mapping.moe.tp_group
        compressed_size = config.moe_hidden_size
        intermediate_size = config.moe_intermediate_size // self.tp_size

        self.gate = MuseSparkRouter(
            hidden_size=config.hidden_size,
            num_experts=config.num_local_experts,
            multiplier=config.output_multiplier,
            solution=None,
        )
        self.pre_expert_proj = _MuseGatheredLinear(
            input_size=config.hidden_size,
            output_size=compressed_size,
            prefix=add_prefix("pre_expert_proj", prefix),
            tp_rank=mapping.attn.tp_rank,
            tp_size=mapping.attn.tp_size,
            tp_group=mapping.attn.tp_group,
            layer_id=layer_id,
            layer_count=config.num_hidden_layers,
            replicate_for_owned_rows=True,
        )
        self.pre_expert_norm = MuseSparkRMSNorm(
            hidden_size=compressed_size,
            eps=config.rms_norm_eps,
            has_weight=True,
            weight_offset=ZERO_CENTERED_GAIN_OFFSET,
        )
        experts_prefix = add_prefix("experts", prefix)
        self.quantized_experts = quant_config is not None and not (
            should_ignore_quant_layer(experts_prefix, quant_config.ignored_layers)
            or should_exclude_quant_module(
                experts_prefix,
                quant_config.exclude_modules,
            )
        )
        if self.quantized_experts:
            self.experts = MoELayer(
                top_k=self.topk,
                num_experts=self.num_experts,
                hidden_size=compressed_size,
                intermediate_size=config.moe_intermediate_size,
                quant_config=quant_config,
                layer_index=layer_id,
                prefix=experts_prefix,
                tp_rank=self.tp_rank,
                tp_size=self.tp_size,
                ep_rank=self.ep_rank,
                ep_size=self.ep_size,
                # "swiglu" would force one shared gate/up scale; this model is plain silu(gate) * up.
                activation="silu",
                routing_mode="precomputed_topk",
            )
            self.expert_quant_kind = self.experts._quant_kind
            if self.expert_quant_kind not in {"mxfp4", "nvfp4"}:
                raise ValueError(
                    "Muse Spark quantized experts require NVFP4 or MXFP4 weights"
                )
            if not self.experts.supports_deferred_finalize:
                raise ValueError(
                    "Muse Spark normalizes each expert route before the "
                    "top-k reduction, so its MoE kernel must defer the finalize"
                )
        else:
            self.expert_quant_kind = "unquant"
            self.experts = nn.Module()
            self.experts.w13_weight = nn.Parameter(
                torch.empty(
                    self.experts_per_rank,
                    2 * intermediate_size,
                    compressed_size,
                )
            )
            self.experts.w2_weight = nn.Parameter(
                torch.empty(
                    self.experts_per_rank,
                    compressed_size,
                    intermediate_size,
                )
            )
        self.post_expert_norm = MuseSparkRMSNorm(
            hidden_size=compressed_size,
            eps=config.post_norm_eps,
            has_weight=True,
            weight_offset=ZERO_CENTERED_GAIN_OFFSET,
        )

        self.post_expert_proj = _MuseGatheredLinear(
            input_size=compressed_size,
            output_size=config.hidden_size,
            prefix=add_prefix("post_expert_proj", prefix),
            tp_rank=mapping.attn.tp_rank,
            tp_size=mapping.attn.tp_size,
            tp_group=mapping.attn.tp_group,
            layer_id=layer_id,
            layer_count=config.num_hidden_layers,
            replicate_for_owned_rows=False,
        )
        # Decode-sized batches route in the top-k kernel; each token count keeps its own GEMM workspace.
        self.prepared_routing: dict[int, PreparedNvfp4Routing] = {}
        self.iris_state: MuseIrisState | None = None

    def _reduce_expert_routes(
        self,
        routes: torch.Tensor,
        route_map: torch.Tensor | None,
        num_tokens: int,
        route_weights: torch.Tensor,
        *,
        owned_rows: bool,
    ) -> torch.Tensor:
        """Complete TP routes before normalization, or sum normalized EP routes."""
        norm = self.post_expert_norm
        iris = self.iris_state
        if self.tp_size > 1:
            assert norm.weight is not None
            width = norm.weight.shape[0]
            if owned_rows:
                assert iris is not None
                routes = iris.reduce_scatter_owned_routes(routes, route_map, num_tokens)
                num_tokens = routes.shape[0]
                first = iris.all_reduce.rank_in_group * num_tokens
                route_weights = route_weights[first : first + num_tokens]
                if route_weights.shape[0] < num_tokens:
                    route_weights = torch.nn.functional.pad(
                        route_weights, (0, 0, 0, num_tokens - route_weights.shape[0])
                    )
            elif route_map is not None:
                output = _acquire_all_reduce_output(
                    (num_tokens * self.topk, width), routes, self.tp_group
                )
                routes = gather_moe_routes(
                    routes,
                    route_map,
                    num_tokens,
                    self.topk,
                    width,
                    output=(
                        output.reshape(num_tokens, self.topk, width)
                        if output is not None
                        else None
                    ),
                )
            # Each TP rank contributes only its intermediate-column slice.
            # RMSNorm must see the complete route before top-k weighting.
            if not owned_rows:
                # Routes already in the acquired producer views reduce without staging.
                (routes,) = _muse_all_reduce(
                    iris,
                    (routes.reshape(num_tokens * self.topk, width),),
                    self.tp_group,
                )
            # Consume borrowed collective storage before the next projection.
            return routed_input_scale_rmsnorm_sum(
                routes.reshape(num_tokens, self.topk, width),
                route_weights,
                norm.weight,
                norm.eps,
                norm.weight_offset,
                pdl_enabled(),
                None,
            )
        if owned_rows:
            assert iris is not None and iris.can_use_owned_rows(num_tokens)
            if route_map is not None and norm.weight is not None:
                reduced = iris.reduce_scatter_owned_routed_rmsnorm_sum(
                    routes,
                    route_map,
                    num_tokens,
                    self.topk,
                    route_weights,
                    norm.weight,
                    norm.eps,
                    norm.weight_offset,
                )
                if reduced is not None:
                    return reduced
        if route_map is not None and norm.weight is None:
            raise ValueError("routed gathered reduction requires an affine norm")
        # Write the sum straight into the prepared all-reduce input that the
        # collective below acquires again, instead of copying it there.
        output = (
            _acquire_all_reduce_output(
                (
                    num_tokens,
                    routes.shape[-1] if route_map is None else norm.weight.shape[0],
                ),
                routes,
                self.ep_group,
            )
            if current_platform().is_cdna4 and (owned_rows or self.ep_size > 1)
            else None
        )
        if route_map is None:
            combined = routed_input_scale_rmsnorm_sum(
                routes,
                route_weights,
                norm.weight,
                norm.eps,
                norm.weight_offset,
                pdl_enabled(),
                output,
            )
        else:
            if (
                not owned_rows
                and iris is not None
                and iris.can_all_reduce_routed_rmsnorm_sum(
                    routes,
                    route_map,
                    num_tokens,
                    self.topk,
                    route_weights,
                    norm.weight,
                    norm.weight_offset,
                )
            ):
                return iris.all_reduce_routed_rmsnorm_sum(
                    routes,
                    route_map,
                    num_tokens,
                    self.topk,
                    route_weights,
                    norm.weight,
                    norm.eps,
                    norm.weight_offset,
                )
            combined = routed_input_scale_rmsnorm_sum_gathered(
                routes,
                route_map,
                num_tokens,
                self.topk,
                route_weights,
                norm.weight,
                norm.eps,
                norm.weight_offset,
                pdl_enabled(),
                output,
            )
            if owned_rows:
                return iris.reduce_scatter_owned_rows(combined)
        if not owned_rows and self.ep_size <= 1:
            return combined
        if not current_platform().is_cdna4:
            return all_reduce(combined, self.ep_group)
        output = _acquire_all_reduce_output(
            tuple(combined.shape), combined, self.ep_group
        )
        if output is not None:
            if output.data_ptr() != combined.data_ptr():
                output.copy_(combined)
            combined = output
        if owned_rows:
            return iris.reduce_scatter_owned_rows(combined)
        return _muse_all_reduce(self.iris_state, (combined,), self.ep_group)[0]

    def _quantized_combined(
        self,
        compressed: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        compressed_quantized: tuple[torch.Tensor, torch.Tensor] | None,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        router_logits: torch.Tensor | None,
        num_global_tokens: int,
        max_num_tokens_per_gpu: int,
        prepared: PreparedNvfp4Routing | None,
        owned_rows: bool,
    ) -> torch.Tensor:
        """Run FP4 experts and complete sharded routes before normalization."""
        norm_weight = self.post_expert_norm.weight
        iris = self.iris_state
        deferred_out = None
        if (
            self.tp_size > 1
            and self.ep_size == 1
            and iris is not None
            and norm_weight is not None
            and norm_weight.shape == (4096,)
            and self.experts.deferred_output_is_token_order
            and self.experts.supports_deferred_out
        ):
            deferred_out = self._acquire_route_destination(
                topk_ids.shape[0], owned_rows
            )
        gemm2_out, _, expanded_idx = self.experts(
            hidden_states=compressed,
            topk_output=StandardTopKOutput(topk_weights, topk_ids, router_logits),
            num_global_tokens=num_global_tokens,
            max_num_tokens_per_gpu=max_num_tokens_per_gpu,
            do_finalize=False,
            x_quantized=compressed_quantized,
            routing_prepared=prepared,
            deferred_out=deferred_out,
        )
        if deferred_out is not None:
            assert gemm2_out.data_ptr() == deferred_out.data_ptr()
        if prepared is None:
            self._prepare_routing(compressed_quantized, topk_ids, topk_weights)
        if (
            self.tp_size > 1
            and self.ep_size == 1
            and self.experts.deferred_output_is_token_order
            and norm_weight is not None
            and gemm2_out.shape == (topk_ids.numel(), norm_weight.shape[0])
            and gemm2_out.is_contiguous()
        ):
            # EP1 owns every valid Muse route. Canonical, unpadded rows need
            # no map gather before the TP reduction.
            gemm2_out = gemm2_out.view(*topk_ids.shape, norm_weight.shape[0])
            expanded_idx = None
        # A supplied destination is canonical; gathering it would read itself.
        assert deferred_out is None or expanded_idx is None
        # Other deferred outputs retain their map and any hidden-width padding.
        return self._reduce_expert_routes(
            routes=gemm2_out,
            route_map=expanded_idx,
            num_tokens=topk_ids.shape[0],
            route_weights=topk_weights,
            owned_rows=owned_rows,
        )

    def _acquire_route_destination(
        self, num_tokens: int, owned_rows: bool
    ) -> torch.Tensor | None:
        """Borrow the TP route reduction's input as ``[tokens * topk, width]``."""
        iris = self.iris_state
        norm_weight = self.post_expert_norm.weight
        assert iris is not None and norm_weight is not None
        if owned_rows:
            return (
                iris.acquire_owned_route_input(num_tokens)
                if iris.owned_route_max_rows
                else None
            )
        return _acquire_all_reduce_output(
            (num_tokens * self.topk, norm_weight.shape[0]), norm_weight, self.tp_group
        )

    def _prepare_routing(
        self,
        compressed_quantized: tuple[torch.Tensor, torch.Tensor],
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> None:
        """Prepare the routed GEMMs for this token count once, outside graph capture."""
        tokens = topk_ids.shape[0]
        if (
            self.expert_quant_kind != "nvfp4"
            or tokens > FUSED_ROUTING_MAX_TOKENS
            or tokens in self.prepared_routing
            or torch.cuda.is_current_stream_capturing()
        ):
            return
        self.prepared_routing[tokens] = prepare_nvfp4_routing(
            self.experts,
            compressed_quantized,
            topk_ids,
            topk_weights,
            None,
            pdl_enabled(),
        )

    def _select_experts(
        self, router_logits: torch.Tensor, prepared: PreparedNvfp4Routing | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Top-k route weights and ids; with prepared GEMMs the same launch fills their routing."""
        if prepared is not None:
            return moe_sigmoid_bias_topk_routing(
                router_logits,
                self.gate.e_score_correction_bias,
                self.topk,
                self.route_scale,
                True,
                self.expert_start,
                self.experts_per_rank,
                prepared.routing,
                pdl_enabled(),
            )
        return moe_topk(
            router_logits=router_logits,
            top_k=self.topk,
            score_function="sigmoid",
            selection_method="topk",
            renormalize=True,
            routed_scaling_factor=self.route_scale,
            correction_bias=self.gate.e_score_correction_bias,
            hash_indices_table=None,
            input_ids=None,
            logical_to_physical_map=None,
            topk_indices_dtype=torch.int32,
            topk_weights_dtype=torch.float32,
            override=None,
            solution=None,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_input: torch.Tensor,
        num_global_tokens: int,
        max_num_tokens_per_gpu: int,
        owned_global_rows: int | None,
        project: bool,
    ) -> tuple[torch.Tensor, int | None]:
        """Route BF16 ``hidden_states`` (router reads the FP32 ``router_input``) through the experts.

        The returned projection may defer its gather to the residual boundary.
        Without ``project`` the combined expert rows are returned unprojected
        for a boundary that applies ``post_expert_proj`` itself.
        """
        # The graph phase warms the fork serially, so capture never first-touches it.
        with self.stream_fork.scope(
            enable=get_is_cuda_graph_phase(), overlap=get_is_capture_mode()
        ) as fork:
            norm_weight = self.pre_expert_norm.weight
            if owned_global_rows is not None:
                local_rows = hidden_states.shape[0]
                output = hidden_states.new_empty(
                    (local_rows, self.pre_expert_proj.output_size)
                )
                compressed = self.pre_expert_proj.forward_owned_rows(
                    hidden_states, output
                )
                if (
                    self.expert_quant_kind == "mxfp4"
                    and self.experts.supports_prequantized_input
                ):
                    compressed_quantized = rmsnorm_mxfp4_quant(
                        compressed,
                        norm_weight,
                        self.pre_expert_norm.eps,
                        self.pre_expert_norm.weight_offset,
                    )
                else:
                    compressed = self.pre_expert_norm(compressed, compressed.dtype)
                    compressed_quantized = None
            elif (
                self.expert_quant_kind == "mxfp4"
                and self.experts.supports_prequantized_input
                and norm_weight is not None
            ):
                compressed, compressed_quantized = self.pre_expert_proj.forward_mxfp4(
                    hidden_states,
                    norm_weight,
                    self.pre_expert_norm.eps,
                    self.pre_expert_norm.weight_offset,
                )
            elif self.expert_quant_kind == "nvfp4":
                width = self.pre_expert_proj.output_size
                compressed, sentinel = self.pre_expert_proj.forward_with_mailbox(
                    hidden_states, width & (width - 1) == 0
                )
                compressed_quantized = rmsnorm_nvfp4_quant(
                    compressed,
                    norm_weight,
                    self.pre_expert_norm.eps,
                    self.pre_expert_norm.weight_offset,
                    self.experts.w13_input_scale_quant,
                    sentinel,
                    pdl_enabled(),
                )
            else:
                compressed = self.pre_expert_proj(hidden_states)
                compressed_quantized = None
                # MXFP4 kernels dynamically quantize this normalized BF16 row
                # into OCP E2M1 values with group-32 E8M0 scales.
                compressed = self.pre_expert_norm(compressed, compressed.dtype)
            with fork.branch():
                router_logits = self.gate(router_input, pdl_enabled())
                prepared = (
                    self.prepared_routing.get(hidden_states.shape[0])
                    if owned_global_rows is None
                    else None
                )
                topk_weights, topk_ids = self._select_experts(router_logits, prepared)
        if owned_global_rows is not None:
            iris = self.iris_state
            assert iris is not None
            expert_input = (
                compressed_quantized if compressed_quantized is not None else compressed
            )
            compressed, topk_ids, topk_weights = iris.gather_owned_moe_inputs(
                expert_input,
                topk_ids,
                topk_weights,
                owned_global_rows,
                min(
                    local_rows,
                    max(
                        0,
                        owned_global_rows - iris.all_reduce.rank_in_group * local_rows,
                    ),
                ),
            )
            compressed_quantized = None
            router_logits = None
        if self.quantized_experts:
            combined = self._quantized_combined(
                (
                    compressed.contiguous()
                    if isinstance(compressed, torch.Tensor)
                    else compressed
                ),
                compressed_quantized,
                topk_weights,
                topk_ids,
                router_logits,
                num_global_tokens,
                max_num_tokens_per_gpu,
                prepared,
                owned_global_rows is not None,
            )
        else:
            routes = bf16_moe_routes(
                compressed.contiguous(),
                self.experts.w13_weight,
                self.experts.w2_weight,
                topk_ids.contiguous(),
                self.expert_start,
                output=(
                    self._acquire_route_destination(
                        compressed.shape[0], owned_global_rows is not None
                    )
                    if self.tp_size > 1 and self.iris_state is not None
                    else None
                ),
                batch_invariant=global_server_args_dict["numerics"]
                in BITWISE_ENVELOPES,
            )
            combined = self._reduce_expert_routes(
                routes=routes,
                route_map=None,
                num_tokens=compressed.shape[0],
                route_weights=topk_weights,
                owned_rows=owned_global_rows is not None,
            )
        if not project:
            assert owned_global_rows is None
            return combined, None
        if owned_global_rows is not None:
            output = iris.acquire_owned_post_expert_output(
                owned_global_rows, self.post_expert_proj.output_size
            )
            return self.post_expert_proj.forward_owned_rows(combined, output), None
        return self.post_expert_proj.forward_with_mailbox(
            combined,
            boundary_gathers_mailbox(
                combined.shape[0], self.post_expert_proj.output_size, 1
            ),
        )


class MuseSparkResidualGate(nn.Module):
    """Per-channel residual gate and its pointer-stable derived coefficients."""

    def __init__(self, hidden_size: int, temperature: float) -> None:
        super().__init__()
        self.gate = nn.Parameter(torch.empty(hidden_size))
        self.gate_temperature = temperature
        self._coefficients: tuple[torch.Tensor, torch.Tensor] | None = None

    def refresh_gate(self) -> None:
        """Refresh pointer-stable gate coefficients after loading new weights."""
        alpha, beta = residual_gate_coefficients(
            self.gate,
            self.gate_temperature,
            RESIDUAL_GATE_EPS,
        )
        if self._coefficients is None:
            self._coefficients = alpha, beta
        else:
            self._coefficients[0].copy_(alpha)
            self._coefficients[1].copy_(beta)

    def _gate_coefficients(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._coefficients is None:
            self.refresh_gate()
        assert self._coefficients is not None
        return self._coefficients


class MuseSparkDecoderLayer(nn.Module):
    """One unified attention/MoE layer with an FP32 residual stream."""

    def __init__(
        self,
        config: MuseSparkConfig,
        mapping: Mapping,
        layer_id: int,
        quant_config: QuantizationConfig | None,
        prefix: str,
        alt_stream: torch.cuda.Stream | None,
    ) -> None:
        super().__init__()
        self.input_layernorm = MuseSparkRMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
            has_weight=True,
            weight_offset=ZERO_CENTERED_GAIN_OFFSET,
        )
        self.comm_manager = CommManager(
            mapping=mapping,
            layer_id=layer_id,
            is_moe=True,
            prev_is_moe=layer_id > 0,
            dense_batch_invariant=False,
            input_layernorm=None,
            post_attn_layernorm=None,
            query_sharded=False,
        )
        self.self_attn = MuseSparkAttention(
            config,
            mapping,
            layer_id,
            quant_config,
            add_prefix("self_attn", prefix),
            self.comm_manager.use_all_reduce(True),
        )
        self.pre_feedforward_layernorm = MuseSparkRMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
            has_weight=True,
            weight_offset=ZERO_CENTERED_GAIN_OFFSET,
        )
        self.mlp = MuseSparkMoE(
            config,
            mapping,
            quant_config,
            layer_id,
            add_prefix("mlp", prefix),
            alt_stream,
        )
        self.post_attention_layernorm = MuseSparkRMSNorm(
            hidden_size=config.hidden_size,
            eps=config.post_norm_eps,
            has_weight=False,
            weight_offset=ZERO_CENTERED_GAIN_OFFSET,
        )
        self.post_feedforward_layernorm = MuseSparkRMSNorm(
            hidden_size=config.hidden_size,
            eps=config.post_norm_eps,
            has_weight=True,
            weight_offset=ZERO_CENTERED_GAIN_OFFSET,
        )
        self.residual_gate_temperature = config.residual_gate_temperature
        self.post_attention_residual_gate = MuseSparkResidualGate(
            config.hidden_size, self.residual_gate_temperature
        )
        self.post_feedforward_residual_gate = MuseSparkResidualGate(
            config.hidden_size, self.residual_gate_temperature
        )
        self.iris_state: MuseIrisState | None = None

    def refresh_residual_gates(self) -> None:
        """Re-derive ``(alpha, beta)`` in place so captured graphs read reloaded gates."""
        self.post_attention_residual_gate.refresh_gate()
        self.post_feedforward_residual_gate.refresh_gate()

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        ctx: ForwardContext,
        attention_input: torch.Tensor | None,
        next_input_layernorm: MuseSparkRMSNorm | None,
        owned_global_rows: int | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Advance full or owned FP32 residual rows and prepare the next attention input."""
        iris = None
        owned_qkvg = False
        if owned_global_rows is not None:
            iris = self.iris_state
            assert iris is not None
            # Layer zero receives full input; later large batches project owned rows.
            owned_qkvg = (
                self.comm_manager.layer_id > 0
                and owned_global_rows >= MUSE_OWNED_QKVG_MIN_TOKENS
            )
        else:
            if ctx.forward_mode.is_mixed():
                # One call writes K/V for extend rows only; decode rows lose theirs.
                raise ValueError(
                    "Muse Spark does not run mixed prefill/decode rounds; "
                    "serve without --enable-mixed-batch"
                )
            if hidden_states.shape[0] == 0:
                return hidden_states, None
        if not owned_qkvg:
            # The input norm is row-wise, so either side of the comm agrees.
            if attention_input is None and owned_global_rows is None:
                norm = self.input_layernorm
                attention_input = norm(
                    self.comm_manager.pre_attn_comm(hidden_states, ctx),
                    norm.weight.dtype,
                )
            else:
                attention_input = self.comm_manager.pre_attn_comm(attention_input, ctx)
        attention_output, attention_sentinel, attention_ranks = self.self_attn(
            positions, attention_input, ctx, owned_global_rows if owned_qkvg else None
        )
        fused = None
        if owned_global_rows is not None:
            attention_output = iris.reduce_scatter_owned_rows(attention_output)
        elif (
            self.self_attn.o_proj.tp_size > 1
            and not self.self_attn.o_proj.reduce_results
            and self.pre_feedforward_layernorm.weight is not None
            and self.comm_manager.use_all_reduce(self.comm_manager.is_moe)
        ):
            iris = self.iris_state
            pre_norm = self.pre_feedforward_layernorm
            post_norm = self.post_attention_layernorm
            alpha, beta = self.post_attention_residual_gate._gate_coefficients()
            if iris is not None and iris.can_all_reduce_symmetric_boundary(
                attention_output,
                hidden_states,
                post_norm.weight,
                alpha,
                beta,
                pre_norm.weight,
            ):
                fused = iris.all_reduce_symmetric_boundary(
                    attention_output,
                    hidden_states,
                    post_norm.weight,
                    alpha,
                    beta,
                    pre_norm.weight,
                    post_norm.eps,
                    post_norm.weight_offset,
                    pre_norm.eps,
                    pre_norm.weight_offset,
                )
        if fused is not None:
            hidden_states, router_input, mlp_input = fused
        else:
            if owned_global_rows is None and attention_sentinel is None:
                if (
                    current_platform().is_cdna4
                    and self.self_attn.tp_size > 1
                    and self.comm_manager.use_all_reduce(True)
                ):
                    # Every Muse layer is MoE, so the layout stays the same.
                    (attention_output,) = _muse_all_reduce(
                        self.iris_state,
                        (attention_output,),
                        self.self_attn.o_proj.tp_group,
                    )
                else:
                    attention_output, hidden_states = self.comm_manager.post_attn_comm(
                        attention_output, hidden_states, ctx
                    )
            hidden_states, router_input, mlp_input = post_norm_residual_pre_norm(
                hidden_states,
                attention_output,
                self.post_attention_layernorm.weight,
                self.post_attention_layernorm.eps,
                self.post_attention_layernorm.weight_offset,
                *self.post_attention_residual_gate._gate_coefficients(),
                self.pre_feedforward_layernorm.weight,
                self.pre_feedforward_layernorm.eps,
                self.pre_feedforward_layernorm.weight_offset,
                pdl_enabled(),
                attention_sentinel,
                attention_ranks,
            )
        if owned_global_rows is None:
            mlp_input = self.comm_manager.pre_mlp_comm(mlp_input, ctx)
        else:
            assert router_input is not None and mlp_input is not None
        num_global_tokens, max_num_tokens_per_gpu = self.comm_manager.get_num_tokens(
            ctx
        )
        post_norm = self.post_feedforward_layernorm
        alpha, beta = self.post_feedforward_residual_gate._gate_coefficients()
        next_weight, next_eps, next_weight_offset = (
            (None, 0.0, 0.0)
            if next_input_layernorm is None
            else (
                next_input_layernorm.weight,
                next_input_layernorm.eps,
                next_input_layernorm.weight_offset,
            )
        )
        projection = self.mlp.post_expert_proj
        # One decode row projects only this rank's post-expert columns; the
        # boundary gathers them, which beats streaming the replicated weight.
        project_in_boundary = (
            owned_global_rows is None
            and self.iris_state is not None
            and projection.replicated
            and not projection.gather_column_view
            and not _gemm_dispatch_required()
            and resolve_kernel_override("residual", "post_norm_residual_pre_norm", None)
            is None
            and self.iris_state.can_gemv_all_gather_boundary(
                mlp_input.shape[0],
                projection.weight,
                hidden_states,
                post_norm.weight,
                alpha,
                beta,
                next_weight,
            )
        )
        mlp_output, sentinel = self.mlp(
            mlp_input,
            router_input,
            num_global_tokens,
            max_num_tokens_per_gpu,
            owned_global_rows,
            project=not project_in_boundary,
        )
        if project_in_boundary:
            return self.iris_state.gemv_all_gather_boundary(
                mlp_output,
                projection.weight,
                hidden_states,
                post_norm.weight,
                alpha,
                beta,
                next_weight,
                post_norm.eps,
                post_norm.weight_offset,
                next_eps,
                next_weight_offset,
            )
        boundary = (
            post_norm_residual_pre_norm_bf16
            if current_platform().is_cdna4
            else post_norm_residual_pre_norm
        )
        boundary_outputs = boundary(
            hidden_states,
            mlp_output,
            post_norm.weight,
            post_norm.eps,
            post_norm.weight_offset,
            alpha,
            beta,
            next_weight,
            next_eps,
            next_weight_offset,
            pdl_enabled(),
            None if owned_global_rows is not None else sentinel,
            1,
        )
        hidden_states, next_attention_input = boundary_outputs[0], boundary_outputs[-1]
        if (
            owned_global_rows is not None
            and next_attention_input is not None
            and owned_global_rows < MUSE_OWNED_QKVG_MIN_TOKENS
        ):
            next_attention_input = iris.all_gather_owned_rows(
                next_attention_input, owned_global_rows, False
            )
        return hidden_states, next_attention_input


def _widen_modelopt_exclusions(quant_config: QuantizationConfig | None) -> None:
    """Rewrite ModelOpt substring exclusions into TokenSpeed's glob form, in place."""
    if quant_config is None or not quant_config.exclude_modules:
        return
    quant_config.exclude_modules = [
        pattern if "*" in pattern else f"*{pattern}*"
        for pattern in quant_config.exclude_modules
    ]


class MuseSparkModel(BaseTransformerModel):
    """Text decoder using TokenSpeed's ordinary MHA cache route."""

    layer_cls = MuseSparkDecoderLayer

    def __init__(
        self,
        config: MuseSparkConfig,
        mapping: Mapping,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> None:
        _widen_modelopt_exclusions(quant_config)
        exclude_fully_excluded_fused_modules(
            quant_config,
            num_layers=config.num_hidden_layers,
            num_experts=config.num_local_experts,
            prefix=prefix,
        )
        super().__init__(
            config=config,
            mapping=mapping,
            quant_config=quant_config,
            prefix=prefix,
        )
        self.norm = MuseSparkRMSNorm(
            hidden_size=config.hidden_size,
            eps=config.rms_norm_eps,
            has_weight=True,
            weight_offset=OUTPUT_GAIN_OFFSET,
        )
        self._final_norm_op = self._build_final_norm_op()
        self.iris_state: MuseIrisState | None = None

    def resolve_layers(
        self,
        config: MuseSparkConfig,
        quant_config: QuantizationConfig | None,
        prefix: str,
    ) -> nn.ModuleList:
        """Build the decoder layers around one shared auxiliary stream."""
        self.alt_stream = torch.cuda.Stream() if torch.cuda.is_available() else None
        return make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: self.layer_cls(
                config=config,
                layer_id=idx,
                mapping=self.mapping,
                quant_config=quant_config,
                prefix=prefix,
                alt_stream=self.alt_stream,
            ),
            prefix=add_prefix("layers", prefix),
        )

    def _embed_hidden(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor | None,
    ) -> torch.Tensor:
        """Create the FP32 residual stream used by every execution mode."""
        hidden_states = (
            self.embed_tokens(input_ids, reduce_results=True)
            if input_embeds is None
            else input_embeds
        )
        # The prefill graph lands raw embeddings here too; the norm is idempotent.
        return weightless_rmsnorm(
            hidden_states, self.config.rms_norm_eps, torch.float32
        )

    def _can_use_owned_rows(self, rows: int, ctx: ForwardContext) -> bool:
        """Choose one layout for all decoder layers from the full token count."""
        if rows < MUSE_OWNED_ROWS_MIN_TOKENS:
            return False
        mapping = self.mapping
        if not (
            current_platform().is_cdna4
            and not ctx.forward_mode.is_idle()
            and not ctx.forward_mode.is_mixed()
            and mapping.pp_size == 1
            and mapping.attn.tp_size == mapping.moe.tp_ep_size
            and mapping.attn.tp_size in (4, 8)
            and (mapping.moe.tp_size == 1 or mapping.moe.ep_size == 1)
            and mapping.attn.tp_group == mapping.moe.tp_ep_group
            and self.config.hidden_size == 8192
            and self.config.moe_hidden_size == 4096
            and self.config.num_local_experts == 256
            and self.config.num_experts_per_tok == 8
            and not self.layers_to_capture
        ):
            return False
        for index, layer in enumerate(self.layers):
            projection = layer.mlp.post_expert_proj
            pre_expert = layer.mlp.pre_expert_proj
            if not (
                layer.mlp.expert_quant_kind in ("mxfp4", "unquant")
                and projection.workspace is None
                and pre_expert.weight.shape == (4096, 8192)
                and pre_expert.bias is None
                and isinstance(pre_expert.quant_method, UnquantizedLinearMethod)
                and projection.weight.shape == (8192, 4096)
                and projection.bias is None
                and isinstance(projection.quant_method, UnquantizedLinearMethod)
                and (
                    rows < MUSE_OWNED_QKVG_MIN_TOKENS
                    or index == 0
                    or layer.self_attn.qkv_proj.replicate_for_owned_rows
                )
            ):
                return False
        iris = self.iris_state
        return (
            iris is not None
            and iris.can_use_owned_rows(rows)
            and (mapping.moe.tp_size == 1 or rows <= iris.owned_route_max_rows)
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        ctx: ForwardContext,
        input_embeds: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, None]:
        hidden_states = self._embed_hidden(
            input_ids=input_ids,
            input_embeds=input_embeds,
        )
        rows, width = hidden_states.shape
        owned_global_rows = rows if self._can_use_owned_rows(rows, ctx) else None
        attention_input = None
        if owned_global_rows is not None:
            local_rows = (
                rows + self.mapping.attn.tp_size - 1
            ) // self.mapping.attn.tp_size
            first_row = self.mapping.attn.tp_rank * local_rows
            valid_rows = min(local_rows, max(0, rows - first_row))
            norm = self.layers[0].input_layernorm
            attention_input = norm(hidden_states, norm.weight.dtype)
            stream = torch.zeros(
                (local_rows, width),
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            stream[:valid_rows].copy_(hidden_states[first_row : first_row + valid_rows])
            hidden_states = stream
        for layer_index, layer in enumerate(self.layers):
            # Each FFN boundary emits the next layer's attention input rows.
            next_input_layernorm = (
                self.layers[layer_index + 1].input_layernorm
                if layer_index + 1 < len(self.layers)
                else None
            )
            # Preserve the owned path's direct call without Module hooks.
            layer_forward = layer.forward if owned_global_rows is not None else layer
            hidden_states, attention_input = layer_forward(
                positions,
                hidden_states,
                ctx,
                attention_input,
                next_input_layernorm,
                owned_global_rows,
            )
        if owned_global_rows is not None or not ctx.forward_mode.is_idle():
            hidden_states = self.norm(hidden_states, self.embed_tokens.weight.dtype)
        if owned_global_rows is not None:
            iris = self.iris_state
            assert iris is not None
            hidden_states = iris.all_gather_owned_rows(hidden_states, rows, False)
        return hidden_states, None


_CHECKPOINT_PREFIXES = (
    ("model.language_model.", "model."),
    ("model.vision_tower.", "vision_tower."),
    ("model.vision_adapter.", "vision_tower.adapter."),
    ("model.vision_projection.", "vision_projection."),
    ("model.audio_tower.", "audio_tower."),
)
_MULTIMODAL_TOWERS = frozenset({"vision_tower", "vision_projection", "audio_tower"})
_QKV_SHARDS = (
    ("q_proj", "q"),
    ("k_proj", "k"),
    ("v_proj", "v"),
    ("gate_proj", "g"),
)
_EXPERT_PARAMETERS = {
    "gate_up_proj": "w13_weight",
    "gate_up_proj_scale": "w13_weight_scale",
    "gate_up_proj_weight_scale": "w13_weight_scale",
    "gate_up_proj_weight_scale_2": "w13_weight_scale_2",
    "gate_up_proj_input_scale": "w13_input_scale",
    "down_proj": "w2_weight",
    "down_proj_scale": "w2_weight_scale",
    "down_proj_weight_scale": "w2_weight_scale",
    "down_proj_weight_scale_2": "w2_weight_scale_2",
    "down_proj_input_scale": "w2_input_scale",
}
# Quark exports one tensor per expert projection, e.g. ``mlp.experts.7.gate_proj.weight``.
_PER_EXPERT_TENSOR = re.compile(
    r"(?P<expert>\d+)\.(?P<projection>gate_proj|up_proj|down_proj)"
    r"\.(?P<kind>weight|weight_scale)"
)
# Projection -> (fused parameter stem, half of the fused rows it fills or None for all).
# The fused gate/up weight stacks the gate rows before the up rows.
_PER_EXPERT_TARGETS = {
    "gate_proj": ("w13", 0),
    "up_proj": ("w13", 1),
    "down_proj": ("w2", None),
}
_FUSED_ATTENTION_MEMBERS = ("q_proj", "k_proj", "v_proj", "gate_proj")
_FUSED_EXPERT_MEMBERS = ("gate_proj", "up_proj", "down_proj")


def _rename_checkpoint_prefix(name: str) -> str:
    for prefix, replacement in _CHECKPOINT_PREFIXES:
        if name.startswith(prefix):
            return replacement + name.removeprefix(prefix)
    return name


def _layer_id(parameter_name: str) -> int:
    return int(
        parameter_name.split(".layers.", maxsplit=1)[1].split(".", maxsplit=1)[0]
    )


def resolve_per_expert_tensor(name: str) -> tuple[str, int, int | None] | None:
    """Return where one expert's own projection tensor lands in the fused expert weights.

    Args:
        name: A tensor name as the checkpoint spells it.

    Returns:
        ``None`` unless ``name`` holds a single expert's projection, as in
        ``mlp.experts.7.gate_proj.weight``. Otherwise the fused parameter's
        dotted name, the expert index, and the half of the fused rows the
        tensor fills (``0`` for gate, ``1`` for up), or ``None`` when it fills
        every row.
    """
    head, _, expert_tensor = _rename_checkpoint_prefix(name).partition(".mlp.experts.")
    match = _PER_EXPERT_TENSOR.fullmatch(expert_tensor)
    if match is None:
        return None
    stem, half = _PER_EXPERT_TARGETS[match["projection"]]
    suffix = "_weight" if match["kind"] == "weight" else "_weight_scale"
    return f"{head}.mlp.experts.{stem}{suffix}", int(match["expert"]), half


def exclude_fully_excluded_fused_modules(
    quant_config: QuantizationConfig | None,
    num_layers: int,
    num_experts: int,
    prefix: str,
) -> None:
    """Exclude each fused module whose every member projection the checkpoint excludes.

    Quark lists excluded projections one by one (``self_attn.q_proj``,
    ``mlp.experts.7.gate_proj``), but this model builds one QKVG projection and
    one expert module per layer, and matches exclusions against those fused
    names. A fused module with only some members excluded cannot be built.

    Args:
        quant_config: The checkpoint's quantization config; its
            ``ignored_layers`` gains the fused names in place.
        num_layers: Decoder layer count.
        num_experts: Routed experts per layer.
        prefix: The decoder model's module prefix.
    """
    if quant_config is None or not quant_config.ignored_layers:
        return
    ignored = set(quant_config.ignored_layers)
    fused_exclusions = []
    for layer_id in range(num_layers):
        layer = add_prefix(f"layers.{layer_id}", prefix)
        fused_modules = (
            (
                f"{layer}.self_attn.qkv_proj",
                [f"{layer}.self_attn.{member}" for member in _FUSED_ATTENTION_MEMBERS],
            ),
            (
                f"{layer}.mlp.experts",
                [
                    f"{layer}.mlp.experts.{expert}.{member}"
                    for expert in range(num_experts)
                    for member in _FUSED_EXPERT_MEMBERS
                ],
            ),
        )
        for fused, members in fused_modules:
            excluded = sum(member in ignored for member in members)
            if excluded == len(members):
                fused_exclusions.append(fused)
            elif excluded:
                raise ValueError(
                    f"checkpoint excludes only some projections of '{fused}'"
                )
    quant_config.ignored_layers = [*quant_config.ignored_layers, *fused_exclusions]


def resolve_parameter_name(name: str) -> tuple[str, str | None]:
    """Return the parameter a checkpoint tensor fills and the shard it lands in.

    Args:
        name: A tensor name as the checkpoint spells it.

    Returns:
        The parameter's dotted name and, for a projection packed into a wider
        GEMM, that projection's shard id; ``None`` when the tensor fills the
        whole parameter.
    """
    name = _rename_checkpoint_prefix(name)
    head, _, expert_tensor = name.partition(".mlp.experts.")
    if expert_tensor:
        if expert_tensor == "post_expert_norm.weight":
            return f"{head}.mlp.post_expert_norm.weight", None
        parameter = _EXPERT_PARAMETERS.get(expert_tensor)
        if parameter is None:
            raise ValueError(f"checkpoint tensor '{name}' has no parameter")
        return f"{head}.mlp.experts.{parameter}", None
    for projection, shard_id in _QKV_SHARDS:
        needle = f".{projection}."
        if needle in name:
            return name.replace(needle, ".qkv_proj."), shard_id
    return name, None


def restore_rotary_pairs(weight: torch.Tensor, head_dim: int) -> torch.Tensor:
    """Interleave a query or key projection's rotary row pairs within each head.

    Args:
        weight: The ``[heads * head_dim, hidden]`` projection as the checkpoint
            stores it, each head's even-indexed rows before its odd-indexed ones.
        head_dim: Per-head width.

    Returns:
        The same rows reordered so each rotated pair is adjacent, which is the
        layout the interleaved rotary embedding reads.
    """
    rows, columns = weight.shape
    return (
        weight.reshape(rows // head_dim, 2, head_dim // 2, columns)
        .transpose(1, 2)
        .reshape(rows, columns)
    )


def load_expert_input_scale(
    parameter: nn.Parameter, loaded_weight: torch.Tensor, expert_start: int
) -> None:
    """Fill an expert activation scale, per expert where the projection reads its own rows.

    Args:
        parameter: The layer's activation scale, one element or one per expert.
        loaded_weight: The calibrated scale of every expert; a gated projection
            carries one column per half.
        expert_start: First expert this rank owns.
    """
    scales = loaded_weight.detach().to(dtype=torch.float32, device=parameter.device)
    if scales.dim() == 2 and not torch.equal(scales[:, 0], scales[:, 1]):
        raise ValueError("a gated projection's halves read one set of rows")
    if scales.dim() == 2:
        scales = scales[:, 0]
    if parameter.data.numel() == 1:
        # Every expert reads the same rows, quantized once before routing, so the
        # widest scale is the only one that keeps them all in range.
        parameter.data.fill_(float(scales.amax()))
        return
    local = parameter.data.numel()
    parameter.data.copy_(scales[expert_start : expert_start + local])


def shard_expert_projection(
    loaded_weight: torch.Tensor,
    target: torch.Tensor,
    tp_rank: int,
    tp_size: int,
    *,
    gate_up: bool,
    fused_gate_up: bool,
) -> torch.Tensor:
    """Select one TP slice in the checkpoint's packed-weight or scale units.

    Gate and up own matching intermediate rows; the down projection owns the
    corresponding input columns. Fused gate/up tensors need a slice from each
    half, rather than one contiguous slice of their concatenation.

    Args:
        loaded_weight: Full expert projection or its block scales, after EP slicing.
        target: Local parameter view, including only the selected gate/up half
            for a separately stored projection.
        tp_rank: Rank within the expert tensor-parallel group.
        tp_size: Number of expert tensor-parallel ranks.
        gate_up: Select output rows when true, or down-projection input columns.
        fused_gate_up: Whether the loaded tensor concatenates both gate and up.

    Returns:
        The checkpoint shard corresponding to ``target``.
    """
    if tp_size == 1:
        return loaded_weight
    dim = -2 if gate_up else -1
    width = target.shape[dim]
    if fused_gate_up:
        width //= 2
        gate, up = loaded_weight.chunk(2, dim=dim)
        return torch.cat(
            (
                gate.narrow(dim, tp_rank * width, width),
                up.narrow(dim, tp_rank * width, width),
            ),
            dim=dim,
        )
    return loaded_weight.narrow(dim, tp_rank * width, width)


class MuseSparkForConditionalGeneration(BaseCausalLM):
    """Muse Spark conditional generation model with shared vision and audio towers."""

    model_cls = MuseSparkModel

    def prepare_communication_runtime(self, max_num_tokens: int) -> bool:
        """Prepare and bind Muse's TP4/TP8 state before cache sizing and capture."""
        from tokenspeed.runtime.distributed.comm_backend.auto import AutoBackend

        mapping = self.mapping
        backend = get_global_backend()
        group = mapping.attn.tp_group
        if (
            not current_platform().is_cdna4
            or mapping.attn.tp_size not in (4, 8)
            or group != mapping.moe.tp_ep_group
            or not isinstance(backend, AutoBackend)
            or not backend.can_prepare_all_reduce_buffers(group, torch.bfloat16)
        ):
            return False
        from tokenspeed_kernel.ops.communication._iris.muse_spark import (
            prepare_muse_communication,
        )

        capacity = min(max_num_tokens, 8192) * self.config.hidden_size
        owned_route_max_rows = (
            min(max_num_tokens, 8192)
            if mapping.moe.tp_size > 1
            and mapping.moe.ep_size == 1
            and self.config.moe_hidden_size == 4096
            and self.config.num_experts_per_tok == 8
            else 0
        )
        state = self.model.iris_state
        if state is None:
            state = prepare_muse_communication(
                group=process_group_manager.get_device_process_group(group),
                rank_in_group=group.index(dist.get_rank()),
                staged_max_numel=min(capacity, (512 << 10) // torch.bfloat16.itemsize),
                producer_direct_max_numel=capacity,
                enable_lamport=True,
                owned_route_max_rows=owned_route_max_rows,
                all_reduce=backend.iris.state(group, torch.bfloat16),
                device=torch.device(f"cuda:{torch.cuda.current_device()}"),
            )
            backend.iris.install(group, state.all_reduce)
            backend.iris.install_all_gather(group, state.gather)
            backend.iris.install_broadcast(group, state.broadcast)
        elif (
            state.all_reduce.producer_direct_max_numel < capacity
            or state.owned_route_max_rows < owned_route_max_rows
        ):
            raise RuntimeError(
                "Muse communication was prepared below the requested capacity"
            )
        self.model.iris_state = state
        for layer in self.model.layers:
            layer.iris_state = state
            layer.self_attn.iris_state = state
            layer.self_attn.qkv_proj.iris_state = state
            layer.mlp.iris_state = state
            for projection in (layer.mlp.pre_expert_proj, layer.mlp.post_expert_proj):
                if isinstance(projection, _MuseGatheredLinear):
                    projection.iris_state = state
        return True

    def __init__(
        self,
        config: MuseSparkConfig,
        mapping: Mapping,
        quant_config: QuantizationConfig | None,
        is_multimodal_active: bool,
        mm_attention_backend: str | None,
    ) -> None:
        if mapping.pp_size > 1:
            raise ValueError(
                "Muse Spark builds every layer on each rank; pp_size must be 1"
            )
        super().__init__(
            config=config,
            mapping=mapping,
            quant_config=quant_config,
            prefix="",
            encoder_only=False,
        )
        self.is_multimodal_active = is_multimodal_active
        self._loaded_audio: set[str] = set()
        self.multimodal_embedder = (
            MultimodalEmbedder(encoder_mapping=mapping.vision)
            if is_multimodal_active
            else None
        )
        if not is_multimodal_active:
            self.vision_tower = None
            self.vision_projection = None
            self.audio_tower = None
            return

        self.vision_tower = MuseSparkVisionTower(
            raw_config=config.vision_config,
            out_hidden_size=config.out_hidden_size,
            projector_hidden_size=config.projector_hidden_size,
            mapping=mapping,
            prefix="vision_tower",
            mm_attention_backend=mm_attention_backend,
        )
        vision = mapping.vision
        self.vision_projection = RowParallelLinear(
            input_size=config.projector_hidden_size,
            output_size=config.hidden_size,
            bias=False,
            input_is_parallel=False,
            skip_bias_add=False,
            params_dtype=None,
            quant_config=None,
            prefix="vision_projection",
            tp_rank=vision.tp_rank,
            tp_size=vision.tp_size,
            tp_group=vision.tp_group,
            reduce_results=True,
        )
        self.audio_tower = MuseSparkAudioTower.from_config(
            config.audio_config, config.hidden_size
        )

    def get_vision_feature(self, items: list[MultimodalDataItem]) -> torch.Tensor:
        value = self.vision_tower(items)
        value, _ = self.vision_projection(value)
        return value

    def get_audio_feature(self, items: list[MultimodalDataItem]) -> torch.Tensor:
        """Encode the items' mono 24 kHz waveforms into concatenated decoder-width tokens."""
        weight = self.audio_tower.proj.weight
        waveforms = [
            item.feature.to(device=weight.device, dtype=weight.dtype) for item in items
        ]
        return self.audio_tower(waveforms)

    def get_multimodal_encoder_specs(self) -> dict[Modality, EncoderSpec]:
        specs: dict[Modality, EncoderSpec] = {}
        if self.audio_tower is not None:
            specs[Modality.AUDIO] = EncoderSpec(
                fn=self.get_audio_feature,
                deepstack=False,
                make_warmup_items=self.audio_tower.make_warmup_items,
            )
        if self.vision_tower is None:
            return specs
        for modality in (Modality.IMAGE, Modality.VIDEO):
            specs[modality] = EncoderSpec(
                fn=self.get_vision_feature,
                deepstack=False,
                make_warmup_items=partial(
                    self.vision_tower.make_warmup_items, modality
                ),
            )
        return specs

    def pad_input_ids(
        self,
        input_ids: list[int],
        mm_inputs: MultimodalInputs,
    ) -> list[int]:
        return pad_input_tokens(input_ids=input_ids, mm_inputs=mm_inputs)

    @torch.no_grad()
    def multimodal_input_embeds(
        self,
        input_ids: torch.Tensor,
        ctx: ForwardContext,
        multimodal_context: MultimodalForwardContext | None,
    ) -> torch.Tensor | None:
        """Return the token embeddings merged with new encoder outputs, or None without any.

        The prefill graph lands the result in its static embedding buffer.
        """
        if multimodal_context is None or not multimodal_context.has_extend_inputs():
            return None
        if not self.is_multimodal_active:
            raise RuntimeError(
                "Muse Spark received multimodal inputs while its encoders are disabled"
            )
        if ctx.forward_mode.is_decode_or_idle():
            return None
        # No encoder spec is deepstack, so the embedder returns no model kwargs.
        input_embeds, _ = self.multimodal_embedder.apply(
            input_ids=input_ids,
            text_embedding=self.model.embed_tokens,
            ctx=multimodal_context,
            encoders=self.get_multimodal_encoder_specs(),
            multimodal_model=self,
        )
        return input_embeds

    def forward(
        self,
        ctx: ForwardContext,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        input_embeds = self.multimodal_input_embeds(
            input_ids, ctx, kwargs.pop("multimodal_context", None)
        )
        if input_embeds is not None:
            kwargs["input_embeds"] = input_embeds
        return super().forward(ctx, input_ids, positions, **kwargs)

    @classmethod
    def get_model_config_for_expert_location(
        cls, config: MuseSparkConfig
    ) -> ModelConfigForExpertLocation:
        """Describe the fixed contiguous expert layout of ``config`` to the runtime."""
        return ModelConfigForExpertLocation(
            num_layers=config.num_hidden_layers,
            num_logical_experts=config.num_local_experts,
            num_groups=None,
        )

    def resolve_logits_processor(self, config: MuseSparkConfig) -> LogitsProcessor:
        return LogitsProcessor(
            config,
            skip_all_gather=self.mapping.attn.has_dp,
            do_argmax=False,
            logit_scale=config.output_multiplier,
            tp_rank=self.mapping.attn.tp_rank,
            tp_size=self.mapping.attn.tp_size,
            tp_group=self.mapping.attn.tp_group,
            dp_lm_head_tp=False,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> None:
        params = dict(self.named_parameters())
        for name, loaded_weight in weights:
            if name.endswith((".self_attn.q_proj.weight", ".self_attn.k_proj.weight")):
                loaded_weight = restore_rotary_pairs(
                    loaded_weight, self.config.head_dim
                )
            per_expert = resolve_per_expert_tensor(name)
            if per_expert is not None:
                parameter_name, expert_id, half = per_expert
                moe = self.model.layers[_layer_id(parameter_name)].mlp
                local_expert = expert_id - moe.expert_start
                if not 0 <= local_expert < moe.experts_per_rank:
                    continue
                parameter = params.get(parameter_name)
                if parameter is None:
                    raise ValueError(f"checkpoint tensor '{name}' has no parameter")
                target = parameter.data[local_expert]
                if half is not None:
                    rows = target.shape[0] // 2
                    target = target.narrow(0, half * rows, rows)
                loaded_weight = shard_expert_projection(
                    loaded_weight,
                    target,
                    moe.tp_rank,
                    moe.tp_size,
                    gate_up=half is not None,
                    fused_gate_up=False,
                )
                if target.shape != loaded_weight.shape:
                    raise ValueError(
                        f"checkpoint tensor '{name}' has shape "
                        f"{tuple(loaded_weight.shape)}; expected {tuple(target.shape)}"
                    )
                target.copy_(loaded_weight)
                continue
            parameter_name, shard_id = resolve_parameter_name(name)
            tower = parameter_name.split(".", maxsplit=1)[0]
            if tower in _MULTIMODAL_TOWERS and not self.is_multimodal_active:
                continue
            parameter = params.get(parameter_name)
            if parameter is None:
                raise ValueError(f"checkpoint tensor '{name}' has no parameter")
            if tower == "audio_tower":
                self._loaded_audio.add(parameter_name)
            if shard_id is not None:
                parameter.weight_loader(parameter, loaded_weight, shard_id)
                continue
            if ".mlp.experts." in parameter_name:
                moe = self.model.layers[_layer_id(parameter_name)].mlp
                if parameter_name.endswith("_input_scale"):
                    load_expert_input_scale(parameter, loaded_weight, moe.expert_start)
                    continue
                loaded_weight = loaded_weight.narrow(
                    0, moe.expert_start, moe.experts_per_rank
                )
                if parameter_name.endswith(("_weight", "_weight_scale")):
                    gate_up = ".w13_" in parameter_name
                    loaded_weight = shard_expert_projection(
                        loaded_weight,
                        parameter,
                        moe.tp_rank,
                        moe.tp_size,
                        gate_up=gate_up,
                        fused_gate_up=gate_up,
                    )
                default_weight_loader(parameter, loaded_weight)
                continue
            if parameter_name.endswith(".mlp.gate.weight"):
                loaded_weight = loaded_weight.T.contiguous()
            loader = getattr(parameter, "weight_loader", default_weight_loader)
            loader(parameter, loaded_weight)
        # Reloaded gates (weight updates) must reach the cached coefficients.
        for layer in self.model.layers:
            layer.refresh_residual_gates()

    def post_quant_warmup(self) -> None:
        """Validate loaded audio weights and warm bounded expert kernel families."""
        # A missing or partial tower would encode audio with random weights.
        if self.audio_tower is not None:
            audio_parameters = {
                name
                for name, _ in self.named_parameters()
                if name.startswith("audio_tower.")
            }
            if self._loaded_audio != audio_parameters:
                missing = sorted(audio_parameters - self._loaded_audio)
                raise ValueError(f"checkpoint is missing audio tensors: {missing[:5]}")
        bf16_mlps = [
            layer.mlp for layer in self.model.layers if not layer.mlp.quantized_experts
        ]
        experts_to_warm = (
            []
            if global_server_args_dict["disable_autotune"]
            else [
                layer.mlp.experts
                for layer in self.model.layers
                if layer.mlp.quantized_experts
                and layer.mlp.experts.plan.get("warmup") is not None
            ]
        )
        if not bf16_mlps and not experts_to_warm:
            return
        # Match the global routing span reserved by execution/device.py.
        chunk = int(global_server_args_dict["chunked_prefill_size"])
        max_tokens = (
            chunk
            if chunk > 0
            else int(global_server_args_dict["max_prefill_tokens"])
            + int(global_server_args_dict["max_model_len"])
        )
        decode_rows = max(
            int(global_server_args_dict["max_cudagraph_capture_size"] or 0),
            int(global_server_args_dict["max_num_seqs"] or 0),
        )
        if global_server_args_dict["speculative_algorithm"]:
            decode_rows *= int(global_server_args_dict["speculative_num_draft_tokens"])
        max_tokens = max(max_tokens, decode_rows)
        # BF16 expert layers compile their route kernels before graph capture.
        # This is JIT warmup, not tuning, so disable_autotune does not skip it.
        for mlp in bf16_mlps:
            bf16_moe_routes_warmup(
                mlp.experts.w13_weight,
                mlp.experts.w2_weight,
                mlp.topk,
                mlp.expert_start,
                max_num_tokens=max_tokens,
                batch_invariant=global_server_args_dict["numerics"]
                in BITWISE_ENVELOPES,
            )
        for experts in experts_to_warm:
            if not experts._weights_processed:
                raise RuntimeError("MoE weights must be processed before warmup")
            moe_warmup(experts.plan, experts, max_num_tokens=max_tokens)


EntryClass = MuseSparkForConditionalGeneration

__all__ = [
    "MuseSparkForConditionalGeneration",
    "resolve_parameter_name",
    "restore_rotary_pairs",
]
