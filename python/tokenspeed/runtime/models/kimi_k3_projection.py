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

"""Kimi checkpoint layouts for DP projections; communication stays in DP Linear."""

from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

from tokenspeed.runtime.distributed.mapping import DenseLayerMapping, Mapping
from tokenspeed.runtime.distributed.process_group_manager import (
    process_group_manager as pg_manager,
)
from tokenspeed.runtime.layers.dense.fp8 import Fp8LinearMethod
from tokenspeed.runtime.layers.dense.unquant import UnquantizedLinearMethod
from tokenspeed.runtime.layers.linear import DPColumnParallelLinear, LinearBase
from tokenspeed.runtime.layers.quantization.base_config import QuantizeMethodBase
from tokenspeed.runtime.layers.quantization.fp8 import Fp8Config
from tokenspeed.runtime.models.deepseek_v3 import DeepseekV3FusedQkvAProjWithMqa
from tokenspeed.runtime.utils import ceil_div

if TYPE_CHECKING:
    from tokenspeed.runtime.execution.context import ForwardContext


def validate_projection_settings(
    mapping: Mapping, qkv_size: int, output_size: int
) -> tuple[DenseLayerMapping | None, DenseLayerMapping | None]:
    """Agree on target-model settings before constructing any sharded projection.

    Disabled ranks also join the world agreement on parsed sizes, so peers cannot
    enter subgroup preparation with different projection graphs. Return the
    independent QKV and output mappings; None selects the ordinary projection.
    """
    sizes = (qkv_size, output_size)
    if dist.is_initialized() and mapping.world_size > 1:
        pg_manager.init_process_group(mapping.world_group, backend="gloo")
        group = pg_manager.get_process_group("gloo", mapping.world_group)
        gathered = [None] * group.size()
        dist.all_gather_object(gathered, sizes, group=group)
        if any(value != sizes for value in gathered):
            raise ValueError(
                f"Kimi projection TP settings differ across ranks: {gathered}"
            )
    if any(size < 1 or mapping.world_size % size for size in sizes):
        raise ValueError(
            f"Kimi projection TP sizes {sizes} must be positive divisors of "
            f"world size {mapping.world_size}"
        )
    # Full-world DP implies TP1 and PP1; head and expert sharding are independent.
    if max(sizes) > 1 and (
        mapping.attn.dp_size != mapping.world_size
        or mapping.linear_attn.dp_size != mapping.world_size
        or mapping.attn.head_tp_size != 1
        or mapping.moe.ep_size != mapping.world_size
    ):
        raise ValueError(
            "Kimi projection TP requires attention TP1/DPworld, "
            "linear attention TP1, MoE TP1/EPworld and PP1"
        )
    qkv, output = (
        (
            None
            if size == 1
            else DenseLayerMapping(
                rank=mapping.rank,
                world_size=mapping.world_size,
                tp_size=size,
                dp_size=mapping.world_size // size,
            )
        )
        for size in sizes
    )
    return qkv, output


def projection_storage_rows(rows: int, tp_size: int, block_fp8: bool) -> int:
    """Pad at the end so each output shard meets its GEMM/scale alignment."""
    alignment = 128 if block_fp8 else 16
    return ceil_div(rows, tp_size * alignment) * tp_size * alignment


def projection_fp8_config() -> Fp8Config:
    """Describe the fused checkpoint's block scales, independent of its alias."""
    return Fp8Config(
        is_checkpoint_fp8_serialized=True,
        activation_scheme="dynamic",
        ignored_layers=[],
        weight_block_size=[128, 128],
        scale_fmt=None,
    )


def validate_projection_quantization(
    linear: LinearBase, method: QuantizeMethodBase | None, prefix: str
) -> None:
    """Require the method to match the supported BF16/block-FP8 buffer.

    Output projections pass their resolved quant_method. Merged QKV projections
    pass methods resolved from the original checkpoint config and prefixes to
    check that their source formats match the locally configured merged buffer.
    """
    if method is None or isinstance(method, UnquantizedLinearMethod):
        supported = linear.weight.dtype == torch.bfloat16
    else:
        supported = (
            isinstance(method, Fp8LinearMethod)
            and method.block_quant
            and method.quant_config.weight_block_size == [128, 128]
            and linear.weight.dtype == torch.float8_e4m3fn
        )
    if not supported:
        raise ValueError(
            f"Kimi projection TP supports BF16 or 128x128 block-FP8 weights: {prefix}"
        )


def _load_intersection(
    destination: torch.Tensor,
    source: torch.Tensor,
    segment_start: int,
    shard_start: int,
) -> None:
    """Copy the overlap of a global checkpoint segment and a local row shard."""
    start = max(segment_start, shard_start)
    end = min(segment_start + source.shape[0], shard_start + destination.shape[0])
    if start < end:
        destination[start - shard_start : end - shard_start].copy_(
            source[start - segment_start : end - segment_start]
        )


class KimiKDAColumnProj(DPColumnParallelLinear):
    """Contiguous shard of [q, k, v, g, f_a, b], not an attention-head shard.

    Each source segment and scale grid is intersected with the same fused-row
    shard. Padding codes are zero and their scales are one; no requantization
    is needed because every segment starts on a 128-row block boundary.
    """

    def __init__(
        self,
        hidden: int,
        proj: int,
        heads: int,
        head_dim: int,
        parallel: DenseLayerMapping,
        block_fp8: bool,
        prefix: str,
    ):
        if block_fp8 and (proj % 128 or head_dim % 128 or hidden % 128):
            raise ValueError("KDA block-FP8 projection segments must be 128-aligned")
        used_rows = 4 * proj + head_dim + heads
        super().__init__(
            input_size=hidden,
            output_size=used_rows,
            padded_output_size=projection_storage_rows(
                used_rows, parallel.tp_size, block_fp8
            ),
            parallel=parallel,
            params_dtype=torch.bfloat16,
            quant_config=projection_fp8_config() if block_fp8 else None,
            prefix=prefix,
        )
        self.used_rows = used_rows
        self.fp8_block_quant = block_fp8
        self.fp8_channel_quant = False
        self._segments = {
            "q": (0, proj),
            "k": (proj, proj),
            "v": (2 * proj, proj),
            "g": (3 * proj, proj),
            "f_a": (4 * proj, head_dim),
            "b": (4 * proj + head_dim, heads),
        }
        self._loaded_weight_shards: set[str] = set()
        self._loaded_scale_shards: set[str] = set()
        self.weight.data.zero_()
        if block_fp8:
            self.weight_scale_inv.data.fill_(1)

    def weight_loader(
        self, param: torch.nn.Parameter, source: torch.Tensor, shard_id: str
    ) -> None:
        self.weight_loader_v2(param, source, shard_id)

    def weight_loader_v2(
        self, param: torch.nn.Parameter, source: torch.Tensor, shard_id: str
    ) -> None:
        offset, rows = self._segments[shard_id]
        if param is self.weight:
            if source.dtype != param.dtype:
                raise TypeError(
                    f"KDA projection expects {param.dtype}, got {source.dtype}"
                )
            if source.shape != (rows, self.input_size):
                raise ValueError(
                    f"KDA {shard_id} weight shape {tuple(source.shape)}, "
                    f"expected {(rows, self.input_size)}"
                )
            block = 1
            loaded = self._loaded_weight_shards
        elif self.fp8_block_quant and param is self.weight_scale_inv:
            if source.shape != (ceil_div(rows, 128), self.input_size // 128):
                raise ValueError(
                    f"KDA {shard_id} scale shape {tuple(source.shape)}, "
                    f"expected {(ceil_div(rows, 128), self.input_size // 128)}"
                )
            block = 128
            loaded = self._loaded_scale_shards
        else:
            raise ValueError("Unexpected KDA projection parameter")
        _load_intersection(
            param.data,
            source,
            offset // block,
            self.tp_rank * (self.output_size_per_partition // block),
        )
        loaded.add(shard_id)

    def verify_load_complete(self) -> None:
        """Keep the merged checkpoint completeness check before GEMM preparation."""
        expected = set(self._segments)
        if self._loaded_weight_shards != expected or (
            self.fp8_block_quant and self._loaded_scale_shards != expected
        ):
            raise RuntimeError("Incomplete column-sharded KDA checkpoint")


class KimiMLAReplicatedProj(DeepseekV3FusedQkvAProjWithMqa):
    """Kimi-only tuple-return adapter; the DeepSeek parent's API stays unchanged.

    Accept the same optional scale/dtype inputs as the parent, plus the context
    consumed by the DP alternative. Empty owners need no local projection GEMM.
    """

    def forward(
        self,
        x: torch.Tensor,
        block_scale: torch.Tensor | None = None,
        output_dtype: torch.dtype | None = None,
        *,
        ctx: "ForwardContext",
    ) -> tuple[torch.Tensor, None]:
        if x.shape[0] == 0:
            return (
                x.new_empty((0, self.output_size), dtype=output_dtype or x.dtype),
                None,
            )
        return super().forward(x, block_scale, output_dtype), None


class KimiMLAColumnProj(DPColumnParallelLinear):
    """DP MLA projection with streamed BF16 or assembled block-FP8 loading."""

    def __init__(
        self,
        hidden: int,
        rows: int,
        parallel: DenseLayerMapping,
        block_fp8: bool,
        prefix: str,
    ):
        super().__init__(
            input_size=hidden,
            output_size=rows,
            padded_output_size=projection_storage_rows(
                rows, parallel.tp_size, block_fp8
            ),
            parallel=parallel,
            params_dtype=torch.bfloat16,
            quant_config=projection_fp8_config() if block_fp8 else None,
            prefix=prefix,
        )
        self.weight.data.zero_()
        if block_fp8:
            self.weight_scale_inv.data.fill_(1)

    def weight_loader(
        self,
        param: torch.nn.Parameter,
        source: torch.Tensor,
        begin_size: int | None = None,
    ) -> None:
        if begin_size is None:
            super().weight_loader(param, source)
            return
        # BF16 q_a / kv_a / gate stream into the canonical fused layout.
        if source.dtype != param.dtype:
            raise TypeError(f"MLA projection expects {param.dtype}, got {source.dtype}")
        _load_intersection(
            param.data,
            source,
            begin_size,
            self.tp_rank * self.output_size_per_partition,
        )

    def forward(
        self,
        x: torch.Tensor,
        block_scale: torch.Tensor | None = None,
        output_dtype: torch.dtype | None = None,
        *,
        ctx: "ForwardContext",
    ) -> tuple[torch.Tensor, None]:
        # DEP inputs carry complete activation rows, never a prequantized TP slice.
        if block_scale is not None or output_dtype not in (None, x.dtype):
            raise ValueError("DP MLA projection requires unquantized local activations")
        return super().forward(x, ctx=ctx)
