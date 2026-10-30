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

"""Explicit cost rules for dense matmul, attention and collective traffic."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod

from tokenspeed_kernel.op_profile import (
    CostRegistry,
    OpCost,
    OpRecord,
    TensorDescription,
)


def _infer_matmul_compute_type(
    op: OpRecord, *operands: TensorDescription
) -> str | None:
    """Select a peak for uncast BF16/FP16 GPU matrix products.

    This selects a normalization peak, not the kernel's actual instruction mix.
    FP32/TF32, quantized, mixed-dtype and autocast execution need more metadata.
    """
    dtype, device = operands[0].dtype, operands[0].device
    if (
        op.autocast_enabled
        or not device.startswith("cuda:")
        or any(tensor.dtype != dtype or tensor.device != device for tensor in operands)
    ):
        return None
    return {"torch.bfloat16": "bf16_tensor", "torch.float16": "fp16_tensor"}.get(dtype)


@dataclass(frozen=True)
class DenseMatmulCost:
    """Plain dense 2D GEMM cost using explicit argument roles.

    No padding, quantization, fusion or implicit workspace is modeled. HBM
    traffic remains unknown; logical traffic assumes each operand read once.
    """

    lhs: str
    rhs: str
    output: str

    def __call__(self, op: OpRecord) -> OpCost:
        try:
            lhs = op.inputs[self.lhs]
            rhs = op.inputs[self.rhs]
            output = op.outputs[self.output]
        except KeyError as exc:
            raise ValueError(f"Missing GEMM tensor role: {exc}") from exc
        if len(lhs.shape) != 2 or len(rhs.shape) != 2:
            raise ValueError("DenseMatmulCost requires 2D operands")

        m, k = lhs.shape
        rhs_k, n = rhs.shape
        if k != rhs_k or output.shape != (m, n):
            raise ValueError("GEMM shapes do not match")

        flops = 2 * m * n * k
        return OpCost(
            useful_flops=flops,
            executed_flops=flops,
            logical_read_bytes=lhs.logical_bytes + rhs.logical_bytes,
            logical_write_bytes=output.logical_bytes,
            estimated_hbm_bytes=None,
            rule="dense_mm.v1",
            assumptions=(
                "unquantized dense A@B without padding or fusion",
                "each operand read once; output written once",
                "logical bytes do not measure HBM traffic",
            ),
            compute_type=_infer_matmul_compute_type(op, lhs, rhs),
        )


@dataclass(frozen=True)
class _DenseProjectionCost:
    """Cost of A @ B.transpose(-1, -2) for registered torch_mm and torch_bmm."""

    ndim: int
    hbm_policy: str

    def __call__(self, op: OpRecord) -> OpCost:
        try:
            lhs = op.inputs["A"]
            rhs = op.inputs["B"]
            output = op.outputs["return"]
        except KeyError as exc:
            raise ValueError(f"Missing dense projection tensor role: {exc}") from exc
        if len(lhs.shape) != self.ndim or len(rhs.shape) != self.ndim:
            raise ValueError("Dense projection operand ranks do not match the rule")
        batch_shape = lhs.shape[:-2]
        if lhs.dtype != rhs.dtype or batch_shape != rhs.shape[:-2]:
            raise ValueError("Dense projection operands must match dtype and batch")

        m, k = lhs.shape[-2:]
        n, rhs_k = rhs.shape[-2:]
        if k != rhs_k or output.shape != (*batch_shape, m, n):
            raise ValueError(
                "Dense projection shapes do not match A @ B.transpose(-1, -2)"
            )

        batch_size = lhs.shape[0] if self.ndim == 3 else 1
        read_bytes = lhs.logical_bytes + rhs.logical_bytes
        read_bytes += sum(
            op.inputs[name].logical_bytes
            for name in ("bias", "alpha")
            if name in op.inputs
        )
        assumptions = (
            "useful dense matmul FLOPs only; bias, alpha and dtype conversions excluded",
            "output dtype does not select the compute peak",
            "compute peak inferred from matching BF16/FP16 GPU operands",
            "logical IO does not establish physical HBM traffic",
        )
        estimated_hbm_bytes = None
        if self.hbm_policy == "logical_io_once":
            estimated_hbm_bytes = read_bytes + output.logical_bytes
            assumptions += (
                "HBM estimate assumes one operand read and one final output write; "
                "cache reuse, reloads, intermediate buffers and workspace excluded",
            )
        return OpCost(
            useful_flops=2 * batch_size * m * n * k,
            executed_flops=None,
            logical_read_bytes=read_bytes,
            logical_write_bytes=output.logical_bytes,
            estimated_hbm_bytes=estimated_hbm_bytes,
            rule=f"dense_projection_{self.ndim}d.v1",
            assumptions=assumptions,
            compute_type=_infer_matmul_compute_type(op, lhs, rhs),
        )


def register_compute_costs(registry: CostRegistry, *, hbm_policy: str) -> None:
    """Register costs for the opted-in torch_mm and torch_bmm implementations.

    Args:
        registry: Cost registry receiving the two named rules.
        hbm_policy: 'unavailable' or the rough estimate 'logical_io_once'.

    Compute types are inferred from matching BF16/FP16 GPU operands. Unsupported
    dtypes and autocast leave MFU unavailable; output dtype does not select a peak.
    Counts matrix products only; bias, scaling and conversion FLOPs are excluded.
    Registers the rules in place and returns None.
    """
    if hbm_policy not in {"unavailable", "logical_io_once"}:
        raise ValueError(f"Unsupported HBM policy: {hbm_policy}")
    for name, ndim in (("torch_mm.v1", 2), ("torch_bmm.v1", 3)):
        registry.register(
            name,
            _DenseProjectionCost(ndim=ndim, hbm_policy=hbm_policy),
        )


@dataclass(frozen=True)
class CollectiveTraffic:
    """NCCL-tests normalized traffic, not measured link or HBM bytes.

    total_bytes is the full AllReduce/ReduceScatter input or AllGather output,
    already including element size. group identifies the participating ranks.
    """

    collective: str
    total_bytes: int
    world_size: int
    group: str

    def __post_init__(self) -> None:
        if self.collective not in {
            "all_reduce_sum",
            "reduce_scatter_sum",
            "all_gather",
        }:
            raise ValueError(f"Unsupported collective: {self.collective}")
        if type(self.world_size) is not int or self.world_size < 1:
            raise ValueError("world_size must be a positive integer")
        if type(self.total_bytes) is not int or self.total_bytes < 0:
            raise ValueError("total_bytes must be a nonnegative integer byte count")
        if not self.group.strip():
            raise ValueError("Communication group identity is required")

    @property
    def bus_bytes(self) -> float:
        """Return per-rank equivalent bytes; divide by seconds to obtain busbw."""
        phases = 2 if self.collective == "all_reduce_sum" else 1
        return self.total_bytes * phases * (self.world_size - 1) / self.world_size


@dataclass(frozen=True)
class CollectiveCost:
    """Describe one rank's synchronous collective using its full-array tensor.

    collective selects all_reduce_sum, reduce_scatter_sum or all_gather.
    tensor and tensor_source (input/output) identify the full input for reduction
    or full output for gathering; in-place output buffers can be input arguments.
    world_size is the participating group size, not the machine's GPU count.
    group must uniquely identify that group within the profile session.
    """

    collective: str
    tensor: str
    tensor_source: str
    world_size: int
    group: str

    def __post_init__(self) -> None:
        # Validate group metadata before a runtime tensor is available.
        CollectiveTraffic(
            collective=self.collective,
            total_bytes=0,
            world_size=self.world_size,
            group=self.group,
        )
        if self.tensor_source not in {"input", "output"}:
            raise ValueError("tensor_source must be input or output")

    def __call__(self, op: OpRecord) -> OpCost:
        tensors = op.inputs if self.tensor_source == "input" else op.outputs
        if self.tensor not in tensors:
            raise ValueError(f"Missing collective full-array tensor: {self.tensor}")
        tensor = tensors[self.tensor]
        numel = prod(tensor.shape)
        if self.collective != "all_reduce_sum" and numel % self.world_size:
            raise ValueError("Full collective array must split evenly across ranks")

        total_bytes = tensor.logical_bytes
        shard_bytes = total_bytes // self.world_size
        read_bytes = shard_bytes if self.collective == "all_gather" else total_bytes
        write_bytes = (
            shard_bytes if self.collective == "reduce_scatter_sum" else total_bytes
        )
        return OpCost(
            useful_flops=0,
            executed_flops=None,
            logical_read_bytes=read_bytes,
            logical_write_bytes=write_bytes,
            estimated_hbm_bytes=None,
            rule="nccl_collective.v1",
            assumptions=(
                "total bytes already include dtype element size",
                "NCCL-tests per-rank normalized bus bytes, not physical link traffic",
                "equal partitions for ReduceScatter and AllGather",
                "reduction arithmetic excluded from model useful FLOPs",
                "collective work must join the timed stream before scope exit",
            ),
            communication=CollectiveTraffic(
                collective=self.collective,
                total_bytes=total_bytes,
                world_size=self.world_size,
                group=self.group,
            ),
        )


@dataclass(frozen=True)
class AttentionMatmulCost:
    """Fixed-length attention cost for explicit [batch, heads, sequence, dim] roles.

    q/k/v/output name recorded tensor arguments and return paths. mask selects
    full, causal_upper_left or causal_lower_right alignment for unpadded sequences.
    Compute type is inferred from Q/K/V GPU dtypes; unknown precision stays None.
    Counts useful QK^T and PV FLOPs with an FMA counted as two. GQA uses query heads;
    scaling, softmax, projections, dropout, padding and recompute are excluded.
    """

    q: str
    k: str
    v: str
    output: str
    mask: str

    def __call__(self, op: OpRecord) -> OpCost:
        try:
            q = op.inputs[self.q]
            k = op.inputs[self.k]
            v = op.inputs[self.v]
            output = op.outputs[self.output]
        except KeyError as exc:
            raise ValueError(f"Missing attention tensor role: {exc}") from exc
        if any(len(tensor.shape) != 4 for tensor in (q, k, v, output)):
            raise ValueError(
                "AttentionMatmulCost requires [batch, heads, sequence, dim] tensors"
            )

        batch_size, query_heads, query_length, qk_head_dim = q.shape
        key_batch, kv_heads, kv_length, key_head_dim = k.shape
        value_batch, value_heads, value_length, value_head_dim = v.shape
        if not (
            batch_size == key_batch == value_batch
            and qk_head_dim == key_head_dim
            and kv_heads == value_heads
            and kv_heads > 0
            and query_heads % kv_heads == 0
            and kv_length == value_length
            and output.shape == (batch_size, query_heads, query_length, value_head_dim)
        ):
            raise ValueError("Attention tensor shapes do not match")

        dimensions = (
            batch_size,
            query_heads,
            query_length,
            kv_length,
            qk_head_dim,
            value_head_dim,
        )
        if any(type(dim) is not int or dim < 1 for dim in dimensions):
            raise ValueError("Attention dimensions must be positive integers")

        triangle = min(query_length, kv_length)
        if self.mask == "full":
            pairs = query_length * kv_length
        elif self.mask == "causal_upper_left":
            # Row i sees min(i + 1, kv_length) keys.
            pairs = (
                triangle * (triangle + 1) // 2 + (query_length - triangle) * kv_length
            )
        elif self.mask == "causal_lower_right":
            # Row i sees max(0, kv_length - query_length + i + 1) keys.
            pairs = triangle * (2 * kv_length - triangle + 1) // 2
        else:
            raise ValueError(f"Unsupported attention mask: {self.mask}")

        flops = 2 * batch_size * query_heads * pairs * (qk_head_dim + value_head_dim)
        return OpCost(
            useful_flops=flops,
            executed_flops=None,
            logical_read_bytes=q.logical_bytes + k.logical_bytes + v.logical_bytes,
            logical_write_bytes=output.logical_bytes,
            estimated_hbm_bytes=None,
            rule="attention_matmuls.v1",
            assumptions=(
                f"useful QK^T and PV only; mask={self.mask}",
                "FMA counts as two; GQA uses query heads; QK and V dimensions are separate",
                "softmax, scale, projections, padding and recompute excluded",
                "logical tensor IO does not establish HBM traffic or executed FLOPs",
            ),
            compute_type=_infer_matmul_compute_type(op, q, k, v),
        )
