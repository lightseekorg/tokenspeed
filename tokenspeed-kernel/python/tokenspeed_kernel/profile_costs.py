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

from dataclasses import dataclass, replace
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


@dataclass(frozen=True)
class BlockScaledMatmulCost:
    """E4M3 A[M,K] @ B[N,K].T with FP32 1x128 / 128x128 scales.

    The scope starts after activation quantization and includes any scale layout
    conversion performed by the selected GEMM. Only boundary IO is estimated.
    """

    hbm_policy: str

    def __call__(self, op: OpRecord) -> OpCost:
        try:
            lhs, rhs = (op.inputs[name] for name in ("A", "B"))
            lhs_scale, rhs_scale = (
                op.inputs[name] for name in ("A_scales", "B_scales")
            )
            output = op.outputs["return"]
        except KeyError as exc:
            raise ValueError(f"Missing block-scaled GEMM tensor: {exc}") from exc
        if len(lhs.shape) != 2 or len(rhs.shape) != 2:
            raise ValueError("Block-scaled GEMM requires 2D operands")
        m, k = lhs.shape
        n, rhs_k = rhs.shape
        if (
            min(m, n, k) < 1
            or k != rhs_k
            or k % 128
            or n % 64
            or output.shape != (m, n)
            or lhs_scale.shape != (m, k // 128)
            or rhs_scale.shape != ((n + 127) // 128, k // 128)
        ):
            raise ValueError("Block-scaled GEMM shapes or 128-wide scales disagree")
        if (
            lhs.dtype != "torch.float8_e4m3fn"
            or rhs.dtype != lhs.dtype
            or output.dtype != "torch.bfloat16"
            or lhs_scale.dtype != "torch.float32"
            or rhs_scale.dtype != "torch.float32"
            or any(t.device != lhs.device for t in (rhs, lhs_scale, rhs_scale, output))
        ):
            raise ValueError(
                "Expected colocated E4M3 operands, FP32 scales and BF16 output"
            )
        reads = sum(t.logical_bytes for t in (lhs, rhs, lhs_scale, rhs_scale))
        return OpCost(
            useful_flops=2 * m * n * k,
            executed_flops=None,
            logical_read_bytes=reads,
            logical_write_bytes=output.logical_bytes,
            estimated_hbm_bytes=(
                reads + output.logical_bytes
                if self.hbm_policy == "logical_io_once"
                else None
            ),
            rule="fp8_block_mm.v1",
            assumptions=(
                "useful matmul FLOPs; quantization, scale arithmetic and padding excluded",
                "activation quantization occurs before this GEMM scope",
                "FP8 E4M3 operands select the compute peak; BF16 output does not",
                "boundary IO read/write once; temporary scale layouts and reloads excluded",
                "logical IO does not establish physical HBM traffic",
            ),
            compute_type=(
                "fp8_e4m3_tensor"
                if lhs.device.startswith("cuda:") and not op.autocast_enabled
                else None
            ),
        )


@dataclass(frozen=True)
class FP8TokenGroupQuantCost:
    """FP16/BF16 to E4M3 with one FP32 scale per 128 input elements.

    Counts per-element rescaling only, not amax comparisons or scale formation.
    Canonical scale copies are inside the op, but boundary IO excludes their
    intermediate buffers, just as it excludes GEMM's temporary scale layouts.
    Bound implementations enforce 128-wide groups; output scale geometry is
    checked below without collecting the same group size again.
    """

    hbm_policy: str

    def __call__(self, op: OpRecord) -> OpCost:
        granularity = op.scalar_args.get(
            "granularity", op.static_traits.get("granularity")
        )
        scale_encoding = op.scalar_args.get(
            "scale_encoding", op.static_traits.get("scale_encoding")
        )
        if (
            granularity not in {"token_group", "token_group_128"}
            or scale_encoding != "float32"
        ):
            raise ValueError(
                "FP8 quantization cost requires 128-wide groups and FP32 scales"
            )
        try:
            x = op.inputs["x"]
            values, scales = (op.outputs[f"return[{i}]"] for i in (0, 1))
        except KeyError as exc:
            raise ValueError(f"Missing FP8 quantization tensor: {exc}") from exc
        if (
            len(x.shape) != 2
            or min(x.shape) < 1
            or x.shape[-1] % 128
            or values.shape != x.shape
            or scales.shape != (x.shape[0], x.shape[1] // 128)
            or x.dtype not in {"torch.float16", "torch.bfloat16"}
            or values.dtype != "torch.float8_e4m3fn"
            or scales.dtype != "torch.float32"
            or any(t.device != x.device for t in (values, scales))
        ):
            raise ValueError(
                "FP8 quantization requires matching values and [M,K/128] FP32 scales"
            )
        writes = values.logical_bytes + scales.logical_bytes
        return OpCost(
            useful_flops=prod(x.shape),
            executed_flops=None,
            logical_read_bytes=x.logical_bytes,
            logical_write_bytes=writes,
            estimated_hbm_bytes=(
                x.logical_bytes + writes
                if self.hbm_policy == "logical_io_once"
                else None
            ),
            rule="fp8_token_group_128.v1",
            assumptions=(
                "one FP32 rescale per input element; amax, scale formation and FP8 conversion excluded",
                "boundary input read and FP8 values/FP32 scales write once",
                "temporary scale layout copies, padded scale storage and physical reloads excluded",
                "FP8 output does not select a Tensor Core compute peak",
            ),
            compute_type=(
                "fp32_simt"
                if x.device.startswith("cuda:") and not op.autocast_enabled
                else None
            ),
        )


@dataclass(frozen=True)
class MLANoPEPrologueCost:
    """Native-cache absorbed MLA: assemble query tails and store new latent rows.

    Counts touched slices, never the allocated cache capacity or unchanged
    absorbed query channels. Masked writes, RoPE and quantized caches require
    separate rules; no device mask or slot values are read back here.
    """

    hbm_policy: str

    def __call__(self, op: OpRecord) -> OpCost:
        if any(name.startswith(("rotary.", "expanded.")) for name in op.inputs):
            raise ValueError(
                "MLA NoPE prologue cost excludes RoPE and expanded attention"
            )
        if "cache.write_mask" in op.inputs:
            raise ValueError("MLA prologue masked write count is unavailable")
        try:
            query, q_pe, latent = (
                op.inputs[name] for name in ("query", "q_pe", "latent_cache")
            )
            cache, slots = (
                op.inputs[f"cache.{name}"] for name in ("kv_cache", "slots")
            )
            output = op.outputs["return.query"]
        except KeyError as exc:
            raise ValueError(f"Missing MLA NoPE prologue tensor: {exc}") from exc
        if len(query.shape) != 3 or len(q_pe.shape) != 3 or len(latent.shape) != 2:
            raise ValueError("MLA prologue requires query/q_pe[T,H,D] and latent[T,D]")
        tokens, heads, width = query.shape
        if (
            min(tokens, heads, width) < 1
            or q_pe.shape[:2] != (tokens, heads)
            or not 0 <= q_pe.shape[-1] < width
            or latent.shape != (tokens, width)
            or output.shape != query.shape
            or slots.shape != (tokens,)
            or slots.dtype not in {"torch.int32", "torch.int64"}
            or len(cache.shape) not in (3, 4)
            or cache.shape[-2:] != (1, width)
            or prod(cache.shape[:-2]) < tokens
        ):
            raise ValueError("MLA prologue input, cache and output geometry disagree")
        if (
            query.dtype not in {"torch.float16", "torch.bfloat16"}
            or any(t.dtype != query.dtype for t in (q_pe, latent, output))
            or cache.dtype not in {query.dtype, "torch.bfloat16"}
            or any(
                t.device != query.device for t in (q_pe, latent, output, cache, slots)
            )
        ):
            raise ValueError(
                "MLA NoPE prologue requires colocated native FP16/BF16 tensors"
            )
        cache_row_bytes = cache.logical_bytes // prod(cache.shape[:-2])
        reads = q_pe.logical_bytes + latent.logical_bytes + slots.logical_bytes
        writes = q_pe.logical_bytes + tokens * cache_row_bytes
        return OpCost(
            useful_flops=0,
            executed_flops=None,
            logical_read_bytes=reads,
            logical_write_bytes=writes,
            estimated_hbm_bytes=(
                reads + writes if self.hbm_policy == "logical_io_once" else None
            ),
            rule="mla_nope_prologue.v1",
            assumptions=(
                "NoPE copies query tail channels and stores one latent row per token",
                "absorbed query prefix and previously cached tokens are not accessed",
                "slot indices included; cache capacity and index arithmetic excluded",
                "casts, sanitization comparisons and data movement have no model FLOPs",
                "logical IO estimate assumes one read/write of touched slices, excluding cache effects",
            ),
        )


@dataclass(frozen=True)
class MLACacheCopyCost:
    """Scatter/gather split latent and RoPE rows through an existing cache.

    Both entry points return None. Tensor roles determine reads and writes;
    only the rows selected by loc are accessed, not the whole cache capacity.
    """

    hbm_policy: str
    scatter: bool

    def __call__(self, op: OpRecord) -> OpCost:
        if self.scatter and "write_mask" in op.inputs:
            raise ValueError("MLA cache masked write count is unavailable")
        try:
            cache, slots, nope, rope = (
                op.inputs[name]
                for name in ("kv_buffer", "loc", "cache_k_nope", "cache_k_rope")
            )
        except KeyError as exc:
            raise ValueError(f"Missing MLA cache copy tensor: {exc}") from exc
        if (
            len(slots.shape) != 1
            or slots.dtype not in {"torch.int32", "torch.int64"}
            or slots.stride != (1,)
            or len(cache.shape) not in (3, 4)
            or min(cache.shape) < 1
            or cache.shape[-2] != 1
            or cache.stride[-1] != 1
        ):
            raise ValueError("MLA copy requires slot indices and native cache rows")
        rows = slots.shape[0]
        if (
            any(
                len(t.shape) not in (2, 3)
                or t.shape[0] != rows
                or prod(t.shape[1:-1]) != 1
                or t.stride[-1] != 1
                for t in (nope, rope)
            )
            or cache.shape[-1] != nope.shape[-1] + rope.shape[-1]
        ):
            raise ValueError("MLA cache copy input/output row geometry disagrees")
        if any(t.device != cache.device for t in (slots, nope, rope)):
            raise ValueError("MLA cache copy requires colocated tensors")
        cache_bytes = rows * (cache.logical_bytes // prod(cache.shape[:-2]))
        split_bytes = nope.logical_bytes + rope.logical_bytes
        reads, writes = (
            (split_bytes, cache_bytes) if self.scatter else (cache_bytes, split_bytes)
        )
        reads += slots.logical_bytes
        return OpCost(
            useful_flops=0,
            executed_flops=None,
            logical_read_bytes=reads,
            logical_write_bytes=writes,
            estimated_hbm_bytes=(
                reads + writes if self.hbm_policy == "logical_io_once" else None
            ),
            rule=f"mla_cache_{'scatter' if self.scatter else 'gather'}.v1",
            assumptions=(
                "one combined cache row per slot lookup, counting each tensor's storage dtype",
                "destination arguments are writes only; untouched rows and page gaps excluded",
                "slot indices included; repeated slots counted per access without reading their values",
                "casts, sanitization comparisons and index arithmetic have no model FLOPs",
                "logical IO estimate excludes physical reloads and cache effects",
            ),
        )


@dataclass(frozen=True)
class _FlashMLACost:
    """Adapt FlashMLA metadata to the common attention matmul count.

    QK and PV share latent cache storage: the value is a prefix of the key.
    The logical IO estimate reads each request's visible cache rows once,
    shared across query heads and positions; split/combine workspace is excluded.
    """

    hbm_policy: str

    def __call__(self, op: OpRecord) -> OpCost:
        if op.scalar_args.get("is_fp8_kvcache") is not False or any(
            name in op.inputs
            for name in (
                "indices",
                "attn_sink",
                "extra_k_cache",
                "extra_indices_in_kvcache",
                "topk_length",
                "extra_topk_length",
            )
        ):
            raise ValueError(
                "Paged MLA cost supports dense, native-cache attention only"
            )
        lengths = op.tensor_values.get("cache_seqlens")
        if lengths is None:
            raise ValueError(
                "cache_seqlens snapshot unavailable; CUDA lengths require NvtxTimeline and a small contiguous integer vector"
            )
        causal = op.scalar_args.get("causal")
        value_dim = op.scalar_args.get("head_dim_v")
        if type(causal) is not bool or type(value_dim) is not int or value_dim < 1:
            raise ValueError("Paged MLA requires causal and head_dim_v metadata")
        try:
            q, cache, table, seq = (
                op.inputs[name]
                for name in ("q", "k_cache", "block_table", "cache_seqlens")
            )
            output, lse = (op.outputs[f"return[{i}]"] for i in (0, 1))
        except KeyError as exc:
            raise ValueError(f"Missing paged MLA tensor: {exc}") from exc
        if len(q.shape) != 4 or len(cache.shape) != 4 or len(table.shape) != 2:
            raise ValueError(
                "Paged MLA requires q[B,Q,H,D], cache[P,S,1,D] and block_table[B,P]"
            )
        batch, query_length, heads, key_dim = q.shape
        pages, page_size, kv_heads, cache_dim = cache.shape
        if (
            min(*q.shape, *cache.shape) < 1
            or kv_heads != 1
            or cache_dim != key_dim
            or value_dim > key_dim
            or table.shape[0] != batch
            or seq.shape != (batch,)
            or output.shape != (batch, query_length, heads, value_dim)
            or lse.shape != (batch, heads, query_length)
            or len(lengths) != batch
        ):
            raise ValueError("Paged MLA shapes or visible lengths disagree")
        if (
            q.dtype not in {"torch.float16", "torch.bfloat16"}
            or cache.dtype != q.dtype
            or output.dtype != q.dtype
            or lse.dtype != "torch.float32"
            or table.dtype not in {"torch.int32", "torch.int64"}
            or seq.dtype not in {"torch.int32", "torch.int64"}
            or any(t.device != q.device for t in (cache, table, seq, output, lse))
        ):
            raise ValueError(
                "Paged MLA requires colocated matching native tensors and integer metadata"
            )
        flops = cache_rows = used_pages = 0
        mask = "causal_lower_right" if causal else "full"
        for length in lengths:
            if type(length) is not int or not 0 <= length <= page_size * table.shape[1]:
                raise ValueError("Paged MLA shapes or visible lengths disagree")
            flops += AttentionMatmulCost._matmul_flops(
                query_heads=heads,
                query_length=query_length,
                kv_length=length,
                qk_head_dim=key_dim,
                value_head_dim=value_dim,
                mask=mask,
            )
            cache_rows += length
            used_pages += (length + page_size - 1) // page_size
        cache_bytes = cache_rows * cache.logical_bytes // (pages * page_size)
        table_bytes = used_pages * (4 if table.dtype == "torch.int32" else 8)
        reads = q.logical_bytes + cache_bytes + table_bytes + seq.logical_bytes
        writes = output.logical_bytes + lse.logical_bytes
        return OpCost(
            useful_flops=flops,
            executed_flops=None,
            logical_read_bytes=reads,
            logical_write_bytes=writes,
            estimated_hbm_bytes=(
                reads + writes if self.hbm_policy == "logical_io_once" else None
            ),
            rule="flashmla_decode.v1",
            assumptions=(
                "useful QK/PV matmuls with lower-right causal alignment and actual visible lengths",
                "single-token decode uses every visible KV row, without causal halving",
                "softmax, scaling, split/combine arithmetic and padding excluded from useful FLOPs",
                "each request's visible latent rows read once, shared by QK/PV and all query heads/positions",
                "HBM estimate excludes split/combine workspace, tile reloads and cross-request cache reuse",
                "cache/table allocation capacity is not traffic; only referenced logical rows/pages counted",
            ),
            compute_type=_infer_matmul_compute_type(op, q, cache),
        )


@dataclass(frozen=True)
class _VarlenAttentionCost:
    """Adapt packed Q/K/V and cumulative lengths to the shared QK/PV count."""

    hbm_policy: str

    def __call__(self, op: OpRecord) -> OpCost:
        if (
            op.scalar_args.get("window_size") != (-1, -1)
            or op.scalar_args.get("attention_chunk") != 0
            or any(
                name in op.inputs
                for name in (
                    "seqused_q",
                    "seqused_k",
                    "qv",
                    "q_descale",
                    "k_descale",
                    "v_descale",
                    "sinks",
                )
            )
        ):
            raise ValueError(
                "Variable-length cost supports full or causal native attention only"
            )
        causal = op.scalar_args.get("causal")
        if type(causal) is not bool:
            raise ValueError("Variable-length attention requires causal metadata")
        try:
            q, k, v = (op.inputs[name] for name in ("q", "k", "v"))
            output = (
                op.outputs["return"]
                if "return" in op.outputs
                else op.outputs["return[0]"]
            )
        except KeyError as exc:
            raise ValueError(
                f"Missing variable-length attention tensor: {exc}"
            ) from exc
        if (
            any(len(t.shape) != 3 or min(t.shape) < 1 for t in (q, k, v))
            or k.shape[:2] != v.shape[:2]
            or q.shape[-1] != k.shape[-1]
            or q.shape[1] % k.shape[1]
            or output.shape != (*q.shape[:2], v.shape[-1])
            or q.dtype not in {"torch.float16", "torch.bfloat16"}
            or any(t.dtype != q.dtype or t.device != q.device for t in (k, v, output))
        ):
            raise ValueError(
                "Variable-length attention requires matching native packed Q/K/V and output"
            )
        offsets = []
        metadata_bytes = 0
        for name, total in (("cu_seqlens_q", q.shape[0]), ("cu_seqlens_k", k.shape[0])):
            values = op.tensor_values.get(name)
            if values is None:
                raise ValueError(
                    f"{name} snapshot unavailable; CUDA lengths require NvtxTimeline"
                )
            description = op.inputs.get(name)
            if (
                description is None
                or description.shape != (len(values),)
                or description.dtype not in {"torch.int32", "torch.int64"}
                or len(values) < 2
                or any(type(value) is not int for value in values)
                or values[0] != 0
                or values[-1] != total
                or any(left > right for left, right in zip(values, values[1:]))
            ):
                raise ValueError(
                    "Cumulative attention lengths do not partition the packed tensors"
                )
            offsets.append(values)
            metadata_bytes += description.logical_bytes
        if len(offsets[0]) != len(offsets[1]):
            raise ValueError(
                "Query and KV cumulative lengths have different request counts"
            )
        flops = sum(
            AttentionMatmulCost._matmul_flops(
                query_heads=q.shape[1],
                query_length=qe - qs,
                kv_length=ke - ks,
                qk_head_dim=q.shape[2],
                value_head_dim=v.shape[2],
                mask="causal_lower_right" if causal else "full",
            )
            for qs, qe, ks, ke in zip(
                offsets[0], offsets[0][1:], offsets[1], offsets[1][1:]
            )
        )
        reads = q.logical_bytes + k.logical_bytes + v.logical_bytes + metadata_bytes
        writes = sum(t.logical_bytes for t in op.outputs.values())
        return OpCost(
            useful_flops=flops,
            executed_flops=None,
            logical_read_bytes=reads,
            logical_write_bytes=writes,
            estimated_hbm_bytes=(
                reads + writes if self.hbm_policy == "logical_io_once" else None
            ),
            rule="attention_varlen.v1",
            assumptions=(
                "packed native Q/K/V with exact per-request lengths and lower-right causal alignment",
                "useful QK/PV work only; softmax, scaling, softcap and padding excluded",
                "logical boundary IO counted once, including cumulative-length arguments and returned LSE",
                "HBM estimate excludes tiling reloads, intermediate workspace and cache effects",
            ),
            compute_type=_infer_matmul_compute_type(op, q, k, v),
        )


@dataclass(frozen=True)
class AttentionStateMergeCost:
    """Merge two partial attention outputs and their FP32 log-sum-exp values.

    Counts the weighted value sum with normalized per-row weights: 3*N useful
    FLOPs. Weight formation and LSE arithmetic are excluded, including the
    Triton implementation's per-element normalization division.
    """

    hbm_policy: str

    def __call__(self, op: OpRecord) -> OpCost:
        try:
            out_a, lse_a, out_b, lse_b = (
                op.inputs[name] for name in ("out_a", "lse_a", "out_b", "lse_b")
            )
            output, lse = (op.outputs[f"return[{i}]"] for i in (0, 1))
        except KeyError as exc:
            raise ValueError(f"Missing attention merge tensor: {exc}") from exc
        if (
            len(out_a.shape) != 3
            or min(out_a.shape) < 1
            or any(t.shape != out_a.shape for t in (out_b, output))
            or any(t.shape != out_a.shape[:2] for t in (lse_a, lse_b, lse))
        ):
            raise ValueError("Attention merge requires values[T,H,D] and LSE[T,H]")
        if (
            out_a.dtype not in {"torch.float16", "torch.bfloat16"}
            or any(t.dtype != out_a.dtype for t in (out_b, output))
            or any(t.dtype != "torch.float32" for t in (lse_a, lse_b, lse))
            or any(t.device != out_a.device for t in (out_b, output, lse_a, lse_b, lse))
        ):
            raise ValueError(
                "Attention merge requires colocated FP16/BF16 values and FP32 LSE"
            )
        reads = sum(t.logical_bytes for t in (out_a, lse_a, out_b, lse_b))
        writes = output.logical_bytes + lse.logical_bytes
        return OpCost(
            useful_flops=3 * prod(output.shape),
            executed_flops=None,
            logical_read_bytes=reads,
            logical_write_bytes=writes,
            estimated_hbm_bytes=(
                reads + writes if self.hbm_policy == "logical_io_once" else None
            ),
            rule="attention_merge_state.v1",
            assumptions=(
                "two multiplies and one add per output value with normalized row weights",
                "weight formation, normalization division, exp/log and LSE arithmetic excluded",
                "both partial values and LSEs read once; merged values and LSE written once",
                "in-place output is a write to existing inputs, not an additional read",
                "FP32 weighted value arithmetic selects the FP32 SIMT reference peak",
                "logical IO estimate excludes physical reloads and cache effects",
            ),
            compute_type=(
                "fp32_simt"
                if out_a.device.startswith("cuda:") and not op.autocast_enabled
                else None
            ),
        )


@dataclass(frozen=True)
class RMSNormCost:
    """Weighted RMSNorm with FP32 arithmetic, optionally fusing two row norms.

    The parallel implementation returns None; its output arguments describe
    writes, including when an output aliases its corresponding input.
    """

    hbm_policy: str
    parallel: bool

    def __call__(self, op: OpRecord) -> OpCost:
        try:
            tensors = (
                [
                    tuple(
                        op.inputs[f"{role}{i}"]
                        for role in ("input", "weight", "output")
                    )
                    for i in (1, 2)
                ]
                if self.parallel
                else [(op.inputs["input"], op.inputs["weight"], op.outputs["return"])]
            )
        except KeyError as exc:
            raise ValueError(f"Missing RMSNorm tensor: {exc}") from exc
        first = tensors[0][0]
        elements = reads = writes = 0
        for x, weight, output in tensors:
            if (
                len(x.shape) not in ((2,) if self.parallel else (2, 3))
                or not all(dim > 0 for dim in x.shape)
                or weight.shape != (x.shape[-1],)
                or output.shape != x.shape
                or x.shape[:-1] != first.shape[:-1]
            ):
                raise ValueError(
                    "RMSNorm requires matching rows, output and last-dimension weight"
                )
            if first.dtype not in {"torch.float16", "torch.bfloat16"} or any(
                t.dtype != first.dtype or t.device != first.device
                for t in (x, weight, output)
            ):
                raise ValueError(
                    "RMSNorm requires colocated matching FP16/BF16 tensors"
                )
            elements += prod(x.shape)
            reads += x.logical_bytes + weight.logical_bytes
            writes += output.logical_bytes
        return OpCost(
            useful_flops=4 * elements,
            executed_flops=None,
            logical_read_bytes=reads,
            logical_write_bytes=writes,
            estimated_hbm_bytes=(
                reads + writes if self.hbm_policy == "logical_io_once" else None
            ),
            rule="rmsnorm_fused_parallel.v1" if self.parallel else "rmsnorm.v1",
            assumptions=(
                "4*N nominal RMSNorm FLOPs: square, sum, normalize and weight multiply",
                "per-row scalars, rsqrt, dtype conversions and reduction padding excluded",
                "FP32 arithmetic selects the FP32 SIMT reference peak, not the storage dtype peak",
                "one input/weight read and output write; in-place output is not an extra input",
                "logical IO estimate excludes physical reloads and cache effects",
            ),
            compute_type=(
                "fp32_simt"
                if first.device.startswith("cuda:") and not op.autocast_enabled
                else None
            ),
        )


@dataclass(frozen=True)
class CausalConv1dPrefillCost:
    """Depthwise convolution over packed x[channels, tokens].

    Channels include every packed Q/K/V projection. The state estimate counts
    one history row per sequence, never the capacity of the allocated pool.
    Device masks are not read back: initialized, nonempty sequences are assumed
    for the IO estimate; first chunks and padded slots can overestimate traffic.
    The Triton implementation accumulates in FP32. MFU uses the FP32 SIMT FMA
    reference peak, including when half-precision products are rounded before
    accumulation; it does not measure the mixed instruction pipeline's occupancy.
    """

    hbm_policy: str

    def __call__(self, op: OpRecord) -> OpCost:
        try:
            x, weight = op.inputs["x"], op.inputs["weight"]
            boundaries = op.inputs["query_start_loc"]
            output = op.outputs["return"]
        except KeyError as exc:
            raise ValueError(f"Missing causal convolution tensor: {exc}") from exc
        if len(x.shape) != 2 or len(weight.shape) != 2:
            raise ValueError("Causal convolution requires x[C,T] and weight[C,W]")
        channels, tokens = x.shape
        weight_channels, width = weight.shape
        if channels != weight_channels or width < 1 or output.shape != x.shape:
            raise ValueError(
                "Causal convolution input, weight and output shapes differ"
            )
        if len(boundaries.shape) != 1 or boundaries.shape[0] < 1:
            raise ValueError("Causal convolution requires sequence boundaries")

        state_bytes = 0
        state = op.inputs.get("conv_states")
        if state is not None:
            if (
                len(state.shape) != 3
                or state.shape[1] != channels
                or state.shape[2] < width - 1
                or not prod(state.shape)
            ):
                raise ValueError("Causal convolution state shape does not match")
            element_bytes = state.logical_bytes // prod(state.shape)
            sequences = boundaries.shape[0] - 1
            state_bytes = sequences * channels * (width - 1) * element_bytes
        read_bytes = x.logical_bytes + weight.logical_bytes + state_bytes
        if "bias" in op.inputs:
            read_bytes += op.inputs["bias"].logical_bytes
        write_bytes = output.logical_bytes + state_bytes
        scalar_dtypes = {"torch.float16", "torch.bfloat16", "torch.float32"}
        compute_type = (
            "fp32_simt"
            if x.device.startswith("cuda:")
            and weight.device == x.device
            and x.dtype in scalar_dtypes
            and weight.dtype in scalar_dtypes
            and not op.autocast_enabled
            else None
        )
        return OpCost(
            useful_flops=2 * tokens * channels * width,
            executed_flops=None,
            logical_read_bytes=read_bytes,
            logical_write_bytes=write_bytes,
            estimated_hbm_bytes=(
                read_bytes + write_bytes
                if self.hbm_policy == "logical_io_once"
                else None
            ),
            rule="causal_conv1d_prefill.v1",
            assumptions=(
                "nominal depthwise convolution FLOPs; bias and SiLU excluded",
                "packed channels include all Q/K/V projections",
                "one input/weight read, output write and history read/write per sequence",
                "history estimate assumes initialized nonempty sequences; device masks unknown",
                "cache reuse, tile reloads and indexing metadata excluded from HBM estimate",
                "FP32 accumulation normalized to the FP32 SIMT FMA reference peak",
                "FP16/BF16 products may round before accumulation; MFU is not instruction-pipeline utilization",
            ),
            compute_type=compute_type,
        )


@dataclass(frozen=True)
class GatedDeltaPrefillCost:
    """Packed gated-delta scan with a value-major [N,H,V,K] state.

    Counts the equivalent recurrence, not the executed chunk factorization:
    state decay, state/key product, value correction, rank-one update and readout.
    Normalization, gate activations and input scaling are excluded. Mixed-precision
    chunk kernels do not establish one compute peak from Q/K/V storage dtypes.
    """

    hbm_policy: str

    def __call__(self, op: OpRecord) -> OpCost:
        try:
            q, k, v = (op.inputs[name] for name in ("q", "k", "v"))
            state = op.inputs["initial_state"]
            output, final_state = (op.outputs[f"return[{i}]"] for i in (0, 1))
        except KeyError as exc:
            raise ValueError(f"Missing gated-delta tensor: {exc}") from exc
        if len(q.shape) != 3 or k.shape != q.shape or len(v.shape) != 3:
            raise ValueError("Gated-delta scan requires packed q/k/v[T,H,D]")
        tokens, heads, key_dim = q.shape
        if v.shape[:2] != (tokens, heads) or output.shape != v.shape:
            raise ValueError("Gated-delta value/output shapes do not match")
        value_dim = v.shape[2]
        if (
            len(state.shape) != 4
            or state.shape[1:] != (heads, value_dim, key_dim)
            or final_state.shape != state.shape
        ):
            raise ValueError("Gated-delta scan requires value-major state[N,H,V,K]")
        reads = sum(
            tensor.logical_bytes
            for name, tensor in op.inputs.items()
            if name
            in {"q", "k", "v", "g", "beta_logits", "A_log", "dt_bias", "initial_state"}
        )
        writes = output.logical_bytes + final_state.logical_bytes
        return OpCost(
            useful_flops=tokens * heads * (7 * key_dim * value_dim + value_dim),
            executed_flops=None,
            logical_read_bytes=reads,
            logical_write_bytes=writes,
            estimated_hbm_bytes=(
                reads + writes if self.hbm_policy == "logical_io_once" else None
            ),
            rule="gated_delta_prefill.v1",
            assumptions=(
                "equivalent recurrence FLOPs; chunk factorization and recomputation excluded",
                "per token/head: K*V decay, 2*K*V prediction, V subtraction, 2*K*V update, 2*K*V readout",
                "normalization, nonlinear gates and featurewise beta scaling excluded",
                "one boundary input read and output write; intermediate chunk buffers excluded",
                "mixed-precision implementation; storage dtype does not select one compute peak",
            ),
        )


def register_compute_costs(registry: CostRegistry, *, hbm_policy: str) -> None:
    """Register costs for opted-in compute, quantization and MLA cache writes.

    Args:
        registry: Cost registry receiving the named rules.
        hbm_policy: 'unavailable' or the rough estimate 'logical_io_once'.

    Compute types are inferred from matching BF16/FP16 operands, or from the
    explicit E4M3 block-scaled contract. Unsupported dtypes and autocast leave
    MFU unavailable; output dtype does not select a peak.
    Projection rules count matrix products only; bias and conversion are excluded.
    RMSNorm uses nominal FP32 arithmetic counts, not its storage precision peak.
    Registers the rules in place and returns None.
    """
    if hbm_policy not in {"unavailable", "logical_io_once"}:
        raise ValueError(f"Unsupported HBM policy: {hbm_policy}")
    for name, ndim in (("torch_mm.v1", 2), ("torch_bmm.v1", 3)):
        registry.register(
            name,
            _DenseProjectionCost(ndim=ndim, hbm_policy=hbm_policy),
        )
    registry.register("fp8_block_mm.v1", BlockScaledMatmulCost(hbm_policy=hbm_policy))
    registry.register("fp8_token_group_128.v1", FP8TokenGroupQuantCost(hbm_policy))
    registry.register("mla_nope_prologue.v1", MLANoPEPrologueCost(hbm_policy))
    for mode, scatter in (("scatter", True), ("gather", False)):
        registry.register(
            f"mla_cache_{mode}.v1", MLACacheCopyCost(hbm_policy, scatter=scatter)
        )
    registry.register("flashmla_decode.v1", _FlashMLACost(hbm_policy))
    registry.register("attention_varlen.v1", _VarlenAttentionCost(hbm_policy))
    registry.register("attention_merge_state.v1", AttentionStateMergeCost(hbm_policy))
    for name, parallel in (("rmsnorm.v1", False), ("rmsnorm_fused_parallel.v1", True)):
        registry.register(name, RMSNormCost(hbm_policy=hbm_policy, parallel=parallel))
    registry.register(
        "causal_conv1d_prefill.v1", CausalConv1dPrefillCost(hbm_policy=hbm_policy)
    )
    registry.register(
        "gated_delta_prefill.v1", GatedDeltaPrefillCost(hbm_policy=hbm_policy)
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

        return self._from_bytes(tensor.logical_bytes)

    def _from_bytes(self, total_bytes: int) -> OpCost:
        """Shared boundary accounting, including a grouped tuple of tensors."""
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


def _collective_group(op: OpRecord) -> tuple[tuple[int, ...], str]:
    ranks = op.scalar_args.get("ranks")
    if (
        not isinstance(ranks, tuple)
        or not ranks
        or any(type(r) is not int for r in ranks)
    ):
        raise ValueError("AllReduce requires the participating group ranks")
    if len(set(ranks)) != len(ranks):
        raise ValueError("AllReduce group ranks must be unique")
    name = op.scalar_args.get("group_name")
    return ranks, f"nccl:{name!r}:{ranks}"


@dataclass(frozen=True)
class _NcclAllReduceCost:
    hbm_policy: str

    def __call__(self, op: OpRecord) -> OpCost:
        ranks, group = _collective_group(op)
        if op.scalar_args.get("is_sum") is not True:
            raise ValueError("AllReduce cost currently models SUM only")
        tensors = [
            tensor
            for name, tensor in op.inputs.items()
            if name in {"tensor", "first", "second"} or name.startswith("tensor[")
        ]
        if not tensors:
            raise ValueError("Missing AllReduce input tensor")
        # Tuple and grouped-pair entries run their reductions in one outer scope.
        total_bytes = sum(t.logical_bytes for t in tensors) if len(ranks) > 1 else 0
        cost = CollectiveCost(
            "all_reduce_sum", "tensor", "input", len(ranks), group
        )._from_bytes(total_bytes)
        if self.hbm_policy == "logical_io_once":
            cost = replace(
                cost,
                estimated_hbm_bytes=cost.logical_read_bytes + cost.logical_write_bytes,
                assumptions=cost.assumptions
                + (
                    "HBM estimate counts one input read and output write; protocol buffers and reloads excluded",
                ),
            )
        return cost


@dataclass(frozen=True)
class AllReduceRMSNormCost:
    """TRT-LLM SUM + residual + RMSNorm, or its unfused fallback.

    Both IPC and MNNVL normalize the full result on every rank, including
    two-shot: only the collective reduction is sharded. Quantization and
    reduce-scattered residual variants require separate cost rules.
    """

    hbm_policy: str

    def __call__(self, op: OpRecord) -> OpCost:
        ranks, group = _collective_group(op)
        fused = op.scalar_args.get("fused")
        if type(fused) is not bool:
            raise ValueError("AllReduce RMSNorm requires the actual fusion decision")
        if op.scalar_args.get("block_quant_fp8", False):
            raise ValueError("Quantized AllReduce RMSNorm cost is not modeled")
        if op.scalar_args.get("residual_reduce_scattered", False):
            raise ValueError("Reduce-scattered residual RMSNorm cost is not modeled")
        try:
            x, residual, weight = (
                op.inputs[name] for name in ("input_tensor", "residual", "weight")
            )
            output, residual_out = (op.outputs[f"return[{i}]"] for i in (0, 1))
        except KeyError as exc:
            raise ValueError(f"Missing AllReduce RMSNorm tensor: {exc}") from exc
        if (
            len(x.shape) != 2
            or min(x.shape) < 1
            or any(t.shape != x.shape for t in (residual, output, residual_out))
            or weight.shape != (x.shape[1],)
            or any(
                t.dtype != x.dtype or t.device != x.device
                for t in (residual, weight, output, residual_out)
            )
        ):
            raise ValueError(
                "AllReduce RMSNorm requires matching [T,H] tensors and weight[H]"
            )
        reads = x.logical_bytes + residual.logical_bytes + weight.logical_bytes
        writes = output.logical_bytes + residual_out.logical_bytes
        if op.scalar_args.get("has_partial_norm_out", False):
            partial = op.outputs.get("return[3]")
            rank = op.scalar_args.get("rank")
            if type(rank) is not int or not 0 <= rank < len(ranks):
                raise ValueError("Partial RMSNorm output requires the group-local rank")
            rows = x.shape[0] // len(ranks) + int(rank < x.shape[0] % len(ranks))
            if (
                partial is None
                or partial.shape != (rows, x.shape[1])
                or partial.dtype != x.dtype
                or partial.device != x.device
            ):
                raise ValueError("Partial RMSNorm output shape or dtype does not match")
            # This fused pattern writes only partial_norm_out, not the allocated
            # norm_out buffer. The fallback instead returns a view of norm_out.
            if fused:
                writes = residual_out.logical_bytes + partial.logical_bytes
        assumptions = (
            "full RMSNorm epilogue on every rank for both one-shot and two-shot",
            "5*T*H useful FLOPs: residual add plus RMSNorm; per-row scalars and rsqrt excluded",
            "collective reduction arithmetic excluded from model useful FLOPs",
            "normalized AllReduce bus bytes use the original input dtype and full tensor size",
            "bandwidth and MFU divide by the whole scope window, not a communication-only time",
            "logical IO counts boundary tensors once; protocol buffers, cache reuse and reloads excluded",
            "FP32 RMSNorm arithmetic uses the FP32 SIMT reference peak; residual rounding may differ",
        )
        if not fused:
            assumptions += (
                "unfused NCCL and PyTorch epilogue; intermediate HBM traffic is not modeled",
            )
        return OpCost(
            useful_flops=5 * prod(x.shape),
            executed_flops=None,
            logical_read_bytes=reads,
            logical_write_bytes=writes,
            estimated_hbm_bytes=(
                reads + writes
                if fused and self.hbm_policy == "logical_io_once"
                else None
            ),
            rule=(
                "all_reduce_residual_rmsnorm.v1"
                if fused
                else "all_reduce_residual_rmsnorm_unfused.v1"
            ),
            assumptions=assumptions,
            compute_type=(
                "fp32_simt"
                if x.device.startswith("cuda:")
                and x.dtype in {"torch.float16", "torch.bfloat16", "torch.float32"}
                and not op.autocast_enabled
                else None
            ),
            communication=CollectiveTraffic(
                "all_reduce_sum", x.logical_bytes, len(ranks), group
            ),
        )


def register_collective_costs(registry: CostRegistry, *, hbm_policy: str) -> None:
    """Register SUM and residual/RMSNorm costs using recorded process-group ranks."""
    if hbm_policy not in {"unavailable", "logical_io_once"}:
        raise ValueError(f"Unsupported HBM policy: {hbm_policy}")
    registry.register("nccl_all_reduce_sum.v1", _NcclAllReduceCost(hbm_policy))
    registry.register(
        "all_reduce_residual_rmsnorm.v1", AllReduceRMSNormCost(hbm_policy)
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

        flops = batch_size * self._matmul_flops(
            query_heads=query_heads,
            query_length=query_length,
            kv_length=kv_length,
            qk_head_dim=qk_head_dim,
            value_head_dim=value_head_dim,
            mask=self.mask,
        )
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

    @staticmethod
    def _matmul_flops(
        *,
        query_heads: int,
        query_length: int,
        kv_length: int,
        qk_head_dim: int,
        value_head_dim: int,
        mask: str,
    ) -> int:
        """Count one request's QK/PV work for explicit tensors or a cache view."""
        triangle = min(query_length, kv_length)
        if mask == "full":
            pairs = query_length * kv_length
        elif mask == "causal_upper_left":
            # Row i sees min(i + 1, kv_length) keys.
            pairs = (
                triangle * (triangle + 1) // 2 + (query_length - triangle) * kv_length
            )
        elif mask == "causal_lower_right":
            # Row i sees max(0, kv_length - query_length + i + 1) keys.
            pairs = triangle * (2 * kv_length - triangle + 1) // 2
        else:
            raise ValueError(f"Unsupported attention mask: {mask}")
        return 2 * query_heads * pairs * (qk_head_dim + value_head_dim)
