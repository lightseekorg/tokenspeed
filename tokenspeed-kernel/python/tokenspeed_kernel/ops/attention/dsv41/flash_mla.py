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

"""Optional native V4.1 selected attention over the shared cache contract."""

from __future__ import annotations

from types import ModuleType

import torch
import torch.nn.functional as F
from tokenspeed_kernel._optional import has_optional_package
from tokenspeed_kernel.ops.attention.dsv41.triton import (
    _PAGED_FORMATS,
    _copy_attention_heads,
)
from tokenspeed_kernel.platform import (
    ArchVersion,
    CapabilityRequirement,
    current_platform,
)
from tokenspeed_kernel.registry import Priority, register_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature

_FLASH_MLA_API: ModuleType | None = None
if current_platform().is_hopper_plus and has_optional_package("flash_mla"):
    import flash_mla.flash_mla_interface as _FLASH_MLA_API

_FUSED_DECODE_CAPABILITY = CapabilityRequirement(
    min_arch_version=ArchVersion(10, 0),
    max_arch_version=ArchVersion(10, 3),
    vendors=frozenset({"nvidia"}),
)
_FUSED_FLASH_MLA_API: ModuleType | None = None
if _FUSED_DECODE_CAPABILITY.satisfied_by(current_platform()):
    try:
        from flash_mla import fused_norm_rope_attn_rope_cast as _FUSED_FLASH_MLA_API
    except ImportError:
        pass


def flash_mla_api() -> ModuleType:
    """Return the optional native V4.1 FlashMLA API."""
    if _FLASH_MLA_API is None:
        raise ImportError(
            "FlashMLA V4.1 requires tokenspeed-flashmla with its native CUDA extension"
        )
    return _FLASH_MLA_API


def is_flash_mla_v41_available() -> bool:
    """Return whether the optional V4.1 API and native extension are available."""
    return _FLASH_MLA_API is not None


def is_flash_mla_fused_decode_available() -> bool:
    """Return whether the native fused decode API is available on this target."""
    return _FUSED_FLASH_MLA_API is not None


def new_flashmla_schedule() -> object:
    """Create caller-owned metadata; capture must record its first preparation."""
    return flash_mla_api().get_mla_metadata()[0]


def _native_query(q, sink):
    if q.ndim != 3 or q.shape[-1] != 512 or q.dtype != torch.bfloat16:
        raise ValueError("V4.1 query must be BF16 [tokens, heads, 512]")
    heads = q.shape[1]
    if not 1 <= heads <= 128:
        raise ValueError("FlashMLA V4.1 requires 1..128 query heads")
    padded = 64 if heads <= 64 else 128
    if sink is None:
        return F.pad(q, (0, 0, 0, padded - heads)).contiguous(), None
    if (
        sink.shape not in ((heads,), (padded,))
        or sink.dtype != torch.float32
        or sink.device != q.device
    ):
        raise ValueError(
            "attention sink must be FP32 [query heads] on the query device"
        )
    if padded == heads:
        return q.contiguous(), sink.contiguous()
    q_native = torch.empty((q.shape[0], padded, 512), dtype=q.dtype, device=q.device)
    sink_native = (
        sink.contiguous()
        if sink.numel() == padded
        else torch.empty(padded, dtype=sink.dtype, device=sink.device)
    )
    _copy_attention_heads(
        q_native,
        q,
        sink if sink.numel() != padded else None,
        sink_native if sink.numel() != padded else None,
    )
    return q_native, sink_native


def _paged(cache, role, stride_alignment):
    # FlashMLA detects the layout from the row width; the widths a role may
    # carry are the codec's. Which of them a build reads is an arch property:
    # every target reads V4 (584), sm100 and above add V4.1 (528 and 288).
    widths = _PAGED_FORMATS[role]
    if cache.ndim != 3 or cache.shape[2] not in widths:
        raise ValueError(
            f"FlashMLA {role} cache rows must be one of {sorted(widths)} bytes"
        )
    row_bytes = cache.shape[2]
    if (
        cache.dtype != torch.uint8
        or cache.shape[1] != 64
        or cache.stride(1) != row_bytes
        or cache.stride(2) != 1
    ):
        raise ValueError(
            "FlashMLA requires contiguous bytes within each page-planar field"
        )
    if cache.stride(0) < 64 * row_bytes or cache.stride(0) % stride_alignment:
        raise ValueError("FlashMLA page stride has incompatible alignment")
    if cache.data_ptr() % 16:
        raise ValueError("FlashMLA cache base must be 16-byte aligned")
    return cache.as_strided(
        (cache.shape[0], 64, 1, row_bytes), (cache.stride(0), row_bytes, row_bytes, 1)
    )


def _slots(slots, lengths, tokens):
    if slots is None or lengths is None:
        raise ValueError("paged attention requires slots and lengths")
    if (
        slots.ndim != 2
        or slots.shape[0] != tokens
        or slots.dtype not in (torch.int32, torch.int64)
    ):
        raise ValueError("paged slots must be integer [tokens, capacity]")
    if lengths.shape != (tokens,) or lengths.dtype not in (torch.int32, torch.int64):
        raise ValueError("paged lengths must be integer [tokens]")
    width = max(64, (slots.shape[1] + 63) // 64 * 64)
    slots = slots.to(torch.int32)
    if width != slots.shape[1]:
        slots = F.pad(slots, (0, width - slots.shape[1]), value=-1)
    return slots.contiguous().unsqueeze(1), lengths.to(torch.int32).contiguous()


@register_kernel(
    "attention",
    "dsv41_prepare_fused_decode_weights",
    name="flashmla_prepare_fused_decode_weights",
    solution="flashmla",
    capability=_FUSED_DECODE_CAPABILITY,
    signatures=frozenset(
        {format_signature(x=dense_tensor_format(torch.float8_e4m3fn))}
    ),
    priority=Priority.SPECIALIZED,
)
def flashmla_prepare_fused_decode_weights(
    q_weight: torch.Tensor,
    q_scales: torch.Tensor,
    wo_a_weight: torch.Tensor,
    wo_a_scales: torch.Tensor,
    num_heads: int,
    num_groups: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
    """Prepare load-time projection weights for fused decode without mutating inputs.

    Q weights are E4M3 [heads * 512, K] with canonical uint8 UE8M0 [heads *
    512, K/32] scales. O weights are E4M3 [groups * N, heads/groups * 512]
    with packed int32 scales from prepare_grouped_fp8_projection.

    Return the permuted Q codes and canonical scales, then the permuted O
    codes [groups, N, heads/groups * 512] and their packed int32 scales.
    """
    from tokenspeed_kernel.ops.gemm import prepare_fp8_projection

    if _FUSED_FLASH_MLA_API is None:
        raise ImportError("Fused decode requires the native FlashMLA fused API")
    # The native output epilogue fixes WV_GROUP_SIZE to eight query heads.
    if num_heads not in (64, 128) or num_heads != 8 * num_groups:
        raise ValueError(
            "Fused decode requires 64 or 128 heads with 8 heads per output group"
        )
    if (
        q_weight.ndim != 2
        or q_weight.shape[0] != num_heads * 512
        or q_weight.shape[1] == 0
        or q_weight.shape[1] % 128
    ):
        raise ValueError("Fused decode Q weight must be [heads * 512, K], K % 128 == 0")
    group_width = num_heads // num_groups * 512
    if (
        wo_a_weight.ndim != 2
        or wo_a_weight.shape[0] == 0
        or wo_a_weight.shape[0] % (num_groups * 64)
        or wo_a_weight.shape[1] != group_width
        or wo_a_weight.dtype != torch.float8_e4m3fn
    ):
        raise ValueError(
            "Fused decode O weight must be E4M3 [groups * N, heads/groups * 512], N % 64 == 0"
        )
    if (
        wo_a_scales.dtype != torch.int32
        or wo_a_scales.shape
        != (num_groups, wo_a_weight.shape[0] // num_groups, group_width // 128)
        or wo_a_scales.device != q_weight.device
        or wo_a_weight.device != q_weight.device
    ):
        raise ValueError(
            "Fused decode O weights require packed per-32 scales on the Q weight device"
        )

    q_weight = q_weight.contiguous()
    packed = prepare_fp8_projection(q_weight, q_scales.contiguous())
    if packed is None:
        return None
    q_permuted, q_packed = _FUSED_FLASH_MLA_API.permute_q_b_proj(
        (q_weight, packed), num_heads, 512
    )
    # Each packed word contains four consecutive UE8M0 bytes. Copy the
    # logical words before viewing their bytes. A singleton packed K axis
    # can be "contiguous" while retaining a non-unit stride, so force a copy.
    q_canonical = q_packed.clone(memory_format=torch.contiguous_format).view(
        torch.uint8
    )
    wo_a_permuted, wo_a_packed = _FUSED_FLASH_MLA_API.permute_wv_proj(
        (wo_a_weight.contiguous().view(num_groups, -1, group_width), wo_a_scales),
        num_heads // num_groups,
        512,
    )
    return q_permuted, q_canonical, wo_a_permuted, wo_a_packed


@register_kernel(
    "attention",
    "dsv41_selected_attention_fp8",
    name="flashmla_selected_attention_fp8",
    solution="flashmla",
    capability=_FUSED_DECODE_CAPABILITY,
    signatures=frozenset({format_signature(x=dense_tensor_format(torch.bfloat16))}),
    priority=Priority.SPECIALIZED,
)
def flashmla_selected_attention_fp8(
    q: torch.Tensor,
    swa_cache: torch.Tensor | None,
    swa_slots: torch.Tensor | None,
    swa_lens: torch.Tensor | None,
    global_cache: torch.Tensor | None,
    global_slots: torch.Tensor | None,
    global_lens: torch.Tensor | None,
    attn_sink: torch.Tensor | None,
    softmax_scale: float,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    num_groups: int,
    prefill_kv: torch.Tensor | None,
    prefill_indices: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate permuted BF16 Q [T,H,512], attend selected cache rows, and inverse-RoPE O.

    SWA and optional global cache inputs are page-planar fields with physical
    slots [T,capacity] and lengths [T]. Alternatively, pass a BF16 workspace
    [rows,1,512] and its indices [T,capacity] with all paged inputs set to None.
    The optional FP32 sink has shape [H]. Positions [T] index the FP32
    [max_positions,64] cosine/sine table, with negative padding positions using
    row zero. Empty selections produce zero.

    The caller supplies bounded metadata: nonnegative slots address their
    cache field, lengths are within the selection capacity, and nonnegative
    positions are within the RoPE table. Paged fields include a readable null
    page. These are the runtime producers' invariants, not device-side scans.

    Return E4M3 output [T,groups,H/groups*512] in the permuted O layout and
    int32 packed per-32 scales [T,groups,H/groups*4] for the grouped projection.
    """
    if _FUSED_FLASH_MLA_API is None:
        raise ImportError("Fused decode requires the native FlashMLA fused API")
    if q.ndim != 3 or q.shape[-1] != 512 or q.dtype != torch.bfloat16:
        raise ValueError("V4.1 query must be BF16 [tokens, heads, 512]")
    tokens, heads, _ = q.shape
    if heads not in (64, 128) or heads != 8 * num_groups:
        raise ValueError(
            "Fused decode requires 64 or 128 heads with 8 heads per output group"
        )
    paged_inputs = (
        swa_cache,
        swa_slots,
        swa_lens,
        global_cache,
        global_slots,
        global_lens,
    )
    if (prefill_kv is None) != (prefill_indices is None):
        raise ValueError("workspace and indices must be supplied together")
    if any(
        tensor is not None and tensor.device != q.device
        for tensor in (
            *paged_inputs,
            positions,
            cos_sin_cache,
            prefill_kv,
            prefill_indices,
        )
    ):
        raise ValueError("all selected-attention tensors must share the query device")
    if positions.shape != (tokens,) or positions.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise ValueError("Fused decode positions must be integer [tokens]")
    if (
        cos_sin_cache.ndim != 2
        or cos_sin_cache.shape[0] == 0
        or cos_sin_cache.shape[1] != 64
        or cos_sin_cache.dtype != torch.float32
    ):
        raise ValueError("Fused decode RoPE table must be FP32 [max_positions,64]")

    q, attn_sink = _native_query(q, attn_sink)
    if prefill_kv is not None:
        if any(tensor is not None for tensor in paged_inputs):
            raise ValueError("workspace attention does not take paged inputs")
        if (
            prefill_kv.ndim != 3
            or prefill_kv.shape[1:] != (1, 512)
            or prefill_kv.dtype != torch.bfloat16
        ):
            raise ValueError("workspace must be BF16 [rows, 1, 512]")
        if (
            prefill_indices.ndim != 2
            or prefill_indices.shape[0] != tokens
            or prefill_indices.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError("workspace indices must be integer [tokens, capacity]")
    else:
        if swa_cache is None:
            raise ValueError("paged attention requires a SWA cache")
        if (global_cache is None) != (global_slots is None) or (
            global_cache is None
        ) != (global_lens is None):
            raise ValueError(
                "global cache, slots and lengths must be supplied together"
            )
        indices, lengths = _slots(swa_slots, swa_lens, tokens)
        k_cache = _paged(swa_cache, "swa", 512)
        extra_cache, extra_indices, extra_lengths = None, None, None
        if global_cache is not None:
            extra_cache = _paged(global_cache, "global", 256)
            extra_indices, extra_lengths = _slots(global_slots, global_lens, tokens)
    if tokens == 0:
        width = heads // num_groups * 512
        return (
            q.new_empty((0, num_groups, width), dtype=torch.float8_e4m3fn),
            q.new_empty((0, num_groups, width // 128), dtype=torch.int32),
        )

    common = dict(
        enable_q_norm=False,
        rms_norm_eps=0.0,
        token_positions=positions.clamp_min(0).to(torch.int32).contiguous(),
        is_rope_neox_style=False,
        rope_dim=64,
        cos_sin_cache=cos_sin_cache.contiguous(),
        n_wv_group=num_groups,
        num_per_channels=32,
        use_tma_aligned_col_major_sf=True,
        round_sf=True,
        use_packed_ue8m0=True,
        q=q,
        sm_scale=softmax_scale,
        d_v=512,
        attn_sink=attn_sink,
    )
    if prefill_kv is not None:
        indices = prefill_indices.to(torch.int32)
        width = max(64, (indices.shape[1] + 63) // 64 * 64)
        if width != indices.shape[1]:
            indices = F.pad(indices, (0, width - indices.shape[1]), value=-1)
        codes, scales, _, _ = _FUSED_FLASH_MLA_API.prefill(
            kv=prefill_kv.contiguous(),
            indices=indices.contiguous().unsqueeze(1),
            topk_length=None,
            **common,
        )
        return codes, scales

    codes, scales, _ = _FUSED_FLASH_MLA_API.decode(
        k_cache=k_cache,
        indices_in_kvcache=indices.squeeze(1),
        topk_length=lengths,
        extra_k_cache=extra_cache,
        extra_indices_in_kvcache=(
            None if extra_indices is None else extra_indices.squeeze(1)
        ),
        extra_topk_length=extra_lengths,
        **common,
    )
    return codes, scales


@register_kernel(
    "attention",
    "dsv41_selected_attention",
    name="flashmla_dsv41_selected_attention",
    solution="flashmla",
    # Hopper builds carry EXTRA_KVCACHE, EXTRA_TOPK_LENGTH and ATTN_SINK -- the
    # two-reader shape this op needs -- but of the packed layouts only the V4
    # cache reader. The caller keeps its rows in a format its target reads, so
    # the floor is the two-reader feature set rather than any one layout.
    capability=CapabilityRequirement(
        min_arch_version=ArchVersion(9, 0),
        max_arch_version=ArchVersion(10, 3),
        vendors=frozenset({"nvidia"}),
    ),
    signatures=frozenset({format_signature(x=dense_tensor_format(torch.bfloat16))}),
    traits={"flashmla_eligible": frozenset({True})},
    priority=Priority.SPECIALIZED,
)
def selected_attention(
    q,
    swa_cache,
    swa_slots,
    swa_lens,
    global_cache,
    global_slots,
    global_lens,
    attn_sink,
    softmax_scale,
    out,
    query_chunk_size,
    schedule,
    prefill_kv,
    prefill_indices,
    *,
    lse=None,
):
    """Use a caller-built prefill workspace or native paged decode, returning BF16.

    Persistent fields and schedules stay caller-owned. Native output views are
    retained when out is None; a supplied destination is filled in place.
    """
    if query_chunk_size <= 0:
        raise ValueError("query_chunk_size must be positive")
    if (prefill_kv is None) != (prefill_indices is None):
        raise ValueError("prefill_kv and prefill_indices must be supplied together")
    prefill = prefill_kv is not None
    inputs = (
        swa_cache,
        swa_slots,
        swa_lens,
        global_cache,
        global_slots,
        global_lens,
        out,
        prefill_kv,
        prefill_indices,
    )
    if any(t is not None and t.device != q.device for t in inputs):
        raise ValueError("all selected-attention tensors must share the query device")
    if out is not None and (out.shape != q.shape or out.dtype != q.dtype):
        raise ValueError("out must match query shape and dtype")
    if q.shape[0] == 0:
        return q if out is None else out
    api = flash_mla_api()
    if prefill:
        if prefill_kv.dtype != torch.bfloat16 or prefill_kv.shape[-1] != 512:
            raise ValueError("prefill_kv must contain BF16 512-dimensional rows")
        if prefill_kv.ndim == 2:
            prefill_kv = prefill_kv[:, None, :]
        if prefill_kv.ndim != 3 or prefill_kv.shape[1] != 1:
            raise ValueError("prefill_kv must have shape [rows, 1, 512]")
        if prefill_indices.ndim != 2 or prefill_indices.shape[0] != q.shape[0]:
            raise ValueError("prefill_indices must have shape [tokens, capacity]")
        if prefill_indices.dtype not in (torch.int32, torch.int64):
            raise ValueError("prefill_indices must be integer")
        indices = prefill_indices.to(torch.int32)
        # The sm90 sparse prefill kernel tiles the selection two blocks at a
        # time and asserts the padded width divides evenly: it rejects 192 and
        # 320 while accepting 128, 256 and 384. Later archs take any 64-multiple.
        align = 128 if current_platform().is_hopper else 64
        width = max(align, (indices.shape[1] + align - 1) // align * align)
        if width != indices.shape[1]:
            indices = F.pad(indices, (0, width - indices.shape[1]), value=-1)
        destination = out
        if destination is None and q.shape[0] > query_chunk_size:
            destination = torch.empty_like(q)
        query_buffer = None
        for start in range(0, q.shape[0], query_chunk_size):
            end = min(start + query_chunk_size, q.shape[0])
            if query_buffer is None:
                q_native, sink_native = _native_query(q[start:end], attn_sink)
                if q_native.shape[1] != q.shape[1]:
                    query_buffer = q_native
            else:
                q_native = query_buffer[: end - start]
                _copy_attention_heads(
                    q_native[:, : q.shape[1]], q[start:end], None, None
                )
            result, _, tile_lse = api.flash_mla_sparse_fwd(
                q=q_native,
                kv=prefill_kv,
                indices=indices[start:end].contiguous().unsqueeze(1),
                sm_scale=softmax_scale,
                d_v=512,
                attn_sink=sink_native,
                topk_length=None,
            )
            if lse is not None:
                lse[start:end].copy_(tile_lse[:, : q.shape[1]])
            if destination is None:
                return result[:, : q.shape[1], :]
            _copy_attention_heads(
                destination[start:end], result[:, : q.shape[1], :], None, None
            )
            # Same-stream calls can reuse Q's zero padding across tiles. Release
            # each padded output before allocating the next one.
            del result, q_native
        return destination
    else:
        q_native, sink_native = _native_query(q, attn_sink)
        if schedule is None:
            raise ValueError("native decode requires a caller-owned schedule")
        if (global_cache is None) != (global_slots is None) or (
            global_cache is None
        ) != (global_lens is None):
            raise ValueError(
                "global cache, slots and lengths must be supplied together"
            )
        indices, lengths = _slots(swa_slots, swa_lens, q.shape[0])
        extra_indices, extra_lengths = (
            (None, None)
            if global_cache is None
            else _slots(global_slots, global_lens, q.shape[0])
        )
        result, native_lse = api.flash_mla_with_kvcache(
            q=q_native.unsqueeze(1),
            k_cache=_paged(swa_cache, "swa", 512),
            block_table=None,
            cache_seqlens=None,
            head_dim_v=512,
            tile_scheduler_metadata=schedule,
            num_splits=None,
            softmax_scale=softmax_scale,
            causal=False,
            is_fp8_kvcache=True,
            indices=indices,
            attn_sink=sink_native,
            extra_k_cache=(
                None if global_cache is None else _paged(global_cache, "global", 256)
            ),
            extra_indices_in_kvcache=extra_indices,
            topk_length=lengths,
            extra_topk_length=extra_lengths,
        )
        if lse is not None:
            lse.copy_(native_lse.squeeze(-1)[:, : q.shape[1]])
        result = result.squeeze(1)
    result = result[:, : q.shape[1], :]
    if out is None:
        return result
    out.copy_(result)
    return out
