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

import math
from dataclasses import dataclass
from functools import cached_property

import torch
from tokenspeed_kernel._optional import has_optional_package
from tokenspeed_kernel.ops.activation.triton import rmsnorm_gated_sigmoid
from tokenspeed_kernel.ops.attention.kda._prefill_capacity import KdaPrefillCapacity
from tokenspeed_kernel.platform import current_platform, pdl_enabled
from tokenspeed_kernel.registry import KernelRegistry
from tokenspeed_kernel.selection import (
    NoKernelFoundError,
    select_kernel,
)
from tokenspeed_kernel.signature import (
    MXFP8_BLOCK_SCALE,
    dense_tensor_format,
    format_signature,
    tensor_format,
)

AttentionResult = torch.Tensor | tuple[torch.Tensor, torch.Tensor | None]

# One UE8M0 scale per 32 consecutive head_dim elements (MXFP8).
MXFP8_ATTENTION_BLOCK_SCALE = MXFP8_BLOCK_SCALE


def _attention_format_signature(**roles: torch.Tensor):
    return format_signature(
        **{role: dense_tensor_format(tensor.dtype) for role, tensor in roles.items()}
    )


def _mxfp8_attention_format_signature(**roles: torch.Tensor):
    return format_signature(
        **{
            role: tensor_format(
                "mxfp8", tensor.dtype, scale=MXFP8_ATTENTION_BLOCK_SCALE
            )
            for role, tensor in roles.items()
        }
    )


def _blockscaled_signature_and_scales(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    q_scale: torch.Tensor | None,
    k_scale: torch.Tensor | None,
    v_scale: torch.Tensor | None,
):
    """Pick dense vs MXFP8 signature and build the scale kwargs splat.

    q_scale selects the block-scaled path; k_scale/v_scale must accompany it.
    Returns (signature, scale_kwargs) for the paged-KV-cache entry points.
    """
    if q_scale is not None:
        assert (
            k_scale is not None and v_scale is not None
        ), "MXFP8 attention requires q_scale, k_scale, and v_scale together"
        signature = _mxfp8_attention_format_signature(
            q=q, k_cache=k_cache, v_cache=v_cache
        )
    else:
        signature = _attention_format_signature(q=q, k_cache=k_cache, v_cache=v_cache)
    return signature, dict(q_scale=q_scale, k_scale=k_scale, v_scale=v_scale)


LSE_LN = math.log2(math.e)


# ===-----------------------------------------------------------------------===#
# KDA Kernels
# ===-----------------------------------------------------------------------===#


@dataclass(frozen=True)
class KdaPrefillResult:
    """Results from a packed KDA prefill.

    Attributes:
        out: Packed output ``[1, total_tokens, heads, value_dim]``.
        final_state: One final recurrent state per packed sequence.
    """

    out: torch.Tensor
    final_state: torch.Tensor


@dataclass(frozen=True)
class KdaOutputNorm:
    """The gated RMSNorm a KDA layer applies to its attention output.

    ``out = rmsnorm(core) * weight * sigmoid(gate)`` per head over ``head_dim``,
    in FP32 on the BF16 core output, rounded once to the output dtype.

    Attributes:
        gate: ``[tokens, heads * head_dim]`` raw output-gate logits with unit
            inner stride; a column view of the input projection is fine.
        weight: ``[head_dim]`` RMSNorm weight.
        eps: RMSNorm epsilon.
    """

    gate: torch.Tensor
    weight: torch.Tensor
    eps: float

    def rows(self, begin: int, end: int) -> KdaOutputNorm:
        """The norm of token rows ``[begin, end)``."""
        return KdaOutputNorm(self.gate[begin:end], self.weight, self.eps)


def apply_kda_output_norm(
    out: torch.Tensor, output_norm: KdaOutputNorm | None
) -> torch.Tensor:
    """``out`` gated-RMSNormed as a ``[tokens, heads * head_dim]`` tensor, or as is.

    Args:
        out: KDA attention output whose leading dimensions flatten to the
            gate's token rows and whose last dimension is ``head_dim`` or
            ``heads * head_dim``.
        output_norm: The layer's output norm, or None to return ``out``.

    Returns:
        ``out`` when ``output_norm`` is None, else the normed output.
    """
    if output_norm is None:
        return out
    tokens, width = output_norm.gate.shape
    head_dim = output_norm.weight.shape[0]
    return rmsnorm_gated_sigmoid(
        out.reshape(tokens, width).contiguous(),
        output_norm.gate,
        output_norm.weight,
        output_norm.eps,
        width // head_dim,
        head_dim,
        enable_pdl=pdl_enabled(),
    )


# Accepted records a request slot keeps beside its verify window by default.
# The runtime's default history capacity is the window plus this many, so a
# verify reads at most this many retained records whatever its width.
KDA_DEFAULT_RETAINED_RECORDS = 12


@dataclass(frozen=True)
class KdaRetainedHistory:
    """Per-request verify records and accepted lengths, outside the cache pages.

    correction is FP32 [slot, head, capacity, value]. key and decay are
    FP32 [slot, head, capacity, key] and hold normalized K and multiplicative
    decay. length is INT32 [slot, 1] and counts the records a slot has
    accepted since the dense base of its request's page; every layer of a
    request shares one count. A verify applies a count only within
    [0, capacity - window], and none otherwise: the prepare ahead of it folded
    those records into the live page. record_slots maps
    each request row to its resident record slot; slot zero is valid.
    live_state_indices maps each request row to its private physical page.
    Cache pages hold only dense checkpoints: the owner folds a slot's
    accepted records into its page before another reader sees that page.
    The index vectors and storage tensors may be strided views.

    Construction validates storage metadata, which must stay fixed for the
    object's lifetime. Tensor contents may be refreshed in place.
    """

    correction: torch.Tensor
    key: torch.Tensor
    decay: torch.Tensor
    length: torch.Tensor
    live_state_indices: torch.Tensor
    record_slots: torch.Tensor

    def __post_init__(self) -> None:
        if self.correction.ndim != 4:
            raise ValueError(
                "retained correction must be [slot, head, capacity, value]"
            )
        slots, heads, capacity, _ = self.correction.shape
        if (
            capacity <= 0
            or self.key.ndim != 4
            or self.key.shape[:3] != (slots, heads, capacity)
            or self.decay.shape != self.key.shape
        ):
            raise ValueError("retained records must share slot/head/capacity geometry")
        if any(
            x.dtype != torch.float32 for x in (self.correction, self.key, self.decay)
        ):
            raise ValueError("retained correction, key, and decay must be FP32")
        if self.length.shape != (slots, 1) or self.length.dtype != torch.int32:
            raise ValueError("retained length must be INT32 [slot, 1]")
        indices = (self.live_state_indices, self.record_slots)
        if (
            self.live_state_indices.ndim != 1
            or self.record_slots.shape != self.live_state_indices.shape
            or any(x.dtype not in (torch.int32, torch.int64) for x in indices)
        ):
            raise ValueError(
                "retained live indices and record slots must be matching integer vectors"
            )
        if any(
            x.device != self.correction.device
            for x in (self.key, self.decay, self.length, *indices)
        ):
            raise ValueError("retained history fields must be on the same device")

    @cached_property
    def packed_record_width(self) -> int | None:
        """Return the common FP32 row width if every head can use a 16-byte bulk copy."""
        width = self.correction.shape[-1]
        for field in (self.correction, self.key, self.decay):
            slot_stride, head_stride, record_stride, channel_stride = field.stride()
            if (
                field.shape[-1] != width
                or record_stride != width
                or channel_stride != 1
                or slot_stride % 4
                or head_stride % 4
                or field.data_ptr() % 16
            ):
                return None
        return width


# Retained commit descriptor ABI: UINT64 [layers, 12].
# Each layer's row contains 3 pointers, 1 page count, and 8 strides:
#   0: original QKV pointer
#   1: convolution state pointer
#   2: page count (the convolution and recurrent pools hold the same pages)
#   3: QKV row stride
#   4: convolution page stride
#   5: convolution channel stride
#   6: convolution tap stride
#   7: v-major recurrent state pointer
#   8: recurrent page stride
#   9: recurrent head stride
#  10: recurrent value stride
#  11: recurrent key stride
# All strides count elements. Per-layer values support different arena layouts.
# A request's accepted count is not a layer's: every layer shares it, and the
# prepare and the commit take it as a tensor.
# Verify already stores U/K/decay records; commit reads them only to fold an
# ended request's accepted records into its live page. Commit updates the
# convolution state, then publishes the request's count. The prepare ahead of
# a verify forward reads the page count and recurrent-state columns.
KDA_RETAINED_COMMIT_DESCRIPTOR_COLUMNS = 12


@dataclass(frozen=True)
class KdaRetainedRecords:
    """Request-owned records and lengths, packed QKV and validity metadata.

    history owns FP32 U/K/decay and accepted lengths. original_qkv optionally
    receives pre-convolution inputs. metadata is INT64 [batch, 3] containing
    slot/generation/absolute start; stamps is INT64 [slots, ring_capacity, 2].
    Stamp capacity is independent of history capacity. query_start_loc gives
    packed request boundaries. Tensor contents may be refreshed in place.
    """

    original_qkv: torch.Tensor | None
    history: KdaRetainedHistory
    metadata: torch.Tensor
    stamps: torch.Tensor
    query_start_loc: torch.Tensor


def kda_recurrent_layout() -> str:
    """Return the recurrent state layout this platform's KDA kernels consume.

    Returns:
        ``"v_major"`` where the paged slab is ``[pages, HV, V, K]``, else
        ``"k_major"``. K equals V for the supported head geometry, so the two
        differ only in which axis is contiguous.
    """
    platform = current_platform()
    v_major = platform.is_nvidia or platform.is_cdna4 or platform.is_cdna5
    return "v_major" if v_major else "k_major"


def kda_paged_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g_raw: torch.Tensor,
    beta_logits: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    *,
    initial_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cu_seqlens_cpu: torch.Tensor,
    capacity: KdaPrefillCapacity | None,
    lower_bound: float | None = -5.0,
    override: str | None = None,
    solution: str | None = None,
    recurrent_layout: str | None = None,
) -> KdaPrefillResult:
    """Run packed KDA prefill through capability-based kernel selection.

    Args:
        q/k/g_raw: Packed tensors ``[1, total_tokens, heads, key_dim]``.
        v: Values ``[1, total_tokens, heads, value_dim]``.
        beta_logits: Raw beta logits ``[1, total_tokens, heads]``.
        A_log/dt_bias: FP32 gate parameters.
        initial_state: One backend-owned recurrent state per sequence, FP32
            or BF16.
        cu_seqlens: Device sequence boundaries ``[num_sequences + 1]``.
        cu_seqlens_cpu: REQUIRED host int64 copy of ``cu_seqlens`` with equal
            contents. Used for capacity admission and backends that require
            host planning, without a synchronizing D2H per layer. Backends
            with device planning consume cu_seqlens directly.
        capacity: Explicit graph planning bounds, or None for exact
            live-length planning. Live boundaries retain their normal meaning;
            input rows past the final live boundary must be zero.
        lower_bound: Optional safe lower bound for log decay.
        override: Optional exact kernel name.
        solution: Optional registered solution name.
        recurrent_layout: Layout of the backend-owned recurrent state; the
            platform default when omitted.

    Returns:
        Packed output and FP32 final state, in the caller's
        ``recurrent_layout``.
    """
    recurrent_layout = recurrent_layout or kda_recurrent_layout()
    if q.ndim != 4 or q.shape[0] != 1:
        raise ValueError("KDA q must be [1, total_tokens, heads, key_dim]")
    # Prefill scans take and return FP32 state. A BF16 state pool is widened
    # here, and the caller narrows the final state when writing it back.
    initial_state = initial_state.float()
    if k.shape != q.shape or g_raw.shape != q.shape:
        raise ValueError("KDA q, k, and g_raw must have identical shapes")
    if v.ndim != 4 or v.shape[:3] != q.shape[:3]:
        raise ValueError("KDA v must match q through the head dimension")
    if beta_logits.shape != q.shape[:-1]:
        raise ValueError("KDA beta logits must be [1, total_tokens, heads]")
    num_sequences = cu_seqlens.numel() - 1
    if initial_state.ndim != 4 or initial_state.shape[0] != num_sequences:
        raise ValueError("KDA initial_state must contain one row per sequence")
    if (
        not isinstance(cu_seqlens_cpu, torch.Tensor)
        or cu_seqlens_cpu.is_cuda
        or cu_seqlens_cpu.dtype != torch.int64
        or cu_seqlens_cpu.numel() != cu_seqlens.numel()
    ):
        raise ValueError(
            "KDA cu_seqlens_cpu must be a host int64 tensor with one entry "
            f"per cu_seqlens boundary; got {type(cu_seqlens_cpu).__name__}"
        )
    if solution == "fla":
        solution = "triton"
    capacity_kwargs = {}
    if capacity is not None:
        capacity.validate(cu_seqlens_cpu, q.shape[1])
    kernel = select_kernel(
        "attention",
        "kda_paged_prefill",
        _attention_format_signature(q=q, k=k, v=v),
        solution=solution,
        override=override,
    )
    spec = KernelRegistry.get().get_by_name(kernel.name)
    supports_capacity = spec is not None and True in spec.traits.get(
        "prefill_capacity", ()
    )
    if capacity is not None and not supports_capacity:
        raise ValueError("Selected KDA kernel does not support planning capacity")
    if supports_capacity:
        capacity_kwargs["capacity"] = capacity
    supported = None if spec is None else spec.traits.get("recurrent_layout")
    # Kernels that declare no layout consume the caller's state as it is.
    relayout = supported is not None and recurrent_layout not in supported
    if relayout:
        initial_state = initial_state.transpose(-1, -2).contiguous()
    result = kernel(
        q=q,
        k=k,
        v=v,
        g_raw=g_raw,
        beta_logits=beta_logits,
        A_log=A_log,
        dt_bias=dt_bias,
        initial_state=initial_state,
        cu_seqlens=cu_seqlens,
        cu_seqlens_cpu=cu_seqlens_cpu,
        lower_bound=lower_bound,
        **capacity_kwargs,
    )
    if relayout:
        # Hand the final state back in the caller's layout (a view; no copy).
        return KdaPrefillResult(result.out, result.final_state.transpose(-1, -2))
    return result


def kda_fused_paged_verify(
    mixed_qkv: torch.Tensor,
    conv_weights: torch.Tensor,
    conv_states: torch.Tensor,
    f_a_out: torch.Tensor | None,
    f_b_weight: torch.Tensor | None,
    beta_logits: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    *,
    state_pool: torch.Tensor,
    read_indices: torch.Tensor,
    num_heads: int,
    head_dim: int,
    draft_token_num: int,
    retained_records: KdaRetainedRecords,
    output_norm: KdaOutputNorm | None,
    recurrent_layout: str,
    lower_bound: float | None,
    override: str | None,
    solution: str | None,
    g_raw: torch.Tensor | None = None,
    conv_qkv: torch.Tensor | None = None,
) -> torch.Tensor:
    """Append candidates to retained history and return packed attention output.

    mixed_qkv and convolution inputs describe packed pre-convolution rows.
    retained_records supplies request-owned history, private live destinations,
    packed request boundaries and candidate validity stamps. read_indices
    identifies immutable source pages; convolution reads them. The round's
    history must already be prepared (resolve_kda_batched_retained_prepare):
    verify reads each row's live page as its dense base, applies its accepted
    records, and writes candidates after them. Verify itself never writes a
    page or a length; commit subsequently publishes accepted candidate length.
    f_a_out/f_b_weight project the gate when g_raw is omitted. NVIDIA split
    producers require both g_raw and conv_qkv. lower_bound selects bounded
    sigmoid decay, or None selects softplus decay. Unsupported geometry or
    kernel selection raises before any state or record is modified.
    """
    if (
        head_dim <= 0
        or num_heads <= 0
        or conv_weights.shape[-1] != 4
        or recurrent_layout != "v_major"
        or draft_token_num <= 0
    ):
        raise ValueError("Unsupported retained KDA verify geometry")
    if conv_states.shape[0] != state_pool.shape[0]:
        raise ValueError("KDA convolution and recurrent pools must hold the same pages")
    if retained_records is None:
        raise ValueError("Retained KDA verify requires retained records")
    if draft_token_num > retained_records.history.correction.shape[2]:
        raise ValueError("Retained KDA candidate window exceeds history capacity")
    if conv_qkv is not None and g_raw is None:
        raise ValueError("Precomputed convolution requires a projected raw gate")
    if g_raw is None and (f_a_out is None or f_b_weight is None):
        raise ValueError(
            "KDA verify requires a raw gate or both gate projection inputs"
        )
    kernel = select_kernel(
        "attention",
        "kda_fused_paged_verify",
        _attention_format_signature(q=mixed_qkv, k=mixed_qkv, v=mixed_qkv),
        traits={
            "num_heads": num_heads,
            "head_dim": head_dim,
            "conv_kernel_size": conv_weights.shape[-1],
            "paged_state": True,
            "recurrent_layout": recurrent_layout,
            **({"split_producers": True} if conv_qkv is not None else {}),
            "retained_history": True,
            "spec_query_len": draft_token_num,
            "batch_size": read_indices.shape[0],
        },
        solution=solution,
        override=override,
    )
    registered = KernelRegistry.get().get_by_name(kernel.name)
    if registered is None or True not in registered.traits.get("retained_history", ()):
        raise ValueError("Selected KDA kernel does not support retained history")
    producers = {}
    if conv_qkv is not None:
        producers["conv_qkv"] = conv_qkv
    if g_raw is not None:
        producers["g_raw"] = g_raw
    return kernel(
        mixed_qkv=mixed_qkv,
        conv_weights=conv_weights,
        conv_states=conv_states,
        f_a_out=f_a_out,
        f_b_weight=f_b_weight,
        beta_logits=beta_logits,
        A_log=A_log,
        dt_bias=dt_bias,
        state_pool=state_pool,
        read_indices=read_indices,
        num_heads=num_heads,
        head_dim=head_dim,
        draft_token_num=draft_token_num,
        lower_bound=lower_bound,
        retained_records=retained_records,
        output_norm=output_norm,
        **producers,
    )


def kda_fused_paged_verify_uses_split_producers(
    dtype: torch.dtype,
    *,
    recurrent_layout: str,
    num_heads: int,
    head_dim: int,
) -> bool:
    """Return whether retained verify uses separate convolution/gate producers.

    dtype and fixed model geometry select the retained verify/commit pair.
    This initialization-time query allocates no GPU storage.
    """
    if not kda_retained_supported(
        dtype,
        recurrent_layout=recurrent_layout,
        num_heads=num_heads,
        head_dim=head_dim,
        conv_kernel_size=4,
    ):
        return False
    probe = torch.empty(0, dtype=dtype, device="meta")
    kernel = select_kernel(
        "attention",
        "kda_fused_paged_verify",
        _attention_format_signature(q=probe, k=probe, v=probe),
        traits={
            "paged_state": True,
            "recurrent_layout": recurrent_layout,
            "num_heads": num_heads,
            "head_dim": head_dim,
            "retained_history": True,
        },
    )
    registered = KernelRegistry.get().get_by_name(kernel.name)
    return registered.traits.get("split_producers") == frozenset({True})


def kda_fused_paged_verify_fuses_producers(
    dtype: torch.dtype,
    *,
    recurrent_layout: str,
    num_heads: int,
    head_dim: int,
    draft_token_num: int,
    batch_size: int,
) -> bool:
    """Whether record verify runs its gate and convolution producers inline.

    Args:
        dtype: Activation dtype used to resolve the registered kernel.
        recurrent_layout: Committed recurrent-state layout.
        num_heads: Per-rank KDA head count.
        head_dim: KDA head width.
        draft_token_num: Tokens per request window.
        batch_size: Requests in the batch (padding included).

    Returns:
        True when the selected record verify for these windows computes the gate
        projection and convolution itself, so the caller passes neither
        ``g_raw`` nor ``conv_qkv``. Records and commit are the same either way.
    """
    probe = torch.empty(0, dtype=dtype, device="meta")
    signature = _attention_format_signature(q=probe, k=probe, v=probe)
    traits = {
        "num_heads": num_heads,
        "head_dim": head_dim,
        "paged_state": True,
        "recurrent_layout": recurrent_layout,
        "retained_history": True,
        "spec_query_len": draft_token_num,
        "batch_size": batch_size,
    }
    try:
        kernel = select_kernel(
            "attention", "kda_fused_paged_verify", signature, traits=traits
        )
    except NoKernelFoundError:
        return False
    registered = KernelRegistry.get().get_by_name(kernel.name)
    return (
        registered is not None
        and registered.traits.get("split_producers") == frozenset({False})
        and registered.traits.get("retained_history") == frozenset({True})
    )


def kda_verify_conv_update(
    mixed_qkv: torch.Tensor,
    conv_weights: torch.Tensor,
    conv_states: torch.Tensor,
    read_indices: torch.Tensor,
    *,
    num_heads: int,
    head_dim: int,
    draft_token_num: int,
    query_start_loc: torch.Tensor,
    bias: torch.Tensor | None = None,
    original_qkv: torch.Tensor | None = None,
    recurrent_layout: str,
) -> torch.Tensor:
    """Materialize the convolution producer used by split KDA verification.

    Args:
        mixed_qkv: Packed raw QKV projection rows.
        conv_weights: Fused four-tap QKV convolution weights.
        conv_states: Committed convolution-state pool.
        read_indices: Committed page index for each request.
        bias: Optional per-channel bias before SiLU; None adds zero.
        original_qkv: Optional separate destination for pre-convolution QKV.
            The producer retains each input row for accepted-history commit.
            None leaves retention to the caller.
        num_heads: Per-rank KDA head count.
        head_dim: KDA head width.
        draft_token_num: Maximum verify positions per request.
        query_start_loc: Packed request boundaries.
        recurrent_layout: Committed recurrent-state layout.

    Returns:
        Convolved and SiLU-activated QKV rows.
    """
    signature = _attention_format_signature(
        q=mixed_qkv,
        k=mixed_qkv,
        v=mixed_qkv,
    )
    kernel = select_kernel(
        "attention",
        "kda_verify_conv_update",
        signature,
        traits={
            "paged_state": True,
            "recurrent_layout": recurrent_layout,
            "split_producers": True,
        },
    )
    return kernel(
        mixed_qkv=mixed_qkv,
        conv_weights=conv_weights,
        conv_states=conv_states,
        read_indices=read_indices,
        num_heads=num_heads,
        head_dim=head_dim,
        draft_token_num=draft_token_num,
        query_start_loc=query_start_loc,
        bias=bias,
        original_qkv=original_qkv,
    )


def kda_materialize_retained_state(
    recurrent: torch.Tensor,
    *,
    history: KdaRetainedHistory,
    source_indices: torch.Tensor,
    recurrent_layout: str,
) -> None:
    """Apply accepted history to gathered dense state in place.

    recurrent is the [request, head, V, K] gather output (K,V for k_major).
    source_indices holds each row's source page, and history.record_slots its
    records and the count history.length gives. Rows with nonpositive source
    pages are skipped. The count stays as it is: every layer of a request
    shares it, so the caller clears it once every layer has gathered
    (kda_reset_retained_length), on the same stream.
    """
    from tokenspeed_kernel.ops.attention.kda._retained import materialize

    materialize(
        recurrent,
        history=history,
        source_indices=source_indices,
        recurrent_layout=recurrent_layout,
        paged=False,
    )


def kda_materialize_retained_pages(
    recurrent_pool: torch.Tensor,
    *,
    history: KdaRetainedHistory,
    source_indices: torch.Tensor,
    recurrent_layout: str,
) -> None:
    """Fold each request's accepted records into its private page.

    recurrent_pool holds [page, head, V, K] state (K,V for k_major).
    source_indices lists one physical page per history.record_slots entry.
    Positive pages and their slots must be unique. Rows with nonpositive pages
    are skipped. Every layer of a request shares its count, which this leaves
    as it is: the caller clears it once every layer has folded
    (kda_reset_retained_length), on the same stream.
    """
    from tokenspeed_kernel.ops.attention.kda._retained import materialize

    materialize(
        recurrent_pool,
        history=history,
        source_indices=source_indices,
        recurrent_layout=recurrent_layout,
        paged=True,
    )


def kda_reset_retained_length(
    length: torch.Tensor,
    *,
    write_indices: torch.Tensor,
    record_slots: torch.Tensor,
    page_count: int,
) -> None:
    """Clear the slot counts of rows whose page received a dense state.

    length is INT32 [slot, 1]. write_indices holds each request row's
    destination page and record_slots its slot; page_count is the State pool's
    page count. Padding rows -- a page at or below 0, or at or past
    page_count -- keep their counts, as materialization skips them. Every
    layer shares a request's count, so the caller clears it once, after every
    layer's fold, in a launch ordered behind their dense writes.
    """
    from tokenspeed_kernel.ops.attention.kda._retained import reset_length

    reset_length(
        length,
        write_indices=write_indices,
        record_slots=record_slots,
        page_count=page_count,
    )


def resolve_kda_batched_retained_commit(
    dtype: torch.dtype, *, num_heads: int, head_dim: int, recurrent_layout: str
):
    """Resolve retained cross-layer convolution and count publication.

    Returns a callable or None for unsupported geometry/platform. Its uint64
    [layers, 12] descriptors hold QKV/conv pointers, the page count, QKV row,
    conv page, conv channel and conv tap strides, then the recurrent state
    pointer and its page, head, value and key strides, in elements.
    record_slots selects each request's slot in replay_length, the count
    every layer shares. The first launch reads it in every layer; a second,
    one-dimensional launch then writes each carried request's count once: the
    records its verify read (the count, or 0 when it lies outside
    [0, history_capacity - draft_token_num]) plus its acceptance. A request
    whose slot is set in request_stopped ended in this forward: its accepted
    records are folded into its live page and its count is cleared. One
    whose commit a layer that carried it rejects instead is reported as
    ``REQUEST_ENDED_INCOMPLETE`` in request_ends, the forward's per-row
    ``apply_stop_conditions`` report. Metadata and all pointers remain stable
    during CUDA graph capture.
    """
    return _resolve_batched_retained(
        "kda_retained_commit",
        dtype,
        num_heads=num_heads,
        head_dim=head_dim,
        recurrent_layout=recurrent_layout,
    )


def resolve_kda_batched_retained_prepare(
    dtype: torch.dtype, *, num_heads: int, head_dim: int, recurrent_layout: str
):
    """Resolve the cross-layer fold that readies retained history for verify.

    Returns a callable or None for unsupported geometry/platform. It takes the
    commit's descriptors, layer groups, record slots, FP32 records and shared
    counts, plus [groups, requests] source and live pages, the packed
    boundaries and the window (draft_token_num) of the verify that follows.
    A request's count H overflows when it lies outside
    [0, capacity - draft_token_num]; the verify and the commit then apply no
    record, so a verify sees at most capacity - draft_token_num records a
    request. In each layer, an overflowing request gets the source state with
    records [0, H) applied in its live page; otherwise a request whose live
    page differs from its source in the layer's group gets the source state
    copied, with no record applied. One launch decides and copies, and only
    reads the counts: the commit alone writes them. Run it once per verify
    round, after the round's pages, boundaries and record slots are
    refreshed and before any layer's verify: a second run would copy the
    unchanged source over the fold.
    """
    return _resolve_batched_retained(
        "kda_retained_prepare",
        dtype,
        num_heads=num_heads,
        head_dim=head_dim,
        recurrent_layout=recurrent_layout,
    )


def _resolve_batched_retained(
    op: str,
    dtype: torch.dtype,
    *,
    num_heads: int,
    head_dim: int,
    recurrent_layout: str,
):
    platform = current_platform()
    if (
        num_heads <= 0
        or head_dim <= 0
        or recurrent_layout != "v_major"
        or not (platform.is_nvidia or platform.is_cdna4 or platform.is_cdna5)
    ):
        return None
    probe = torch.empty(0, dtype=dtype, device="meta")
    try:
        kernel = select_kernel(
            "attention",
            op,
            _attention_format_signature(q=probe, k=probe, v=probe),
            traits={
                "batched_layers": True,
                "retained_history": True,
                "num_heads": num_heads,
                "head_dim": head_dim,
                "recurrent_layout": recurrent_layout,
            },
        )
    except NoKernelFoundError:
        return None
    registered = KernelRegistry.get().get_by_name(kernel.name)
    if registered is None or True not in registered.traits.get("retained_history", ()):
        return None
    return kernel


def kda_retained_supported(
    dtype: torch.dtype,
    *,
    recurrent_layout: str,
    num_heads: int,
    head_dim: int,
    conv_kernel_size: int,
) -> bool:
    """Whether retained verify and the all-layer prepare and commit are available.

    This initialization-time probe uses model geometry and activation dtype;
    it does not allocate GPU storage or execute kernels. Layout, convolution
    width and every capability declaration must match before serving can omit
    per-round raw correction/key-gate workspace.
    """
    geometry = {
        "num_heads": num_heads,
        "head_dim": head_dim,
        "recurrent_layout": recurrent_layout,
    }
    if (
        conv_kernel_size != 4
        or resolve_kda_batched_retained_prepare(dtype, **geometry) is None
        or resolve_kda_batched_retained_commit(dtype, **geometry) is None
    ):
        return False
    probe = torch.empty(0, dtype=dtype, device="meta")
    try:
        kernel = select_kernel(
            "attention",
            "kda_fused_paged_verify",
            _attention_format_signature(q=probe, k=probe, v=probe),
            traits={
                "paged_state": True,
                "retained_history": True,
                "recurrent_layout": recurrent_layout,
                "num_heads": num_heads,
                "head_dim": head_dim,
                "conv_kernel_size": conv_kernel_size,
            },
        )
    except NoKernelFoundError:
        return False
    registered = KernelRegistry.get().get_by_name(kernel.name)
    return registered is not None and True in registered.traits.get(
        "retained_history", ()
    )


# Backend registration (side-effect imports)
# isort: off
import tokenspeed_kernel.ops.attention.kda.triton  # noqa: E402,F401
import tokenspeed_kernel.ops.attention.kda.vllm_recoverssm  # noqa: E402,F401

if has_optional_package("flash_kda"):
    import tokenspeed_kernel.ops.attention.kda.cuda  # noqa: E402,F401
import tokenspeed_kernel.ops.attention.kda.cute_dsl  # noqa: E402,F401
import tokenspeed_kernel.ops.attention.kda.cudnn  # noqa: E402,F401
import tokenspeed_kernel.ops.attention.kda.gluon  # noqa: E402,F401

# isort: on

__all__ = [
    "KdaOutputNorm",
    "KdaPrefillResult",
    "apply_kda_output_norm",
    "kda_fused_paged_verify_fuses_producers",
    "KdaRetainedHistory",
    "KdaRetainedRecords",
    "KDA_RETAINED_COMMIT_DESCRIPTOR_COLUMNS",
    "kda_materialize_retained_pages",
    "kda_materialize_retained_state",
    "kda_reset_retained_length",
    "resolve_kda_batched_retained_commit",
    "resolve_kda_batched_retained_prepare",
    "kda_retained_supported",
    "kda_recurrent_layout",
    "kda_paged_prefill",
    "kda_fused_paged_verify",
    "kda_fused_paged_verify_uses_split_producers",
    "kda_verify_conv_update",
    "KDA_DEFAULT_RETAINED_RECORDS",
]
