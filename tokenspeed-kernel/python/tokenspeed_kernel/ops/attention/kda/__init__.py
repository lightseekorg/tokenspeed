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
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

import torch
from tokenspeed_kernel.platform import current_platform
from tokenspeed_kernel.profiling import ShapeCapture, kernel_scope
from tokenspeed_kernel.registry import KernelRegistry, Priority
from tokenspeed_kernel.selection import (
    NoKernelFoundError,
    select_kernel,
    spec_matches_traits,
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
class KdaFusedDecodeResult:
    """Result from an optional pre-convolution KDA decode fusion.

    Attributes:
        out: Packed decode output ``[1, batch, heads, value_dim]``.
        output_norm_applied: Whether the selected kernel applied the output
            gate and RMSNorm, so the caller must not apply them again.
    """

    out: torch.Tensor
    output_norm_applied: bool


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
        initial_state: One backend-owned recurrent state per sequence.
        cu_seqlens: Device sequence boundaries ``[num_sequences + 1]``.
        cu_seqlens_cpu: REQUIRED host int64 copy of ``cu_seqlens`` with equal
            contents. Every solution plans its chunk indices from it on the
            host; reading the device boundaries instead would issue a
            stream-synchronizing D2H per KDA layer per chunk, which stalls
            the launch thread behind all queued work (and serializes the
            chunk pipeline's stages).
        lower_bound: Optional safe lower bound for log decay.
        override: Optional exact kernel name.
        solution: Optional registered solution name.
        recurrent_layout: Layout of the backend-owned recurrent state; the
            platform default when omitted.

    Returns:
        Packed output and final state, in the caller's ``recurrent_layout``.
    """
    recurrent_layout = recurrent_layout or kda_recurrent_layout()
    if q.ndim != 4 or q.shape[0] != 1:
        raise ValueError("KDA q must be [1, total_tokens, heads, key_dim]")
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
    kernel = select_kernel(
        "attention",
        "kda_paged_prefill",
        _attention_format_signature(q=q, k=k, v=v),
        solution=solution,
        override=override,
    )
    spec = KernelRegistry.get().get_by_name(kernel.name)
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
    )
    if relayout:
        # Hand the final state back in the caller's layout (a view; no copy).
        return KdaPrefillResult(result.out, result.final_state.transpose(-1, -2))
    return result


def kda_paged_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g_raw: torch.Tensor,
    beta_logits: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    *,
    state_pool: torch.Tensor,
    read_indices: torch.Tensor,
    write_indices: torch.Tensor,
    cu_seqlens: torch.Tensor,
    lower_bound: float | None = -5.0,
    override: str | None = None,
    solution: str | None = None,
    recurrent_layout: str | None = None,
) -> torch.Tensor:
    """Run post-convolution KDA decode against an indexed state pool.

    Args:
        q/k/g_raw: Packed tensors ``[1, batch, heads, key_dim]``.
        v: Packed values ``[1, batch, heads, value_dim]``.
        beta_logits: Raw beta logits ``[1, batch, heads]``.
        A_log/dt_bias: FP32 gate parameters.
        state_pool: Backend-owned recurrent-state pool.
        read_indices/write_indices: Independent source/destination rows.
        cu_seqlens: Device boundaries ``[batch + 1]``.
        lower_bound: Optional safe lower bound for log decay.
        override: Optional exact kernel name.
        solution: Optional registered solution name.
        recurrent_layout: Layout of the state pool; the platform default
            when omitted.

    Returns:
        KDA output with the same shape as ``v``.
    """
    recurrent_layout = recurrent_layout or kda_recurrent_layout()
    if q.ndim != 4 or q.shape[0] != 1:
        raise ValueError("KDA decode q must be [1, batch, heads, key_dim]")
    if k.shape != q.shape or g_raw.shape != q.shape:
        raise ValueError("KDA decode q, k, and g_raw must have identical shapes")
    if v.ndim != 4 or v.shape[:3] != q.shape[:3]:
        raise ValueError("KDA decode v must match q through the head dimension")
    if beta_logits.shape != q.shape[:-1]:
        raise ValueError("KDA beta logits must be [1, total_tokens, heads]")
    num_sequences = read_indices.numel()
    if read_indices.ndim != 1 or write_indices.shape != (num_sequences,):
        raise ValueError("KDA decode requires one read/write index per sequence")
    if cu_seqlens.numel() != num_sequences + 1:
        raise ValueError("KDA decode cu_seqlens must contain one boundary per sequence")

    kernel = select_kernel(
        "attention",
        "kda_paged_decode",
        _attention_format_signature(q=q, k=k, v=v),
        traits={
            "indexed_state": True,
            "single_token": q.shape[1] == num_sequences,
            "recurrent_layout": recurrent_layout,
        },
        solution=solution,
        override=override,
    )
    return kernel(
        q=q,
        k=k,
        v=v,
        g_raw=g_raw,
        beta_logits=beta_logits,
        A_log=A_log,
        dt_bias=dt_bias,
        state_pool=state_pool,
        read_indices=read_indices,
        write_indices=write_indices,
        cu_seqlens=cu_seqlens,
        lower_bound=lower_bound,
    )


def try_kda_fused_paged_decode(
    mixed_qkv: torch.Tensor,
    conv_weights: torch.Tensor,
    conv_states: torch.Tensor,
    f_a_out: torch.Tensor,
    f_b_weight: torch.Tensor,
    beta_logits: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    *,
    state_pool: torch.Tensor,
    read_indices: torch.Tensor,
    write_indices: torch.Tensor,
    num_heads: int,
    head_dim: int,
    cu_seqlens: torch.Tensor,
    lower_bound: float | None = -5.0,
    output_gate: torch.Tensor | None = None,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float | None = None,
    recurrent_layout: str | None = None,
    override: str | None = None,
    solution: str | None = None,
) -> KdaFusedDecodeResult | None:
    """Run a registered pre-convolution KDA decode fusion when available.

    ``output_gate``, ``norm_weight``, and ``norm_eps`` request a fused gated
    RMSNorm epilogue. If the selected backend only supports the original core
    fusion, the returned result reports that the caller must apply the
    epilogue.

    Returns ``None`` only when no implementation supports the current
    platform. Otherwise, returns the output and whether output normalization
    was applied. Invalid inputs and execution failures remain visible.
    """
    recurrent_layout = recurrent_layout or kda_recurrent_layout()
    if (output_gate is None) != (norm_weight is None):
        raise ValueError("output_gate and norm_weight must be provided together")
    if output_gate is not None and norm_eps is None:
        raise ValueError("norm_eps is required with fused KDA output normalization")
    if recurrent_layout not in ("k_major", "v_major"):
        raise ValueError(f"unsupported KDA recurrent layout {recurrent_layout!r}")

    signature = _attention_format_signature(
        q=mixed_qkv,
        k=mixed_qkv,
        v=mixed_qkv,
    )
    try:
        kernel = select_kernel(
            "attention",
            "kda_fused_paged_decode",
            signature,
            traits={
                "paged_state": True,
                "fused_output_norm": output_gate is not None,
                "num_heads": num_heads,
                "head_dim": head_dim,
                "conv_kernel_size": conv_weights.shape[-1],
                "recurrent_layout": recurrent_layout,
            },
            solution=solution,
            override=override,
        )
    except NoKernelFoundError:
        if output_gate is None:
            return None
        try:
            kernel = select_kernel(
                "attention",
                "kda_fused_paged_decode",
                signature,
                traits={
                    "paged_state": True,
                    "fused_output_norm": False,
                    "num_heads": num_heads,
                    "head_dim": head_dim,
                    "conv_kernel_size": conv_weights.shape[-1],
                    "recurrent_layout": recurrent_layout,
                },
                solution=solution,
                override=override,
            )
        except NoKernelFoundError:
            return None

    selected_spec = KernelRegistry.get().get_by_name(kernel.name)
    output_norm_applied = (
        output_gate is not None
        and selected_spec is not None
        and spec_matches_traits(
            selected_spec,
            {"fused_output_norm": True},
            require_all_traits=True,
        )
    )

    out = kernel(
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
        write_indices=write_indices,
        num_heads=num_heads,
        head_dim=head_dim,
        cu_seqlens=cu_seqlens,
        lower_bound=lower_bound,
        output_gate=output_gate if output_norm_applied else None,
        norm_weight=norm_weight if output_norm_applied else None,
        norm_eps=norm_eps if output_norm_applied else None,
    )
    return KdaFusedDecodeResult(out=out, output_norm_applied=output_norm_applied)


def try_kda_fused_paged_verify(
    mixed_qkv: torch.Tensor,
    conv_weights: torch.Tensor,
    conv_states: torch.Tensor,
    conv_scratch: torch.Tensor,
    f_a_out: torch.Tensor,
    f_b_weight: torch.Tensor,
    beta_logits: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    *,
    state_pool: torch.Tensor,
    state_scratch: torch.Tensor | None,
    read_indices: torch.Tensor,
    write_indices: torch.Tensor,
    num_heads: int,
    head_dim: int,
    draft_token_num: int,
    lower_bound: float | None = -5.0,
    recurrent_layout: str | None = None,
    override: str | None = None,
    solution: str | None = None,
    store_states: bool = True,
    replay_mixed_qkv: torch.Tensor | None = None,
    replay_gate: torch.Tensor | None = None,
    replay_beta: torch.Tensor | None = None,
    g_raw: torch.Tensor | None = None,
    conv_qkv: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """Run a registered pre-convolution KDA target-verify fusion when available.

    Mirrors ``try_kda_fused_paged_decode`` for the speculative verify batch:
    per-position conv windows and recurrent states land in the verify
    scratches for partial-accept commit. ``store_states`` selects the
    rollback-tape variant and ``recurrent_layout`` defaults to the
    platform's state layout; which producer arrangement runs is the
    registry's choice. Returns ``None`` only when no implementation
    supports the current platform.
    """
    recurrent_layout = recurrent_layout or kda_recurrent_layout()
    if recurrent_layout not in ("k_major", "v_major"):
        raise ValueError(f"unsupported KDA recurrent layout {recurrent_layout!r}")
    split_producers = g_raw is not None or conv_qkv is not None
    if split_producers and (g_raw is None or conv_qkv is None):
        raise ValueError("g_raw and conv_qkv must be provided together")
    signature = _attention_format_signature(
        q=mixed_qkv,
        k=mixed_qkv,
        v=mixed_qkv,
    )
    try:
        kernel = select_kernel(
            "attention",
            "kda_fused_paged_verify",
            signature,
            traits={
                "paged_state": True,
                "store_states": store_states,
                "recurrent_layout": recurrent_layout,
                "num_heads": num_heads,
                "head_dim": head_dim,
                "split_producers": split_producers,
            },
            solution=solution,
            override=override,
        )
    except NoKernelFoundError:
        return None
    kwargs = {}
    if replay_mixed_qkv is not None:
        kwargs.update(
            {
                "replay_mixed_qkv": replay_mixed_qkv,
                "replay_gate": replay_gate,
                "replay_beta": replay_beta,
            }
        )
    if split_producers:
        kwargs.update({"g_raw": g_raw, "conv_qkv": conv_qkv})
    return kernel(
        mixed_qkv=mixed_qkv,
        conv_weights=conv_weights,
        conv_states=conv_states,
        conv_scratch=conv_scratch,
        f_a_out=f_a_out,
        f_b_weight=f_b_weight,
        beta_logits=beta_logits,
        A_log=A_log,
        dt_bias=dt_bias,
        state_pool=state_pool,
        state_scratch=state_scratch,
        read_indices=read_indices,
        write_indices=write_indices,
        num_heads=num_heads,
        head_dim=head_dim,
        draft_token_num=draft_token_num,
        lower_bound=lower_bound,
        **kwargs,
    )


def kda_fused_paged_verify_uses_split_producers(
    dtype: torch.dtype,
    *,
    store_states: bool,
    recurrent_layout: str,
    num_heads: int,
    head_dim: int,
) -> bool:
    """Whether the selected verify implementation accepts split producers.

    Args:
        dtype: Activation dtype used to resolve the registered kernel.
        store_states: Whether verify materializes per-position rollback state.
        recurrent_layout: Committed recurrent-state layout.
        num_heads: Per-rank KDA head count.
        head_dim: KDA head width.

    Returns:
        True when the selected implementation accepts precomputed convolution
        and gate tensors.
    """
    probe = torch.empty(0, dtype=dtype, device="meta")
    signature = _attention_format_signature(q=probe, k=probe, v=probe)
    traits = {
        "paged_state": True,
        "store_states": store_states,
        "recurrent_layout": recurrent_layout,
    }
    traits["num_heads"] = num_heads
    traits["head_dim"] = head_dim
    try:
        kernel = select_kernel(
            "attention", "kda_fused_paged_verify", signature, traits=traits
        )
    except NoKernelFoundError:
        return False
    registered = KernelRegistry.get().get_by_name(kernel.name)
    return bool(
        registered is not None
        and registered.traits.get("split_producers") == frozenset({True})
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
    recurrent_layout: str,
) -> torch.Tensor:
    """Materialize the convolution producer used by split KDA verification.

    Args:
        mixed_qkv: Packed raw QKV projection rows.
        conv_weights: Fused four-tap QKV convolution weights.
        conv_states: Committed convolution-state pool.
        read_indices: Committed page index for each request.
        num_heads: Per-rank KDA head count.
        head_dim: KDA head width.
        draft_token_num: Verify positions per request.
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
            "split_producers": True,
            "recurrent_layout": recurrent_layout,
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
    )


def try_kda_replay_commit(
    mixed_qkv: torch.Tensor,
    conv_weights: torch.Tensor,
    conv_states: torch.Tensor,
    conv_out: torch.Tensor,
    f_a_out: torch.Tensor,
    f_b_weight: torch.Tensor,
    beta_logits: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    *,
    state_pool: torch.Tensor,
    state_out: torch.Tensor,
    read_indices: torch.Tensor,
    write_indices: torch.Tensor,
    accepted_length: torch.Tensor,
    num_heads: int,
    head_dim: int,
    draft_token_num: int,
    lower_bound: float | None = -5.0,
    override: str | None = None,
    solution: str | None = None,
    gate_scratch: torch.Tensor | None = None,
    replay_gate: torch.Tensor | None = None,
    recurrent_layout: str | None = None,
) -> bool:
    """Run a registered KDA speculative replay-commit when available.

    Replays the accepted prefix of a verified draft window from the committed
    page, so the caller never has to keep a recurrent state per draft
    position. Pass the SAME projections the verify pass consumed.
    ``gate_scratch`` is transient fp32 scratch for the hoisted gate
    (``[>= N*T, num_heads*head_dim]``); ``None`` falls back to a
    kernel-module buffer. ``recurrent_layout`` defaults to the platform's
    state layout.

    Returns:
        ``True`` when a kernel ran, ``False`` when none supports the current
        platform (the caller must then fall back to a scratch-based commit).
    """
    recurrent_layout = recurrent_layout or kda_recurrent_layout()
    signature = _attention_format_signature(
        q=mixed_qkv,
        k=mixed_qkv,
        v=mixed_qkv,
    )
    try:
        kernel = select_kernel(
            "attention",
            "kda_replay_commit",
            signature,
            traits={
                "flat_state": True,
                "recurrent_layout": recurrent_layout,
                "num_heads": num_heads,
                "head_dim": head_dim,
            },
            solution=solution,
            override=override,
        )
    except NoKernelFoundError:
        return False
    kwargs = {"replay_gate": replay_gate} if replay_gate is not None else {}
    kernel(
        mixed_qkv=mixed_qkv,
        conv_weights=conv_weights,
        conv_states=conv_states,
        conv_out=conv_out,
        f_a_out=f_a_out,
        f_b_weight=f_b_weight,
        beta_logits=beta_logits,
        A_log=A_log,
        dt_bias=dt_bias,
        state_pool=state_pool,
        state_out=state_out,
        read_indices=read_indices,
        write_indices=write_indices,
        accepted_length=accepted_length,
        num_heads=num_heads,
        head_dim=head_dim,
        draft_token_num=draft_token_num,
        lower_bound=lower_bound,
        gate_scratch=gate_scratch,
        **kwargs,
    )
    return True


def resolve_kda_batched_replay_commit(
    dtype: torch.dtype = torch.bfloat16,
    *,
    num_heads: int | None = None,
    head_dim: int | None = None,
):
    """Resolve the all-layer replay kernel once, or return ``None``.

    Batched kernels dereference descriptor addresses as BF16, so other dtypes
    use the per-layer commit.

    Args:
        dtype: Activation dtype used by the replay payload.
        num_heads: Local KDA head count, when known.
        head_dim: KDA head dimension, when known.
    """
    if dtype is not torch.bfloat16:
        return None
    probe = torch.empty(0, dtype=dtype, device="meta")
    signature = _attention_format_signature(q=probe, k=probe, v=probe)
    traits = {"flat_state": True, "batched_layers": True}
    if num_heads is not None:
        traits["num_heads"] = num_heads
    if head_dim is not None:
        traits["head_dim"] = head_dim
    try:
        return select_kernel(
            "attention",
            "kda_replay_commit",
            signature,
            traits=traits,
            override=(
                "triton_nvidia_kda_batched_replay_commit"
                if current_platform().is_nvidia
                else None
            ),
        )
    except NoKernelFoundError:
        return None


def kda_batched_replay_uses_raw_gate(
    dtype: torch.dtype = torch.bfloat16,
    *,
    num_heads: int | None = None,
    head_dim: int | None = None,
) -> bool:
    """Whether the selected batched replay consumes persistent BF16 raw-g.

    Args:
        dtype: Activation dtype used by the replay payload.
        num_heads: Local KDA head count, when known.
        head_dim: KDA head dimension, when known.
    """
    kernel = resolve_kda_batched_replay_commit(
        dtype, num_heads=num_heads, head_dim=head_dim
    )
    if kernel is None:
        return False
    registered = KernelRegistry.get().get_by_name(kernel.name)
    if registered is None:
        return False
    return registered.traits.get("replay_raw_gate") == frozenset({True})


def kda_replay_commit_supported(
    dtype: torch.dtype = torch.bfloat16,
    *,
    solution: str | None = None,
    recurrent_layout: str | None = None,
    num_heads: int | None = None,
    head_dim: int | None = None,
) -> bool:
    """Whether this platform can run the KDA speculative replay path.

    Lets a caller decide up front whether it can skip allocating a
    per-draft-position state scratch, before any verify batch has run. The
    eager replay path has no decomposed fallback, so it needs both the
    standalone commit kernel and the no-store fused verify it rides on.

    Args:
        dtype: activation dtype the verify batch will use.
        solution: restrict to one registered solution, as in ``select_kernel``.
        recurrent_layout: Layout of the committed state; the platform default
            when omitted. It must match what the caller stores, or the probe
            answers for kernels the backend will not select.
        num_heads: Local KDA head count, when known.
        head_dim: KDA head dimension, when known.

    Returns:
        ``True`` when both kernels are registered for the current platform.
    """
    recurrent_layout = recurrent_layout or kda_recurrent_layout()
    probe = torch.empty(0, dtype=dtype, device="meta")
    signature = _attention_format_signature(q=probe, k=probe, v=probe)
    shape_traits = {}
    if num_heads is not None:
        shape_traits["num_heads"] = num_heads
    if head_dim is not None:
        shape_traits["head_dim"] = head_dim
    try:
        select_kernel(
            "attention",
            "kda_replay_commit",
            signature,
            traits={
                "flat_state": True,
                "recurrent_layout": recurrent_layout,
                **shape_traits,
            },
            solution=solution,
        )
        select_kernel(
            "attention",
            "kda_fused_paged_verify",
            signature,
            traits={
                "paged_state": True,
                "store_states": False,
                "recurrent_layout": recurrent_layout,
                **shape_traits,
            },
            solution=solution,
        )
    except NoKernelFoundError:
        return False
    return True


# Backend registration (side-effect imports)
# isort: off
import tokenspeed_kernel.ops.attention.kda.triton  # noqa: E402,F401
import tokenspeed_kernel.ops.attention.kda.cuda  # noqa: E402,F401
import tokenspeed_kernel.ops.attention.kda.cute_dsl  # noqa: E402,F401
import tokenspeed_kernel.ops.attention.kda.gluon  # noqa: E402,F401

# isort: on

__all__ = [
    "KdaPrefillResult",
    "KdaFusedDecodeResult",
    "kda_recurrent_layout",
    "kda_paged_prefill",
    "kda_paged_decode",
    "try_kda_fused_paged_decode",
    "try_kda_fused_paged_verify",
    "kda_fused_paged_verify_uses_split_producers",
    "kda_verify_conv_update",
    "try_kda_replay_commit",
    "resolve_kda_batched_replay_commit",
    "kda_batched_replay_uses_raw_gate",
    "kda_replay_commit_supported",
]
