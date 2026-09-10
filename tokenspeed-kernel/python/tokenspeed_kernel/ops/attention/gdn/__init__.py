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
from tokenspeed_kernel.platform import current_platform, pdl_enabled
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
# GDN Kernels
# ===-----------------------------------------------------------------------===#


class GdnCheckpointLayout(str, Enum):
    """Backend-native checkpoint layout returned by GDN chunk prefill."""

    NONE = "none"
    FLA = "fla"
    FLASHINFER = "flashinfer"


@dataclass(frozen=True)
class GdnChunkPrefillResult:
    """Structured result for GDN chunk prefill.

    Args:
        out: GDN output tensor.
        final_state: Final recurrent state, when requested.
        h: Optional backend-native intermediate recurrent checkpoints.
        h_cu_starts: Optional cumulative checkpoint starts for FlashInfer layout.
        h_layout: Layout of ``h``.
    """

    out: torch.Tensor
    final_state: torch.Tensor | None
    h: torch.Tensor | None = None
    h_cu_starts: torch.Tensor | None = None
    h_layout: GdnCheckpointLayout = GdnCheckpointLayout.NONE


def gdn_chunk_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    scale: float | None,
    initial_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
    qk_l2norm: bool = False,
    output_final_state: bool = True,
    output_h: bool = False,
    override: str | None = None,
    solution: str | None = None,
) -> GdnChunkPrefillResult:
    """Run Gated Delta Net chunked prefill through kernel selection.

    Args:
        q: Query tensor shaped ``[1, total_tokens, num_q_heads, head_dim]``.
        k: Key tensor shaped ``[1, total_tokens, num_k_heads, head_dim]``.
        v: Value tensor shaped ``[1, total_tokens, num_v_heads, head_v_dim]``.
        g: Log-space forget gate shaped ``[1, total_tokens, num_v_heads]``.
        beta: Beta gate shaped ``[1, total_tokens, num_v_heads]``.
        scale: Attention scale. ``None`` lets the implementation use its default.
        initial_state: Recurrent state, K-last: ``[batch, num_v_heads,
            head_v_dim, head_dim]``. This matches flashinfer's native GDN
            decode/MTP layout (and the runtime's SSM state pool); backends
            whose own math is FLA-native (e.g. Triton) transpose internally.
        cu_seqlens: Cumulative sequence lengths for variable-length prefill.
        qk_l2norm: Whether the selected kernel should L2-normalize Q/K.
        output_final_state: Whether to return the final recurrent state.
        output_h: Whether to return intermediate recurrent checkpoints in the
            selected backend's native layout.
        override: Optional kernel override name.
        solution: Optional kernel solution to force through normal selection.

    Returns:
        ``GdnChunkPrefillResult`` with output, final state (K-last, same
        layout as ``initial_state``), and optional backend-native recurrent
        checkpoints (also K-last).
    """
    head_dim = q.shape[-1]
    head_v_dim = v.shape[-1]
    num_q_heads = q.shape[-2]
    num_v_heads = v.shape[-2]
    traits = {
        "head_dim": head_dim,
        "head_v_dim": head_v_dim,
        "head_v_eq_head_k": head_v_dim == k.shape[-1],
        "num_v_gte_num_q": num_v_heads >= num_q_heads,
        "qk_l2norm": qk_l2norm,
        "output_h": output_h,
    }
    signature = _attention_format_signature(q=q, k=k, v=v)
    kernel = select_kernel(
        "attention",
        "gdn_chunk_prefill",
        signature,
        traits=traits,
        solution=solution,
        override=override,
    )

    shape_params = {
        "batch_size": cu_seqlens.shape[0] - 1,
        "total_tokens": q.shape[1] if q.dim() == 4 else q.shape[0],
        "num_q_heads": num_q_heads,
        "num_v_heads": num_v_heads,
        "head_dim": head_dim,
        "head_v_dim": head_v_dim,
    }
    ShapeCapture.get().record(
        "attention",
        "gdn_chunk_prefill",
        kernel.name,
        q.dtype,
        shape_params,
    )

    with kernel_scope(
        "attention",
        "gdn_chunk_prefill",
        q.dtype,
        kernel_name=kernel.name,
        **shape_params,
    ):
        return kernel(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            initial_state=initial_state,
            cu_seqlens=cu_seqlens,
            qk_l2norm=qk_l2norm,
            output_final_state=output_final_state,
            output_h=output_h,
        )


def gdn_decode_step(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    b: torch.Tensor,
    initial_state: torch.Tensor,
    initial_state_indices: torch.Tensor,
    scale: float | None = None,
    output_state_indices: torch.Tensor | None = None,
    use_qk_l2norm: bool = True,
    override: str | None = None,
    solution: str | None = None,
) -> torch.Tensor:
    """Run one single-token (T=1) GDN decode step through kernel selection.

    Args:
        q: Query tensor shaped ``[B, 1, num_q_heads, head_dim]``.
        k: Key tensor shaped ``[B, 1, num_q_heads, head_dim]``.
        v: Value tensor shaped ``[B, 1, num_v_heads, head_v_dim]``.
        A_log: Floating-point log decay parameter shaped ``[num_v_heads]``.
            Backends that require FP32 normalize it internally.
        a: Input-dependent decay shaped ``[B, 1, num_v_heads]``.
        dt_bias: Floating-point decay bias shaped ``[num_v_heads]``. Backends
            that require FP32 normalize it internally.
        b: Update-gate (beta) input shaped ``[B, 1, num_v_heads]``.
        initial_state: SSM state pool, K-last ``[pool_size, num_v_heads,
            head_v_dim, head_dim]`` (matches the runtime's SSM state pool).
        initial_state_indices: Per-batch read row, shaped ``[B]``. ``-1``
            marks CUDA-graph padding; handled internally, no caller clamp
            needed.
        scale: Attention scale. ``None`` lets the implementation use its default.
        output_state_indices: Per-batch write row, shaped ``[B]``. ``None``
            writes back to ``initial_state_indices`` (the common, non-flat
            pool case); pass distinct rows for flat dual-index state paging.
        use_qk_l2norm: Whether the selected kernel should L2-normalize Q/K.
        override: Optional kernel override name.
        solution: Optional kernel solution to force through normal selection.

    Returns:
        Decode output shaped ``[B, 1, num_v_heads, head_v_dim]`` (q.dtype).
    """
    head_dim = q.shape[-1]
    signature = _attention_format_signature(q=q, k=k, v=v)
    kernel = select_kernel(
        "attention",
        "gdn_decode_step",
        signature,
        traits={"head_dim": head_dim},
        solution=solution,
        override=override,
    )
    with kernel_scope(
        "attention",
        "gdn_decode_step",
        q.dtype,
        kernel_name=kernel.name,
        batch_size=q.shape[0],
        num_v_heads=v.shape[-2],
        head_dim=head_dim,
        head_v_dim=v.shape[-1],
    ):
        return kernel(
            q=q,
            k=k,
            v=v,
            A_log=A_log,
            a=a,
            dt_bias=dt_bias,
            b=b,
            initial_state=initial_state,
            initial_state_indices=initial_state_indices,
            scale=scale,
            output_state_indices=output_state_indices,
            use_qk_l2norm=use_qk_l2norm,
        )


def gdn_decode_mtp(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    b: torch.Tensor,
    initial_state: torch.Tensor,
    initial_state_indices: torch.Tensor,
    scale: float | None = None,
    disable_state_update: bool = True,
    use_qk_l2norm: bool = True,
    intermediate_states_buffer: torch.Tensor | None = None,
    output_state_indices: torch.Tensor | None = None,
    override: str | None = None,
    solution: str | None = None,
) -> torch.Tensor:
    """Run one multi-token (T>1) GDN MTP verify step through kernel selection.

    Args:
        q: Query tensor shaped ``[B, T, num_q_heads, head_dim]``.
        k: Key tensor shaped ``[B, T, num_q_heads, head_dim]``.
        v: Value tensor shaped ``[B, T, num_v_heads, head_v_dim]``.
        A_log: Floating-point log decay parameter shaped ``[num_v_heads]``.
            Backends that require FP32 normalize it internally.
        a: Input-dependent decay shaped ``[B, T, num_v_heads]``.
        dt_bias: Floating-point decay bias shaped ``[num_v_heads]``. Backends
            that require FP32 normalize it internally.
        b: Update-gate (beta) input shaped ``[B, T, num_v_heads]``.
        initial_state: SSM state pool, K-last ``[pool_size, num_v_heads,
            head_v_dim, head_dim]`` (matches the runtime's SSM state pool).
        initial_state_indices: Per-batch read row, shaped ``[B]``. When
            ``output_state_indices`` is not provided and
            ``disable_state_update=False``, the final state is written back to
            that same row. Padding handling is solution and state-dtype
            specific: the portable Triton and FlashInfer FP32 paths suppress
            state reads and writes for negative rows, while FlashInfer's BF16
            fast path redirects them to row 0 and requires the caller to
            reserve that row.
        scale: Attention scale. ``None`` lets the implementation use its default.
        disable_state_update: When True (default), never write back to
            ``initial_state_indices``.
        use_qk_l2norm: Whether the selected kernel should L2-normalize Q/K.
        intermediate_states_buffer: Optional batch-scoped ``[B, T,
            num_v_heads, head_v_dim, head_dim]`` (K-last, same dtype as
            ``initial_state``) buffer that receives every step's post-update
            state at ``buffer[i_n, step]``.
        output_state_indices: Optional per-token state-pool destinations shaped
            ``[B, T]`` with dtype ``torch.int32``. When provided, each
            post-update state ``h_{t+1}`` is written directly to
            ``initial_state[output_state_indices[i, t]]``. Negative entries
            are safe only when the selected solution skips the corresponding
            negative initial-state row; otherwise entries must be
            non-negative. This is mutually exclusive with
            ``intermediate_states_buffer`` and requires
            ``disable_state_update=False``.
        override: Optional kernel override name.
        solution: Optional kernel solution to force through normal selection.

    Returns:
        Decode output shaped ``[B, T, num_v_heads, head_v_dim]`` (q.dtype).
        Outputs for negative initial-state indices are undefined and must be
        ignored, including on the FlashInfer FP32 path.
    """
    if output_state_indices is not None:
        if output_state_indices.shape != q.shape[:2]:
            raise ValueError(
                "output_state_indices must have shape "
                f"{tuple(q.shape[:2])}, got {tuple(output_state_indices.shape)}"
            )
        if output_state_indices.dtype != torch.int32:
            raise ValueError(
                "output_state_indices must have dtype torch.int32, got "
                f"{output_state_indices.dtype}"
            )
        if intermediate_states_buffer is not None:
            raise ValueError(
                "output_state_indices and intermediate_states_buffer are "
                "mutually exclusive"
            )
        if disable_state_update:
            raise ValueError("output_state_indices requires disable_state_update=False")

    head_dim = q.shape[-1]
    signature = _attention_format_signature(q=q, k=k, v=v)
    kernel = select_kernel(
        "attention",
        "gdn_decode_mtp",
        signature,
        traits={"head_dim": head_dim},
        solution=solution,
        override=override,
    )
    with kernel_scope(
        "attention",
        "gdn_decode_mtp",
        q.dtype,
        kernel_name=kernel.name,
        batch_size=q.shape[0],
        seq_len=q.shape[1],
        num_v_heads=v.shape[-2],
        head_dim=head_dim,
        head_v_dim=v.shape[-1],
    ):
        return kernel(
            q=q,
            k=k,
            v=v,
            A_log=A_log,
            a=a,
            dt_bias=dt_bias,
            b=b,
            initial_state=initial_state,
            initial_state_indices=initial_state_indices,
            scale=scale,
            disable_state_update=disable_state_update,
            use_qk_l2norm=use_qk_l2norm,
            intermediate_states_buffer=intermediate_states_buffer,
            output_state_indices=output_state_indices,
        )


def gdn_replay_commit(
    payload: torch.Tensor,
    parameters: torch.Tensor,
    *,
    state_addresses: torch.Tensor,
    state_row_strides: torch.Tensor,
    read_indices: torch.Tensor,
    write_indices: torch.Tensor,
    accepted_length: torch.Tensor,
    draft_token_num: int,
    geometry: tuple[int, int, int, int],
    state_dtype: torch.dtype,
    override: str | None = None,
    solution: str | None = None,
) -> None:
    """Replay every GDN layer's accepted prefix in one kernel launch.

    K/V/a/b share one layer-major allocation. Recurrent slabs may remain
    physically disjoint: ``state_addresses`` and ``state_row_strides`` expose
    them as a layer-indexed table to the kernel. Each program decodes its
    layer, request, and value-head coordinates and writes only the final
    accepted state.

    Args:
        payload: Contiguous packed K/V/a/b storage shaped
            ``[L, token_capacity, H*K + HV*V + 2*HV]``. The first ``B*T``
            rows of each layer hold the current request-major verify window.
        parameters: FP32 A_log/dt_bias table shaped ``[L, 2, HV]``.
        state_addresses: uint64 base-address table shaped ``[L]`` for the
            K-last recurrent-state pools.
        state_row_strides: int64 row strides in elements, shaped ``[L]``.
        read_indices: Committed-state pages shaped ``[L, B]``.
        write_indices: Accepted-state destination pages shaped ``[L, B]``.
        accepted_length: Accepted verified-token count per request, shaped ``[B]``.
        draft_token_num: Number of verify positions per request (``T``).
        geometry: ``(num_k_heads, num_v_heads, head_k_dim, head_v_dim)``.
        state_dtype: Element dtype shared by all recurrent-state pools.
        override: Optional exact registered kernel name.
        solution: Optional registered solution name.
    """
    if payload.dim() != 3 or not payload.is_contiguous():
        raise ValueError("GDN layer replay payload must be contiguous [L, rows, width]")
    num_layers = payload.shape[0]
    batch_size = accepted_length.numel()
    if num_layers == 0 or batch_size == 0:
        return
    num_k_heads, num_v_heads, head_k_dim, head_v_dim = geometry
    if draft_token_num <= 0:
        raise ValueError("draft_token_num must be positive")
    if num_v_heads <= 0 or num_k_heads <= 0 or num_v_heads % num_k_heads:
        raise ValueError("num_v_heads must be divisible by num_k_heads")
    if head_k_dim <= 0 or head_v_dim <= 0:
        raise ValueError("GDN replay head dimensions must be positive")
    payload_width = (
        num_k_heads * head_k_dim + num_v_heads * head_v_dim + 2 * num_v_heads
    )
    if payload.shape[1] < batch_size * draft_token_num:
        raise ValueError("GDN replay payload has insufficient token capacity")
    if payload.shape[2] != payload_width:
        raise ValueError(
            f"GDN replay payload width must be {payload_width}, got {payload.shape[2]}"
        )
    if parameters.shape != (num_layers, 2, num_v_heads):
        raise ValueError(
            "GDN replay parameters must have shape "
            f"{(num_layers, 2, num_v_heads)}, got {tuple(parameters.shape)}"
        )
    if parameters.dtype != torch.float32 or not parameters.is_contiguous():
        raise ValueError("GDN replay parameters must be contiguous torch.float32")
    if state_addresses.shape != (num_layers,) or state_addresses.dtype != torch.uint64:
        raise ValueError(
            "state_addresses must be torch.uint64 with one entry per layer"
        )
    if (
        state_row_strides.shape != (num_layers,)
        or state_row_strides.dtype != torch.int64
    ):
        raise ValueError(
            "state_row_strides must be torch.int64 with one entry per layer"
        )
    if read_indices.shape != (num_layers, batch_size):
        raise ValueError(
            "read_indices must have shape "
            f"{(num_layers, batch_size)}, got {tuple(read_indices.shape)}"
        )
    if write_indices.shape != (num_layers, batch_size):
        raise ValueError(
            "write_indices must have shape "
            f"{(num_layers, batch_size)}, got {tuple(write_indices.shape)}"
        )
    if read_indices.dtype != torch.int32 or write_indices.dtype != torch.int32:
        raise ValueError("GDN replay page tables must have dtype torch.int32")
    if accepted_length.shape != (batch_size,) or accepted_length.dtype != torch.int32:
        raise ValueError("accepted_length must be a one-dimensional torch.int32 tensor")
    if state_dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError(f"unsupported GDN replay state dtype: {state_dtype}")
    tensors = (
        parameters,
        state_addresses,
        state_row_strides,
        read_indices,
        write_indices,
        accepted_length,
    )
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise ValueError(
            "GDN replay address, stride, index, and parameter tables must be contiguous"
        )
    if any(tensor.device != payload.device for tensor in tensors):
        raise ValueError("all GDN replay tensors must reside on the payload device")

    signature = _attention_format_signature(q=payload, k=payload, v=payload)
    kernel = select_kernel(
        "attention",
        "gdn_replay_commit",
        signature,
        traits={"flat_state": True},
        solution=solution,
        override=override,
    )
    with kernel_scope(
        "attention",
        "gdn_replay_commit",
        payload.dtype,
        kernel_name=kernel.name,
        batch_size=batch_size,
        seq_len=draft_token_num,
        num_layers=num_layers,
        num_v_heads=num_v_heads,
        head_dim=head_k_dim,
        head_v_dim=head_v_dim,
    ):
        kernel(
            payload=payload,
            parameters=parameters,
            state_addresses=state_addresses,
            state_row_strides=state_row_strides,
            read_indices=read_indices,
            write_indices=write_indices,
            accepted_length=accepted_length,
            draft_token_num=draft_token_num,
            num_k_heads=num_k_heads,
            num_v_heads=num_v_heads,
            head_k_dim=head_k_dim,
            head_v_dim=head_v_dim,
            state_dtype=state_dtype,
        )


def gdn_replay_commit_supported(
    dtype: torch.dtype = torch.bfloat16,
    *,
    solution: str | None = None,
) -> bool:
    """Whether ReplaySSM can replace per-draft GDN recurrent-state scratch.

    Args:
        dtype: Activation dtype used by the target verify pass.
        solution: Optional registered solution restriction.

    Returns:
        ``True`` when a compatible GDN replay kernel is registered for the
        current platform.
    """
    probe = torch.empty(0, dtype=dtype, device="meta")
    signature = _attention_format_signature(q=probe, k=probe, v=probe)
    try:
        select_kernel(
            "attention",
            "gdn_replay_commit",
            signature,
            traits={"flat_state": True},
            solution=solution,
        )
    except NoKernelFoundError:
        return False
    return True


# Backend registration (side-effect imports)
# isort: off
import tokenspeed_kernel.ops.attention.gdn.flashinfer  # noqa: E402,F401
import tokenspeed_kernel.ops.attention.gdn.triton  # noqa: E402,F401

# isort: on

__all__ = [
    "GdnCheckpointLayout",
    "GdnChunkPrefillResult",
    "gdn_chunk_prefill",
    "gdn_decode_step",
    "gdn_decode_mtp",
    "gdn_replay_commit",
    "gdn_replay_commit_supported",
]
