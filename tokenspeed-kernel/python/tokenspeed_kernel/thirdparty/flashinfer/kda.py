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

"""FlashInfer KDA prefill integration."""

from __future__ import annotations

from functools import lru_cache

import torch
from tokenspeed_kernel.thirdparty.flashinfer.kda_plan import (
    KdaPrefillPlan,
    warm_kda_prefill_plan,
)


@lru_cache(maxsize=None)
def _compile_cutedsl_kda(device: int, gate_dtype: torch.dtype, lower_bound: float):
    import cutlass
    import cutlass.cute as cute
    from flashinfer.kda_kernels import kda_chunked_bt16 as kda

    gate_type = {torch.bfloat16: cutlass.BFloat16, torch.float32: cutlass.Float32}[
        gate_dtype
    ]
    fakes, checkpoints, checkpoint_starts, checkpoint_indices = kda._unified_fakes(
        cutlass.BFloat16, cutlass.Float32, True, True, False, False, False, gate_type
    )
    # The public wrapper updates state in place and retains GPU scratch globally.
    # Use its compiled kernel with separate state output and invocation-owned scratch.
    with torch.cuda.device(device):
        warm_kda_prefill_plan()
        return cute.compile(
            kda.host_unified,
            *fakes,
            kda.make_fake_stream(),
            0,
            0,
            0,
            0,
            0,
            kda.DEFAULT_SCALE,
            checkpoints,
            checkpoint_starts,
            checkpoint_indices,
            True,
            lower_bound * kda.LOG2_E,
            kda.THREADS_PER_CTA,
            None,
            gate_type,
            options="--enable-tvm-ffi",
        )


@lru_cache(maxsize=None)
def _compile_persistent_kda(device: int, heads: int, lower_bound: float):
    from flashinfer.kda_kernels.persistent.forward import _get_compiled
    from tokenspeed_kernel.thirdparty.flashinfer.kda_plan import (
        warm_persistent_kda_plan,
    )

    with torch.cuda.device(device):
        warm_persistent_kda_plan(heads)
        # Lengths and handoffs change on replay; keep their handling in one
        # compiled variant instead of specializing on each batch's partition.
        return _get_compiled(
            heads,
            torch.device("cuda", device),
            gate2=0,
            has_split=True,
            beta_tma=True,
            bf16_dv=True,
            beta_bf16=True,
            qk_rowpair=8,
            reg_mode=2,
            store_final_state=True,
            full_chunks=False,
            lower_bound_m5=lower_bound == -5.0,
            gate_roll=True,
            partial_tma=True,
        )


def _run_persistent_kda(
    compiled,
    q,
    k,
    v,
    g_raw,
    beta_logits,
    A_log,
    dt_bias,
    initial_state,
    cu_seqlens,
    lower_bound,
    sm_count,
):
    from cuda.bindings.driver import CUstream
    from tokenspeed_kernel.thirdparty.flashinfer.kda_plan import persistent_kda_plan

    sequences, heads, dim = initial_state.shape[:3]
    workers = min(sequences * heads, sm_count)
    cu32, offsets, chains, starts, lengths, sources, destinations, flags = (
        persistent_kda_plan(cu_seqlens, heads, workers)
    )
    handoffs = torch.empty(
        (workers if sequences * heads > sm_count else 1, dim, dim),
        dtype=torch.float32,
        device=q.device,
    )
    # Unused export/probe ABI views share invocation-owned dummy storage.
    dummy = torch.empty(64 * 256, dtype=torch.bfloat16, device=q.device)
    out = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    final_state = torch.empty_like(initial_state)
    compiled(
        q[0],
        k[0],
        v[0],
        g_raw.contiguous()[0],
        beta_logits.contiguous()[0],
        A_log.contiguous(),
        dt_bias.reshape(heads, dim).contiguous(),
        initial_state.view(-1, dim, dim),
        final_state.view(-1, dim, dim),
        out[0],
        cu32,
        offsets,
        chains,
        starts,
        lengths,
        sources,
        destinations,
        handoffs,
        flags,
        dummy.view(64, 1, 256),
        dummy.view(torch.float32)[:160].view(1, 160),
        dummy.view(torch.int64)[:64],
        0,
        1,
        workers,
        0,
        0,
        1,
        1,
        dim**-0.5,
        lower_bound * 1.4426950408889634,
        CUstream(torch.cuda.current_stream(q.device).cuda_stream),
    )
    return out, final_state


def _run_cutedsl_kda(
    compiled,
    q,
    k,
    v,
    g_raw,
    beta_logits,
    A_log,
    dt_bias,
    initial_state,
    cu_seqlens,
    sm_count,
    plan,
):
    from cuda.bindings.driver import CUstream
    from flashinfer.kda_kernels import kda_chunked_bt16 as kda

    heads, tokens = q.shape[2], q.shape[1]
    sequences = cu_seqlens.numel() - 1
    route = kda._route_for_workspace(sequences, heads, q.device, "auto")
    # The engine never reads the decomposition workspace; one chunk supplies
    # the ABI's required views without retaining a process-global GPU cache.
    chunks = (tokens + 15) // 16 + sequences - 1 if route == "decomp" else 1
    workspace = torch.empty(
        kda._decomp_ws_bytes(heads, chunks), dtype=torch.uint8, device=q.device
    )
    scratch = kda._partition_workspace(workspace, heads, chunks)
    if plan is None:
        # FlashInfer builds the metadata in this launch.
        cu_chunks = torch.empty(sequences + 1, dtype=torch.int32, device=q.device)
        order = torch.empty(sequences, dtype=torch.int32, device=q.device)
        generate = 1
    else:
        order, cu_chunks, generate = plan.sequence_order, plan.cu_chunks, 0
    num_ctas, chunks_per_cta = kda._k1_grid(chunks, heads)
    out = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    final_state = torch.empty_like(initial_state)
    compiled(
        q,
        k,
        v,
        g_raw.contiguous(),
        A_log.contiguous(),
        dt_bias.reshape(heads, 128).contiguous(),
        beta_logits.contiguous(),
        cu_seqlens,
        order,
        generate,
        None,
        cu_chunks,
        generate,
        scratch["kd"],
        scratch["qd"],
        scratch["w"],
        scratch["qk"],
        scratch["diag"],
        initial_state,
        out,
        final_state,
        CUstream(torch.cuda.current_stream(q.device).cuda_stream),
        1,
        sm_count,
        0,
        num_ctas,
        chunks_per_cta,
        q.shape[-1] ** -0.5,
        None,
        None,
        None,
    )
    return out, final_state


def cutedsl_kda_prefill(
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
    lower_bound: float,
    plan: KdaPrefillPlan | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select ordinary or persistent CuTe prefill from live packed boundaries.

    Args:
        q/k/v: BF16 [1, capacity, heads, 128], with dense head/features and
            16-byte-aligned bases and token strides. Token rows may be strided.
        g_raw: BF16 or FP32 raw gate with the same shape as q.
        beta_logits: BF16 [1, capacity, heads] raw beta logits.
        A_log/dt_bias: FP32 per-head decay and per-channel gate bias.
        initial_state: FP32 [sequences, heads, V, K] gathered cache state.
        cu_seqlens: Device int64 boundaries [sequences + 1]. Live lengths may
            change on replay and their sum may be less than token capacity;
            input rows beyond the live total must be zero, as in the runtime.
            Every sequence must be non-empty; unused capacity slots scan one
            zero-padded token each.
        lower_bound: Negative safe-gate bound, fixed for the model.
        plan: Ordinary-scan metadata built from ``cu_seqlens`` once for the
            batch's layers, or None for the launch to build its own. The
            persistent scan plans inside its launch.

    Returns:
        Packed BF16 output and separate FP32 final state. Input state is
        unchanged; scratch belongs to this call or its CUDA graph pool.
    """
    from flashinfer.kda_kernels import kda_chunked_bt16 as kda

    heads = q.shape[2]
    sequences = cu_seqlens.numel() - 1
    if plan is not None and plan.cu_chunks.numel() != cu_seqlens.numel():
        raise ValueError("KDA prefill plan was built for another batch shape")
    sm_count = kda._device_sm_count(q.device)
    can_compile_persistent = (
        heads % 8 == 0 and g_raw.dtype == torch.bfloat16 and -5.0 <= lower_bound < 0
    )
    compiled_cutedsl = _compile_cutedsl_kda(q.device.index, g_raw.dtype, lower_bound)
    compiled_persistent = None
    if can_compile_persistent:
        compiled_persistent = _compile_persistent_kda(
            q.device.index, heads, lower_bound
        )

    if can_compile_persistent and 2 * sequences * heads > sm_count:
        return _run_persistent_kda(
            compiled_persistent,
            q,
            k,
            v,
            g_raw,
            beta_logits,
            A_log,
            dt_bias,
            initial_state,
            cu_seqlens,
            lower_bound,
            sm_count,
        )
    else:
        return _run_cutedsl_kda(
            compiled_cutedsl,
            q,
            k,
            v,
            g_raw,
            beta_logits,
            A_log,
            dt_bias,
            initial_state,
            cu_seqlens,
            sm_count,
            plan,
        )


def cudnn_kda_prefill(
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
    lower_bound: float | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scan token-strided Q/K/V and return a separate native V-major state.

    Args:
        q/k/v: Packed BF16 [1, tokens, heads, 128] activations.
        g_raw: BF16/FP32 gate with the same shape and 16-byte-aligned token strides.
        beta_logits: Raw BF16 [1, tokens, heads] beta projection.
        A_log/dt_bias: FP32 per-head decay and per-channel gate bias.
        initial_state: FP32 [sequences, heads, V, K] gathered cache state.
        cu_seqlens: Device boundaries [sequences + 1].
        lower_bound: Safe-gate bound forwarded to cuDNN.

    Returns:
        Packed output and FP32 final state; initial_state is not modified.
    """
    from flashinfer.cudnn import cudnn_recurrent_kda

    if lower_bound is None:
        raise ValueError("cuDNN KDA requires a safe-gate lower bound")
    if g_raw.stride(-3) * g_raw.element_size() % 16:
        raise ValueError("cuDNN KDA requires 16-byte-aligned gate token strides")
    # A separate output avoids FlashInfer's in-place state copy-back.
    final_state = torch.empty_like(initial_state)
    output, _ = cudnn_recurrent_kda(
        q,
        k,
        v,
        g_raw,
        beta_logits,
        A_log=A_log,
        dt_bias=dt_bias,
        scale=q.shape[-1] ** -0.5,
        initial_state=initial_state,
        output_state=final_state,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        lower_bound=lower_bound,
        cu_seqlens=cu_seqlens,
        beta_is_logit=True,
    )
    return output, final_state
