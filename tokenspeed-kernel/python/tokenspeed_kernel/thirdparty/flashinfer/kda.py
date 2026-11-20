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
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run FlashInfer's ordinary CuTe prefill with live packed boundaries.

    Args:
        q/k/v: BF16 [1, capacity, heads, 128], with dense head/features and
            16-byte-aligned bases and token strides. Token rows may be strided.
        g_raw: BF16 or FP32 raw gate with the same shape as q.
        beta_logits: BF16 [1, capacity, heads] raw beta logits.
        A_log/dt_bias: FP32 per-head decay and per-channel gate bias.
        initial_state: FP32 [sequences, heads, V, K] gathered cache state.
        cu_seqlens: Device int64 boundaries [sequences + 1]. Live lengths may
            change on replay and their sum may be less than token capacity.
        lower_bound: Negative safe-gate bound, fixed for the model.

    Returns:
        Packed BF16 output and separate FP32 final state. Input state is
        unchanged; scratch belongs to this call or its CUDA graph pool.
    """
    from cuda.bindings.driver import CUstream
    from flashinfer.kda_kernels import kda_chunked_bt16 as kda

    heads, tokens = q.shape[2], q.shape[1]
    sequences = cu_seqlens.numel() - 1
    compiled = _compile_cutedsl_kda(q.device.index, g_raw.dtype, lower_bound)
    route = kda._route_for_workspace(sequences, heads, q.device, "auto")
    # The engine never reads the decomposition workspace; one chunk supplies
    # the ABI's required views without retaining a process-global GPU cache.
    chunks = (tokens + 15) // 16 + sequences - 1 if route == "decomp" else 1
    workspace = torch.empty(
        kda._decomp_ws_bytes(heads, chunks), dtype=torch.uint8, device=q.device
    )
    scratch = kda._partition_workspace(workspace, heads, chunks)
    cu_chunks = torch.empty(sequences + 1, dtype=torch.int32, device=q.device)
    order = torch.empty(sequences, dtype=torch.int32, device=q.device)
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
        1,
        None,
        cu_chunks,
        1,
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
        kda._device_sm_count(q.device),
        0,
        num_ctas,
        chunks_per_cta,
        q.shape[-1] ** -0.5,
        None,
        None,
        None,
    )
    return out, final_state


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
