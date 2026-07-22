# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from tokenspeed_kernel import (
    GdnCheckpointLayout,
    GdnChunkPrefillResult,
    gdn_chunk_prefill,
    gdn_decode_mtp,
    gdn_decode_step,
)


def _fla_chunk_gated_delta_rule():
    from tokenspeed_kernel.ops.attention.triton.linear.chunk import (
        chunk_gated_delta_rule,
    )

    return chunk_gated_delta_rule


def _make_inputs(*, device: str, dtype: torch.dtype, seq_len: int = 130):
    torch.manual_seed(0)
    num_q_heads = 16
    num_v_heads = 32
    head_dim = 128
    q = torch.randn(1, seq_len, num_q_heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(1, seq_len, num_q_heads, head_dim, device=device, dtype=dtype)
    v = torch.randn(1, seq_len, num_v_heads, head_dim, device=device, dtype=dtype)
    beta = torch.rand(1, seq_len, num_v_heads, device=device, dtype=dtype).sigmoid()
    g = F.logsigmoid(
        torch.rand(1, seq_len, num_v_heads, device=device, dtype=torch.float32)
    )
    initial_state = (
        torch.randn(1, num_v_heads, head_dim, head_dim, device=device, dtype=dtype)
        * 0.1
    )
    cu_seqlens = torch.tensor([0, seq_len], device=device, dtype=torch.int32)
    return q, k, v, g, beta, initial_state, cu_seqlens


def _torch_l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x_float = x.float()
    return (
        x_float * torch.rsqrt(x_float.square().sum(dim=-1, keepdim=True).clamp_min(eps))
    ).to(x.dtype)


def _torch_gdn_chunk_prefill_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    scale: float,
    initial_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Independent token-by-token recurrence, used as an oracle.

    ``initial_state``/the returned final state follow ``gdn_chunk_prefill``'s
    public K-last ``[N, Hv, V, K]`` contract (matches the runtime's SSM state
    pool); this reference's own math is plain ``[N, Hv, K, V]`` (K row, V
    col), so it transposes in/out at its boundary, mirroring the production
    Triton wrapper (``triton_gdn_chunk_prefill``).
    """
    q_f = q.float()
    k_f = k.float()
    v_f = v.float()
    g_f = g.float()
    beta_f = beta.float()
    initial_state_kv = initial_state.transpose(-2, -1)

    batch, total_tokens, num_q_heads, _ = q.shape
    assert batch == 1
    num_v_heads = v.shape[2]
    head_v_dim = v.shape[-1]
    group_size = num_v_heads // num_q_heads

    out = torch.empty(
        (1, total_tokens, num_v_heads, head_v_dim),
        device=q.device,
        dtype=torch.float32,
    )
    final_states = []
    starts = cu_seqlens[:-1].to(torch.int64).tolist()
    ends = cu_seqlens[1:].to(torch.int64).tolist()

    for seq_idx, (start, end) in enumerate(zip(starts, ends, strict=True)):
        state = initial_state_kv[seq_idx].float().clone()
        for token_idx in range(start, end):
            for value_head in range(num_v_heads):
                qk_head = value_head // group_size
                q_t = q_f[0, token_idx, qk_head]
                k_t = k_f[0, token_idx, qk_head]
                v_t = v_f[0, token_idx, value_head]
                state_h = torch.exp(g_f[0, token_idx, value_head]) * state[value_head]
                delta = beta_f[0, token_idx, value_head] * (v_t - k_t @ state_h)
                state_h = state_h + k_t[:, None] * delta[None, :]

                out[0, token_idx, value_head] = scale * (q_t @ state_h)
                state[value_head] = state_h
        final_states.append(state)

    final_state_kv = torch.stack(final_states, dim=0).to(initial_state.dtype)
    return out.to(q.dtype), final_state_kv.transpose(-2, -1)


def test_gdn_chunk_prefill_triton_matches_torch_reference(device: str, require):
    # The Triton wrapper should match an independent token-by-token recurrence.
    require("attention", "gdn_chunk_prefill", "triton", torch.bfloat16, "q")

    torch.manual_seed(1234)
    seq_len = 130
    num_q_heads = 16
    num_v_heads = 32
    head_dim = 128
    dtype = torch.bfloat16
    q = torch.randn(1, seq_len, num_q_heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(1, seq_len, num_q_heads, head_dim, device=device, dtype=dtype)
    v = torch.randn(1, seq_len, num_v_heads, head_dim, device=device, dtype=dtype) * 0.5
    g = F.logsigmoid(
        torch.randn(1, seq_len, num_v_heads, device=device, dtype=torch.float32)
    )
    beta = torch.rand(1, seq_len, num_v_heads, device=device, dtype=dtype).sigmoid()
    initial_state = (
        torch.randn(1, num_v_heads, head_dim, head_dim, device=device, dtype=dtype)
        * 0.01
    )
    cu_seqlens = torch.tensor([0, seq_len], device=device, dtype=torch.int32)
    scale = head_dim**-0.5

    result = gdn_chunk_prefill(
        q,
        k,
        v,
        g,
        beta,
        scale=scale,
        initial_state=initial_state.clone(),
        cu_seqlens=cu_seqlens,
        qk_l2norm=True,
        output_final_state=True,
        solution="triton",
    )
    assert isinstance(result, GdnChunkPrefillResult)
    assert result.h_layout is GdnCheckpointLayout.NONE
    ref_out, ref_state = _torch_gdn_chunk_prefill_reference(
        _torch_l2norm(q),
        _torch_l2norm(k),
        v,
        g,
        beta,
        scale=scale,
        initial_state=initial_state.clone(),
        cu_seqlens=cu_seqlens,
    )

    torch.testing.assert_close(
        result.out.float(), ref_out.float(), rtol=2e-2, atol=2e-2
    )
    torch.testing.assert_close(
        result.final_state.float(),
        ref_state.float(),
        rtol=2e-2,
        atol=3e-2,
    )


def test_gdn_chunk_prefill_triton_matches_torch_reference_varlen(device: str, require):
    # Varlen cu_seqlens should reset recurrent state independently per sequence.
    require("attention", "gdn_chunk_prefill", "triton", torch.bfloat16, "q")

    torch.manual_seed(4321)
    seq_lens = [63, 67]
    total_tokens = sum(seq_lens)
    num_q_heads = 4
    num_v_heads = 8
    head_dim = 128
    dtype = torch.bfloat16
    q = torch.randn(1, total_tokens, num_q_heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(1, total_tokens, num_q_heads, head_dim, device=device, dtype=dtype)
    v = (
        torch.randn(1, total_tokens, num_v_heads, head_dim, device=device, dtype=dtype)
        * 0.5
    )
    g = F.logsigmoid(
        torch.randn(1, total_tokens, num_v_heads, device=device, dtype=torch.float32)
    )
    beta = torch.rand(
        1, total_tokens, num_v_heads, device=device, dtype=dtype
    ).sigmoid()
    initial_state = (
        torch.randn(
            len(seq_lens), num_v_heads, head_dim, head_dim, device=device, dtype=dtype
        )
        * 0.01
    )
    cu_seqlens = torch.tensor([0, *torch.tensor(seq_lens).cumsum(0)], device=device)
    cu_seqlens = cu_seqlens.to(torch.int32)
    scale = head_dim**-0.5

    result = gdn_chunk_prefill(
        q,
        k,
        v,
        g,
        beta,
        scale=scale,
        initial_state=initial_state.clone(),
        cu_seqlens=cu_seqlens,
        qk_l2norm=True,
        output_final_state=True,
        solution="triton",
    )
    assert isinstance(result, GdnChunkPrefillResult)
    assert result.h_layout is GdnCheckpointLayout.NONE
    ref_out, ref_state = _torch_gdn_chunk_prefill_reference(
        _torch_l2norm(q),
        _torch_l2norm(k),
        v,
        g,
        beta,
        scale=scale,
        initial_state=initial_state.clone(),
        cu_seqlens=cu_seqlens,
    )

    torch.testing.assert_close(
        result.out.float(), ref_out.float(), rtol=2e-2, atol=2e-2
    )
    torch.testing.assert_close(
        result.final_state.float(),
        ref_state.float(),
        rtol=2e-2,
        atol=3e-2,
    )


@pytest.mark.parametrize("solution", ["triton", "flashinfer"])
def test_gdn_chunk_prefill_matches_fla_reference(device: str, solution: str, require):
    # Each selectable backend should match the FLA reference output/state contract.
    require("attention", "gdn_chunk_prefill", solution, torch.bfloat16, "q")

    q, k, v, g, beta, initial_state, cu_seqlens = _make_inputs(
        device=device,
        dtype=torch.bfloat16,
    )
    result = gdn_chunk_prefill(
        q,
        k,
        v,
        g,
        beta,
        scale=q.shape[-1] ** -0.5,
        initial_state=initial_state.clone(),
        cu_seqlens=cu_seqlens,
        qk_l2norm=True,
        output_final_state=True,
        solution=solution,
    )
    assert isinstance(result, GdnChunkPrefillResult)
    assert result.h_layout is GdnCheckpointLayout.NONE

    # FLA's own native state layout is [N, Hv, K, V] (V-last); gdn_chunk_prefill's
    # public contract is K-last [N, Hv, V, K] (matches the runtime pool +
    # flashinfer), so transpose at this reference's boundary (mirrors
    # triton_gdn_chunk_prefill's own in/out transpose).
    ref_out, ref_state_kv = _fla_chunk_gated_delta_rule()(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=q.shape[-1] ** -0.5,
        initial_state=initial_state.clone().transpose(-2, -1).contiguous(),
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        head_first=False,
        use_qk_l2norm_in_kernel=True,
    )
    ref_state = ref_state_kv.transpose(-2, -1)

    assert result.out.shape == ref_out.shape
    assert result.final_state.shape == ref_state.shape
    # The FLA implementation is nondeterministic due to atomics; mean error is
    # the stable signal, while max error can be noisy on a few elements.
    assert (result.out.float() - ref_out.float()).abs().mean() < 1e-3
    assert (result.final_state.float() - ref_state.float()).abs().mean() < 1e-3


@pytest.mark.parametrize("solution", ["triton", "flashinfer"])
def test_gdn_chunk_prefill_output_h_contract(device: str, solution: str, require):
    # output_h exposes backend-native checkpoint layouts used by hybrid GDN caching.
    require("attention", "gdn_chunk_prefill", solution, torch.bfloat16, "q")

    q, k, v, g, beta, initial_state, cu_seqlens = _make_inputs(
        device=device,
        dtype=torch.bfloat16,
    )
    result = gdn_chunk_prefill(
        q,
        k,
        v,
        g,
        beta,
        scale=q.shape[-1] ** -0.5,
        initial_state=initial_state,
        cu_seqlens=cu_seqlens,
        qk_l2norm=True,
        output_final_state=True,
        output_h=True,
        solution=solution,
    )
    assert isinstance(result, GdnChunkPrefillResult)

    if solution == "triton":
        assert result.h_layout is GdnCheckpointLayout.FLA
        assert result.h is not None
        assert result.h_cu_starts is None
        assert result.h.shape == (1, 3, 32, 128, 128)
    else:
        assert result.h_layout is GdnCheckpointLayout.FLASHINFER
        assert result.h is not None
        assert result.h_cu_starts is not None
        assert result.h.shape == (2, 32, 128, 128)
        torch.testing.assert_close(
            result.h_cu_starts, torch.tensor([0, 2], device=device)
        )

    assert result.out.shape == v.shape
    assert result.final_state.shape == initial_state.shape


# ===-----------------------------------------------------------------------===#
# GDN decode / MTP (K-last, gdn_decode_step / gdn_decode_mtp)
# ===-----------------------------------------------------------------------===#


def _torch_softplus_gate(
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    A_log: torch.Tensor,
    softplus_beta: float = 1.0,
    softplus_threshold: float = 20.0,
) -> torch.Tensor:
    x = a + dt_bias
    beta_x = softplus_beta * x
    softplus_x = torch.where(
        beta_x <= softplus_threshold,
        (1.0 / softplus_beta) * torch.log1p(torch.exp(beta_x)),
        x,
    )
    return -torch.exp(A_log) * softplus_x


def _torch_gdn_decode_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state_klast: torch.Tensor,
    scale: float,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Independent token-by-token oracle for gdn_decode_step/gdn_decode_mtp.

    q, k: [B, T, H, K]; v, a, b: [B, T, HV, V] / [B, T, HV]; state_klast is
    K-last ``[B, HV, V, K]`` -- the same public contract the kernels use, so
    callers don't need to transpose anything at this reference's boundary.

    Returns ``(out [B, T, HV, V], per_step_states)`` where ``per_step_states``
    is a length-T list of K-last ``[B, HV, V, K]`` snapshots, one per
    processed step (post-update).
    """
    B, T, H, K = q.shape
    HV = v.shape[2]
    group = HV // H
    q_f, k_f, v_f, a_f, b_f = (x.float() for x in (q, k, v, a, b))
    A_log_f, dt_bias_f = A_log.float(), dt_bias.float()
    state = state_klast.float().transpose(-2, -1).clone()  # -> [B, HV, K, V]
    out = torch.zeros(B, T, HV, v.shape[-1], dtype=torch.float32, device=q.device)
    per_step_states = []
    for t in range(T):
        g = _torch_softplus_gate(a_f[:, t], dt_bias_f, A_log_f)
        beta = torch.sigmoid(b_f[:, t])
        for bi in range(B):
            for hv in range(HV):
                qh = hv // group
                q_t = q_f[bi, t, qh]
                k_t = k_f[bi, t, qh]
                qn = q_t / torch.sqrt((q_t * q_t).sum() + 1e-6)
                kn = k_t / torch.sqrt((k_t * k_t).sum() + 1e-6)
                st = state[bi, hv] * torch.exp(g[bi, hv])
                delta = beta[bi, hv] * (v_f[bi, t, hv] - kn @ st)
                st = st + kn[:, None] * delta[None, :]
                out[bi, t, hv] = scale * (qn @ st)
                state[bi, hv] = st
        per_step_states.append(state.transpose(-2, -1).clone())
    return out, per_step_states


def _make_decode_inputs(
    *,
    device: str,
    dtype: torch.dtype,
    T: int,
    batch: int = 4,
    num_q_heads: int = 4,
    num_v_heads: int = 8,
    head_dim: int = 128,
    pool_size: int = 16,
    state_dtype: torch.dtype | None = None,
    parameter_dtype: torch.dtype = torch.float32,
    parameter_requires_grad: bool = False,
):
    torch.manual_seed(7)
    q = torch.randn(batch, T, num_q_heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch, T, num_q_heads, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch, T, num_v_heads, head_dim, device=device, dtype=dtype) * 0.5
    a = torch.randn(batch, T, num_v_heads, device=device, dtype=torch.float32)
    b = torch.randn(batch, T, num_v_heads, device=device, dtype=torch.float32)
    A_log = torch.randn(num_v_heads, device=device, dtype=parameter_dtype)
    dt_bias = torch.randn(num_v_heads, device=device, dtype=parameter_dtype)
    if parameter_requires_grad:
        A_log = torch.nn.Parameter(A_log)
        dt_bias = torch.nn.Parameter(dt_bias)
    # K-last pool [pool_size, HV, V, K], small values to keep the recurrence stable.
    state_dtype = dtype if state_dtype is None else state_dtype
    pool = (
        torch.randn(
            pool_size,
            num_v_heads,
            head_dim,
            head_dim,
            device=device,
            dtype=state_dtype,
        )
        * 0.02
    )
    return q, k, v, a, b, A_log, dt_bias, pool


@pytest.mark.parametrize("solution", ["triton", "flashinfer"])
@pytest.mark.parametrize(
    ("state_dtype", "parameter_dtype", "parameter_requires_grad"),
    [
        (torch.bfloat16, torch.float32, False),
        (torch.float32, torch.bfloat16, True),
        (torch.float32, torch.float32, True),
    ],
)
def test_gdn_decode_step_matches_torch_reference(
    device: str,
    solution: str,
    state_dtype: torch.dtype,
    parameter_dtype: torch.dtype,
    parameter_requires_grad: bool,
    require,
):
    # T=1 decode step, in-place state update (output_state_indices=None).
    require("attention", "gdn_decode_step", solution, torch.bfloat16, "q")

    q, k, v, a, b, A_log, dt_bias, pool = _make_decode_inputs(
        device=device,
        dtype=torch.bfloat16,
        T=1,
        state_dtype=state_dtype,
        parameter_dtype=parameter_dtype,
        parameter_requires_grad=parameter_requires_grad,
    )
    read_idx = torch.tensor([1, 3, 5, 7], device=device, dtype=torch.int32)
    scale = q.shape[-1] ** -0.5

    pool_copy = pool.clone()
    out = gdn_decode_step(
        q,
        k,
        v,
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        b=b,
        initial_state=pool_copy,
        initial_state_indices=read_idx,
        scale=scale,
        use_qk_l2norm=True,
        solution=solution,
    )

    ref_out, ref_states = _torch_gdn_decode_reference(
        q, k, v, a, b, A_log, dt_bias, pool[read_idx], scale
    )

    assert out.shape == v.shape
    torch.testing.assert_close(
        out.float(), ref_out.to(out.dtype).float(), rtol=2e-2, atol=2e-2
    )
    torch.testing.assert_close(
        pool_copy[read_idx].float(),
        ref_states[0].to(pool.dtype).float(),
        rtol=2e-2,
        atol=3e-2,
    )
    # Rows never referenced by read_idx must stay untouched.
    untouched = torch.tensor(
        [i for i in range(pool.shape[0]) if i not in read_idx.tolist()],
        device=device,
    )
    torch.testing.assert_close(pool_copy[untouched], pool[untouched])


@pytest.mark.parametrize("solution", ["triton", "flashinfer"])
def test_gdn_decode_step_output_state_indices_remap(
    device: str, solution: str, require
):
    # Dual-index paging: write lands on output_state_indices, not the read row.
    require("attention", "gdn_decode_step", solution, torch.bfloat16, "q")

    q, k, v, a, b, A_log, dt_bias, pool = _make_decode_inputs(
        device=device, dtype=torch.bfloat16, T=1
    )
    read_idx = torch.tensor([1, 3, 5, 7], device=device, dtype=torch.int32)
    out_idx = torch.tensor([8, 9, -1, 10], device=device, dtype=torch.int32)
    scale = q.shape[-1] ** -0.5

    pool_copy = pool.clone()
    out = gdn_decode_step(
        q,
        k,
        v,
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        b=b,
        initial_state=pool_copy,
        initial_state_indices=read_idx,
        output_state_indices=out_idx,
        scale=scale,
        use_qk_l2norm=True,
        solution=solution,
    )

    # The read rows must never be mutated when output_state_indices is given.
    torch.testing.assert_close(pool_copy[read_idx], pool[read_idx])

    ref_out, ref_states = _torch_gdn_decode_reference(
        q, k, v, a, b, A_log, dt_bias, pool[read_idx], scale
    )
    torch.testing.assert_close(
        out.float(), ref_out.to(out.dtype).float(), rtol=2e-2, atol=2e-2
    )
    for bi in range(q.shape[0]):
        oi = int(out_idx[bi])
        if oi < 0:
            continue
        torch.testing.assert_close(
            pool_copy[oi].float(),
            ref_states[0][bi].to(pool.dtype).float(),
            rtol=2e-2,
            atol=3e-2,
        )


@pytest.mark.parametrize("solution", ["triton", "flashinfer"])
def test_gdn_decode_step_padding_index_is_isolated(device: str, solution: str, require):
    # -1 (CUDA-graph padding) rows must not corrupt any other batch entry's
    # read or write, regardless of how each backend treats the padding row
    # itself (skip vs sacrificial-row redirect are both valid per-kernel).
    require("attention", "gdn_decode_step", solution, torch.bfloat16, "q")

    q, k, v, a, b, A_log, dt_bias, pool = _make_decode_inputs(
        device=device, dtype=torch.bfloat16, T=1
    )
    read_idx = torch.tensor([1, -1, 5, 7], device=device, dtype=torch.int32)
    scale = q.shape[-1] ** -0.5

    pool_copy = pool.clone()
    out = gdn_decode_step(
        q,
        k,
        v,
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        b=b,
        initial_state=pool_copy,
        initial_state_indices=read_idx,
        scale=scale,
        use_qk_l2norm=True,
        solution=solution,
    )

    valid = [i for i, r in enumerate(read_idx.tolist()) if r >= 0]
    valid_rows = read_idx[valid]
    valid_idx = torch.tensor(valid, device=device)
    ref_out, ref_states = _torch_gdn_decode_reference(
        q[valid_idx],
        k[valid_idx],
        v[valid_idx],
        a[valid_idx],
        b[valid_idx],
        A_log,
        dt_bias,
        pool[valid_rows],
        scale,
    )
    torch.testing.assert_close(
        out[valid_idx].float(), ref_out.to(out.dtype).float(), rtol=2e-2, atol=2e-2
    )
    torch.testing.assert_close(
        pool_copy[valid_rows].float(),
        ref_states[0].to(pool.dtype).float(),
        rtol=2e-2,
        atol=3e-2,
    )
    # Rows unrelated to any (valid or padding) index stay untouched. Row 0 is
    # excluded because FlashInfer's bf16 path redirects -1 padding there. The
    # radix runtime cannot use that convention because its row 0 may be live;
    # it routes bf16 state to the Triton solution instead.
    referenced = {int(x) for x in valid_rows.tolist()} | {0}
    untouched = torch.tensor(
        [i for i in range(pool.shape[0]) if i not in referenced], device=device
    )
    torch.testing.assert_close(pool_copy[untouched], pool[untouched])


@pytest.mark.parametrize("solution", ["triton", "flashinfer"])
@pytest.mark.parametrize(
    ("state_dtype", "parameter_dtype", "parameter_requires_grad"),
    [
        (torch.bfloat16, torch.float32, False),
        (torch.float32, torch.bfloat16, True),
        (torch.float32, torch.float32, True),
    ],
)
def test_gdn_decode_mtp_intermediate_states_buffer_matches_reference(
    device: str,
    solution: str,
    state_dtype: torch.dtype,
    parameter_dtype: torch.dtype,
    parameter_requires_grad: bool,
    require,
):
    # T>1 MTP verify: every step's post-update state must land in
    # intermediate_states_buffer[i_n, step], and the pool itself must stay
    # untouched (disable_state_update=True, the runtime's target_verify mode).
    require("attention", "gdn_decode_mtp", solution, torch.bfloat16, "q")

    T = 3
    q, k, v, a, b, A_log, dt_bias, pool = _make_decode_inputs(
        device=device,
        dtype=torch.bfloat16,
        T=T,
        state_dtype=state_dtype,
        parameter_dtype=parameter_dtype,
        parameter_requires_grad=parameter_requires_grad,
    )
    read_idx = torch.tensor([1, 3, 5, 7], device=device, dtype=torch.int32)
    scale = q.shape[-1] ** -0.5
    batch, _, num_v_heads, head_dim = v.shape

    scratch = torch.empty(
        batch, T, num_v_heads, head_dim, head_dim, device=device, dtype=pool.dtype
    )
    pool_copy = pool.clone()
    out = gdn_decode_mtp(
        q,
        k,
        v,
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        b=b,
        initial_state=pool_copy,
        initial_state_indices=read_idx,
        scale=scale,
        disable_state_update=True,
        use_qk_l2norm=True,
        intermediate_states_buffer=scratch,
        solution=solution,
    )

    # disable_state_update=True: the read rows must never be mutated.
    torch.testing.assert_close(pool_copy[read_idx], pool[read_idx])

    ref_out, ref_states = _torch_gdn_decode_reference(
        q, k, v, a, b, A_log, dt_bias, pool[read_idx], scale
    )
    assert out.shape == v.shape
    torch.testing.assert_close(
        out.float(), ref_out.to(out.dtype).float(), rtol=2e-2, atol=2e-2
    )
    for t in range(T):
        torch.testing.assert_close(
            scratch[:, t].float(),
            ref_states[t].to(pool.dtype).float(),
            rtol=2e-2,
            atol=3e-2,
        )


@pytest.mark.parametrize("solution", ["triton", "flashinfer"])
@pytest.mark.parametrize("state_dtype", [torch.bfloat16, torch.float32])
def test_gdn_decode_mtp_output_state_indices_scatter_matches_reference(
    device: str,
    solution: str,
    state_dtype: torch.dtype,
    require,
):
    # FlashInfer 0.6.15 and the Triton fallback scatter every post-token state
    # directly into the scheduler-provided pool row, avoiding a dense B*T state
    # scratch buffer plus a follow-up PyTorch scatter.
    require("attention", "gdn_decode_mtp", solution, torch.bfloat16, "q")

    T = 3
    q, k, v, a, b, A_log, dt_bias, pool = _make_decode_inputs(
        device=device,
        dtype=torch.bfloat16,
        T=T,
        pool_size=24,
        state_dtype=state_dtype,
    )
    read_idx = torch.tensor([1, 3, 5, 7], device=device, dtype=torch.int32)
    output_idx = torch.arange(8, 20, device=device, dtype=torch.int32).view(4, T)
    scale = q.shape[-1] ** -0.5

    pool_copy = pool.clone()
    out = gdn_decode_mtp(
        q,
        k,
        v,
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        b=b,
        initial_state=pool_copy,
        initial_state_indices=read_idx,
        output_state_indices=output_idx,
        scale=scale,
        disable_state_update=False,
        use_qk_l2norm=True,
        solution=solution,
    )

    ref_out, ref_states = _torch_gdn_decode_reference(
        q, k, v, a, b, A_log, dt_bias, pool[read_idx], scale
    )
    torch.testing.assert_close(
        out.float(), ref_out.to(out.dtype).float(), rtol=2e-2, atol=2e-2
    )
    # Per-token scatter must not overwrite the read rows.
    torch.testing.assert_close(pool_copy[read_idx], pool[read_idx])
    for t in range(T):
        torch.testing.assert_close(
            pool_copy[output_idx[:, t].long()].float(),
            ref_states[t].to(pool.dtype).float(),
            rtol=2e-2,
            atol=3e-2,
        )


@pytest.mark.parametrize(
    ("solution", "state_dtype"),
    [
        ("triton", torch.bfloat16),
        ("triton", torch.float32),
        ("flashinfer", torch.float32),
    ],
)
def test_gdn_decode_mtp_padding_indices_skip_state_writes(
    device: str, solution: str, state_dtype: torch.dtype, require
):
    require("attention", "gdn_decode_mtp", solution, torch.bfloat16, "q")

    T = 3
    q, k, v, a, b, A_log, dt_bias, pool = _make_decode_inputs(
        device=device,
        dtype=torch.bfloat16,
        T=T,
        pool_size=24,
        state_dtype=state_dtype,
    )
    read_idx = torch.tensor([1, -1, 5, 7], device=device, dtype=torch.int32)
    output_idx = torch.tensor(
        [[8, 9, 10], [-1, -1, -1], [11, 12, 13], [14, 15, 16]],
        device=device,
        dtype=torch.int32,
    )
    scale = q.shape[-1] ** -0.5

    pool_copy = pool.clone()
    out = gdn_decode_mtp(
        q,
        k,
        v,
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        b=b,
        initial_state=pool_copy,
        initial_state_indices=read_idx,
        output_state_indices=output_idx,
        scale=scale,
        disable_state_update=False,
        use_qk_l2norm=True,
        solution=solution,
    )

    valid_batch = torch.tensor([0, 2, 3], device=device)
    valid_read = read_idx[valid_batch]
    ref_out, ref_states = _torch_gdn_decode_reference(
        q[valid_batch],
        k[valid_batch],
        v[valid_batch],
        a[valid_batch],
        b[valid_batch],
        A_log,
        dt_bias,
        pool[valid_read],
        scale,
    )
    torch.testing.assert_close(
        out[valid_batch].float(),
        ref_out.to(out.dtype).float(),
        rtol=2e-2,
        atol=2e-2,
    )
    valid_output = output_idx[valid_batch]
    for t in range(T):
        torch.testing.assert_close(
            pool_copy[valid_output[:, t].long()].float(),
            ref_states[t].to(pool.dtype).float(),
            rtol=2e-2,
            atol=3e-2,
        )

    written = set(valid_output.flatten().tolist())
    untouched = torch.tensor(
        [i for i in range(pool.shape[0]) if i not in written], device=device
    )
    torch.testing.assert_close(pool_copy[untouched], pool[untouched])


@pytest.mark.parametrize("solution", ["triton", "flashinfer"])
def test_gdn_decode_mtp_disable_state_update_false_writes_back(
    device: str, solution: str, require
):
    # disable_state_update=False: the final step's state writes back to the
    # SAME row (initial_state_indices), matching gdn_decode_mtp's contract.
    require("attention", "gdn_decode_mtp", solution, torch.bfloat16, "q")

    T = 2
    q, k, v, a, b, A_log, dt_bias, pool = _make_decode_inputs(
        device=device, dtype=torch.bfloat16, T=T
    )
    read_idx = torch.tensor([1, 3, 5, 7], device=device, dtype=torch.int32)
    scale = q.shape[-1] ** -0.5

    pool_copy = pool.clone()
    gdn_decode_mtp(
        q,
        k,
        v,
        A_log=A_log,
        a=a,
        dt_bias=dt_bias,
        b=b,
        initial_state=pool_copy,
        initial_state_indices=read_idx,
        scale=scale,
        disable_state_update=False,
        use_qk_l2norm=True,
        solution=solution,
    )

    _, ref_states = _torch_gdn_decode_reference(
        q, k, v, a, b, A_log, dt_bias, pool[read_idx], scale
    )
    torch.testing.assert_close(
        pool_copy[read_idx].float(),
        ref_states[-1].to(pool.dtype).float(),
        rtol=2e-2,
        atol=3e-2,
    )
