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
    q_f = q.float()
    k_f = k.float()
    v_f = v.float()
    g_f = g.float()
    beta_f = beta.float()

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
        state = initial_state[seq_idx].float().clone()
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

    return out.to(q.dtype), torch.stack(final_states, dim=0).to(initial_state.dtype)


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

    ref_out, ref_state = _fla_chunk_gated_delta_rule()(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=q.shape[-1] ** -0.5,
        initial_state=initial_state.clone(),
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        head_first=False,
        use_qk_l2norm_in_kernel=True,
    )

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
