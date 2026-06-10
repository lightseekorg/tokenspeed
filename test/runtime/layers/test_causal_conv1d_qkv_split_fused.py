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

"""Correctness tests for fused causal_conv1d + QKV split kernel."""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.ops.attention.triton.gdn_qkv_split import (
    causal_conv1d_qkv_split_gdn_prefill,
    fused_qkv_split_gdn_prefill,
)

from tokenspeed.runtime.layers.attention.linear.causal_conv1d import (
    causal_conv1d_fn,
)

NQ = NK = NV = 16
HEAD_DIM = 128
CONV_DIM = (NQ + NK + NV) * HEAD_DIM  # 6144
CONV_WIDTH = 4

# (seq_lens_list, description)
SEQ_CONFIGS = [
    ([512], "single_seq_512"),
    ([2048], "single_seq_2048"),
    ([8192], "single_seq_8192"),
    ([256, 512], "varlen_2seq"),
    ([128, 256, 512], "varlen_3seq"),
]


def _make_inputs(seq_lens: list[int], dtype: torch.dtype, with_initial_state: bool = False):
    total_tokens = sum(seq_lens)
    batch = len(seq_lens)

    x = torch.randn(CONV_DIM, total_tokens, dtype=dtype, device="cuda")
    x = x.as_strided(x.shape, (1, CONV_DIM))  # channel-last

    weight = torch.randn(CONV_DIM, CONV_WIDTH, dtype=dtype, device="cuda")

    conv_states = torch.zeros(batch, CONV_DIM, CONV_WIDTH - 1, dtype=dtype, device="cuda")
    cache_indices = torch.arange(batch, dtype=torch.int32, device="cuda")
    has_initial_state = torch.zeros(batch, dtype=torch.bool, device="cuda")
    if with_initial_state:
        has_initial_state[:] = True
        conv_states.uniform_(-0.1, 0.1)

    offsets = [0] + list(torch.tensor(seq_lens).cumsum(0).tolist())
    query_start_loc = torch.tensor(offsets, dtype=torch.int32, device="cuda")
    seq_lens_cpu = seq_lens

    return x, weight, conv_states, cache_indices, has_initial_state, query_start_loc, seq_lens_cpu


def _run_reference(x, weight, conv_states, cache_indices, has_initial_state,
                   query_start_loc, seq_lens_cpu, total_seq_len, dtype):
    """Reference: causal_conv1d_fn + fused_qkv_split_gdn_prefill (2 kernels)."""
    # conv_states is mutated in-place; clone so both paths see the same initial state
    conv_states_ref = conv_states.clone()
    mixed_qkv = causal_conv1d_fn(
        x,
        weight,
        None,
        conv_states_ref,
        query_start_loc,
        cache_indices=cache_indices,
        has_initial_state=has_initial_state,
        activation="silu",
        seq_lens_cpu=seq_lens_cpu,
    ).transpose(0, 1)[:total_seq_len]

    return fused_qkv_split_gdn_prefill(
        mixed_qkv,
        num_q_heads=NQ,
        num_k_heads=NK,
        num_v_heads=NV,
        head_q=HEAD_DIM,
        head_k=HEAD_DIM,
        head_v=HEAD_DIM,
    )


def _run_fused(x, weight, conv_states, cache_indices, has_initial_state,
               query_start_loc, seq_lens_cpu, total_seq_len, dtype):
    conv_states_fused = conv_states.clone()
    return causal_conv1d_qkv_split_gdn_prefill(
        x,
        weight,
        None,
        conv_states_fused,
        query_start_loc,
        seq_lens_cpu,
        num_q_heads=NQ,
        num_k_heads=NK,
        num_v_heads=NV,
        head_q=HEAD_DIM,
        head_k=HEAD_DIM,
        head_v=HEAD_DIM,
        total_seq_len=total_seq_len,
        cache_indices=cache_indices,
        has_initial_state=has_initial_state,
        activation="silu",
    )


@pytest.mark.parametrize("seq_lens,_desc", SEQ_CONFIGS)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("with_initial_state", [False, True])
def test_qkv_correctness(seq_lens, _desc, dtype, with_initial_state):
    torch.manual_seed(0)
    total = sum(seq_lens)
    x, weight, conv_states, cache_indices, has_initial_state, query_start_loc, seq_lens_cpu = (
        _make_inputs(seq_lens, dtype, with_initial_state)
    )

    q_ref, k_ref, v_ref = _run_reference(
        x, weight, conv_states, cache_indices, has_initial_state,
        query_start_loc, seq_lens_cpu, total, dtype,
    )
    q, k, v = _run_fused(
        x, weight, conv_states, cache_indices, has_initial_state,
        query_start_loc, seq_lens_cpu, total, dtype,
    )

    tol = dict(atol=1e-3, rtol=1e-3)
    assert q.shape == q_ref.shape, f"Q shape mismatch: {q.shape} vs {q_ref.shape}"
    assert k.shape == k_ref.shape
    assert v.shape == v_ref.shape
    assert torch.allclose(q, q_ref, **tol), f"Q max diff {(q - q_ref).abs().max():.2e}"
    assert torch.allclose(k, k_ref, **tol), f"K max diff {(k - k_ref).abs().max():.2e}"
    assert torch.allclose(v, v_ref, **tol), f"V max diff {(v - v_ref).abs().max():.2e}"


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_conv_state_update(dtype):
    """conv_states must be updated identically by both paths."""
    torch.manual_seed(1)
    seq_lens = [512]
    total = sum(seq_lens)
    x, weight, conv_states, cache_indices, has_initial_state, query_start_loc, seq_lens_cpu = (
        _make_inputs(seq_lens, dtype, with_initial_state=False)
    )

    conv_states_ref = conv_states.clone()
    conv_states_fused = conv_states.clone()

    causal_conv1d_fn(
        x, weight, None, conv_states_ref, query_start_loc,
        cache_indices=cache_indices, has_initial_state=has_initial_state,
        activation="silu", seq_lens_cpu=seq_lens_cpu,
    )
    causal_conv1d_qkv_split_gdn_prefill(
        x, weight, None, conv_states_fused, query_start_loc, seq_lens_cpu,
        NQ, NK, NV, HEAD_DIM, HEAD_DIM, HEAD_DIM,
        total_seq_len=total,
        cache_indices=cache_indices,
        has_initial_state=has_initial_state,
        activation="silu",
    )

    assert torch.allclose(conv_states_ref, conv_states_fused, atol=1e-3, rtol=1e-3), (
        f"conv_states diverged: max diff {(conv_states_ref - conv_states_fused).abs().max():.2e}"
    )
