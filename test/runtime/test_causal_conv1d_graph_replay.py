"""Graph-replay coverage for the continuous-batch prefill convolution."""

from __future__ import annotations

import os
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

from tokenspeed.runtime.layers.attention.linear.causal_conv1d import (
    _causal_conv1d_fwd_kernel,
)

register_cuda_ci(est_time=30, suite="runtime-1gpu")

_TIMED_LENGTHS = [46, 526, 20]


def _reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    lengths: list[int],
) -> torch.Tensor:
    width = weight.shape[1]
    out = torch.empty_like(x)
    start = 0
    for length in lengths:
        sequence = x[:, start : start + length].float()
        padded = F.pad(sequence, (width - 1, 0))
        acc = sum(
            padded[:, offset : offset + length] * weight[:, offset, None].float()
            for offset in range(width)
        )
        out[:, start : start + length] = F.silu(acc).to(x.dtype)
        start += length
    return out[:, :start]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
def test_v9_timed_partition_replays_changed_inputs() -> None:
    torch.manual_seed(29)
    device = torch.device("cuda")
    dim = 32
    width = 4
    total_tokens = sum(_TIMED_LENGTHS)
    x = torch.randn(
        total_tokens,
        dim,
        dtype=torch.bfloat16,
        device=device,
    ).transpose(0, 1)
    weight = torch.randn(dim, width, dtype=torch.bfloat16, device=device)
    conv_states = torch.zeros(3, dim, width - 1, dtype=x.dtype, device=device)
    cache_indices = torch.arange(3, dtype=torch.int32, device=device)
    has_initial_state = torch.zeros(3, dtype=torch.bool, device=device)
    query_start_loc = torch.tensor(
        [0, *torch.tensor(_TIMED_LENGTHS).cumsum(0).tolist()],
        dtype=torch.int32,
        device=device,
    )
    batch_rows: list[int] = []
    chunk_offsets: list[int] = []
    for row, length in enumerate(_TIMED_LENGTHS):
        for chunk in range((length + 7) // 8):
            batch_rows.append(row)
            chunk_offsets.append(chunk)
    batch_ptr = torch.tensor(batch_rows, dtype=torch.int32, device=device)
    token_chunk_offset = torch.tensor(
        chunk_offsets,
        dtype=torch.int32,
        device=device,
    )
    out = torch.empty_like(x)

    def launch() -> None:
        _causal_conv1d_fwd_kernel[(len(batch_rows), 1)](
            x,
            weight,
            None,
            conv_states,
            cache_indices,
            has_initial_state,
            query_start_loc,
            batch_ptr,
            token_chunk_offset,
            out,
            dim,
            total_tokens,
            conv_states.shape[0],
            0,
            x.stride(0),
            x.stride(1),
            weight.stride(0),
            weight.stride(1),
            conv_states.stride(0),
            conv_states.stride(1),
            conv_states.stride(2),
            0,
            out.stride(0),
            out.stride(1),
            -1,
            HAS_BIAS=False,
            KERNEL_WIDTH=width,
            SILU_ACTIVATION=True,
            HAS_INITIAL_STATES=True,
            HAS_CACHE=True,
            IS_CONTINUOUS_BATCHING=True,
            USE_PAD_SLOT=True,
            NP2_STATELEN=4,
            BLOCK_M=8,
            BLOCK_N=256,
            num_stages=2,
        )

    launch()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()

    x.copy_(torch.randn_like(x))
    conv_states.zero_()
    out.zero_()
    graph.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(
        out,
        _reference(x, weight, _TIMED_LENGTHS),
        atol=2e-2,
        rtol=2e-2,
    )
