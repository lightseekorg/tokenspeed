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

"""FlashInfer persistent KDA prefill through kda_paged_prefill vs the portable FLA scan.

The FlashInfer solution keeps V-major [N, H, V, K] states while FLA uses
[N, H, K, V]; the reference's states are converted at the comparison boundary.
"""

from __future__ import annotations

from importlib.util import find_spec

import pytest
import torch
from tokenspeed_kernel.ops.attention.kda import kda_paged_prefill
from tokenspeed_kernel.ops.attention.kda.flashinfer import flashinfer_kda_supported

HEADS = 8
DIM = 128
LOWER_BOUND = -5.0
OUTPUT_MAX_ERROR = 2e-3
STATE_MAX_ERROR = 5e-3

requires_flashinfer_kda = pytest.mark.skipif(
    not (
        torch.cuda.is_available()
        and find_spec("fla") is not None
        and flashinfer_kda_supported()
    ),
    reason="requires an sm100/sm103 GPU and flash-linear-attention",
)


def _inputs(lengths: list[int], *, heads: int = HEADS, seed: int = 20261108):
    torch.manual_seed(seed)
    tokens = sum(lengths)
    q = torch.randn(1, tokens, heads, DIM, device="cuda").bfloat16()
    k = torch.randn(1, tokens, heads, DIM, device="cuda").bfloat16()
    v = (torch.randn(1, tokens, heads, DIM, device="cuda") * 0.1).bfloat16()
    g_raw = torch.randn(1, tokens, heads, DIM, device="cuda", dtype=torch.bfloat16)
    # Serving hands beta over as a slice of the merged projection.
    beta = torch.randn(1, tokens, heads + DIM, device="cuda", dtype=torch.bfloat16)[
        ..., -heads:
    ]
    A_log = torch.randn(heads, device="cuda", dtype=torch.float32) * 0.25
    dt_bias = (
        torch.randn(heads, DIM, device="cuda", dtype=torch.float32) * 1.4 - 4.6
    ).clamp_(-9.0, -0.7)
    boundaries_cpu = torch.tensor([0, *lengths], dtype=torch.int64).cumsum(0)
    state = torch.randn(len(lengths), heads, DIM, DIM, device="cuda") * 0.05
    return (q, k, v, g_raw, beta, A_log, dt_bias), state, boundaries_cpu


def _flashinfer(inputs, state, boundaries_cpu, boundaries=None):
    return kda_paged_prefill(
        *inputs,
        initial_state=state,
        cu_seqlens=boundaries_cpu.cuda() if boundaries is None else boundaries,
        cu_seqlens_cpu=boundaries_cpu,
        capacity=None,
        lower_bound=LOWER_BOUND,
        solution="flashinfer",
        recurrent_layout="v_major",
    )


def _assert_matches_portable(lengths: list[int], fresh_rows: tuple[int, ...] = ()):
    from tokenspeed_kernel.ops.attention.kda._triton.fla import kda_chunk_prefill

    inputs, state, boundaries_cpu = _inputs(lengths)
    for row in fresh_rows:
        state[row].zero_()
    expected_out, expected_state = kda_chunk_prefill(
        *inputs,
        initial_state=state.transpose(-1, -2).contiguous(),
        cu_seqlens=boundaries_cpu.cuda(),
        cu_seqlens_cpu=boundaries_cpu,
        lower_bound=LOWER_BOUND,
        beta_is_logit=True,
    )
    actual = _flashinfer(inputs, state.clone(), boundaries_cpu)
    torch.cuda.synchronize()
    out_err = (actual.out.float() - expected_out.float()).abs().max().item()
    state_err = (
        (actual.final_state.float() - expected_state.transpose(-1, -2).float())
        .abs()
        .max()
        .item()
    )
    assert out_err <= OUTPUT_MAX_ERROR, f"output max error {out_err}"
    assert state_err <= STATE_MAX_ERROR, f"state max error {state_err}"


@requires_flashinfer_kda
def test_unaligned_single_sequence_matches_portable() -> None:
    _assert_matches_portable([500])


@requires_flashinfer_kda
def test_fresh_state_matches_portable() -> None:
    _assert_matches_portable([512], fresh_rows=(0,))


@requires_flashinfer_kda
def test_varlen_batch_matches_portable() -> None:
    _assert_matches_portable([130, 517, 64], fresh_rows=(1,))


@requires_flashinfer_kda
@pytest.mark.parametrize("lengths", [[1], [1, 1, 3]])
def test_single_token_sequences_match_portable(lengths) -> None:
    """A fully cached prompt prefills one token; batches can hold only those."""
    _assert_matches_portable(lengths)


@requires_flashinfer_kda
def test_known_boundaries_issue_no_device_sync() -> None:
    """Every KDA layer of a forward shares its boundaries: after the first
    layer plans them, a layer must not synchronize the launch stream."""
    inputs, state, boundaries_cpu = _inputs([700, 300])
    boundaries = boundaries_cpu.cuda()
    _flashinfer(inputs, state.clone(), boundaries_cpu, boundaries)
    layer_state = state.clone()
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        _flashinfer(inputs, layer_state, boundaries_cpu, boundaries)
    finally:
        torch.cuda.set_sync_debug_mode("default")


@requires_flashinfer_kda
def test_rejects_heads_not_divisible_by_eight() -> None:
    inputs, state, boundaries_cpu = _inputs([64], heads=12)
    with pytest.raises(ValueError, match="heads divisible by 8"):
        _flashinfer(inputs, state, boundaries_cpu)
