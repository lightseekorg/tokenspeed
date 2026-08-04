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

"""CuteDSL KDA prefill drop-in vs the portable FLA scan.

``cutedsl_kda_chunk_prefill`` mirrors ``triton.linear.kda.kda_chunk_prefill``
(same signature, same FLA-native state convention), so every case runs both
and compares directly — including the token-major views and the
state-layout round trip the wrapper performs internally.
"""

from __future__ import annotations

from importlib.util import find_spec

import pytest
import torch
from tokenspeed_kernel.ops.attention.kda.cute_dsl import (
    cutedsl_kda_chunk_prefill,
    is_cutedsl_kda_installed,
)

HEADS = 12
DIM = 128
LOWER_BOUND = -5.0
OUTPUT_MAX_ERROR = 2e-3
STATE_MAX_ERROR = 5e-3

requires_cutedsl_kda = pytest.mark.skipif(
    not (
        torch.cuda.is_available()
        and find_spec("fla") is not None
        and is_cutedsl_kda_installed()
    ),
    reason="requires CUDA, flash-linear-attention, and a tokenspeed-cutedsl-kda build",
)


def _make_inputs(tokens: int, *, seed: int = 20260725):
    torch.manual_seed(seed)
    q = torch.randn(1, tokens, HEADS, DIM, device="cuda").bfloat16()
    k = torch.randn(1, tokens, HEADS, DIM, device="cuda").bfloat16()
    v = (torch.randn(1, tokens, HEADS, DIM, device="cuda") * 0.1).bfloat16()
    g_raw = torch.randn(1, tokens, HEADS, DIM, device="cuda", dtype=torch.bfloat16)
    beta = torch.randn(1, tokens, HEADS, device="cuda", dtype=torch.bfloat16)
    A_log = torch.randn(HEADS, device="cuda", dtype=torch.float32) * 0.25
    dt_bias = (
        torch.randn(HEADS, DIM, device="cuda", dtype=torch.float32) * 1.4 - 4.6
    ).clamp_(-9.0, -0.7)
    return q, k, v, g_raw, beta, A_log, dt_bias


def _assert_matches_portable(
    inputs,
    initial_state,
    cu_seqlens,
    *,
    out_max_error: float = OUTPUT_MAX_ERROR,
    state_max_error: float = STATE_MAX_ERROR,
):
    from tokenspeed_kernel.ops.attention.kda.triton.fla import (
        kda_chunk_prefill,
    )

    kwargs = dict(cu_seqlens=cu_seqlens, lower_bound=LOWER_BOUND, beta_is_logit=True)
    expected_out, expected_state = kda_chunk_prefill(
        *inputs,
        initial_state=None if initial_state is None else initial_state.clone(),
        **kwargs,
    )
    actual_out, actual_state = cutedsl_kda_chunk_prefill(
        *inputs,
        initial_state=None if initial_state is None else initial_state.clone(),
        **kwargs,
    )
    torch.cuda.synchronize()
    out_err = (actual_out.float() - expected_out.float()).abs().max().item()
    state_err = (actual_state.float() - expected_state.float()).abs().max().item()
    assert out_err <= out_max_error, f"output max error {out_err}"
    assert state_err <= state_max_error, f"state max error {state_err}"


@requires_cutedsl_kda
def test_cutedsl_kda_unaligned_single_sequence_matches_portable() -> None:
    """500 tokens (no alignment requirement), FLA-native nonzero state."""
    tokens = 500
    inputs = _make_inputs(tokens)
    initial_state = torch.randn(1, HEADS, DIM, DIM, device="cuda") * 0.05
    cu_seqlens = torch.tensor([0, tokens], device="cuda", dtype=torch.int32)
    _assert_matches_portable(inputs, initial_state, cu_seqlens)


@requires_cutedsl_kda
def test_cutedsl_kda_fresh_state_matches_portable() -> None:
    """initial_state=None starts from zero on both paths."""
    tokens = 512
    inputs = _make_inputs(tokens)
    cu_seqlens = torch.tensor([0, tokens], device="cuda", dtype=torch.int32)
    _assert_matches_portable(inputs, None, cu_seqlens)


@requires_cutedsl_kda
def test_cutedsl_kda_varlen_batch_matches_portable() -> None:
    """Packed varlen sequences with mixed unaligned lengths and states."""
    boundaries = [0, 130, 130 + 517, 130 + 517 + 64]
    tokens = boundaries[-1]
    inputs = _make_inputs(tokens)
    initial_state = torch.randn(3, HEADS, DIM, DIM, device="cuda") * 0.05
    initial_state[1].zero_()  # one fresh sequence in the batch
    cu_seqlens = torch.tensor(boundaries, device="cuda", dtype=torch.int32)
    _assert_matches_portable(inputs, initial_state, cu_seqlens)


@requires_cutedsl_kda
def test_cutedsl_kda_repeated_token_content_stays_bounded() -> None:
    """Period-1 token repetition with no-decay heads.

    Near-identical keys make the intra-chunk inverse depend on
    alternating-sign cancellation; a low-precision inverse blows the state
    up geometrically (we have seen 1e2-scale outputs from kernels with this
    bug on ~1k tokens of repeated words — common agentic traffic, and an
    earlier CuteDSL KDA drop failed exactly here). Guard that the packaged build
    tracks the portable scan on exactly that profile. The
    bounds are blow-up guards, not ulp bounds: on this extreme profile two
    bounded chunked implementations legitimately drift a few 1e-2 apart
    (state magnitude ~9e-2) while a broken inverse lands at 1e+2.
    """
    tokens = 2048
    torch.manual_seed(0)
    base_q = torch.randn(1, HEADS, DIM, device="cuda")
    base_k = torch.randn(1, HEADS, DIM, device="cuda")
    noise = 0.01
    q = (
        base_q.expand(tokens, HEADS, DIM)[None]
        + noise * torch.randn(1, tokens, HEADS, DIM, device="cuda")
    ).bfloat16()
    k = (
        base_k.expand(tokens, HEADS, DIM)[None]
        + noise * torch.randn(1, tokens, HEADS, DIM, device="cuda")
    ).bfloat16()
    v = (torch.randn(1, tokens, HEADS, DIM, device="cuda") * 0.1).bfloat16()
    g_raw = torch.randn(1, tokens, HEADS, DIM, device="cuda", dtype=torch.bfloat16)
    beta = torch.full((1, tokens, HEADS), 2.0, device="cuda", dtype=torch.bfloat16)
    A_log = torch.rand(HEADS, device="cuda") * 0.4 + 0.05  # no-decay heads
    dt_bias = torch.rand(HEADS, DIM, device="cuda") * -7.5 - 1.0
    inputs = (q, k, v, g_raw, beta, A_log, dt_bias)
    cu_seqlens = torch.tensor([0, tokens], device="cuda", dtype=torch.int32)
    _assert_matches_portable(
        inputs, None, cu_seqlens, out_max_error=2e-2, state_max_error=1e-1
    )
