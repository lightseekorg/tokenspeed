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

from itertools import accumulate

import pytest
import torch
from tokenspeed_kernel.numerics.torch.attention.kda import kda_delta_rule, kda_gate
from utils import is_cdna4, is_cdna5

if is_cdna4():
    from tokenspeed_kernel_amd.ops.gfx950.attention.kda.prefill import (
        launch_gluon_kda_paged_prefill_gfx950 as launch_kda_prefill,
    )
elif is_cdna5():
    from tokenspeed_kernel_amd.ops.gfx1250.attention.kda.prefill import (
        launch_gluon_kda_paged_prefill_gfx1250 as launch_kda_prefill,
    )
else:
    pytest.skip(
        "AMD CDNA4 or CDNA5 is required for Gluon KDA tests", allow_module_level=True
    )


def _reference_sequence(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    raw_g: torch.Tensor,
    beta_logits: torch.Tensor,
    state: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    # V-major: state is [heads, V, K] (value-major), matching the physical
    # recurrent-state pool layout.
    q, k = q.float(), k.float()
    q = q * torch.rsqrt(torch.sum(q * q, dim=-1, keepdim=True) + 1e-6)
    k = k * torch.rsqrt(torch.sum(k * k, dim=-1, keepdim=True) + 1e-6)
    return kda_delta_rule(
        q * q.shape[-1] ** -0.5,
        k,
        v,
        kda_gate(raw_g, a_log, dt_bias, lower_bound=lower_bound),
        beta_logits.float().sigmoid(),
        state,
    )


@pytest.mark.parametrize("lower_bound", [-5.0, None])
def test_kda_prefill_matches_packed_recurrent_reference(
    lower_bound: float | None,
) -> None:
    torch.manual_seed(31)
    lengths = [0, 1, 15, 16, 17, 63, 64, 65]
    total_tokens = sum(lengths)
    heads = 2
    dim = 128
    q = torch.randn(1, total_tokens, heads, dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    raw_g = torch.randn_like(q)
    beta = torch.randn(1, total_tokens, heads, device="cuda", dtype=torch.bfloat16)
    initial_state = torch.randn(
        len(lengths), heads, dim, dim, device="cuda", dtype=torch.float32
    )
    a_log = torch.randn(heads, device="cuda", dtype=torch.float32) * 0.1 - 2.0
    dt_bias = torch.randn(heads, dim, device="cuda", dtype=torch.float32)
    cu_seqlens = torch.tensor(
        [0, *accumulate(lengths)], device="cuda", dtype=torch.int32
    )

    expected_outputs = []
    expected_states = []
    for sequence, (begin, end) in enumerate(
        zip(cu_seqlens.tolist(), cu_seqlens.tolist()[1:])
    ):
        output, state = _reference_sequence(
            q[0, begin:end],
            k[0, begin:end],
            v[0, begin:end],
            raw_g[0, begin:end],
            beta[0, begin:end],
            initial_state[sequence],
            a_log,
            dt_bias,
            lower_bound,
        )
        expected_outputs.append(output)
        expected_states.append(state)

    actual_output, actual_state = launch_kda_prefill(
        q,
        k,
        v,
        raw_g,
        beta,
        a_log,
        dt_bias,
        initial_state=initial_state,
        cu_seqlens=cu_seqlens,
        lower_bound=lower_bound,
    )

    torch.testing.assert_close(
        actual_output[0].float(),
        torch.cat(expected_outputs),
        atol=2e-3,
        rtol=2e-2,
    )
    torch.testing.assert_close(
        actual_state,
        torch.stack(expected_states),
        atol=4e-3,
        rtol=2e-2,
    )
    assert actual_state.dtype == torch.float32
    assert actual_state.shape == (len(lengths), heads, dim, dim)
    assert actual_state.stride()[-2:] == (dim, 1)
