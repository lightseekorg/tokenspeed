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

"""gfx950 Gluon GDN decode / MTP verify at Qwen3.8 TP2 shapes."""

import pytest
import torch
import torch.nn.functional as F
from tokenspeed_kernel.numerics.torch.attention.gdn import (
    gdn_gate,
    torch_gdn_recurrence,
)
from utils import assert_no_triton_compile, is_cdna4

if not is_cdna4():
    pytest.skip("AMD CDNA4 is required", allow_module_level=True)

from tokenspeed_kernel.ops.attention.gdn import (  # noqa: E402
    gdn_decode_mtp,
    gdn_decode_step,
)
from tokenspeed_kernel_amd.ops.gfx950.attention.gdn.decode import (  # noqa: E402
    gluon_gdn_decode_gfx950,
)

DEVICE = "cuda"
# Qwen3.8-Flash-Next per TP2 rank: 8 key heads, 24 value heads, 128-wide heads.
H, HV, D = 8, 24, 128


def _packed_inputs(batch: int, tokens: int, seed: int = 0):
    """q/k/v as strided views of one packed conv output, like the runtime."""
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    packed = torch.randn(
        batch * tokens, (2 * H + HV) * D, device=DEVICE, generator=gen
    ).bfloat16()
    q, k, v = packed.split((H * D, H * D, HV * D), dim=-1)
    gates = torch.randn(batch, tokens, HV, 2, device=DEVICE, generator=gen)
    return (
        q.view(batch, tokens, H, D),
        k.view(batch, tokens, H, D),
        v.view(batch, tokens, HV, D),
        gates[..., 0].bfloat16(),
        gates[..., 1].bfloat16(),
        torch.randn(HV, device=DEVICE, generator=gen),
        torch.randn(HV, device=DEVICE, generator=gen),
    )


def _reference(q, k, v, a, b, A_log, dt_bias, state):
    """fp64 recurrence; returns outputs [B, T, HV, V] and states [T, B, HV, V, K]."""
    out, states = torch_gdn_recurrence(
        F.normalize(q.double(), dim=-1, eps=1e-6),
        F.normalize(k.double(), dim=-1, eps=1e-6),
        v,
        gdn_gate(a.double(), dt_bias.double(), A_log.double()),
        torch.sigmoid(b.double()),
        state,
        scale=D**-0.5,
        parents=None,
        compute_dtype=torch.float64,
    )
    return out, states.transpose(0, 1)


def test_verify_reads_committed_pool_and_stages_scratch():
    """The runtime's verify: read committed rows in place, write T scratch rows."""
    batch, tokens = 16, 4
    q, k, v, a, b, A_log, dt_bias = _packed_inputs(batch, tokens)
    pool = torch.randn(40, HV, D, D, device=DEVICE) * 0.05
    committed = pool.clone()
    scratch = torch.zeros(batch * (tokens + 1), HV, D, D, device=DEVICE)
    reads = torch.randperm(39, device=DEVICE)[:batch].int() + 1
    rows = (torch.arange(batch, device=DEVICE, dtype=torch.int32) * (tokens + 1))[
        :, None
    ] + torch.arange(1, tokens + 1, device=DEVICE, dtype=torch.int32)
    out = gdn_decode_mtp(
        q=q, k=k, v=v, a=a, b=b, A_log=A_log, dt_bias=dt_bias,
        initial_state=pool, initial_state_indices=reads,
        disable_state_update=False, output_state_indices=rows,
        output_state=scratch, tree_ancestors=None,
    )  # fmt: skip
    expected, states = _reference(q, k, v, a, b, A_log, dt_bias, pool[reads.long()])
    torch.testing.assert_close(out.double(), expected, rtol=2e-2, atol=1e-2)
    for t in range(tokens):
        torch.testing.assert_close(
            scratch[rows[:, t].long()].double(), states[t], rtol=1e-4, atol=2e-5
        )
    assert torch.equal(pool, committed)


def test_decode_and_verify_compile_once_across_batch_sizes():
    """Batch size and request rows are runtime values: no per-batch compiles."""
    pool = torch.randn(80, HV, D, D, device=DEVICE) * 0.05
    scratch = torch.zeros(80, HV, D, D, device=DEVICE)

    def run(batch: int) -> None:
        reads = torch.arange(batch, device=DEVICE, dtype=torch.int32)
        q, k, v, a, b, A_log, dt_bias = _packed_inputs(batch, 1)
        gdn_decode_step(
            q=q, k=k, v=v, a=a, b=b, A_log=A_log, dt_bias=dt_bias,
            initial_state=pool, initial_state_indices=reads,
            output_state_indices=reads + 40,
        )  # fmt: skip
        q, k, v, a, b, A_log, dt_bias = _packed_inputs(batch, 4)
        rows = torch.arange(batch * 4, device=DEVICE, dtype=torch.int32).view(batch, 4)
        gdn_decode_mtp(
            q=q, k=k, v=v, a=a, b=b, A_log=A_log, dt_bias=dt_bias,
            initial_state=pool, initial_state_indices=reads,
            disable_state_update=False, output_state_indices=rows,
            output_state=scratch, tree_ancestors=None,
        )  # fmt: skip
        gdn_decode_mtp(
            q=q, k=k, v=v, a=a, b=b, A_log=A_log, dt_bias=dt_bias,
            initial_state=pool, initial_state_indices=reads,
            disable_state_update=True, tree_ancestors=None,
        )  # fmt: skip

    run(1)
    run(16)
    with assert_no_triton_compile(gluon_gdn_decode_gfx950):
        for batch in (2, 3, 5, 8, 13, 17):
            run(batch)
