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
    _attention_format_signature,
    gdn_decode_mtp,
    gdn_decode_step,
    gdn_verify_fuses_output_norm,
)
from tokenspeed_kernel.ops.attention.kda import KdaOutputNorm  # noqa: E402
from tokenspeed_kernel.selection import select_kernel  # noqa: E402
from tokenspeed_kernel_amd.ops.gfx950.attention.gdn.decode import (  # noqa: E402
    gluon_gdn_decode_gfx950,
)

DEVICE = "cuda"
# Qwen3.8-Flash-Next per TP2 rank: 8 key heads, 24 value heads, 128-wide heads.
H, HV, D = 8, 24, 128
# The same model at TP1.
H_TP1, HV_TP1 = 16, 48


def _packed_inputs(
    batch: int, tokens: int, seed: int = 0, heads: int = H, value_heads: int = HV
):
    """q/k/v as strided views of one packed conv output, a/b as views of one
    projection row, like the runtime."""
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    packed = torch.randn(
        batch * tokens, (2 * heads + value_heads) * D, device=DEVICE, generator=gen
    ).bfloat16()
    q, k, v = packed.split((heads * D, heads * D, value_heads * D), dim=-1)
    gates = torch.randn(
        batch, tokens, 2 * value_heads, device=DEVICE, generator=gen
    ).bfloat16()
    return (
        q.view(batch, tokens, heads, D),
        k.view(batch, tokens, heads, D),
        v.view(batch, tokens, value_heads, D),
        gates[..., value_heads:],
        gates[..., :value_heads],
        torch.randn(value_heads, device=DEVICE, generator=gen),
        torch.randn(value_heads, device=DEVICE, generator=gen),
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


@pytest.mark.parametrize(
    ("op", "value_head_dim", "expected"),
    [
        ("gdn_decode_step", 128, "gluon_gdn_decode_step_gfx950"),
        ("gdn_decode_step", 256, "gluon_gdn_decode_step_gfx950"),
        ("gdn_decode_step", 272, "triton_gdn_decode_step"),
        ("gdn_decode_step", 136, "triton_gdn_decode_step"),
        ("gdn_decode_mtp", 128, "gluon_gdn_decode_mtp_gfx950"),
        ("gdn_decode_mtp", 256, "gluon_gdn_decode_mtp_gfx950"),
        ("gdn_decode_mtp", 272, "triton_gdn_decode_mtp"),
        ("gdn_decode_mtp", 136, "triton_gdn_decode_mtp"),
    ],
)
def test_value_heads_the_launcher_rejects_select_triton(op, value_head_dim, expected):
    """Value heads outside the launcher's multiples of 16 up to 256 route to
    the Triton solution instead of failing in the gfx950 launcher."""
    probe = torch.empty(0, dtype=torch.bfloat16, device="meta")
    traits = {
        "head_dim": D,
        "value_head_dim": value_head_dim,
        "state_dtype": torch.bfloat16,
        "padding_safe": False,
        "strided_state": False,
    }
    if op == "gdn_decode_mtp":
        traits.update(verify_tokens=4, tree=False, output_only=False)
    kernel = select_kernel(
        "attention",
        op,
        _attention_format_signature(q=probe, k=probe, v=probe),
        traits=traits,
    )
    assert kernel.name == expected


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

              pending_replay=None,
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

            pending_replay=None,
        )  # fmt: skip
        gdn_decode_mtp(
            q=q, k=k, v=v, a=a, b=b, A_log=A_log, dt_bias=dt_bias,
            initial_state=pool, initial_state_indices=reads,
            disable_state_update=True, tree_ancestors=None,

            pending_replay=None,
        )  # fmt: skip

    run(1)
    run(16)
    with assert_no_triton_compile(gluon_gdn_decode_gfx950):
        for batch in (2, 3, 5, 8, 13, 17):
            run(batch)


@pytest.mark.parametrize("tokens", [1, 2, 3, 4, 5, 8])
def test_output_only_verify_reads_bf16_pool(tokens):
    """ReplaySSM verify at TP1: BF16 pool read in place, outputs only.

    Outputs must match an fp64 recurrence to BF16 rounding; a negative read
    row starts from a zero state, and the pool is left untouched.
    """
    batch = 6
    q, k, v, a, b, A_log, dt_bias = _packed_inputs(
        batch, tokens, seed=tokens, heads=H_TP1, value_heads=HV_TP1
    )
    gen = torch.Generator(device=DEVICE).manual_seed(100 + tokens)
    pool = (torch.randn(9, HV_TP1, D, D, device=DEVICE, generator=gen) * 0.3).bfloat16()
    committed = pool.clone()
    reads = torch.tensor([3, -1, 0, 8, 5, 1], device=DEVICE, dtype=torch.int32)
    out = gdn_decode_mtp(
        q=q, k=k, v=v, a=a, b=b, A_log=A_log, dt_bias=dt_bias,
        initial_state=pool, initial_state_indices=reads,
        disable_state_update=True, tree_ancestors=None, pending_replay=None,
    )  # fmt: skip
    initial = pool[reads.clamp(min=0).long()].double()
    initial[reads < 0] = 0
    expected, _ = _reference(q, k, v, a, b, A_log, dt_bias, initial)
    # BF16 output rounding is 2^-9 relative; the FP32 math stays far below it.
    torch.testing.assert_close(out.double(), expected, rtol=4e-3, atol=2e-5)
    assert torch.equal(pool, committed)


def test_output_only_verify_far_pool_rows():
    """Pool rows past 2^31 elements: the row offset is computed in int64."""
    tokens, row_elements = 4, HV * D * D
    rows = (2**31) // row_elements + 3
    pool = torch.empty(rows, HV, D, D, device=DEVICE, dtype=torch.bfloat16)
    gen = torch.Generator(device=DEVICE).manual_seed(7)
    reads = torch.tensor([rows - 1, rows - 3], device=DEVICE, dtype=torch.int32)
    pool[reads.long()] = (
        torch.randn(2, HV, D, D, device=DEVICE, generator=gen) * 0.3
    ).bfloat16()
    q, k, v, a, b, A_log, dt_bias = _packed_inputs(2, tokens, seed=11)
    out = gdn_decode_mtp(
        q=q, k=k, v=v, a=a, b=b, A_log=A_log, dt_bias=dt_bias,
        initial_state=pool, initial_state_indices=reads,
        disable_state_update=True, tree_ancestors=None, pending_replay=None,
    )  # fmt: skip
    expected, _ = _reference(q, k, v, a, b, A_log, dt_bias, pool[reads.long()].double())
    torch.testing.assert_close(out.double(), expected, rtol=4e-3, atol=2e-5)


def test_verify_compiles_once_per_token_bucket():
    """T is a runtime value within its power-of-2 bucket (3 and 4 share one)."""
    pool = (torch.randn(8, HV, D, D, device=DEVICE) * 0.05).bfloat16()

    def run(batch: int, tokens: int) -> None:
        q, k, v, a, b, A_log, dt_bias = _packed_inputs(batch, tokens)
        gdn_decode_mtp(
            q=q, k=k, v=v, a=a, b=b, A_log=A_log, dt_bias=dt_bias,
            initial_state=pool,
            initial_state_indices=torch.arange(
                batch, device=DEVICE, dtype=torch.int32
            ),
            disable_state_update=True, tree_ancestors=None, pending_replay=None,
        )  # fmt: skip

    run(1, 4)
    with assert_no_triton_compile(gluon_gdn_decode_gfx950):
        for batch, tokens in ((2, 3), (7, 4), (5, 3)):
            run(batch, tokens)


def test_output_only_verify_fuses_gated_output_norm():
    """The verify applies rmsnorm(o) * w * sigmoid(z) per head, z a strided view."""
    batch, tokens, eps = 5, 4, 1e-6
    q, k, v, a, b, A_log, dt_bias = _packed_inputs(
        batch, tokens, seed=3, heads=H_TP1, value_heads=HV_TP1
    )
    gen = torch.Generator(device=DEVICE).manual_seed(4)
    pool = (
        torch.randn(batch, HV_TP1, D, D, device=DEVICE, generator=gen) * 0.3
    ).bfloat16()
    reads = torch.tensor([2, 0, -1, 4, 1], device=DEVICE, dtype=torch.int32)
    # The gate is a column view of a wider projection row, like z.
    projection = torch.randn(
        batch * tokens, 2 * HV_TP1 * D, device=DEVICE, generator=gen
    ).bfloat16()
    gate = projection[:, HV_TP1 * D :]
    weight = (1 + 0.1 * torch.randn(D, device=DEVICE, generator=gen)).bfloat16()
    assert gdn_verify_fuses_output_norm(
        torch.bfloat16,
        state_dtype=torch.bfloat16,
        traits={
            "head_dim": D,
            "value_head_dim": D,
            "strided_state": False,
            "verify_tokens": tokens,
            "tree": False,
            "output_only": True,
        },
        solution=None,
    )
    out = gdn_decode_mtp(
        q=q, k=k, v=v, a=a, b=b, A_log=A_log, dt_bias=dt_bias,
        initial_state=pool, initial_state_indices=reads,
        disable_state_update=True, tree_ancestors=None, pending_replay=None,
        output_norm=KdaOutputNorm(gate, weight, eps),
    )  # fmt: skip
    initial = pool[reads.clamp(min=0).long()].double()
    initial[reads < 0] = 0
    raw, _ = _reference(q, k, v, a, b, A_log, dt_bias, initial)
    expected = (
        raw
        * torch.rsqrt(raw.square().mean(-1, keepdim=True) + eps)
        * weight.double()
        * torch.sigmoid(gate.double()).view(batch, tokens, HV_TP1, D)
    )
    torch.testing.assert_close(out.double(), expected, rtol=4e-3, atol=2e-4)
