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

"""gfx950 Gluon GDN chunked prefill (two-level segment scan).

The segment scan composes chunk transfers in a different order from FLA's
serial chunk scan, so its states and outputs differ by rounding. The
tolerance (see the kernel README): every state (final, checkpoints) and the
output are within 5e-3 relative Frobenius error of the serial Triton scan,
and their error against an FP64 recurrence is at most 1.25x the serial
scan's own.
"""

import pytest
import torch
import torch.nn.functional as F
from utils import assert_no_triton_compile, is_cdna4

if not is_cdna4():
    pytest.skip("AMD CDNA4 is required", allow_module_level=True)

from tokenspeed_kernel.ops.attention.gdn import (  # noqa: E402
    GdnStateCheckpoints,
    _attention_format_signature,
    _chunk_prefill_traits,
    gdn_chunk_prefill,
)
from tokenspeed_kernel.ops.attention.gdn._triton.chunk_delta_h import (  # noqa: E402
    chunk_gated_delta_rule_fwd_h,
)
from tokenspeed_kernel.ops.attention.gdn._triton.chunk_o import (  # noqa: E402
    chunk_fwd_o,
)
from tokenspeed_kernel.ops.attention.gdn._triton.chunk_scaled_dot_kkt import (  # noqa: E402
    chunk_scaled_dot_kkt_fwd,
)
from tokenspeed_kernel.ops.attention.gdn._triton.cumsum import (  # noqa: E402
    chunk_local_cumsum,
)
from tokenspeed_kernel.ops.attention.gdn._triton.l2norm import (  # noqa: E402
    l2norm_fwd,
)
from tokenspeed_kernel.ops.attention.gdn._triton.solve_tril import (  # noqa: E402
    solve_tril,
)
from tokenspeed_kernel.ops.attention.gdn._triton.wy_fast import (  # noqa: E402
    recompute_w_u_fwd,
)
from tokenspeed_kernel.selection import select_kernel  # noqa: E402
from tokenspeed_kernel_amd.ops.gfx950.attention.gdn.chunk_prefill import (  # noqa: E402
    gdn_chunk_prefill_segment_chunks,
    gluon_gdn_chunk_prefill_carry_gfx950,
    gluon_gdn_chunk_prefill_gfx950,
    gluon_gdn_chunk_prefill_transfer_gfx950,
    launch_gluon_gdn_chunk_prefill_gfx950,
)

DEVICE = "cuda"
D = 128
# Qwen3.8-Flash-Next at TP1: 16 key heads, 48 value heads.
H, HV = 16, 48
# The serial-scan tolerance of every state and output (relative Frobenius).
REL_TOL = 5e-3
# Allowed growth of the error against an FP64 recurrence over the serial scan's.
FP64_ERROR_RATIO = 1.25


def _rel(actual: torch.Tensor, expected: torch.Tensor) -> float:
    actual, expected = actual.double(), expected.double()
    return ((actual - expected).norm() / expected.norm()).item()


def _inputs(lens, *, heads=H, value_heads=HV, decay=0.02, state_dtype, seed=0):
    """Raw GDN inputs with weak decay (long memory: the hardest case for a
    scan that composes many chunk transfers)."""
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    tokens = sum(lens)
    q = torch.randn(1, tokens, heads, D, device=DEVICE, generator=gen).bfloat16()
    k = torch.randn(1, tokens, heads, D, device=DEVICE, generator=gen).bfloat16()
    v = torch.randn(1, tokens, value_heads, D, device=DEVICE, generator=gen).bfloat16()
    g = -torch.rand(1, tokens, value_heads, device=DEVICE, generator=gen) * decay
    beta = torch.rand(1, tokens, value_heads, device=DEVICE, generator=gen).bfloat16()
    h0 = torch.randn(len(lens), value_heads, D, D, device=DEVICE, generator=gen)
    cu = torch.tensor([0, *torch.tensor(lens).cumsum(0).tolist()], dtype=torch.int32)
    return q, k, v, g, beta, (h0 * 0.1).to(state_dtype), cu.to(DEVICE)


def _checkpoints(lens, every, guard_rows=0):
    counts = [n // every for n in lens]
    cu_starts = torch.tensor([0, *torch.tensor(counts).cumsum(0).tolist()])
    # NaN rows show a checkpoint the scan failed to store (or, in leading
    # guard rows, one it stored out of range).
    rows = guard_rows + sum(counts)
    states = torch.full((rows, HV, D, D), float("nan"), device=DEVICE)
    return GdnStateCheckpoints(
        every, states[guard_rows:], cu_starts.to(DEVICE, torch.int32)
    )


def _wy_inputs(q, k, v, g, beta, cu):
    """FLA's WY preparation: normalized q/k, cumulative gates, w and u."""
    q, k = l2norm_fwd(q), l2norm_fwd(k)
    g = chunk_local_cumsum(g, chunk_size=64, cu_seqlens=cu)
    a = chunk_scaled_dot_kkt_fwd(
        k=k, beta=beta, g_cumsum=g, cu_seqlens=cu, output_dtype=torch.float32
    )
    a = solve_tril(A=a, cu_seqlens=cu, output_dtype=k.dtype)
    w, u = recompute_w_u_fwd(k=k, v=v, beta=beta, A=a, g_cumsum=g, cu_seqlens=cu)
    return q, k, w, u, g


def test_gdn_chunk_prefill_selects_the_segment_scan_for_128_wide_heads():
    probe = torch.empty(0, dtype=torch.bfloat16, device="meta")
    signature = _attention_format_signature(q=probe, k=probe, v=probe)

    def selected(output_h, checkpoints, head_dim=D):
        traits = _chunk_prefill_traits(
            head_dim, head_dim, H, HV, output_h, True, checkpoints
        )
        return select_kernel(
            "attention", "gdn_chunk_prefill", signature, traits=traits
        ).name

    assert selected(False, False) == "gluon_gdn_chunk_prefill_gfx950"
    assert selected(False, True) == "gluon_gdn_chunk_prefill_gfx950"
    # Per-chunk states never leave the chip; other head dims are not compiled.
    assert selected(True, False) == "triton_gdn_chunk_prefill"
    assert selected(False, False, head_dim=64) == "triton_gdn_chunk_prefill"


@pytest.mark.parametrize(
    "lens, every",
    [
        ([300], 64),  # one segment: the serial fused scan
        ([8192], 1024),  # five segments
        ([3000, 1000, 4192], 256),  # mixed lengths, partial last chunks
        ([300, 77, 2000, 5], 128),  # short sequences beside a split one
        ([2048, 2048, 2048, 2048], 512),  # a segment per sequence
    ],
)
@pytest.mark.parametrize("state_dtype", [torch.bfloat16, torch.float32])
def test_gdn_chunk_prefill_matches_the_serial_scan(lens, every, state_dtype):
    q, k, v, g, beta, h0, cu = _inputs(lens, state_dtype=state_dtype)

    def run(solution):
        checkpoints = _checkpoints(lens, every)
        result = gdn_chunk_prefill(
            q, k, v, g, beta, scale=None, initial_state=h0, cu_seqlens=cu,
            qk_l2norm=True, state_checkpoints=checkpoints, solution=solution,
        )  # fmt: skip
        return result.out, result.final_state, checkpoints.states

    out, final, states = run("gluon")
    ref_out, ref_final, ref_states = run("triton")
    assert not states.isnan().any()
    assert _rel(out, ref_out) < REL_TOL
    assert _rel(final, ref_final) < REL_TOL
    assert _rel(states, ref_states) < REL_TOL


def _fp64_scan(q, k, w, u, g, h0, lens, scale):
    """FP64 chunk recurrence on WY inputs: (o [T, HV, V], K-last final states)."""
    rep = w.shape[2] // q.shape[2]
    out = torch.zeros(sum(lens), w.shape[2], D, dtype=torch.float64, device=DEVICE)
    finals = []
    bos = 0
    for n, length in enumerate(lens):
        h = h0[n].double().transpose(-1, -2)  # [HV, K, V]
        for t0 in range(0, length, 64):
            rows = slice(bos + t0, bos + min(t0 + 64, length))
            qc = q[0, rows].double().repeat_interleave(rep, 1).transpose(0, 1)
            kc = k[0, rows].double().repeat_interleave(rep, 1).transpose(0, 1)
            wc = w[0, rows].double().transpose(0, 1)
            uc = u[0, rows].double().transpose(0, 1)
            gc = g[0, rows].double().transpose(0, 1)  # [HV, BT]
            v_new = uc - wc @ h
            att = torch.tril(
                (qc @ kc.transpose(1, 2)) * (gc[:, :, None] - gc[:, None, :]).exp()
            )
            out[rows] = (
                ((qc @ h) * gc.exp()[:, :, None] + att @ v_new) * scale
            ).transpose(0, 1)
            g_last = gc[:, -1]
            decay = (g_last[:, None] - gc).exp()[:, :, None]
            h = h * g_last.exp()[:, None, None] + kc.transpose(1, 2) @ (v_new * decay)
        finals.append(h.transpose(-1, -2))
        bos += length
    return out, torch.stack(finals)


@pytest.mark.parametrize("lens", [[4096], [2500, 1600]])
def test_gdn_chunk_prefill_error_against_fp64_stays_that_of_the_serial_scan(lens):
    # Fewer heads: more segments per sequence (one program per head and
    # segment fills the GPU), the most transfers composed.
    heads, value_heads = 4, 12
    q, k, v, g, beta, h0, cu = _inputs(
        lens, heads=heads, value_heads=value_heads, decay=0.005,
        state_dtype=torch.float32,
    )  # fmt: skip
    q, k, w, u, g = _wy_inputs(q, k, v, g, beta, cu)
    scale = D**-0.5
    out, final = launch_gluon_gdn_chunk_prefill_gfx950(
        q, k, w, u, g, initial_state=h0, cu_seqlens=cu, scale=scale,
        checkpoint_states=None, checkpoint_cu_starts=None,
        checkpoint_every_n_tokens=None,
    )  # fmt: skip
    h, v_new, ref_final = chunk_gated_delta_rule_fwd_h(
        k=k, w=w, u=u, g=g, initial_state=h0.transpose(-1, -2).contiguous(),
        output_final_state=True, cu_seqlens=cu,
    )  # fmt: skip
    ref_out = chunk_fwd_o(q=q, k=k, v=v_new, h=h, g=g, scale=scale, cu_seqlens=cu)
    exact_out, exact_final = _fp64_scan(q, k, w, u, g, h0, lens, scale)

    serial_out = _rel(ref_out[0], exact_out)
    serial_final = _rel(ref_final.transpose(-1, -2), exact_final)
    assert _rel(out[0], exact_out) <= FP64_ERROR_RATIO * serial_out
    assert _rel(final, exact_final) <= FP64_ERROR_RATIO * serial_final


def test_gdn_chunk_prefill_does_not_recompile_across_lengths():
    def call(lens, every):
        q, k, v, g, beta, h0, cu = _inputs(lens, state_dtype=torch.bfloat16)
        gdn_chunk_prefill(
            q, k, v, g, beta, scale=None, initial_state=h0, cu_seqlens=cu,
            qk_l2norm=True,
            state_checkpoints=None if every is None else _checkpoints(lens, every),
        )  # fmt: skip

    # Warm both checkpoint specializations on a split call.
    call([4096], None)
    call([4096], 512)
    kernels = (
        gluon_gdn_chunk_prefill_transfer_gfx950,
        gluon_gdn_chunk_prefill_carry_gfx950,
        gluon_gdn_chunk_prefill_gfx950,
    )
    with assert_no_triton_compile(*kernels):
        for lens, every in (
            ([64], None),
            ([1], 64),
            ([8192], 256),
            ([700, 4100, 33], 64),
            ([2048] * 4, None),
            ([5000, 17], 1024),
        ):
            call(lens, every)
    torch.cuda.synchronize()


def test_gdn_chunk_prefill_offsets_past_int32():
    """A sequence whose u/w/o offsets pass 2**31 elements: only its own rows
    are touched (the call starts there), over several segments so the
    transfer and carry passes run too."""
    length = 4096 + 700
    # First token whose value-head row offset exceeds 2**31 elements.
    start = (2**31 // (HV * D) // 64 + 1) * 64
    tokens = start + length
    lens = [length]
    q, k, v, g, beta, h0, cu = _inputs(lens, state_dtype=torch.bfloat16)
    q, k, w, u, g = _wy_inputs(q, k, v, g, beta, cu)
    every = 256
    scale = D**-0.5

    def far(x):
        big = torch.empty(1, tokens, *x.shape[2:], dtype=x.dtype, device=DEVICE)
        big[:, start:] = x
        return big

    # A guard row before the checkpoints: nothing is stored out of range.
    checkpoints = _checkpoints(lens, every, guard_rows=1)
    far_out, far_final = launch_gluon_gdn_chunk_prefill_gfx950(
        far(q), far(k), far(w), far(u), far(g), initial_state=h0,
        cu_seqlens=cu + start, scale=scale,
        checkpoint_states=checkpoints.states,
        checkpoint_cu_starts=checkpoints.cu_starts,
        checkpoint_every_n_tokens=every,
    )  # fmt: skip
    near = _checkpoints(lens, every)
    out, final = launch_gluon_gdn_chunk_prefill_gfx950(
        q, k, w, u, g, initial_state=h0, cu_seqlens=cu, scale=scale,
        checkpoint_states=near.states, checkpoint_cu_starts=near.cu_starts,
        checkpoint_every_n_tokens=every,
    )  # fmt: skip
    # The far call has more slots and longer segments: the same tolerance.
    assert _rel(far_out[:, start:], out) < REL_TOL
    assert _rel(far_final, final) < REL_TOL
    assert not checkpoints.states.isnan().any()
    assert checkpoints.states.untyped_storage().nbytes() > checkpoints.states.nbytes
    guard = torch.empty(0, device=DEVICE).set_(checkpoints.states.untyped_storage())
    assert guard[: HV * D * D].isnan().all()
    assert _rel(checkpoints.states, near.states) < REL_TOL


def test_segment_length_follows_the_call():
    def seg(tokens, seqs, every=None):
        return gdn_chunk_prefill_segment_chunks(tokens, seqs, HV, 256, every)

    # Short calls, and calls of many short sequences, scan unsplit.
    assert seg(512, 1) == 8
    assert seg(4 * 2048, 4) == 32
    # A long sequence splits so that about one program per CU runs.
    assert seg(8192, 1) == 26
    assert seg(4096, 1) == 13
    # A length within 25% that divides or is a multiple of the checkpoint
    # interval (in chunks) is preferred.
    assert seg(8192, 1, every=2048) == 32
    assert seg(8192, 1, every=64 * 13) == 26
    assert seg(8192, 1, every=64 * 7) == 28
