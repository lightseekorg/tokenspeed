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

"""The EAGLE-4 draft layer and final norm in one launch, against the
regular path's math with its BF16 roundings."""

from __future__ import annotations

import pytest
import torch
from utils import assert_no_triton_compile, is_cdna4

if not is_cdna4():
    pytest.skip("AMD CDNA4 is required", allow_module_level=True)

from tokenspeed_kernel_amd.ops.gfx950.monokernel.eagle4 import layer  # noqa: E402

HIDDEN, HEADS, HEAD, INTER = 5120, 40, 128, 16384
PAGE, WINDOW, EPS = 64, 4096, 1e-6
POSITIONS = 8192


def _rms(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    x = x.float()
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + EPS) * w.float()


def _bf(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.bfloat16).float()


def _weights(gen: torch.Generator, device) -> dict[str, torch.Tensor]:
    def mat(n: int, k: int) -> torch.Tensor:
        w = torch.randn(n, k, generator=gen) / k**0.5
        return w.to(torch.bfloat16).to(device)

    def gain() -> torch.Tensor:
        return (
            (1 + 0.1 * torch.randn(HIDDEN, generator=gen)).to(torch.bfloat16).to(device)
        )

    return {
        "input_norm": gain(),
        "hidden_norm": gain(),
        "post_norm": gain(),
        "final_norm": gain(),
        "qkv": mat(3 * HIDDEN, 2 * HIDDEN),
        "o_proj": mat(HIDDEN, HIDDEN),
        # Larger gate and up rows keep silu off its near-linear range.
        "gate_up": mat(2 * INTER, HIDDEN) * 2,
        "down": mat(HIDDEN, INTER),
    }


def _rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """NeoX RoPE of ``x [rows, heads, 128]`` as the attention prologue."""
    x1, x2 = x[..., :64], x[..., 64:]
    c, s = cos[:, None, :], sin[:, None, :]
    return torch.cat((x1 * c - x2 * s, x2 * c + x1 * s), dim=-1)


def _reference(w, embeds, hidden, positions, seq_len, slots, table, kc, vc, cos_sin):
    """``(normed, k, v)`` of the regular path (k and v: the rows' BF16 heads)."""
    rows = embeds.shape[0]
    xin = torch.cat(
        (_bf(_rms(embeds, w["input_norm"])), _bf(_rms(hidden, w["hidden_norm"]))), -1
    )
    qkv = _bf(xin @ w["qkv"].float().T)
    q, k, v = (t.view(rows, HEADS, HEAD) for t in qkv.split(HIDDEN, -1))
    cos = cos_sin[positions.long(), :64]
    sin = cos_sin[positions.long(), 64:]
    q, k = _bf(_rope(q, cos, sin)), _bf(_rope(k, cos, sin))
    out = torch.empty(rows, HEADS, HEAD, device=embeds.device)
    for r in range(rows):
        end = int(seq_len[r])
        pos = torch.arange(max(end - WINDOW, 0), end - 1, device=embeds.device)
        at = table[r, pos // PAGE].long() * PAGE + pos % PAGE
        keys = torch.cat((kc[at].float(), k[r][None]), 0)  # [n, heads, 128]
        values = torch.cat((vc[at].float(), v[r][None]), 0)
        scores = torch.einsum("hd,nhd->hn", q[r], keys) / HEAD**0.5
        out[r] = torch.einsum("hn,nhd->hd", scores.softmax(-1), values)
    o = _bf(_bf(out.view(rows, HIDDEN)) @ w["o_proj"].float().T)
    s = o + hidden.float()
    hin = _bf(_rms(s, w["post_norm"]))
    gu = _bf(hin @ w["gate_up"].float().T)
    g, u = gu[:, :INTER], gu[:, INTER:]
    act = _bf(torch.nn.functional.silu(g) * u)
    d = _bf(act @ w["down"].float().T)
    normed = _rms(d + _bf(s), w["final_norm"]).to(torch.bfloat16)
    return normed, k.to(torch.bfloat16), v.to(torch.bfloat16)


def _case(rows: int, seq_lens: list[int], seed: int, device):
    gen = torch.Generator().manual_seed(seed)
    w = _weights(gen, device)
    pages_per_row = (max(seq_lens) + PAGE - 1) // PAGE
    # Every row's pages, shuffled across the pool.
    order = torch.randperm(rows * pages_per_row, generator=gen).to(torch.int32)
    table = order.view(rows, pages_per_row).to(device)
    slot_count = rows * pages_per_row * PAGE
    kc = (
        torch.randn(slot_count, HEADS, HEAD, generator=gen)
        .to(torch.bfloat16)
        .to(device)
    )
    vc = (
        torch.randn(slot_count, HEADS, HEAD, generator=gen)
        .to(torch.bfloat16)
        .to(device)
    )
    seq_len = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    own = seq_len - 1
    slots = (
        table[torch.arange(rows, device=device), own // PAGE].long() * PAGE + own % PAGE
    )
    positions = (own + 3).to(torch.int32)  # positions need not equal the cache index
    freqs = 1.0 / 1e6 ** (torch.arange(0, 64, dtype=torch.float32) / 64)
    angles = torch.arange(POSITIONS, dtype=torch.float32)[:, None] * freqs
    cos_sin = torch.cat((angles.cos(), angles.sin()), -1).to(device)
    embeds = torch.randn(rows, HIDDEN, generator=gen).to(torch.bfloat16).to(device)
    hidden = (
        (3 * torch.randn(rows, HIDDEN, generator=gen)).to(torch.bfloat16).to(device)
    )
    return w, embeds, hidden, positions, seq_len, slots, table, kc, vc, cos_sin


def _launch(packed, case, scratch, step, slot_layer, normed, compile_only=False):
    _, embeds, hidden, positions, seq_len, slots, table, kc, vc, cos_sin = case
    layer.launch_gluon_eagle4_draft_layer_gfx950(
        packed,
        embeds=embeds,
        hidden=hidden,
        normed=normed,
        positions=positions,
        seq_len=seq_len,
        slots=slots,
        page_table=table,
        k_cache=kc,
        v_cache=vc,
        cos_sin=cos_sin,
        scratch=scratch,
        step=step,
        layer=slot_layer,
        trace_buf=None,
        compile_only=compile_only,
    )


@pytest.mark.parametrize(
    "seq_lens",
    [
        [1],  # the row's own key only (6 attention splits, 5 empty)
        [4500, 300],  # 3 splits of a full window and of a short one
        [700, 1, 4096, 4097, 6000, 65],  # windows shorter than, at, past 4096
        [130 + 37 * i for i in range(16)],  # 16 rows (the GEMV's limit)
    ],
)
def test_matches_regular_path(seq_lens):
    device = torch.device("cuda")
    rows = len(seq_lens)
    case = _case(rows, seq_lens, seed=rows, device=device)
    w, embeds, hidden, positions, seq_len, slots, table, kc, vc, cos_sin = case
    want, want_k, want_v = _reference(
        w, embeds, hidden, positions, seq_len, slots, table, kc, vc, cos_sin
    )
    packed = layer.pack_eagle4_draft_layer(w)
    scratch = torch.zeros(
        layer.layer_plan(rows).scratch_pairs, dtype=torch.int64, device=device
    )
    step = torch.zeros(1, dtype=torch.int32, device=device)
    normed = torch.empty(rows, HIDDEN, dtype=torch.bfloat16, device=device)
    # Two launches over one scratch: the second's tags differ from the first's.
    for slot_layer in (0, 1):
        normed.fill_(0)
        _launch(packed, case, scratch, step, slot_layer, normed)
        torch.cuda.synchronize()
        # The partials sum in another order than the reference's matmuls, so
        # a BF16 rounding of a projection can differ by one unit.
        torch.testing.assert_close(normed.float(), want.float(), atol=3e-2, rtol=2e-2)
    torch.testing.assert_close(kc[slots].float(), want_k.float(), atol=2e-2, rtol=1e-2)
    torch.testing.assert_close(vc[slots].float(), want_v.float(), atol=2e-2, rtol=1e-2)


def test_no_recompile_across_steps():
    """The layer slot, the step and the page table's width are runtime
    arguments: serving them takes no new compile."""
    device = torch.device("cuda")
    case = _case(2, [300, 90], seed=7, device=device)
    packed = layer.pack_eagle4_draft_layer(case[0])
    scratch = torch.zeros(
        layer.layer_plan(2).scratch_pairs, dtype=torch.int64, device=device
    )
    step = torch.zeros(1, dtype=torch.int32, device=device)
    normed = torch.empty(2, HIDDEN, dtype=torch.bfloat16, device=device)
    _launch(packed, case, scratch, step, 0, normed)
    wide = list(case)
    wide[6] = torch.cat((case[6], torch.zeros_like(case[6])), 1)  # a wider table
    with assert_no_triton_compile(layer.gluon_eagle4_draft_layer_gfx950):
        for slot_layer in range(1, 7):
            _launch(
                packed,
                tuple(wide) if slot_layer % 2 else case,
                scratch,
                step,
                slot_layer,
                normed,
            )
        step.add_(1)
        _launch(packed, case, scratch, step, 0, normed)
    torch.cuda.synchronize()


def test_layer_plan_rows():
    plan = layer.layer_plan(5)
    assert plan.rows == 5 and plan.grid == 256
    assert [layer.attn_splits(r) for r in (1, 2, 6, 8, 16)] == [6, 3, 1, 3, 2]
    with pytest.raises(ValueError):
        layer.layer_plan(17)
