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

"""DSA over a dense BF16 KV cache with a per-head learnable sink.

Shapes follow HY V4 (64 heads, 576-wide latent+rope rows, top-2048). The sink is
one extra softmax candidate without a value vector, so the reference appends it
as a logit column and drops its probability from the output.
"""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.ops.attention.dsa import dsa_decode, dsa_prefill
from tokenspeed_kernel.ops.attention.dsa import triton as dsa_triton

HEADS = 64
KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
QK_NOPE_HEAD_DIM = 192
ROW = KV_LORA_RANK + QK_ROPE_HEAD_DIM
TOPK = 2048
SLOTS = 8192
SCALE = ROW**-0.5


def _reference(q, kv, slots, lens, sink):
    out = torch.zeros(q.shape[0], HEADS, KV_LORA_RANK, device=q.device)
    lse = torch.full((q.shape[0], HEADS), float("-inf"), device=q.device)
    for row in range(q.shape[0]):
        selected = slots[row, : int(lens[row])]
        selected = selected[selected >= 0].long()
        if selected.numel() == 0:
            continue
        keys = kv[selected].float()
        logits = q[row].float() @ keys.T * SCALE
        if sink is not None:
            logits = torch.cat([logits, sink[:, None]], dim=1)
        lse[row] = torch.logsumexp(logits, dim=1)
        probs = torch.softmax(logits, dim=1)[:, : selected.numel()]
        out[row] = probs @ keys[:, :KV_LORA_RANK]
    return out, lse


CASES = {
    "short": dict(lens=[1, 2, 5, 13, 20, 33, 64, 7]),
    "empty_rows": dict(lens=[0, 3, 0, 100]),
    "invalid_slots": dict(lens=[2048, 1500, 900, 64, 20, 2048, 300, 1], invalid=0.3),
    "full": dict(lens=[TOPK] * 8),
    "mtp": dict(lens=[10, 11, 500, 501, 2047, 2048, 1, 2], q_len=2),
}


@pytest.mark.parametrize("case", sorted(CASES))
@pytest.mark.parametrize("with_sink", [True, False], ids=["sink", "no_sink"])
@pytest.mark.parametrize("mode", ["decode", "prefill"])
@pytest.mark.parametrize("solution", ["triton", "flashmla"])
def test_dsa_dense_kv_sink(device, require, solution, mode, with_sink, case):
    spec = CASES[case]
    q_len = spec.get("q_len", 1)
    if mode == "prefill" and q_len != 1:
        pytest.skip("prefill rows are single-token")
    if solution == "flashmla" and mode == "decode" and not with_sink:
        pytest.skip("FlashMLA dense-KV decode is selected only with a sink")
    api_name = f"dsa_{mode}"
    require("attention", api_name, solution, torch.bfloat16, "q")

    generator = torch.Generator(device=device).manual_seed(0)
    kv = (torch.randn(SLOTS, ROW, device=device, generator=generator) * 0.5).to(
        torch.bfloat16
    )
    sink = (
        torch.randn(HEADS, device=device, dtype=torch.float32, generator=generator)
        if with_sink
        else None
    )
    rows = len(spec["lens"])
    q = torch.randn(rows, HEADS, ROW, device=device, generator=generator).to(
        torch.bfloat16
    )
    slots = torch.stack(
        [
            torch.randperm(SLOTS, device=device, generator=generator)[:TOPK]
            for _ in range(rows)
        ]
    ).to(torch.int32)
    if spec.get("invalid"):
        mask = torch.rand(slots.shape, device=device, generator=generator)
        slots[mask < spec["invalid"]] = -1
    lens = torch.tensor(spec["lens"], device=device, dtype=torch.int32)

    kwargs = dict(
        q=q,
        kv_cache=kv,
        sparse_kv_cache=None,
        topk_slots=slots,
        topk_lens=lens,
        max_seqlen_k=SLOTS,
        qk_nope_head_dim=QK_NOPE_HEAD_DIM,
        kv_lora_rank=KV_LORA_RANK,
        qk_rope_head_dim=QK_ROPE_HEAD_DIM,
        softmax_scale=SCALE,
        page_size=64,
        return_lse=True,
        solution=solution,
        slot_order="selection",
        attn_sink=sink,
    )
    if mode == "decode":
        out, lse = dsa_decode(q_len_per_req=q_len, **kwargs)
    else:
        out, lse = dsa_prefill(**kwargs)

    ref_out, ref_lse = _reference(q, kv, slots, lens, sink)
    finite = torch.isfinite(ref_lse)
    assert out.shape == (rows, HEADS, KV_LORA_RANK)
    torch.testing.assert_close(out.float(), ref_out, rtol=0.0, atol=2e-2)
    torch.testing.assert_close(lse[finite], ref_lse[finite], rtol=0.0, atol=1e-2)
    # Rows with nothing selected must stay empty even when the sink is set.
    assert torch.isneginf(lse[~finite]).all()
    assert (out[~finite.all(dim=1)] == 0).all()


@pytest.mark.parametrize("packed", [False, True], ids=["dense", "packed"])
@pytest.mark.parametrize("mode", ["decode", "prefill"])
def test_strong_sink_does_not_turn_live_rows_into_empty_rows(
    device, require, packed, mode
):
    """A sink can underflow key mass without making selected slots disappear."""
    require("attention", f"dsa_{mode}", "triton", torch.bfloat16, "q")
    q = torch.zeros(4, 3, ROW, dtype=torch.bfloat16, device=device)
    slots = torch.full((4, TOPK), -1, dtype=torch.int32, device=device)
    slots[0, :2] = torch.tensor([1, 2], dtype=torch.int32, device=device)
    slots[1, :2] = slots[0, :2]  # Valid storage, but outside a zero live length.
    slots[3, 1] = 3  # A valid slot after an invalid one still makes a live row.
    lens = torch.tensor([2, 0, 2, 2], dtype=torch.int32, device=device)
    sink = torch.tensor([200.0, -200.0, 0.0], device=device)
    kv = sparse_kv = None
    if packed:
        # FP8 latent ones, one FP32 scale per 128 values, and BF16 RoPE zeros.
        scale_bytes = 4 * (KV_LORA_RANK // 128)
        sparse_kv = torch.zeros(
            64,
            KV_LORA_RANK + scale_bytes + 2 * QK_ROPE_HEAD_DIM,
            dtype=torch.uint8,
            device=device,
        )
        sparse_kv[:, :KV_LORA_RANK].fill_(0x38)  # E4M3 encoding of 1.0.
        sparse_kv[:, KV_LORA_RANK : KV_LORA_RANK + scale_bytes].view(
            torch.float32
        ).fill_(1.0)
    else:
        kv = torch.zeros(64, ROW, dtype=torch.bfloat16, device=device)
        kv[:, :KV_LORA_RANK].fill_(1.0)
    operation = dsa_decode if mode == "decode" else dsa_prefill
    out, lse = operation(
        q=q,
        kv_cache=kv,
        sparse_kv_cache=sparse_kv,
        topk_slots=slots,
        topk_lens=lens,
        max_seqlen_k=64,
        qk_nope_head_dim=QK_NOPE_HEAD_DIM,
        kv_lora_rank=KV_LORA_RANK,
        qk_rope_head_dim=QK_ROPE_HEAD_DIM,
        softmax_scale=SCALE,
        page_size=64,
        attn_sink=sink,
        return_lse=True,
        slot_order="selection",
        override=f"triton_dsa_{mode}",
    )
    # Every key logit is zero. Positive/negative/zero sinks exercise
    # underflow, almost unattenuated values, and an ordinary softmax.
    for row, num_keys in ((0, 2), (3, 1)):
        log_keys = torch.full_like(sink, float(num_keys)).log()
        expected_lse = torch.logaddexp(log_keys, sink)
        expected_out = (log_keys - expected_lse).exp().bfloat16()
        torch.testing.assert_close(lse[row], expected_lse, atol=1e-5, rtol=0.0)
        torch.testing.assert_close(
            out[row], expected_out[:, None].expand_as(out[row]), atol=0.0, rtol=0.0
        )
    assert torch.isneginf(lse[1:3]).all()
    assert torch.count_nonzero(out[1:3]) == 0


def _portable_dsa_output_call(q, out, packed):
    return dsa_triton._run_dsa(
        q=q,
        kv_cache=None if packed else torch.zeros(64, 128, dtype=torch.bfloat16),
        packed_kv_cache=torch.zeros(64, 132, dtype=torch.uint8) if packed else None,
        topk_slots=torch.zeros(2, 512, dtype=torch.int32),
        topk_lens=torch.ones(2, dtype=torch.int32),
        kv_lora_rank=128,
        qk_rope_head_dim=0,
        softmax_scale=0.1,
        k_scale=1.0,
        sink=None,
        out=out,
        return_lse=True,
    )


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("case", ["strided", "dtype", "count", "device"])
def test_portable_dsa_rejects_invalid_output_before_launch(monkeypatch, packed, case):
    """The host wrapper rejects incompatible storage without launching a kernel."""
    q = torch.ones(2, 2, 128, dtype=torch.bfloat16)
    if case == "strided":
        out = torch.full((4, 128), -3.0, dtype=torch.bfloat16).t()
    elif case == "dtype":
        out = torch.empty(2, 256, dtype=torch.float32)
    elif case == "count":
        out = torch.empty(511, dtype=torch.bfloat16)
    else:
        out = torch.empty(512, dtype=torch.bfloat16, device="meta")

    def forbidden(*args, **kwargs):
        pytest.fail("invalid output reached a kernel launcher")

    monkeypatch.setattr(dsa_triton, "_run_dense_kv", forbidden)
    monkeypatch.setattr(dsa_triton, "_run_packed_kv", forbidden)
    with pytest.raises(ValueError, match="output buffer"):
        _portable_dsa_output_call(q, out, packed)


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("q_dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_portable_dsa_writes_the_callers_output_storage(monkeypatch, packed, q_dtype):
    q = torch.ones(2, 2, 128).to(q_dtype)
    out = torch.full((2, 256), -3.0, dtype=torch.bfloat16)
    expected = torch.arange(512, dtype=torch.float32).bfloat16().view(2, 2, 128)
    expected_lse = torch.zeros(2, 2)
    calls = []

    def compute(*args, **kwargs):
        calls.append(1)
        return expected, expected_lse

    monkeypatch.setattr(dsa_triton, "_run_dense_kv", compute)
    monkeypatch.setattr(dsa_triton, "_run_packed_kv", compute)
    result, lse = _portable_dsa_output_call(q, out, packed)
    assert calls == [1]
    assert result is out and lse is expected_lse
    torch.testing.assert_close(out.view_as(expected), expected, atol=0, rtol=0)
