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

import pytest
import torch
from tokenspeed_kernel.ops.attention import dsv41
from tokenspeed_kernel.ops.attention.dsv4._triton.dcp import normalize_dcp_partials
from tokenspeed_kernel.ops.attention.dsv41 import triton as portable
from tokenspeed_kernel.platform import current_platform


@pytest.mark.parametrize("compact", [False, True])
@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("heads", [8, 64])
@pytest.mark.parametrize("scale", [1.0, 8.0])
def test_selected_attention_partial_lse(native, heads, scale, compact):
    if not torch.cuda.is_available():
        pytest.skip("GPU required")
    if native:
        from tokenspeed_kernel.ops.attention.dsv41 import flash_mla

        if not flash_mla.is_flash_mla_v41_available():
            pytest.skip("native V4.1 FlashMLA unavailable")
        implementation = flash_mla.selected_attention
        # Match the serving cache: Hopper uses V4, Blackwell uses V4.1.
        if current_platform().is_hopper:
            fmt_swa, fmt_global, width_swa, width_global = (
                "swa_v4",
                "global_v4",
                584,
                584,
            )
        else:
            fmt_swa, fmt_global, width_swa, width_global = "swa", "global", 528, 288
    else:
        implementation = portable.selected_attention
        fmt_swa, fmt_global, width_swa, width_global = "swa", "global", 528, 288
    torch.manual_seed(37)
    q = torch.randn(3, heads, 512, device="cuda", dtype=torch.bfloat16) * scale
    sw = torch.zeros(3, 64, width_swa, dtype=torch.uint8, device="cuda")
    gl = torch.zeros(10, 64, width_global, dtype=torch.uint8, device="cuda")
    for cache, fmt in [(sw, fmt_swa), (gl, fmt_global)]:
        rows = torch.randn(
            cache.shape[0] * 64, 512, device="cuda", dtype=torch.bfloat16
        )
        dsv41.cache_scatter(
            rows, cache, torch.arange(rows.shape[0], device="cuda"), fmt
        )
    ss = torch.arange(128, device="cuda").expand(3, -1).clone()
    gs = (torch.arange(512, device="cuda") + 64).expand(3, -1).clone()
    # Holes before/after page boundaries, partial scan lengths, and an empty shard.
    gs[:, 1::3] = -1
    sl = torch.tensor([128, 17, 0], dtype=torch.int32, device="cuda")
    glen = torch.tensor([512, 65, 0], dtype=torch.int32, device="cuda")
    lse = torch.empty(3, heads, dtype=torch.float32, device="cuda")
    used_slots, used_lens = dsv41.compact_slots(gs, glen) if compact else (gs, glen)
    out = implementation(
        q,
        sw,
        ss,
        sl,
        gl,
        used_slots,
        used_lens,
        None,
        512**-0.5,
        None,
        1,
        dsv41.new_attention_schedule() if native else None,
        None,
        None,
        lse=lse,
    )
    valid_s = torch.arange(128, device="cuda")[None, :] < sl[:, None]
    valid_g = (torch.arange(512, device="cuda")[None, :] < glen[:, None]) & (gs >= 0)
    out, lse = normalize_dcp_partials(out, lse, sl, valid_g.sum(-1, dtype=torch.int32))
    kv = torch.cat(
        [
            dsv41.cache_gather(sw, ss, fmt_swa, None),
            dsv41.cache_gather(gl, gs, fmt_global, None),
        ],
        dim=1,
    ).double()
    valid = torch.cat([valid_s, valid_g], dim=1)
    logits = torch.einsum("thd,tkd->thk", q.double(), kv) * 512**-0.5
    logits = logits.masked_fill(~valid[:, None, :], -torch.inf)
    expected_lse = torch.logsumexp(logits, dim=-1)
    weights = logits.softmax(-1).nan_to_num()
    expected = torch.einsum("thk,tkd->thd", weights, kv).to(torch.bfloat16)
    torch.testing.assert_close(lse, expected_lse.float(), rtol=2e-5, atol=2e-4)
    # FlashMLA multiplies V by BF16-rounded probabilities, which peaked
    # logits (scale=8) push just past the portable kernel's bound.
    torch.testing.assert_close(
        out, expected, rtol=0.008, atol=0.008 if native else 0.004
    )
    assert not out[2].any()
    assert torch.isneginf(lse[2]).all()


@pytest.mark.parametrize("width", [0, 1, 65, 512])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_compact_slots_graph(width, dtype):
    if not torch.cuda.is_available():
        pytest.skip("GPU required")
    torch.manual_seed(37)
    backing = torch.randint(-3, 4096, (6, width * 2), device="cuda", dtype=dtype)
    slots = backing[:, ::2]
    slots[:, 1::3] = -1
    lengths = torch.tensor([width, width // 2, 0, 1, width, width], device="cuda")[::1]
    slots[4].fill_(-1)
    dsv41.compact_slots(slots, lengths)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        packed, counts = dsv41.compact_slots(slots, lengths)
    for step in range(3):
        if step:
            lengths.copy_(torch.roll(lengths, 1))
            slots[:, ::4] = -1
        graph.replay()
        cpu = slots.cpu()
        lens = lengths.cpu()
        expected = torch.full_like(cpu, -1)
        expected_counts = []
        for row in range(cpu.shape[0]):
            values = cpu[row, : int(lens[row])]
            values = values[values >= 0]
            expected[row, : values.numel()] = values
            expected_counts.append(values.numel())
        torch.testing.assert_close(packed.cpu(), expected, rtol=0, atol=0)
        assert counts.cpu().tolist() == expected_counts
        cpu_packed, cpu_counts = dsv41.compact_slots(cpu, lens)
        torch.testing.assert_close(cpu_packed, expected, rtol=0, atol=0)
        assert cpu_counts.tolist() == expected_counts


def test_compact_slots_no_shape_recompile():
    if not torch.cuda.is_available():
        pytest.skip("GPU required")
    from utils import assert_no_triton_compile

    def run(rows, width):
        slots = torch.arange(rows * width * 2, device="cuda", dtype=torch.int64).view(
            rows, width * 2
        )[:, ::2]
        lengths = torch.full((rows * 2,), width, device="cuda", dtype=torch.int32)[::2]
        packed, counts = dsv41.compact_slots(slots, lengths)
        torch.testing.assert_close(packed, slots, rtol=0, atol=0)
        assert counts.tolist() == [width] * rows

    run(2, 384)
    with assert_no_triton_compile(portable._compact_slots_kernel):
        run(6, 257)
        run(24, 512)
