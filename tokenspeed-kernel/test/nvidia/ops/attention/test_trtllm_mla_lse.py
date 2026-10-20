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


"""trtllm-gen MLA decode with its log-sum-exp, as a draft-tree cascade's prefix."""

import contextlib
import math

import pytest
import torch
from tokenspeed_kernel.ops.attention.mha.flashinfer import (
    trtllm_batch_decode_with_kv_cache_mla,
)
from tokenspeed_kernel.ops.tuning import autotune
from tokenspeed_kernel.platform import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform().is_blackwell, reason="trtllm-gen MLA decode needs Blackwell"
)

LATENT, ROPE, PAGE, HEADS = 512, 64, 64, 16


def test_lse_decode_inside_tuning_fits_the_runtime_workspace():
    """Tuning would sweep up to 8192 requests, each reserving 256 rows of LSE stats
    per head; the LSE call keeps its tactic instead, in the runtime's 256 MiB."""
    gen = torch.Generator().manual_seed(0)
    rows, prefix_lens = 8, [0, 3, 200]
    scale = (128 + ROPE) ** -0.5
    pages = 4
    table = torch.stack(
        [torch.arange(pages) + 1 + b * pages for b in range(len(prefix_lens))]
    ).int()
    cache = torch.randn(
        (len(prefix_lens) * pages + 1) * PAGE, LATENT + ROPE, generator=gen
    )
    cache = cache.bfloat16()
    q = torch.randn(
        len(prefix_lens), rows, HEADS, LATENT + ROPE, generator=gen
    ).bfloat16()
    workspace = torch.zeros(256 << 20, dtype=torch.uint8, device="cuda")
    with autotune(tune_mode=True, tuning_buckets=None, round_up=None):
        out, lse = trtllm_batch_decode_with_kv_cache_mla(
            query=q.cuda(),
            kv_cache=cache.cuda().view(-1, 1, PAGE, LATENT + ROPE),
            workspace_buffer=workspace,
            qk_nope_head_dim=128,
            kv_lora_rank=LATENT,
            qk_rope_head_dim=ROPE,
            block_tables=table.cuda(),
            seq_lens=torch.tensor(prefix_lens, dtype=torch.int32, device="cuda"),
            max_seq_len=pages * PAGE,
            bmm1_scale=scale,
            return_lse=True,
        )
    out = out.float().cpu().view(len(prefix_lens), rows, HEADS, LATENT)
    lse = lse.float().cpu().view(len(prefix_lens), rows, HEADS)
    for b, plen in enumerate(prefix_lens):
        pos = torch.arange(plen)
        keys = cache[(table[b, pos // PAGE].long() * PAGE + pos % PAGE)].float()
        for r in range(rows):
            # Causal: row r sees the first prefix - rows + 1 + r keys.
            covered = plen - rows + 1 + r
            if covered <= 0:
                assert torch.isneginf(lse[b, r]).all()
                continue
            s = q[b, r].float() @ keys[:covered].T * scale
            torch.testing.assert_close(
                out[b, r],
                torch.softmax(s, -1) @ keys[:covered, :LATENT],
                atol=1e-2,
                rtol=1e-2,
            )
            torch.testing.assert_close(
                lse[b, r], torch.logsumexp(s, -1) / math.log(2), atol=1e-3, rtol=1e-3
            )


def test_lse_guard_sees_return_lse_by_position_and_by_keyword(monkeypatch):
    """A positional return_lse enters the untuned scope like a keyword one."""
    from tokenspeed_kernel.ops.attention.mha import flashinfer as wrappers

    entered = []

    @contextlib.contextmanager
    def record(*ops):
        entered.append(ops)
        yield

    monkeypatch.setattr(wrappers, "untuned", record)

    def decode(query, kv_cache, return_lse=False):
        return return_lse

    guarded = wrappers._untuned_with_lse(decode)
    for args, kwargs in (((0, 0, True), {}), ((0, 0), {"return_lse": True})):
        entered.clear()
        assert guarded(*args, **kwargs)
        assert entered == [("trtllm_batch_decode_mla",)]
    entered.clear()
    guarded(0, 0, False)
    guarded(0, 0)
    assert entered == []
