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

"""Real NPU cached Prefill leaf, first-chunk boundary, and failure contracts."""

from unittest.mock import Mock

import pytest
import torch
from tokenspeed_kernel_npu.ops import mla_packed as adapter

pytestmark = pytest.mark.skipif(
    not hasattr(torch, "npu") or not torch.npu.is_available(), reason="requires NPU"
)


def inputs(queries, prefixes, page):
    torch.manual_seed(423)
    lengths = [q + p for q, p in zip(queries, prefixes)]
    pages = (max(lengths) + page - 1) // page + 3
    backing = (
        torch.randn(pages + 4, page, 1, 576, device="npu", dtype=torch.bfloat16) * 0.2
    )
    cache = backing[2:-2]
    table = torch.stack(
        [torch.randperm(pages, device="npu", dtype=torch.int32) for _ in queries]
    )
    q = torch.randn(sum(queries), 32, 576, device="npu", dtype=torch.bfloat16) * 0.2
    return (
        dict(
            q=q,
            kv_cache=cache,
            page_table=table,
            cache_seqlens=torch.tensor(lengths, dtype=torch.int64, device="npu"),
            cu_seqlens_q=torch.tensor(
                [0, *torch.tensor(queries).cumsum(0).tolist()],
                dtype=torch.int32,
                device="npu",
            ),
            cu_seqlens_kv=torch.tensor(
                [0, *torch.tensor(lengths).cumsum(0).tolist()],
                dtype=torch.int32,
                device="npu",
            ),
            max_seqlen_q=max(queries),
            max_seqlen_k=max(lengths),
            qk_nope_head_dim=128,
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            softmax_scale=192**-0.5,
            is_causal=True,
            logit_cap=0.0,
            return_lse=True,
            out=None,
        ),
        backing,
    )


@pytest.mark.parametrize(
    "queries,prefixes", [([1, 7, 17], [0, 65, 65400]), ([4096], [61440])]
)
@pytest.mark.parametrize("page", [64, 128])
@torch.inference_mode()
def test_actual_extend_leaf(monkeypatch, queries, prefixes, page):
    from tokenspeed_kernel_npu.ops import mla

    kwargs, backing = inputs(queries, prefixes, page)
    before = backing.clone()
    op = adapter._packed_prefill_op()
    assert op is not None, "matching packed Prefill extension is required"
    calls = []

    def tracked(*args, **kw):
        assert args[2].data_ptr() == kwargs["kv_cache"].data_ptr()
        calls.append(args[5])
        return op(*args, **kw)

    output = torch.empty(
        kwargs["q"].shape[0], 32, 512, dtype=torch.bfloat16, device="npu"
    )
    kwargs["out"] = output
    monkeypatch.setattr(adapter, "_packed_prefill_op", lambda: tracked)
    actual, lse = mla.mla_extend_with_kvcache(**kwargs)
    torch.npu.synchronize()
    assert actual.data_ptr() == output.data_ptr()
    saved, saved_lse = actual.clone(), lse.clone()
    monkeypatch.setattr(adapter, "_packed_prefill_op", lambda: None)
    expected, expected_lse = mla.mla_extend_with_kvcache(**kwargs)
    torch.npu.synchronize()
    assert len(calls) == 1
    torch.testing.assert_close(saved, expected, rtol=0.01, atol=0.0015)
    torch.testing.assert_close(saved_lse, expected_lse, rtol=1e-5, atol=5e-5)
    assert torch.equal(backing, before)


@torch.inference_mode()
def test_execution_error_is_not_retried(monkeypatch):
    from tokenspeed_kernel_npu.ops import mla

    kwargs, backing = inputs([7], [65], 64)
    before = backing.clone()
    failed = Mock(side_effect=RuntimeError("injected execution error"))
    fallback = Mock()
    monkeypatch.setattr(adapter, "_packed_prefill_op", lambda: failed)
    monkeypatch.setattr(mla.torch_npu, "npu_fused_infer_attention_score", fallback)
    with pytest.raises(RuntimeError, match="injected execution error"):
        mla.mla_extend_with_kvcache(**kwargs)
    failed.assert_called_once()
    fallback.assert_not_called()
    assert torch.equal(backing, before)


@torch.inference_mode()
def test_explicit_first_chunk_keeps_value_width_128(monkeypatch):
    from tokenspeed_kernel_npu.ops import mla

    torch.manual_seed(424)
    tokens, heads = 17, 32
    q = torch.randn(tokens, heads, 192, dtype=torch.bfloat16, device="npu") * 0.2
    k = torch.randn_like(q) * 0.2
    v = torch.randn(tokens, heads, 128, dtype=q.dtype, device=q.device) * 0.2
    cu = torch.tensor([0, tokens], dtype=torch.int32, device=q.device)
    loader = Mock(
        side_effect=AssertionError("explicit first chunk must not call packed Prefill")
    )
    monkeypatch.setattr(adapter, "_packed_prefill_op", loader)
    actual, lse = mla.mla_prefill(
        q, k, v, cu, cu, tokens, tokens, 192**-0.5, None, True, 0.0, True, None
    )
    scores = torch.einsum("thd,shd->hts", q.float(), k.float()) * 192**-0.5
    scores.masked_fill_(
        torch.triu(
            torch.ones(tokens, tokens, dtype=torch.bool, device=q.device), diagonal=1
        ),
        -torch.inf,
    )
    expected = torch.einsum("hts,shd->thd", scores.softmax(-1), v.float())
    torch.testing.assert_close(actual.float(), expected, rtol=0.01, atol=0.0015)
    torch.testing.assert_close(lse, scores.logsumexp(-1).t(), rtol=1e-5, atol=5e-5)
    assert actual.shape[-1] == 128
    loader.assert_not_called()
