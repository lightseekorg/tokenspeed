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
from tokenspeed_kernel.ops.attention.dsa import dsa_decode, dsa_prefill
from tokenspeed_kernel.ops.attention.dsv4._triton.dcp import (
    combine_dcp_attention_partials,
    dispatch_dcp_attention_partials,
)
from tokenspeed_kernel.ops.kvcache.flashmla import set_flashmla_kv_buffer
from tokenspeed_kernel.platform import current_platform
from tokenspeed_kernel.selection import NoKernelFoundError

pytestmark = pytest.mark.skipif(
    not current_platform().is_hopper,
    reason="Hopper FlashMLA sparse attention",
)


def _case(*, packed, rope_dim, heads, topk):
    torch.manual_seed(193)
    num_slots = max(256, (topk + 63) // 64 * 64)
    keys = torch.randn(
        (num_slots, 1, 512 + rope_dim), device="cuda", dtype=torch.bfloat16
    )
    query = torch.randn((6, heads, 512 + rope_dim), device="cuda", dtype=torch.bfloat16)
    slots = torch.full((6, topk), -1, device="cuda", dtype=torch.int32)
    slots[0, :4] = torch.tensor([0, 63, 64, 127], device="cuda")
    slots[1, :3] = torch.tensor([17, 128, 193], device="cuda")
    slots[2, :3] = torch.tensor([64, 65, 193], device="cuda")
    slots[3, :2] = torch.tensor([70, 130], device="cuda")
    # The fourth row has valid indices beyond its length; the fifth has no
    # indices despite a nonzero length; the sixth has indices but zero length.
    slots[5, :4] = slots[0, :4]
    lengths = torch.tensor([4, 3, 3, 1, 2, 0], device="cuda", dtype=torch.int32)
    if topk > 2048:
        query[0].zero_()
        slots[0] = torch.arange(topk, device="cuda", dtype=torch.int32)
        lengths[0] = topk
        # Each non-tile-aligned tail contributes enough to expose truncation.
        for offset in range(topk - 2048):
            keys[2048 + offset, :, :512] = 64 * 2**offset
    if packed:
        cache = torch.zeros((num_slots, 1, 656), device="cuda", dtype=torch.uint8)
        set_flashmla_kv_buffer(
            cache,
            torch.arange(num_slots, device="cuda", dtype=torch.int64),
            keys[..., :512],
            keys[..., 512:],
            sanitize=False,
            write_mask=None,
        )
        # Read the public byte format without using the production gather.
        quantized = (
            cache[:, 0, :512]
            .view(torch.float8_e4m3fn)
            .float()
            .reshape(num_slots, 4, 128)
        )
        scales = cache[:, 0, 512:528].view(torch.float32).bfloat16()
        latent = (quantized * scales[..., None]).bfloat16().reshape(num_slots, 512)
        reference_kv = torch.cat((latent, keys[:, 0, 512:]), dim=-1)
    else:
        cache = keys
        reference_kv = keys[:, 0]
    args = dict(
        q=query,
        kv_cache=None if packed else cache,
        sparse_kv_cache=cache if packed else None,
        topk_slots=slots,
        topk_lens=lengths,
        max_seqlen_k=num_slots,
        qk_nope_head_dim=256 if rope_dim == 0 else 128,
        kv_lora_rank=512,
        qk_rope_head_dim=rope_dim,
        softmax_scale=(512 + rope_dim) ** -0.5,
        page_size=64,
        return_lse=True,
        solution="flashmla",
        slot_order="selection",
    )
    return args, reference_kv


def _reference(args, keys, sink):
    query, slots, lengths = args["q"], args["topk_slots"], args["topk_lens"]
    valid = (slots >= 0) & (
        torch.arange(slots.shape[1], device=slots.device)[None, :] < lengths[:, None]
    )
    selected = keys.float()[slots.clamp_min(0).long()]
    scores = (
        torch.einsum("thd,tkd->thk", query.float(), selected) * args["softmax_scale"]
    ).masked_fill(~valid[:, None, :], -torch.inf)
    lse = torch.logsumexp(scores, dim=-1)
    if sink is not None:
        lse = torch.logaddexp(lse, sink[None, :])
    probabilities = (scores - lse[..., None]).exp().nan_to_num()
    output = torch.einsum("thk,tkd->thd", probabilities, selected[..., :512])
    return output, lse


def _sink(heads):
    sink = torch.linspace(-2, 8, heads, device="cuda", dtype=torch.float32)
    sink[0] = -torch.inf
    sink[-1] = torch.inf
    return sink


def _assert_reference(actual, lse, args, keys, sink):
    expected, expected_lse = _reference(args, keys, sink)
    torch.testing.assert_close(actual.float(), expected, atol=0.015, rtol=0.015)
    torch.testing.assert_close(lse, expected_lse, atol=2e-3, rtol=2e-3)
    assert actual.dtype == torch.bfloat16 and lse.dtype == torch.float32
    assert not actual[4:].any()
    if sink is not None:
        assert not actual[:, -1].any()
        torch.testing.assert_close(lse[4:], sink.expand(2, -1), rtol=0, atol=0)
    else:
        assert torch.isneginf(lse[4:]).all()


@pytest.mark.parametrize(
    "packed,phase,q_len,rope_dim,heads,topk",
    [
        (False, "prefill", 1, 64, 8, 512),
        (False, "decode", 1, 64, 72, 512),
        (False, "decode", 3, 64, 8, 2051),
        (True, "prefill", 1, 0, 72, 2051),
        (True, "decode", 1, 64, 8, 512),
        (True, "decode", 3, 64, 72, 512),
    ],
)
@pytest.mark.parametrize("with_sink", [False, True])
def test_sparse_attention_sink_matches_reference(
    packed, phase, q_len, rope_dim, heads, topk, with_sink
):
    args, keys = _case(packed=packed, rope_dim=rope_dim, heads=heads, topk=topk)
    attention = dsa_prefill if phase == "prefill" else dsa_decode
    if phase == "decode":
        args["q_len_per_req"] = q_len
    sink = _sink(heads) if with_sink else None
    if not packed and phase == "decode" and sink is None:
        # Adding sink support must not select FlashMLA for the existing
        # BF16 no-sink decode route. Its native numerical reference remains
        # available through the already-supported sparse prefill operation.
        with pytest.raises(NoKernelFoundError):
            attention(**args, attn_sink=None)
        attention = dsa_prefill
        args.pop("q_len_per_req")
    actual, lse = attention(**args, attn_sink=sink)
    _assert_reference(actual, lse, args, keys, sink)
    supplied_out = torch.empty_like(actual)
    returned = attention(
        **dict(args, return_lse=False), attn_sink=sink, out=supplied_out
    )
    assert returned is supplied_out
    torch.testing.assert_close(returned, actual, atol=0, rtol=0)
    if sink is not None:
        # A -inf sink is the no-sink limit, including after head padding.
        if not packed and phase == "decode":
            prefill_args = {
                key: value for key, value in args.items() if key != "q_len_per_req"
            }
            no_sink, _ = dsa_prefill(**prefill_args)
        else:
            no_sink, _ = attention(**args)
        torch.testing.assert_close(actual[:, 0], no_sink[:, 0], atol=0, rtol=0)


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("phase,q_len", [("prefill", 1), ("decode", 3)])
def test_sparse_attention_sink_cuda_graph_reloads_inputs(packed, phase, q_len):
    args, keys = _case(packed=packed, rope_dim=64, heads=8, topk=512)
    attention = dsa_prefill if phase == "prefill" else dsa_decode
    if phase == "decode":
        args["q_len_per_req"] = q_len
    sink = _sink(8)
    # Compile and allocate native metadata before capturing the graph.
    attention(**args, attn_sink=sink)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual, lse = attention(**args, attn_sink=sink)
    args["q"].mul_(0.5)
    args["topk_slots"][:4, :2] = torch.tensor([9, 65], device="cuda")
    args["topk_lens"][:4].fill_(2)
    sink[1:-1].add_(2)
    graph.replay()
    _assert_reference(actual, lse, args, keys, sink)
    args["topk_lens"].zero_()
    graph.replay()
    assert not actual.any()
    torch.testing.assert_close(lse, sink.expand_as(lse), rtol=0, atol=0)


@pytest.mark.parametrize("packed", [False, True])
def test_sparse_context_partials_apply_sink_once(packed):
    args, keys = _case(packed=packed, rope_dim=64, heads=8, topk=512)
    sink = _sink(8)
    degree = 4
    slots = args["topk_slots"]
    records = []
    for rank in range(degree):
        # Keep one shard empty to test the reduction identity as well.
        owned = (slots >= 0) & ((slots // 64) % (degree - 1) == rank)
        partial_args = dict(args, topk_slots=torch.where(owned, slots, -1))
        partial_attention = dsa_prefill
        if packed:
            partial_attention = dsa_decode
            partial_args["q_len_per_req"] = 3
        partial, lse = partial_attention(**partial_args, attn_sink=None)
        records.append(dispatch_dcp_attention_partials(partial, lse, degree=degree))
    expected, _ = _reference(args, keys, sink)
    heads_per_rank = args["q"].shape[1] // degree
    merged = []
    for owner in range(degree):
        head_slice = slice(owner * heads_per_rank, (owner + 1) * heads_per_rank)
        # This is the payload the all-to-all delivers to each head owner.
        received = torch.stack([payload[owner] for payload in records])
        result = combine_dcp_attention_partials(
            received,
            head_dim=512,
            partial_dtype=torch.bfloat16,
            output_dtype=torch.bfloat16,
            sink=sink[head_slice],
        )
        torch.testing.assert_close(
            result.float(), expected[:, head_slice], atol=0.015, rtol=0.015
        )
        merged.append(result)
    full, _ = dsa_decode(**args, q_len_per_req=3, attn_sink=sink)
    torch.testing.assert_close(torch.cat(merged, dim=1), full, atol=0.015, rtol=0.015)
