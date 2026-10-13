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
from tokenspeed_kernel.ops.kvcache.triton_cache_placement import (
    virtual_slots_to_local,
)
from tokenspeed_kernel.platform import current_platform

platform = current_platform()


def _pack_sparse_kv(
    latent: torch.Tensor,
    rope: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    kv_lora_rank = latent.shape[1]
    qk_rope_head_dim = rope.shape[1]
    scale = latent.float().abs().amax(dim=1, keepdim=True).clamp_min(1.0e-6) / 448.0
    latent_fp8 = (latent.float() / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    row_bytes = kv_lora_rank + kv_lora_rank // 128 * 4 + qk_rope_head_dim * 2
    sparse = torch.empty(
        (latent.shape[0], row_bytes),
        dtype=torch.uint8,
        device=latent.device,
    )
    sparse[:, :kv_lora_rank].copy_(latent_fp8.view(torch.uint8))
    scale_start = kv_lora_rank
    scale_end = scale_start + kv_lora_rank // 128 * 4
    sparse[:, scale_start:scale_end].view(torch.float32).copy_(scale)
    sparse[:, scale_end:].view(torch.bfloat16).copy_(rope)
    return sparse, latent_fp8.float() * scale


@pytest.mark.skipif(
    not (platform.is_nvidia and platform.is_hopper_plus),
    reason="The flashmla DSA solution registers on Hopper+ NVIDIA GPUs only",
)
@pytest.mark.parametrize(
    "phase,q_len",
    [
        ("prefill", 1),
        ("decode", 1),
        ("decode", 3),
        ("dense_decode", 1),
        ("dense_decode", 3),
    ],
)
@pytest.mark.parametrize("degree", [1, 2, 4, 8])
def test_flashmla_dsa_dcp_partials(phase, q_len, degree):
    if phase == "dense_decode" and not platform.is_hopper:
        pytest.skip("FlashMLA dense-KV partial decode registers on Hopper only")
    packed = phase == "decode"
    attention = dsa_prefill if phase == "prefill" else dsa_decode
    torch.manual_seed(17)
    latent = torch.randn((256, 512), device="cuda", dtype=torch.bfloat16)
    rope = torch.randn((256, 64), device="cuda", dtype=torch.bfloat16)
    query = torch.randn((3, 4, 576), device="cuda", dtype=torch.bfloat16)
    dense = torch.cat((latent, rope), dim=-1)
    sparse, reference_latent = (
        _pack_sparse_kv(latent, rope) if packed else (None, latent)
    )
    slots = torch.full((3, 512), -1, device="cuda", dtype=torch.int32)
    slots[0, :4] = torch.tensor([64, 65, 128, 193], device="cuda")
    slots[1, :1] = 70
    slots[1, 1] = 130  # Excluded by topk_lens, even when the slot is valid.
    # Third row and some ranks have no candidates.
    kwargs = dict(
        q=query,
        kv_cache=None if packed else dense,
        sparse_kv_cache=sparse,
        topk_lens=torch.tensor([4, 1, 0], device="cuda", dtype=torch.int32),
        max_seqlen_k=256,
        qk_nope_head_dim=128,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        softmax_scale=576**-0.5,
        page_size=64,
        return_lse=True,
        solution="flashmla",
        slot_order="selection",
    )
    if phase != "prefill":
        kwargs["q_len_per_req"] = q_len
    reference, ref_lse = attention(topk_slots=slots, **kwargs)
    slots[1, 1] = -1
    reference_kv = torch.cat((reference_latent.float(), rope.float()), dim=-1)
    scores = (
        torch.einsum(
            "thd,tkd->thk", query.float(), reference_kv[slots.clamp_min(0).long()]
        )
        * 576**-0.5
    )
    expected_lse = torch.logsumexp(
        scores.masked_fill((slots < 0)[:, None, :], -float("inf")), dim=-1
    )
    torch.testing.assert_close(ref_lse, expected_lse, atol=2e-3, rtol=2e-3)
    probs = torch.softmax(
        scores.masked_fill((slots < 0)[:, None, :], -float("inf")), dim=-1
    ).nan_to_num()
    expected = torch.einsum(
        "thk,tkd->thd", probs, reference_latent.float()[slots.clamp_min(0).long()]
    )
    torch.testing.assert_close(reference.float(), expected, atol=0.015, rtol=0.015)
    # FlashMLA serves sinkless dense-KV decode only when it returns the LSE.
    if phase != "dense_decode":
        tensor_kwargs = dict(kwargs, return_lse=False)
        plain = attention(topk_slots=slots, **tensor_kwargs)
        torch.testing.assert_close(plain, reference)
    supplied_out = torch.empty_like(reference)
    returned_out, _ = attention(topk_slots=slots, out=supplied_out, **kwargs)
    assert returned_out is supplied_out
    torch.testing.assert_close(returned_out, reference)

    pages = dense.shape[0] // 64
    cache_key = "sparse_kv_cache" if packed else "kv_cache"
    outputs, lses = [], []
    for rank in range(degree):
        owned = (slots >= 64) & ((slots // 64 - 1) % degree == rank)
        out, lse = attention(topk_slots=torch.where(owned, slots, -1), **kwargs)
        outputs.append(out.float())
        lses.append(lse)
        # The rank's own cache holds the null page, then the pages it owns.
        local_pages = [0] + [p for p in range(1, pages) if (p - 1) % degree == rank]
        local_slots, local_owned = virtual_slots_to_local(
            slots,
            rows_per_page=64,
            virtual_block_count=pages,
            degree=degree,
            rank=rank,
        )
        local_kwargs = dict(kwargs)
        local_kwargs[cache_key] = (
            kwargs[cache_key].view(pages, 64, -1)[local_pages].flatten(0, 1)
        )
        local_out, local_lse = attention(
            topk_slots=torch.where(local_owned, local_slots, -1), **local_kwargs
        )
        torch.testing.assert_close(local_out, out)
        torch.testing.assert_close(local_lse, lse)
    lses = torch.stack(lses)
    merged_lse = torch.logsumexp(lses, dim=0)
    weights = torch.where(torch.isfinite(lses), (lses - merged_lse).exp(), 0.0)
    merged = (torch.stack(outputs) * weights[..., None]).sum(0)
    torch.testing.assert_close(merged, reference.float(), atol=0.015, rtol=0.015)
    torch.testing.assert_close(merged_lse, ref_lse, atol=2e-3, rtol=2e-3)
    assert not reference[2].any()
    assert torch.isneginf(ref_lse[2]).all()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_out, graph_lse = attention(topk_slots=slots, **kwargs)
    slots.fill_(-1)
    graph.replay()
    assert not graph_out.any()
    assert torch.isneginf(graph_lse).all()


@pytest.mark.skipif(
    not (platform.is_nvidia and platform.is_hopper_plus),
    reason="The flashmla DSA solution registers on Hopper+ NVIDIA GPUs only",
)
@pytest.mark.parametrize("heads", [2, 64])
@pytest.mark.parametrize("q_len", [1, 3])
def test_flashmla_packed_decode_lse_counts_the_sink(heads, q_len):
    generator = torch.Generator(device="cuda").manual_seed(29)
    latent = torch.randn((1024, 512), device="cuda", generator=generator)
    rope = torch.randn((1024, 64), device="cuda", generator=generator)
    sparse, _ = _pack_sparse_kv(latent.to(torch.bfloat16), rope.to(torch.bfloat16))
    lens = torch.tensor([512, 37, 0, 3, 200, 1], device="cuda", dtype=torch.int32)
    tokens = lens.numel()
    query = torch.randn((tokens, heads, 576), device="cuda", generator=generator)
    slots = torch.stack(
        [
            torch.randperm(1024, device="cuda", generator=generator)[:512]
            for _ in range(tokens)
        ]
    ).to(torch.int32)
    slots.masked_fill_(torch.arange(512, device="cuda") >= lens[:, None], -1)
    # On the scale of the short rows' LSE, so the sink dominates their softmax.
    sink = torch.randn(heads, device="cuda", generator=generator) + 4.0
    kwargs = dict(
        q=query.to(torch.bfloat16),
        kv_cache=None,
        sparse_kv_cache=sparse,
        topk_slots=slots,
        topk_lens=lens,
        max_seqlen_k=1024,
        qk_nope_head_dim=128,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        softmax_scale=576**-0.5,
        page_size=64,
        q_len_per_req=q_len,
        return_lse=True,
        slot_order="selection",
    )
    out, lse = dsa_decode(solution="flashmla", attn_sink=sink, **kwargs)
    ref_out, ref_lse = dsa_decode(solution="triton", attn_sink=sink, **kwargs)
    _, sinkless_lse = dsa_decode(solution="triton", **kwargs)
    live = lens > 0
    assert (ref_lse[live] > sinkless_lse[live]).all()
    torch.testing.assert_close(out.float(), ref_out.float(), atol=0.015, rtol=0.015)
    torch.testing.assert_close(lse[live], ref_lse[live], atol=2e-3, rtol=2e-3)
    assert not out[~live].any()
    assert torch.isneginf(lse[~live]).all()
