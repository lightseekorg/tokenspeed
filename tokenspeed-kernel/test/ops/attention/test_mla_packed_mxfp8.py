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

"""Packed MLA prologue against independent rotation and group-32 encoding."""

import pytest
import torch
from tokenspeed_kernel.ops.attention.prologue import (
    LatentKVCache,
    PackedMXFP8Rows,
    RopeStyle,
    Rotary,
    latent_store,
    mla_prologue,
)


def reference_rotate(x, table, positions, style):
    cos, sin = table[positions].double().chunk(2, -1)
    cos, sin = cos[:, None], sin[:, None]
    if style is RopeStyle.NEOX:
        a, b = x.double().chunk(2, -1)
    else:
        a, b = x.double()[..., ::2], x.double()[..., 1::2]
    # Match the documented fused multiply-add association, independently
    # of the production rotation and cache-write kernels.
    first = (a * cos - (b * sin).float().double()).float().to(x.dtype)
    second = (b * cos + (a * sin).float().double()).float().to(x.dtype)
    if style is RopeStyle.NEOX:
        return torch.cat((first, second), -1)
    return torch.stack((first, second), -1).flatten(-2)


def reference_pack(x):
    width = x.shape[-1]
    groups = x.float().reshape(x.shape[0], width // 32, 32)
    # A mathematical power-of-two ceiling, not the writer's bit encoding.
    scale = torch.pow(
        2.0, torch.ceil(torch.log2(groups.abs().amax(-1).clamp_min(1.0e-4) / 448.0))
    )
    codes = (torch.log2(scale) + 127).to(torch.uint8)
    payload = (groups / scale[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    size = ((width + width // 32 + 15) // 16) * 16
    packed = torch.zeros(x.shape[0], 1, size, dtype=torch.uint8, device=x.device)
    packed[:, 0, :width] = payload.flatten(1).view(torch.uint8)
    packed[:, 0, width : width + width // 32] = codes
    return packed


@pytest.mark.parametrize("rank,rope", [(256, 64), (512, 64), (512, 128)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("style", [RopeStyle.NEOX, RopeStyle.GPTJ])
def test_packed_prologue_and_gathered_store(rank, rope, dtype, style):
    torch.manual_seed(409)
    device = "cuda"
    n, heads = 5, 4
    query = torch.randn(n, heads, rank + rope, dtype=dtype, device=device)
    qpe = torch.randn(n, heads, rope, dtype=dtype, device=device)
    latent = torch.randn(n, rank + rope, dtype=dtype, device=device)
    latent[0].zero_()
    latent[1].mul_(1.0e-4)
    angles = torch.randn(128, rope // 2, device=device)
    table = torch.cat((angles.cos(), angles.sin()), -1)
    positions = torch.tensor([3, 4, 17, 63, 90], device=device)
    expected_q = torch.cat(
        (query[..., :rank], reference_rotate(qpe, table, positions, style)), -1
    )
    expected_k = torch.cat(
        (
            latent[:, :rank],
            reference_rotate(latent[:, None, rank:], table, positions, style)[:, 0],
        ),
        -1,
    )
    expected_bytes = reference_pack(expected_k)
    # Nonidentity slots; negative masked padding must not write the last row.
    slots = torch.tensor([11, 3, 7, -1, 16], device=device)
    mask = torch.tensor([True, False, True, False, True], device=device)
    storage = torch.full(
        (20, 1, expected_bytes.shape[-1] + 16), 0x5A, device=device, dtype=torch.uint8
    )
    rows = storage[..., : expected_bytes.shape[-1]]
    target = LatentKVCache(PackedMXFP8Rows(rows), False, slots, mask)
    rotary = Rotary(table, positions, style, None)
    out = mla_prologue(
        query.clone(),
        qpe.clone(),
        latent.clone(),
        expanded=None,
        rotary=rotary,
        cache=target,
        solution=None,
        override=None,
    )
    torch.testing.assert_close(out.query, expected_q, rtol=0, atol=0)
    expected_storage = torch.full_like(storage, 0x5A)
    expected_storage[slots[mask], :, : rows.shape[-1]] = expected_bytes[mask]
    assert torch.equal(storage, expected_storage)
    storage.fill_(0x5A)
    prepared = mla_prologue(
        query.clone(),
        qpe.clone(),
        latent.clone(),
        expanded=None,
        rotary=rotary,
        cache=None,
        solution="composite",
        override=None,
    )
    latent_store(prepared.latent, kv_lora_rank=rank, cache=target)
    assert torch.equal(storage, expected_storage)


@pytest.mark.parametrize(
    "shape,dtype",
    [
        ((8, 1, 607), torch.uint8),
        ((8, 1, 608), torch.bfloat16),
        ((8, 608), torch.uint8),
    ],
)
def test_packed_descriptor_rejects_bad_layout(shape, dtype):
    query = torch.zeros(1, 2, 576, dtype=torch.bfloat16, device="cuda")
    target = LatentKVCache(
        PackedMXFP8Rows(torch.empty(shape, dtype=dtype, device="cuda")),
        False,
        torch.zeros(1, dtype=torch.int64, device="cuda"),
        None,
    )
    with pytest.raises(ValueError, match="packed MXFP8"):
        mla_prologue(
            query,
            query[..., 512:],
            query[:, 0],
            expanded=None,
            rotary=None,
            cache=target,
            solution=None,
            override=None,
        )


def test_packed_store_graph_replay_and_token_count_no_compile():
    from tokenspeed_kernel.ops.kvcache._triton.mxfp8_latent import _store
    from utils import assert_no_triton_compile

    cache = torch.full((128, 1, 608), 0x5A, dtype=torch.uint8, device="cuda")
    latent = torch.randn(32, 576, dtype=torch.bfloat16, device="cuda")
    locations = torch.arange(32, dtype=torch.int64, device="cuda")
    target = LatentKVCache(PackedMXFP8Rows(cache), True, locations, None)
    latent[0, :3] = torch.tensor(
        [float("nan"), float("inf"), -float("inf")], device="cuda"
    )
    expected = reference_pack(torch.nan_to_num(latent))
    latent_store(latent, kv_lora_rank=512, cache=target)
    assert torch.equal(cache[:32], expected)
    with assert_no_triton_compile(_store):
        for count in (3, 7, 17, 31):
            latent_store(
                latent[:count],
                kv_lora_rank=512,
                cache=LatentKVCache(
                    PackedMXFP8Rows(cache), True, locations[:count], None
                ),
            )
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            latent_store(latent, kv_lora_rank=512, cache=target)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        latent_store(latent, kv_lora_rank=512, cache=target)
    latent.normal_()
    locations.add_(48)
    expected = reference_pack(latent)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(cache[48:80], expected)
