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

"""The one-launch MLA context write stores the per-layer chain's bytes.

The chain it replaces is FlashInfer's RMSNorm, the CUDA RoPE, then the latent
store.
"""

import pytest
import torch
from tokenspeed_kernel.ops.embedding import apply_k_rope
from tokenspeed_kernel.ops.kvcache.triton import (
    LatentPlanes,
    _mla_latent_norm_rope_scatter_kernel,
    mla_latent_norm_rope_scatter,
    set_mla_kv_buffer_triton,
)
from tokenspeed_kernel.ops.layernorm import rmsnorm
from utils import assert_no_triton_compile

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the reference chain runs CUDA kernels"
)

EPS = 1e-5
SENTINEL = 3.0


def _planes(layers, slots, width, dtype, page_size):
    """Latent planes full of a sentinel: dense rows, or pages of a padded backing."""
    if page_size == 1:
        return [
            torch.full((slots, 1, width), SENTINEL, device="cuda").to(dtype)
            for _ in range(layers)
        ]
    pages = slots // page_size
    return [
        torch.full((pages, page_size + 2, 1, width), SENTINEL, device="cuda").to(dtype)[
            :, :page_size
        ]
        for _ in range(layers)
    ]


def _inputs(rows, layers, rank, rope, seed, *, poison):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    width = rank + rope
    # Each layer's rows on their own, as the K3 draft lays its GEMM outputs out.
    latent = (
        torch.randn(layers, rows, width, device="cuda", generator=generator) * 2
    ).to(torch.bfloat16)
    if poison and rows > 1:
        latent[0, 0, 3] = float("nan")
        latent[layers - 1, 1, rank + 2] = float("inf")
    weight = (torch.randn(layers, rank, device="cuda", generator=generator) + 1).to(
        torch.bfloat16
    )
    cos_sin = torch.rand(4096, rope, device="cuda", generator=generator) * 2 - 1
    positions = torch.randint(4096, (rows,), device="cuda", generator=generator)
    return latent, weight, cos_sin, positions


def _slots(rows, slots, masked, seed):
    """Distinct non-zero slots, with masked rows at slot 0.

    A masked row points at slot 0, as an unowned row resolves under KV
    parallelism.
    """
    generator = torch.Generator(device="cuda").manual_seed(seed + 1)
    loc = torch.randperm(slots - 1, device="cuda", generator=generator)[:rows] + 1
    if not masked:
        return loc, None
    owned = torch.arange(rows, device="cuda") % 3 != 1
    return loc.masked_fill(~owned, 0), owned


def _chain(
    latent, weight, cos_sin, positions, loc, owned, planes, rope, is_neox, sanitize
):
    """The per-layer chain: FlashInfer RMSNorm, CUDA RoPE, latent store."""
    rank = weight.shape[-1]
    for layer, plane in enumerate(planes):
        rows = latent[layer]
        nope = rmsnorm(rows[:, :rank].contiguous(), weight[layer], EPS)
        k_pe = rows[:, rank:].reshape(-1, 1, rope).clone()
        rotated = apply_k_rope(positions, k_pe, rope, cos_sin, is_neox=is_neox)
        set_mla_kv_buffer_triton(
            plane,
            loc,
            nope,
            rotated.reshape(-1, rope),
            enable_pdl=False,
            sanitize=sanitize,
            write_mask=owned,
        )


def _fused(latent, weight, cos_sin, positions, loc, owned, planes, is_neox, sanitize):
    mla_latent_norm_rope_scatter(
        latent.transpose(0, 1),
        weight,
        torch.full((weight.shape[0],), EPS, device="cuda"),
        cos_sin,
        positions,
        loc,
        LatentPlanes.from_buffers(planes),
        write_mask=owned,
        is_neox=is_neox,
        sanitize=sanitize,
    )


def _assert_same_bytes(actual, expected):
    for got, want in zip(actual, expected, strict=True):
        assert torch.equal(got.view(torch.uint8), want.view(torch.uint8)), (
            (got.view(torch.uint8) != want.view(torch.uint8)).sum().item()
        )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("rank,rope", [(512, 64)])
@pytest.mark.parametrize("is_neox", [False, True])
@pytest.mark.parametrize("sanitize", [False, True])
@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("page_size", [1, 16])
@pytest.mark.parametrize("rows", [1, 9, 130])
def test_the_fused_write_stores_the_per_layer_chains_bytes(
    dtype, rank, rope, is_neox, sanitize, masked, page_size, rows
):
    layers, slots = 3, 512
    seed = rows * 7 + rank
    latent, weight, cos_sin, positions = _inputs(
        rows, layers, rank, rope, seed, poison=sanitize
    )
    loc, owned = _slots(rows, slots, masked, seed)
    expected = _planes(layers, slots, rank + rope, dtype, page_size)
    actual = _planes(layers, slots, rank + rope, dtype, page_size)

    _chain(
        latent,
        weight,
        cos_sin,
        positions,
        loc,
        owned,
        expected,
        rope,
        is_neox,
        sanitize,
    )
    _fused(latent, weight, cos_sin, positions, loc, owned, actual, is_neox, sanitize)

    _assert_same_bytes(actual, expected)
    if page_size > 1:
        # The page padding past each page's rows stays untouched.
        for plane in actual:
            padding = torch.as_strided(
                plane,
                (plane.shape[0], 2, 1, plane.shape[-1]),
                plane.stride(),
                plane.storage_offset() + page_size * plane.stride(1),
            )
            assert torch.all(padding.float() == SENTINEL)


def test_graph_replay_follows_the_refreshed_rows_and_slots():
    rank, rope, rows, layers, slots = 512, 64, 8, 5, 64
    latent, weight, cos_sin, positions = _inputs(
        rows, layers, rank, rope, 1, poison=False
    )
    loc, owned = _slots(rows, slots, True, 1)
    planes = _planes(layers, slots, rank + rope, torch.float8_e4m3fn, 1)
    eps = torch.full((layers,), EPS, device="cuda")
    table = LatentPlanes.from_buffers(planes)
    stacked = latent.transpose(0, 1)

    def write():
        mla_latent_norm_rope_scatter(
            stacked,
            weight,
            eps,
            cos_sin,
            positions,
            loc,
            table,
            write_mask=owned,
            is_neox=False,
            sanitize=True,
        )

    write()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        write()
    fresh, _, _, fresh_positions = _inputs(rows, layers, rank, rope, 2, poison=False)
    fresh_loc, _ = _slots(rows, slots, True, 2)
    latent.copy_(fresh)
    positions.copy_(fresh_positions)
    loc.copy_(fresh_loc.masked_fill(~owned, 0))
    for plane in planes:
        plane.fill_(SENTINEL)
    graph.replay()
    expected = _planes(layers, slots, rank + rope, torch.float8_e4m3fn, 1)
    _chain(latent, weight, cos_sin, positions, loc, owned, expected, rope, False, True)
    _assert_same_bytes(planes, expected)


def test_new_row_counts_reuse_the_compiled_kernel():
    rank, rope, layers, slots = 512, 64, 5, 256
    planes = _planes(layers, slots, rank + rope, torch.bfloat16, 1)

    def write(rows, masked):
        latent, weight, cos_sin, positions = _inputs(
            rows, layers, rank, rope, rows, poison=False
        )
        loc, owned = _slots(rows, slots, masked, rows)
        _fused(latent, weight, cos_sin, positions, loc, owned, planes, False, False)

    for masked in (False, True):
        write(3, masked)
        write(16, masked)
    with assert_no_triton_compile(_mla_latent_norm_rope_scatter_kernel):
        for rows in (1, 5, 8, 37, 200):
            for masked in (False, True):
                write(rows, masked)


def test_slots_past_int32_offsets_land_in_place():
    """A slot whose element offset passes 2**31 is formed in 64 bits."""
    rank, rope, rows = 512, 64, 2
    width = rank + rope
    slots = (2**31) // width + 4
    plane = torch.zeros((slots, 1, width), dtype=torch.float8_e4m3fn, device="cuda")
    latent, weight, cos_sin, positions = _inputs(rows, 1, rank, rope, 5, poison=False)
    loc = torch.tensor([slots - 1, slots - 3], device="cuda")
    _fused(latent, weight, cos_sin, positions, loc, None, [plane], False, False)
    expected = torch.zeros(
        (rows + 2, 1, width), dtype=torch.float8_e4m3fn, device="cuda"
    )
    _chain(
        latent,
        weight,
        cos_sin,
        positions,
        torch.tensor([3, 1], device="cuda"),
        None,
        [expected],
        rope,
        False,
        False,
    )
    assert torch.equal(plane[-4:].view(torch.uint8), expected.view(torch.uint8))
    assert not torch.any(plane[: slots - 4].view(torch.uint8)[:4096])


def test_planes_must_share_one_geometry():
    dense = torch.zeros((8, 1, 576), dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError, match="one geometry"):
        LatentPlanes.from_buffers([dense, dense.to(torch.float8_e4m3fn)])
    with pytest.raises(ValueError, match="rows"):
        LatentPlanes.from_buffers([dense.view(8, 576)])
    planes = LatentPlanes.from_buffers([dense, torch.zeros_like(dense)])
    latent = torch.zeros((2, 2, 576), dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError, match="cover 1 layers"):
        mla_latent_norm_rope_scatter(
            latent[:, :1],
            torch.zeros((1, 512), dtype=torch.bfloat16, device="cuda"),
            torch.zeros(1, device="cuda"),
            torch.zeros((16, 64), device="cuda"),
            torch.zeros(2, dtype=torch.int64, device="cuda"),
            torch.zeros(2, dtype=torch.int64, device="cuda"),
            planes,
            write_mask=None,
            is_neox=False,
            sanitize=False,
        )
