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
from tokenspeed_kernel.ops.attention.prologue import (
    LatentKVCache,
    MLAExpandedKV,
    RopeStyle,
    Rotary,
    mla_prologue,
)
from tokenspeed_kernel.ops.kvcache.flashmla import (
    _get_flashmla_kv_buffer,
    _set_flashmla_kv_buffer,
    get_flashmla_kv_buffer,
    set_flashmla_kv_buffer,
)
from tokenspeed_kernel.platform import current_platform
from utils import assert_no_triton_compile

pytestmark = pytest.mark.skipif(
    not current_platform().is_hopper_plus, reason="Hopper+ CUDA required"
)


def _reference_rows(latent, rope):
    groups = latent.float().reshape(-1, 4, 128)
    scales = groups.abs().amax(dim=-1).clamp_min(1e-12) / 448.0
    values = (groups / scales[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    packed = torch.zeros(
        (latent.shape[0], 656), device=latent.device, dtype=torch.uint8
    )
    packed[:, :512] = values.reshape(-1, 512).view(torch.uint8)
    packed[:, 512:528] = scales.contiguous().view(torch.uint8)
    if rope.shape[-1]:
        packed[:, 528:] = rope.reshape(-1, 64).view(torch.uint8)
    return packed, (values.float() * scales[..., None]).reshape(-1, 1, 512)


@pytest.mark.parametrize("rope_dim", [0, 64])
@pytest.mark.parametrize("rows", [0, 1, 17, 129])
@pytest.mark.parametrize("paged", [False, True])
def test_packed_rows_round_trip_and_preserve_masked_storage(rope_dim, rows, paged):
    torch.manual_seed(17)
    # The extra page row must remain untouched by writes through the strided view.
    backing = torch.full((4, 65, 1, 656), 53, dtype=torch.uint8, device="cuda")
    cache = backing[:, :64] if paged else backing.view(-1, 1, 656)
    latent = torch.randn((rows, 1, 512 + rope_dim), dtype=torch.bfloat16, device="cuda")
    latent[..., :512] *= torch.tensor(
        [0.01, 1, 10, 100], device="cuda"
    ).repeat_interleave(128)
    slots = (torch.arange(rows, device="cuda", dtype=torch.int64) * 17 + 7) % 256
    mask = torch.arange(rows, device="cuda") % 3 != 1
    before = backing.clone()
    set_flashmla_kv_buffer(
        cache,
        slots,
        latent[..., :512],
        latent[..., 512:],
        sanitize=False,
        write_mask=mask,
    )
    packed, decoded = _reference_rows(latent[..., :512], latent[..., 512:])
    expected = before.clone()
    if paged:
        expected[slots[mask] // 64, slots[mask] % 64, 0] = packed[mask]
    else:
        expected.view(-1, 656)[slots[mask]] = packed[mask]
    torch.testing.assert_close(backing, expected, atol=0, rtol=0)
    nope, rope = get_flashmla_kv_buffer(cache, slots[mask], rope_dim, torch.bfloat16)
    torch.testing.assert_close(nope, decoded[mask].bfloat16(), atol=0, rtol=0)
    torch.testing.assert_close(rope, latent[mask, :, 512:], atol=0, rtol=0)
    if rope_dim == 0:
        assert rope.shape == (int(mask.sum()), 1, 0)


@pytest.mark.parametrize("rope_dim", [0, 64])
def test_invalid_slots_and_nonfinite_masked_rows_do_not_write(rope_dim):
    cache = torch.full((8, 1, 656), 37, dtype=torch.uint8, device="cuda")
    values = torch.full(
        (4, 1, 512 + rope_dim), float("nan"), dtype=torch.bfloat16, device="cuda"
    )
    slots = torch.tensor([-1, 8, 0, 7], dtype=torch.int64, device="cuda")
    mask = torch.tensor([True, True, False, True], device="cuda")
    values[3] = 2
    set_flashmla_kv_buffer(
        cache,
        slots,
        values[..., :512],
        values[..., 512:],
        sanitize=False,
        write_mask=mask,
    )
    assert torch.all(cache[:7] == 37)
    nope, rope = get_flashmla_kv_buffer(cache, slots[:2], rope_dim, torch.float32)
    assert not nope.any() and not rope.any()
    nope, rope = get_flashmla_kv_buffer(cache, slots[3:], rope_dim, torch.float32)
    torch.testing.assert_close(nope, torch.full_like(nope, 2), rtol=1e-6, atol=0)
    torch.testing.assert_close(rope, torch.full_like(rope, 2), rtol=0, atol=0)


@pytest.mark.parametrize("rope_dim", [0, 64])
def test_prologue_keeps_bf16_queries_and_replays_updated_rows(rope_dim):
    cache = torch.zeros((128, 1, 656), dtype=torch.uint8, device="cuda")
    latent = torch.randn((3, 512 + rope_dim), dtype=torch.bfloat16, device="cuda")
    query = torch.randn((3, 8, 512 + rope_dim), dtype=torch.bfloat16, device="cuda")
    slots = torch.tensor([1, 4, 9], dtype=torch.int32, device="cuda")
    mask = torch.tensor([True, False, True], device="cuda")
    target = LatentKVCache(kv_cache=cache, slots=slots, write_mask=mask, sanitize=False)

    def run():
        return mla_prologue(
            query,
            query[..., 512:],
            latent,
            expanded=None,
            rotary=None,
            cache=target,
            solution=None,
            override=None,
        )

    result = run()
    assert result.query.dtype == torch.bfloat16
    torch.testing.assert_close(result.query, query, rtol=0, atol=0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = run()
    latent.mul_(2)
    slots.copy_(torch.tensor([11, 8, 2], dtype=torch.int32, device="cuda"))
    mask.logical_not_()
    cache.zero_()
    graph.replay()
    packed, _ = _reference_rows(latent[:, :512], latent[:, 512:])
    expected = torch.zeros_like(cache)
    expected[slots[mask].long(), 0] = packed[mask]
    torch.testing.assert_close(cache, expected, rtol=0, atol=0)
    torch.testing.assert_close(result.query, query, rtol=0, atol=0)


def test_packed_writer_sanitizes_and_does_not_specialize_on_batch_size():
    cache = torch.zeros((128, 1, 656), dtype=torch.uint8, device="cuda")
    values = torch.randn((32, 1, 576), dtype=torch.bfloat16, device="cuda")
    values[:, :, 0] = float("nan")
    values[:, :, 1] = float("inf")
    values[:, :, 512] = -float("inf")
    slots = torch.arange(1, 33, dtype=torch.int64, device="cuda")

    def run(n):
        set_flashmla_kv_buffer(
            cache,
            slots[:n],
            values[:n, :, :512],
            values[:n, :, 512:],
            sanitize=True,
            write_mask=None,
        )
        return get_flashmla_kv_buffer(cache, slots[:n], 64, torch.bfloat16)

    run(1)
    with assert_no_triton_compile(_set_flashmla_kv_buffer, _get_flashmla_kv_buffer):
        for n in (3, 17, 32):
            nope, rope = run(n)
            assert torch.isfinite(nope).all() and torch.isfinite(rope).all()
            assert not nope[..., 0].any()
            assert torch.all(rope[..., 0] == -torch.finfo(torch.bfloat16).max)


@pytest.mark.parametrize("style", [RopeStyle.NEOX, RopeStyle.GPTJ])
@pytest.mark.parametrize("expanded", [False, True])
def test_packed_prologue_quantizes_the_rotated_native_rows(style, expanded):
    torch.manual_seed(91)
    rows, heads, rope = 3, 8, 64
    q_nope = 128 if expanded else 512
    query = torch.randn(
        (rows, heads, q_nope + rope), dtype=torch.bfloat16, device="cuda"
    )
    latent = torch.randn((rows, 512 + rope), dtype=torch.bfloat16, device="cuda")
    slots = torch.tensor([7, 1, 8], dtype=torch.int64, device="cuda")
    phase = torch.randn((16, rope // 2), device="cuda")
    rotary = Rotary(
        torch.cat((phase.cos(), phase.sin()), -1),
        torch.tensor([2, 4, 9], device="cuda"),
        style,
        None,
    )
    expanded_kv = (
        MLAExpandedKV(
            torch.randn((rows, heads, q_nope), dtype=query.dtype, device="cuda"),
            torch.randn((rows, heads, 128), dtype=query.dtype, device="cuda"),
        )
        if expanded
        else None
    )
    native = torch.zeros((64, 1, 576), dtype=torch.bfloat16, device="cuda")
    packed = torch.zeros((64, 1, 656), dtype=torch.uint8, device="cuda")

    def run(cache):
        q = query.clone()
        return mla_prologue(
            q,
            q[..., q_nope:],
            latent.clone(),
            expanded=expanded_kv,
            rotary=rotary,
            cache=LatentKVCache(cache, False, slots, None),
            solution="composite",
            override=None,
        )

    reference = run(native)
    result = run(packed)
    torch.testing.assert_close(result.query, reference.query, rtol=0, atol=0)
    if expanded:
        torch.testing.assert_close(result.key, reference.key, rtol=0, atol=0)
        torch.testing.assert_close(result.value, reference.value, rtol=0, atol=0)
    encoded, _ = _reference_rows(native[slots, :, :512], native[slots, :, 512:])
    torch.testing.assert_close(packed[slots, 0], encoded, rtol=0, atol=0)
