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
    LatentNorm,
    MLAExpandedKV,
    RopeStyle,
    Rotary,
    mla_prologue,
)
from tokenspeed_kernel.ops.attention.prologue.checks import check_mla_request
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
    scales = (groups.abs().amax(dim=-1) / 448.0).clamp_min(1e-4)
    scales = torch.exp2(torch.ceil(torch.log2(scales)))
    limit = torch.where(scales < 2.0**120, 448.0, 240.0)[..., None]
    values = (groups / scales[..., None]).clamp(-limit, limit).to(torch.float8_e4m3fn)
    packed = torch.zeros(
        (latent.shape[0], 656), device=latent.device, dtype=torch.uint8
    )
    packed[:, :512] = values.reshape(-1, 512).view(torch.uint8)
    packed[:, 512:528] = scales.contiguous().view(torch.uint8)
    if rope.shape[-1]:
        packed[:, 528:] = rope.reshape(-1, 64).view(torch.uint8)
    return packed, (values.float() * scales[..., None]).reshape(-1, 1, 512)


def _normalized_latent(latent, norm):
    values = latent.float()
    return (
        values
        * torch.rsqrt(values.square().mean(-1, keepdim=True) + norm.eps)
        * norm.weight.float()
    )


@pytest.mark.parametrize("cache_dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_latent_norm_rejects_other_formats_after_a_cached_layout_check(cache_dtype):
    query = torch.zeros((1, 8, 512), dtype=torch.bfloat16, device="cuda")
    latent = torch.zeros((1, 512), dtype=query.dtype, device="cuda")
    target = LatentKVCache(
        torch.zeros((8, 1, 512), dtype=cache_dtype, device="cuda"),
        False,
        torch.tensor([1], dtype=torch.int64, device="cuda"),
        None,
    )
    check_mla_request(query, query[..., 512:], latent, None, None, target)
    norm = LatentNorm(torch.ones(512, dtype=query.dtype, device="cuda"), 1e-6)
    with pytest.raises(ValueError, match="absorbed FlashMLA"):
        check_mla_request(query, query[..., 512:], latent, None, None, target, norm)


@pytest.mark.parametrize("written_rows", [0, 2])
def test_fused_norm_preserves_raw_latent_and_only_writes_selected_leading_rows(
    written_rows,
):
    latent = torch.randn((5, 512), dtype=torch.bfloat16, device="cuda")
    original = latent.clone()
    query = torch.randn((5, 8, 512), dtype=torch.bfloat16, device="cuda")
    cache = torch.full((8, 1, 656), 37, dtype=torch.uint8, device="cuda")
    slots = torch.tensor([7, 1], dtype=torch.int64, device="cuda")[:written_rows]
    mask = torch.tensor([True, False], device="cuda")[:written_rows]
    norm = LatentNorm(torch.ones(512, dtype=query.dtype, device="cuda"), 1e-6)
    expected = cache.clone()
    encoded, _ = _reference_rows(_normalized_latent(latent, norm), latent[:, 512:])
    expected[slots[mask], 0] = encoded[:written_rows][mask]
    out = mla_prologue(
        query,
        query[..., 512:],
        latent,
        expanded=None,
        rotary=None,
        cache=LatentKVCache(cache, False, slots, mask),
        solution=None,
        override=None,
        norm=norm,
    )
    torch.testing.assert_close(cache, expected, atol=0, rtol=0)
    torch.testing.assert_close(latent, original, atol=0, rtol=0)
    torch.testing.assert_close(out.query, query, atol=0, rtol=0)


@pytest.mark.parametrize("boundary", ["fp8_rounding", "scale_rounding"])
def test_fused_norm_quantizes_fp32_without_a_bf16_intermediate(boundary):
    values = torch.ones((1, 512), dtype=torch.bfloat16, device="cuda")
    weight = torch.full((512,), 1.1875, dtype=torch.bfloat16, device="cuda")
    if boundary == "scale_rounding":
        values[0, -1] = 0.99609375
        weight.fill_(1)
        weight[0] = 448
    norm = LatentNorm(weight, 1e-6)
    normalized = _normalized_latent(values, norm)
    rope = values.new_empty((1, 0))
    expected, _ = _reference_rows(normalized, rope)
    rounded, _ = _reference_rows(normalized.bfloat16(), rope)
    # These inputs distinguish direct FP32 quantization from the old two-step path.
    assert not torch.equal(expected, rounded)
    if boundary == "scale_rounding":
        assert expected[:, 512:528].view(torch.float32)[0, 0] == 2
        assert rounded[:, 512:528].view(torch.float32)[0, 0] == 1
    cache = torch.zeros((1, 1, 656), dtype=torch.uint8, device="cuda")
    before = values.clone()
    set_flashmla_kv_buffer(
        cache,
        torch.zeros(1, dtype=torch.int64, device="cuda"),
        values,
        rope,
        sanitize=False,
        write_mask=None,
        norm=norm,
    )
    torch.testing.assert_close(cache[:, 0], expected, atol=0, rtol=0)
    torch.testing.assert_close(values, before, atol=0, rtol=0)


@pytest.mark.parametrize("sanitize", [False, True])
def test_fused_norm_preserves_scale_floor_and_finite_extreme_dequantization(sanitize):
    values = torch.ones((3, 512), dtype=torch.bfloat16, device="cuda")
    values[0].zero_()
    values[1].fill_(1e-8)
    weights = (torch.tensor([224, 247, 248, 255], device="cuda") * 2.0**120).bfloat16()
    norm = LatentNorm(weights.repeat_interleave(128), 1e-6)
    rope = values.new_empty((3, 0))
    cache = torch.zeros((3, 1, 656), dtype=torch.uint8, device="cuda")
    slots = torch.arange(3, dtype=torch.int64, device="cuda")
    set_flashmla_kv_buffer(
        cache, slots, values, rope, sanitize=sanitize, write_mask=None, norm=norm
    )
    expected, _ = _reference_rows(_normalized_latent(values, norm), rope)
    torch.testing.assert_close(cache[:, 0], expected, atol=0, rtol=0)
    assert (cache[0, 0, 512:528].view(torch.float32) == 2**-13).all()
    decoded, _ = get_flashmla_kv_buffer(cache, slots, 0, torch.bfloat16)
    assert torch.isfinite(decoded).all()
    assert (cache[2, 0, 128:512].view(torch.float8_e4m3fn).float() == 240).all()


def test_packed_scales_apply_floor_and_round_up_to_powers_of_two():
    maxima = torch.tensor(
        [
            [0, 1e-6, 224, 448],
            [450, 896, 900, 0.75],
            [2**-24, 2**-23, 3 * 2**-24, 2**-22],
            [0.02734375, 0.03125, 0.0546875, 0.054931640625],
        ],
        dtype=torch.bfloat16,
        device="cuda",
    )
    values = torch.zeros((4, 4, 128), dtype=torch.bfloat16, device="cuda")
    values[:, :, 0] = maxima
    values[:, :, 1] = -maxima
    cache = torch.zeros((4, 1, 656), dtype=torch.uint8, device="cuda")
    slots = torch.arange(4, dtype=torch.int64, device="cuda")
    set_flashmla_kv_buffer(
        cache,
        slots,
        values.reshape(4, 512),
        values.new_empty((4, 0)),
        sanitize=False,
        write_mask=None,
    )
    scales = cache[:, 0, 512:528].view(torch.float32)
    expected = torch.tensor(
        [
            [2**-13, 2**-13, 0.5, 1],
            [2, 2, 4, 2**-9],
            [2**-13] * 4,
            [2**-13, 2**-13, 2**-13, 2**-12],
        ],
        dtype=torch.float32,
        device="cuda",
    )
    torch.testing.assert_close(scales, expected, atol=0, rtol=0)
    torch.testing.assert_close(scales.bfloat16().float(), scales, atol=0, rtol=0)
    assert not cache[:, 0, 528:].any()
    # The floor makes the smallest decoded value 2**-22; the half-way tie is zero.
    decoded, _ = get_flashmla_kv_buffer(cache, slots, 0, torch.bfloat16)
    decoded = decoded.reshape(4, 4, 128)
    smallest = values.new_tensor([0, 0, 2**-22, 2**-22])
    torch.testing.assert_close(decoded[2, :, 0], smallest, atol=0, rtol=0)
    torch.testing.assert_close(decoded[2, :, 1], -smallest, atol=0, rtol=0)


@pytest.mark.parametrize("sanitize", [False, True])
def test_packed_extreme_bf16_values_dequantize_finitely(sanitize):
    # Last full-range scale, just below overflow, the rounding tie, and BF16 max.
    magnitudes = torch.tensor([224, 247, 248, 255], dtype=torch.float32, device="cuda")
    groups = torch.stack((magnitudes, -magnitudes)) * 2.0**120
    values = groups.bfloat16()[..., None].expand(2, 4, 128).reshape(2, 512)
    cache = torch.zeros((2, 1, 656), dtype=torch.uint8, device="cuda")
    slots = torch.arange(2, dtype=torch.int64, device="cuda")
    set_flashmla_kv_buffer(
        cache,
        slots,
        values,
        values.new_empty((2, 0)),
        sanitize=sanitize,
        write_mask=None,
    )
    positive_codes = magnitudes.new_tensor([448, 240, 240, 240])
    codes = torch.stack((positive_codes, -positive_codes))
    scales = magnitudes.new_tensor(
        [2.0**119, 2.0**120, 2.0**120, 2.0**120]
    ).expand(2, 4)
    torch.testing.assert_close(
        cache[:, 0, :512].view(torch.float8_e4m3fn).float().reshape(2, 4, 128),
        codes[..., None].expand(2, 4, 128),
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        cache[:, 0, 512:528].view(torch.float32), scales, atol=0, rtol=0
    )
    nope, _ = get_flashmla_kv_buffer(cache, slots, 0, torch.bfloat16)
    expected = (
        (codes * scales).bfloat16()[..., None].expand(2, 4, 128).reshape(2, 1, 512)
    )
    assert torch.isfinite(nope).all()
    torch.testing.assert_close(nope, expected, atol=0, rtol=0)


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
@pytest.mark.parametrize("fused_norm", [False, True])
def test_prologue_keeps_bf16_queries_and_replays_updated_rows(rope_dim, fused_norm):
    cache = torch.zeros((128, 1, 656), dtype=torch.uint8, device="cuda")
    latent = torch.randn((3, 512 + rope_dim), dtype=torch.bfloat16, device="cuda")
    query = torch.randn((3, 8, 512 + rope_dim), dtype=torch.bfloat16, device="cuda")
    slots = torch.tensor([1, 4, 9], dtype=torch.int32, device="cuda")
    mask = torch.tensor([True, False, True], device="cuda")
    target = LatentKVCache(kv_cache=cache, slots=slots, write_mask=mask, sanitize=False)
    norm = (
        LatentNorm(torch.ones(512, dtype=torch.bfloat16, device="cuda"), 1e-6)
        if fused_norm
        else None
    )

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
            norm=norm,
        )

    result = run()
    assert result.query.dtype == torch.bfloat16
    torch.testing.assert_close(result.query, query, rtol=0, atol=0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = run()
    latent.mul_(2)
    if norm is not None:
        norm.weight.mul_(1.25)
    slots.copy_(torch.tensor([11, 8, 2], dtype=torch.int32, device="cuda"))
    mask.logical_not_()
    cache.zero_()
    graph.replay()
    normalized = (
        latent[:, :512] if norm is None else _normalized_latent(latent[:, :512], norm)
    )
    packed, _ = _reference_rows(normalized, latent[:, 512:])
    expected = torch.zeros_like(cache)
    expected[slots[mask].long(), 0] = packed[mask]
    torch.testing.assert_close(cache, expected, rtol=0, atol=0)
    torch.testing.assert_close(result.query, query, rtol=0, atol=0)


@pytest.mark.parametrize("fused_norm", [False, True])
def test_packed_writer_sanitizes_and_does_not_specialize_on_batch_size(fused_norm):
    cache = torch.zeros((128, 1, 656), dtype=torch.uint8, device="cuda")
    values = torch.randn((32, 1, 576), dtype=torch.bfloat16, device="cuda")
    values[:, :, 0] = float("nan")
    values[:, :, 1] = float("inf")
    values[:, :, 512] = -float("inf")
    slots = torch.arange(1, 33, dtype=torch.int64, device="cuda")
    norm = (
        LatentNorm(torch.ones(512, dtype=torch.bfloat16, device="cuda"), 1e-6)
        if fused_norm
        else None
    )

    def run(n):
        set_flashmla_kv_buffer(
            cache,
            slots[:n],
            values[:n, :, :512],
            values[:n, :, 512:],
            sanitize=True,
            write_mask=None,
            norm=norm,
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
@pytest.mark.parametrize(
    "expanded,fused_norm", [(False, False), (True, False), (False, True)]
)
def test_packed_prologue_quantizes_the_rotated_native_rows(style, expanded, fused_norm):
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
    norm = (
        LatentNorm(torch.ones(512, dtype=torch.bfloat16, device="cuda"), 1e-6)
        if fused_norm
        else None
    )

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
            norm=norm if cache.dtype == torch.uint8 else None,
        )

    reference = run(native)
    result = run(packed)
    torch.testing.assert_close(result.query, reference.query, rtol=0, atol=0)
    if expanded:
        torch.testing.assert_close(result.key, reference.key, rtol=0, atol=0)
        torch.testing.assert_close(result.value, reference.value, rtol=0, atol=0)
    normalized = (
        native[slots, :, :512]
        if norm is None
        else _normalized_latent(latent[:, :512], norm)
    )
    encoded, _ = _reference_rows(normalized, native[slots, :, 512:])
    torch.testing.assert_close(packed[slots, 0], encoded, rtol=0, atol=0)
