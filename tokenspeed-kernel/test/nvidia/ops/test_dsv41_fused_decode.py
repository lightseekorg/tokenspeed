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

from __future__ import annotations

import importlib.util

import pytest
import torch
from tokenspeed_kernel.ops.attention import dsv41
from tokenspeed_kernel.ops.attention.dsv41.flash_mla import (
    is_flash_mla_fused_decode_available,
)
from tokenspeed_kernel.ops.gemm import (
    grouped_fp8_projection,
    grouped_fp8_projection_quantize_fp8,
    mm,
    prepare_grouped_fp8_projection,
    prepare_mxfp8_weight_scales,
)
from tokenspeed_kernel.ops.gemm.flashinfer import has_flashinfer_mxfp8
from tokenspeed_kernel.platform import pdl_enabled

pytestmark = pytest.mark.skipif(
    not is_flash_mla_fused_decode_available(),
    reason="requires fused FlashMLA on SM100 or SM103",
)

_FP8 = torch.float8_e4m3fn
_SOFTMAX_SCALE = 512**-0.5


@pytest.fixture
def projection_dependencies():
    if importlib.util.find_spec("deep_gemm") is None or not has_flashinfer_mxfp8():
        pytest.skip("requires DeepGEMM and FlashInfer MXFP8")
    previous = pdl_enabled()
    pdl_enabled(True)
    try:
        yield
    finally:
        pdl_enabled(previous)


def _weights(device, heads, groups):
    # Fingerprint the first three code bytes of each Q row to recover the
    # native row permutation independently of its packed scale layout.
    q = torch.randn(heads * 512, 256, device=device).to(_FP8)
    q_bytes = q.view(torch.uint8)
    rows = torch.arange(q.shape[0], device=device)
    for digit in range(3):
        q_bytes[:, digit] = (rows >> (6 * digit)) & 63
    q_scales = torch.randint(
        121, 125, (8, q.shape[0]), device=device, dtype=torch.uint8
    ).T

    width = heads // groups * 512
    o = torch.randn(groups * 128, width, device=device).to(_FP8)
    o_scales = torch.randint(
        119, 124, (o.shape[0], width // 32), device=device, dtype=torch.uint8
    )
    o_packed = prepare_grouped_fp8_projection(o, o_scales, groups)
    assert o_packed is not None

    original = [
        tensor.clone() for tensor in (q_bytes, q_scales, o.view(torch.uint8), o_packed)
    ]
    weights = dsv41.prepare_fused_decode_weights(
        q, q_scales, o, o_packed, heads, groups
    )
    assert weights is not None
    q_perm, q_perm_scales, o_perm, o_perm_packed = weights
    for got, want in zip(
        (q_bytes, q_scales, o.view(torch.uint8), o_packed), original, strict=True
    ):
        torch.testing.assert_close(got, want, rtol=0, atol=0)
    fingerprints = q_perm.view(torch.uint8)[:, :3].to(torch.int64)
    permuted_rows = (
        fingerprints[:, 0] + (fingerprints[:, 1] << 6) + (fingerprints[:, 2] << 12)
    )
    torch.testing.assert_close(permuted_rows.sort().values, rows, rtol=0, atol=0)
    torch.testing.assert_close(q_perm_scales, q_scales[permuted_rows], rtol=0, atol=0)

    q_packed = prepare_mxfp8_weight_scales(q, q_scales.contiguous())
    q_perm_packed = prepare_mxfp8_weight_scales(q_perm, q_perm_scales)
    assert q_packed is not None and q_perm_packed is not None
    return (q, q_packed, o.view(groups, 128, width), o_packed), (
        q_perm,
        q_perm_packed,
        o_perm,
        o_perm_packed,
    )


def _rotate(x, positions, cos_sin):
    cos, sin = cos_sin[positions.clamp_min(0)].chunk(2, dim=-1)
    even, odd = x[..., -64:].float().reshape(*x.shape[:-1], 32, 2).unbind(-1)
    tail = torch.stack(
        (
            even * cos[:, None] - odd * sin[:, None],
            odd * cos[:, None] + even * sin[:, None],
        ),
        dim=-1,
    )
    return torch.cat((x[..., :-64], tail.flatten(-2).to(x.dtype)), dim=-1)


def _selected_rows(cache, slots, lengths, fmt):
    valid = (slots >= 0) & (
        torch.arange(slots.shape[1], device=slots.device) < lengths[:, None]
    )
    return dsv41.cache_gather(cache, slots.masked_fill(~valid, -1), fmt, None), valid


def _attention_reference(q, kv, valid, sink):
    # Explicit FP32 products avoid TF32 settings and keep the sink in the
    # softmax denominator only. The workspace and page codecs supply BF16 KV.
    values = kv.float()[:, None]
    logits = (q.float()[:, :, None] * values).sum(-1) * _SOFTMAX_SCALE
    logits.masked_fill_(~valid[:, None], -torch.inf)
    sink_logits = sink[None, :, None].expand(q.shape[0], -1, -1)
    probs = torch.cat((logits, sink_logits), dim=-1).softmax(-1)[..., :-1]
    return (probs[..., None] * values).sum(-2).to(torch.bfloat16)


def _dequantize(codes, scales):
    exponents = scales.contiguous().view(torch.uint8).float() - 127
    return (
        codes.float().unflatten(-1, (-1, 32)) * torch.exp2(exponents[..., None])
    ).flatten(-2)


def _assert_projection_close(actual, expected, relative_rms, max_error):
    actual, expected = actual.float(), expected.float()
    rms = expected.square().mean().sqrt()
    assert rms.item() > 0
    error = (actual - expected).abs()
    assert (error.square().mean().sqrt() / rms).item() < relative_rms
    if max_error is not None:
        assert (error.max() / rms).item() < max_error


def _assert_quantized_projection(bf16, codes, scales):
    """The FP8 GEMM epilogue must equal an independent cast of its BF16 output."""
    blocks = bf16.float().flatten(1).unflatten(-1, (-1, 32))
    amax = blocks.abs().amax(-1).clamp_min(1e-4)
    mantissa, exponent = torch.frexp(amax)
    # 448 = 0.875 * 2**9. Select the smallest power-of-two scale that fits
    # the group without a log2 approximation at an exact exponent boundary.
    scale_exponent = exponent - 9 + (mantissa > 0.875).to(exponent.dtype)
    reference_scales = (scale_exponent + 127).to(torch.uint8)
    reference_codes = (
        (blocks / torch.exp2(scale_exponent.float())[..., None]).flatten(-2).to(_FP8)
    )
    torch.testing.assert_close(
        codes.view(torch.uint8), reference_codes.view(torch.uint8), atol=0, rtol=0
    )
    torch.testing.assert_close(
        scales.contiguous().view(torch.uint8), reference_scales, atol=0, rtol=0
    )


@pytest.mark.parametrize(
    "layout,heads", [("swa", 64), ("global", 128), ("workspace", 64)]
)
def test_fused_decode_projection_and_graph(
    device, projection_dependencies, layout, heads
):
    torch.manual_seed(316)
    groups = heads // 8
    tokens = 5
    original, permuted = _weights(device, heads, groups)
    x = torch.randn(tokens, 256, device=device).to(_FP8)
    x_scales = torch.randint(125, 129, (tokens, 8), device=device, dtype=torch.uint8)
    positions = torch.tensor([11, 17, 7, 2, -1], device=device, dtype=torch.int64)
    angle = torch.arange(32, device=device)[:, None] * torch.linspace(
        0.012, 0.5, 32, device=device
    )
    cos_sin = torch.cat((angle.cos(), angle.sin()), dim=-1)
    inverse = torch.cat((angle.cos(), -angle.sin()), dim=-1)
    sink = torch.linspace(-1.0, 0.7, heads, device=device, dtype=torch.float32)

    rows = torch.randn(128, 512, device=device, dtype=torch.bfloat16) * 0.5
    slots = torch.full((tokens, 67), -1, device=device, dtype=torch.int32)
    slots[0, :6] = torch.tensor([66, 2, -1, 7, 100, 101], device=device)
    slots[1, :4] = torch.tensor([95, 1, 3, 77], device=device)
    lens = torch.tensor([5, 3, 2, 0, 0], device=device, dtype=torch.int32)
    if layout == "workspace":
        slots[0, 5] = slots[1, 3] = -1
        workspace = rows[:, None]
        workspace_slots = slots
        swa_args = (None, None, None)
    else:
        cache = torch.empty((2, 64, 528), device=device, dtype=torch.uint8)
        dsv41.cache_scatter(rows, cache, torch.arange(128, device=device), "swa")
        workspace, workspace_slots = None, None
        swa_args = (cache, slots, lens)
        slots[3, 0] = 39  # Valid physical slot beyond the declared zero length.
    if layout == "global":
        global_cache = torch.empty((2, 64, 288), device=device, dtype=torch.uint8)
        dsv41.cache_scatter(
            rows.flip(0) * 1.5, global_cache, torch.arange(128, device=device), "global"
        )
        global_slots = torch.full((tokens, 71), -1, device=device, dtype=torch.int64)
        global_slots[0, :4] = torch.tensor([4, 93, -1, 17], device=device)
        global_slots[2, :2] = torch.tensor([14, 97], device=device)
        global_lens = torch.tensor([4, 0, 2, 0, 0], device=device, dtype=torch.int64)
        global_args = (global_cache, global_slots, global_lens)
    else:
        global_args = (None, None, None)

    def project_q(weights):
        return mm(
            x,
            weights[0],
            A_scales=x_scales,
            B_scales=weights[1],
            out_dtype=torch.bfloat16,
            quant="mxfp8",
            block_size=[1, 32],
            prepacked_scales=True,
        ).view(tokens, heads, 512)

    def run_fused():
        codes, scales = dsv41.selected_attention_fp8(
            project_q(permuted),
            *swa_args,
            *global_args,
            sink,
            _SOFTMAX_SCALE,
            positions,
            cos_sin,
            groups,
            workspace,
            workspace_slots,
        )
        projected = grouped_fp8_projection(
            codes, scales, permuted[2], permuted[3], None
        )
        projected_fp8 = grouped_fp8_projection_quantize_fp8(
            codes, scales, permuted[2], permuted[3]
        )
        return codes, scales, projected, *projected_fp8

    def check_reference(got):
        q = _rotate(project_q(original), positions, cos_sin)
        if workspace is not None:
            valid = (slots >= 0) & (slots < workspace.shape[0])
            kv = workspace[slots.clamp(0, workspace.shape[0] - 1), 0]
        else:
            kv, valid = _selected_rows(*swa_args, "swa")
            if layout == "global":
                extra_kv, extra_valid = _selected_rows(*global_args, "global")
                kv, valid = torch.cat((kv, extra_kv), dim=1), torch.cat(
                    (valid, extra_valid), dim=1
                )
        ref = _attention_reference(q, kv, valid, sink)
        plain = dsv41.selected_attention(
            q,
            *swa_args,
            *global_args,
            sink,
            _SOFTMAX_SCALE,
            None,
            16,
            dsv41.new_attention_schedule(),
            workspace,
            workspace_slots,
            lse=None,
        )
        torch.testing.assert_close(plain.float(), ref.float(), atol=0.008, rtol=0.012)

        codes, scales = dsv41.rope_quantize_fp8_grouped(
            plain, positions, inverse, groups
        )
        out = grouped_fp8_projection(codes, scales, original[2], original[3], None)
        out_fp8, out_scales = grouped_fp8_projection_quantize_fp8(
            codes, scales, original[2], original[3]
        )
        # Fused attention can round at a different BF16/FP8 boundary. Check the
        # complete projection's RMS and worst element rather than FP8 byte
        # identity across those boundaries.
        _assert_projection_close(got[2], out, relative_rms=0.035, max_error=0.16)
        # Compare each epilogue against its own BF16 input bit for bit. A
        # small cross-path BF16 difference can cross an FP8 rounding midpoint;
        # max error / global RMS is not a quantization bound (and changes when
        # zero rows are appended). Keep the cross-path RMS check unchanged.
        _assert_quantized_projection(got[2], got[3], got[4])
        _assert_quantized_projection(out, out_fp8, out_scales)
        _assert_projection_close(
            _dequantize(got[3], got[4]),
            _dequantize(out_fp8, out_scales),
            relative_rms=0.055,
            max_error=None,
        )
        empty = ~valid.any(dim=1)
        assert not got[0].float()[empty].count_nonzero().item()
        assert not got[2][empty].count_nonzero().item()

    initial = run_fused()
    check_reference(initial)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run_fused()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            replayed = run_fused()
    torch.cuda.current_stream().wait_stream(stream)

    x.view(torch.uint8).copy_(x.view(torch.uint8).flip(0))
    positions[:3].copy_(torch.tensor([23, 9, 1], device=device))
    slots[0, 0] = 9
    slots[2, 0] = 76
    lens[2] = 1
    if layout == "global":
        global_slots[0, 0] = 30
        global_lens[2] = 0

    refreshed = run_fused()
    check_reference(refreshed)
    graph.replay()
    assert not torch.equal(refreshed[2], initial[2])
    for got, want in zip(replayed, refreshed, strict=True):
        if got.dtype == _FP8:
            got, want = got.view(torch.uint8), want.view(torch.uint8)
        torch.testing.assert_close(got, want, atol=0, rtol=0)


@pytest.mark.parametrize("layout", ["swa", "workspace"])
@pytest.mark.parametrize("tokens", [0, 3])
def test_fused_decode_zero_capacity(device, layout, tokens):
    q = torch.zeros((tokens, 64, 512), device=device, dtype=torch.bfloat16)
    slots = torch.empty((tokens, 0), device=device, dtype=torch.int32)
    positions = torch.full((tokens,), -1, device=device, dtype=torch.int32)
    cos_sin = torch.cat(
        (torch.ones((1, 32), device=device), torch.zeros((1, 32), device=device)), dim=1
    )
    if layout == "workspace":
        workspace = torch.zeros((64, 1, 512), device=device, dtype=torch.bfloat16)
        swa_args = (None, None, None)
        workspace_args = (workspace, slots)
    else:
        cache = torch.zeros((1, 64, 528), device=device, dtype=torch.uint8)
        swa_args = (cache, slots, torch.zeros_like(positions))
        workspace_args = (None, None)
    codes, scales = dsv41.selected_attention_fp8(
        q,
        *swa_args,
        None,
        None,
        None,
        None,
        _SOFTMAX_SCALE,
        positions,
        cos_sin,
        8,
        *workspace_args,
    )
    assert codes.shape == (tokens, 8, 4096) and codes.dtype == _FP8
    assert scales.shape == (tokens, 8, 32) and scales.dtype == torch.int32
    assert not codes.float().count_nonzero().item()


def test_fused_decode_rejects_unsupported_output_group_width(device):
    # Native attention writes groups of eight heads even though its Python
    # signature accepts an arbitrary group count. Reject before entering it.
    q = torch.empty((0, 64, 512), dtype=torch.bfloat16, device=device)
    with pytest.raises(ValueError, match="8 heads"):
        dsv41.selected_attention_fp8(
            q,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            _SOFTMAX_SCALE,
            torch.empty(0, dtype=torch.int32, device=device),
            torch.zeros((1, 64), dtype=torch.float32, device=device),
            4,
            torch.empty((0, 1, 512), dtype=torch.bfloat16, device=device),
            torch.empty((0, 64), dtype=torch.int32, device=device),
        )
    with pytest.raises(ValueError, match="8 heads"):
        dsv41.prepare_fused_decode_weights(
            torch.empty((1, 128), dtype=_FP8, device=device),
            torch.empty((1, 4), dtype=torch.uint8, device=device),
            torch.empty((1, 128), dtype=_FP8, device=device),
            torch.empty((1, 1, 1), dtype=torch.int32, device=device),
            64,
            4,
        )
