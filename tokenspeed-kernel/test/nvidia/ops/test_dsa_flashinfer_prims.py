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

"""Validate Prims-TS DSA through the public selected-attention facade."""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel.ops.attention.dsa import dsa_decode, dsa_prefill
from tokenspeed_kernel.ops.attention.dsa.flashinfer_prims import has_prims_sparse_mla
from tokenspeed_kernel.platform import pdl_enabled
from tokenspeed_kernel.workspace import capture_session

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="requires SM100 or SM103",
)
requires_prims = pytest.mark.skipif(
    not has_prims_sparse_mla(),
    reason="requires a FlashInfer build with PR #7984",
)


@pytest.fixture(autouse=True)
def _exact_reference_math():
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    old_pdl = pdl_enabled()
    torch.backends.cuda.matmul.allow_tf32 = False
    pdl_enabled(True)
    yield
    torch.backends.cuda.matmul.allow_tf32 = old_tf32
    pdl_enabled(old_pdl)


def _call(q, kv, slots, lengths, *, mode, width, out, lse, k_scale, solution):
    kwargs = {"q_len_per_req": width} if mode == "decode" else {}
    return (dsa_decode if mode == "decode" else dsa_prefill)(
        q=q,
        kv_cache=kv,
        sparse_kv_cache=None,
        topk_slots=slots,
        topk_lens=lengths,
        max_seqlen_k=4096,
        qk_nope_head_dim=128,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        softmax_scale=192**-0.5,
        page_size=64,
        logit_cap=0.0,
        k_scale=k_scale,
        return_lse=lse,
        out=out,
        solution=solution,
        slot_order="selection",
        **kwargs,
    )


def _reference(q, kv, slots, lengths, k_scale):
    q = q.reshape(-1, q.shape[-2], 576).double()
    kv = kv.reshape(-1, 576).double()
    outputs, lses, bounds = [], [], []
    for row in range(q.shape[0]):
        visible = slots[row] if lengths is None else slots[row, : int(lengths[row])]
        selected = kv[visible[visible >= 0].long()]
        if selected.shape[0] == 0:
            outputs.append(
                torch.zeros((q.shape[1], 512), device=q.device, dtype=q.dtype)
            )
            lses.append(
                torch.full((q.shape[1],), -torch.inf, device=q.device, dtype=q.dtype)
            )
            bounds.append(torch.zeros_like(outputs[-1]))
        else:
            logits = q[row] @ selected.T * (192**-0.5) * k_scale
            probabilities = logits.softmax(-1)
            outputs.append(probabilities @ selected[:, :512])
            lses.append(logits.logsumexp(-1))
            # FP8 PV rounds P before the MMA: bound by P @ |V| plus E4M3 underflow.
            absolute_values = selected[:, :512].abs()
            sensitivity = probabilities @ absolute_values
            underflow = probabilities.amax(-1, keepdim=True) * absolute_values.sum(0)
            bounds.append(
                (2**-4 + 3 * 2**-8 + 1e-5) * sensitivity
                + (2**-10 / 448) * underflow
                + 1e-6
            )
    return torch.stack(outputs), torch.stack(lses), torch.stack(bounds)


def _check_fp8_output(actual, reference, bound):
    error = (actual.double() - reference).abs()
    assert torch.isfinite(actual).all()
    assert (error <= bound).all(), (
        f"FP8 error exceeds roundoff bound by "
        f"{(error / bound.clamp_min(1e-30)).max().item():.3f}x"
    )


@requires_prims
@pytest.mark.parametrize("heads", [12, 96])
@pytest.mark.parametrize(
    ("mode", "width"), [("decode", 1), ("decode", 6), ("decode", 8), ("prefill", 1)]
)
@pytest.mark.parametrize("with_lengths", [False, True])
def test_prims_sparse_attention_graph_and_reference(heads, mode, width, with_lengths):
    torch.manual_seed(7984)
    # Three requests also exercise a partially filled power-of-two plan.
    rows = 3 * width if mode == "decode" else 8
    shape = (3, width, heads, 576) if mode == "decode" else (rows, heads, 576)
    q = (torch.randn(shape, device="cuda") * 0.5).to(torch.float8_e4m3fn)
    kv = (torch.randn(4096, 576, device="cuda") * 0.25).to(torch.float8_e4m3fn)
    slots = torch.stack(
        [torch.randperm(4096, device="cuda")[:2048] for _ in range(rows)]
    ).int()
    # Holes before live entries, an empty row, and valid IDs past the length to ignore.
    slots[0].fill_(-1)
    slots[1, :64] = -1
    lengths = torch.full((rows,), 2048, device="cuda", dtype=torch.int32)
    if rows > 2:
        lengths[2] = 65
    lengths = lengths if with_lengths else None
    out = torch.empty((*shape[:-1], 512), device="cuda", dtype=torch.bfloat16)

    def run():
        return _call(
            q,
            kv,
            slots,
            lengths,
            mode=mode,
            width=width,
            out=out,
            lse=True,
            k_scale=0.7,
            solution="flashinfer_prims",
        )

    def check(output, lse):
        reference, ref_lse, bound = _reference(q, kv, slots, lengths, 0.7)
        assert output is out
        _check_fp8_output(output.reshape(rows, heads, 512), reference, bound)
        torch.testing.assert_close(lse.double(), ref_lse, rtol=1e-4, atol=2e-4)
        torch.testing.assert_close(
            output.reshape(rows, heads, 512)[0],
            torch.zeros_like(output.reshape(rows, heads, 512)[0]),
            rtol=0,
            atol=0,
        )
        assert torch.isneginf(lse[0]).all()

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        output, lse = run()
    stream.synchronize()
    check(output, lse)
    graph = torch.cuda.CUDAGraph()
    with capture_session(graph), torch.cuda.graph(graph, stream=stream):
        captured, captured_lse = run()
    q.copy_((torch.randn(shape, device="cuda") * 0.5).to(q.dtype))
    kv.copy_((torch.randn(kv.shape, device="cuda") * 0.25).to(kv.dtype))
    slots[1, 64:128] = -1
    if lengths is not None and rows > 2:
        lengths[2] = 127
    graph.replay()
    torch.cuda.synchronize()
    check(captured, captured_lse)


@requires_prims
@pytest.mark.parametrize("layout", ["flat", "latent", "nhd", "hnd"])
def test_prims_cache_layout_and_optional_output(layout):
    torch.manual_seed(79)
    q = (torch.randn(3, 12, 576, device="cuda") * 0.5).to(torch.float8_e4m3fn)
    kv = (torch.randn(4096, 576, device="cuda") * 0.25).to(q.dtype)
    cache = {
        "flat": kv,
        "latent": kv.view(4096, 1, 576),
        "nhd": kv.view(64, 64, 1, 576),
        "hnd": kv.view(64, 1, 64, 576),
    }[layout]
    slots = (
        torch.arange(512, device="cuda", dtype=torch.int32).expand(3, -1).contiguous()
    )
    actual = _call(
        q,
        cache,
        slots,
        None,
        mode="prefill",
        width=1,
        out=None,
        lse=False,
        k_scale=1.0,
        solution="flashinfer_prims",
    )
    expected, _, bound = _reference(q, kv, slots, None, 1.0)
    assert actual.shape == (3, 12, 512)
    _check_fp8_output(actual, expected, bound)


@requires_prims
def test_prims_separate_streams_keep_independent_scratch():
    torch.manual_seed(84)
    q = (torch.randn(8, 12, 576, device="cuda") * 0.5).to(torch.float8_e4m3fn)
    kv = (torch.randn(4096, 576, device="cuda") * 0.25).to(q.dtype)
    slots = (
        torch.arange(512, device="cuda", dtype=torch.int32).expand(8, -1).contiguous()
    )
    outputs = [
        torch.empty(8, 12, 512, device="cuda", dtype=torch.bfloat16) for _ in range(2)
    ]
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    graphs = [torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()]
    queries = [q, (-q.float()).to(q.dtype)]
    for stream, graph, query, output in zip(streams, graphs, queries, outputs):
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            _call(
                query,
                kv,
                slots,
                None,
                mode="decode",
                width=1,
                out=output,
                lse=False,
                k_scale=1.0,
                solution="flashinfer_prims",
            )
        stream.synchronize()
        with capture_session(graph), torch.cuda.graph(graph, stream=stream):
            _call(
                query,
                kv,
                slots,
                None,
                mode="decode",
                width=1,
                out=output,
                lse=False,
                k_scale=1.0,
                solution="flashinfer_prims",
            )
    for stream, graph in zip(streams, graphs):
        with torch.cuda.stream(stream):
            graph.replay()
    torch.cuda.synchronize()
    for query, output in zip(queries, outputs):
        expected, _, bound = _reference(query, kv, slots, None, 1.0)
        _check_fp8_output(output, expected, bound)


@requires_prims
def test_prims_rejects_disabled_pdl():
    q = torch.zeros(1, 12, 576, device="cuda", dtype=torch.float8_e4m3fn)
    kv = torch.zeros(64, 576, device="cuda", dtype=q.dtype)
    slots = torch.full((1, 512), -1, device="cuda", dtype=torch.int32)
    pdl_enabled(False)
    with pytest.raises(ValueError, match="requires PDL enabled"):
        _call(
            q,
            kv,
            slots,
            None,
            mode="decode",
            width=1,
            out=None,
            lse=False,
            k_scale=1.0,
            solution="flashinfer_prims",
        )


def test_fp8_decode_with_lse_meets_the_reference_on_the_default_kernel():
    # Decode CP needs an LSE: Prims-TS when FlashInfer provides it, Triton otherwise.
    torch.manual_seed(7)
    q = (torch.randn(12, 12, 576, device="cuda") * 0.5).to(torch.float8_e4m3fn)
    kv = (torch.randn(4096, 576, device="cuda") * 0.25).to(q.dtype)
    slots = torch.stack(
        [torch.randperm(4096, device="cuda")[:2048] for _ in range(12)]
    ).int()
    output, lse = _call(
        q,
        kv,
        slots,
        None,
        mode="decode",
        width=4,
        out=None,
        lse=True,
        k_scale=1.0,
        solution=None,
    )
    reference, ref_lse, bound = _reference(q, kv, slots, None, 1.0)
    _check_fp8_output(output, reference, bound)
    torch.testing.assert_close(lse.double(), ref_lse, rtol=1e-4, atol=2e-4)
