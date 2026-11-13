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

"""Relative-attention (rel_mha) operator tests.

The rel_mha family is MHA plus a learned per-query relative-distance
pre-softmax bias (TML/Inkling), taking the bias table as a first-class
``rel_logits`` tensor. How the bias is applied (fused sheared-bias FA4 path
vs generic score_mod gather) is an implementation detail of the registered
kernels; these tests pin the op outputs against a torch reference either
way. Extents that violate the fused-path constraints (not a multiple of
128) exercise the gather fallback; the fused-specific parity tests live in
test_attention_rel_bias_fused.py.

Also locks the plain mha ops' interface: no relative-bias arguments.
"""

from __future__ import annotations

import pytest
import torch
from l2_clip_helpers import triton_l2_clip
from rel_mha_helpers import (
    DTYPE,
    HEAD_DIM,
    NUM_KV_HEADS,
    NUM_Q_HEADS,
    PAGE,
    build_paged,
    cu_seqlens,
    require_fa4,
)
from tokenspeed_kernel.numerics.torch.attention.rmha import (
    rel_attention_logits,
    torch_rel_attention,
)
from tokenspeed_kernel.ops.attention.mha import (
    mha_decode_with_kvcache,
    mha_extend_with_kvcache,
    mha_prefill,
)
from tokenspeed_kernel.ops.attention.rmha import (
    rel_mha_decode_with_kvcache,
    rel_mha_extend_with_kvcache,
    rel_mha_gate_sink,
    rel_mha_head_norms,
    rel_mha_prefill,
)
from tokenspeed_kernel.ops.attention.rmha._triton.gate_sink import _gate_sink_kernel
from tokenspeed_kernel.ops.attention.rmha._triton.head_norms import (
    _rel_mha_head_norms_kernel,
)
from tokenspeed_kernel.ops.attention.rmha.triton import log_scaling_tau
from tokenspeed_kernel.ops.layernorm.precision import rmsnorm_precision
from utils import assert_no_triton_compile

torch.manual_seed(7)

TOL = 2e-2


@pytest.mark.parametrize(
    "solution,rel_extent,window_left",
    [
        (None, 64, -1),
        (None, 32, 31),
        ("triton", 64, -1),
        ("triton", 32, 31),
    ],
    ids=["default-full", "default-swa32", "triton-full", "triton-swa32"],
)
def test_rel_mha_prefill(
    device: str,
    require,
    solution: str | None,
    rel_extent: int,
    window_left: int,
) -> None:
    if solution is None:
        require_fa4(require)
    else:
        require("attention", "rel_mha_prefill", solution, DTYPE, "q")
    q_lens = [128, 200, 65]
    scale = 1.0 / HEAD_DIM
    total = sum(q_lens)
    q = torch.randn(total, NUM_Q_HEADS, HEAD_DIM, device=device, dtype=DTYPE) * 0.5
    k = torch.randn(total, NUM_KV_HEADS, HEAD_DIM, device=device, dtype=DTYPE) * 0.5
    v = torch.randn(total, NUM_KV_HEADS, HEAD_DIM, device=device, dtype=DTYPE) * 0.5
    rel_logits = (
        torch.randn(total, NUM_Q_HEADS, rel_extent, device=device, dtype=DTYPE) * 0.5
    )
    cu_cpu = [0]
    for length in q_lens:
        cu_cpu.append(cu_cpu[-1] + length)
    cu = torch.tensor(cu_cpu, device=device, dtype=torch.int32)

    out = rel_mha_prefill(
        q=q,
        k=k,
        v=v,
        rel_logits=rel_logits,
        cu_seqlens=cu,
        cu_seqlens_cpu=cu_cpu,
        max_seqlen=max(q_lens),
        window_left=window_left,
        softmax_scale=scale,
        solution=solution,
        output_dtype=None,
    )

    for i, length in enumerate(q_lens):
        s = cu_cpu[i]
        ref = torch_rel_attention(
            q[s : s + length],
            k[s : s + length],
            v[s : s + length],
            rel_logits[s : s + length],
            rel_extent,
            window_left,
            scale,
        )[0].to(DTYPE)
        err = (out[s : s + length].float() - ref.float()).abs().max().item()
        assert err < TOL, f"seq {i}: max_err={err:.4e}"


@pytest.mark.parametrize(
    "solution,page,rel_extent,q_lens,kv_lens,window_left",
    [
        (None, PAGE, 64, [64, 200, 1], [300, 400, 139], -1),
        ("gluon", 256, 512, [64, 32], [300, 511], 255),
        ("triton", PAGE, 64, [64, 32], [300, 139], 63),
    ],
    ids=["default", "gluon-page256", "triton-swa63"],
)
def test_rel_mha_extend_with_kvcache(
    device: str,
    require,
    solution: str | None,
    page: int,
    rel_extent: int,
    q_lens: list[int],
    kv_lens: list[int],
    window_left: int,
) -> None:
    """Paged extend with cached prefix (kv longer than q) exercises the
    seqlen_k - seqlen_q relative-distance offset."""
    if solution is None:
        require_fa4(require)
    else:
        require("attention", "rel_mha_extend_with_kvcache", solution, DTYPE, "q")
    scale = 1.0 / HEAD_DIM
    total_q = sum(q_lens)
    q = torch.randn(total_q, NUM_Q_HEADS, HEAD_DIM, device=device, dtype=DTYPE) * 0.5
    rel_logits = (
        torch.randn(total_q, NUM_Q_HEADS, rel_extent, device=device, dtype=DTYPE) * 0.5
    )
    cu_q = cu_seqlens(q_lens, device)
    cu_kv = cu_seqlens(kv_lens, device)
    cache_seqlens = torch.tensor(kv_lens, device=device, dtype=torch.int32)
    k_cache, v_cache, page_table, ks, vs = build_paged(kv_lens, device, page)

    out = rel_mha_extend_with_kvcache(
        q=q,
        cu_seqlens_q=cu_q,
        cu_seqlens_kv=cu_kv,
        k_cache=k_cache,
        v_cache=v_cache,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        max_seqlen_q=max(q_lens),
        max_seqlen_k=max(kv_lens),
        rel_logits=rel_logits,
        window_left=window_left,
        softmax_scale=scale,
        solution=solution,
        output_dtype=None,
    )

    for i, (ql, _) in enumerate(zip(q_lens, kv_lens)):
        s = int(cu_q[i].item())
        ref = torch_rel_attention(
            q[s : s + ql],
            ks[i],
            vs[i],
            rel_logits[s : s + ql],
            rel_extent,
            window_left,
            scale,
        )[0].to(DTYPE)
        err = (out[s : s + ql].float() - ref.float()).abs().max().item()
        assert err < TOL, f"seq {i}: max_err={err:.4e}"


@pytest.mark.parametrize(
    "solution,page,rel_extent,kv_lens,window_left",
    [
        (None, PAGE, 64, [77, 260, 129, 33] * 4, -1),
        ("gluon", 256, 512, [255, 256, 300, 511], 255),
        ("triton", PAGE, 64, [77, 260, 129, 33], -1),
        ("triton", PAGE, 64, [77, 260, 129, 33], 63),
    ],
    ids=["default", "gluon-page256", "triton-full", "triton-swa63"],
)
def test_rel_mha_decode_with_kvcache(
    device: str,
    require,
    solution: str | None,
    page: int,
    rel_extent: int,
    kv_lens: list[int],
    window_left: int,
    gate=None,
    sink=None,
) -> None:
    """Paged decode via the varlen path: cu_seqlens_q maps each request's
    query row into rel_logits at its batch-flattened position."""
    if solution is None:
        require_fa4(require)
    else:
        require("attention", "rel_mha_decode_with_kvcache", solution, DTYPE, "q")
    batch = len(kv_lens)
    scale = 1.0 / HEAD_DIM
    q = torch.randn(batch, NUM_Q_HEADS, HEAD_DIM, device=device, dtype=DTYPE) * 0.5
    rel_logits = (
        torch.randn(batch, NUM_Q_HEADS, rel_extent, device=device, dtype=DTYPE) * 0.5
    )
    cu_seqlens_q = torch.arange(batch + 1, device=device, dtype=torch.int32)
    cache_seqlens = torch.tensor(kv_lens, device=device, dtype=torch.int32)
    k_cache, v_cache, page_table, ks, vs = build_paged(kv_lens, device, page)

    out = rel_mha_decode_with_kvcache(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        max_seqlen_k=max(kv_lens),
        rel_logits=rel_logits,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=1,
        window_left=window_left,
        softmax_scale=scale,
        solution=solution,
        output_dtype=None,
        return_lse=False,
        gate=None,
        sink=None,
    )

    assert out.shape == q.shape
    for i in range(batch):
        ref = torch_rel_attention(
            q[i : i + 1],
            ks[i],
            vs[i],
            rel_logits[i : i + 1],
            rel_extent,
            window_left,
            scale,
        )[0].to(DTYPE)
        err = (out[i : i + 1].float() - ref.float()).abs().max().item()
        assert err < TOL, f"request {i}: max_err={err:.4e}"


def test_mha_ops_interface_has_no_rel_args(device: str, require) -> None:
    """The plain mha ops must not accept relative-bias arguments — that is
    the rel_mha family's contract. Use default dispatch so each architecture
    exercises a supported plain-MHA backend. Regression guard for interface
    creep."""
    require_fa4(require)
    q_lens = [70, 130]
    kv_lens = [198, 130]
    total_q = sum(q_lens)
    q = torch.randn(total_q, NUM_Q_HEADS, HEAD_DIM, device=device, dtype=DTYPE)
    k = torch.randn(total_q, NUM_KV_HEADS, HEAD_DIM, device=device, dtype=DTYPE)
    v = torch.randn(total_q, NUM_KV_HEADS, HEAD_DIM, device=device, dtype=DTYPE)
    cu_cpu = [0]
    for length in q_lens:
        cu_cpu.append(cu_cpu[-1] + length)
    cu = torch.tensor(cu_cpu, device=device, dtype=torch.int32)

    common = dict(
        q=q,
        k=k,
        v=v,
        cu_seqlens=cu,
        cu_seqlens_cpu=cu_cpu,
        max_seqlen=max(q_lens),
    )
    out = mha_prefill(**common)
    assert out.shape == q.shape
    assert not torch.isnan(out).any()
    for bad_kwarg in ("score_mod", "aux_tensors", "rel_logits"):
        with pytest.raises(TypeError):
            mha_prefill(**common, **{bad_kwarg: None})

    k_cache, v_cache, page_table, _, _ = build_paged(kv_lens, device, PAGE)
    cache_seqlens = torch.tensor(kv_lens, device=device, dtype=torch.int32)
    out = mha_extend_with_kvcache(
        q=q,
        cu_seqlens_q=cu,
        cu_seqlens_kv=cu_seqlens(kv_lens, device),
        k_cache=k_cache,
        v_cache=v_cache,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        max_seqlen_q=max(q_lens),
        max_seqlen_k=max(kv_lens),
        is_causal=True,
    )
    assert out.shape == q.shape
    assert not torch.isnan(out).any()

    q_decode = torch.randn(2, NUM_Q_HEADS, HEAD_DIM, device=device, dtype=DTYPE)
    out = mha_decode_with_kvcache(
        q=q_decode,
        k_cache=k_cache,
        v_cache=v_cache,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        max_seqlen_k=max(kv_lens),
        max_seqlen_q=1,
    )
    assert out.shape == q_decode.shape
    assert not torch.isnan(out).any()


@pytest.mark.parametrize("head_dim", [128, 96], ids=["d128", "d96"])
@pytest.mark.parametrize("window_left", [-1, 31], ids=["full", "swa32"])
def test_triton_rel_mha_prefill_lse(
    device: str, require, head_dim: int, window_left: int
) -> None:
    """Triton prefill output and LSE, including a non-power-of-two head dim."""
    require("attention", "rel_mha_prefill", "triton", DTYPE, "q")
    q_lens, rel_extent, scale = [150, 64, 1], 48, 1.0 / head_dim
    total = sum(q_lens)
    q = torch.randn(total, NUM_Q_HEADS, head_dim, device=device, dtype=DTYPE) * 0.5
    k = torch.randn(total, NUM_KV_HEADS, head_dim, device=device, dtype=DTYPE) * 0.5
    v = torch.randn(total, NUM_KV_HEADS, head_dim, device=device, dtype=DTYPE) * 0.5
    rel_logits = torch.randn(total, NUM_Q_HEADS, rel_extent, device=device, dtype=DTYPE)
    cu = cu_seqlens(q_lens, device)
    out, lse = rel_mha_prefill(
        q=q,
        k=k,
        v=v,
        rel_logits=rel_logits,
        cu_seqlens=cu,
        cu_seqlens_cpu=cu.tolist(),
        max_seqlen=max(q_lens),
        window_left=window_left,
        return_lse=True,
        softmax_scale=scale,
        solution="triton",
        output_dtype=None,
    )
    for s, e in zip(cu.tolist()[:-1], cu.tolist()[1:]):
        args = (q[s:e], k[s:e], rel_logits[s:e], rel_extent, window_left, scale)
        ref_lse = rel_attention_logits(*args).logsumexp(-1).transpose(0, 1)
        torch.testing.assert_close(lse[s:e], ref_lse, atol=1e-4, rtol=1e-4)
        ref = torch_rel_attention(q[s:e], k[s:e], v[s:e], *args[2:])[0].to(DTYPE)
        torch.testing.assert_close(out[s:e], ref, atol=TOL, rtol=0)


def test_triton_rel_mha_extend_lse_ignores_page_padding(device: str, require) -> None:
    """Multi-block windowed extend; NaN in page padding must not reach the output."""
    require("attention", "rel_mha_extend_with_kvcache", "triton", DTYPE, "q")
    q_lens, kv_lens, rel_extent, window_left = [150, 7], [430, 7], 64, 100
    scale = 1.0 / HEAD_DIM
    k_cache, v_cache, page_table, ks, vs = build_paged(kv_lens, device, PAGE)
    for i, length in enumerate(kv_lens):
        last_page, used = page_table[i, (length - 1) // PAGE], (length - 1) % PAGE + 1
        k_cache[last_page, used:] = float("nan")
        v_cache[last_page, used:] = float("nan")
    total = sum(q_lens)
    q = torch.randn(total, NUM_Q_HEADS, HEAD_DIM, device=device, dtype=DTYPE) * 0.5
    rel_logits = torch.randn(total, NUM_Q_HEADS, rel_extent, device=device, dtype=DTYPE)
    cu = cu_seqlens(q_lens, device)
    out, lse = rel_mha_extend_with_kvcache(
        q=q,
        cu_seqlens_q=cu,
        cu_seqlens_kv=cu_seqlens(kv_lens, device),
        k_cache=k_cache,
        v_cache=v_cache,
        page_table=page_table,
        cache_seqlens=torch.tensor(kv_lens, device=device, dtype=torch.int32),
        max_seqlen_q=max(q_lens),
        max_seqlen_k=max(kv_lens),
        rel_logits=rel_logits,
        window_left=window_left,
        return_lse=True,
        softmax_scale=scale,
        solution="triton",
        output_dtype=None,
    )
    for i, (s, e) in enumerate(zip(cu.tolist()[:-1], cu.tolist()[1:])):
        args = (rel_logits[s:e], rel_extent, window_left, scale)
        ref_lse = (
            rel_attention_logits(q[s:e], ks[i], *args).logsumexp(-1).transpose(0, 1)
        )
        torch.testing.assert_close(lse[s:e], ref_lse, atol=1e-4, rtol=1e-4)
        ref = torch_rel_attention(q[s:e], ks[i], vs[i], *args)[0].to(DTYPE)
        torch.testing.assert_close(out[s:e], ref, atol=TOL, rtol=0)


@pytest.mark.parametrize("mode", ["prefill", "extend", "decode"])
def test_triton_fp32_output_and_natural_lse(device: str, require, mode: str) -> None:
    operator = (
        "rel_mha_prefill" if mode == "prefill" else f"rel_mha_{mode}_with_kvcache"
    )
    require("attention", operator, "triton", DTYPE, "q")
    k_cache, v_cache, table, keys, values = build_paged([9], device, PAGE)
    q = torch.randn(1, NUM_Q_HEADS, HEAD_DIM, dtype=DTYPE, device=device) * 0.5
    relative = torch.randn(1, NUM_Q_HEADS, 8, dtype=DTYPE, device=device) * 0.25
    cu = cu_seqlens([1], device)
    options = dict(
        window_left=-1,
        softmax_scale=HEAD_DIM**-0.5,
        output_dtype=torch.float32,
        return_lse=True,
        solution="triton",
    )
    if mode == "prefill":
        k, v = keys[0][-1:], values[0][-1:]
        output, lse = rel_mha_prefill(q, k, v, relative, cu, [0, 1], 1, **options)
    else:
        k, v = keys[0], values[0]
        cached = dict(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            page_table=table,
            cache_seqlens=torch.tensor([9], device=device, dtype=torch.int32),
            max_seqlen_k=9,
            rel_logits=relative,
            cu_seqlens_q=cu,
            max_seqlen_q=1,
        )
        if mode == "extend":
            output, lse = rel_mha_extend_with_kvcache(
                **cached, cu_seqlens_kv=cu_seqlens([9], device), **options
            )
        else:
            output, lse = rel_mha_decode_with_kvcache(
                **cached, **options, gate=None, sink=None
            )
    scores = rel_attention_logits(q, k, relative, 8, -1, HEAD_DIM**-0.5)
    repeated_values = v.repeat_interleave(NUM_Q_HEADS // NUM_KV_HEADS, dim=1)
    expected = torch.einsum("hqk,khd->qhd", scores.softmax(-1), repeated_values.float())
    assert output.dtype == torch.float32
    assert lse.dtype == torch.float32
    torch.testing.assert_close(output, expected, atol=3e-3, rtol=3e-3)
    torch.testing.assert_close(
        lse, scores.logsumexp(-1).transpose(0, 1), atol=1e-4, rtol=1e-4
    )
    if mode != "prefill":
        assert not torch.equal(output, output.to(torch.bfloat16).float())


@pytest.mark.parametrize("group_size", (3, 6, 10))
@pytest.mark.parametrize("window_left", (-1, 7))
def test_triton_decode_partial_query_head_groups(
    device: str, require, group_size: int, window_left: int
) -> None:
    require("attention", "rel_mha_decode_with_kvcache", "triton", DTYPE, "q")
    heads, kv_heads, width, length, page = group_size * 2, 2, 128, 19, 16
    q = torch.randn((1, heads, width), dtype=DTYPE, device=device) * 0.5
    k = torch.randn((2, page, kv_heads, width), dtype=DTYPE, device=device) * 0.5
    v = torch.randn_like(k)
    relative = torch.randn((1, heads, 16), dtype=DTYPE, device=device) * 0.25
    keys, values = k.flatten(0, 1)[:length], v.flatten(0, 1)[:length]
    output, lse = rel_mha_decode_with_kvcache(
        q=q,
        k_cache=k,
        v_cache=v,
        page_table=torch.tensor([[0, 1]], dtype=torch.int32, device=device),
        cache_seqlens=torch.tensor([length], dtype=torch.int32, device=device),
        max_seqlen_k=length,
        rel_logits=relative,
        cu_seqlens_q=cu_seqlens([1], device),
        max_seqlen_q=1,
        window_left=window_left,
        softmax_scale=width**-0.5,
        output_dtype=torch.float32,
        return_lse=True,
        gate=None,
        sink=None,
        solution="triton",
    )
    scores = rel_attention_logits(q, keys, relative, 16, window_left, width**-0.5)
    expected = torch.einsum(
        "hqk,khd->qhd",
        scores.softmax(-1),
        values.repeat_interleave(group_size, 1).float(),
    )
    torch.testing.assert_close(output, expected, atol=3e-3, rtol=3e-3)
    torch.testing.assert_close(
        lse, scores.logsumexp(-1).transpose(0, 1), atol=1e-4, rtol=1e-4
    )


# NVIDIA follows the reference kernels' PTX sigmoid instead
# (nvidia/ops/attention/test_rel_mha_gate_sink_nvidia.py).
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is None,
    reason="the torch formula is the gate/sink epilogue on AMD",
)
# Past 256 tokens a program walks several tokens of its head.
@pytest.mark.parametrize("tokens", [37, 600])
def test_gate_sink_epilogue_matches_the_torch_formula_bitwise(tokens):
    generator = torch.Generator().manual_seed(11)
    heads, head_dim = 12, 128
    attention = torch.randn((tokens, heads * head_dim), generator=generator) * 4
    # Gate and sink are strided column slices of one projection, as in the model.
    projected = (
        torch.randn((tokens, heads * head_dim + heads + 64), generator=generator) * 8
    ).to(torch.bfloat16)
    gate = projected[:, : heads * head_dim]
    sink = projected[:, heads * head_dim : heads * head_dim + heads]
    lse = torch.randn((tokens, heads), generator=generator) * 20
    lse[0, 0], lse[1, 1] = float("inf"), -float("inf")
    actual = rel_mha_gate_sink(
        attention.cuda(),
        gate.cuda(),
        lse.cuda(),
        sink.cuda(),
        head_dim=head_dim,
        out_dtype=torch.bfloat16,
    )
    reference = attention.cuda().view(tokens, heads, head_dim) * torch.sigmoid(
        gate.cuda().view(tokens, heads, head_dim).float()
    )
    reference = reference * torch.sigmoid(lse.cuda() - sink.cuda().float()).unsqueeze(
        -1
    )
    assert torch.equal(actual, reference.flatten(1).to(torch.bfloat16))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires an admitted GPU")
@pytest.mark.parametrize("token_scale", [False, True])
@pytest.mark.parametrize("value_norm", [False, True])
# Past 256 tokens a program walks several tokens of its head.
@pytest.mark.parametrize("tokens", [19, 600])
def test_head_norms_match_the_separate_clips_and_norm_bitwise(
    token_scale, value_norm, tokens
):
    generator = torch.Generator(device="cuda").manual_seed(23)
    q_heads, kv_heads, head_dim, rel_dim = 12, 2, 128, 16

    def rows(width, scale):
        return (
            torch.randn((tokens, width), generator=generator, device="cuda") * scale
        ).to(torch.bfloat16)

    # Q and the relative heads are column slices of one wide projection row.
    projection = rows(q_heads * (head_dim + rel_dim) + 40, 3.0)
    q = projection[:, : q_heads * head_dim].view(tokens, q_heads, head_dim)
    rel = projection[:, q_heads * head_dim : q_heads * (head_dim + rel_dim)]
    rel = rel.view(tokens, q_heads, rel_dim)
    kv = rows(2 * kv_heads * head_dim, 40.0)
    k = kv[:, : kv_heads * head_dim].view(tokens, kv_heads, head_dim)
    v = kv[:, kv_heads * head_dim :].view(tokens, kv_heads, head_dim)
    q_scale = (
        torch.rand((q_heads, head_dim), generator=generator, device="cuda") + 0.5
    ).bfloat16()
    rel_scale = (
        torch.rand((q_heads, rel_dim), generator=generator, device="cuda") + 0.5
    ).bfloat16()
    tau = (
        torch.rand(tokens, generator=generator, device="cuda") + 1.0
        if token_scale
        else None
    )
    actual = rel_mha_head_norms(
        q,
        k,
        v,
        rel,
        q_scale=q_scale,
        k_scale=None,
        rel_scale=rel_scale,
        output_scale=tau,
        q_threshold=22.6,
        k_threshold=22.6,
        rel_threshold=8.0,
        clip_eps=1e-6,
        value_norm_eps=1e-6 if value_norm else None,
        rel_projection=None,
    )
    expected = (
        triton_l2_clip(q, 22.6, 1e-6, q_scale, tau),
        triton_l2_clip(k, 22.6, 1e-6, None, None),
        (
            rmsnorm_precision(v, None, 1e-6, None, v.dtype, mean_by_reciprocal=False)
            if value_norm
            else v
        ),
        triton_l2_clip(rel, 8.0, 1e-6, rel_scale, tau),
    )
    for got, want in zip(actual, expected):
        assert torch.equal(got, want)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires an admitted GPU")
@pytest.mark.parametrize("token_scale", [False, True])
@pytest.mark.parametrize("tokens", [1, 8, 19, 600])
def test_head_norms_project_relative_heads_like_the_matmul(token_scale, tokens):
    generator = torch.Generator(device="cuda").manual_seed(tokens)
    q_heads, kv_heads, head_dim, rel_dim, extent = 12, 2, 128, 16, 1024
    q = torch.randn((tokens, q_heads, head_dim), generator=generator, device="cuda")
    kv = torch.randn((tokens, kv_heads, head_dim), generator=generator, device="cuda")
    rel = torch.randn((tokens, q_heads, rel_dim), generator=generator, device="cuda")
    rel = (rel * 4.0).bfloat16()
    rel_scale = (
        torch.rand((q_heads, rel_dim), generator=generator, device="cuda") + 0.5
    ).bfloat16()
    projection = torch.randn((rel_dim, extent), generator=generator, device="cuda")
    projection = projection.bfloat16()
    tau = (
        torch.rand(tokens, generator=generator, device="cuda") + 1.0
        if token_scale
        else None
    )
    norms = dict(
        q_scale=None,
        k_scale=None,
        rel_scale=rel_scale,
        output_scale=tau,
        q_threshold=22.6,
        k_threshold=22.6,
        rel_threshold=8.0,
        clip_eps=1e-6,
        value_norm_eps=1e-6,
    )
    q, kv = q.bfloat16(), kv.bfloat16()
    *heads, logits = rel_mha_head_norms(
        q, kv, kv, rel, rel_projection=projection, **norms
    )
    *plain_heads, clipped = rel_mha_head_norms(
        q, kv, kv, rel, rel_projection=None, **norms
    )
    assert torch.equal(clipped, triton_l2_clip(rel, 8.0, 1e-6, rel_scale, tau))
    assert logits.shape == (tokens, q_heads, extent)
    assert torch.equal(logits, torch.matmul(clipped, projection))
    for got, want in zip(heads, plain_heads):
        assert torch.equal(got, want)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires an admitted GPU")
def test_head_epilogues_do_not_recompile_across_token_counts():
    projection = torch.randn((16, 1024), device="cuda").bfloat16()

    def run(tokens):
        q = torch.randn((tokens, 12, 128), device="cuda").bfloat16()
        k = torch.randn((tokens, 2, 128), device="cuda").bfloat16()
        rel = torch.randn((tokens, 12, 16), device="cuda").bfloat16()
        norms = dict(q_threshold=8.0, k_threshold=8.0, rel_threshold=8.0)
        rel_mha_head_norms(
            q,
            k,
            k.clone(),
            rel,
            q_scale=None,
            k_scale=None,
            rel_scale=None,
            output_scale=None,
            clip_eps=1e-6,
            value_norm_eps=1e-6,
            rel_projection=projection,
            **norms,
        )
        rel_mha_gate_sink(
            torch.randn((tokens, 1536), device="cuda"),
            q.view(tokens, -1),
            torch.randn((tokens, 12), device="cuda"),
            torch.randn((tokens, 12), device="cuda").bfloat16(),
            head_dim=128,
            out_dtype=torch.bfloat16,
        )

    # Fused (short) and GEMM (long) projection batches take one specialization each.
    run(8)
    run(129)
    with assert_no_triton_compile(_rel_mha_head_norms_kernel), assert_no_triton_compile(
        _gate_sink_kernel
    ):
        for tokens in (16, 33, 128, 300, 2048):
            run(tokens)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires an admitted GPU")
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize(("n_floor", "alpha"), [(128, 0.1), (100, 0.37)])
def test_log_scaling_tau_in_torch_order_is_the_torch_statement(dtype, n_floor, alpha):
    generator = torch.Generator().manual_seed(n_floor)
    positions = torch.cat(
        [
            torch.arange(300),
            torch.randint(0, 1 << 20, (4000,), generator=generator),
            torch.tensor([(1 << 24) - 1]),
        ]
    ).to(device="cuda", dtype=dtype)
    expected = 1.0 + alpha * ((positions + 1).float() / n_floor).clamp_min(1).log()
    exact = log_scaling_tau(positions, n_floor, alpha, torch_order=True)
    assert torch.equal(exact, expected)
    fused = log_scaling_tau(positions, n_floor, alpha, torch_order=False)
    torch.testing.assert_close(fused, expected, rtol=1e-6, atol=1e-6)
