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
from rel_mha_reference import (
    DTYPE,
    HEAD_DIM,
    NUM_KV_HEADS,
    NUM_Q_HEADS,
    PAGE,
    build_paged,
    cu_seqlens,
    ref_rel_attn,
    require_fa4,
)
from tokenspeed_kernel.ops.attention.mha import (
    mha_decode_with_kvcache,
    mha_extend_with_kvcache,
    mha_prefill,
)
from tokenspeed_kernel.ops.attention.rmha import (
    rel_mha_decode_with_kvcache,
    rel_mha_extend_with_kvcache,
    rel_mha_prefill,
)

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
    )

    for i, length in enumerate(q_lens):
        s = cu_cpu[i]
        ref = ref_rel_attn(
            q[s : s + length],
            k[s : s + length],
            v[s : s + length],
            rel_logits[s : s + length],
            rel_extent,
            window_left,
            scale,
        )
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
    )

    for i, (ql, _) in enumerate(zip(q_lens, kv_lens)):
        s = int(cu_q[i].item())
        ref = ref_rel_attn(
            q[s : s + ql],
            ks[i],
            vs[i],
            rel_logits[s : s + ql],
            rel_extent,
            window_left,
            scale,
        )
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
    )

    assert out.shape == q.shape
    for i in range(batch):
        ref = ref_rel_attn(
            q[i : i + 1],
            ks[i],
            vs[i],
            rel_logits[i : i + 1],
            rel_extent,
            window_left,
            scale,
        )
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
