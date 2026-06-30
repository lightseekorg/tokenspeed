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

import math

import pytest
import torch
from tokenspeed_kernel import (
    dsa_decode,
    dsa_decode_topk,
    dsa_plan,
    dsa_prefill,
    dsa_prefill_topk,
)
from tokenspeed_kernel.ops.attention.triton.dsa_sparse_layout import (
    workspace_topk_to_global_slots as dsa_workspace_topk_to_global_slots,
)

torch.manual_seed(42)


def _pack_index_k_cache(
    index_k: torch.Tensor,
    page_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    head_dim = index_k.shape[1]
    num_groups = head_dim // 128
    row_bytes = head_dim + num_groups * 4
    num_slots = index_k.shape[0]
    num_pages = num_slots // page_size
    packed = torch.empty(
        (num_slots, row_bytes),
        device=index_k.device,
        dtype=torch.uint8,
    )
    x = index_k.float().reshape(num_slots, num_groups, 128)
    scale = x.abs().amax(dim=-1, keepdim=True).clamp_min(1.0e-6) / 448.0
    x_fp8 = (x / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)

    flat = packed.reshape(-1)
    page_bytes = page_size * row_bytes
    fp8_view = torch.as_strided(
        flat.view(torch.float8_e4m3fn),
        (num_pages, page_size, head_dim),
        (page_bytes, head_dim, 1),
    )
    scale_view = torch.as_strided(
        flat.view(torch.float32),
        (num_pages, page_size, num_groups),
        (page_bytes // 4, num_groups, 1),
        (page_size * head_dim) // 4,
    )
    fp8_view.copy_(x_fp8.reshape(num_pages, page_size, head_dim))
    scale_view.copy_(scale.reshape(num_pages, page_size, num_groups))
    return packed, (x_fp8.float() * scale).reshape_as(index_k)


@pytest.mark.parametrize("solution", ["triton", "deep_gemm"])
def test_dsa_decode_topk(device: str, solution: str, require) -> None:
    require("attention", "dsa_decode_topk", solution, torch.bfloat16, "q")
    if solution == "deep_gemm":
        pytest.skip("DeepGEMM DSA paged top-k requires packed FP8 cache fixture")

    page_size = 64
    topk = 512
    q = torch.randn((3, 2, 128), device=device, dtype=torch.bfloat16)
    weights = torch.randn((3, 2), device=device, dtype=torch.float32)
    index_k = torch.randn((4 * page_size, 128), device=device, dtype=torch.bfloat16)
    seq_lens = torch.tensor([20, 65, 3], device=device, dtype=torch.int32)
    block_table = torch.tensor(
        [[1, 3], [0, 2], [2, 1]], device=device, dtype=torch.int32
    )
    out = torch.empty((3, topk), device=device, dtype=torch.int32)
    lens_out = torch.empty((3,), device=device, dtype=torch.int32)

    topk_slots, topk_lens = dsa_decode_topk(
        q,
        weights,
        seq_lens,
        block_table,
        page_size=page_size,
        topk=topk,
        softmax_scale=128**-0.5,
        index_k_cache=index_k,
        out=out,
        lens_out=lens_out,
        solution=solution,
    )

    expected = torch.full_like(topk_slots, -1)
    expected_lens = torch.minimum(seq_lens, torch.full_like(seq_lens, topk))
    for token in range(q.shape[0]):
        scores = []
        slots = []
        for offset in range(int(seq_lens[token].item())):
            page = int(block_table[token, offset // page_size].item())
            slot = page * page_size + offset % page_size
            per_head = (q[token].float() * index_k[slot].float()).sum(dim=-1)
            scores.append((per_head * weights[token]).sum() * (128**-0.5))
            slots.append(slot)
        local = torch.topk(
            torch.stack(scores), int(expected_lens[token].item())
        ).indices
        expected[token, : local.numel()] = torch.tensor(
            [slots[int(i)] for i in local.tolist()], device=device, dtype=torch.int32
        )

    assert topk_slots.data_ptr() == out.data_ptr()
    assert topk_lens.data_ptr() == lens_out.data_ptr()
    torch.testing.assert_close(topk_lens.cpu(), expected_lens.cpu())
    torch.testing.assert_close(topk_slots[:, :65].cpu(), expected[:, :65].cpu())
    assert (topk_slots[0, int(expected_lens[0].item()) :] == -1).all()


def test_dsa_decode_topk_large_context(device: str, require) -> None:
    require("attention", "dsa_decode_topk", "triton", torch.bfloat16, "q")

    page_size = 64
    pages = 1024
    topk = 512
    q = torch.randn((1, 2, 128), device=device, dtype=torch.bfloat16)
    weights = torch.randn((1, 2), device=device, dtype=torch.float32)
    index_k = torch.randn((page_size, 128), device=device, dtype=torch.bfloat16)
    seq_lens = torch.tensor([5], device=device, dtype=torch.int32)
    block_table = torch.zeros((1, pages), device=device, dtype=torch.int32)

    topk_slots, topk_lens = dsa_decode_topk(
        q,
        weights,
        seq_lens,
        block_table,
        page_size=page_size,
        topk=topk,
        softmax_scale=128**-0.5,
        index_k_cache=index_k,
        solution="triton",
    )

    scores = []
    for offset in range(int(seq_lens[0].item())):
        per_head = (q[0].float() * index_k[offset].float()).sum(dim=-1)
        scores.append((per_head * weights[0]).sum() * (128**-0.5))
    expected = torch.topk(torch.stack(scores), int(seq_lens[0].item())).indices.to(
        torch.int32
    )

    torch.testing.assert_close(topk_lens.cpu(), torch.tensor([5], dtype=torch.int32))
    torch.testing.assert_close(topk_slots[0, :5].cpu(), expected.cpu())
    assert (topk_slots[0, 5:] == -1).all()


def test_dsa_decode_topk_fp8(device: str, require) -> None:
    require("attention", "dsa_decode_topk", "triton", torch.bfloat16, "q")

    page_size = 64
    topk = 512
    q = torch.randn((3, 2, 128), device=device, dtype=torch.bfloat16)
    weights = torch.randn((3, 2), device=device, dtype=torch.float32)
    packed_index_k, index_k = _pack_index_k_cache(
        torch.randn((4 * page_size, 128), device=device, dtype=torch.bfloat16),
        page_size,
    )
    seq_lens = torch.tensor([20, 65, 3], device=device, dtype=torch.int32)
    block_table = torch.tensor(
        [[1, 3], [0, 2], [2, 1]], device=device, dtype=torch.int32
    )

    topk_slots, topk_lens = dsa_decode_topk(
        q,
        weights,
        seq_lens,
        block_table,
        page_size=page_size,
        topk=topk,
        softmax_scale=128**-0.5,
        index_k_with_scale_cache=packed_index_k,
        solution="triton",
    )

    expected = torch.full_like(topk_slots, -1)
    expected_lens = torch.minimum(seq_lens, torch.full_like(seq_lens, topk))
    for token in range(q.shape[0]):
        scores = []
        slots = []
        for offset in range(int(seq_lens[token].item())):
            page = int(block_table[token, offset // page_size].item())
            slot = page * page_size + offset % page_size
            per_head = (q[token].float() * index_k[slot].float()).sum(dim=-1)
            scores.append((per_head * weights[token]).sum() * (128**-0.5))
            slots.append(slot)
        local = torch.topk(
            torch.stack(scores), int(expected_lens[token].item())
        ).indices
        expected[token, : local.numel()] = torch.tensor(
            [slots[int(i)] for i in local.tolist()], device=device, dtype=torch.int32
        )

    torch.testing.assert_close(topk_lens.cpu(), expected_lens.cpu())
    torch.testing.assert_close(topk_slots[:, :65].cpu(), expected[:, :65].cpu())
    assert (topk_slots[0, int(expected_lens[0].item()) :] == -1).all()


@pytest.mark.parametrize("solution", ["triton", "deep_gemm"])
def test_dsa_prefill_topk(device: str, solution: str, require) -> None:
    require("attention", "dsa_prefill_topk", solution, torch.bfloat16, "q")
    if solution == "deep_gemm":
        pytest.skip("DeepGEMM DSA workspace top-k requires gathered FP8 fixture")

    topk = 512
    q = torch.randn((3, 2, 128), device=device, dtype=torch.bfloat16)
    weights = torch.randn((3, 2), device=device, dtype=torch.float32)
    index_k = torch.randn((256, 128), device=device, dtype=torch.bfloat16)
    kv_workspace_slots = torch.arange(85, device=device, dtype=torch.int64) + 17
    row_starts = torch.tensor([0, 10, 70], device=device, dtype=torch.int32)
    row_ends = torch.tensor([20, 75, 85], device=device, dtype=torch.int32)
    out = torch.empty((3, topk), device=device, dtype=torch.int32)
    lens_out = torch.empty((3,), device=device, dtype=torch.int32)

    workspace_indices, topk_lens = dsa_prefill_topk(
        q,
        weights,
        kv_workspace_slots,
        row_starts,
        row_ends,
        topk=topk,
        softmax_scale=128**-0.5,
        index_k_cache=index_k,
        out=out,
        lens_out=lens_out,
        solution=solution,
    )

    expected = torch.full_like(workspace_indices, -1)
    expected_lens = torch.minimum(
        row_ends - row_starts, torch.full_like(row_ends, topk)
    )
    for token in range(q.shape[0]):
        scores = []
        rows = []
        for row in range(int(row_starts[token].item()), int(row_ends[token].item())):
            slot = int(kv_workspace_slots[row].item())
            per_head = (q[token].float() * index_k[slot].float()).sum(dim=-1)
            scores.append((per_head * weights[token]).sum() * (128**-0.5))
            rows.append(row)
        local = torch.topk(
            torch.stack(scores), int(expected_lens[token].item())
        ).indices
        expected[token, : local.numel()] = torch.tensor(
            [rows[int(i)] for i in local.tolist()], device=device, dtype=torch.int32
        )

    assert workspace_indices.data_ptr() == out.data_ptr()
    assert topk_lens.data_ptr() == lens_out.data_ptr()
    torch.testing.assert_close(topk_lens.cpu(), expected_lens.cpu())
    torch.testing.assert_close(workspace_indices[:, :65].cpu(), expected[:, :65].cpu())
    assert (workspace_indices[0, int(expected_lens[0].item()) :] == -1).all()


def test_dsa_prefill_topk_fp8(device: str, require) -> None:
    require("attention", "dsa_prefill_topk", "triton", torch.bfloat16, "q")

    page_size = 64
    topk = 512
    q = torch.randn((3, 2, 128), device=device, dtype=torch.bfloat16)
    weights = torch.randn((3, 2), device=device, dtype=torch.float32)
    packed_index_k, index_k = _pack_index_k_cache(
        torch.randn((4 * page_size, 128), device=device, dtype=torch.bfloat16),
        page_size,
    )
    kv_workspace_slots = torch.arange(85, device=device, dtype=torch.int64) + 17
    row_starts = torch.tensor([0, 10, 70], device=device, dtype=torch.int32)
    row_ends = torch.tensor([20, 75, 85], device=device, dtype=torch.int32)

    workspace_indices, topk_lens = dsa_prefill_topk(
        q,
        weights,
        kv_workspace_slots,
        row_starts,
        row_ends,
        topk=topk,
        softmax_scale=128**-0.5,
        index_k_with_scale_cache=packed_index_k,
        page_size=page_size,
        solution="triton",
    )

    expected = torch.full_like(workspace_indices, -1)
    expected_lens = torch.minimum(
        row_ends - row_starts, torch.full_like(row_ends, topk)
    )
    for token in range(q.shape[0]):
        scores = []
        rows = []
        for row in range(int(row_starts[token].item()), int(row_ends[token].item())):
            slot = int(kv_workspace_slots[row].item())
            per_head = (q[token].float() * index_k[slot].float()).sum(dim=-1)
            scores.append((per_head * weights[token]).sum() * (128**-0.5))
            rows.append(row)
        local = torch.topk(
            torch.stack(scores), int(expected_lens[token].item())
        ).indices
        expected[token, : local.numel()] = torch.tensor(
            [rows[int(i)] for i in local.tolist()], device=device, dtype=torch.int32
        )

    torch.testing.assert_close(topk_lens.cpu(), expected_lens.cpu())
    torch.testing.assert_close(workspace_indices[:, :65].cpu(), expected[:, :65].cpu())
    assert (workspace_indices[0, int(expected_lens[0].item()) :] == -1).all()


def test_dsa_plan_triton(device: str, require) -> None:
    require("attention", "dsa_plan", "triton", torch.int32, "seq_lens")

    seq_lens = torch.tensor([20, 65, 3], device=device, dtype=torch.int32)
    plan = dsa_plan(seq_lens, page_size=64, solution="triton")

    assert isinstance(plan, torch.Tensor)
    assert plan.shape == (3, 4)
    assert plan.dtype == torch.int32
    torch.testing.assert_close(plan[:, 0].cpu(), seq_lens.cpu())
    torch.testing.assert_close(plan[:, 1].cpu(), seq_lens.cpu())
    torch.testing.assert_close(plan[:, 2].cpu(), torch.full((3,), 1, dtype=torch.int32))
    torch.testing.assert_close(
        plan[:, 3].cpu(), torch.full((3,), 64, dtype=torch.int32)
    )

    plan_ptr = plan.data_ptr()
    refreshed = dsa_plan(seq_lens + 1, page_size=64, out=plan, solution="triton")
    assert refreshed is plan
    assert plan.data_ptr() == plan_ptr
    torch.testing.assert_close(plan[:, 0].cpu(), (seq_lens + 1).cpu())

    with pytest.raises(RuntimeError, match="DSA paged top-k plan changed shape"):
        dsa_plan(seq_lens[:2], page_size=64, out=plan, solution="triton")


def test_dsa_plan_returns_none_without_kernel(device: str) -> None:
    seq_lens = torch.tensor([1], device=device, dtype=torch.int32)

    assert dsa_plan(seq_lens, page_size=64, solution="missing") is None


def test_dsa_workspace_topk_to_global_slots(device: str) -> None:
    workspace_indices = torch.tensor(
        [[2, -1, 0], [1, 3, -1]],
        device=device,
        dtype=torch.int32,
    )
    kv_workspace_slots = torch.tensor(
        [10, 20, 30, 40],
        device=device,
        dtype=torch.int64,
    )
    out = torch.empty_like(workspace_indices)

    slots = dsa_workspace_topk_to_global_slots(
        workspace_indices=workspace_indices,
        kv_workspace_slots=kv_workspace_slots,
        out=out,
    )

    expected = torch.tensor(
        [[30, -1, 10], [20, 40, -1]],
        device=device,
        dtype=torch.int32,
    )
    assert slots.data_ptr() == out.data_ptr()
    torch.testing.assert_close(slots.cpu(), expected.cpu())


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


def _dsa_reference(
    q: torch.Tensor,
    latent: torch.Tensor,
    rope: torch.Tensor,
    topk_slots: torch.Tensor,
    topk_lens: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    refs = []
    kv_lora_rank = latent.shape[1]
    for token in range(q.shape[0]):
        valid_slots = topk_slots[token, : int(topk_lens[token].item())].long()
        q_nope = q[token, :, :kv_lora_rank].float()
        q_rope = q[token, :, kv_lora_rank:].float()
        k_nope = latent.index_select(0, valid_slots).float()
        k_rope = rope.index_select(0, valid_slots).float()
        scores = torch.einsum("hd,kd->hk", q_nope, k_nope)
        scores += torch.einsum("hd,kd->hk", q_rope, k_rope)
        probs = torch.softmax(scores * softmax_scale, dim=-1)
        refs.append(torch.matmul(probs, k_nope))
    return torch.stack(refs, dim=0).to(torch.bfloat16)


@pytest.mark.parametrize(
    "mode,api_name",
    [
        pytest.param("decode", "dsa_decode", id="decode"),
        pytest.param("prefill", "dsa_prefill", id="prefill"),
    ],
)
@pytest.mark.parametrize("solution", ["triton"])
@pytest.mark.parametrize(
    "q_dtype",
    [
        pytest.param(torch.bfloat16, id="q_bf16"),
        pytest.param(torch.float8_e4m3fn, id="q_fp8"),
    ],
)
def test_dsa_with_kvcache(
    device: str,
    mode: str,
    api_name: str,
    solution: str,
    q_dtype: torch.dtype,
    require,
) -> None:
    require("attention", api_name, solution, q_dtype, "q")

    tokens = 3
    num_heads = 2
    num_slots = 16
    topk = 512
    kv_lora_rank = 128
    qk_rope_head_dim = 64
    qk_nope_head_dim = 128
    softmax_scale = 1.0 / math.sqrt(qk_nope_head_dim + qk_rope_head_dim)
    q_bf16 = torch.randn(
        tokens,
        num_heads,
        kv_lora_rank + qk_rope_head_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    q = q_bf16.to(q_dtype)
    latent = torch.randn(num_slots, kv_lora_rank, device=device, dtype=torch.bfloat16)
    rope = torch.randn(num_slots, qk_rope_head_dim, device=device, dtype=torch.bfloat16)
    sparse_kv, dequant_latent = _pack_sparse_kv(latent, rope)
    topk_slots = torch.full((tokens, topk), -1, device=device, dtype=torch.int32)
    topk_lens = torch.tensor([5, 7, 4], device=device, dtype=torch.int32)
    for token in range(tokens):
        count = int(topk_lens[token].item())
        topk_slots[token, :count] = torch.randperm(num_slots, device=device)[:count]

    api = dsa_decode if mode == "decode" else dsa_prefill
    out = api(
        q=q,
        kv_cache=None,
        sparse_kv_cache=sparse_kv,
        topk_slots=topk_slots,
        topk_lens=topk_lens,
        max_seqlen_k=num_slots,
        qk_nope_head_dim=qk_nope_head_dim,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        softmax_scale=softmax_scale,
        page_size=64,
        solution=solution,
    )

    ref = _dsa_reference(
        q,
        dequant_latent,
        rope,
        topk_slots,
        topk_lens,
        softmax_scale,
    )
    assert out.shape == (tokens, num_heads, kv_lora_rank)
    assert out.dtype == torch.bfloat16
    torch.testing.assert_close(out.float(), ref.float(), rtol=8e-2, atol=8e-2)


@pytest.mark.parametrize(
    "q_dtype",
    [
        pytest.param(torch.bfloat16, id="q_bf16"),
        pytest.param(torch.float8_e4m3fn, id="q_fp8"),
    ],
)
def test_dsa_decode_dense_kvcache(device: str, q_dtype: torch.dtype, require) -> None:
    require("attention", "dsa_decode", "triton", q_dtype, "q")

    tokens = 3
    num_heads = 2
    num_slots = 16
    topk = 512
    kv_lora_rank = 128
    qk_rope_head_dim = 64
    qk_nope_head_dim = 128
    softmax_scale = 1.0 / math.sqrt(qk_nope_head_dim + qk_rope_head_dim)
    q_bf16 = torch.randn(
        tokens,
        num_heads,
        kv_lora_rank + qk_rope_head_dim,
        device=device,
        dtype=torch.bfloat16,
    )
    q = q_bf16.to(q_dtype)
    latent = torch.randn(num_slots, kv_lora_rank, device=device, dtype=torch.bfloat16)
    rope = torch.randn(num_slots, qk_rope_head_dim, device=device, dtype=torch.bfloat16)
    kv_cache = torch.cat([latent, rope], dim=-1).to(q_dtype)
    dequant_latent = kv_cache[:, :kv_lora_rank].float().to(torch.bfloat16)
    dequant_rope = kv_cache[:, kv_lora_rank:].float().to(torch.bfloat16)
    topk_slots = torch.full((tokens, topk), -1, device=device, dtype=torch.int32)
    topk_lens = torch.tensor([5, 7, 4], device=device, dtype=torch.int32)
    for token in range(tokens):
        count = int(topk_lens[token].item())
        topk_slots[token, :count] = torch.randperm(num_slots, device=device)[:count]

    out = dsa_decode(
        q=q,
        kv_cache=kv_cache,
        sparse_kv_cache=None,
        topk_slots=topk_slots,
        topk_lens=topk_lens,
        max_seqlen_k=num_slots,
        qk_nope_head_dim=qk_nope_head_dim,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        softmax_scale=softmax_scale,
        page_size=64,
        solution="triton",
    )

    ref = _dsa_reference(
        q,
        dequant_latent,
        dequant_rope,
        topk_slots,
        topk_lens,
        softmax_scale,
    )
    assert out.shape == (tokens, num_heads, kv_lora_rank)
    assert out.dtype == torch.bfloat16
    torch.testing.assert_close(out.float(), ref.float(), rtol=8e-2, atol=8e-2)
