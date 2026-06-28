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
    dsa_prefill,
    dsa_top_paged,
    dsa_topk,
)

torch.manual_seed(42)


@pytest.mark.parametrize("solution", ["triton", "deep_gemm"])
def test_dsa_top_paged(device: str, solution: str, require) -> None:
    require("attention", "dsa_top_paged", solution, torch.bfloat16, "q")
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

    topk_slots, topk_lens = dsa_top_paged(
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


@pytest.mark.parametrize("solution", ["triton", "deep_gemm"])
def test_dsa_topk(device: str, solution: str, require) -> None:
    require("attention", "dsa_topk", solution, torch.bfloat16, "q")
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

    workspace_indices, topk_lens = dsa_topk(
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
