from __future__ import annotations

import pytest
import torch

from tokenspeed_kernel.ops.attention.flash_mla import (
    flash_mla_with_kvcache,
    get_mla_metadata,
)
from tokenspeed_kernel.ops.attention.triton.dsa import (
    GLM_DSA_SPARSE_DECODE_ROW_BYTES,
    glm_dsa_pack_sparse_decode_kv,
)
from tokenspeed_kernel.registry import error_fn


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_flash_mla_sparse_decode_matches_fp8_reference() -> None:
    if flash_mla_with_kvcache is error_fn or get_mla_metadata is error_fn:
        pytest.skip("FlashMLA sparse decode is unavailable")

    torch.manual_seed(42)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    batch_size = 1
    num_q_heads = 64
    head_dim = 576
    value_dim = 512
    page_size = 64
    topk = 64
    num_pages = 1
    num_slots = page_size * num_pages
    softmax_scale = head_dim**-0.5

    q = (torch.randn(batch_size, 1, num_q_heads, head_dim, device=device) * 0.5).to(
        dtype
    )
    kv = (torch.randn(num_slots, head_dim, device=device) * 0.25).to(dtype)
    indices = torch.arange(topk, device=device, dtype=torch.int32).view(
        batch_size, 1, topk
    )

    packed = torch.zeros(
        (num_slots, GLM_DSA_SPARSE_DECODE_ROW_BYTES),
        device=device,
        dtype=torch.uint8,
    )
    glm_dsa_pack_sparse_decode_kv(
        out=packed,
        loc=torch.arange(num_slots, device=device),
        cache_k_nope=kv[:, :value_dim],
        cache_k_rope=kv[:, value_dim:],
    )

    metadata, _ = get_mla_metadata()
    out, _ = flash_mla_with_kvcache(
        q=q,
        k_cache=packed.view(num_pages, page_size, 1, GLM_DSA_SPARSE_DECODE_ROW_BYTES),
        block_table=None,
        cache_seqlens=None,
        head_dim_v=value_dim,
        tile_scheduler_metadata=metadata,
        softmax_scale=softmax_scale,
        is_fp8_kvcache=True,
        indices=indices,
    )

    rows = packed.view(num_slots, GLM_DSA_SPARSE_DECODE_ROW_BYTES)
    nope_fp8 = rows[:, :value_dim].contiguous().view(torch.float8_e4m3fn)
    nope = nope_fp8.view(num_slots, value_dim).float()
    scales = (
        rows[:, value_dim : value_dim + 16]
        .contiguous()
        .view(torch.float32)
        .view(num_slots, 4)
    )
    nope = (nope.view(num_slots, 4, 128) * scales.unsqueeze(-1)).view(
        num_slots, value_dim
    )
    rope = (
        rows[:, value_dim + 16 :]
        .contiguous()
        .view(torch.bfloat16)
        .view(num_slots, head_dim - value_dim)
        .float()
    )
    kv_ref = torch.cat([nope, rope], dim=-1)
    selected = kv_ref.index_select(0, indices[0, 0].long())
    logits = torch.einsum("hd,td->ht", q[0, 0].float(), selected) * softmax_scale
    probs = torch.softmax(logits, dim=-1)
    ref = torch.einsum("ht,td->hd", probs, selected[:, :value_dim]).view(
        batch_size, 1, num_q_heads, value_dim
    )

    diff = (out.float() - ref).abs()
    cosine = torch.nn.functional.cosine_similarity(
        out.float().flatten(),
        ref.flatten(),
        dim=0,
    )
    assert diff.mean().item() < 2.5e-4
    assert diff.max().item() < 2.5e-3
    assert cosine.item() > 0.999
