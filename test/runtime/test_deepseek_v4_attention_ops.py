# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.

import math
import unittest

import torch

from tokenspeed.runtime.layers.attention.deepseek_v4_ops import (
    DEEPSEEK_V4_SWA_SCALE_DIM,
    DEEPSEEK_V4_SWA_TOKEN_STRIDE,
    DeepseekV4AttentionOpUnavailable,
    deepseek_v4_combine_dense_swa_indices,
    deepseek_v4_combine_topk_swa_indices,
    deepseek_v4_compressed_slot_mapping,
    deepseek_v4_compute_global_topk_indices_and_lens,
    deepseek_v4_csa_compress_kv_cache_insert,
    deepseek_v4_csa_indexer_cache_insert,
    deepseek_v4_decode_swa_indices_and_lens,
    deepseek_v4_dequantize_and_gather_k_cache,
    deepseek_v4_hca_compress_kv_cache_insert,
    deepseek_v4_indexer_topk_reference,
    deepseek_v4_prepare_indexer_q_mxfp4,
    deepseek_v4_prepare_indexer_q_reference,
    dequantize_deepseek_v4_fp8_ds_mla_cache,
    fused_qnorm_rope_kv_insert,
    read_deepseek_v4_indexer_fp8_cache,
    read_deepseek_v4_indexer_mxfp4_cache,
    save_deepseek_v4_compressor_state,
    write_deepseek_v4_indexer_fp8_cache,
    write_deepseek_v4_indexer_mxfp4_cache,
)

HEAD_DIM = 512
NOPE_DIM = 448
ROPE_DIM = 64
FP8_MAX = 448.0


def _apply_gptj_rope_with_nope(
    x: torch.Tensor,
    positions: torch.Tensor,
    cos_sin: torch.Tensor,
    nope_dim: int,
) -> torch.Tensor:
    out = x.float().clone()
    cos = cos_sin[positions, : ROPE_DIM // 2]
    sin = cos_sin[positions, ROPE_DIM // 2 :]
    even = out[..., nope_dim::2].clone()
    odd = out[..., nope_dim + 1 :: 2].clone()
    while cos.ndim < even.ndim:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    out[..., nope_dim::2] = even * cos - odd * sin
    out[..., nope_dim + 1 :: 2] = even * sin + odd * cos
    return out


def _apply_gptj_rope(
    x: torch.Tensor,
    positions: torch.Tensor,
    cos_sin: torch.Tensor,
) -> torch.Tensor:
    return _apply_gptj_rope_with_nope(x, positions, cos_sin, NOPE_DIM)


def _apply_inverse_gptj_rope_with_nope(
    x: torch.Tensor,
    positions: torch.Tensor,
    cos_sin: torch.Tensor,
    nope_dim: int,
) -> torch.Tensor:
    out = x.float().clone()
    rope_dim = out.shape[-1] - nope_dim
    cos = cos_sin[positions, : rope_dim // 2]
    sin = cos_sin[positions, rope_dim // 2 : rope_dim]
    even = out[..., nope_dim::2].clone()
    odd = out[..., nope_dim + 1 :: 2].clone()
    while cos.ndim < even.ndim:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    out[..., nope_dim::2] = even * cos + odd * sin
    out[..., nope_dim + 1 :: 2] = odd * cos - even * sin
    return out


def _q_reference(
    q: torch.Tensor, positions: torch.Tensor, cos_sin: torch.Tensor, eps: float
) -> torch.Tensor:
    q_float = q.float()
    scale = torch.rsqrt(q_float.square().mean(dim=-1, keepdim=True) + eps)
    return _apply_gptj_rope(q_float * scale, positions, cos_sin).to(q.dtype)


def _k_reference(
    kv: torch.Tensor, positions: torch.Tensor, cos_sin: torch.Tensor
) -> torch.Tensor:
    return _apply_gptj_rope(kv.float(), positions, cos_sin).to(kv.dtype)


def _fp8_bytes_and_scale(block: torch.Tensor) -> tuple[torch.Tensor, int]:
    absmax = max(float(block.abs().max()), 1.0e-4)
    exponent = math.ceil(math.log2(absmax / FP8_MAX))
    scaled = torch.clamp(block * (2.0**-exponent), -FP8_MAX, FP8_MAX)
    return scaled.to(torch.float8_e4m3fn).view(torch.uint8), int(
        max(min(exponent + 127, 255), 0)
    )


def _fp8_pow2_bytes_and_scale(block: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scale = max(float(block.abs().max()) / FP8_MAX, 1.0e-10)
    scale = torch.tensor(2.0 ** math.ceil(math.log2(scale)), device=block.device)
    scaled = torch.clamp(block / scale, -FP8_MAX, FP8_MAX)
    return scaled.to(torch.float8_e4m3fn).view(torch.uint8), scale.float()


def _e2m1_nibbles(x: torch.Tensor) -> torch.Tensor:
    abs_x = torch.clamp(x.abs(), max=6.0)
    code = torch.zeros_like(abs_x, dtype=torch.uint8)
    for idx, boundary in enumerate((0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)):
        code = torch.where(abs_x > boundary, idx + 1, code)
    sign = ((x < 0) & (code != 0)).to(torch.uint8)
    return code | (sign << 3)


def _e2m1_values(nibbles: torch.Tensor) -> torch.Tensor:
    table = nibbles.new_tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32
    )
    return table[(nibbles & 0x7).long()] * torch.where((nibbles & 0x8) != 0, -1.0, 1.0)


def _mxfp4_bytes_and_scales(
    row: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    packed_blocks = []
    scales = []
    dequant_blocks = []
    for block_id in range(4):
        block = row[block_id * 32 : (block_id + 1) * 32].float()
        even = block[0::2]
        odd = block[1::2]
        absmax = max(float(torch.maximum(even.abs().max(), odd.abs().max())), 1.0e-4)
        exponent = min(max(math.ceil(math.log2(absmax / 6.0)), -127), 127)
        scale = 2.0**exponent
        lo = _e2m1_nibbles(even / scale)
        hi = _e2m1_nibbles(odd / scale)
        packed = lo | (hi << 4)
        dequant = torch.empty(32, device=row.device, dtype=torch.float32)
        dequant[0::2] = _e2m1_values(lo) * scale
        dequant[1::2] = _e2m1_values(hi) * scale
        packed_blocks.append(packed)
        scales.append(row.new_tensor(exponent + 127, dtype=torch.uint8))
        dequant_blocks.append(dequant)
    return torch.cat(packed_blocks), torch.stack(scales), torch.cat(dequant_blocks)


def _hadamard_rotate(row: torch.Tensor) -> torch.Tensor:
    from tokenspeed_kernel.thirdparty.fast_hadamard_transform import hadamard_transform

    shape = row.shape
    return hadamard_transform(
        row.to(torch.bfloat16).reshape(-1, shape[-1]).contiguous(),
        scale=shape[-1] ** -0.5,
    ).reshape(shape)


def _expected_overlap_normed(
    kv: torch.Tensor,
    score: torch.Tensor,
    ape: torch.Tensor,
    position: int,
    compress_ratio: int,
    head_dim: int,
    rms_weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    window = 2 * compress_ratio
    state_width = 2 * head_dim
    assert kv.shape[-1] == state_width and score.shape[-1] == state_width
    ape_slots = ape.view(-1, head_dim)
    kv_rows = []
    score_rows = []
    for offset in range(window):
        pos = position - window + 1 + offset
        if pos < 0:
            continue
        head_offset = head_dim if offset >= compress_ratio else 0
        ape_offset = compress_ratio if offset >= compress_ratio else 0
        ape_slot = ape_offset + (pos % compress_ratio)
        kv_rows.append(kv[pos, head_offset : head_offset + head_dim].float())
        score_rows.append(
            score[pos, head_offset : head_offset + head_dim].float()
            + ape_slots[ape_slot]
        )
    weights = torch.softmax(torch.stack(score_rows, dim=0), dim=0)
    compressed = torch.sum(torch.stack(kv_rows, dim=0) * weights, dim=0)
    variance = compressed.square().sum() / head_dim
    return compressed * torch.rsqrt(variance + eps) * rms_weight


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class DeepseekV4AttentionOpsTest(unittest.TestCase):
    def test_fused_qnorm_rope_kv_insert_matches_reference(self):
        torch.manual_seed(1234)
        dtype = torch.bfloat16
        device = torch.device("cuda")
        num_tokens = 4
        num_insert = 3
        num_heads = 2
        block_size = 4
        eps = 1.0e-6

        q = torch.randn(num_tokens, num_heads, HEAD_DIM, device=device, dtype=dtype)
        kv = torch.randn(num_tokens, HEAD_DIM, device=device, dtype=dtype)
        q_before = q.clone()
        kv_before = kv.clone()
        positions = torch.tensor([0, 3, 5, 7], dtype=torch.int64, device=device)
        slot_mapping = torch.tensor([0, 2, -1], dtype=torch.int64, device=device)
        cos_sin = torch.randn(16, ROPE_DIM, device=device, dtype=torch.float32) * 0.1
        cache = torch.zeros(2, block_size * 584, device=device, dtype=torch.uint8)

        try:
            fused_qnorm_rope_kv_insert(
                q=q,
                kv=kv,
                swa_kv_cache_2d=cache,
                slot_mapping=slot_mapping,
                positions=positions,
                cos_sin_cache=cos_sin,
                rms_norm_eps=eps,
                block_size=block_size,
            )
        except DeepseekV4AttentionOpUnavailable as exc:
            self.skipTest(str(exc))
        torch.cuda.synchronize()

        expected_q = _q_reference(q_before, positions, cos_sin, eps)
        torch.testing.assert_close(
            q.float(), expected_q.float(), atol=3.0e-2, rtol=3.0e-2
        )

        expected_k = _k_reference(
            kv_before[:num_insert], positions[:num_insert], cos_sin
        )
        for token_idx, slot in enumerate(slot_mapping.tolist()):
            if slot < 0:
                continue
            block = slot // block_size
            pos = slot % block_size
            base = block * cache.stride(0) + pos * 576
            scale_base = block * cache.stride(0) + block_size * 576 + pos * 8
            flat_cache = cache.view(-1)
            token_bytes = flat_cache[base : base + 576]
            scale_bytes = flat_cache[scale_base : scale_base + 8]
            for qblock in range(7):
                start = qblock * 64
                expected_bytes, expected_scale = _fp8_bytes_and_scale(
                    expected_k[token_idx, start : start + 64].float()
                )
                torch.testing.assert_close(
                    token_bytes[start : start + 64].cpu(),
                    expected_bytes.cpu(),
                    atol=0,
                    rtol=0,
                )
                self.assertEqual(int(scale_bytes[qblock]), expected_scale)
            self.assertEqual(int(scale_bytes[7]), 0)
            expected_rope = expected_k[token_idx, NOPE_DIM:].view(torch.uint8)
            torch.testing.assert_close(
                token_bytes[NOPE_DIM:].cpu(),
                expected_rope.cpu(),
                atol=0,
                rtol=0,
            )

        # The fourth token was DP-style padding for KV insert: Q is still updated,
        # but no cache row is written for it.
        self.assertEqual(int(cache.view(-1)[3 * 576 : 4 * 576].sum()), 0)

    def test_hca_compressor_state_insert_matches_reference(self):
        torch.manual_seed(4321)
        device = torch.device("cuda")
        dtype = torch.bfloat16
        compress_ratio = 128
        state_block_size = 8
        kv_cache_block_size = 2
        num_tokens = compress_ratio
        num_state_blocks = num_tokens // state_block_size
        eps = 1.0e-6

        kv = torch.randn(num_tokens, HEAD_DIM, device=device, dtype=dtype)
        score = torch.randn(num_tokens, HEAD_DIM, device=device, dtype=dtype) * 0.1
        ape = (
            torch.randn(compress_ratio, HEAD_DIM, device=device, dtype=torch.float32)
            * 0.01
        )
        state_cache = torch.zeros(
            num_state_blocks,
            state_block_size,
            HEAD_DIM * 2,
            device=device,
            dtype=torch.float32,
        )
        positions = torch.arange(num_tokens, device=device, dtype=torch.int64)
        state_slots = positions.clone()

        save_deepseek_v4_compressor_state(
            kv=kv,
            score=score,
            ape=ape,
            state_cache=state_cache,
            slot_mapping=state_slots,
            positions=positions,
            block_size=state_block_size,
            compress_ratio=compress_ratio,
        )

        torch.testing.assert_close(
            state_cache[0, 0, :HEAD_DIM], kv[0].float(), atol=0, rtol=0
        )
        torch.testing.assert_close(
            state_cache[0, 0, HEAD_DIM:],
            score[0].float() + ape[0],
            atol=0,
            rtol=0,
        )

        token_to_req_indices = torch.zeros(num_tokens, device=device, dtype=torch.int32)
        block_table = torch.arange(
            num_state_blocks, device=device, dtype=torch.int32
        ).view(1, -1)
        kv_slots = torch.full((num_tokens,), -1, device=device, dtype=torch.int64)
        kv_slots[-2] = 1
        kv_slots[-1] = 0
        cos_sin = torch.randn(256, ROPE_DIM, device=device, dtype=torch.float32) * 0.05
        rms_weight = (
            torch.randn(HEAD_DIM, device=device, dtype=torch.float32) * 0.1 + 1.0
        )
        cache = torch.zeros(
            1,
            kv_cache_block_size
            * (DEEPSEEK_V4_SWA_TOKEN_STRIDE + DEEPSEEK_V4_SWA_SCALE_DIM),
            device=device,
            dtype=torch.uint8,
        )

        deepseek_v4_hca_compress_kv_cache_insert(
            state_cache=state_cache,
            token_to_req_indices=token_to_req_indices,
            positions=positions,
            compressor_slot_mapping=state_slots,
            block_table=block_table,
            compressor_block_size=state_block_size,
            rms_norm_weight=rms_weight,
            rms_norm_eps=eps,
            cos_sin_cache=cos_sin,
            kv_cache_2d=cache,
            kv_slot_mapping=kv_slots,
            kv_cache_block_size=kv_cache_block_size,
            compress_ratio=compress_ratio,
        )

        weights = torch.softmax(score.float() + ape, dim=0)
        compressed = torch.sum(kv.float() * weights, dim=0)
        variance = compressed.square().sum() / HEAD_DIM
        normed = compressed * torch.rsqrt(variance + eps) * rms_weight
        quant_input = normed.to(torch.bfloat16).float()
        expected_rope = (
            _apply_gptj_rope(
                normed.view(1, -1),
                torch.tensor([0], device=device, dtype=torch.int64),
                cos_sin,
            )[0, NOPE_DIM:]
            .to(torch.bfloat16)
            .view(torch.uint8)
        )

        flat_cache = cache.view(-1)
        scale_base = kv_cache_block_size * DEEPSEEK_V4_SWA_TOKEN_STRIDE
        for qblock in range(7):
            start = qblock * 64
            expected_bytes, expected_scale = _fp8_bytes_and_scale(
                quant_input[start : start + 64]
            )
            torch.testing.assert_close(
                flat_cache[start : start + 64].cpu(),
                expected_bytes.cpu(),
                atol=0,
                rtol=0,
            )
            self.assertEqual(int(flat_cache[scale_base + qblock]), expected_scale)
        self.assertEqual(int(flat_cache[scale_base + 7]), 0)
        torch.testing.assert_close(
            flat_cache[NOPE_DIM:DEEPSEEK_V4_SWA_TOKEN_STRIDE].cpu(),
            expected_rope.cpu(),
            atol=0,
            rtol=0,
        )
        self.assertEqual(
            int(
                flat_cache[
                    DEEPSEEK_V4_SWA_TOKEN_STRIDE : 2 * DEEPSEEK_V4_SWA_TOKEN_STRIDE
                ].sum()
            ),
            0,
        )

    def test_csa_compressor_state_insert_matches_reference(self):
        torch.manual_seed(5678)
        device = torch.device("cuda")
        dtype = torch.bfloat16
        compress_ratio = 4
        state_width = HEAD_DIM * 2
        state_block_size = 8
        kv_cache_block_size = 2
        num_tokens = compress_ratio * 2
        eps = 1.0e-6

        kv = torch.randn(num_tokens, state_width, device=device, dtype=dtype)
        score = torch.randn(num_tokens, state_width, device=device, dtype=dtype) * 0.1
        ape = (
            torch.randn(compress_ratio, state_width, device=device, dtype=torch.float32)
            * 0.01
        )
        state_cache = torch.zeros(
            1,
            state_block_size,
            state_width * 2,
            device=device,
            dtype=torch.float32,
        )
        positions = torch.arange(num_tokens, device=device, dtype=torch.int64)
        state_slots = positions.clone()

        save_deepseek_v4_compressor_state(
            kv=kv,
            score=score,
            ape=ape,
            state_cache=state_cache,
            slot_mapping=state_slots,
            positions=positions,
            block_size=state_block_size,
            compress_ratio=compress_ratio,
        )

        token_to_req_indices = torch.zeros(num_tokens, device=device, dtype=torch.int32)
        block_table = torch.zeros(1, 1, device=device, dtype=torch.int32)
        kv_slots = torch.full((num_tokens,), -1, device=device, dtype=torch.int64)
        kv_slots[compress_ratio - 1] = 0
        kv_slots[num_tokens - 1] = 1
        cos_sin = torch.randn(16, ROPE_DIM, device=device, dtype=torch.float32) * 0.05
        rms_weight = (
            torch.randn(HEAD_DIM, device=device, dtype=torch.float32) * 0.1 + 1.0
        )
        cache = torch.zeros(
            1,
            kv_cache_block_size
            * (DEEPSEEK_V4_SWA_TOKEN_STRIDE + DEEPSEEK_V4_SWA_SCALE_DIM),
            device=device,
            dtype=torch.uint8,
        )

        deepseek_v4_csa_compress_kv_cache_insert(
            state_cache=state_cache,
            token_to_req_indices=token_to_req_indices,
            positions=positions,
            compressor_slot_mapping=state_slots,
            block_table=block_table,
            compressor_block_size=state_block_size,
            rms_norm_weight=rms_weight,
            rms_norm_eps=eps,
            cos_sin_cache=cos_sin,
            kv_cache_2d=cache,
            kv_slot_mapping=kv_slots,
            kv_cache_block_size=kv_cache_block_size,
            compress_ratio=compress_ratio,
        )

        flat_cache = cache.view(-1)
        for slot, position in ((0, compress_ratio - 1), (1, num_tokens - 1)):
            normed = _expected_overlap_normed(
                kv, score, ape, position, compress_ratio, HEAD_DIM, rms_weight, eps
            )
            quant_input = normed.to(torch.bfloat16).float()
            expected_rope = (
                _apply_gptj_rope(
                    normed.view(1, -1),
                    torch.tensor(
                        [(position // compress_ratio) * compress_ratio],
                        device=device,
                        dtype=torch.int64,
                    ),
                    cos_sin,
                )[0, NOPE_DIM:]
                .to(torch.bfloat16)
                .view(torch.uint8)
            )
            base = slot * DEEPSEEK_V4_SWA_TOKEN_STRIDE
            scale_base = kv_cache_block_size * DEEPSEEK_V4_SWA_TOKEN_STRIDE
            scale_base += slot * DEEPSEEK_V4_SWA_SCALE_DIM
            for qblock in range(7):
                start = qblock * 64
                expected_bytes, expected_scale = _fp8_bytes_and_scale(
                    quant_input[start : start + 64]
                )
                torch.testing.assert_close(
                    flat_cache[base + start : base + start + 64].cpu(),
                    expected_bytes.cpu(),
                    atol=0,
                    rtol=0,
                )
                self.assertEqual(int(flat_cache[scale_base + qblock]), expected_scale)
            self.assertEqual(int(flat_cache[scale_base + 7]), 0)
            torch.testing.assert_close(
                flat_cache[base + NOPE_DIM : base + DEEPSEEK_V4_SWA_TOKEN_STRIDE].cpu(),
                expected_rope.cpu(),
                atol=0,
                rtol=0,
            )

    def test_indexer_fp8_cache_and_topk_reference(self):
        torch.manual_seed(6789)
        device = torch.device("cuda")
        dtype = torch.bfloat16
        block_size = 64

        index_k = torch.randn(3, 128, device=device, dtype=dtype)
        cache = torch.zeros(1, block_size * 132, device=device, dtype=torch.uint8)
        slots = torch.tensor([0, 2, -1], device=device, dtype=torch.int64)

        write_deepseek_v4_indexer_fp8_cache(index_k, cache, slots, block_size)

        flat_cache = cache.view(-1)
        for token_idx, slot in enumerate(slots.tolist()):
            if slot < 0:
                continue
            value_base = slot * 128
            scale_base = block_size * 128 + slot * 4
            expected_bytes, expected_scale = _fp8_pow2_bytes_and_scale(
                index_k[token_idx].float()
            )
            torch.testing.assert_close(
                flat_cache[value_base : value_base + 128].cpu(),
                expected_bytes.cpu(),
                atol=0,
                rtol=0,
            )
            torch.testing.assert_close(
                flat_cache[scale_base : scale_base + 4].cpu(),
                expected_scale.reshape(1).view(torch.uint8).cpu(),
                atol=0,
                rtol=0,
            )
        self.assertEqual(int(flat_cache[128:256].sum()), 0)

        q = torch.randn(4, 3, 128, device=device, dtype=dtype)
        k = torch.randn(6, 128, device=device, dtype=dtype)
        weights = torch.randn(4, 3, device=device, dtype=torch.float32)
        lengths = torch.tensor([3, 3, 2, 2], device=device, dtype=torch.int64)
        row_starts = torch.tensor([0, 1, 2, 3], device=device, dtype=torch.int64)

        topk = deepseek_v4_indexer_topk_reference(
            q, k, weights, top_k=2, lengths=lengths, row_starts=row_starts
        )

        logits = torch.einsum("thd,kd->thk", q.float(), k.float()).relu()
        logits = (logits * weights.unsqueeze(-1)).sum(dim=1)
        cols = torch.arange(k.shape[0], device=device)
        valid = (cols.unsqueeze(0) >= row_starts.unsqueeze(1)) & (
            cols.unsqueeze(0) < (row_starts + lengths).unsqueeze(1)
        )
        logits = logits.masked_fill(~valid, -float("inf"))
        expected = torch.topk(logits, k=2, dim=-1, sorted=False).indices.to(torch.int32)
        torch.testing.assert_close(topk.cpu(), expected.cpu(), atol=0, rtol=0)

    def test_indexer_mxfp4_cache_matches_reference(self):
        torch.manual_seed(7890)
        device = torch.device("cuda")
        dtype = torch.bfloat16
        block_size = 64

        index_k = torch.randn(3, 128, device=device, dtype=dtype)
        cache = torch.zeros(1, block_size * 68, device=device, dtype=torch.uint8)
        slots = torch.tensor([0, 2, -1], device=device, dtype=torch.int64)

        write_deepseek_v4_indexer_mxfp4_cache(index_k, cache, slots, block_size)

        flat_cache = cache.view(-1)
        for token_idx, slot in enumerate(slots.tolist()):
            if slot < 0:
                continue
            value_base = slot * 64
            scale_base = block_size * 64 + slot * 4
            expected_bytes, expected_scales, expected_dequant = _mxfp4_bytes_and_scales(
                index_k[token_idx]
            )
            torch.testing.assert_close(
                flat_cache[value_base : value_base + 64].cpu(),
                expected_bytes.cpu(),
                atol=0,
                rtol=0,
            )
            torch.testing.assert_close(
                flat_cache[scale_base : scale_base + 4].cpu(),
                expected_scales.cpu(),
                atol=0,
                rtol=0,
            )
            dequant = read_deepseek_v4_indexer_mxfp4_cache(
                cache, slots[token_idx : token_idx + 1], block_size
            )[0]
            torch.testing.assert_close(
                dequant.cpu(),
                expected_dequant.cpu(),
                atol=0,
                rtol=0,
            )
        self.assertEqual(int(flat_cache[64:128].sum()), 0)

    def test_csa_indexer_cache_insert_matches_reference(self):
        torch.manual_seed(8901)
        device = torch.device("cuda")
        dtype = torch.bfloat16
        compress_ratio = 4
        head_dim = 128
        state_width = head_dim * 2
        state_block_size = 8
        kv_cache_block_size = 64
        num_tokens = compress_ratio * 2
        eps = 1.0e-6

        kv = torch.randn(num_tokens, state_width, device=device, dtype=dtype)
        score = torch.randn(num_tokens, state_width, device=device, dtype=dtype) * 0.1
        ape = (
            torch.randn(compress_ratio, state_width, device=device, dtype=torch.float32)
            * 0.01
        )
        state_cache = torch.zeros(
            1,
            state_block_size,
            state_width * 2,
            device=device,
            dtype=torch.float32,
        )
        positions = torch.arange(num_tokens, device=device, dtype=torch.int64)
        state_slots = positions.clone()
        save_deepseek_v4_compressor_state(
            kv=kv,
            score=score,
            ape=ape,
            state_cache=state_cache,
            slot_mapping=state_slots,
            positions=positions,
            block_size=state_block_size,
            compress_ratio=compress_ratio,
        )

        token_to_req_indices = torch.zeros(num_tokens, device=device, dtype=torch.int32)
        block_table = torch.zeros(1, 1, device=device, dtype=torch.int32)
        kv_slots = torch.full((num_tokens,), -1, device=device, dtype=torch.int64)
        kv_slots[compress_ratio - 1] = 0
        kv_slots[num_tokens - 1] = 1
        cos_sin = torch.randn(16, ROPE_DIM, device=device, dtype=torch.float32) * 0.05
        rms_weight = (
            torch.randn(head_dim, device=device, dtype=torch.float32) * 0.1 + 1.0
        )
        cache_fp4 = torch.zeros(
            1, kv_cache_block_size * 68, device=device, dtype=torch.uint8
        )

        deepseek_v4_csa_indexer_cache_insert(
            state_cache=state_cache,
            token_to_req_indices=token_to_req_indices,
            positions=positions,
            compressor_slot_mapping=state_slots,
            block_table=block_table,
            compressor_block_size=state_block_size,
            rms_norm_weight=rms_weight,
            rms_norm_eps=eps,
            cos_sin_cache=cos_sin,
            kv_cache_2d=cache_fp4,
            kv_slot_mapping=kv_slots,
            kv_cache_block_size=kv_cache_block_size,
            use_fp4_cache=True,
            compress_ratio=compress_ratio,
        )

        for slot, position in ((0, compress_ratio - 1), (1, num_tokens - 1)):
            normed = _expected_overlap_normed(
                kv, score, ape, position, compress_ratio, head_dim, rms_weight, eps
            )
            rotated = _apply_gptj_rope_with_nope(
                normed.view(1, -1),
                torch.tensor(
                    [(position // compress_ratio) * compress_ratio],
                    device=device,
                    dtype=torch.int64,
                ),
                cos_sin,
                head_dim - ROPE_DIM,
            )[0]
            rotated = _hadamard_rotate(rotated)
            _, _, expected_dequant = _mxfp4_bytes_and_scales(rotated)
            dequant = read_deepseek_v4_indexer_mxfp4_cache(
                cache_fp4,
                torch.tensor([slot], device=device, dtype=torch.int64),
                kv_cache_block_size,
            )[0]
            torch.testing.assert_close(
                dequant.cpu(), expected_dequant.cpu(), atol=0, rtol=0
            )

        cache_fp8 = torch.zeros(
            1, kv_cache_block_size * 132, device=device, dtype=torch.uint8
        )
        deepseek_v4_csa_indexer_cache_insert(
            state_cache=state_cache,
            token_to_req_indices=token_to_req_indices,
            positions=positions,
            compressor_slot_mapping=state_slots,
            block_table=block_table,
            compressor_block_size=state_block_size,
            rms_norm_weight=rms_weight,
            rms_norm_eps=eps,
            cos_sin_cache=cos_sin,
            kv_cache_2d=cache_fp8,
            kv_slot_mapping=kv_slots,
            kv_cache_block_size=kv_cache_block_size,
            use_fp4_cache=False,
            compress_ratio=compress_ratio,
        )
        fp8_rows = read_deepseek_v4_indexer_fp8_cache(
            cache_fp8,
            torch.tensor([0, 1], device=device, dtype=torch.int64),
            kv_cache_block_size,
        )
        self.assertGreater(float(fp8_rows.abs().sum()), 0.0)

    def test_fp8_ds_mla_cache_dequant_and_inv_rope_output_reference(self):
        torch.manual_seed(9012)
        device = torch.device("cuda")
        dtype = torch.bfloat16
        block_size = 4
        slots = torch.tensor([0, 2, -1], device=device, dtype=torch.int64)
        positions = torch.tensor([0, 3, 7], device=device, dtype=torch.int64)
        cos_sin = torch.randn(16, ROPE_DIM, device=device, dtype=torch.float32) * 0.05
        rows = torch.randn(3, HEAD_DIM, device=device, dtype=dtype)
        cache = torch.zeros(
            1,
            block_size * (DEEPSEEK_V4_SWA_TOKEN_STRIDE + DEEPSEEK_V4_SWA_SCALE_DIM),
            device=device,
            dtype=torch.uint8,
        )
        for token_idx, slot in enumerate(slots.tolist()):
            if slot < 0:
                continue
            flat = cache.view(-1)
            token_base = slot * DEEPSEEK_V4_SWA_TOKEN_STRIDE
            scale_base = (
                block_size * DEEPSEEK_V4_SWA_TOKEN_STRIDE
                + slot * DEEPSEEK_V4_SWA_SCALE_DIM
            )
            rotated = _apply_gptj_rope(
                rows[token_idx : token_idx + 1],
                positions[token_idx : token_idx + 1],
                cos_sin,
            )[0]
            for qblock in range(7):
                start = qblock * 64
                fp8_bytes, scale = _fp8_bytes_and_scale(
                    rotated[start : start + 64].float()
                )
                flat[token_base + start : token_base + start + 64].copy_(fp8_bytes)
                flat[scale_base + qblock] = scale
            flat[
                token_base + NOPE_DIM : token_base + DEEPSEEK_V4_SWA_TOKEN_STRIDE
            ].copy_(rotated[NOPE_DIM:].to(torch.bfloat16).view(torch.uint8))

        dequant = dequantize_deepseek_v4_fp8_ds_mla_cache(cache, slots, block_size)
        self.assertEqual(float(dequant[2].abs().sum()), 0.0)
        self.assertGreater(float(dequant[:2].abs().sum()), 0.0)

    def test_indexer_q_prepare_matches_fp4_weight_contract(self):
        torch.manual_seed(9123)
        device = torch.device("cuda")
        dtype = torch.bfloat16
        positions = torch.tensor([1, 5], device=device, dtype=torch.int64)
        cos_sin = torch.randn(16, ROPE_DIM, device=device, dtype=torch.float32) * 0.05
        q = torch.randn(2, 3, 128, device=device, dtype=dtype)
        weights = torch.randn(2, 3, device=device, dtype=torch.float32)
        q_fp4, weights_fp4 = deepseek_v4_prepare_indexer_q_reference(
            q, positions, cos_sin, weights, 0.25, 3**-0.5, use_fp4=True
        )
        self.assertEqual(q_fp4.shape, q.shape)
        rotated = _apply_gptj_rope_with_nope(
            q,
            positions,
            cos_sin,
            nope_dim=128 - ROPE_DIM,
        )
        rotated = _hadamard_rotate(rotated)
        expected_fp4 = torch.empty_like(rotated, dtype=torch.float32)
        for token_idx in range(rotated.shape[0]):
            for head_idx in range(rotated.shape[1]):
                _, _, dequant = _mxfp4_bytes_and_scales(rotated[token_idx, head_idx])
                expected_fp4[token_idx, head_idx].copy_(dequant)
        torch.testing.assert_close(
            q_fp4.float().cpu(),
            expected_fp4.to(torch.bfloat16).float().cpu(),
            atol=0,
            rtol=0,
        )
        torch.testing.assert_close(
            weights_fp4.cpu(),
            (weights * 0.25 * (3**-0.5)).cpu(),
            atol=1.0e-6,
            rtol=1.0e-6,
        )
        q_fp8, weights_fp8 = deepseek_v4_prepare_indexer_q_reference(
            q, positions, cos_sin, weights, 0.25, 3**-0.5, use_fp4=False
        )
        self.assertEqual(q_fp8.shape, q.shape)
        self.assertGreater(float(weights_fp8.abs().sum()), 0.0)

    def test_indexer_q_prepare_mxfp4_returns_deepgemm_layout(self):
        torch.manual_seed(9124)
        device = torch.device("cuda")
        dtype = torch.bfloat16
        positions = torch.tensor([1, 5], device=device, dtype=torch.int64)
        cos_sin = torch.randn(16, ROPE_DIM, device=device, dtype=torch.float32) * 0.05
        q = torch.randn(2, 3, 128, device=device, dtype=dtype)
        weights = torch.randn(2, 3, device=device, dtype=torch.float32)

        (q_packed, q_scales), weights_out = deepseek_v4_prepare_indexer_q_mxfp4(
            q,
            positions,
            cos_sin,
            weights,
            0.25,
            3**-0.5,
        )

        self.assertEqual(q_packed.shape, (2, 3, 64))
        self.assertEqual(q_packed.dtype, torch.uint8)
        self.assertEqual(q_scales.shape, (2, 3))
        self.assertEqual(q_scales.dtype, torch.int32)
        torch.testing.assert_close(
            weights_out.cpu(),
            (weights * 0.25 * (3**-0.5)).cpu(),
            atol=1.0e-6,
            rtol=1.0e-6,
        )

        rotated = _apply_gptj_rope_with_nope(
            q,
            positions,
            cos_sin,
            nope_dim=128 - ROPE_DIM,
        )
        rotated = _hadamard_rotate(rotated)
        expected_packed = torch.empty_like(q_packed)
        expected_scales = torch.empty(2, 3, 4, device=device, dtype=torch.uint8)
        for token_idx in range(rotated.shape[0]):
            for head_idx in range(rotated.shape[1]):
                packed, scales, _ = _mxfp4_bytes_and_scales(
                    rotated[token_idx, head_idx]
                )
                expected_packed[token_idx, head_idx].copy_(packed)
                expected_scales[token_idx, head_idx].copy_(scales)

        self.assertTrue(torch.equal(q_packed.cpu(), expected_packed.cpu()))
        self.assertTrue(
            torch.equal(
                q_scales.cpu(),
                expected_scales.contiguous().view(torch.int32).squeeze(-1).cpu(),
            )
        )

    def test_sparse_prefill_combine_topk_swa_indices_matches_reference(self):
        device = torch.device("cuda")
        topk_indices = torch.tensor(
            [
                [0, 1, 2],
                [1, 0, 2],
                [0, 1, 2],
                [1, 0, 2],
                [1, 0, 2],
            ],
            device=device,
            dtype=torch.int32,
        )
        query_start_loc = torch.tensor([0, 2, 5], device=device, dtype=torch.int32)
        seq_lens = torch.tensor([6, 8], device=device, dtype=torch.int32)
        gather_lens = torch.tensor([5, 6], device=device, dtype=torch.int32)
        window_size = 4
        compress_ratio = 4
        compressed_base = 3
        workspace_width = 9

        actual, actual_lens = deepseek_v4_combine_topk_swa_indices(
            topk_indices=topk_indices,
            query_start_loc=query_start_loc,
            seq_lens=seq_lens,
            gather_lens=gather_lens,
            window_size=window_size,
            compress_ratio=compress_ratio,
            topk=topk_indices.shape[-1],
            workspace_width=workspace_width,
            compressed_base=compressed_base,
        )
        torch.cuda.synchronize()

        expected = torch.full_like(actual, -1)
        expected_lens = torch.empty_like(actual_lens)
        for req_idx in range(seq_lens.numel()):
            query_start = int(query_start_loc[req_idx].item())
            query_end = int(query_start_loc[req_idx + 1].item())
            query_len = query_end - query_start
            seq_len = int(seq_lens[req_idx].item())
            gather_len = int(gather_lens[req_idx].item())
            start_pos = seq_len - query_len
            gather_start = seq_len - gather_len
            request_base = req_idx * workspace_width
            for token_idx in range(query_start, query_end):
                pos = start_pos + token_idx - query_start
                topk_len = min((pos + 1) // compress_ratio, topk_indices.shape[-1])
                swa_len = min(pos + 1, window_size)
                cursor = 0
                if topk_len:
                    expected[token_idx, :topk_len] = (
                        request_base + topk_indices[token_idx, :topk_len]
                    )
                    cursor = topk_len
                for offset in range(swa_len):
                    expected[token_idx, cursor + offset] = (
                        request_base
                        + compressed_base
                        + offset
                        + pos
                        - swa_len
                        + 1
                        - gather_start
                    )
                expected_lens[token_idx] = topk_len + swa_len

        torch.testing.assert_close(actual.cpu(), expected.cpu(), atol=0, rtol=0)
        torch.testing.assert_close(
            actual_lens.cpu(), expected_lens.cpu(), atol=0, rtol=0
        )

    def test_sparse_prefill_combine_dense_swa_indices_matches_reference(self):
        device = torch.device("cuda")
        positions = torch.tensor([4, 5, 5, 6, 7], device=device, dtype=torch.int64)
        token_to_req_indices = torch.tensor(
            [0, 0, 1, 1, 1],
            device=device,
            dtype=torch.int32,
        )
        seq_lens = torch.tensor([6, 8], device=device, dtype=torch.int32)
        compressed_lens = torch.tensor([2, 2], device=device, dtype=torch.int32)
        gather_lens = torch.tensor([5, 6], device=device, dtype=torch.int32)
        window_size = 4
        compress_ratio = 3
        compressed_base = 2
        workspace_width = 8

        actual, actual_lens = deepseek_v4_combine_dense_swa_indices(
            positions=positions,
            token_to_req_indices=token_to_req_indices,
            seq_lens=seq_lens,
            compressed_lens=compressed_lens,
            gather_lens=gather_lens,
            window_size=window_size,
            compress_ratio=compress_ratio,
            workspace_width=workspace_width,
            compressed_base=compressed_base,
        )
        torch.cuda.synchronize()

        expected = torch.full_like(actual, -1)
        expected_lens = torch.empty_like(actual_lens)
        for token_idx, position in enumerate(positions.cpu().tolist()):
            req_idx = int(token_to_req_indices[token_idx].item())
            seq_len = int(seq_lens[req_idx].item())
            gather_len = int(gather_lens[req_idx].item())
            gather_start = seq_len - gather_len
            request_base = req_idx * workspace_width
            compressed_len = min(
                (position + 1) // compress_ratio,
                int(compressed_lens[req_idx].item()),
            )
            cursor = 0
            for offset in range(compressed_len):
                expected[token_idx, cursor] = request_base + offset
                cursor += 1
            swa_len = min(position + 1, window_size)
            for offset in range(swa_len):
                expected[token_idx, cursor + offset] = (
                    request_base
                    + compressed_base
                    + offset
                    + position
                    - swa_len
                    + 1
                    - gather_start
                )
            expected_lens[token_idx] = compressed_len + swa_len

        torch.testing.assert_close(actual.cpu(), expected.cpu(), atol=0, rtol=0)
        torch.testing.assert_close(
            actual_lens.cpu(), expected_lens.cpu(), atol=0, rtol=0
        )

    def test_decode_swa_indices_and_lens_matches_reference(self):
        device = torch.device("cuda")
        query_start_loc = torch.tensor([0, 1, 2], device=device, dtype=torch.int32)
        seq_lens = torch.tensor([70, 3], device=device, dtype=torch.int32)
        token_to_req_indices = torch.tensor([0, 1], device=device, dtype=torch.int32)
        block_table = torch.tensor(
            [[10, 11], [20, 21]],
            device=device,
            dtype=torch.int32,
        )
        out_indices = torch.empty((2, 4), device=device, dtype=torch.int32)
        out_lens = torch.empty((2,), device=device, dtype=torch.int32)

        actual, actual_lens = deepseek_v4_decode_swa_indices_and_lens(
            query_start_loc=query_start_loc,
            seq_lens=seq_lens,
            token_to_req_indices=token_to_req_indices,
            block_table=block_table,
            window_size=4,
            block_size=64,
            out_indices=out_indices,
            out_lens=out_lens,
        )
        torch.cuda.synchronize()

        self.assertEqual(actual.data_ptr(), out_indices.data_ptr())
        self.assertEqual(actual_lens.data_ptr(), out_lens.data_ptr())
        self.assertTrue(
            torch.equal(
                actual.cpu(),
                torch.tensor(
                    [
                        [706, 707, 708, 709],
                        [1280, 1281, 1282, -1],
                    ],
                    dtype=torch.int32,
                ),
            )
        )
        self.assertTrue(
            torch.equal(actual_lens.cpu(), torch.tensor([4, 3], dtype=torch.int32))
        )
        compact_actual, compact_lens = deepseek_v4_decode_swa_indices_and_lens(
            query_start_loc=query_start_loc,
            seq_lens=seq_lens,
            token_to_req_indices=token_to_req_indices,
            block_table=torch.tensor([[11], [20]], device=device, dtype=torch.int32),
            block_table_base_offsets=torch.tensor(
                [1, 0], device=device, dtype=torch.int32
            ),
            window_size=4,
            block_size=64,
        )
        torch.cuda.synchronize()

        torch.testing.assert_close(compact_actual.cpu(), actual.cpu(), atol=0, rtol=0)
        torch.testing.assert_close(
            compact_lens.cpu(), actual_lens.cpu(), atol=0, rtol=0
        )

    def test_compute_global_topk_indices_and_lens_matches_reference(self):
        device = torch.device("cuda")
        topk_indices = torch.tensor(
            [
                [0, 1, -1, 5],
                [3, -1, -1, -1],
                [4, 2, 1, -1],
            ],
            device=device,
            dtype=torch.int32,
        )
        token_to_req_indices = torch.tensor([0, 1, 0], device=device, dtype=torch.int32)
        block_table = torch.tensor(
            [
                [10, 11],
                [20, 21],
            ],
            device=device,
            dtype=torch.int32,
        )

        actual, actual_lens = deepseek_v4_compute_global_topk_indices_and_lens(
            topk_indices=topk_indices,
            token_to_req_indices=token_to_req_indices,
            block_table=block_table,
            block_size=4,
        )
        torch.cuda.synchronize()

        expected = torch.empty_like(topk_indices)
        expected_lens = torch.empty_like(actual_lens)
        for token_idx in range(topk_indices.shape[0]):
            req_idx = int(token_to_req_indices[token_idx].item())
            count = 0
            for topk_idx in range(topk_indices.shape[1]):
                local_idx = int(topk_indices[token_idx, topk_idx].item())
                if local_idx < 0:
                    expected[token_idx, topk_idx] = -1
                    continue
                block_idx = local_idx // 4
                offset = local_idx % 4
                expected[token_idx, topk_idx] = (
                    int(block_table[req_idx, block_idx].item()) * 4 + offset
                )
                count += 1
            expected_lens[token_idx] = count

        torch.testing.assert_close(actual.cpu(), expected.cpu(), atol=0, rtol=0)
        torch.testing.assert_close(
            actual_lens.cpu(), expected_lens.cpu(), atol=0, rtol=0
        )

    def test_compressed_slot_mapping_matches_page_reference(self):
        device = torch.device("cuda")
        query_start_loc = torch.tensor([0, 3, 5], device=device, dtype=torch.int32)
        seq_lens = torch.tensor([8, 6], device=device, dtype=torch.int32)
        block_table = torch.tensor(
            [
                [10, 11, 12],
                [20, 21, 22],
            ],
            device=device,
            dtype=torch.int32,
        )
        out = torch.empty(8, device=device, dtype=torch.int64)

        actual = deepseek_v4_compressed_slot_mapping(
            num_tokens=5,
            query_start_loc=query_start_loc,
            seq_lens=seq_lens,
            block_table=block_table,
            block_size=4,
            compress_ratio=4,
            out=out,
        )
        torch.cuda.synchronize()

        self.assertEqual(actual.data_ptr(), out.data_ptr())
        self.assertTrue(
            torch.equal(
                actual.cpu(),
                torch.tensor([-1, -1, 10 * 4 + 1, -1, -1], dtype=torch.int64),
            )
        )
        self.assertTrue(torch.equal(out[5:].cpu(), torch.full((3,), -1)))

    def test_sparse_prefill_dequantize_and_gather_matches_reference(self):
        torch.manual_seed(9234)
        device = torch.device("cuda")
        block_size = 4
        num_blocks = 3
        cache = torch.zeros(
            num_blocks,
            block_size * (DEEPSEEK_V4_SWA_TOKEN_STRIDE + DEEPSEEK_V4_SWA_SCALE_DIM),
            device=device,
            dtype=torch.uint8,
        )
        rows = torch.randn(num_blocks * block_size, HEAD_DIM, device=device)
        flat_cache = cache.view(-1)
        for slot in range(rows.shape[0]):
            page = slot // block_size
            pos = slot % block_size
            page_base = page * cache.stride(0)
            token_base = page_base + pos * DEEPSEEK_V4_SWA_TOKEN_STRIDE
            scale_base = (
                page_base
                + block_size * DEEPSEEK_V4_SWA_TOKEN_STRIDE
                + pos * DEEPSEEK_V4_SWA_SCALE_DIM
            )
            for qblock in range(7):
                start = qblock * 64
                fp8_bytes, scale = _fp8_bytes_and_scale(rows[slot, start : start + 64])
                flat_cache[token_base + start : token_base + start + 64].copy_(
                    fp8_bytes
                )
                flat_cache[scale_base + qblock] = scale
            flat_cache[scale_base + 7] = 0
            flat_cache[
                token_base + NOPE_DIM : token_base + DEEPSEEK_V4_SWA_TOKEN_STRIDE
            ].copy_(rows[slot, NOPE_DIM:].to(torch.bfloat16).view(torch.uint8))

        seq_lens = torch.tensor([9, 8], device=device, dtype=torch.int32)
        gather_lens = torch.tensor([3, 2], device=device, dtype=torch.int32)
        block_table = torch.tensor([[0, 1], [2, 0]], device=device, dtype=torch.int32)
        block_table_base_offsets = torch.tensor(
            [1, 1], device=device, dtype=torch.int32
        )
        out = torch.zeros(2, 5, HEAD_DIM, device=device, dtype=torch.bfloat16)
        deepseek_v4_dequantize_and_gather_k_cache(
            out=out,
            cache_2d=cache,
            seq_lens=seq_lens,
            gather_lens=gather_lens,
            block_table=block_table,
            block_table_base_offsets=block_table_base_offsets,
            block_size=block_size,
            offset=1,
        )
        torch.cuda.synchronize()

        expected_slots = []
        for req_idx in range(seq_lens.numel()):
            request_slots = []
            start = int(seq_lens[req_idx].item() - gather_lens[req_idx].item())
            end = int(seq_lens[req_idx].item())
            base = int(block_table_base_offsets[req_idx].item())
            for pos in range(start, end):
                page = pos // block_size
                offset = pos % block_size
                request_slots.append(
                    int(block_table[req_idx, page - base].item()) * block_size + offset
                )
            expected_slots.append(
                torch.tensor(request_slots, device=device, dtype=torch.int64)
            )

        for req_idx, slots in enumerate(expected_slots):
            expected_rows = dequantize_deepseek_v4_fp8_ds_mla_cache(
                cache, slots, block_size
            )
            gathered = out[req_idx, 1 : 1 + slots.numel()]
            torch.testing.assert_close(
                gathered.float().cpu(),
                expected_rows.float().cpu(),
                atol=0,
                rtol=0,
            )


if __name__ == "__main__":
    unittest.main()
