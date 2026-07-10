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
from types import SimpleNamespace
from unittest import mock

import torch
from tokenspeed_kernel.ops.attention.cuda.deepseek_v4 import (
    fused_qnorm_rope_kv_insert as cuda_fused_qnorm_rope_kv_insert,
)
from tokenspeed_kernel.ops.attention.cuda.deepseek_v4 import (
    has_indexer_mxfp4_paged_gather,
    has_persistent_topk,
    indexer_mxfp4_paged_gather,
    persistent_topk,
)
from tokenspeed_kernel.ops.attention.triton.deepseek_v4 import (
    _deepseek_v4_decode_swa_indices_and_lens_kernel,
    _deepseek_v4_dequantize_and_gather_k_kernel,
    _deepseek_v4_fused_csa_indexer_mxfp4_cache_kernel,
    _deepseek_v4_fused_sparse_compress_cache_kernel,
    _deepseek_v4_gather_launch_config,
)
from tokenspeed_kernel.ops.attention.trtllm.deepseek_v4 import (
    has_trtllm_deepseek_v4_c128_prefill_compress_cache,
    supports_trtllm_deepseek_v4_c128_prefill_compress_cache,
)
from tokenspeed_kernel.ops.transform import hadamard_transform

from tokenspeed.runtime.configs.deepseek_v4_cache_spec import (
    deepseek_v4_swa_scale_dim,
    deepseek_v4_swa_token_stride,
)
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.attention.deepseek_v4_ops import (
    deepseek_v4_combine_dense_swa_indices,
    deepseek_v4_combine_topk_swa_indices,
    deepseek_v4_compressed_slot_mapping,
    deepseek_v4_compute_global_topk_indices_and_lens,
    deepseek_v4_csa_compress_kv_cache_insert,
    deepseek_v4_csa_indexer_cache_insert,
    deepseek_v4_decode_swa_indices_and_lens,
    deepseek_v4_dequantize_and_gather_k_cache,
    deepseek_v4_hca_compress_kv_cache_insert,
    deepseek_v4_prepare_indexer_q_mxfp4,
    dequantize_deepseek_v4_fp8_ds_mla_cache,
    fused_qnorm_rope_kv_insert,
    has_fused_qnorm_rope_kv_insert_padded,
    read_deepseek_v4_indexer_fp8_cache,
    read_deepseek_v4_indexer_mxfp4_cache,
    save_deepseek_v4_compressor_state,
    write_deepseek_v4_indexer_fp8_cache,
    write_deepseek_v4_indexer_mxfp4_cache,
)
from tokenspeed.runtime.layers.attention.kv_cache.deepseek_v4 import (
    _mask_invalid_graph_tokens,
)
from tokenspeed.runtime.models.deepseek_v4 import (
    DeepseekV4Attention,
    _deepseek_v4_sanitize_swa_slot_mapping,
)
from tokenspeed.runtime.utils.cuda_stream import StreamFork

HEAD_DIM = 512
NOPE_DIM = 448
ROPE_DIM = 64
FP8_MAX = 448.0
SWA_TOKEN_STRIDE = deepseek_v4_swa_token_stride(HEAD_DIM, ROPE_DIM)
SWA_SCALE_DIM = deepseek_v4_swa_scale_dim(HEAD_DIM, ROPE_DIM)


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
    shape = row.shape
    return hadamard_transform(
        row.to(torch.bfloat16).reshape(-1, shape[-1]).contiguous(),
        scale=shape[-1] ** -0.5,
    ).reshape(shape)


def _prepare_indexer_q_reference(
    index_q: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    weights: torch.Tensor,
    softmax_scale: float,
    head_scale: float,
    use_fp4: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    rope_dim = int(cos_sin_cache.shape[-1])
    rotated = _apply_gptj_rope_with_nope(
        index_q,
        positions,
        cos_sin_cache,
        nope_dim=index_q.shape[-1] - rope_dim,
    )
    rotated = _hadamard_rotate(rotated).float()
    weights_out = weights.float().clone()
    if weights_out.dim() == 3:
        weights_out = weights_out.squeeze(-1)

    if use_fp4:
        q_out = torch.empty_like(rotated, dtype=torch.float32)
        for token_idx in range(rotated.shape[0]):
            for head_idx in range(rotated.shape[1]):
                _, _, dequant = _mxfp4_bytes_and_scales(rotated[token_idx, head_idx])
                q_out[token_idx, head_idx].copy_(dequant)
        weights_out *= softmax_scale * head_scale
    else:
        q_out = torch.empty_like(rotated, dtype=torch.float32)
        q_scales = torch.empty_like(weights_out, dtype=torch.float32)
        for token_idx in range(rotated.shape[0]):
            for head_idx in range(rotated.shape[1]):
                q_bytes, scale = _fp8_pow2_bytes_and_scale(rotated[token_idx, head_idx])
                q_out[token_idx, head_idx].copy_(
                    q_bytes.view(torch.float8_e4m3fn).float() * scale
                )
                q_scales[token_idx, head_idx] = scale
        weights_out *= q_scales * softmax_scale * head_scale
    return q_out.to(index_q.dtype), weights_out


def _indexer_topk_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    top_k: int,
    lengths: torch.Tensor | None = None,
    row_starts: torch.Tensor | None = None,
) -> torch.Tensor:
    if weights.dim() == 3:
        weights = weights.squeeze(-1)
    logits = torch.einsum("thd,kd->thk", q.float(), k.float()).relu()
    logits = (logits * weights.float().unsqueeze(-1)).sum(dim=1)
    if lengths is not None:
        if row_starts is None:
            row_starts = torch.zeros_like(lengths)
        cols = torch.arange(k.shape[0], device=k.device)
        valid = (cols.unsqueeze(0) >= row_starts.unsqueeze(1)) & (
            cols.unsqueeze(0) < (row_starts + lengths).unsqueeze(1)
        )
        logits = logits.masked_fill(~valid, -float("inf"))
    return torch.topk(logits, k=top_k, dim=-1, sorted=False).indices.to(torch.int32)


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


class DeepseekV4AttentionOpsCpuValidationTest(unittest.TestCase):
    def test_dynamic_block_table_widths_are_not_specialized(self):
        cases = (
            (
                _deepseek_v4_fused_sparse_compress_cache_kernel,
                ("block_table_stride", "block_table_width"),
            ),
            (
                _deepseek_v4_fused_csa_indexer_mxfp4_cache_kernel,
                ("block_table_stride", "block_table_width"),
            ),
            (
                _deepseek_v4_dequantize_and_gather_k_kernel,
                ("block_table_stride", "max_blocks_per_seq"),
            ),
            (
                _deepseek_v4_decode_swa_indices_and_lens_kernel,
                ("block_table_stride", "max_blocks_per_seq"),
            ),
        )
        for kernel, parameter_names in cases:
            with self.subTest(kernel=kernel.__name__):
                parameters = {parameter.name: parameter for parameter in kernel.params}
                for parameter_name in parameter_names:
                    parameter = parameters[parameter_name]
                    self.assertFalse(parameter.is_constexpr)
                    self.assertTrue(parameter.do_not_specialize)

        hca_capacity = {
            parameter.name: parameter
            for parameter in _deepseek_v4_fused_sparse_compress_cache_kernel.params
        }["HCA_BLOCK_TABLE_CAPACITY"]
        self.assertTrue(hca_capacity.is_constexpr)

    def test_sparse_prefill_gather_launch_config(self):
        cases = (
            (1, 0, (128, 4)),
            (1, 512, (128, 4)),
            (1, 513, (512, 1)),
            (1, 3072, (512, 1)),
            (1, 3073, (1024, 1)),
            (1, 6144, (1024, 1)),
            (1, 6145, (2048, 1)),
            (2, 6145, (2048, 1)),
            (3, 6145, (1024, 1)),
            (4, 8192, (1024, 1)),
            (4, 12288, (1024, 1)),
            (4, 12289, (2048, 1)),
            (8, 2048, (512, 1)),
            (8, 8192, (2048, 1)),
        )
        for num_reqs, max_rows, expected in cases:
            with self.subTest(num_reqs=num_reqs, max_rows=max_rows):
                self.assertEqual(
                    _deepseek_v4_gather_launch_config(num_reqs, max_rows),
                    expected,
                )

    def test_padded_q_runtime_route_is_pure_prefill_only(self):
        def run_case(
            mode: ForwardMode,
            *,
            capability: bool,
            padded_heads: int = 64,
        ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
            attention = object.__new__(DeepseekV4Attention)
            torch.nn.Module.__init__(attention)
            attention.attention_kind = "csa"
            attention.rotary_emb = SimpleNamespace(
                cos_sin_cache=torch.zeros(1, ROPE_DIM, dtype=torch.float32)
            )
            attention.stream_fork = StreamFork(None)
            attention.compressor = None
            attention.indexer = None
            attention.padded_heads = padded_heads
            attention.cache_layer_index = 0
            attention.compress_ratio = 4
            attention.num_local_heads = 16
            attention.head_dim = HEAD_DIM
            attention.swa_window = 64
            attention.scale = 1.0
            attention.attn_sink = torch.zeros(1)

            q = torch.randn(2, 16, HEAD_DIM, dtype=torch.bfloat16)
            kv = torch.randn(2, HEAD_DIM, dtype=torch.bfloat16)
            qr = torch.empty(2, 16, ROPE_DIM, dtype=torch.bfloat16)
            attention._project_q_kv = mock.Mock(return_value=(q, kv, qr))
            attention._insert_swa_cache = mock.Mock()
            attention._project_attention_output = mock.Mock(
                side_effect=lambda output, *_: output
            )

            backend_method = {
                ForwardMode.EXTEND: "forward_deepseek_v4_prefill",
                ForwardMode.MIXED: "forward_deepseek_v4_mixed",
                ForwardMode.DECODE: "forward_deepseek_v4_decode",
            }[mode]
            backend = SimpleNamespace(
                forward_metadata=SimpleNamespace(
                    is_valid_token=None,
                    token_to_req_indices=None,
                )
            )
            setattr(
                backend, backend_method, mock.Mock(side_effect=lambda **kw: kw["q"])
            )
            pool = SimpleNamespace(
                swa_block_size=4,
                get_swa_kv_buffer=mock.Mock(
                    return_value=torch.empty(1, 4 * 584, dtype=torch.uint8)
                ),
            )
            slots = torch.arange(2, dtype=torch.int64)
            ctx = SimpleNamespace(
                dsa_compressor_slot_cache={},
                dsa_swa_slot_mapping=slots,
                token_to_kv_pool=pool,
                attn_backend=backend,
                forward_mode=mode,
            )
            with mock.patch(
                "tokenspeed.runtime.models.deepseek_v4."
                "has_fused_qnorm_rope_kv_insert_padded",
                return_value=capability,
            ):
                output = attention(
                    torch.arange(2, dtype=torch.int64),
                    torch.empty(2, 1),
                    ctx,
                    torch.arange(2, dtype=torch.int64),
                    swa_slot_mapping=slots,
                )
            q_out = attention._insert_swa_cache.call_args.kwargs["q_out"]
            backend_q = getattr(backend, backend_method).call_args.kwargs["q"]
            return q, q_out, backend_q

        cases = (
            (ForwardMode.EXTEND, True, 64, True),
            (ForwardMode.EXTEND, False, 64, False),
            (ForwardMode.MIXED, True, 64, False),
            (ForwardMode.DECODE, True, 64, False),
            (ForwardMode.EXTEND, True, 16, False),
        )
        for mode, capability, padded_heads, expect_padded in cases:
            with self.subTest(
                mode=mode,
                capability=capability,
                padded_heads=padded_heads,
            ):
                q, q_out, backend_q = run_case(
                    mode,
                    capability=capability,
                    padded_heads=padded_heads,
                )
                if expect_padded:
                    self.assertIsNotNone(q_out)
                    self.assertEqual(q_out.shape, (2, padded_heads, HEAD_DIM))
                    self.assertIs(backend_q, q_out)
                else:
                    self.assertIsNone(q_out)
                    self.assertIs(backend_q, q)

    def test_swa_slot_mapping_guard_masks_out_of_range_slots(self):
        cache = torch.empty((2, 4 * SWA_TOKEN_STRIDE), dtype=torch.uint8)
        slots = torch.tensor([-3, -1, 0, 7, 8, 99], dtype=torch.int64)

        sanitized = _deepseek_v4_sanitize_swa_slot_mapping(slots, cache, 4)

        torch.testing.assert_close(
            sanitized,
            torch.tensor([-1, -1, 0, 7, -1, -1], dtype=torch.int64),
        )

    def test_swa_slot_mapping_guard_masks_invalid_graph_tokens(self):
        cache = torch.empty((2, 4 * SWA_TOKEN_STRIDE), dtype=torch.uint8)
        slots = torch.tensor([0, 1, 2, 9, -1, 3], dtype=torch.int64)
        is_valid_token = torch.tensor([True, False, True, True, True, False])

        sanitized = _deepseek_v4_sanitize_swa_slot_mapping(
            slots,
            cache,
            4,
            is_valid_token=is_valid_token,
        )

        torch.testing.assert_close(
            sanitized,
            torch.tensor([0, -1, 2, -1, -1, -1], dtype=torch.int64),
        )

    def test_slot_mapping_guard_masks_invalid_graph_tokens(self):
        slots = torch.tensor([0, 1, -1, 3], dtype=torch.int64)
        is_valid_token = torch.tensor([True, False, True, False])

        sanitized = _mask_invalid_graph_tokens(
            slots,
            is_valid_token,
        )

        torch.testing.assert_close(
            sanitized,
            torch.tensor([0, -1, -1, -1], dtype=torch.int64),
        )

    def test_slot_mapping_guard_expands_per_request_validity(self):
        slots = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.int64)
        is_valid_token = torch.tensor([True, False])

        sanitized = _mask_invalid_graph_tokens(
            slots,
            is_valid_token,
        )

        torch.testing.assert_close(
            sanitized,
            torch.tensor([0, 1, 2, -1, -1, -1], dtype=torch.int64),
        )

    def test_indexer_q_mxfp4_requires_cuda(self):
        q = torch.zeros(1, 1, 128, dtype=torch.bfloat16)
        positions = torch.zeros(1, dtype=torch.int64)
        cos_sin = torch.zeros(1, ROPE_DIM, dtype=torch.float32)
        weights = torch.ones(1, 1, dtype=torch.float32)

        with self.assertRaisesRegex(ValueError, "only supports CUDA"):
            deepseek_v4_prepare_indexer_q_mxfp4(
                q,
                positions,
                cos_sin,
                weights,
                1.0,
                1.0,
            )

    def test_indexer_mxfp4_cache_writer_requires_cuda(self):
        index_k = torch.zeros(1, 128, dtype=torch.bfloat16)
        cache = torch.zeros(1, 64 * 68, dtype=torch.uint8)
        slots = torch.zeros(1, dtype=torch.int64)

        with self.assertRaisesRegex(ValueError, "only supports CUDA"):
            write_deepseek_v4_indexer_mxfp4_cache(index_k, cache, slots, 64)

    def test_compressor_state_writer_requires_cuda(self):
        kv = torch.zeros(1, HEAD_DIM, dtype=torch.bfloat16)
        score = torch.zeros_like(kv)
        ape = torch.zeros(128, HEAD_DIM, dtype=torch.float32)
        state_cache = torch.zeros(1, 128, HEAD_DIM * 2, dtype=torch.float32)
        slots = torch.zeros(1, dtype=torch.int64)
        positions = torch.zeros(1, dtype=torch.int64)

        with self.assertRaisesRegex(ValueError, "only supports CUDA"):
            save_deepseek_v4_compressor_state(
                kv=kv,
                score=score,
                ape=ape,
                state_cache=state_cache,
                slot_mapping=slots,
                positions=positions,
                block_size=128,
                compress_ratio=128,
            )

    def test_compress_cache_insert_requires_cuda(self):
        state_cache = torch.zeros(1, 128, HEAD_DIM * 2, dtype=torch.float32)
        token_to_req_indices = torch.zeros(1, dtype=torch.int32)
        positions = torch.zeros(1, dtype=torch.int64)
        compressor_slots = torch.zeros(1, dtype=torch.int64)
        block_table = torch.zeros(1, 1, dtype=torch.int32)
        rms_weight = torch.ones(HEAD_DIM, dtype=torch.float32)
        cos_sin = torch.zeros(1, ROPE_DIM, dtype=torch.float32)
        kv_cache = torch.zeros(
            1,
            64 * (SWA_TOKEN_STRIDE + SWA_SCALE_DIM),
            dtype=torch.uint8,
        )
        kv_slots = torch.zeros(1, dtype=torch.int64)

        with self.assertRaisesRegex(ValueError, "only supports CUDA"):
            deepseek_v4_hca_compress_kv_cache_insert(
                state_cache=state_cache,
                token_to_req_indices=token_to_req_indices,
                positions=positions,
                compressor_slot_mapping=compressor_slots,
                block_table=block_table,
                compressor_block_size=128,
                rms_norm_weight=rms_weight,
                rms_norm_eps=1.0e-6,
                cos_sin_cache=cos_sin,
                kv_cache_2d=kv_cache,
                kv_slot_mapping=kv_slots,
                kv_cache_block_size=64,
                compress_ratio=128,
            )

    def test_csa_indexer_cache_insert_requires_cuda(self):
        state_cache = torch.zeros(1, 4, 128 * 4, dtype=torch.float32)
        token_to_req_indices = torch.zeros(1, dtype=torch.int32)
        positions = torch.zeros(1, dtype=torch.int64)
        compressor_slots = torch.zeros(1, dtype=torch.int64)
        block_table = torch.zeros(1, 1, dtype=torch.int32)
        rms_weight = torch.ones(128, dtype=torch.float32)
        cos_sin = torch.zeros(1, ROPE_DIM, dtype=torch.float32)
        kv_cache = torch.zeros(1, 64 * 68, dtype=torch.uint8)
        kv_slots = torch.zeros(1, dtype=torch.int64)

        with self.assertRaisesRegex(ValueError, "only supports CUDA"):
            deepseek_v4_csa_indexer_cache_insert(
                state_cache=state_cache,
                token_to_req_indices=token_to_req_indices,
                positions=positions,
                compressor_slot_mapping=compressor_slots,
                block_table=block_table,
                compressor_block_size=4,
                rms_norm_weight=rms_weight,
                rms_norm_eps=1.0e-6,
                cos_sin_cache=cos_sin,
                kv_cache_2d=kv_cache,
                kv_slot_mapping=kv_slots,
                kv_cache_block_size=64,
                use_fp4_cache=True,
                compress_ratio=4,
            )


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
        except RuntimeError as exc:
            if "fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert" in str(exc):
                self.skipTest(str(exc))
            raise
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

    def test_fused_qnorm_rope_kv_insert_padded_matches_inplace_bitwise(self):
        self.assertTrue(
            has_fused_qnorm_rope_kv_insert_padded(),
            "current tokenspeed-kernel build is missing the padded-Q symbol",
        )

        device = torch.device("cuda")
        num_tokens = 9
        block_size = 4
        eps = 1.0e-6
        positions = torch.arange(num_tokens, dtype=torch.int64, device=device)
        slot_mapping = torch.tensor(
            [0, 2, -1, 7, 8, 15, 16, 100], dtype=torch.int64, device=device
        )
        cos_sin = torch.randn(32, ROPE_DIM, device=device, dtype=torch.float32) * 0.1

        for num_heads, padded_heads in (
            (16, 64),
            (24, 64),
            (32, 64),
            (64, 64),
            (65, 128),
            (96, 128),
            (128, 128),
        ):
            for dtype in (torch.bfloat16, torch.float16):
                with self.subTest(
                    dtype=dtype,
                    num_heads=num_heads,
                    padded_heads=padded_heads,
                ):
                    torch.manual_seed(5678)
                    q_input = torch.randn(
                        num_tokens,
                        num_heads,
                        HEAD_DIM,
                        device=device,
                        dtype=dtype,
                    )
                    kv = torch.randn(num_tokens, HEAD_DIM, device=device, dtype=dtype)
                    q_inplace = q_input.clone()
                    q_source = q_input.clone()
                    q_out = torch.full(
                        (num_tokens, padded_heads, HEAD_DIM),
                        float("nan"),
                        device=device,
                        dtype=dtype,
                    )
                    cache_inplace = torch.zeros(
                        4, block_size * 584, device=device, dtype=torch.uint8
                    )
                    cache_padded = torch.zeros_like(cache_inplace)

                    fused_qnorm_rope_kv_insert(
                        q=q_inplace,
                        kv=kv,
                        swa_kv_cache_2d=cache_inplace,
                        slot_mapping=slot_mapping,
                        positions=positions,
                        cos_sin_cache=cos_sin,
                        rms_norm_eps=eps,
                        block_size=block_size,
                    )
                    fused_qnorm_rope_kv_insert(
                        q=q_source,
                        q_out=q_out,
                        kv=kv,
                        swa_kv_cache_2d=cache_padded,
                        slot_mapping=slot_mapping,
                        positions=positions,
                        cos_sin_cache=cos_sin,
                        rms_norm_eps=eps,
                        block_size=block_size,
                    )
                    torch.cuda.synchronize()

                    self.assertTrue(torch.equal(q_source, q_input))
                    self.assertTrue(torch.equal(q_out[:, :num_heads], q_inplace))
                    self.assertEqual(
                        int(
                            torch.count_nonzero(q_out[:, num_heads:].view(torch.int16))
                        ),
                        0,
                    )
                    self.assertTrue(torch.equal(cache_padded, cache_inplace))

                    if (
                        num_heads == 16
                        and dtype == torch.bfloat16
                        and torch.cuda.get_device_capability()[0] >= 9
                    ):
                        q_pdl_source = q_input.clone()
                        q_pdl_out = torch.full_like(q_out, float("nan"))
                        cache_pdl = torch.zeros_like(cache_inplace)
                        cuda_fused_qnorm_rope_kv_insert(
                            q=q_pdl_source,
                            q_out=q_pdl_out,
                            kv=kv,
                            k_cache=cache_pdl,
                            slot_mapping=slot_mapping,
                            positions=positions,
                            cos_sin_cache=cos_sin,
                            rms_norm_eps=eps,
                            block_size=block_size,
                            enable_pdl=True,
                        )
                        torch.cuda.synchronize()
                        self.assertTrue(torch.equal(q_pdl_source, q_input))
                        self.assertTrue(torch.equal(q_pdl_out, q_out))
                        self.assertTrue(torch.equal(cache_pdl, cache_inplace))

    def test_fused_qnorm_rope_kv_insert_padded_rejects_q_out_overlap(self):
        self.assertTrue(
            has_fused_qnorm_rope_kv_insert_padded(),
            "current tokenspeed-kernel build is missing the padded-Q symbol",
        )

        device = torch.device("cuda")
        num_tokens = 1
        num_heads = 16
        padded_heads = 64
        block_size = 4
        cache_row_bytes = block_size * 584
        for input_name in (
            "q",
            "kv",
            "k_cache",
            "slot_mapping",
            "positions",
            "cos_sin_cache",
        ):
            with self.subTest(input_name=input_name):
                if input_name == "k_cache":
                    cache_stride = cache_row_bytes + 128
                    # Start q_out beyond a naive k_cache.numel() range but
                    # inside the second strided row's real byte envelope.
                    q_out_offset = 2 * cache_row_bytes + 32
                    q_out_num_bytes = (
                        num_tokens * padded_heads * HEAD_DIM * torch.bfloat16.itemsize
                    )
                    shared_storage = torch.empty(
                        q_out_offset + q_out_num_bytes,
                        device=device,
                        dtype=torch.uint8,
                    )
                    q_out = (
                        shared_storage[q_out_offset : q_out_offset + q_out_num_bytes]
                        .view(torch.bfloat16)
                        .view(num_tokens, padded_heads, HEAD_DIM)
                    )
                    cache_alias = torch.as_strided(
                        shared_storage,
                        size=(2, cache_row_bytes),
                        stride=(cache_stride, 1),
                    )
                else:
                    q_out = torch.empty(
                        num_tokens,
                        padded_heads,
                        HEAD_DIM,
                        device=device,
                        dtype=torch.bfloat16,
                    )
                    cache_alias = torch.as_strided(
                        q_out.view(torch.uint8).view(-1),
                        size=(2, cache_row_bytes),
                        stride=(cache_row_bytes + 128, 1),
                    )
                q_out_flat = q_out.view(-1)
                q_out_int64 = q_out.view(torch.int64).view(-1)
                q_out_float32 = q_out.view(torch.float32).view(-1)
                inputs = {
                    "q": torch.randn(
                        num_tokens,
                        num_heads,
                        HEAD_DIM,
                        device=device,
                        dtype=q_out.dtype,
                    ),
                    "kv": torch.randn(
                        num_tokens,
                        HEAD_DIM,
                        device=device,
                        dtype=q_out.dtype,
                    ),
                    "k_cache": torch.zeros(
                        1,
                        cache_row_bytes,
                        device=device,
                        dtype=torch.uint8,
                    ),
                    "slot_mapping": torch.zeros(
                        num_tokens, device=device, dtype=torch.int64
                    ),
                    "positions": torch.zeros(
                        num_tokens, device=device, dtype=torch.int64
                    ),
                    "cos_sin_cache": torch.zeros(
                        1, ROPE_DIM, device=device, dtype=torch.float32
                    ),
                }
                aliases = {
                    "q": q_out_flat[: num_tokens * num_heads * HEAD_DIM].view(
                        num_tokens, num_heads, HEAD_DIM
                    ),
                    "kv": q_out_flat[: num_tokens * HEAD_DIM].view(
                        num_tokens, HEAD_DIM
                    ),
                    "k_cache": cache_alias,
                    "slot_mapping": q_out_int64[:num_tokens],
                    "positions": q_out_int64[:num_tokens],
                    "cos_sin_cache": q_out_float32[:ROPE_DIM].view(1, ROPE_DIM),
                }
                inputs[input_name] = aliases[input_name]

                with self.assertRaisesRegex(
                    RuntimeError, f"q_out must not overlap {input_name}"
                ):
                    cuda_fused_qnorm_rope_kv_insert(
                        q_out=q_out,
                        rms_norm_eps=1.0e-6,
                        block_size=block_size,
                        **inputs,
                    )

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
        live_block_table = torch.arange(
            num_state_blocks, device=device, dtype=torch.int32
        ).view(1, -1)
        block_table = torch.full((1, 128), -1, device=device, dtype=torch.int32)
        block_table[:, : live_block_table.shape[1]].copy_(live_block_table)
        kv_slots = torch.full((num_tokens,), -1, device=device, dtype=torch.int64)
        kv_slots[-2] = 1
        kv_slots[-1] = 0
        cos_sin = torch.randn(256, ROPE_DIM, device=device, dtype=torch.float32) * 0.05
        rms_weight = (
            torch.randn(HEAD_DIM, device=device, dtype=torch.float32) * 0.1 + 1.0
        )
        cache = torch.zeros(
            1,
            kv_cache_block_size * (SWA_TOKEN_STRIDE + SWA_SCALE_DIM),
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
        scale_base = kv_cache_block_size * SWA_TOKEN_STRIDE
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
            flat_cache[NOPE_DIM:SWA_TOKEN_STRIDE].cpu(),
            expected_rope.cpu(),
            atol=0,
            rtol=0,
        )
        self.assertEqual(
            int(flat_cache[SWA_TOKEN_STRIDE : 2 * SWA_TOKEN_STRIDE].sum()),
            0,
        )

        for width in (16, 129, 256):
            with self.subTest(width=width):
                table_storage = torch.full(
                    (1, width * 2),
                    -1,
                    device=device,
                    dtype=torch.int32,
                )
                candidate_block_table = table_storage[:, ::2]
                candidate_block_table[:, : live_block_table.shape[1]].copy_(
                    live_block_table
                )
                candidate_cache = torch.zeros_like(cache)
                deepseek_v4_hca_compress_kv_cache_insert(
                    state_cache=state_cache,
                    token_to_req_indices=token_to_req_indices,
                    positions=positions,
                    compressor_slot_mapping=state_slots,
                    block_table=candidate_block_table,
                    compressor_block_size=state_block_size,
                    rms_norm_weight=rms_weight,
                    rms_norm_eps=eps,
                    cos_sin_cache=cos_sin,
                    kv_cache_2d=candidate_cache,
                    kv_slot_mapping=kv_slots,
                    kv_cache_block_size=kv_cache_block_size,
                    compress_ratio=compress_ratio,
                )
                torch.cuda.synchronize()
                self.assertTrue(torch.equal(candidate_cache, cache))

    def test_hca_trtllm_prefill_dispatches_fallback_and_empty(self):
        device = torch.device("cuda")
        common = {
            "state_cache": torch.zeros(
                (1, 1, HEAD_DIM * 2),
                dtype=torch.float32,
                device=device,
            ),
            "token_to_req_indices": torch.zeros(1, dtype=torch.int32, device=device),
            "positions": torch.zeros(1, dtype=torch.int64, device=device),
            "compressor_slot_mapping": torch.zeros(1, dtype=torch.int64, device=device),
            "block_table": torch.zeros((1, 1), dtype=torch.int32, device=device),
            "compressor_block_size": 1,
            "rms_norm_weight": torch.ones(HEAD_DIM, device=device),
            "rms_norm_eps": 1.0e-6,
            "cos_sin_cache": torch.ones((1, ROPE_DIM), device=device),
            "kv_cache_2d": torch.zeros(
                (1, SWA_TOKEN_STRIDE + SWA_SCALE_DIM),
                dtype=torch.uint8,
                device=device,
            ),
            "kv_slot_mapping": torch.zeros(1, dtype=torch.int64, device=device),
            "kv_cache_block_size": 1,
            "prefill_scratch": torch.zeros(
                (1, HEAD_DIM * 2),
                dtype=torch.float32,
                device=device,
            ),
            "query_start_loc": torch.tensor([0, 1], dtype=torch.int32, device=device),
            "seq_lens": torch.ones(1, dtype=torch.int32, device=device),
        }
        module = "tokenspeed.runtime.layers.attention.deepseek_v4_ops"

        with (
            mock.patch(
                f"{module}._supports_trtllm_c128_prefill_compress_cache",
                return_value=False,
            ) as supports,
            mock.patch(f"{module}._trtllm_c128_prefill_compress_cache") as native,
            mock.patch(
                f"{module}._triton_fused_sparse_compress_cache_insert"
            ) as triton,
        ):
            deepseek_v4_hca_compress_kv_cache_insert(**common, max_outputs=1)
        supports.assert_called_once()
        native.assert_not_called()
        triton.assert_called_once()

        with (
            mock.patch(
                f"{module}._supports_trtllm_c128_prefill_compress_cache"
            ) as supports,
            mock.patch(f"{module}._trtllm_c128_prefill_compress_cache") as native,
            mock.patch(
                f"{module}._triton_fused_sparse_compress_cache_insert"
            ) as triton,
        ):
            deepseek_v4_hca_compress_kv_cache_insert(**common, max_outputs=0)
        supports.assert_not_called()
        native.assert_not_called()
        triton.assert_not_called()

    def test_hca_trtllm_prefill_matches_triton_for_varlen_pages(self):
        torch.manual_seed(20260709)
        device = torch.device("cuda")
        major, _minor = torch.cuda.get_device_capability(device)
        if major != 10 or not has_trtllm_deepseek_v4_c128_prefill_compress_cache():
            self.skipTest("TRT-LLM C128 prefill op requires SM100 and its native DSO")
        state_block_size = 32
        kv_cache_block_size = 8
        table_width = 128
        eps = 1.0e-6
        # (query start position, final sequence length, first logical state page)
        requests = (
            (0, 256, 0),
            (1, 256, 0),
            (127, 256, 0),
            (385, 640, 12),
        )
        query_lens = [end - start for start, end, _ in requests]
        query_offsets = [0]
        for query_len in query_lens:
            query_offsets.append(query_offsets[-1] + query_len)

        query_start_loc = torch.tensor(
            query_offsets,
            dtype=torch.int32,
            device=device,
        )
        seq_lens = torch.tensor(
            [end for _, end, _ in requests],
            dtype=torch.int32,
            device=device,
        )
        positions = torch.cat(
            [
                torch.arange(start, end, dtype=torch.int64, device=device)
                for start, end, _ in requests
            ]
        )
        token_to_req = torch.repeat_interleave(
            torch.arange(len(requests), dtype=torch.int32, device=device),
            torch.tensor(query_lens, dtype=torch.int64, device=device),
        )

        block_table = torch.full(
            (len(requests), table_width),
            -1,
            dtype=torch.int32,
            device=device,
        )
        base_offsets = torch.tensor(
            [base for _, _, base in requests],
            dtype=torch.int32,
            device=device,
        )
        state_cache = torch.full(
            (80, state_block_size, HEAD_DIM * 2),
            float("nan"),
            dtype=torch.float32,
            device=device,
        )
        next_physical_page = 7
        dim = torch.arange(HEAD_DIM, dtype=torch.float32, device=device)
        for req_idx, (start, end, base) in enumerate(requests):
            first_page = (start // 128) * 128 // state_block_size
            last_page = end // state_block_size
            for logical_page in range(first_page, last_page):
                physical_page = next_physical_page
                next_physical_page += 2
                block_table[req_idx, logical_page - base] = physical_page
                row_positions = torch.arange(
                    logical_page * state_block_size,
                    (logical_page + 1) * state_block_size,
                    dtype=torch.float32,
                    device=device,
                )
                state_cache[physical_page, :, :HEAD_DIM] = (
                    torch.sin(row_positions[:, None] * 0.013 + dim[None, :] * 0.007)
                    + req_idx * 0.05
                )
                state_cache[physical_page, :, HEAD_DIM:] = 0.7 * torch.cos(
                    row_positions[:, None] * 0.009 - dim[None, :] * 0.005
                )

        angles = torch.arange(
            max(end for _, end, _ in requests) + 1,
            dtype=torch.float32,
            device=device,
        )[:, None] * (
            0.0003
            + torch.arange(ROPE_DIM // 2, dtype=torch.float32, device=device)[None, :]
            * 0.00001
        )
        cos_sin = torch.cat((torch.cos(angles), torch.sin(angles)), dim=1)
        kv_slots = torch.full(
            (positions.numel(),),
            -1,
            dtype=torch.int64,
            device=device,
        )
        boundary_rows: list[int] = []
        boundary_slots: list[int] = []
        slot_plan = ((3, 2), (5, 1), (7, 3), (9, 0))
        for req_idx, (start, end, _) in enumerate(requests):
            first_window = start // 128
            for output_idx in range(end // 128 - first_window):
                boundary_position = (first_window + output_idx + 1) * 128 - 1
                boundary_row = query_offsets[req_idx] + boundary_position - start
                page, first_offset = slot_plan[req_idx]
                slot = page * kv_cache_block_size + first_offset + output_idx
                kv_slots[boundary_row] = slot
                boundary_rows.append(boundary_row)
                boundary_slots.append(slot)

        compressor_slots = torch.zeros_like(kv_slots)
        invalid_state_row = boundary_rows[2]
        invalid_state_slot = boundary_slots[2]
        invalid_kv_row = boundary_rows[4]
        compressor_slots[invalid_state_row] = -1
        kv_slots[invalid_kv_row] = -1
        writable_boundary_rows = [
            row
            for row in boundary_rows
            if row not in (invalid_state_row, invalid_kv_row)
        ]
        scratch_source = torch.randn(
            positions.numel(),
            HEAD_DIM * 2,
            dtype=torch.float32,
            device=device,
        )
        base_weight = 0.75 + 0.5 * torch.rand(HEAD_DIM, device=device)
        writable_boundary_rows_tensor = torch.tensor(
            writable_boundary_rows,
            dtype=torch.int64,
            device=device,
        )
        unmodified_rows = torch.ones(
            positions.numel(),
            dtype=torch.bool,
            device=device,
        )
        unmodified_rows[writable_boundary_rows_tensor] = False

        for weight_dtype in (torch.bfloat16, torch.float32):
            with self.subTest(weight_dtype=weight_dtype):
                rms_weight = base_weight.to(weight_dtype)
                scratch = scratch_source.clone()
                candidate_cache = torch.zeros(
                    16,
                    kv_cache_block_size * (SWA_TOKEN_STRIDE + SWA_SCALE_DIM),
                    dtype=torch.uint8,
                    device=device,
                )
                reference_cache = torch.zeros_like(candidate_cache)
                self.assertTrue(
                    supports_trtllm_deepseek_v4_c128_prefill_compress_cache(
                        state_cache,
                        scratch,
                        positions,
                        compressor_slots,
                        query_start_loc,
                        seq_lens,
                        block_table,
                        rms_weight,
                        cos_sin,
                        candidate_cache,
                        kv_slots,
                        block_table_base_offsets=base_offsets,
                        state_block_size=state_block_size,
                        kv_block_size=kv_cache_block_size,
                        max_outputs=2,
                    ),
                    "legal SM100 fixture must select the native C128 path",
                )

                deepseek_v4_hca_compress_kv_cache_insert(
                    state_cache=state_cache,
                    token_to_req_indices=token_to_req,
                    positions=positions,
                    compressor_slot_mapping=compressor_slots,
                    block_table=block_table,
                    compressor_block_size=state_block_size,
                    rms_norm_weight=rms_weight,
                    rms_norm_eps=eps,
                    cos_sin_cache=cos_sin,
                    kv_cache_2d=candidate_cache,
                    kv_slot_mapping=kv_slots,
                    kv_cache_block_size=kv_cache_block_size,
                    prefill_scratch=scratch,
                    query_start_loc=query_start_loc,
                    seq_lens=seq_lens,
                    max_outputs=2,
                    block_table_base_offsets=base_offsets,
                )
                deepseek_v4_hca_compress_kv_cache_insert(
                    state_cache=state_cache,
                    token_to_req_indices=token_to_req,
                    positions=positions,
                    compressor_slot_mapping=compressor_slots,
                    block_table=block_table,
                    compressor_block_size=state_block_size,
                    rms_norm_weight=rms_weight,
                    rms_norm_eps=eps,
                    cos_sin_cache=cos_sin,
                    kv_cache_2d=reference_cache,
                    kv_slot_mapping=kv_slots,
                    kv_cache_block_size=kv_cache_block_size,
                    block_table_base_offsets=base_offsets,
                )
                torch.cuda.synchronize()

                candidate_values = dequantize_deepseek_v4_fp8_ds_mla_cache(
                    candidate_cache,
                    kv_slots[kv_slots >= 0],
                    kv_cache_block_size,
                    head_dim=HEAD_DIM,
                    rope_dim=ROPE_DIM,
                )
                reference_values = dequantize_deepseek_v4_fp8_ds_mla_cache(
                    reference_cache,
                    kv_slots[kv_slots >= 0],
                    kv_cache_block_size,
                    head_dim=HEAD_DIM,
                    rope_dim=ROPE_DIM,
                )
                torch.testing.assert_close(
                    candidate_values,
                    reference_values,
                    atol=0.125,
                    rtol=0.05,
                )
                torch.testing.assert_close(
                    candidate_values[:, NOPE_DIM:],
                    reference_values[:, NOPE_DIM:],
                    atol=0.01,
                    rtol=0.01,
                )
                for slot in kv_slots[kv_slots >= 0].tolist():
                    page = slot // kv_cache_block_size
                    offset = (
                        kv_cache_block_size * SWA_TOKEN_STRIDE
                        + (slot % kv_cache_block_size) * SWA_SCALE_DIM
                    )
                    self.assertTrue(
                        torch.equal(
                            candidate_cache[page, offset : offset + SWA_SCALE_DIM],
                            reference_cache[page, offset : offset + SWA_SCALE_DIM],
                        )
                    )
                skipped_value = dequantize_deepseek_v4_fp8_ds_mla_cache(
                    candidate_cache,
                    torch.tensor(
                        [invalid_state_slot],
                        dtype=torch.int64,
                        device=device,
                    ),
                    kv_cache_block_size,
                    head_dim=HEAD_DIM,
                    rope_dim=ROPE_DIM,
                )
                self.assertEqual(float(skipped_value.abs().sum()), 0.0)
                self.assertTrue(
                    torch.equal(
                        scratch[unmodified_rows],
                        scratch_source[unmodified_rows],
                    )
                )
                self.assertTrue(
                    torch.equal(
                        scratch[writable_boundary_rows_tensor, HEAD_DIM:],
                        scratch_source[writable_boundary_rows_tensor, HEAD_DIM:],
                    )
                )
                for req_idx, (start, end, base) in enumerate(requests):
                    first_window = start // 128
                    for output_idx in range(end // 128 - first_window):
                        window_start = (first_window + output_idx) * 128
                        window_positions = torch.arange(
                            window_start,
                            window_start + 128,
                            dtype=torch.int64,
                            device=device,
                        )
                        logical_pages = torch.div(
                            window_positions,
                            state_block_size,
                            rounding_mode="floor",
                        )
                        physical_pages = block_table[req_idx, logical_pages - base].to(
                            torch.int64
                        )
                        rows = state_cache[
                            physical_pages,
                            torch.remainder(window_positions, state_block_size),
                        ]
                        expected = torch.sum(
                            rows[:, :HEAD_DIM]
                            * torch.softmax(rows[:, HEAD_DIM:], dim=0),
                            dim=0,
                        )
                        boundary_position = (first_window + output_idx + 1) * 128 - 1
                        boundary_row = (
                            query_offsets[req_idx] + boundary_position - start
                        )
                        if boundary_row not in writable_boundary_rows:
                            continue
                        torch.testing.assert_close(
                            scratch[boundary_row, :HEAD_DIM],
                            expected,
                            atol=1.0e-6,
                            rtol=1.0e-5,
                        )

                first_result = candidate_cache.clone()
                deepseek_v4_hca_compress_kv_cache_insert(
                    state_cache=state_cache,
                    token_to_req_indices=token_to_req,
                    positions=positions,
                    compressor_slot_mapping=compressor_slots,
                    block_table=block_table,
                    compressor_block_size=state_block_size,
                    rms_norm_weight=rms_weight,
                    rms_norm_eps=eps,
                    cos_sin_cache=cos_sin,
                    kv_cache_2d=candidate_cache,
                    kv_slot_mapping=kv_slots,
                    kv_cache_block_size=kv_cache_block_size,
                    prefill_scratch=scratch,
                    query_start_loc=query_start_loc,
                    seq_lens=seq_lens,
                    max_outputs=2,
                    block_table_base_offsets=base_offsets,
                )
                torch.cuda.synchronize()
                self.assertTrue(torch.equal(candidate_cache, first_result))

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
            kv_cache_block_size * (SWA_TOKEN_STRIDE + SWA_SCALE_DIM),
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
            base = slot * SWA_TOKEN_STRIDE
            scale_base = kv_cache_block_size * SWA_TOKEN_STRIDE
            scale_base += slot * SWA_SCALE_DIM
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
                flat_cache[base + NOPE_DIM : base + SWA_TOKEN_STRIDE].cpu(),
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

        topk = _indexer_topk_reference(
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

    def _assert_persistent_topk_matches_torch(
        self,
        logits: torch.Tensor,
        lengths: torch.Tensor,
        output: torch.Tensor,
        topk: int,
    ) -> None:
        for row_idx, raw_len in enumerate(lengths.cpu().tolist()):
            row_output = output[row_idx].cpu()
            if raw_len <= topk:
                expected = torch.full((topk,), -1, dtype=torch.int32)
                if raw_len > 0:
                    expected[:raw_len] = torch.arange(raw_len, dtype=torch.int32)
                self.assertTrue(torch.equal(row_output, expected))
                continue

            expected = (
                torch.topk(
                    logits[row_idx, :raw_len],
                    k=topk,
                    dim=-1,
                    sorted=False,
                )
                .indices.to(torch.int32)
                .sort()
                .values.cpu()
            )
            self.assertTrue(torch.equal(row_output.sort().values, expected))

    def test_persistent_topk_matches_torch_across_decode_medium_large_paths(self):
        if not has_persistent_topk():
            self.skipTest("DeepSeek V4 persistent top-k op is not available")

        torch.manual_seed(6790)
        device = torch.device("cuda")
        topk = 512
        lengths = torch.tensor(
            [0, 7, 513, 9000, 33000], device=device, dtype=torch.int32
        )
        stride = 33024
        logits = torch.randn(
            (lengths.numel(), stride), device=device, dtype=torch.float32
        )
        for row_idx, raw_len in enumerate(lengths.cpu().tolist()):
            if raw_len < stride:
                logits[row_idx, raw_len:] = 1.0e6
        output = torch.full(
            (lengths.numel(), topk), -77, device=device, dtype=torch.int32
        )
        workspace = torch.empty((1024 * 1024,), device=device, dtype=torch.uint8)

        persistent_topk(
            logits,
            lengths,
            output,
            workspace,
            topk,
            int(lengths.max().item()),
        )
        torch.cuda.synchronize()

        self._assert_persistent_topk_matches_torch(logits, lengths, output, topk)

    def test_persistent_topk_matches_torch_for_batch_gt_32(self):
        if not has_persistent_topk():
            self.skipTest("DeepSeek V4 persistent top-k op is not available")

        torch.manual_seed(6791)
        device = torch.device("cuda")
        topk = 512
        num_rows = 36
        stride = 544
        lengths = torch.tensor(
            [0, 17] + [520 + (idx % 24) for idx in range(num_rows - 2)],
            device=device,
            dtype=torch.int32,
        )
        logits = torch.randn((num_rows, stride), device=device, dtype=torch.float32)
        for row_idx, raw_len in enumerate(lengths.cpu().tolist()):
            if raw_len < stride:
                logits[row_idx, raw_len:] = 1.0e6
        output = torch.full((num_rows, topk), -77, device=device, dtype=torch.int32)
        workspace = torch.empty((1024 * 1024,), device=device, dtype=torch.uint8)

        persistent_topk(
            logits,
            lengths,
            output,
            workspace,
            topk,
            int(lengths.max().item()),
        )
        torch.cuda.synchronize()

        self._assert_persistent_topk_matches_torch(logits, lengths, output, topk)

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

    def test_indexer_mxfp4_paged_gather_matches_paged_layout(self):
        if not has_indexer_mxfp4_paged_gather():
            self.skipTest("DeepSeek V4 paged MXFP4 gather op is not available")

        device = torch.device("cuda")
        block_size = 4
        value_bytes = 64
        scale_bytes = 4
        num_blocks = 3
        kv_cache = torch.zeros(
            num_blocks,
            block_size * (value_bytes + scale_bytes),
            device=device,
            dtype=torch.uint8,
        )

        value_rows = {}
        scale_rows = {}
        for block_idx in range(num_blocks):
            for row_idx in range(block_size):
                values = (
                    (
                        torch.arange(value_bytes, device=device, dtype=torch.int16)
                        + block_idx * 37
                        + row_idx * 11
                    )
                    .remainder(251)
                    .to(torch.uint8)
                )
                scales = torch.tensor(
                    [block_idx, row_idx, block_idx * 17 + row_idx, 200 + block_idx],
                    device=device,
                    dtype=torch.uint8,
                )
                value_base = row_idx * value_bytes
                scale_base = block_size * value_bytes + row_idx * scale_bytes
                kv_cache[block_idx, value_base : value_base + value_bytes].copy_(values)
                kv_cache[block_idx, scale_base : scale_base + scale_bytes].copy_(scales)
                value_rows[(block_idx, row_idx)] = values
                scale_rows[(block_idx, row_idx)] = scales

        block_table = torch.tensor([[2, 0], [1, 0]], device=device, dtype=torch.int32)
        cu_seq_lens = torch.tensor([0, 5, 7], device=device, dtype=torch.int32)
        values_out = torch.full(
            (8, value_bytes), 0xCC, device=device, dtype=torch.uint8
        )
        scales_out = torch.full(
            (8, scale_bytes), 0xDD, device=device, dtype=torch.uint8
        )

        indexer_mxfp4_paged_gather(
            kv_cache,
            values_out,
            scales_out,
            block_table,
            cu_seq_lens,
            block_size,
        )
        torch.cuda.synchronize()

        expected_plan = [
            (2, 0),
            (2, 1),
            (2, 2),
            (2, 3),
            (0, 0),
            (1, 0),
            (1, 1),
        ]
        expected_values = torch.stack([value_rows[item] for item in expected_plan])
        expected_scales = torch.stack([scale_rows[item] for item in expected_plan])
        self.assertTrue(torch.equal(values_out[:7].cpu(), expected_values.cpu()))
        self.assertTrue(torch.equal(scales_out[:7].cpu(), expected_scales.cpu()))
        self.assertTrue(
            torch.equal(values_out[7].cpu(), torch.full((64,), 0xCC, dtype=torch.uint8))
        )
        self.assertTrue(
            torch.equal(scales_out[7].cpu(), torch.full((4,), 0xDD, dtype=torch.uint8))
        )

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
            block_size * (SWA_TOKEN_STRIDE + SWA_SCALE_DIM),
            device=device,
            dtype=torch.uint8,
        )
        for token_idx, slot in enumerate(slots.tolist()):
            if slot < 0:
                continue
            flat = cache.view(-1)
            token_base = slot * SWA_TOKEN_STRIDE
            scale_base = block_size * SWA_TOKEN_STRIDE + slot * SWA_SCALE_DIM
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
            flat[token_base + NOPE_DIM : token_base + SWA_TOKEN_STRIDE].copy_(
                rotated[NOPE_DIM:].to(torch.bfloat16).view(torch.uint8)
            )

        dequant = dequantize_deepseek_v4_fp8_ds_mla_cache(
            cache,
            slots,
            block_size,
            head_dim=HEAD_DIM,
            rope_dim=ROPE_DIM,
        )
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
        q_fp4, weights_fp4 = _prepare_indexer_q_reference(
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
        q_fp8, weights_fp8 = _prepare_indexer_q_reference(
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

    def test_large_indexer_q_prepare_dispatches_to_trtllm_kernels(self):
        from tokenspeed_kernel.ops.attention.trtllm.deepseek_v4 import (
            has_trtllm_deepseek_v4_indexer_q_prepare,
            trtllm_deepseek_v4_indexer_q_prepare_mxfp4,
        )

        if (
            torch.cuda.get_device_capability()[0] != 10
            or not has_trtllm_deepseek_v4_indexer_q_prepare()
        ):
            self.skipTest("TRT-LLM indexer Q kernels require NVIDIA SM100/SM10x")

        torch.manual_seed(9125)
        num_tokens = 128
        num_heads = 64
        index_q = torch.randn(
            num_tokens,
            num_heads,
            128,
            device="cuda",
            dtype=torch.bfloat16,
        )
        original_q = index_q.clone()
        expected_q = index_q.clone()
        positions = torch.arange(num_tokens, device="cuda", dtype=torch.int64)
        cos_sin = torch.randn(
            num_tokens,
            ROPE_DIM,
            device="cuda",
            dtype=torch.float32,
        )
        weights = torch.randn(
            num_tokens,
            num_heads,
            device="cuda",
            dtype=torch.float32,
        )
        kwargs = {
            "positions": positions,
            "cos_sin_cache": cos_sin,
            "weights": weights,
            "softmax_scale": 0.25,
            "head_scale": num_heads**-0.5,
        }

        expected = trtllm_deepseek_v4_indexer_q_prepare_mxfp4(
            index_q=expected_q,
            enable_pdl=True,
            **kwargs,
        )
        actual = deepseek_v4_prepare_indexer_q_mxfp4(
            index_q=index_q,
            **kwargs,
        )

        self.assertTrue(torch.equal(actual[0][0], expected[0][0]))
        self.assertTrue(torch.equal(actual[0][1], expected[0][1]))
        self.assertTrue(torch.equal(actual[1], expected[1]))
        self.assertTrue(torch.equal(index_q, expected_q))
        self.assertFalse(torch.equal(index_q, original_q))

    def test_small_indexer_q_prepare_keeps_triton_fallback(self):
        from tokenspeed_kernel.ops.attention.trtllm.deepseek_v4 import (
            has_trtllm_deepseek_v4_indexer_q_prepare,
        )

        if (
            torch.cuda.get_device_capability()[0] != 10
            or not has_trtllm_deepseek_v4_indexer_q_prepare()
        ):
            self.skipTest("TRT-LLM indexer Q kernels require NVIDIA SM100/SM10x")

        torch.manual_seed(9126)
        num_tokens = 127
        index_q = torch.randn(
            num_tokens,
            64,
            128,
            device="cuda",
            dtype=torch.bfloat16,
        )
        original_q = index_q.clone()
        positions = torch.arange(num_tokens, device="cuda", dtype=torch.int64)
        cos_sin = torch.randn(
            num_tokens,
            ROPE_DIM,
            device="cuda",
            dtype=torch.float32,
        )
        weights = torch.randn(num_tokens, 64, device="cuda", dtype=torch.float32)

        deepseek_v4_prepare_indexer_q_mxfp4(
            index_q=index_q,
            positions=positions,
            cos_sin_cache=cos_sin,
            weights=weights,
            softmax_scale=0.25,
            head_scale=64**-0.5,
        )

        self.assertTrue(torch.equal(index_q, original_q))

    def test_large_indexer_q_prepare_keeps_triton_fallback_when_unsupported(self):
        num_tokens = 128
        index_q = torch.empty(
            num_tokens,
            64,
            128,
            device="cuda",
            dtype=torch.bfloat16,
        )
        positions = torch.zeros(num_tokens, device="cuda", dtype=torch.int64)
        cos_sin = torch.empty(1, ROPE_DIM, device="cuda", dtype=torch.float32)
        weights = torch.empty(num_tokens, 64, device="cuda", dtype=torch.float32)
        sentinel = object()

        with (
            mock.patch(
                "tokenspeed.runtime.layers.attention.deepseek_v4_ops."
                "_supports_trtllm_indexer_q_prepare",
                return_value=False,
            ) as supports,
            mock.patch(
                "tokenspeed.runtime.layers.attention.deepseek_v4_ops."
                "_trtllm_indexer_q_prepare_mxfp4"
            ) as trtllm_prepare,
            mock.patch(
                "tokenspeed.runtime.layers.attention.deepseek_v4_ops."
                "_triton_fused_indexer_q_rope_hadamard_mxfp4",
                return_value=sentinel,
            ) as triton_prepare,
        ):
            actual = deepseek_v4_prepare_indexer_q_mxfp4(
                index_q,
                positions,
                cos_sin,
                weights,
                0.25,
                64**-0.5,
            )

        self.assertIs(actual, sentinel)
        supports.assert_called_once()
        trtllm_prepare.assert_not_called()
        triton_prepare.assert_called_once()

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
        block_table_storage = torch.full((2, 4), -1, device=device, dtype=torch.int64)
        block_table_storage[:, :2].copy_(
            torch.tensor([[10, 11], [20, 21]], device=device, dtype=torch.int64)
        )
        block_table = block_table_storage[:, :2]
        self.assertNotEqual(block_table.stride(0), block_table.shape[1])
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

    def test_decode_swa_indices_and_lens_masks_invalid_tokens(self):
        device = torch.device("cuda")
        query_start_loc = torch.tensor([0, 1, 2], device=device, dtype=torch.int32)
        seq_lens = torch.tensor([70, 3], device=device, dtype=torch.int32)
        token_to_req_indices = torch.tensor([0, 1], device=device, dtype=torch.int32)
        is_valid_token = torch.tensor([True, False], device=device)
        block_table = torch.tensor(
            [[10, 11], [20, 21]],
            device=device,
            dtype=torch.int32,
        )
        out_indices = torch.full((2, 4), -123, device=device, dtype=torch.int32)
        out_lens = torch.empty((2,), device=device, dtype=torch.int32)

        actual, actual_lens = deepseek_v4_decode_swa_indices_and_lens(
            query_start_loc=query_start_loc,
            seq_lens=seq_lens,
            token_to_req_indices=token_to_req_indices,
            block_table=block_table,
            window_size=4,
            block_size=64,
            is_valid_token=is_valid_token,
            out_indices=out_indices,
            out_lens=out_lens,
        )
        torch.cuda.synchronize()

        self.assertTrue(
            torch.equal(actual_lens.cpu(), torch.tensor([4, 0], dtype=torch.int32))
        )
        self.assertTrue(
            torch.equal(
                actual[0].cpu(),
                torch.tensor([706, 707, 708, 709], dtype=torch.int32),
            )
        )
        self.assertTrue(
            torch.equal(actual[1].cpu(), torch.full((4,), -123, dtype=torch.int32))
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
        block_table_storage = torch.full((2, 4), -1, device=device, dtype=torch.int64)
        block_table_storage[:, :2].copy_(
            torch.tensor([[10, 11], [20, 21]], device=device, dtype=torch.int64)
        )
        block_table = block_table_storage[:, :2]
        self.assertNotEqual(block_table.stride(0), block_table.shape[1])

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

    def test_compute_global_topk_indices_and_lens_masks_invalid_tokens(self):
        device = torch.device("cuda")
        topk_indices = torch.tensor(
            [
                [0, 1, -1, 5],
                [3, -1, -1, -1],
            ],
            device=device,
            dtype=torch.int32,
        )
        token_to_req_indices = torch.tensor([0, 1], device=device, dtype=torch.int32)
        is_valid_token = torch.tensor([True, False], device=device)
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
            is_valid_token=is_valid_token,
        )
        torch.cuda.synchronize()

        self.assertTrue(
            torch.equal(actual_lens.cpu(), torch.tensor([3, 0], dtype=torch.int32))
        )
        self.assertTrue(
            torch.equal(
                actual[0].cpu(),
                torch.tensor([40, 41, -1, 45], dtype=torch.int32),
            )
        )

    def test_compressed_slot_mapping_matches_page_reference(self):
        device = torch.device("cuda")
        query_start_loc = torch.tensor([0, 3, 5], device=device, dtype=torch.int32)
        seq_lens = torch.tensor([8, 6], device=device, dtype=torch.int32)
        block_table_storage = torch.full((2, 6), -1, device=device, dtype=torch.int32)
        block_table_storage[:, ::2].copy_(
            torch.tensor([[10, 11, 12], [20, 21, 22]], device=device)
        )
        block_table = block_table_storage[:, ::2]
        self.assertEqual(block_table.stride(-1), 2)
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
            block_size * (SWA_TOKEN_STRIDE + SWA_SCALE_DIM),
            device=device,
            dtype=torch.uint8,
        )
        rows = torch.randn(num_blocks * block_size, HEAD_DIM, device=device)
        flat_cache = cache.view(-1)
        for slot in range(rows.shape[0]):
            page = slot // block_size
            pos = slot % block_size
            page_base = page * cache.stride(0)
            token_base = page_base + pos * SWA_TOKEN_STRIDE
            scale_base = page_base + block_size * SWA_TOKEN_STRIDE + pos * SWA_SCALE_DIM
            for qblock in range(7):
                start = qblock * 64
                fp8_bytes, scale = _fp8_bytes_and_scale(rows[slot, start : start + 64])
                flat_cache[token_base + start : token_base + start + 64].copy_(
                    fp8_bytes
                )
                flat_cache[scale_base + qblock] = scale
            flat_cache[scale_base + 7] = 0
            flat_cache[token_base + NOPE_DIM : token_base + SWA_TOKEN_STRIDE].copy_(
                rows[slot, NOPE_DIM:].to(torch.bfloat16).view(torch.uint8)
            )

        seq_lens = torch.tensor([9, 8], device=device, dtype=torch.int32)
        gather_lens = torch.tensor([3, 2], device=device, dtype=torch.int32)
        block_table_storage = torch.full((2, 4), -1, device=device, dtype=torch.int32)
        block_table_storage[:, ::2].copy_(
            torch.tensor([[0, 1], [2, 0]], device=device, dtype=torch.int32)
        )
        block_table = block_table_storage[:, ::2]
        self.assertEqual(block_table.stride(-1), 2)
        block_table_base_offsets = torch.tensor(
            [1, 1], device=device, dtype=torch.int32
        )
        sentinel = 7.0
        out = torch.full(
            (2, 5, HEAD_DIM),
            sentinel,
            device=device,
            dtype=torch.bfloat16,
        )
        deepseek_v4_dequantize_and_gather_k_cache(
            out=out,
            cache_2d=cache,
            seq_lens=seq_lens,
            gather_lens=gather_lens,
            block_table=block_table,
            block_table_base_offsets=block_table_base_offsets,
            block_size=block_size,
            offset=1,
            max_gather_len=3,
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
                cache,
                slots,
                block_size,
                head_dim=HEAD_DIM,
                rope_dim=ROPE_DIM,
            )
            gathered = out[req_idx, 1 : 1 + slots.numel()]
            torch.testing.assert_close(
                gathered.float().cpu(),
                expected_rows.float().cpu(),
                atol=0,
                rtol=0,
            )
            self.assertTrue(torch.all(out[req_idx, 0] == sentinel).item())
            self.assertTrue(
                torch.all(out[req_idx, 1 + slots.numel() :] == sentinel).item()
            )

    def test_sparse_prefill_dequantize_and_gather_long_grid_matches_reference(self):
        torch.manual_seed(20260709)
        device = torch.device("cuda")
        block_size = 64
        gather_lens = torch.tensor([6145, 3073], device=device, dtype=torch.int32)
        seq_lens = gather_lens + 128
        max_pages = math.ceil(int(seq_lens.max().item()) / block_size)
        num_reqs = int(seq_lens.numel())
        cache = torch.zeros(
            (
                num_reqs * max_pages,
                block_size * (SWA_TOKEN_STRIDE + SWA_SCALE_DIM),
            ),
            device=device,
            dtype=torch.uint8,
        )
        token_data = cache[:, : block_size * SWA_TOKEN_STRIDE].view(
            num_reqs * max_pages,
            block_size,
            SWA_TOKEN_STRIDE,
        )
        token_data[:, :, :NOPE_DIM].random_(0, 127)
        token_data[:, :, NOPE_DIM:].copy_(
            torch.randn(
                num_reqs * max_pages,
                block_size,
                ROPE_DIM,
                device=device,
                dtype=torch.bfloat16,
            )
            .view(torch.uint8)
            .reshape(num_reqs * max_pages, block_size, 2 * ROPE_DIM)
        )
        cache[:, block_size * SWA_TOKEN_STRIDE :].random_(124, 131)
        block_table = torch.arange(
            num_reqs * max_pages,
            device=device,
            dtype=torch.int32,
        ).view(num_reqs, max_pages)
        sentinel = 7.0
        output_offset = 3
        output_tail = 5
        out = torch.full(
            (
                num_reqs,
                output_offset + int(gather_lens.max().item()) + output_tail,
                HEAD_DIM,
            ),
            sentinel,
            device=device,
            dtype=torch.bfloat16,
        )

        deepseek_v4_dequantize_and_gather_k_cache(
            out=out,
            cache_2d=cache,
            seq_lens=seq_lens,
            gather_lens=gather_lens,
            block_table=block_table,
            block_size=block_size,
            offset=output_offset,
            max_gather_len=int(gather_lens.max().item()),
        )
        torch.cuda.synchronize()

        for req_idx, gather_len in enumerate(gather_lens.tolist()):
            positions = torch.arange(
                int(seq_lens[req_idx].item()) - gather_len,
                int(seq_lens[req_idx].item()),
                device=device,
                dtype=torch.int64,
            )
            slots = (
                block_table[req_idx, positions // block_size].to(torch.int64)
                * block_size
                + positions % block_size
            )
            expected = dequantize_deepseek_v4_fp8_ds_mla_cache(
                cache,
                slots,
                block_size,
                head_dim=HEAD_DIM,
                rope_dim=ROPE_DIM,
            )
            torch.testing.assert_close(
                out[req_idx, output_offset : output_offset + gather_len].float().cpu(),
                expected.float().cpu(),
                atol=0,
                rtol=0,
            )
            self.assertTrue(torch.all(out[req_idx, :output_offset] == sentinel).item())
            self.assertTrue(
                torch.all(out[req_idx, output_offset + gather_len :] == sentinel).item()
            )

        full_out = torch.full(
            (num_reqs, int(gather_lens.max().item()) + output_tail, HEAD_DIM),
            sentinel,
            device=device,
            dtype=torch.bfloat16,
        )
        deepseek_v4_dequantize_and_gather_k_cache(
            out=full_out,
            cache_2d=cache,
            seq_lens=gather_lens,
            gather_lens=None,
            block_table=block_table,
            block_size=block_size,
            offset=0,
            max_gather_len=int(gather_lens.max().item()),
        )
        torch.cuda.synchronize()

        for req_idx, gather_len in enumerate(gather_lens.tolist()):
            slots = block_table[req_idx, 0].to(torch.int64) * block_size + torch.arange(
                gather_len, device=device, dtype=torch.int64
            )
            expected = dequantize_deepseek_v4_fp8_ds_mla_cache(
                cache,
                slots,
                block_size,
                head_dim=HEAD_DIM,
                rope_dim=ROPE_DIM,
            )
            torch.testing.assert_close(
                full_out[req_idx, :gather_len].float().cpu(),
                expected.float().cpu(),
                atol=0,
                rtol=0,
            )
            self.assertTrue(
                torch.all(full_out[req_idx, gather_len:] == sentinel).item()
            )


if __name__ == "__main__":
    unittest.main()
