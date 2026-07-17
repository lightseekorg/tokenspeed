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
import os
import sys
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=90, suite="runtime-1gpu")

from tokenspeed_kernel.ops.attention.cuda.deepseek_v4 import (
    has_indexer_mxfp4_paged_gather,
    has_persistent_topk,
    indexer_mxfp4_paged_gather,
    persistent_topk,
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
    read_deepseek_v4_indexer_fp8_cache,
    read_deepseek_v4_indexer_mxfp4_cache,
    save_deepseek_v4_compressor_state,
    write_deepseek_v4_indexer_fp8_cache,
    write_deepseek_v4_indexer_mxfp4_cache,
)
from tokenspeed.runtime.layers.attention.kv_cache.deepseek_v4 import (
    _group_slot_mapping_from_raw,
    _mask_invalid_graph_tokens,
)
from tokenspeed.runtime.models import deepseek_v4 as deepseek_v4_model
from tokenspeed.runtime.models.deepseek_v4 import (
    DeepseekV4Attention,
    DeepseekV4Compressor,
    DeepseekV4Indexer,
    _deepseek_v4_sanitize_swa_slot_mapping,
    _DeepseekV4PregraphBuffers,
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


def _dequantize_packed_indexer_q(
    values: torch.Tensor,
    scales: torch.Tensor,
) -> torch.Tensor:
    """Decode the four MXFP4 blocks used by the indexer Q tests."""
    scale_bytes = scales.contiguous().view(torch.uint8).reshape(*scales.shape, 4)
    out = torch.empty(
        (*values.shape[:-1], values.shape[-1] * 2),
        device=values.device,
        dtype=torch.float32,
    )
    for block_id in range(4):
        packed = values[..., block_id * 16 : (block_id + 1) * 16]
        scale = torch.pow(
            2.0,
            scale_bytes[..., block_id].to(torch.float32) - 127.0,
        ).unsqueeze(-1)
        target = out[..., block_id * 32 : (block_id + 1) * 32]
        target[..., 0::2] = _e2m1_values(packed & 0x0F) * scale
        target[..., 1::2] = _e2m1_values(packed >> 4) * scale
    return out


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
    def test_attention_phase1_uses_bucket_then_core_slices_every_tensor(self):
        bucket_tokens = 5
        real_tokens = 3

        for ambient, expected_rows in ((True, real_tokens), (False, bucket_tokens)):
            with self.subTest(ambient=ambient):
                attention = object.__new__(DeepseekV4Attention)
                torch.nn.Module.__init__(attention)
                attention.attention_kind = "csa"
                attention.rotary_emb = SimpleNamespace(
                    cos_sin_cache=torch.zeros(1, ROPE_DIM, dtype=torch.float32)
                )
                attention.stream_fork = StreamFork(None)
                attention.padded_heads = 16
                attention.cache_layer_index = 0
                attention.compress_ratio = 4
                attention.pregraph_buffers = None
                attention.num_local_heads = 16
                attention.head_dim = HEAD_DIM
                attention.swa_window = 64
                attention.scale = 1.0
                attention.attn_sink = torch.zeros(1)

                q = torch.randn(bucket_tokens, 16, HEAD_DIM, dtype=torch.bfloat16)
                kv = torch.randn(bucket_tokens, HEAD_DIM, dtype=torch.bfloat16)
                qr = torch.randn(bucket_tokens, 32, dtype=torch.bfloat16)
                compressor_score = torch.randn(bucket_tokens, 17)
                indexer_compressor_score = torch.randn(bucket_tokens, 19)
                attention._project_q_kv = mock.Mock(return_value=(q, kv, qr))

                compressor = mock.Mock()
                compressor.compute_kv_score.return_value = compressor_score
                attention.compressor = compressor
                indexer = mock.Mock()
                indexer.compressor.compute_kv_score.return_value = (
                    indexer_compressor_score
                )
                packed_q_values = torch.full(
                    (bucket_tokens, 2, 64), 0x11, dtype=torch.uint8
                )
                packed_q_scales = torch.full(
                    (bucket_tokens, 2), 0x01010101, dtype=torch.int32
                )
                packed_weights = torch.ones(bucket_tokens, 2)
                packed_q_values[real_tokens:].fill_(0xFF)
                packed_q_scales[real_tokens:].fill_(torch.iinfo(torch.int32).max)
                packed_weights[real_tokens:].fill_(float("nan"))
                indexer._prepare_packed_inputs.return_value = (
                    packed_q_values,
                    packed_q_scales,
                    packed_weights,
                )
                indexer.side_effect = lambda **kw: torch.zeros(
                    (kw["positions"].shape[0], 4), dtype=torch.int32
                )
                attention.indexer = indexer
                attention._insert_swa_cache = mock.Mock()
                attention._project_attention_output = mock.Mock(
                    side_effect=lambda output, *_: output
                )

                metadata = SimpleNamespace(
                    is_valid_token=None,
                    token_to_req_indices=torch.arange(real_tokens),
                )
                backend = SimpleNamespace(
                    forward_metadata=metadata,
                    forward_deepseek_v4_prefill=mock.Mock(
                        side_effect=lambda **kw: kw["q"]
                    ),
                )
                pool = SimpleNamespace(
                    swa_block_size=4,
                    get_swa_kv_buffer=mock.Mock(
                        return_value=torch.empty(1, 4 * 584, dtype=torch.uint8)
                    ),
                    get_indexer_block_size=mock.Mock(return_value=64),
                )
                ctx = SimpleNamespace(
                    dsa_compressor_slot_cache=None,
                    dsa_swa_slot_mapping=None,
                    token_to_kv_pool=pool,
                    attn_backend=backend,
                    forward_mode=ForwardMode.EXTEND,
                    dsa_pregraph_writes=False,
                )
                positions = torch.arange(bucket_tokens, dtype=torch.int64)
                hidden_states = torch.randn(bucket_tokens, 8)
                out_cache_loc = torch.arange(bucket_tokens, dtype=torch.int64) + 10
                swa_slot_mapping = torch.arange(bucket_tokens, dtype=torch.int64) + 20

                with mock.patch(
                    "tokenspeed.runtime.models.deepseek_v4.current_forward_ctx",
                    return_value=ctx if ambient else None,
                ):
                    attention(
                        positions,
                        hidden_states,
                        ctx,
                        out_cache_loc,
                        swa_slot_mapping=swa_slot_mapping,
                    )

                self.assertEqual(
                    attention._project_q_kv.call_args.args[0].shape[0], bucket_tokens
                )
                self.assertEqual(
                    compressor.compute_kv_score.call_args.args[0].shape[0],
                    bucket_tokens,
                )
                self.assertEqual(
                    indexer.compressor.compute_kv_score.call_args.args[0].shape[0],
                    bucket_tokens,
                )

                compressor_args = compressor.call_args.kwargs
                self.assertEqual(
                    compressor_args["hidden_states"].shape[0], expected_rows
                )
                self.assertEqual(compressor_args["positions"].shape[0], expected_rows)
                self.assertEqual(
                    compressor_args["out_cache_loc"].shape[0], expected_rows
                )
                self.assertEqual(compressor_args["kv_score"].shape[0], expected_rows)

                indexer_args = indexer.call_args.kwargs
                self.assertEqual(indexer_args["hidden_states"].shape[0], expected_rows)
                self.assertEqual(indexer_args["qr"].shape[0], expected_rows)
                self.assertEqual(indexer_args["positions"].shape[0], expected_rows)
                self.assertEqual(indexer_args["out_cache_loc"].shape[0], expected_rows)
                self.assertEqual(
                    indexer_args["indexer_compressor_kv_score"].shape[0],
                    expected_rows,
                )
                prepared_sources = {
                    "packed_q_values": packed_q_values,
                    "packed_q_scales": packed_q_scales,
                    "packed_weights": packed_weights,
                }
                if ambient:
                    indexer._prepare_packed_inputs.assert_called_once_with(
                        hidden_states=hidden_states,
                        qr=qr,
                        positions=positions,
                        cos_sin_cache=attention.rotary_emb.cos_sin_cache,
                    )
                    for name, source in prepared_sources.items():
                        actual = indexer_args[name]
                        self.assertEqual(actual.shape[0], real_tokens)
                        self.assertEqual(actual.data_ptr(), source.data_ptr())
                        self.assertIs(actual._base, source)
                    self.assertTrue(
                        bool((indexer_args["packed_q_values"] == 0x11).all())
                    )
                    self.assertTrue(
                        bool((indexer_args["packed_q_scales"] == 0x01010101).all())
                    )
                    self.assertTrue(
                        bool(torch.isfinite(indexer_args["packed_weights"]).all())
                    )
                else:
                    indexer._prepare_packed_inputs.assert_not_called()
                    for name in prepared_sources:
                        self.assertIsNone(indexer_args[name])

                insert_args = attention._insert_swa_cache.call_args.kwargs
                self.assertEqual(insert_args["q"].shape[0], expected_rows)
                self.assertEqual(insert_args["kv"].shape[0], expected_rows)
                self.assertEqual(insert_args["positions"].shape[0], expected_rows)
                self.assertEqual(insert_args["slot_mapping"].shape[0], expected_rows)
                backend_args = backend.forward_deepseek_v4_prefill.call_args.kwargs
                self.assertEqual(backend_args["q"].shape[0], expected_rows)
                attention._project_attention_output.assert_called_once()
                output_args = attention._project_attention_output.call_args.args
                self.assertEqual(output_args[0].shape[0], expected_rows)
                # The real-token slice is local to the eager core. The output
                # projection sits after the break and therefore retains the
                # outer bucket-shaped positions tensor; during CUDA-graph
                # replay the raw attention output lands in a bucket-shaped
                # handoff buffer before this call as well.
                self.assertIs(output_args[1], positions)
                self.assertEqual(output_args[1].shape[0], bucket_tokens)
                self.assertIs(output_args[2], attention.rotary_emb.cos_sin_cache)
                if ambient:
                    leading_views = (
                        (insert_args["q"], q),
                        (insert_args["kv"], kv),
                        (indexer_args["qr"], qr),
                        (compressor_args["kv_score"], compressor_score),
                        (
                            indexer_args["indexer_compressor_kv_score"],
                            indexer_compressor_score,
                        ),
                        (indexer_args["packed_q_values"], packed_q_values),
                        (indexer_args["packed_q_scales"], packed_q_scales),
                        (indexer_args["packed_weights"], packed_weights),
                    )
                    for actual, source in leading_views:
                        self.assertEqual(actual.data_ptr(), source.data_ptr())
                        self.assertIs(actual._base, source)

    def test_indexer_forward_threads_prepared_triplet_to_custom_op(self):
        num_tokens = 2
        indexer = object.__new__(DeepseekV4Indexer)
        torch.nn.Module.__init__(indexer)
        indexer.compress_ratio = 4
        indexer.use_fp4_cache = True
        compressor = mock.Mock()
        compressor.norm = SimpleNamespace(
            weight=torch.ones(1),
            variance_epsilon=1.0e-6,
        )
        indexer.compressor = compressor
        expected_topk = torch.full((num_tokens, 2), 7, dtype=torch.int32)
        indexer._forward_sparse_indexer_custom_op = mock.Mock(
            return_value=expected_topk
        )

        state_cache = torch.empty(1, 1)
        indexer_cache = torch.empty(1, 1, dtype=torch.uint8)
        state_slots = torch.arange(num_tokens, dtype=torch.int64)
        state_table = torch.zeros(1, 1, dtype=torch.int32)
        compressed_slots = torch.arange(num_tokens, dtype=torch.int64)
        cache_metadata = SimpleNamespace(
            compressed_slot_mapping=mock.Mock(return_value=compressed_slots)
        )
        metadata = SimpleNamespace(
            cache=cache_metadata,
            token_to_req_indices=torch.zeros(num_tokens, dtype=torch.int32),
            is_valid_token=None,
            query_start_loc=torch.tensor([0, num_tokens], dtype=torch.int32),
            seq_lens=torch.tensor([num_tokens], dtype=torch.int32),
        )
        pool = SimpleNamespace(
            get_indexer_state_buffer=mock.Mock(return_value=state_cache),
            get_indexer_block_size=mock.Mock(return_value=4),
            get_indexer_kv_buffer_2d=mock.Mock(return_value=indexer_cache),
        )
        ctx = SimpleNamespace(
            token_to_kv_pool=pool,
            attn_backend=SimpleNamespace(
                forward_metadata=metadata,
                forward_prefill_metadata=metadata,
            ),
            forward_mode=ForwardMode.MIXED,
            dsa_pregraph_writes=False,
        )
        packed_q_values = torch.full((num_tokens, 2, 2), 1, dtype=torch.uint8)
        packed_q_scales = torch.full((num_tokens, 2), 2, dtype=torch.int32)
        packed_weights = torch.full((num_tokens, 2), 3.0)

        with mock.patch.object(
            deepseek_v4_model,
            "deepseek_v4_csa_indexer_cache_insert",
        ):
            actual = indexer(
                hidden_states=torch.zeros(num_tokens, 4),
                qr=torch.zeros(num_tokens, 4),
                positions=torch.arange(num_tokens, dtype=torch.int64),
                ctx=ctx,
                out_cache_loc=torch.arange(num_tokens, dtype=torch.int64),
                layer_index=0,
                cos_sin_cache=torch.empty(1, ROPE_DIM),
                compressor_slot_cache={
                    "indexer_state": (state_slots, state_table, 4, None)
                },
                packed_q_values=packed_q_values,
                packed_q_scales=packed_q_scales,
                packed_weights=packed_weights,
            )

        self.assertIs(actual, expected_topk)
        custom_args = indexer._forward_sparse_indexer_custom_op.call_args.kwargs
        self.assertIs(custom_args["packed_q_values"], packed_q_values)
        self.assertIs(custom_args["packed_q_scales"], packed_q_scales)
        self.assertIs(custom_args["packed_weights"], packed_weights)

    def test_compressed_slot_mapping_memo_is_shared_by_block_size(self):
        num_tokens = 2
        positions = torch.arange(num_tokens, dtype=torch.int64)
        state_slots = torch.arange(num_tokens, dtype=torch.int64)
        state_table = torch.zeros(1, 1, dtype=torch.int32)
        compressed_slots_64 = torch.tensor([11, 12], dtype=torch.int64)
        compressed_slots_32 = torch.tensor([21, 22], dtype=torch.int64)
        slots_by_block_size = {
            64: compressed_slots_64,
            32: compressed_slots_32,
        }

        def compressed_slot_mapping(*args, kv_cache_block_size, **kwargs):
            del args, kwargs
            return slots_by_block_size[kv_cache_block_size]

        cache_metadata = SimpleNamespace(
            compressed_slot_mapping=mock.Mock(side_effect=compressed_slot_mapping),
            compressor_state_block_tables={4: state_table},
            compressor_state_base_logical_pages={4: None},
        )
        metadata = SimpleNamespace(
            cache=cache_metadata,
            token_to_req_indices=torch.zeros(num_tokens, dtype=torch.int32),
            is_valid_token=None,
            query_start_loc=torch.tensor([0, num_tokens], dtype=torch.int32),
            seq_lens=torch.tensor([num_tokens], dtype=torch.int32),
        )
        compressed_block_size = mock.Mock(return_value=64)
        pool = SimpleNamespace(
            get_indexer_state_buffer=mock.Mock(return_value=torch.empty(1, 1)),
            get_indexer_block_size=mock.Mock(return_value=64),
            get_indexer_kv_buffer_2d=mock.Mock(
                return_value=torch.empty(1, 1, dtype=torch.uint8)
            ),
            get_compressed_block_size=compressed_block_size,
            get_compressed_kv_buffer_2d=mock.Mock(
                return_value=torch.empty(1, 1, dtype=torch.uint8)
            ),
        )
        ctx = SimpleNamespace(
            token_to_kv_pool=pool,
            attn_backend=SimpleNamespace(forward_metadata=metadata),
            forward_mode=ForwardMode.MIXED,
            dsa_pregraph_writes=False,
        )
        slot_cache = {
            "indexer_state": (state_slots, state_table, 4, None),
        }

        indexer = object.__new__(DeepseekV4Indexer)
        torch.nn.Module.__init__(indexer)
        indexer.compress_ratio = 4
        indexer.use_fp4_cache = True
        indexer.compressor = mock.Mock()
        indexer.compressor.norm = SimpleNamespace(
            weight=torch.ones(1),
            variance_epsilon=1.0e-6,
        )
        indexer._forward_sparse_indexer_custom_op = mock.Mock(
            return_value=torch.zeros(num_tokens, 1, dtype=torch.int32)
        )
        packed_q_values = torch.zeros(num_tokens, 1, 1, dtype=torch.uint8)
        packed_q_scales = torch.zeros(num_tokens, 1, dtype=torch.int32)
        packed_weights = torch.zeros(num_tokens, 1)

        compressor = SimpleNamespace(
            compress_ratio=4,
            coff=1,
            head_dim=2,
            ape=torch.empty((4, 2), dtype=torch.float32),
            norm=SimpleNamespace(
                weight=torch.ones(2),
                variance_epsilon=1.0e-6,
            ),
        )
        kv_score = torch.empty((num_tokens, 4), dtype=torch.float32)

        with (
            mock.patch.object(
                deepseek_v4_model,
                "deepseek_v4_csa_indexer_cache_insert",
            ) as indexer_insert,
            mock.patch.object(
                deepseek_v4_model,
                "save_deepseek_v4_compressor_state",
            ),
            mock.patch.object(
                deepseek_v4_model,
                "deepseek_v4_csa_compress_kv_cache_insert",
            ) as compressor_insert,
        ):
            indexer(
                hidden_states=torch.zeros(num_tokens, 4),
                qr=torch.zeros(num_tokens, 4),
                positions=positions,
                ctx=ctx,
                out_cache_loc=state_slots,
                layer_index=0,
                cos_sin_cache=torch.empty(1, ROPE_DIM),
                compressor_slot_cache=slot_cache,
                packed_q_values=packed_q_values,
                packed_q_scales=packed_q_scales,
                packed_weights=packed_weights,
            )
            DeepseekV4Compressor.forward(
                compressor,
                hidden_states=torch.empty(num_tokens, 1),
                positions=positions,
                ctx=ctx,
                out_cache_loc=state_slots,
                layer_index=0,
                cos_sin_cache=torch.empty(1, ROPE_DIM),
                state_cache=torch.empty(1, 1),
                state_block_size=4,
                state_slot_mapping=state_slots,
                compressor_slot_cache=slot_cache,
                kv_score=kv_score,
            )

            key_64 = ("compressed_slot_mapping", 4, 64)
            self.assertIs(slot_cache[key_64], compressed_slots_64)
            self.assertIs(
                indexer_insert.call_args.kwargs["kv_slot_mapping"],
                compressed_slots_64,
            )
            self.assertIs(
                compressor_insert.call_args.kwargs["kv_slot_mapping"],
                compressed_slots_64,
            )
            cache_metadata.compressed_slot_mapping.assert_called_once()

            compressed_block_size.return_value = 32
            DeepseekV4Compressor.forward(
                compressor,
                hidden_states=torch.empty(num_tokens, 1),
                positions=positions,
                ctx=ctx,
                out_cache_loc=state_slots,
                layer_index=0,
                cos_sin_cache=torch.empty(1, ROPE_DIM),
                state_cache=torch.empty(1, 1),
                state_block_size=4,
                state_slot_mapping=state_slots,
                compressor_slot_cache=slot_cache,
                kv_score=kv_score,
            )

        key_32 = ("compressed_slot_mapping", 4, 32)
        self.assertIs(slot_cache[key_32], compressed_slots_32)
        self.assertEqual(cache_metadata.compressed_slot_mapping.call_count, 2)
        self.assertIs(
            compressor_insert.call_args.kwargs["kv_slot_mapping"],
            compressed_slots_32,
        )

    def test_pregraph_buffers_allocate_disarmed_and_reuse(self):
        buffers = _DeepseekV4PregraphBuffers()
        self.assertFalse(buffers.allocated)
        buffers.allocate(4, 2, 3, 2, torch.device("cpu"))
        self.assertTrue(buffers.allocated)
        self.assertEqual(buffers.token_to_req.tolist(), [-1] * 4)
        self.assertEqual(buffers.indexer.state_slot_mapping.tolist(), [-1] * 4)
        self.assertEqual(buffers.indexer.kv_slot_mapping.tolist(), [-1] * 4)
        self.assertEqual(buffers.indexer.state_block_table.tolist(), [[-1] * 3] * 2)
        self.assertEqual(buffers.indexer.state_base_pages.tolist(), [0, 0])
        self.assertEqual(buffers.c4.state_slot_mapping.tolist(), [-1] * 4)
        self.assertEqual(buffers.c4.kv_slot_mapping.tolist(), [-1] * 4)
        self.assertEqual(buffers.c4.state_block_table.tolist(), [[-1] * 2] * 2)
        self.assertEqual(buffers.c4.state_base_pages.tolist(), [0, 0])

        # A smaller re-allocation keeps the captured storage and re-disarms it.
        buffers.token_to_req.fill_(5)
        buffers.c4.state_slot_mapping.fill_(5)
        table = buffers.indexer.state_block_table
        buffers.allocate(2, 1, 3, 2, torch.device("cpu"))
        self.assertIs(buffers.indexer.state_block_table, table)
        self.assertEqual(buffers.token_to_req.tolist(), [-1] * 4)
        self.assertEqual(buffers.c4.state_slot_mapping.tolist(), [-1] * 4)

    def test_model_pregraph_hook_arms_pure_extend_and_disarms_unsafe(self):
        model = object.__new__(deepseek_v4_model.DeepseekV4Model)
        torch.nn.Module.__init__(model)
        model.pregraph_buffers = _DeepseekV4PregraphBuffers()
        model._pregraph_indexer_cache_layer_index = 0
        model._pregraph_indexer_compress_ratio = 4

        num_tokens = 4
        positions = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
        token_to_req = torch.tensor([0, 0, 1, 1], dtype=torch.int32)
        state_table = torch.tensor([[3, 4], [5, 6]], dtype=torch.int32)
        c4_state_table = torch.tensor([[7], [9]], dtype=torch.int32)
        compressed_slots = torch.tensor([9, -1, 13, 15], dtype=torch.int64)
        cache_metadata = SimpleNamespace(
            indexer_state_block_table=state_table,
            indexer_state_base_logical_page=None,
            compressor_state_block_tables={4: c4_state_table},
            compressor_state_base_logical_pages={4: None},
            compressed_slot_mapping=mock.Mock(return_value=compressed_slots),
        )
        metadata = SimpleNamespace(
            cache=cache_metadata,
            token_to_req_indices=token_to_req,
            is_valid_token=None,
            num_prefill_tokens=num_tokens,
            decode_token_count=mock.Mock(return_value=0),
            query_start_loc=torch.tensor([0, 2, 4], dtype=torch.int32),
            seq_lens=torch.tensor([2, 4], dtype=torch.int32),
        )
        pool = SimpleNamespace(
            get_indexer_state_block_size=mock.Mock(return_value=2),
            get_indexer_block_size=mock.Mock(return_value=64),
            get_compressor_state_block_size=mock.Mock(return_value=4),
            get_compressed_block_size=mock.Mock(return_value=64),
        )
        ctx = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            token_to_kv_pool=pool,
            req_to_page=torch.zeros(8, 2, dtype=torch.int32),
            attn_backend=SimpleNamespace(
                forward_metadata=metadata,
                forward_prefill_metadata=metadata,
            ),
            dsa_compressor_slot_cache=None,
        )

        # The capture call sizes and disarms the buffers from dummy metadata.
        model.prepare_prefill_graph_replay(
            ctx, torch.zeros(6, dtype=torch.int64), capture=True
        )
        buffers = model.pregraph_buffers
        self.assertTrue(buffers.allocated)
        self.assertFalse(ctx.dsa_pregraph_writes)
        self.assertEqual(buffers.token_to_req.shape[0], 6)
        self.assertEqual(list(buffers.indexer.state_block_table.shape), [6, 2])
        self.assertEqual(list(buffers.c4.state_block_table.shape), [6, 1])

        # A safe pure-extend replay stages live metadata and arms the flag.
        model.prepare_prefill_graph_replay(ctx, positions, capture=False)
        self.assertTrue(ctx.dsa_pregraph_writes)
        self.assertEqual(
            buffers.token_to_req[:num_tokens].tolist(), token_to_req.tolist()
        )
        self.assertEqual(buffers.token_to_req[num_tokens:].tolist(), [-1, -1])
        expected_state = _group_slot_mapping_from_raw(
            positions, token_to_req, state_table, 2
        )
        self.assertEqual(
            buffers.indexer.state_slot_mapping[:num_tokens].tolist(),
            expected_state.tolist(),
        )
        self.assertEqual(
            buffers.indexer.state_slot_mapping[num_tokens:].tolist(), [-1, -1]
        )
        self.assertEqual(
            buffers.indexer.kv_slot_mapping[:num_tokens].tolist(),
            compressed_slots.tolist(),
        )
        self.assertEqual(
            buffers.indexer.state_block_table[:2].tolist(), state_table.tolist()
        )
        self.assertEqual(buffers.indexer.state_block_table[2:].tolist(), [[-1, -1]] * 4)
        expected_c4_state = _group_slot_mapping_from_raw(
            positions, token_to_req, c4_state_table, 4
        )
        self.assertEqual(
            buffers.c4.state_slot_mapping[:num_tokens].tolist(),
            expected_c4_state.tolist(),
        )
        self.assertEqual(buffers.c4.state_slot_mapping[num_tokens:].tolist(), [-1, -1])
        # Equal compressed block sizes share one mapping computation.
        self.assertEqual(
            buffers.c4.kv_slot_mapping[:num_tokens].tolist(),
            compressed_slots.tolist(),
        )
        self.assertEqual(
            buffers.c4.state_block_table[:2].tolist(), c4_state_table.tolist()
        )
        self.assertEqual(buffers.c4.state_block_table[2:].tolist(), [[-1]] * 4)
        cache_metadata.compressed_slot_mapping.assert_called_once()
        mapping_kwargs = cache_metadata.compressed_slot_mapping.call_args.kwargs
        self.assertEqual(mapping_kwargs["kv_cache_block_size"], 64)
        self.assertFalse(mapping_kwargs["use_decode_cache"])
        model.finish_prefill_graph_replay(ctx)
        self.assertFalse(ctx.dsa_pregraph_writes)

        # Mixed forwards disarm (isolates the is_mixed() guard: MIXED is also
        # not a pure extend, so this alone would not distinguish the two, but
        # the DECODE case below covers the not-is_extend() guard).
        ctx.forward_mode = ForwardMode.MIXED
        model.prepare_prefill_graph_replay(ctx, positions, capture=False)
        self.assertFalse(ctx.dsa_pregraph_writes)
        self.assertEqual(buffers.indexer.state_slot_mapping.tolist(), [-1] * 6)

        # Pure decode disarms (isolates the not-is_extend() guard).
        ctx.forward_mode = ForwardMode.DECODE
        buffers.indexer.state_slot_mapping[:num_tokens].fill_(0)
        model.prepare_prefill_graph_replay(ctx, positions, capture=False)
        self.assertFalse(ctx.dsa_pregraph_writes)
        self.assertEqual(buffers.indexer.state_slot_mapping.tolist(), [-1] * 6)
        ctx.forward_mode = ForwardMode.EXTEND

        # Base pages shorter than the live table's request count disarm (the
        # kernel would otherwise index out of the base-page tensor).
        cache_metadata.indexer_state_base_logical_page = torch.zeros(
            1, dtype=torch.int32
        )
        buffers.indexer.state_slot_mapping[:num_tokens].fill_(0)
        model.prepare_prefill_graph_replay(ctx, positions, capture=False)
        self.assertFalse(ctx.dsa_pregraph_writes)
        self.assertEqual(buffers.indexer.state_slot_mapping.tolist(), [-1] * 6)
        cache_metadata.indexer_state_base_logical_page = None

        # A live table wider than the captured buffers disarms too.
        ctx.forward_mode = ForwardMode.EXTEND
        cache_metadata.indexer_state_block_table = torch.zeros(2, 3, dtype=torch.int32)
        buffers.indexer.state_slot_mapping[:num_tokens].fill_(0)
        model.prepare_prefill_graph_replay(ctx, positions, capture=False)
        self.assertFalse(ctx.dsa_pregraph_writes)
        self.assertEqual(buffers.indexer.state_slot_mapping.tolist(), [-1] * 6)

        # A missing compressor-state table disarms (arm-both-or-none).
        cache_metadata.indexer_state_block_table = state_table
        cache_metadata.compressor_state_block_tables = {}
        buffers.c4.state_slot_mapping[:num_tokens].fill_(0)
        model.prepare_prefill_graph_replay(ctx, positions, capture=False)
        self.assertFalse(ctx.dsa_pregraph_writes)
        self.assertEqual(buffers.c4.state_slot_mapping.tolist(), [-1] * 6)

        # Unequal compressed block sizes compute and stage a second C4
        # mapping.
        cache_metadata.compressor_state_block_tables = {4: c4_state_table}
        c4_compressed_slots = torch.tensor([21, 22, -1, 25], dtype=torch.int64)
        slots_by_block_size = {64: compressed_slots, 32: c4_compressed_slots}

        def compressed_by_size(*args, kv_cache_block_size, **kwargs):
            del args, kwargs
            return slots_by_block_size[kv_cache_block_size]

        cache_metadata.compressed_slot_mapping = mock.Mock(
            side_effect=compressed_by_size
        )
        pool.get_compressed_block_size.return_value = 32
        model.prepare_prefill_graph_replay(ctx, positions, capture=False)
        self.assertTrue(ctx.dsa_pregraph_writes)
        self.assertEqual(cache_metadata.compressed_slot_mapping.call_count, 2)
        self.assertEqual(
            buffers.indexer.kv_slot_mapping[:num_tokens].tolist(),
            compressed_slots.tolist(),
        )
        self.assertEqual(
            buffers.c4.kv_slot_mapping[:num_tokens].tolist(),
            c4_compressed_slots.tolist(),
        )
        model.finish_prefill_graph_replay(ctx)

    def test_insert_cache_pregraph_binds_staged_buffers_to_kernels(self):
        num_tokens = 2
        buffers = _DeepseekV4PregraphBuffers()
        buffers.allocate(num_tokens, 1, 1, 1, torch.device("cpu"))
        pool = SimpleNamespace(
            get_indexer_state_buffer=mock.Mock(return_value=torch.empty(1, 1)),
            get_indexer_state_block_size=mock.Mock(return_value=2),
            get_indexer_block_size=mock.Mock(return_value=64),
            get_indexer_kv_buffer_2d=mock.Mock(
                return_value=torch.empty(1, 1, dtype=torch.uint8)
            ),
            get_compressor_state_buffer=mock.Mock(return_value=torch.empty(1, 1)),
            get_compressor_state_block_size=mock.Mock(return_value=4),
            get_compressed_kv_buffer_2d=mock.Mock(
                return_value=torch.empty(1, 1, dtype=torch.uint8)
            ),
            get_compressed_block_size=mock.Mock(return_value=64),
        )
        ctx = SimpleNamespace(token_to_kv_pool=pool)
        positions = torch.arange(num_tokens, dtype=torch.int64)

        indexer = object.__new__(DeepseekV4Indexer)
        torch.nn.Module.__init__(indexer)
        indexer.compress_ratio = 4
        indexer.use_fp4_cache = True
        indexer.compressor = SimpleNamespace(
            coff=1,
            head_dim=2,
            ape=torch.empty(4, 2),
            norm=SimpleNamespace(weight=torch.ones(2), variance_epsilon=1.0e-6),
        )
        with (
            mock.patch.object(
                deepseek_v4_model, "save_deepseek_v4_compressor_state"
            ) as save_state,
            mock.patch.object(
                deepseek_v4_model, "deepseek_v4_csa_indexer_cache_insert"
            ) as indexer_insert,
        ):
            indexer.insert_cache_pregraph(
                positions=positions,
                ctx=ctx,
                layer_index=0,
                cos_sin_cache=torch.empty(1, ROPE_DIM),
                kv_score=torch.zeros(num_tokens, 4),
                buffers=buffers,
            )
        self.assertIs(
            save_state.call_args.kwargs["slot_mapping"]._base,
            buffers.indexer.state_slot_mapping,
        )
        kwargs = indexer_insert.call_args.kwargs
        self.assertIs(
            kwargs["compressor_slot_mapping"]._base,
            buffers.indexer.state_slot_mapping,
        )
        self.assertIs(kwargs["kv_slot_mapping"]._base, buffers.indexer.kv_slot_mapping)
        self.assertIs(kwargs["block_table"], buffers.indexer.state_block_table)
        self.assertIs(
            kwargs["block_table_base_offsets"], buffers.indexer.state_base_pages
        )
        self.assertIs(kwargs["token_to_req_indices"]._base, buffers.token_to_req)
        self.assertTrue(kwargs["use_fp4_cache"])

        compressor = object.__new__(DeepseekV4Compressor)
        torch.nn.Module.__init__(compressor)
        compressor.compress_ratio = 4
        compressor.coff = 1
        compressor.head_dim = 2
        compressor.ape = torch.empty(4, 2)
        compressor.norm = SimpleNamespace(weight=torch.ones(2), variance_epsilon=1.0e-6)
        with (
            mock.patch.object(
                deepseek_v4_model, "save_deepseek_v4_compressor_state"
            ) as save_state,
            mock.patch.object(
                deepseek_v4_model, "deepseek_v4_csa_compress_kv_cache_insert"
            ) as compressor_insert,
        ):
            compressor.insert_cache_pregraph(
                positions=positions,
                ctx=ctx,
                layer_index=0,
                cos_sin_cache=torch.empty(1, ROPE_DIM),
                kv_score=torch.zeros(num_tokens, 4),
                buffers=buffers,
            )
        self.assertIs(
            save_state.call_args.kwargs["slot_mapping"]._base,
            buffers.c4.state_slot_mapping,
        )
        kwargs = compressor_insert.call_args.kwargs
        self.assertIs(
            kwargs["compressor_slot_mapping"]._base,
            buffers.c4.state_slot_mapping,
        )
        self.assertIs(kwargs["kv_slot_mapping"]._base, buffers.c4.kv_slot_mapping)
        self.assertIs(kwargs["block_table"], buffers.c4.state_block_table)
        self.assertIs(kwargs["block_table_base_offsets"], buffers.c4.state_base_pages)
        self.assertIs(kwargs["token_to_req_indices"]._base, buffers.token_to_req)

    def test_indexer_forward_skips_cache_writes_when_pregraph_armed(self):
        num_tokens = 2
        indexer = object.__new__(DeepseekV4Indexer)
        torch.nn.Module.__init__(indexer)
        indexer.compress_ratio = 4
        indexer.use_fp4_cache = True
        indexer.compressor = mock.Mock()
        expected_topk = torch.full((num_tokens, 2), 7, dtype=torch.int32)
        indexer._forward_sparse_indexer_custom_op = mock.Mock(
            return_value=expected_topk
        )
        packed_q_values = torch.zeros(num_tokens, 1, 1, dtype=torch.uint8)
        packed_q_scales = torch.zeros(num_tokens, 1, dtype=torch.int32)
        packed_weights = torch.zeros(num_tokens, 1)

        cache_metadata = SimpleNamespace(compressed_slot_mapping=mock.Mock())
        metadata = SimpleNamespace(
            cache=cache_metadata,
            token_to_req_indices=torch.zeros(num_tokens, dtype=torch.int32),
            is_valid_token=None,
            query_start_loc=torch.tensor([0, num_tokens], dtype=torch.int32),
            seq_lens=torch.tensor([num_tokens], dtype=torch.int32),
        )
        pool = SimpleNamespace(
            get_indexer_state_buffer=mock.Mock(return_value=torch.empty(1, 1)),
            get_indexer_block_size=mock.Mock(return_value=64),
            get_indexer_kv_buffer_2d=mock.Mock(
                return_value=torch.empty(1, 1, dtype=torch.uint8)
            ),
        )
        ctx = SimpleNamespace(
            token_to_kv_pool=pool,
            attn_backend=SimpleNamespace(
                forward_metadata=metadata,
                forward_prefill_metadata=metadata,
            ),
            forward_mode=ForwardMode.EXTEND,
            dsa_pregraph_writes=True,
        )

        with (
            mock.patch.object(
                deepseek_v4_model,
                "deepseek_v4_csa_indexer_cache_insert",
            ) as indexer_insert,
            mock.patch.object(
                deepseek_v4_model,
                "save_deepseek_v4_compressor_state",
            ) as save_state,
        ):
            actual = indexer(
                hidden_states=torch.zeros(num_tokens, 4),
                qr=torch.zeros(num_tokens, 4),
                positions=torch.arange(num_tokens, dtype=torch.int64),
                ctx=ctx,
                out_cache_loc=torch.arange(num_tokens, dtype=torch.int64),
                layer_index=0,
                cos_sin_cache=torch.empty(1, ROPE_DIM),
                compressor_slot_cache={},
                packed_q_values=packed_q_values,
                packed_q_scales=packed_q_scales,
                packed_weights=packed_weights,
            )

        self.assertIs(actual, expected_topk)
        indexer.compressor.assert_not_called()
        indexer_insert.assert_not_called()
        save_state.assert_not_called()
        cache_metadata.compressed_slot_mapping.assert_not_called()
        custom_kwargs = indexer._forward_sparse_indexer_custom_op.call_args.kwargs
        self.assertEqual(custom_kwargs["indexer_block_size"], 64)
        # The skip path must thread the caller's already-packed triplet through
        # unchanged (not drop/recompute it inside the sparse indexer op).
        self.assertIs(custom_kwargs["packed_q_values"], packed_q_values)
        self.assertIs(custom_kwargs["packed_q_scales"], packed_q_scales)
        self.assertIs(custom_kwargs["packed_weights"], packed_weights)

    def test_attention_forward_pregraph_insert_gate(self):
        # The gate is load-bearing: the staging buffers stay armed with the
        # last replay's mappings after finish clears the ctx flag, so the
        # captured-write kernels must fire ONLY under an ambient breakable
        # capture/replay with allocated buffers -- never on eager forwards.
        num_tokens = 4
        positions = torch.arange(num_tokens, dtype=torch.int64)

        def build_attention(pregraph_buffers, use_fp4_cache):
            attention = object.__new__(DeepseekV4Attention)
            torch.nn.Module.__init__(attention)
            attention.attention_kind = "csa"
            attention.rotary_emb = SimpleNamespace(
                cos_sin_cache=torch.zeros(1, ROPE_DIM, dtype=torch.float32)
            )
            attention.stream_fork = StreamFork(None)
            attention.cache_layer_index = 0
            attention.compress_ratio = 4
            attention.pregraph_buffers = pregraph_buffers
            attention._project_q_kv = mock.Mock(
                return_value=(
                    torch.zeros(num_tokens, 1, 2),
                    torch.zeros(num_tokens, 2),
                    torch.zeros(num_tokens, 2),
                )
            )
            compressor = mock.Mock()
            compressor_score = torch.zeros(num_tokens, 2)
            compressor.compute_kv_score.return_value = compressor_score
            attention.compressor = compressor
            indexer = mock.Mock()
            indexer.use_fp4_cache = use_fp4_cache
            indexer_score = torch.zeros(num_tokens, 3)
            indexer.compressor.compute_kv_score.return_value = indexer_score
            indexer._prepare_packed_inputs.return_value = (None, None, None)
            attention.indexer = indexer
            attention._forward_attention_core = mock.Mock(
                return_value=torch.zeros(num_tokens, 2)
            )
            attention._project_attention_output = mock.Mock(
                side_effect=lambda output, *_: output
            )
            return attention, indexer, indexer_score, compressor, compressor_score

        allocated = _DeepseekV4PregraphBuffers()
        allocated.allocate(num_tokens, 1, 1, 1, torch.device("cpu"))
        unallocated = _DeepseekV4PregraphBuffers()
        cases = (
            # (name, ambient, buffers, use_fp4, expect_indexer, expect_c4)
            ("armed", True, allocated, True, True, True),
            ("eager", False, allocated, True, False, False),
            ("unallocated", True, unallocated, True, False, False),
            # Gate structure check: with fp4 off only the indexer part is
            # suppressed. In production fp8 configs the buffers are never
            # allocated (the model hook is inert without an fp4 indexer
            # layer), so allocated+fp8 is defense-in-depth, not a reachable
            # serving state.
            ("fp8_cache", True, allocated, False, False, True),
        )
        for name, ambient, buffers, use_fp4, expect_indexer, expect_c4 in cases:
            with self.subTest(case=name):
                attention, indexer, indexer_score, compressor, compressor_score = (
                    build_attention(buffers, use_fp4)
                )
                ctx = SimpleNamespace()
                with mock.patch(
                    "tokenspeed.runtime.models.deepseek_v4.current_forward_ctx",
                    return_value=ctx if ambient else None,
                ):
                    attention(
                        positions,
                        torch.zeros(num_tokens, 8),
                        ctx,
                        torch.arange(num_tokens, dtype=torch.int64),
                    )
                if expect_indexer:
                    indexer.insert_cache_pregraph.assert_called_once()
                    kwargs = indexer.insert_cache_pregraph.call_args.kwargs
                    self.assertIs(kwargs["buffers"], buffers)
                    self.assertIs(kwargs["kv_score"], indexer_score)
                    self.assertIs(kwargs["positions"], positions)
                    self.assertEqual(kwargs["layer_index"], 0)
                else:
                    indexer.insert_cache_pregraph.assert_not_called()
                if expect_c4:
                    compressor.insert_cache_pregraph.assert_called_once()
                    kwargs = compressor.insert_cache_pregraph.call_args.kwargs
                    self.assertIs(kwargs["buffers"], buffers)
                    self.assertIs(kwargs["kv_score"], compressor_score)
                    self.assertIs(kwargs["positions"], positions)
                    self.assertEqual(kwargs["layer_index"], 0)
                else:
                    compressor.insert_cache_pregraph.assert_not_called()

    def test_attention_core_skips_csa_compressor_when_pregraph_armed(self):
        num_tokens = 2
        cases = (
            # (kind, ratio, armed, expect_compressor_called)
            ("csa", 4, True, False),
            ("csa", 4, False, True),
            # HCA (C128) layers keep the eager compressor even when armed.
            ("hca", 128, True, True),
        )
        for kind, ratio, armed, expect_called in cases:
            with self.subTest(kind=kind, armed=armed):
                attention = object.__new__(DeepseekV4Attention)
                torch.nn.Module.__init__(attention)
                attention.attention_kind = kind
                attention.stream_fork = StreamFork(None)
                attention.cache_layer_index = 0
                attention.compress_ratio = ratio
                attention.padded_heads = 2
                attention.num_local_heads = 2
                attention.head_dim = HEAD_DIM
                attention.swa_window = 64
                attention.scale = 1.0
                attention.attn_sink = torch.zeros(1)
                attention.compressor = mock.Mock()
                attention.indexer = None
                attention._insert_swa_cache = mock.Mock()

                metadata = SimpleNamespace(
                    token_to_req_indices=torch.arange(num_tokens, dtype=torch.int32),
                    is_valid_token=None,
                )
                backend = SimpleNamespace(
                    forward_metadata=metadata,
                    forward_deepseek_v4_prefill=mock.Mock(
                        side_effect=lambda **kw: kw["q"]
                    ),
                )
                pool = SimpleNamespace(
                    swa_block_size=4,
                    get_swa_kv_buffer=mock.Mock(
                        return_value=torch.empty(1, 1, dtype=torch.uint8)
                    ),
                )
                ctx = SimpleNamespace(
                    dsa_compressor_slot_cache=None,
                    dsa_swa_slot_mapping=None,
                    token_to_kv_pool=pool,
                    attn_backend=backend,
                    forward_mode=ForwardMode.EXTEND,
                    dsa_pregraph_writes=armed,
                )
                with mock.patch(
                    "tokenspeed.runtime.models.deepseek_v4.current_forward_ctx",
                    return_value=None,
                ):
                    attention._forward_attention_core(
                        torch.arange(num_tokens, dtype=torch.int64),
                        torch.zeros(num_tokens, 8),
                        torch.zeros(num_tokens, 2, HEAD_DIM),
                        torch.zeros(num_tokens, HEAD_DIM),
                        torch.zeros(num_tokens, 4),
                        torch.zeros(1, ROPE_DIM),
                        ctx,
                        torch.arange(num_tokens, dtype=torch.int64),
                        None,
                        None,
                        swa_slot_mapping=torch.arange(num_tokens, dtype=torch.int64),
                        compressor_slot_cache={},
                    )
                if expect_called:
                    attention.compressor.assert_called_once()
                else:
                    attention.compressor.assert_not_called()
                attention._insert_swa_cache.assert_called_once()

    def test_attention_core_returns_raw_output_without_projection(self):
        attention = object.__new__(DeepseekV4Attention)
        torch.nn.Module.__init__(attention)
        attention.attention_kind = "swa"
        attention.stream_fork = StreamFork(None)
        attention.compressor = None
        attention.indexer = None
        attention.padded_heads = 2
        attention.cache_layer_index = 0
        attention.compress_ratio = 1
        attention.num_local_heads = 2
        attention.head_dim = HEAD_DIM
        attention.swa_window = 64
        attention.scale = 1.0
        attention.attn_sink = torch.zeros(1)
        attention._insert_swa_cache = mock.Mock()
        attention._project_attention_output = mock.Mock()

        positions = torch.arange(2, dtype=torch.int64)
        hidden_states = torch.randn(2, 8)
        q = torch.randn(2, 2, HEAD_DIM, dtype=torch.bfloat16)
        kv = torch.randn(2, HEAD_DIM, dtype=torch.bfloat16)
        qr = torch.randn(2, 32, dtype=torch.bfloat16)
        cos_sin_cache = torch.zeros(1, ROPE_DIM, dtype=torch.float32)
        raw_output = torch.randn_like(q)
        backend = SimpleNamespace(
            forward_metadata=SimpleNamespace(token_to_req_indices=None),
            forward_deepseek_v4_prefill=mock.Mock(return_value=raw_output),
        )
        pool = SimpleNamespace(
            swa_block_size=4,
            get_swa_kv_buffer=mock.Mock(
                return_value=torch.empty(1, 4 * 584, dtype=torch.uint8)
            ),
        )
        ctx = SimpleNamespace(
            dsa_compressor_slot_cache={},
            dsa_swa_slot_mapping=None,
            token_to_kv_pool=pool,
            attn_backend=backend,
            forward_mode=ForwardMode.EXTEND,
        )

        output = attention._forward_attention_core(
            positions,
            hidden_states,
            q,
            kv,
            qr,
            cos_sin_cache,
            ctx,
            torch.arange(2, dtype=torch.int64),
            None,
            None,
            torch.arange(2, dtype=torch.int64),
            {},
        )

        self.assertIs(output, raw_output)
        attention._project_attention_output.assert_not_called()

    def test_attention_forward_projects_core_output_once(self):
        attention = object.__new__(DeepseekV4Attention)
        torch.nn.Module.__init__(attention)
        attention.attention_kind = "swa"
        attention.stream_fork = StreamFork(None)
        attention.compressor = None
        attention.indexer = None
        attention.rotary_emb = SimpleNamespace(
            cos_sin_cache=torch.zeros(1, ROPE_DIM, dtype=torch.float64)
        )

        positions = torch.arange(3, dtype=torch.int64)
        hidden_states = torch.randn(3, 8)
        q = torch.randn(3, 2, HEAD_DIM, dtype=torch.bfloat16)
        kv = torch.randn(3, HEAD_DIM, dtype=torch.bfloat16)
        qr = torch.randn(3, 32, dtype=torch.bfloat16)
        raw_output = torch.randn_like(q)
        projected_output = torch.randn(3, 8)
        attention._project_q_kv = mock.Mock(return_value=(q, kv, qr))
        attention._forward_attention_core = mock.Mock(return_value=raw_output)
        attention._project_attention_output = mock.Mock(return_value=projected_output)

        output = attention(
            positions,
            hidden_states,
            object(),
            torch.arange(3, dtype=torch.int64),
        )

        self.assertIs(output, projected_output)
        attention._forward_attention_core.assert_called_once()
        attention._project_attention_output.assert_called_once()
        core_args = attention._forward_attention_core.call_args.args
        project_args = attention._project_attention_output.call_args.args
        self.assertIs(core_args[0], positions)
        self.assertIs(project_args[0], raw_output)
        self.assertIs(project_args[1], positions)
        self.assertIs(core_args[5], project_args[2])
        self.assertEqual(project_args[2].dtype, torch.float32)

    def test_attention_empty_input_skips_phase1(self):
        attention = object.__new__(DeepseekV4Attention)
        torch.nn.Module.__init__(attention)
        attention._project_q_kv = mock.Mock()
        attention._forward_attention_core = mock.Mock()
        attention._project_attention_output = mock.Mock()
        hidden_states = torch.empty(0, 8)

        output = attention(
            torch.empty(0, dtype=torch.int64),
            hidden_states,
            None,
            torch.empty(0, dtype=torch.int64),
        )

        self.assertIs(output, hidden_states)
        attention._project_q_kv.assert_not_called()
        attention._forward_attention_core.assert_not_called()
        attention._project_attention_output.assert_not_called()

    def test_swa_slot_mapping_guard_masks_out_of_range_slots(self):
        slots = torch.tensor([-3, -1, 0, 7, 8, 99], dtype=torch.int64)

        sanitized = _deepseek_v4_sanitize_swa_slot_mapping(slots, capacity=2 * 4)

        torch.testing.assert_close(
            sanitized,
            torch.tensor([-1, -1, 0, 7, -1, -1], dtype=torch.int64),
        )

    def test_swa_slot_mapping_guard_masks_invalid_graph_tokens(self):
        slots = torch.tensor([0, 1, 2, 9, -1, 3], dtype=torch.int64)
        is_valid_token = torch.tensor([True, False, True, True, True, False])

        sanitized = _deepseek_v4_sanitize_swa_slot_mapping(
            slots,
            capacity=2 * 4,
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

    def test_indexer_custom_op_routes_complete_triplet_or_legacy(self):
        device = torch.device("cuda")
        num_tokens = 3
        indexer = object.__new__(DeepseekV4Indexer)
        torch.nn.Module.__init__(indexer)
        indexer.use_fp4_cache = True
        indexer.compress_ratio = 4
        indexer.topk_tokens = 2
        indexer.topk_buffer = None
        indexer._persistent_topk_workspace = torch.empty(
            0, device=device, dtype=torch.uint8
        )
        empty_workspace = torch.empty((0, 0), device=device, dtype=torch.uint8)
        indexer._prefill_gather_workspace = mock.Mock(
            return_value=(empty_workspace, empty_workspace)
        )

        def triplet(seed):
            return (
                torch.full((num_tokens, 2, 2), seed, device=device, dtype=torch.uint8),
                torch.full((num_tokens, 2), seed + 1, device=device, dtype=torch.int32),
                torch.full((num_tokens, 2), float(seed + 2), device=device),
            )

        legacy = triplet(1)
        supplied = triplet(11)
        indexer._prepare_packed_inputs = mock.Mock(return_value=legacy)
        block_table = torch.zeros((1, 1), device=device, dtype=torch.int32)
        metadata = SimpleNamespace(
            cache=SimpleNamespace(
                compressed_block_table=mock.Mock(return_value=block_table)
            ),
            token_to_req_indices=torch.zeros(
                num_tokens, device=device, dtype=torch.int32
            ),
            seq_lens_cpu=None,
            query_lens_cpu=None,
        )
        prefill_metadata = SimpleNamespace(max_gather_rows=mock.Mock(return_value=0))
        ctx = SimpleNamespace(forward_mode=ForwardMode.EXTEND)
        hidden_states = torch.zeros(num_tokens, 4, device=device)
        qr = torch.zeros(num_tokens, 4, device=device)
        positions = torch.arange(num_tokens, device=device, dtype=torch.int64)
        cos_sin_cache = torch.empty(1, ROPE_DIM, device=device)
        sparse_indexer = mock.Mock(
            side_effect=lambda **kwargs: kwargs["topk_indices_buffer"]
        )

        def run(prepared=(None, None, None)):
            return indexer._forward_sparse_indexer_custom_op(
                hidden_states=hidden_states,
                qr=qr,
                positions=positions,
                metadata=metadata,
                ctx=ctx,
                indexer_cache=torch.empty(1, 1, device=device, dtype=torch.uint8),
                indexer_block_size=1,
                cos_sin_cache=cos_sin_cache,
                packed_q_values=prepared[0],
                packed_q_scales=prepared[1],
                packed_weights=prepared[2],
            )

        with (
            mock.patch.object(
                deepseek_v4_model,
                "_deepseek_v4_deepgemm_fp4_indexer_available",
                return_value=True,
            ),
            mock.patch.object(
                deepseek_v4_model,
                "_deepseek_v4_indexer_prefill_metadata",
                return_value=prefill_metadata,
            ),
            mock.patch.object(
                deepseek_v4_model,
                "_deepseek_v4_sparse_attn_indexer",
                sparse_indexer,
            ),
        ):
            run()
            indexer._prepare_packed_inputs.assert_called_once_with(
                hidden_states=hidden_states,
                qr=qr,
                positions=positions,
                cos_sin_cache=cos_sin_cache,
            )
            sparse_args = sparse_indexer.call_args.kwargs
            self.assertIs(sparse_args["packed_q_values"], legacy[0])
            self.assertIs(sparse_args["packed_q_scales"], legacy[1])
            self.assertIs(sparse_args["packed_weights"], legacy[2])

            indexer._prepare_packed_inputs.reset_mock()
            sparse_indexer.reset_mock()
            run(supplied)
            indexer._prepare_packed_inputs.assert_not_called()
            sparse_args = sparse_indexer.call_args.kwargs
            self.assertIs(sparse_args["packed_q_values"], supplied[0])
            self.assertIs(sparse_args["packed_q_scales"], supplied[1])
            self.assertIs(sparse_args["packed_weights"], supplied[2])

            partials = (
                (supplied[0], supplied[1], None),
                (supplied[0], None, supplied[2]),
                (None, supplied[1], supplied[2]),
            )
            for partial in partials:
                with self.subTest(partial=tuple(x is not None for x in partial)):
                    indexer._prepare_packed_inputs.reset_mock()
                    sparse_indexer.reset_mock()
                    with self.assertRaisesRegex(ValueError, "provided together"):
                        run(partial)
                    indexer._prepare_packed_inputs.assert_not_called()
                    sparse_indexer.assert_not_called()

    def test_sanitized_insert_write_safety_under_graph_replay(self):
        # Full producer -> sanitize -> CUDA graph replay -> cache write path.
        # Slots and validity mutate between replays through static buffers;
        # sentinel bytes prove only legal cache slots are ever written.
        from tokenspeed.runtime.models.deepseek_v4 import (
            _deepseek_v4_sanitize_swa_slot_mapping,
        )

        torch.manual_seed(7)
        device = torch.device("cuda")
        dtype = torch.bfloat16
        num_tokens = 4
        num_heads = 2
        block_size = 4
        num_blocks = 2
        capacity = num_blocks * block_size
        eps = 1.0e-6
        sentinel = 0xAB
        token_bytes = 576
        scale_bytes = 8
        block_stride = block_size * (token_bytes + scale_bytes)
        guard_bytes = 4096

        storage = torch.full(
            (num_blocks * block_stride + guard_bytes,),
            sentinel,
            device=device,
            dtype=torch.uint8,
        )
        cache = storage[: num_blocks * block_stride].view(num_blocks, block_stride)
        guard = storage[num_blocks * block_stride :]

        q = torch.randn(num_tokens, num_heads, HEAD_DIM, device=device, dtype=dtype)
        kv = torch.randn(num_tokens, HEAD_DIM, device=device, dtype=dtype)
        positions = torch.tensor([0, 3, 5, 7], dtype=torch.int64, device=device)
        cos_sin = torch.randn(16, ROPE_DIM, device=device, dtype=torch.float32) * 0.1
        raw_slots = torch.zeros(num_tokens, dtype=torch.int64, device=device)
        is_valid = torch.ones(num_tokens, dtype=torch.bool, device=device)

        def run_once():
            sanitized = _deepseek_v4_sanitize_swa_slot_mapping(
                raw_slots, capacity, is_valid
            )
            fused_qnorm_rope_kv_insert(
                q=q,
                kv=kv,
                swa_kv_cache_2d=cache,
                slot_mapping=sanitized,
                positions=positions,
                cos_sin_cache=cos_sin,
                rms_norm_eps=eps,
                block_size=block_size,
            )

        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            try:
                run_once()
            except RuntimeError as exc:
                if "fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert" in str(exc):
                    self.skipTest(str(exc))
                raise
            run_once()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            run_once()

        def written_mask(slots):
            mask = torch.zeros(storage.numel(), dtype=torch.bool)
            for slot in slots:
                if slot < 0 or slot >= capacity:
                    continue
                block, pos = divmod(slot, block_size)
                base = block * block_stride
                start = base + pos * token_bytes
                mask[start : start + token_bytes] = True
                sstart = base + block_size * token_bytes + pos * scale_bytes
                mask[sstart : sstart + scale_bytes] = True
            return mask

        replays = [
            # (raw slots, validity, slots the kernel may legally write)
            ([0, 5, 99, -1], [True, True, True, True], [0, 5]),
            ([2, 0, 7, 3], [True, False, True, True], [2, 7, 3]),
        ]
        for slots, valid, expected in replays:
            storage.fill_(sentinel)
            raw_slots.copy_(torch.tensor(slots, dtype=torch.int64, device=device))
            is_valid.copy_(torch.tensor(valid, dtype=torch.bool, device=device))
            graph.replay()
            torch.cuda.synchronize()
            host = storage.cpu()
            allowed = written_mask(expected)
            outside = host[~allowed]
            self.assertTrue(
                bool((outside == sentinel).all()),
                f"bytes outside legal slots {expected} were modified",
            )
            for slot in expected:
                block, pos = divmod(slot, block_size)
                region = host[
                    block * block_stride
                    + pos * token_bytes : block * block_stride
                    + (pos + 1) * token_bytes
                ]
                self.assertFalse(
                    bool((region == sentinel).all()),
                    f"legal slot {slot} was not written",
                )
        self.assertTrue(bool((guard.cpu() == sentinel).all()))

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


if __name__ == "__main__":
    unittest.main()
