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

"""DeepSeek V4-Flash Hopper (SM90) FP8 paths.

Covers the pieces added for running V4-Flash on SM90 without FP4 tensor
cores: the FP8 indexer query prep + paged gather, the BF16->block-FP8
weight quantization used when a checkpoint ships ``attn.wo_a`` unquantized,
and the Lamport all-reduce dtype gate that keeps fp32 payloads off a
bf16-armed one-shot workspace.
"""

from __future__ import annotations

import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

from tokenspeed.runtime.layers.attention.deepseek_v4_ops import (
    _write_deepseek_v4_indexer_fp8_cache_capturable,
    deepseek_v4_prepare_indexer_q_fp8,
    gather_paged_indexer_fp8_cache,
    read_deepseek_v4_indexer_fp8_cache,
)
from tokenspeed.runtime.models.deepseek_v4 import DeepseekV4ForCausalLM

register_cuda_ci(est_time=30, suite="runtime-1gpu")

INDEX_HEAD_DIM = 128
BLOCK = 64
ROW_BYTES = INDEX_HEAD_DIM + 4  # fp8 values + fp32 per-token scale


def _fill_fp8_indexer_cache(
    num_reqs: int,
    context_lens: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, dict[int, torch.Tensor]]:
    """Build a paged FP8 indexer cache via the production insert helper."""

    max_ctx = int(context_lens.max())
    blocks_per_req = (max_ctx + BLOCK - 1) // BLOCK
    pages = num_reqs * blocks_per_req + 1
    cache = torch.zeros((pages, BLOCK * ROW_BYTES), dtype=torch.uint8, device=device)
    block_table = torch.zeros(
        (num_reqs, blocks_per_req), dtype=torch.int32, device=device
    )
    slots_by_req: dict[int, torch.Tensor] = {}
    for req in range(num_reqs):
        n = int(context_lens[req])
        base = 1 + req * blocks_per_req
        num_blocks = (n + BLOCK - 1) // BLOCK
        block_table[req, :num_blocks] = torch.arange(
            base, base + num_blocks, dtype=torch.int32, device=device
        )
        rows = torch.randn((n, INDEX_HEAD_DIM), device=device, dtype=torch.bfloat16)
        token_idx = torch.arange(n, device=device)
        slots = (
            block_table[req, token_idx // BLOCK].to(torch.int64) * BLOCK
            + token_idx % BLOCK
        )
        _write_deepseek_v4_indexer_fp8_cache_capturable(
            rows,
            cache,
            slots,
            torch.ones(n, dtype=torch.bool, device=device),
            block_size=BLOCK,
        )
        slots_by_req[req] = slots
    return cache, block_table, slots_by_req


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestGatherPagedIndexerFp8Cache(unittest.TestCase):
    def test_gather_matches_slot_read(self):
        torch.manual_seed(0)
        device = torch.device("cuda")
        context_lens = torch.tensor([37, 64, 130], dtype=torch.int32, device=device)
        cache, block_table, slots_by_req = _fill_fp8_indexer_cache(
            len(context_lens), context_lens, device
        )
        cu_seq_lens = torch.zeros(
            len(context_lens) + 1, dtype=torch.int32, device=device
        )
        cu_seq_lens[1:] = torch.cumsum(context_lens, 0)

        k_fp8, k_scale = gather_paged_indexer_fp8_cache(
            cache, block_table, cu_seq_lens, BLOCK
        )
        self.assertEqual(k_fp8.dtype, torch.float8_e4m3fn)
        self.assertEqual(k_fp8.shape, (int(cu_seq_lens[-1]), INDEX_HEAD_DIM))
        self.assertEqual(k_scale.shape, (int(cu_seq_lens[-1]),))

        for req in range(len(context_lens)):
            start, end = int(cu_seq_lens[req]), int(cu_seq_lens[req + 1])
            gathered = k_fp8[start:end].float() * k_scale[start:end].unsqueeze(1)
            reference = read_deepseek_v4_indexer_fp8_cache(
                cache, slots_by_req[req], block_size=BLOCK
            )
            torch.testing.assert_close(gathered, reference, rtol=0, atol=0)

    def test_empty_gather(self):
        device = torch.device("cuda")
        cache = torch.zeros((1, BLOCK * ROW_BYTES), dtype=torch.uint8, device=device)
        block_table = torch.zeros((1, 1), dtype=torch.int32, device=device)
        cu_seq_lens = torch.zeros(2, dtype=torch.int32, device=device)
        k_fp8, k_scale = gather_paged_indexer_fp8_cache(
            cache, block_table, cu_seq_lens, BLOCK
        )
        self.assertEqual(k_fp8.shape, (0, INDEX_HEAD_DIM))
        self.assertEqual(k_scale.shape, (0,))

    def test_write_and_gather_on_strided_arena_view(self):
        """The indexer cache is a strided field view of a larger LCM arena.

        Regression: computing page * stride(0) offsets into reshape(-1) is
        only valid on contiguous caches. On a strided view reshape(-1) copies
        just the logical elements, so stride-based offsets read/write out of
        bounds (crashed serving with a device-side IndexKernel assert).
        """

        torch.manual_seed(0)
        device = torch.device("cuda")
        pages = 7
        row_bytes = BLOCK * ROW_BYTES
        arena_stride = row_bytes + 1024  # field view narrower than the arena
        arena = torch.zeros((pages, arena_stride), dtype=torch.uint8, device=device)
        cache = arena[:, :row_bytes]
        self.assertFalse(cache.is_contiguous())

        n = 100
        rows = torch.randn((n, INDEX_HEAD_DIM), device=device, dtype=torch.bfloat16)
        # Use the highest pages so stride-based addressing would land far
        # outside the logical view.
        slots = torch.arange(
            pages * BLOCK - n, pages * BLOCK, device=device, dtype=torch.int64
        )
        _write_deepseek_v4_indexer_fp8_cache_capturable(
            rows,
            cache,
            slots,
            torch.ones(n, dtype=torch.bool, device=device),
            block_size=BLOCK,
        )
        # Bytes outside the field view must stay untouched.
        self.assertEqual(int(arena[:, row_bytes:].sum()), 0)

        reference = read_deepseek_v4_indexer_fp8_cache(cache, slots, block_size=BLOCK)
        rel_err = (reference - rows.float()).abs().max() / rows.float().abs().max()
        # fp8e4m3 with power-of-two row scales quantizes to ~2^-4 relative
        # granularity near the top of a binade.
        self.assertLess(float(rel_err), 0.07)

        block_table = torch.arange(pages, device=device, dtype=torch.int32).view(1, -1)
        cu_seq_lens = torch.tensor([0, pages * BLOCK], dtype=torch.int32, device=device)
        k_fp8, k_scale = gather_paged_indexer_fp8_cache(
            cache, block_table, cu_seq_lens, BLOCK
        )
        gathered = k_fp8[-n:].float() * k_scale[-n:].unsqueeze(1)
        torch.testing.assert_close(gathered, reference, rtol=0, atol=0)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestPrepareIndexerQFp8(unittest.TestCase):
    def test_scales_fold_into_weights(self):
        torch.manual_seed(0)
        device = torch.device("cuda")
        tokens, heads = 5, 64
        rope_dim = 64
        index_q = torch.randn(
            (tokens, heads, INDEX_HEAD_DIM), device=device, dtype=torch.bfloat16
        )
        positions = torch.arange(tokens, device=device)
        cos_sin = torch.randn((64, rope_dim), device=device, dtype=torch.float32)
        weights = torch.rand((tokens, heads), device=device, dtype=torch.float32)
        softmax_scale = INDEX_HEAD_DIM**-0.5
        head_scale = heads**-0.5

        q_fp8, weights_out = deepseek_v4_prepare_indexer_q_fp8(
            index_q, positions, cos_sin, weights, softmax_scale, head_scale
        )
        self.assertEqual(q_fp8.dtype, torch.float8_e4m3fn)
        self.assertEqual(q_fp8.shape, index_q.shape)
        torch.testing.assert_close(
            weights_out, weights * softmax_scale * head_scale, rtol=1e-6, atol=0
        )

    def test_zero_tokens(self):
        device = torch.device("cuda")
        q_fp8, weights_out = deepseek_v4_prepare_indexer_q_fp8(
            torch.empty((0, 64, INDEX_HEAD_DIM), device=device, dtype=torch.bfloat16),
            torch.empty((0,), device=device, dtype=torch.int64),
            torch.randn((16, 64), device=device, dtype=torch.float32),
            torch.empty((0, 64), device=device, dtype=torch.float32),
            0.1,
            0.125,
        )
        self.assertEqual(q_fp8.shape, (0, 64, INDEX_HEAD_DIM))
        self.assertEqual(weights_out.shape, (0, 64))


class TestBlockQuantFp8Weight(unittest.TestCase):
    def test_round_trip_within_fp8_error(self):
        torch.manual_seed(0)
        weight = torch.randn(256, 384, dtype=torch.bfloat16) * 3.0
        qweight, scale_inv = DeepseekV4ForCausalLM._block_quant_fp8_weight(weight)
        self.assertEqual(qweight.dtype, torch.float8_e4m3fn)
        self.assertEqual(qweight.shape, weight.shape)
        self.assertEqual(scale_inv.shape, (2, 3))

        expanded = scale_inv.repeat_interleave(128, 0)[:256]
        expanded = expanded.repeat_interleave(128, 1)[:, :384]
        dequant = qweight.float() * expanded
        rel_err = (dequant - weight.float()).abs().max() / weight.float().abs().max()
        self.assertLess(float(rel_err), 0.05)

    def test_non_multiple_of_block(self):
        torch.manual_seed(0)
        weight = torch.randn(130, 200, dtype=torch.bfloat16)
        qweight, scale_inv = DeepseekV4ForCausalLM._block_quant_fp8_weight(weight)
        self.assertEqual(qweight.shape, (130, 200))
        self.assertEqual(scale_inv.shape, (2, 2))


class TestLamportAllreduceDtypeGate(unittest.TestCase):
    def _backend_with_resources(self, use_fp32_lamport: bool):
        from tokenspeed.runtime.distributed.comm_backend.trtllm_allreduce import (
            TrtllmAllReduceBackend,
        )

        backend = TrtllmAllReduceBackend(fallback=None)
        resources = {
            "workspace": object(),
            "mnnvl": None,
            "rank": 0,
            "world_size": 8,
            "max_token_num": 64,
            "hidden_dim": 4096,
            "use_fp32_lamport": use_fp32_lamport,
        }
        return backend, resources

    def test_fp32_payload_on_bf16_workspace_falls_back(self):
        backend, resources = self._backend_with_resources(use_fp32_lamport=False)
        tensor = torch.zeros(2, 256, dtype=torch.float32)
        self.assertIsNone(backend._lamport_allreduce(tensor, resources))

    def test_bf16_payload_on_fp32_workspace_falls_back(self):
        backend, resources = self._backend_with_resources(use_fp32_lamport=True)
        tensor = torch.zeros(2, 256, dtype=torch.bfloat16)
        self.assertIsNone(backend._lamport_allreduce(tensor, resources))

    def test_zero_token_short_circuits_before_gate(self):
        backend, resources = self._backend_with_resources(use_fp32_lamport=False)
        tensor = torch.zeros(0, 256, dtype=torch.float32)
        self.assertIs(backend._lamport_allreduce(tensor, resources), tensor)


if __name__ == "__main__":
    unittest.main()
