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

"""Blackwell correctness coverage for the FP8 MLA tree-mask kernel path."""

from __future__ import annotations

import math
import os
import sys
import unittest

import torch

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(
    est_time=300,
    suite="runtime-1gpu",
    disabled_on_runners=["amd-*"],
    disabled_on_runners_reason="TokenSpeed MLA decode requires NVIDIA Blackwell",
)

_HAS_SM100 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10

KV_LORA = 512
ROPE = 64
HEADS = 128
PAGE_SIZE = 64
Q_LEN = 4


def _build_mask(seq_lens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    ancestors = (
        torch.tril(torch.ones(Q_LEN, Q_LEN, dtype=torch.bool)),
        torch.tensor(
            [
                [1, 0, 0, 0],
                [1, 1, 0, 0],
                [1, 0, 1, 0],
                [1, 1, 0, 1],
            ],
            dtype=torch.bool,
        ),
    )
    offsets = torch.empty(len(ancestors), dtype=torch.int32, device="cuda")
    masks = []
    offset = 0
    for batch_idx, ancestor in enumerate(ancestors):
        seq_len = int(seq_lens[batch_idx])
        history = seq_len - Q_LEN
        mask = torch.ones(Q_LEN, seq_len, dtype=torch.bool, device="cuda")
        mask[:, 1:history:3] = False
        mask[:, history:] = ancestor.to("cuda")
        offsets[batch_idx] = offset
        masks.append(mask.flatten())
        offset += Q_LEN * seq_len
    return torch.cat(masks).contiguous(), offsets


def _reference(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    custom_mask: torch.Tensor,
    cmask_off: torch.Tensor,
) -> torch.Tensor:
    output = torch.empty(
        query.shape[0],
        Q_LEN,
        HEADS,
        KV_LORA,
        dtype=torch.float32,
        device="cuda",
    )
    query_float = query.float()
    scale = 1.0 / math.sqrt(KV_LORA + ROPE)
    for batch_idx in range(query.shape[0]):
        seq_len = int(seq_lens[batch_idx])
        mask_start = int(cmask_off[batch_idx])
        mask = custom_mask[mask_start : mask_start + Q_LEN * seq_len].view(
            Q_LEN, seq_len
        )
        request_kv = kv_cache[block_tables[batch_idx]].reshape(-1, KV_LORA + ROPE)[
            :seq_len
        ]
        scores = torch.einsum("qhd,kd->qhk", query_float[batch_idx], request_kv.float())
        probabilities = torch.softmax(
            (scores * scale).masked_fill(~mask[:, None, :], float("-inf")),
            dim=-1,
        )
        output[batch_idx] = torch.einsum(
            "qhk,kd->qhd", probabilities, request_kv[:, :KV_LORA].float()
        )
    return output


def _relative_max_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    denominator = expected.abs().max().clamp_min(1e-6)
    return ((actual - expected).abs().max() / denominator).item()


@unittest.skipUnless(_HAS_SM100, "TokenSpeed MLA decode requires Blackwell SM100")
class TestMLATreeMaskDecode(unittest.TestCase):
    def test_tree_mask_and_default_fp8_abi_match_reference(self):
        from tokenspeed_mla import tokenspeed_mla_decode

        torch.manual_seed(617)
        dtype = torch.float8_e4m3fn
        # Both lengths are non-tile-aligned and span at least three 128-column
        # KV tiles, exercising global mask indexing for k_index > 0.
        seq_lens = torch.tensor([300, 391], dtype=torch.int32, device="cuda")
        max_seq_len = int(seq_lens.max())
        pages_per_request = (max_seq_len + PAGE_SIZE - 1) // PAGE_SIZE
        block_tables = torch.arange(
            len(seq_lens) * pages_per_request, dtype=torch.int32, device="cuda"
        ).view(len(seq_lens), pages_per_request)
        query = (
            torch.randn(
                len(seq_lens),
                Q_LEN,
                HEADS,
                KV_LORA + ROPE,
                device="cuda",
            )
            * 0.3
        ).to(dtype)
        kv_cache = (
            torch.randn(
                len(seq_lens) * pages_per_request,
                PAGE_SIZE,
                KV_LORA + ROPE,
                device="cuda",
            )
            * 0.3
        ).to(dtype)
        workspace = torch.empty(1 << 29, dtype=torch.int8, device="cuda")
        custom_mask, cmask_off = _build_mask(seq_lens)

        def decode(**mask_kwargs):
            return tokenspeed_mla_decode(
                query=query,
                kv_cache=kv_cache,
                workspace_buffer=workspace,
                kv_lora_rank=KV_LORA,
                qk_rope_head_dim=ROPE,
                block_tables=block_tables,
                seq_lens=seq_lens,
                max_seq_len=max_seq_len,
                softmax_scale=1.0 / math.sqrt(KV_LORA + ROPE),
                causal_mask=False,
                **mask_kwargs,
            ).float()

        expected_tree = _reference(
            query, kv_cache, block_tables, seq_lens, custom_mask, cmask_off
        )
        actual_tree = decode(custom_mask=custom_mask, cmask_off=cmask_off)
        derived_offset_tree = decode(custom_mask=custom_mask)

        full_mask = torch.ones_like(custom_mask)
        expected_full = _reference(
            query, kv_cache, block_tables, seq_lens, full_mask, cmask_off
        )
        actual_full = decode()

        self.assertLess(_relative_max_error(actual_tree, expected_tree), 6e-2)
        self.assertLess(_relative_max_error(actual_full, expected_full), 6e-2)
        torch.testing.assert_close(actual_tree, derived_offset_tree, rtol=0, atol=0)

        self.assertGreater(_relative_max_error(expected_tree, expected_full), 6e-2)
        self.assertGreater(_relative_max_error(actual_tree, actual_full), 6e-2)


if __name__ == "__main__":
    unittest.main()
