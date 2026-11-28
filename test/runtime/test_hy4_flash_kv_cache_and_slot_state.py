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

"""KV cache recipe, hybrid pool, router, and dense context slot tests for Hy4-flash."""

from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=15, suite="runtime-1gpu")

from tokenspeed.runtime.configs.hy4_flash_config import (  # noqa: E402
    INDEXED_ATTENTION,
    SLIDING_ATTENTION,
    HYV4FlashTextConfig,
)
from tokenspeed.runtime.execution.drafter.dfly import (  # noqa: E402
    materialize_dense_context_slots,
)
from tokenspeed.runtime.layers.attention.backends.hy4_flash import (  # noqa: E402
    Hy4FlashCacheGroupRouter,
)
from tokenspeed.runtime.layers.attention.configs.hy4_flash import (  # noqa: E402
    Hy4FlashAttentionConfig,
)
from tokenspeed.runtime.layers.attention.kv_cache.hybrid_hy4_flash import (  # noqa: E402
    HybridHy4FlashTokenToKVPool,
    _Hy4KPoolTailWorkspace,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.hy4_flash import (  # noqa: E402
    HY4_FLASH_LOGICAL_BLOCK_TOKENS,
    Hy4FlashPoolOptions,
)


def test_hy4_flash_attention_config_groups_dsa_and_swa_layers():
    cfg = HYV4FlashTextConfig(
        num_hidden_layers=6,
        num_attention_heads=32,
        swa_num_attention_heads=64,
        sliding_window=512,
        layer_types=[
            INDEXED_ATTENTION,
            INDEXED_ATTENTION,
            SLIDING_ATTENTION,
            SLIDING_ATTENTION,
            SLIDING_ATTENTION,
            INDEXED_ATTENTION,
        ],
    )
    assert cfg.dsa_layer_ids == (0, 1, 5)
    assert cfg.swa_layer_ids == (2, 3, 4)
    assert HY4_FLASH_LOGICAL_BLOCK_TOKENS == 64
    assert HY4_FLASH_LOGICAL_BLOCK_TOKENS // cfg.index_kpool == 16
    assert Hy4FlashAttentionConfig is not None


def test_hybrid_hy4_pool_index_k_views_and_tail_state():
    pool = HybridHy4FlashTokenToKVPool.__new__(HybridHy4FlashTokenToKVPool)
    pool.layer_num = 2
    pool.index_head_dim = 128
    pool.index_kpool = 4
    pool._index_k_page_size = 16
    pool._field_layer_offset = 0
    pool._layer_types = (INDEXED_ATTENTION, SLIDING_ATTENTION)
    pool.layerwise_load_tracker = None

    # Layout per page: 16 rows * 128 FP8 bytes + 16 rows * 4 scale bytes = 2112 bytes
    row_bytes = 16 * 128 + 16 * 4
    buf = torch.zeros((4, 16, row_bytes // 16), dtype=torch.uint8)
    pool._index_k = [buf, None]
    pool._hy4_kpool_tail_workspace = _Hy4KPoolTailWorkspace(
        options=Hy4FlashPoolOptions(
            index_kpool=4,
            tail_extra_slots=0,
            index_head_dim=128,
            num_request_slots=8,
            dsa_layer_ids=(0,),
        ),
        token_keys=torch.randn((1, 8, 4, 128), dtype=torch.bfloat16),
        group_pos0=torch.randint(0, 100, (8, 3), dtype=torch.int64),
        row_by_layer={0: 0},
    )

    values, scales = pool.index_k_block_views(buf)
    assert values.shape == (4, 16, 128)
    assert values.dtype == torch.float8_e4m3fn
    assert scales.shape == (4, 16, 1)

    idx_buf, tail_keys, pos0 = pool.get_hy4_kpool_buffers(0)
    assert idx_buf is buf
    assert tail_keys.shape == (8, 4, 128)
    assert pos0.shape == (8, 3)

    # SWA layer must raise loudly if index_k is requested
    with pytest.raises(ValueError, match="SWA layer 1"):
        pool.get_index_k_buffer(1)

    rows = pool.kpool_tail_slot_rows(3)
    assert len(rows) == 2
    assert rows[0].shape == (1, 4, 128)
    assert rows[1].shape == (3,)


def test_hy4_router_shared_topk_slots_lifecycle():
    router = Hy4FlashCacheGroupRouter.__new__(Hy4FlashCacheGroupRouter)
    router._hy4_shared_topk_slots = {}

    t25 = torch.tensor([[0, 1, 2, -1]], dtype=torch.int32)
    router.record_shared_topk_slots(25, t25)
    assert router.get_shared_topk_slots(25) is t25
    assert router.get_shared_topk_slots(29) is None

    router.clear_shared_topk_slots_for_layer(25)
    assert router.get_shared_topk_slots(25) is None

    router.record_shared_topk_slots(33, t25)
    router.clear_shared_topk_slots()
    assert router.get_shared_topk_slots(33) is None


def test_materialize_dense_context_slots_across_pages():
    page_table = torch.tensor([[3, 1, 7], [0, 5, 2]], dtype=torch.int32)
    ctx_lens = torch.tensor([6, 2], dtype=torch.int32)
    slots = materialize_dense_context_slots(
        page_table, ctx_lens, page_size=4, max_ctx_len=6
    )
    expected = torch.tensor(
        [[12, 13, 14, 15, 4, 5], [0, 1, -1, -1, -1, -1]], dtype=torch.int32
    )
    assert torch.equal(slots, expected)

    empty = materialize_dense_context_slots(
        page_table, torch.tensor([0, 0], dtype=torch.int32), page_size=4, max_ctx_len=0
    )
    assert empty.shape == (2, 0)


def test_hy4_decode_workspace_retires_and_uses_memory_saver_region():
    from contextlib import contextmanager

    from tokenspeed.runtime.models.hy4_flash import HYV4FlashCompressedIndexer

    entered: list[str] = []

    class _RecorderAdapter:
        @contextmanager
        def named_region(self, name: str):
            entered.append(name)
            yield

    indexer = HYV4FlashCompressedIndexer.__new__(HYV4FlashCompressedIndexer)
    indexer._decode_topk_workspace = None
    indexer._decode_topk_lens_workspace = None
    indexer._retired_decode_workspaces = []

    adapter = _RecorderAdapter()
    slots4, lens4 = indexer._get_decode_workspaces(
        4, 16, torch.device("cpu"), memory_saver_adapter=adapter
    )
    assert slots4.shape == (4, 16)
    assert lens4.shape == (4,)
    assert entered == ["workspace"]
    assert indexer._retired_decode_workspaces == []

    # Shrinking must slice the existing buffer without reallocating.
    slots2, lens2 = indexer._get_decode_workspaces(
        2, 16, torch.device("cpu"), memory_saver_adapter=adapter
    )
    assert slots2.shape == (2, 16)
    assert slots2.data_ptr() == slots4.data_ptr()
    assert lens2.data_ptr() == lens4.data_ptr()
    assert entered == ["workspace"]

    # Growing must retire the prior buffers and allocate in the workspace region.
    slots8, lens8 = indexer._get_decode_workspaces(
        8, 16, torch.device("cpu"), memory_saver_adapter=adapter
    )
    assert slots8.shape == (8, 16)
    assert lens8.shape == (8,)
    assert entered == ["workspace", "workspace"]
    assert len(indexer._retired_decode_workspaces) == 2


def test_dsa_validate_dense_context_uses_host_max_seq_len():
    import pytest

    from tokenspeed.runtime.layers.attention.backends.paged.dsa import DSABackend

    dsa = DSABackend.__new__(DSABackend)
    dsa.index_topk = 2048

    # Host-known bound avoids touching seq_lens.
    dsa._validate_dense_context(None, 2, max_seq_len=2048)
    with pytest.raises(NotImplementedError, match="index_topk"):
        dsa._validate_dense_context(None, 2, max_seq_len=2049)
