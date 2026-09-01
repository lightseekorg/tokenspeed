from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.attention.backends import trtllm_mla
from tokenspeed.runtime.layers.attention.backends.trtllm_mla import (
    TRTLLMMLABackend,
    TRTLLMMLADecodeMetadata,
)


def _backend(*, draft_block_decode: bool) -> TRTLLMMLABackend:
    backend = TRTLLMMLABackend.__new__(TRTLLMMLABackend)
    backend.is_draft = True
    backend.draft_block_decode = draft_block_decode
    backend.spec_num_tokens = 4
    backend.max_context_len = 64
    backend.kernel_page_size = 2
    backend.kv_lora_rank = 2
    backend.qk_nope_head_dim = 2
    backend.qk_rope_head_dim = 2
    backend.v_head_dim = 2
    backend.kv_cache_dim = 4
    backend.scaling = 1.0
    backend.data_type = torch.bfloat16
    backend.trtllm_workspace = torch.empty(1, dtype=torch.uint8)
    backend.device = torch.device("cpu")
    backend.max_num_pages = backend._calc_padded_blocks(backend.max_context_len)
    backend._cache_groups_bound = False
    backend.decode_cuda_graph_metadata = {}
    backend.decode_cuda_graph_out_cache_loc = None
    backend.forward_decode_metadata = None
    return backend


def test_eager_draft_page_table_is_not_expanded_twice() -> None:
    backend = _backend(draft_block_decode=True)
    kernel_page_table = torch.tensor([[6, 7, 10, 11, 14, 15]], dtype=torch.int32)

    backend.init_cuda_graph_state(max_bs=1)
    backend.refresh_decode_metadata(
        1,
        1,
        torch.tensor([1], dtype=torch.int64),
        torch.tensor([12], dtype=torch.int32),
        forward_mode=ForwardMode.DECODE,
        page_table=kernel_page_table,
    )

    block_table = backend.forward_decode_metadata.page_table
    assert block_table[0, :6].tolist() == [6, 7, 10, 11, 14, 15]
    assert block_table[0, 6:].eq(0).all()


def test_graph_replay_draft_page_table_is_not_expanded_twice() -> None:
    backend = _backend(draft_block_decode=True)
    backend.init_cuda_graph_state(max_bs=2)
    backend.init_forward_metadata_capture_cuda_graph(
        bs=2,
        req_pool_indices=torch.tensor([0, 1], dtype=torch.int64),
        seq_lens=torch.tensor([1, 1], dtype=torch.int32),
        forward_mode=ForwardMode.DECODE,
    )
    kernel_page_table = torch.tensor(
        [[6, 7, 10, 11], [14, 15, 18, 19]], dtype=torch.int32
    )

    # DraftPageStaging publishes kernel pages; the replay must copy them as-is
    # (identity), never re-expand. seq_lens fit the 4-page (page_size=2) table.
    backend.refresh_decode_metadata(
        2,
        2,
        torch.tensor([0, 1], dtype=torch.int64),
        torch.tensor([8, 8], dtype=torch.int32),
        forward_mode=ForwardMode.DECODE,
        page_table=kernel_page_table,
        for_graph_replay=True,
    )

    block_table = backend.forward_decode_metadata.page_table
    assert block_table[:, :4].tolist() == kernel_page_table.tolist()
    assert block_table[:, 4:].eq(0).all()


def _layer() -> SimpleNamespace:
    return SimpleNamespace(
        layer_id=0,
        tp_q_head_num=1,
        head_dim=4,
        v_head_dim=2,
        scaling=1.0,
    )


def test_fill_block_decode_seq_lens_publishes_clamped_lengths() -> None:
    backend = _backend(draft_block_decode=True)
    backend.cuda_graph_seq_lens = torch.zeros(2, dtype=torch.int32)

    backend.fill_block_decode_seq_lens(2, torch.tensor([3, 99], dtype=torch.int32))

    assert backend.cuda_graph_seq_lens.tolist() == [4, 64]


def test_block_decode_keeps_every_metadata_row_and_uses_uniform_lengths() -> None:
    backend = _backend(draft_block_decode=True)
    backend.forward_decode_metadata = TRTLLMMLADecodeMetadata(
        # Reproduce the old DFlash convention from the failing startup log.
        # Block decode must ignore this discriminator and retain both rows.
        num_extends=2,
        page_table=torch.tensor([[1, 2], [3, 4]], dtype=torch.int32),
        max_seq_len_k=64,
        seq_lens_k=torch.tensor([11, 23], dtype=torch.int32),
    )
    q = torch.zeros(8, 1, 4, dtype=torch.bfloat16)
    layer = _layer()
    pool = SimpleNamespace(
        get_key_buffer=lambda _layer_id: torch.zeros(4, 4, dtype=torch.bfloat16)
    )
    captured = {}

    def fake_decode(**kwargs):
        captured.update(kwargs)
        return torch.zeros(8, 1, 2, dtype=torch.bfloat16)

    with mock.patch.object(
        trtllm_mla, "trtllm_batch_decode_with_kv_cache_mla", fake_decode
    ):
        output = backend.forward_decode(
            q=q,
            k=None,
            v=None,
            layer=layer,
            out_cache_loc=torch.empty(0, dtype=torch.int32),
            token_to_kv_pool=pool,
            bs=2,
            save_kv_cache=False,
        )

    assert output.shape == (8, 2)
    assert captured["block_tables"].tolist() == [
        [1, 2],
        [1, 2],
        [1, 2],
        [1, 2],
        [3, 4],
        [3, 4],
        [3, 4],
        [3, 4],
    ]
    assert captured["seq_lens"].tolist() == [11] * 4 + [23] * 4
    assert captured["max_seq_len"] == 64


def test_non_block_draft_keeps_causal_catch_up_offsets() -> None:
    backend = _backend(draft_block_decode=False)
    backend.forward_decode_metadata = TRTLLMMLADecodeMetadata(
        num_extends=0,
        page_table=torch.tensor([[1, 2]], dtype=torch.int32),
        max_seq_len_k=60,
        seq_lens_k=torch.tensor([11], dtype=torch.int32),
    )
    q = torch.zeros(4, 1, 4, dtype=torch.bfloat16)
    layer = _layer()
    pool = SimpleNamespace(
        get_key_buffer=lambda _layer_id: torch.zeros(4, 4, dtype=torch.bfloat16)
    )
    captured = {}

    def fake_decode(**kwargs):
        captured.update(kwargs)
        return torch.zeros(4, 1, 2, dtype=torch.bfloat16)

    with mock.patch.object(
        trtllm_mla, "trtllm_batch_decode_with_kv_cache_mla", fake_decode
    ):
        backend.forward_decode(
            q=q,
            k=None,
            v=None,
            layer=layer,
            out_cache_loc=torch.empty(0, dtype=torch.int32),
            token_to_kv_pool=pool,
            bs=1,
            save_kv_cache=False,
        )

    assert captured["seq_lens"].tolist() == [11, 12, 13, 14]
    assert captured["max_seq_len"] == 64
