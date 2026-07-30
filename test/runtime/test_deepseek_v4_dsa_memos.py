import torch

from tokenspeed.runtime.layers.attention.backends.deepseek_v4 import (
    DeepseekV4AttentionBackend,
)
from tokenspeed.runtime.layers.attention.deepseek_v4.metadata import (
    DeepseekV4ForwardMetadata,
)
from tokenspeed.runtime.layers.attention.kv_cache.deepseek_v4 import (
    DeepseekV4CacheMetadata,
)


def _make_metadata() -> DeepseekV4ForwardMetadata:
    one = torch.zeros(1, dtype=torch.int32)
    return DeepseekV4ForwardMetadata(
        req_pool_indices=one,
        seq_lens=one,
        query_lens=one,
        query_start_loc=torch.zeros(2, dtype=torch.int32),
        token_to_req_indices=one,
        cache=DeepseekV4CacheMetadata(
            page_size=1,
            block_table=torch.zeros((1, 1), dtype=torch.int32),
        ),
    )


def test_metadata_dsa_memos_default_to_none():
    metadata = _make_metadata()

    assert metadata.swa_slot_mapping is None
    assert metadata.compressor_slot_cache is None


def test_reset_cross_layer_memos_clears_all_tracked_metadata():
    backend = DeepseekV4AttentionBackend.__new__(DeepseekV4AttentionBackend)
    forward = _make_metadata()
    prefill = _make_metadata()
    draft = _make_metadata()
    for metadata in (forward, prefill, draft):
        metadata.swa_slot_mapping = torch.zeros(1, dtype=torch.int64)
        metadata.compressor_slot_cache = {"indexer_state": object()}
    backend.forward_metadata = forward
    backend.forward_prefill_metadata = prefill
    backend.forward_decode_metadata = forward  # decode forwards alias forward_metadata
    backend._draft_decode_metadata = draft

    backend.reset_cross_layer_memos()

    for metadata in (forward, prefill, draft):
        assert metadata.swa_slot_mapping is None
        assert metadata.compressor_slot_cache is None
