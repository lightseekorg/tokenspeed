import torch

from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.logits_processor import LogitsMetadata


def _make_ctx() -> ForwardContext:
    return ForwardContext(
        attn_backend=None,
        token_to_kv_pool=None,
        bs=1,
        num_extends=0,
        input_num_tokens=1,
        forward_mode=ForwardMode.DECODE,
    )


def test_logits_metadata_derives_basic_fields_from_forward_context():
    gather_ids = torch.tensor([0], dtype=torch.int64)

    metadata = LogitsMetadata.from_forward_context(_make_ctx(), gather_ids=gather_ids)

    assert metadata.forward_mode == ForwardMode.DECODE
    assert metadata.gather_ids is gather_ids


def test_logits_metadata_gather_ids_defaults_to_none():
    metadata = LogitsMetadata.from_forward_context(_make_ctx())

    assert metadata.gather_ids is None


def test_forward_context_has_no_gather_ids_field():
    assert not hasattr(_make_ctx(), "gather_ids")


def test_forward_context_has_no_dsa_memo_fields():
    ctx = _make_ctx()
    assert not hasattr(ctx, "dsa_swa_slot_mapping")
    assert not hasattr(ctx, "dsa_compressor_slot_cache")
