import torch

from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.logits_processor import LogitsMetadata


def test_logits_metadata_derives_basic_fields_from_forward_context():
    gather_ids = torch.tensor([0], dtype=torch.int64)
    ctx = ForwardContext(
        attn_backend=None,
        token_to_kv_pool=None,
        bs=1,
        num_extends=0,
        input_num_tokens=1,
        forward_mode=ForwardMode.DECODE,
        gather_ids=gather_ids,
    )

    metadata = LogitsMetadata.from_forward_context(
        ctx,
        torch.tensor([1], dtype=torch.int32),
    )

    assert metadata.forward_mode == ForwardMode.DECODE
    assert metadata.gather_ids is gather_ids
    assert metadata.extend_seq_lens.tolist() == [1]
