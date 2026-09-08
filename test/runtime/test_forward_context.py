from types import SimpleNamespace

import torch

from tokenspeed.runtime.execution.context import ForwardContext
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.execution.model_executor import ModelExecutor
from tokenspeed.runtime.layers.logits_processor import LogitsMetadata


def test_logits_metadata_derives_basic_fields_from_forward_context():
    gather_ids = torch.tensor([0], dtype=torch.int64)
    ctx = ForwardContext(
        attn_backend=None,
        token_to_kv_pool=None,
        indexer_runtime=None,
        bs=1,
        num_extends=0,
        input_num_tokens=1,
        forward_mode=ForwardMode.DECODE,
        gather_ids=gather_ids,
    )

    metadata = LogitsMetadata.from_forward_context(ctx)

    assert metadata.forward_mode == ForwardMode.DECODE
    assert metadata.gather_ids is gather_ids


def test_eager_idle_forwards_keep_target_and_draft_runtimes_separate():
    contexts = []
    model_runner = SimpleNamespace(forward=lambda ctx, **kwargs: contexts.append(ctx))
    target_runtime = SimpleNamespace(workspace=torch.empty(4))
    draft_runtime = SimpleNamespace(workspace=torch.empty(2))
    executor = SimpleNamespace(
        attn_backend=None,
        token_to_kv_pool=None,
        indexer_runtime=target_runtime,
        device="cpu",
        model_runner=model_runner,
        input_buffers=SimpleNamespace(req_pool_indices_buf=torch.empty(0)),
        runtime_states=SimpleNamespace(
            valid_cache_lengths=torch.empty(0), vocab_size=8
        ),
        forward_step=SimpleNamespace(can_run=lambda **kwargs: False),
        drafter=SimpleNamespace(
            attn_backend=None,
            token_to_kv_pool=None,
            indexer_runtime=draft_runtime,
            draft_model_runner=model_runner,
            spec_num_steps=2,
        ),
    )
    ModelExecutor.execute_idle_forward(
        executor,
        SimpleNamespace(
            global_num_tokens=[0], global_batch_size=[0], all_decode_or_idle=True
        ),
    )

    assert len(contexts) == 3
    assert contexts[0].indexer_runtime is target_runtime
    assert all(ctx.indexer_runtime is draft_runtime for ctx in contexts[1:])
    assert all(ctx.forward_mode == ForwardMode.IDLE for ctx in contexts)
    assert all(
        not isinstance(value, torch.Tensor)
        for ctx in contexts
        for value in vars(ctx).values()
    )
