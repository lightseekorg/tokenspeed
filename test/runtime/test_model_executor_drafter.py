from types import SimpleNamespace

import torch

from tokenspeed.runtime.execution.model_executor import ModelExecutor


def test_run_drafter_skips_single_rank_mid_chunk_extends():
    def fail_run(**_kwargs):
        raise AssertionError("mid-chunk extend must not run the drafter")

    executor = SimpleNamespace(
        input_buffers=SimpleNamespace(all_extends_mid_chunk=True),
        config=SimpleNamespace(data_parallel_size=1),
        drafter=SimpleNamespace(run=fail_run),
    )
    ctx = SimpleNamespace(num_extends=1, bs=1)

    ModelExecutor._run_drafter_and_store(
        executor,
        ctx,
        logits_output=None,
        output_tokens=torch.tensor([1], dtype=torch.int32),
        accept_lengths=torch.tensor([1], dtype=torch.int32),
    )
