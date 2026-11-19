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

"""Graph padding can exceed the scheduler's live-token issue budget."""

import sys
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, suite="runtime-1gpu")

from tokenspeed.runtime.execution.input_buffer import InputBuffers
from tokenspeed.runtime.execution.model_executor import ModelExecutorConfig
from tokenspeed.runtime.execution.prefill_graph import get_prefill_token_buckets


def _config() -> ModelExecutorConfig:
    return ModelExecutorConfig(
        max_req_pool_size=129,
        output_length=4,
        enforce_eager=False,
        enable_dp_local_cuda_graph=False,
        prefix_granularity=256,
        max_num_seqs=128,
        chunked_prefill_size=2048,
        vocab_size=32,
        input_vocab_size=32,
        context_len=4096,
        physical_context_len=4096,
        device="cpu",
        gpu_id=0,
        global_rank=0,
        cudagraph_capture_sizes=[1, 8, 32, 128],
        disable_cuda_graph_padding=False,
        spec_topk=1,
        max_cudagraph_capture_size=128,
        model_is_mrope=False,
        autotune_cache_key=None,
        prefill_only=False,
        decode_only_attention=False,
        prefill_graph_capture_batch_sizes=[1, 8, 32],
        input_logprob_chunk_tokens=1024,
        enable_speculative_sampling=False,
        query_shard_size=1,
        query_shard_rank=0,
        enable_memory_saver=False,
        prefill_graph_max_tokens=2080,
        prefill_graph_capture_sizes=[128, 2048, 2080],
    )


def test_explicit_graph_ceiling_preserves_native_scan_padding_room():
    config = _config()
    assert get_prefill_token_buckets(config) == [128, 2048, 2080]
    assert config.chunked_prefill_size == 2048


@pytest.mark.parametrize(
    "overrides, expected",
    [
        ({}, 2080),
        ({"disable_prefill_graph": True}, 2048),
        ({"enforce_eager": True}, 2048),
        ({"prefill_graph_max_tokens": 0}, 2048),
        ({"chunked_prefill_size": 128, "prefill_graph_max_tokens": 0}, 512),
        (
            {
                "chunked_prefill_size": 128,
                "prefill_graph_max_tokens": 0,
                "data_parallel_size": 2,
            },
            256,
        ),
    ],
)
def test_input_storage_covers_live_decode_and_padded_prefill(overrides, expected):
    config = replace(_config(), **overrides)
    buffers = InputBuffers(
        max_bs=config.max_num_seqs // config.data_parallel_size,
        max_num_tokens=config.input_token_capacity,
        state_write_padding_pool_index=config.max_req_pool_size,
        device=config.device,
        reads_state_checkpoints=config.state_checkpoint_in_forward,
    )
    assert buffers.input_ids_buf.numel() == expected
    assert buffers.positions_buf.numel() == expected
    assert buffers.input_ids_buf[-1].item() == 1
    assert buffers.positions_buf[-1].item() == 0
