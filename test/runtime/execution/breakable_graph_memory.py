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

"""Exercise breakable-graph storage growth and real allocator sleep/wake."""

import gc
from dataclasses import dataclass

import torch

from tokenspeed.runtime.execution.breakable_cuda_graph import (
    BreakableCapture,
    break_point,
)
from tokenspeed.runtime.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter


@dataclass
class Scale:
    value: float


@torch.inference_mode()
def main():
    saver = TorchMemorySaverAdapter.create(True, device="cuda")
    storage, captures, pool = {}, [], None
    with saver.region(tag="weights", enable_cpu_backup=True):
        weight = torch.ones(16, device="cuda")
    for rows in (8, 32):
        with saver.named_region("executor_inputs"):
            inputs = torch.ones((rows, 16), device="cuda")
        scale = Scale(2.0)

        @break_point
        def transform(value, scale):
            return value * scale.value

        def forward(inputs=inputs, scale=scale):
            return transform(inputs + weight, scale) + 3

        forward()
        torch.cuda.synchronize()
        cap = BreakableCapture(
            pool=pool, handoff_storage=storage, memory_saver_adapter=saver
        )
        with cap:
            output = forward()
        pool = cap.pool
        captures.append((cap, inputs, output, scale))
    addresses = [
        tensor.data_ptr()
        for _, inputs, output, _ in captures
        for tensor in (inputs, output)
    ] + [weight.data_ptr()]
    for cycle in range(3):
        weight.fill_(cycle + 1)
        for cap, inputs, output, scale in captures:
            inputs.fill_(cycle + 2)
            scale.value = cycle + 3.0
            cap.replay(valid_rows=inputs.shape[0])
            expected = torch.full(inputs.shape, (2 * cycle + 3) * scale.value + 3)
            assert torch.equal(output.cpu(), expected)
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        saver.pause(tag="cuda_graph")
        saver.pause(tag="weights")
        try:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            ordinary = sum(
                segment["total_size"]
                for segment in torch.cuda.memory_snapshot()
                if tuple(segment["segment_pool_id"]) == (0, 0)
            )
            assert ordinary == 0, ordinary
        finally:
            saver.resume(tag="weights")
            saver.resume(tag="cuda_graph")
        assert addresses == [
            tensor.data_ptr()
            for _, inputs, output, _ in captures
            for tensor in (inputs, output)
        ] + [weight.data_ptr()]
        for cap, inputs, output, scale in captures:
            cap.replay(valid_rows=inputs.shape[0])
            expected = torch.full(inputs.shape, (2 * cycle + 3) * scale.value + 3)
            assert torch.equal(output.cpu(), expected)
    prefill_storage(saver)
    print(
        "PASS breakable graph: growth, live values, three sleep/wake cycles, zero ordinary allocator bytes"
    )


class ToyInner(torch.nn.Module):
    """Stand in for model weights while using the real PrefillGraph owner."""

    def __init__(self):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(4, 16, device="cuda")
        self.with_aux = False

    @break_point
    def add_metadata(self, value, ctx):
        return (
            value + ctx.query_shard.local_index_of_row[:, None] + ctx.gather_ids.sum()
        )

    def forward(self, input_ids, positions, ctx, *, input_embeds):
        hidden = self.add_metadata(input_embeds + 1, ctx) + 2
        return hidden, [hidden * 2] if self.with_aux else None


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = ToyInner()


@dataclass(frozen=True)
class ToyModelConfig:
    requires_request_token_history: bool = False


@dataclass(frozen=True)
class ToyRunner:
    model: ToyModel
    model_config: ToyModelConfig
    is_generation: bool = True


class NoAttention:
    """The weight stub has no attention or KV payload to initialize."""

    def init_prefill_graph_state(self, **kwargs):
        pass

    def init_forward_metadata(self, **kwargs):
        pass

    def admits_prefill_graph(self, *args):
        return False

    def prepare_prefill_metadata(self, *args, capture):
        return False


@dataclass(frozen=True)
class EmptyArena:
    cache_group_specs: tuple = ()


@dataclass(frozen=True)
class EmptyPool:
    arena: EmptyArena
    layerwise_load_tracker: None = None
    layer_accesses_wait_for_load: bool = False


def prefill_config():
    """Small executor capacities shared by the real-owner allocator regressions."""
    from tokenspeed.runtime.execution.model_executor import ModelExecutorConfig

    return ModelExecutorConfig(
        max_req_pool_size=3,
        output_length=1,
        enforce_eager=False,
        enable_dp_local_cuda_graph=False,
        acl_graph_memory_policy="discard",
        prefix_granularity=1,
        max_num_seqs=2,
        chunked_prefill_size=32,
        vocab_size=4,
        input_vocab_size=4,
        context_len=32,
        physical_context_len=32,
        device="cuda",
        gpu_id=0,
        global_rank=1,
        cudagraph_capture_sizes=[1],
        disable_cuda_graph_padding=False,
        spec_topk=1,
        max_cudagraph_capture_size=1,
        model_is_mrope=False,
        autotune_cache_key=None,
        prefill_only=False,
        decode_only_attention=False,
        prefill_graph_capture_batch_sizes=[1],
        input_logprob_chunk_tokens=32,
        enable_speculative_sampling=False,
        query_shard_size=2,
        query_shard_rank=0,
        enable_memory_saver=True,
        prefill_graph_max_tokens=32,
        prefill_graph_capture_sizes=[8, 32],
    )


def prefill_storage(saver):
    """Walk real input, output and query-shard allocation and recapture paths."""
    from tokenspeed.runtime.execution.breakable_cuda_graph import active_forward
    from tokenspeed.runtime.execution.input_buffer import InputBuffers
    from tokenspeed.runtime.execution.memory_delta import NULL_MEMORY_DELTA_OBSERVER
    from tokenspeed.runtime.execution.prefill_graph import PrefillGraph

    config = prefill_config()
    with saver.named_region("executor_inputs"):
        buffers = InputBuffers(2, 32, 2, device="cuda")
    with saver.region(tag="weights", enable_cpu_backup=True):
        model = ToyModel()
    graph = PrefillGraph(
        model_runner=ToyRunner(model, ToyModelConfig()),
        attn_backend=NoAttention(),
        token_to_kv_pool=EmptyPool(EmptyArena()),
        input_buffers=buffers,
        config=config,
        memory_saver_adapter=saver,
        num_warmup=1,
    )
    # Recapture changes the output spec after clearing its old captures, as
    # reload-forward does. Output storage is allocated lazily on each pass.
    for with_aux in (False, True):
        model.model.with_aux = with_aux
        graph.release_graphs()
        graph.capture(entries=None, observer=NULL_MEMORY_DELTA_OBSERVER)
        contexts = {
            bucket: graph.make_dummy_batch(bucket, 1) for bucket, _ in graph._captures
        }
        persistent = [graph._input_embeds_buf, *graph._outputs]
        for ctx in contexts.values():
            persistent.extend([ctx.gather_ids, ctx.query_shard.local_index_of_row])
        addresses = [tensor.data_ptr() for tensor in persistent]

        def check(cycle):
            graph._input_embeds_buf.fill_(cycle)
            for (bucket, _), (capture, output) in graph._captures.items():
                ctx = contexts[bucket]
                with active_forward(ctx):
                    capture.replay(valid_rows=bucket)
                expected = (
                    cycle
                    + 3
                    + ctx.query_shard.local_index_of_row.cpu()[:, None]
                    + ctx.gather_ids.cpu().sum()
                ).expand(bucket, 16)
                assert torch.equal(output.hidden_states.cpu(), expected)
                if with_aux:
                    assert torch.equal(output.aux_hidden_states[0].cpu(), expected * 2)

        for cycle in range(3):
            check(cycle)
            gc.collect()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            saver.pause(tag="cuda_graph")
            saver.pause(tag="weights")
            try:
                torch.cuda.empty_cache()
                ordinary = sum(
                    segment["total_size"]
                    for segment in torch.cuda.memory_snapshot()
                    if tuple(segment["segment_pool_id"]) == (0, 0)
                )
                assert ordinary == 0, ordinary
            finally:
                saver.resume(tag="weights")
                saver.resume(tag="cuda_graph")
            assert addresses == [tensor.data_ptr() for tensor in persistent]
            check(cycle)
    print(
        "PASS PrefillGraph: query shards, lazy outputs, recapture, three sleep/wake cycles"
    )


if __name__ == "__main__":
    main()
