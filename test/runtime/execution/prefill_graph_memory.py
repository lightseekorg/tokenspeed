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

"""Real graph owners and allocation sites across growth and allocator sleep/wake.

Tiny model/attention substitutes avoid checkpoint and process-group dependencies;
capture, buffer allocation, metadata snapshots and release/resume use runtime code.
Each case runs in its own preload-configured subprocess.
"""

from __future__ import annotations

import gc
import sys
from dataclasses import dataclass, fields, is_dataclass, replace
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from test.runtime.execution.breakable_graph_memory import (  # noqa: E402
    EmptyPool,
    NoAttention,
    ToyModel,
    ToyModelConfig,
    ToyRunner,
    prefill_config,
)

from tokenspeed.runtime.execution.breakable_cuda_graph import (  # noqa: E402
    BreakableCapture,
    active_forward,
    break_point,
)
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode  # noqa: E402
from tokenspeed.runtime.execution.input_buffer import InputBuffers  # noqa: E402
from tokenspeed.runtime.execution.memory_delta import (  # noqa: E402
    NULL_MEMORY_DELTA_OBSERVER,
)
from tokenspeed.runtime.utils.torch_memory_saver_adapter import (  # noqa: E402
    TorchMemorySaverAdapter,
)


@dataclass(frozen=True)
class Arena:
    memory_saver_adapter: TorchMemorySaverAdapter
    cache_group_specs: tuple = ()


def device_tensors(value):
    if isinstance(value, torch.Tensor):
        return [value] if value.is_cuda else []
    if is_dataclass(value):
        return [
            t
            for field in fields(value)
            for t in device_tensors(getattr(value, field.name))
        ]
    if isinstance(value, dict):
        return [t for child in value.values() for t in device_tensors(child)]
    if isinstance(value, (list, tuple)):
        return [t for child in value for t in device_tensors(child)]
    return []


def sleep_wake(saver, persistent, run):
    from tokenspeed.runtime.engine.memory_occupation import MemoryRegion
    from tokenspeed.runtime.engine.rust_host import HostDataPlane

    # A real data-plane transition over the graph's real allocator regions;
    # this isolated workload has no model/communication lifecycle to initialize.
    host = HostDataPlane.__new__(HostDataPlane)
    host._device = torch.cuda
    host._replica_memory_transitions = False
    host._regions = {
        tag: MemoryRegion(
            release=partial(saver.pause, tag=tag),
            resume=partial(saver.resume, tag=tag),
        )
        for tag in ("cuda_graph", "weights")
    }
    addresses = [tensor.data_ptr() for tensor in persistent]
    for cycle in range(3):
        expected = run(cycle)
        contents = [tensor.cpu().clone() for tensor in persistent]
        gc.collect()
        torch.cuda.synchronize()
        host.release_memory(["cuda_graph", "weights"])
        try:
            torch.cuda.synchronize()
            ordinary = sum(
                segment["total_size"]
                for segment in torch.cuda.memory_snapshot()
                if tuple(segment["segment_pool_id"]) == (0, 0)
            )
            assert ordinary == 0, ordinary
        finally:
            host.resume_memory(["weights", "cuda_graph"])
        assert addresses == [tensor.data_ptr() for tensor in persistent]
        for tensor, before in zip(persistent, contents, strict=True):
            torch.testing.assert_close(
                tensor.cpu(), before, rtol=0, atol=0, equal_nan=True
            )
        actual = run(cycle)
        for before, after in zip(expected, actual, strict=True):
            assert torch.equal(before, after)


def eagle(saver, *, split):
    from test.runtime.drafter_prefill_graph_test_utils import (
        WIDTH,
        ToyDraftModel,
        ToySplitDraftModel,
        capture_input,
        make_eagle,
    )
    from test.runtime.test_drafter_prefill_graph_cuda import _inputs, _round

    class ManagedMetadata:
        @break_point
        def index(self, x, ctx):
            # The toy indexer, like production metadata initialization, owns its
            # retained row layout. Model weights and a real indexer are absent.
            with saver.named_region("attention_state"):
                return super().index(x, ctx)

    class Draft(ManagedMetadata, ToyDraftModel):
        pass

    class SplitDraft(ManagedMetadata, ToySplitDraftModel):
        pass

    with saver.named_region("runtime_states"):
        model = (SplitDraft if split else Draft)("cuda")
        drafter, backend = make_eagle(device="cuda", model=model)
    stream = torch.cuda.Stream()
    retained = []
    old_graphs = []
    for rows, width in ((8, WIDTH), (32, WIDTH + 16)):
        capture = capture_input(
            drafter, backend, [rows], entries=None, stream=stream, width=width
        )
        prepare = capture.prepare_batch

        def prepare_batch(tokens, bs):
            with saver.named_region("attention_state"):
                prepare(tokens, bs)

        capture = replace(
            capture, memory_saver_adapter=saver, prepare_batch=prepare_batch
        )
        drafter.capture_extend_graphs(capture, observer=NULL_MEMORY_DELTA_OBSERVER)
        retained.extend(device_tensors(drafter._extend_buffers))
        if drafter._kv_tile is not None:
            retained.extend(drafter._kv_tile.inputs())
            old_graphs.append(drafter._kv_tile.graph)
        old_graphs.extend(drafter._decode_steps_graphs.values())

        def run(cycle):
            with saver.named_region("attention_state"):
                backend.cache.zero_()
                model.fold.fill_(cycle + 1)
                inputs = _inputs([5, 3], torch.Generator().manual_seed(10 + cycle))
                hidden = torch.nn.functional.pad(inputs[2], (0, width - WIDTH))
                window, _ = _round(
                    drafter,
                    backend,
                    [5, 3],
                    [0, 10],
                    True,
                    (inputs[0], inputs[1], hidden),
                )
                return window.cpu(), backend.cache.cpu()

        sleep_wake(saver, retained + list(model.parameters()), run)
    print(
        f"PASS {'split' if split else 'unsplit'} Eagle: capture, static growth, live values, sleep/wake"
    )


def dflash(saver):
    from tokenspeed.runtime.execution.drafter.dflash import DFlash

    # Only the model/head/process-group environment is absent. Exercise the
    # real lazy allocator, including dtype/capacity replacement and retained views.
    drafter = DFlash.__new__(DFlash)
    drafter.input_buffers = SimpleNamespace(max_bs=2)
    drafter.logits_processor = SimpleNamespace(tp_size=2)
    drafter.spec_num_tokens = 4
    drafter.token_to_kv_pool = EmptyPool(Arena(saver))
    drafter._greedy_gathered_max = drafter._greedy_gathered_ids = None
    drafter._greedy_gather_cap = 0
    retained, captures = [], []
    for bs, dtype in ((2, torch.int32), (4, torch.int64)):
        drafter.input_buffers.max_bs = bs
        maximum, ids = drafter._ensure_greedy_gather_buffers(
            torch.float32, dtype, torch.device("cuda")
        )
        retained.extend((maximum[:], ids[:]))
        with saver.named_region("executor_inputs"):
            source = torch.zeros_like(maximum)
        cap = BreakableCapture(memory_saver_adapter=saver)
        with cap:
            output = source + maximum + ids
        captures.append((cap, source, maximum, ids, output))

        def run(cycle):
            results = []
            for capture, source, maximum, ids, output in captures:
                source.fill_(cycle + 1)
                maximum.fill_(cycle + 2)
                ids.fill_(cycle + 3)
                capture.replay()
                result = output.cpu()
                assert torch.equal(result, torch.full_like(result, 3 * cycle + 6))
                results.append(result)
            return results

        sleep_wake(saver, retained, run)
    print(
        "PASS DFlash: lazy TP scratch growth, retained views, graph replay, sleep/wake"
    )


def make_prefill(saver, inner, backend):
    from tokenspeed.runtime.execution.prefill_graph import PrefillGraph

    config = replace(prefill_config(), query_shard_size=1, query_shard_rank=0)
    with saver.named_region("executor_inputs"):
        buffers = InputBuffers(2, 32, 2, device="cuda")
    with saver.named_region("runtime_states"):
        model = ToyModel()
        model.model = inner
    return PrefillGraph(
        model_runner=ToyRunner(model, ToyModelConfig()),
        attn_backend=backend,
        token_to_kv_pool=EmptyPool(Arena(saver)),
        input_buffers=buffers,
        config=config,
        memory_saver_adapter=saver,
        num_warmup=1,
    )


def narrowing(saver):
    from tokenspeed.runtime.models.deepseek_v41 import DeepseekV41Model, V41RowState

    class Inner(DeepseekV41Model):
        """Tiny row-wise stages with the production V4.1 static allocator."""

        max_decoder_rows_per_request = 4

        def __init__(self):
            torch.nn.Module.__init__(self)
            self.embed_tokens = torch.nn.Embedding(4, 16, device="cuda")
            self.config = SimpleNamespace(hidden_size=16, hc_mult=2)
            self.decoder_uses_engram = True
            self.engram_hash = SimpleNamespace(
                offsets=torch.zeros((1, 2), device="cuda", dtype=torch.int64)
            )
            self.dspark_capture_layers = (0,)
            self.ced_decoder_start = 0

        def encoder_forward(self, input_ids, positions, ctx, *, input_embeds):
            rows = positions.shape[0]
            return V41RowState(
                input_embeds[:, None, :].repeat(1, 2, 1),
                input_embeds[:, :2] + 1,
                positions + 1,
                None,
                torch.ones((rows, 1, 2), dtype=torch.int64, device="cuda"),
                torch.ones(rows, dtype=torch.bool, device="cuda"),
                [input_embeds * 2],
            )

        def decoder_rows(self, ctx):
            return min(ctx.input_num_tokens, ctx.bs * self.max_decoder_rows_per_request)

        def narrowing_forward(self, state, ctx):
            return state.leading(self.decoder_rows(ctx))

        def decoder_forward(self, state, ctx):
            hidden = state.hidden.sum(1) + state.pre_mix.sum(1)[:, None]
            hidden = (
                hidden + state.positions[:, None] + state.hashes.sum((1, 2))[:, None]
            )
            return hidden * state.engram_token_mask[:, None], state.captured

        def finish_forward(self, hidden, captured, ctx):
            return hidden, captured

    with saver.named_region("runtime_states"):
        inner = Inner()
    graph = make_prefill(saver, inner, NoAttention())
    graph.capture(entries=None, observer=NULL_MEMORY_DELTA_OBSERVER)
    contexts = {bucket: graph.make_dummy_batch(bucket, 1) for bucket in graph._encoders}
    persistent = [graph._input_embeds_buf, *graph._outputs]
    for decoder in graph._decoders.values():
        persistent.extend(device_tensors(decoder.statics))

    def run(cycle):
        graph._input_embeds_buf.fill_(cycle + 1)
        results = []
        for bucket, encoder in graph._encoders.items():
            ctx = contexts[bucket]
            with active_forward(ctx):
                encoder.capture.replay()
                state = inner.narrowing_forward(encoder.state, ctx)
                decoder = graph._decoders[graph._decoder_bucket(state.rows)]
                state.land_into(decoder.statics)
                decoder.capture.replay(valid_rows=state.rows)
            output = decoder.output.hidden_states[: state.rows].cpu()
            expected = 4 * (cycle + 1) + torch.arange(1, state.rows + 1)[:, None] + 4
            assert torch.equal(output, expected.expand_as(output))
            results.append(output)
        return results

    sleep_wake(saver, persistent, run)
    print(
        "PASS narrowing PrefillGraph: production V4.1 statics, optional fields, replay, sleep/wake"
    )


def capacity(saver):
    from tokenspeed.runtime.layers.attention.backends.state.kda import KdaAttnBackend
    from tokenspeed.runtime.layers.attention.backends.state.mamba import (
        MambaForwardMetadata,
    )

    class Backend(KdaAttnBackend):
        """A weight-free state consumer using production capacity/checkpoint allocation."""

        _capacity_layout_when_uncaptured = True

        def __init__(self):
            self.cache_pool = EmptyPool(Arena(saver))
            self.device = "cuda"
            self.forward_metadata = None
            self._state_group_ids = ("state",)
            self.states = torch.zeros(4, 2, device="cuda")
            self._prefill_metadata = {}
            self.init_prefill_graph_state(8, 2)

        def _admits_capacity_prefill(self):
            return True

        def _capacity_checkpoint_interval(self):
            return 4

        def _state_layer_ids(self):
            return [0]

        def _state_components(self, layer):
            return None, self.states

        def _refresh_captured_prefill(self, metadata):
            pass

        def init_forward_metadata(
            self,
            *,
            bs,
            extend_seq_lens,
            extend_seq_lens_cpu,
            req_pool_indices,
            **kwargs,
        ):
            bounds = torch.cat(
                (
                    torch.zeros(1, dtype=torch.int32, device="cuda"),
                    extend_seq_lens[:bs].cumsum(0).int(),
                )
            )
            self.forward_metadata = MambaForwardMetadata(
                query_start_loc=bounds,
                scan_query_start_loc=bounds.long(),
                query_start_loc_int64=bounds.long(),
                extend_seq_lens_cpu=extend_seq_lens_cpu[:bs].clone(),
                cu_extend_seq_lens_cpu=torch.cat(
                    (
                        torch.zeros(1, dtype=torch.int64),
                        extend_seq_lens_cpu[:bs].long().cumsum(0),
                    )
                ),
                req_pool_indices=req_pool_indices[:bs],
                state_in_blocks_by_group={
                    "state": torch.zeros(bs, dtype=torch.int32, device="cuda")
                },
                state_out_blocks_by_group={
                    "state": torch.ones(bs, dtype=torch.int32, device="cuda")
                },
            )

    class Inner(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = torch.nn.Embedding(4, 16, device="cuda")

        def forward(self, input_ids, positions, ctx, *, input_embeds):
            metadata = ctx.attn_backend.forward_metadata
            return input_embeds + metadata.query_start_loc[-1], None

    with saver.named_region("runtime_states"):
        backend, inner = Backend(), Inner()
    graph = make_prefill(saver, inner, backend)
    graph.capture(entries=None, observer=NULL_MEMORY_DELTA_OBSERVER)
    retained = list(backend._prefill_metadata.values())
    assert len(retained) == 2
    contexts = {
        bucket: graph.make_dummy_batch(bucket, 1) for bucket, _ in graph._captures
    }
    # Grow checkpoint scratch through the backend's ordinary uncaptured-shape
    # entry while earlier retained views still reference the old allocation.
    with saver.named_region("attention_state"):
        backend.init_forward_metadata(
            bs=1,
            extend_seq_lens=torch.tensor([64], device="cuda"),
            extend_seq_lens_cpu=torch.tensor([64]),
            req_pool_indices=graph.input_buffers.req_pool_indices_buf[:1],
        )
    backend.prepare_prefill_metadata(64, 1, ForwardMode.EXTEND, capture=False)
    retained.append(backend.forward_metadata)
    persistent = device_tensors(retained) + [graph._input_embeds_buf, *graph._outputs]

    def run(cycle):
        graph._input_embeds_buf.fill_(cycle + 1)
        results = []
        for (bucket, bs), (capture, output) in graph._captures.items():
            if bs is None:
                # This consumer reads retained metadata inline; the production
                # ordinary fallback instead runs attention at an eager break.
                continue
            with saver.named_region("attention_state"):
                graph.make_dummy_batch(bucket, bs)
            backend.prepare_prefill_metadata(
                bucket, bs, ForwardMode.EXTEND, capture=False
            )
            with active_forward(contexts[bucket]):
                capture.replay(valid_rows=bucket)
            result = output.hidden_states.cpu()
            torch.testing.assert_close(
                result, torch.full_like(result, cycle + 1 + bucket), rtol=0, atol=0
            )
            results.append(result)
        return results

    sleep_wake(saver, persistent, run)
    print(
        "PASS recurrent PrefillGraph: retained metadata, checkpoint growth, replay, sleep/wake"
    )


if __name__ == "__main__":
    saver = TorchMemorySaverAdapter.create(True, device="cuda")
    with torch.inference_mode():
        case = sys.argv[1]
        if case in ("eagle", "split_eagle"):
            eagle(saver, split=case == "split_eagle")
        else:
            {"dflash": dflash, "narrowing": narrowing, "capacity": capacity}[case](
                saver
            )
