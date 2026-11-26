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

"""KDA retained decode records over the execution-slot lifecycle.

The real C++ scheduler plans every round: PD decode landings, decode permits,
finishes, aborts, forced retractions with their state folds, restores, and
fused-role mixed batches. The real KDA backend runs every forward over the
production cache pool, and the real executor stages each round's rows
(``InputBuffers`` through ``ModelExecutor._fill_batch_inputs``), clearing a
fresh execution slot's count before the forward, as it does in serving. The
retained records are keyed by execution slot (``1..max_bs``: the decode permit
on a PD decode node, the request row on every other role), so a decode node's
resident requests without a permit hold no records.

Every request reads its own input stream, and every round is checked against
the independent tokenwise oracle of ``test_kda_state_eviction_spec``: the
round's accepted outputs and, through the slot's records, each request's
logical state. Each round also checks every slot's accepted count, so records
kept by request row, a count left to a slot's next owner, a fold that skipped
the victim's slot, or records sized by resident requests fail here.
"""

from __future__ import annotations

import os
import sys
from dataclasses import replace

# The CI runner executes each registered test file as a standalone script.
_TEST_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _TEST_DIR)
sys.path.insert(0, os.path.dirname(_TEST_DIR))
from test.runtime.conftest import KIMI_STATE_GROUPS, block_tables_for, requires_cuda
from test.runtime.test_kda_state_eviction_spec import (
    _COMPONENTS,
    _D,
    _FA,
    _H,
    _LAYERS,
    _P,
    _WIDTH,
    _assert_states,
    _reference,
    _transfer_kda_pool,
)
from test.runtime.test_kimi_k3_kda import (
    _assert_close,
    _backend_config,
    _logical_checkpoint,
    requires_fla,
)

import numpy as np
import pytest
import torch
from ci_system.ci_register import register_cuda_ci
from tokenspeed_kernel.platform import current_platform

ts = pytest.importorskip("tokenspeed_engine.scheduler")

from test.runtime.scheduler_driver import (  # noqa: E402
    advance_scheduler,
    make_abort_event,
    make_extend_result_event,
    make_finish_event,
    make_update_reserve_tokens_event,
)

from tokenspeed.runtime.cache.transfer.ops import RestoreOp, SnapshotOp  # noqa: E402
from tokenspeed.runtime.engine.scheduler_utils import (  # noqa: E402
    cache_ops_from_plan,
    pool_to_cache_groups,
)
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode  # noqa: E402
from tokenspeed.runtime.execution.input_buffer import InputBuffers  # noqa: E402
from tokenspeed.runtime.execution.model_executor import ModelExecutor  # noqa: E402
from tokenspeed.runtime.execution.request_slots import RequestSlotLayout  # noqa: E402
from tokenspeed.runtime.execution.runtime_states import RuntimeStates  # noqa: E402
from tokenspeed.runtime.execution.types import StopCriteria  # noqa: E402
from tokenspeed.runtime.layers.attention.backends.state.kda import (  # noqa: E402
    KdaAttnBackend,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import (  # noqa: E402
    STATE_LIVE_SLOT,
)

register_cuda_ci(est_time=240, suite="runtime-1gpu")

pytestmark = [requires_cuda, requires_fla]

Role = ts.SchedulerConfig.Role

# Tokens a prefill node computed before a decode node's landing.
_PROMPT = 5
# Input rows of each request's stream, the oracle's horizon.
_POSITIONS = 48
_USABLE_PAGES = 96
_MAX_NEW_TOKENS = 32
_VOCAB = 1024
_MAX_TOKENS = 64
# The scheduler's own decode round never ends a request.
_NO_STOP = StopCriteria((), 1 << 20)


class _Request:
    """A request's input stream, its tokenwise oracle and its progress."""

    def __init__(self, rid: str, admission: int, params) -> None:
        self.rid = rid
        self.admission = admission
        generator = torch.Generator(device="cpu").manual_seed(4099 + 31 * admission)
        self.stream = {
            name: torch.randn((_POSITIONS, columns), generator=generator).to(
                "cuda", torch.bfloat16
            )
            for name, columns in (
                ("mixed_qkv", 3 * _H * _D),
                ("f_a_out", _FA),
                ("beta_raw", _H),
            )
        }
        # outputs[layer][position] and states[tokens][(layer, component)].
        self.outputs, self.states = _reference(self.stream, _POSITIONS, params)
        # Tokens whose state the request's pages (and records) hold.
        self.consumed = 0
        self.row: int | None = None
        # The State pages a decode node's landing wrote, per group.
        self.landing: dict[str, int] | None = None


class _Graph:
    """A decode CUDA graph captured at ``bucket`` rows."""

    def __init__(self, bucket, graph, inputs, outputs) -> None:
        self.bucket = bucket
        self.graph = graph
        self.inputs = inputs
        self.outputs = outputs


def _executor(backend, *, max_bs: int, residents: int) -> ModelExecutor:
    """The executor's input staging over its real buffers, the backend bound
    to them by the executor's own binding step; nothing below the attention
    backend runs."""
    executor = ModelExecutor.__new__(ModelExecutor)
    # ModelExecutorConfig.device: a landing's lengths are staged onto it.
    executor.device = "cuda"
    executor.attn_backend = backend
    executor.device_module = torch.cuda
    executor.default_stream = torch.cuda.current_stream()
    executor.execution_stream = executor.default_stream
    padding = RequestSlotLayout(residents).padding
    executor.input_buffers = InputBuffers(
        max_bs=max_bs,
        max_num_tokens=_MAX_TOKENS,
        state_write_padding_pool_index=padding,
        device="cuda",
    )
    executor.runtime_states = RuntimeStates(
        req_pool_size=padding,
        input_vocab_size=_VOCAB,
        vocab_size=_VOCAB,
        output_length=_WIDTH,
        device="cuda",
    )
    executor.request_ends = torch.zeros(max_bs, dtype=torch.int32, device="cuda")
    executor._bind_forward_rows()
    return executor


class _Node:
    """One engine rank: the scheduler, the KDA backend over the production
    pool, the executor's row staging and, with forced retractions, the Host
    cache executor that stores and restores retraction images."""

    def __init__(
        self,
        role,
        *,
        max_bs: int,
        residents: int,
        mixed: bool = False,
        retraction_interval: int | None = None,
    ) -> None:
        self.role = role
        self.max_bs = max_bs
        self.width = _WIDTH
        self.pool = _transfer_kda_pool(_USABLE_PAGES, torch.float32)
        self.contract = self.pool.arena.runtime_contract
        config, spec = _backend_config("cuda", spec_tokens=self.width)
        config = replace(
            config,
            max_bs=max_bs,
            max_resident_requests=residents,
            pd_disaggregation_enabled=role == Role.D,
        )
        self.backend = KdaAttnBackend(
            config,
            spec,
            enable_prefill_graph=False,
            kda_backend="auto" if current_platform().is_amd else "fla",
        )
        self.backend.set_kv_pool(self.pool)
        self.executor = _executor(self.backend, max_bs=max_bs, residents=residents)
        self.backend.init_cuda_graph_state(max_bs=max_bs)
        self.backend.preallocate_verify_workspace(max_bs, self.width)
        generator = torch.Generator(device="cpu").manual_seed(107)

        def random_tensor(shape, dtype, scale):
            return torch.randn(shape, generator=generator).to("cuda", dtype) * scale

        self.params = {
            layer: {
                "conv_weights": random_tensor((3 * _H * _D, 4), torch.bfloat16, 0.1),
                "f_b_weight": random_tensor((_H * _D, _FA), torch.bfloat16, 0.05),
                "A_log": random_tensor((_H,), torch.float32, 0.1),
                "dt_bias": random_tensor((_H * _D,), torch.float32, 0.1),
            }
            for layer in _LAYERS
        }
        self.padding = RequestSlotLayout(max_bs).padding
        # Every execution slot's accepted count as the backend must hold it.
        self.counts = [0] * RequestSlotLayout(max_bs).capacity
        self.requests: dict[str, _Request] = {}
        self.host = None
        if retraction_interval is not None:
            from tokenspeed.runtime.cache.l2.executor import HostCacheExecutor
            from tokenspeed.runtime.cache.l2.sizing import RetractionPoolRequest

            self.host = HostCacheExecutor(
                self.pool,
                draft_pool=None,
                l2_tier=False,
                host_ratio=0,
                host_size_gb=0,
                snapshot_pool=RetractionPoolRequest(
                    host_gb=0,
                    ratio=1,
                    max_retracted_requests=1,
                    tail_lcm_blocks_per_request=0,
                ),
                slot_state_exporters=(self.backend,),
                io_backend="direct",
                attn_tp_rank=0,
                kvp_rank=0,
                nprocs_per_node=1,
                world_cpu_group=None,
                block_alignment=1,
            )
        self.scheduler = ts.Scheduler(
            self._scheduler_config(residents, mixed, retraction_interval)
        )

    def _scheduler_config(self, residents, mixed, retraction_interval):
        config = ts.SchedulerConfig()
        config.role = self.role
        config.prefix_granularity = _P
        config.num_device_pages = _USABLE_PAGES + 1
        config.num_host_pages = 0
        config.max_scheduled_tokens = 32
        config.max_batch_size = self.max_bs
        config.max_resident_requests = residents
        config.decode_input_tokens = self.width
        config.overlap_schedule_depth = 0
        config.enable_mixed_prefill_decode = mixed
        config.disable_l2_cache = True
        config.disable_prefix_cache = True
        # As the data plane configures them: the scheduler lists a retraction
        # victim's live blocks in the groups whose decode state is deferred.
        deferred = self.backend.deferred_decode_state_groups()
        groups = pool_to_cache_groups(self.pool)
        for group in groups:
            group.deferred_decode_state = group.group_id in deferred
            if self.role == Role.D:
                group.transfer_policy = (
                    ts.CacheTransferPolicy.LatestSnapshot
                    if group.group_id in deferred
                    else ts.CacheTransferPolicy.FullSuffix
                )
        config.cache_groups = groups
        if self.host is None:
            config.num_snapshot_pages = 1  # the null page alone: no pool
            config.max_retracted_requests = 0
        else:
            config.num_snapshot_pages = self.host.num_snapshot_pages
            config.max_retracted_requests = self.host.max_retracted_requests
            config.host_group_packing = list(self.host.host_group_packing)
            config.debug_force_retraction_interval = retraction_interval
            config.debug_force_retraction_max_per_request = 1
        return config

    def close(self) -> None:
        if self.host is not None:
            self.host.shutdown()

    # ------------------------------------------------------------ requests

    def submit(self, *rids: str, prompt: int = _PROMPT) -> None:
        specs = []
        for rid in rids:
            request = _Request(rid, len(self.requests) + 1, self.params)
            self.requests[rid] = request
            spec = ts.RequestSpec()
            spec.request_id = rid
            spec.tokens = [100 * request.admission + i for i in range(prompt)]
            spec.max_new_tokens = _MAX_NEW_TOKENS
            specs.append(spec)
        self.scheduler.submit_requests(specs)
        if self.role == Role.D:
            advance_scheduler(
                self.scheduler, [ts.PD.BootstrappedEvent(rid) for rid in rids]
            )

    def admit(self, rid: str):
        """Plan the next round, which admits ``rid``'s landing; returns the
        round's forward batch, if any."""
        plan = self.scheduler.next_execution_plan()
        landing = plan.remote_prefill
        assert landing is not None and list(landing.request_ids) == [rid]
        return self.execute(plan)

    def land(self, *rids: str) -> None:
        """The prefill node's transfer lands: the prompt's State in the pages
        the admission named, then the scheduler hears the request landed."""
        for rid in rids:
            request = self.requests[rid]
            for layer in _LAYERS:
                page = request.landing[self.pool.state_group_by_layer[layer]]
                for component in _COMPONENTS:
                    self.pool.get_component(layer, component)[page].copy_(
                        request.states[_PROMPT][layer, component]
                    )
            request.consumed = _PROMPT
            self.executor.mark_remote_cache_ready(request.row)
            advance_scheduler(self.scheduler, [ts.PD.RemotePrefillDoneEvent(rid, 42)])

    def finish(self, *rids: str) -> None:
        advance_scheduler(self.scheduler, [make_finish_event(rid) for rid in rids])

    def abort(self, *rids: str) -> None:
        advance_scheduler(self.scheduler, [make_abort_event(rid) for rid in rids])

    # ------------------------------------------------------------ plans

    def plan(self):
        """Plan and execute the next round's device work; returns its batch."""
        return self.execute(self.scheduler.next_execution_plan())

    def execute(self, plan, *, after_folds=None):
        """The plan's device work in the data plane's order (``DeviceHandle``):
        the retraction folds, the image stores, the copies and zeroing, the
        restores, then a landing's slots. ``after_folds`` runs right after the
        folds, before the zeroing may reuse a victim's freed pages. Returns
        the plan's forward batch."""
        stream = self.executor.execution_stream
        folds = {
            group: torch.from_numpy(rows.copy())
            for group, rows in plan.state_folds_arrays().items()
            if len(rows)
        }
        if folds:
            self.backend.fold_deferred_state(folds, stream=stream)
            for rows in folds.values():
                for slot in rows[:, 2].tolist():
                    if slot >= 0:
                        self.counts[slot] = 0
            torch.cuda.synchronize()
            self.assert_counts("retraction folds")
        if after_folds is not None:
            after_folds()
        operations = []
        if self.host is not None:
            operations = cache_ops_from_plan(
                plan, request_admission=lambda rid: self.requests[rid].admission
            ).device
            self.host.submit_write_backs(
                operations, prerequisite_stream=stream, fence_stream=stream
            )
        else:
            assert not list(plan.cache)
        copies = {
            group: pairs
            for group, pairs in plan.pages_to_copy_arrays().items()
            if len(pairs)
        }
        if copies:
            self.pool.arena.copy_blocks(copies)
        zeros = {
            group: np.asarray(pages, dtype=np.int64)
            for group, pages in plan.pages_to_zero.items()
            if len(pages)
        }
        if zeros:
            self.pool.zero_new_blocks(zeros)
        if self.host is not None:
            self.host.submit_load_backs(operations, prerequisite_stream=stream)
        landing = plan.remote_prefill
        if landing is not None:
            self._admit_landing(landing)
        # The Host executor's transfer streams have landed before their
        # completion events reach the scheduler.
        torch.cuda.synchronize()
        if self.host is not None:
            acknowledgments = self.host.poll_results()
            if acknowledgments:
                advance_scheduler(self.scheduler, acknowledgments)
        self.last_operations = operations
        self.assert_counts("plan's device work")
        batches = [batch for batch in plan.forward if batch.request_ids]
        assert len(batches) <= 1
        return batches[0] if batches else None

    def _admit_landing(self, landing) -> None:
        """A landing row holds no execution slot; its request row starts a new
        owner (``DeviceHandle._prepare_remote_prefill_slots``)."""
        rows = list(landing.request_pool_indices)
        assert landing.num_extends() == len(rows)
        assert list(landing.execution_slots) == [-1] * len(rows)
        assert list(landing.execution_slot_fresh) == [False] * len(rows)
        tables = dict(landing.block_tables)
        for index, rid in enumerate(landing.request_ids):
            request = self.requests[rid]
            request.row = rows[index]
            request.landing = {
                group: tables[group][index][STATE_LIVE_SLOT]
                for group in KIMI_STATE_GROUPS
            }
        self.executor.prepare_remote_cache_slots(rows)
        self.executor.reset_remote_prefill_cache_lengths(landing)

    # ------------------------------------------------------------ rounds

    def expect(self, batch, **rows) -> None:
        """``rid=(request row, execution slot or a set of them, fresh)`` for
        exactly the batch's rows."""
        assert batch is not None
        got = {
            rid: (row, slot, fresh)
            for rid, row, slot, fresh in zip(
                batch.request_ids,
                batch.request_pool_indices,
                batch.execution_slots,
                batch.execution_slot_fresh,
                strict=True,
            )
        }
        assert set(got) == set(rows), got
        for rid, (row, slot, fresh) in rows.items():
            got_row, got_slot, got_fresh = got[rid]
            slots = slot if isinstance(slot, set) else {slot}
            assert (got_row, got_fresh) == (row, fresh) and got_slot in slots, (
                rid,
                got[rid],
            )

    def round(self, batch, accepted=None, *, ending=(), graph: _Graph | None = None):
        """One forward in the executor's order, checked against the oracle.

        ``accepted`` maps each decode row to its accepted tokens; ``ending``
        names decode rows whose request this round ends on the device (the
        sampling epilogue marks them stopped before the commit).
        """
        accepted = dict(accepted or {})
        ids = list(batch.request_ids)
        rows = list(batch.request_pool_indices)
        slots = list(batch.execution_slots)
        fresh = list(batch.execution_slot_fresh)
        lengths = list(batch.input_lengths)
        bs, num_extends = len(ids), batch.num_extends()
        requests = [self.requests[rid] for rid in ids]
        assert set(accepted) == set(ids[num_extends:]), (accepted, ids)
        # Distinct execution slots in 1..max_bs.
        assert sorted(set(slots)) == sorted(slots), slots
        assert all(1 <= slot <= self.max_bs for slot in slots), slots
        for request, row in zip(requests, rows, strict=True):
            assert request.row in (None, row)
            request.row = row
        for request, length in zip(requests, lengths, strict=True):
            assert request.consumed + length <= _POSITIONS
        # The forward's prologue (ModelExecutor.execute_forward_op): a new
        # admission on a row rebinds it, then the batch's inputs are staged
        # and its fresh execution slots cleared, on the execution stream.
        runtime = self.executor.runtime_states
        stream = self.executor.execution_stream
        rebound = runtime.bind_stop_criteria(
            [request.admission for request in requests], rows, [_NO_STOP] * bs
        )
        if rebound is not None:
            self.backend.reset_request_workspace(rebound, stream=stream)
        for request, row in zip(requests, rows, strict=True):
            runtime.valid_cache_lengths[row] = request.consumed
        self.executor._fill_batch_inputs(batch, sum(lengths))
        buffers = self.executor.input_buffers
        assert buffers.execution_slots_buf.tolist() == slots + [self.padding] * (
            self.max_bs - bs
        )
        assert buffers.fresh_execution_slots().tolist() == [
            slot for slot, is_fresh in zip(slots, fresh) if is_fresh
        ]
        for slot, is_fresh in zip(slots, fresh, strict=True):
            if is_fresh:
                self.counts[slot] = 0
        self.assert_counts("fresh-slot reset")

        tables = {
            group: np.asarray(
                [list(row) for row in batch.block_tables[group]], dtype=np.int32
            )
            for group in KIMI_STATE_GROUPS
        }
        delivered = block_tables_for(self.contract, tables, "cuda")
        inputs = {
            name: torch.cat(
                [
                    request.stream[name][request.consumed : request.consumed + length]
                    for request, length in zip(requests, lengths, strict=True)
                ]
            )
            for name in requests[0].stream
        }
        if graph is not None:
            outputs = self._replay(graph, bs, delivered, inputs)
        elif num_extends:
            self.backend.init_forward_metadata(
                bs,
                num_extends,
                buffers.req_pool_indices_buf[:bs],
                buffers.seq_lens_buf[:bs],
                ForwardMode.from_num_extends(num_extends, bs),
                block_tables=delivered,
                extend_seq_lens=buffers.extend_seq_lens_buf[:num_extends],
                extend_seq_lens_cpu=buffers.extend_seq_lens_cpu[:num_extends],
                extend_prefix_lens=buffers.extend_prefix_lens_buf[:num_extends],
                extend_prefix_lens_cpu=buffers.extend_prefix_lens_cpu[:num_extends],
                extend_replay_lens_cpu=buffers.extend_replay_lens_cpu[:num_extends],
                extend_prompt_lens_cpu=buffers.extend_prompt_lens_cpu[:num_extends],
                extend_with_prefix=bool(
                    buffers.extend_prefix_lens_cpu[:num_extends].any()
                ),
                query_shard=None,
            )
            outputs = self._layers(
                inputs, bs, ForwardMode.from_num_extends(num_extends, bs)
            )
        else:
            self.backend.refresh_decode_metadata(
                bs,
                bs,
                buffers.req_pool_indices_buf[:bs],
                buffers.seq_lens_buf[:bs],
                forward_mode=ForwardMode.DECODE,
                block_tables=delivered,
                for_graph_replay=False,
            )
            outputs = self._layers(inputs, bs, ForwardMode.DECODE)

        if num_extends < bs:
            for index in range(num_extends, bs):
                if ids[index] in ending:
                    runtime.stopped[rows[index]] = True
            self.backend.commit_state_after_acceptance(
                torch.tensor(
                    [accepted.get(rid, 1) for rid in ids],
                    dtype=torch.int32,
                    device="cuda",
                ),
                num_extends=num_extends,
                accepted_path=None,
            )
        capacity = self.backend._retained_capacity
        for index, (rid, slot) in enumerate(zip(ids, slots, strict=True)):
            if index < num_extends or rid in ending:
                # A prefill stores a dense state; an ended request's commit
                # folds its records into its live page.
                self.counts[slot] = 0
            else:
                # A round accepts one token at least; a window that would
                # overflow the history folds the records first.
                old = self.counts[slot]
                self.counts[slot] = (0 if old + self.width > capacity else old) + (
                    accepted[rid]
                )
        torch.cuda.synchronize()
        self.assert_counts(f"round of {ids}")

        offset = 0
        for index, (request, length, slot) in enumerate(
            zip(requests, lengths, slots, strict=True)
        ):
            begin = request.consumed
            kept = length if index < num_extends else accepted[request.rid]
            end = begin + kept
            for layer in _LAYERS:
                _assert_close(
                    outputs[layer][offset : offset + kept],
                    request.outputs[layer][begin:end],
                    f"{request.rid} layer {layer} outputs {begin}:{end}",
                )
            live = {
                group: tables[group][index][STATE_LIVE_SLOT]
                for group in KIMI_STATE_GROUPS
            }
            _assert_states(
                self.states(live, slot),
                request.states[end],
                f"{request.rid} on execution slot {slot} at {end} tokens",
            )
            request.consumed = end
            offset += length

        for index, request in enumerate(requests):
            if index < num_extends:
                # Every prompt here fits one chunk, whose result is a token.
                assert (
                    batch.extend_prefix_lens[index] + lengths[index]
                    == batch.prefill_lengths[index]
                )
                events = [make_extend_result_event(request.rid, [7])]
            else:
                kept = accepted[request.rid]
                events = [
                    make_extend_result_event(request.rid, [7] * kept),
                    make_update_reserve_tokens_event(request.rid, kept),
                ]
            advance_scheduler(self.scheduler, events)
        return {rid: slot for rid, slot in zip(ids, slots, strict=True)}

    def _layers(self, inputs, bs: int, mode) -> dict[int, torch.Tensor]:
        outputs = {}
        for layer in _LAYERS:
            kwargs = dict(
                layer=None,
                token_to_kv_pool=self.pool,
                bs=bs,
                mixed_qkv=inputs["mixed_qkv"].clone(),
                f_a_out=inputs["f_a_out"],
                beta_raw=inputs["beta_raw"],
                g_raw=None,
                bias=None,
                activation="silu",
                key_dim=_H * _D,
                value_dim=_H * _D,
                attention_tp_size=1,
                head_k_dim=_D,
                head_v_dim=_D,
                lower_bound=-5.0,
                layer_id=layer,
                seq_len=inputs["mixed_qkv"].shape[0],
                a=None,
                b=None,
                **self.params[layer],
            )
            if mode.is_decode():
                output = self.backend.forward_decode(None, None, None, **kwargs)
            else:
                output = self.backend.forward_extend(
                    None, None, None, forward_mode=mode, save_kv_cache=True, **kwargs
                )
            outputs[layer] = output.reshape(-1, _H, _D).clone()
        return outputs

    def capture(self, bucket: int) -> _Graph:
        """Capture a decode graph at ``bucket`` rows the way the forward
        runner does: dummy inputs, every execution slot padding, a warmup
        forward on the capture stream, then the recorded forward."""
        buffers = self.executor.input_buffers
        buffers.fill_dummy_decode_buffers(
            batch_size=bucket, total_tokens=bucket * self.width
        )
        assert buffers.execution_slots_buf.tolist() == [self.padding] * self.max_bs
        self.backend.init_forward_metadata_capture_cuda_graph(
            bucket,
            buffers.req_pool_indices_buf[:bucket],
            buffers.seq_lens_buf[:bucket],
            ForwardMode.DECODE,
            block_tables={
                group: torch.full((bucket, 2), -1, dtype=torch.int32, device="cuda")
                for group in KIMI_STATE_GROUPS
            },
        )
        generator = torch.Generator(device="cpu").manual_seed(8191)
        inputs = {
            name: torch.randn(
                (bucket * self.width, tensor.shape[1]), generator=generator
            ).to("cuda", torch.bfloat16)
            for name, tensor in next(iter(self.requests.values())).stream.items()
        }
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            self._layers(inputs, bucket, ForwardMode.DECODE)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outputs = self._layers(inputs, bucket, ForwardMode.DECODE)
        return _Graph(bucket, graph, inputs, outputs)

    def _replay(self, graph: _Graph, bs: int, delivered, inputs):
        """Refresh the captured buffers for ``bs`` live rows and replay; the
        padding rows past them take large random inputs."""
        assert bs <= graph.bucket
        buffers = self.executor.input_buffers
        self.backend.refresh_decode_metadata(
            graph.bucket,
            bs,
            buffers.req_pool_indices_buf[:bs],
            buffers.seq_lens_buf[:bs],
            forward_mode=ForwardMode.DECODE,
            for_graph_replay=True,
            block_tables=delivered,
        )
        live = bs * self.width
        for name, value in inputs.items():
            graph.inputs[name][:live].copy_(value)
            graph.inputs[name][live:].normal_(0, 4)
        graph.graph.replay()
        return {layer: output[:live].clone() for layer, output in graph.outputs.items()}

    # ------------------------------------------------------------ state

    def assert_counts(self, what: str) -> None:
        counts = self.backend._replay_length[:, 0].tolist()
        assert counts == self.counts, f"{what}: counts {counts}, want {self.counts}"

    def states(self, pages: dict[str, int], slot: int):
        """The pages' dense states with ``slot``'s accepted records applied."""
        return {
            (layer, component): (
                _logical_checkpoint(self.backend, layer, page, slot)
                if component == "recurrent_state"
                else self.pool.get_component(layer, component)[page].clone()
            )
            for layer in _LAYERS
            for page in (pages[self.pool.state_group_by_layer[layer]],)
            for component in _COMPONENTS
        }

    def dense_states(self, pages: dict[str, int]):
        """The pages' dense states alone (bit-exact copies)."""
        return {
            (layer, component): self.pool.get_component(layer, component)[
                pages[self.pool.state_group_by_layer[layer]]
            ].clone()
            for layer in _LAYERS
            for component in _COMPONENTS
        }


def _live_pages(batch, rid: str) -> dict[str, int]:
    index = list(batch.request_ids).index(rid)
    return {
        group: batch.block_tables[group][index][STATE_LIVE_SLOT]
        for group in KIMI_STATE_GROUPS
    }


def _folds(plan) -> dict[str, list[list[int]]]:
    """The plan's state folds: (request row, live page, execution slot)."""
    return {
        group: rows.tolist()
        for group, rows in plan.state_folds_arrays().items()
        if len(rows)
    }


def _assert_bitwise(actual, expected, what: str) -> None:
    for key, value in expected.items():
        assert torch.equal(
            actual[key].view(torch.uint8), value.view(torch.uint8)
        ), f"{what}: {key} changed"


def _assert_stale_records_corrupt(node: _Node, request: _Request, slot: int) -> None:
    """Applying ``slot``'s still-counted records to ``request``'s landed
    state would leave the oracle: the fresh reset the test relies on is not
    vacuous."""
    assert node.counts[slot] > 0
    stale = node.states(request.landing, slot)
    with pytest.raises(AssertionError):
        _assert_states(stale, request.states[request.consumed], "stale records")


def test_landings_without_permits_keep_their_state_until_their_first_decode():
    """A decode node with two permits and two extra resident slots. Two
    requests decode on permits 1 and 2 from request rows 3 and 4; two more
    are admitted into rows 1 and 2 -- the indices of those permits -- and
    land while both permits are held. Neither admission nor landing touches
    any record or count: the decoders continue exactly, and the waiting
    requests' landed states are bit for bit what the transfer wrote when they
    finally decode on the permits the decoders release."""
    node = _Node(Role.D, max_bs=2, residents=4)
    node.submit("p", "q", "c", "d")
    for rid in ("p", "q", "c", "d"):
        assert node.admit(rid) is None
    assert [node.requests[rid].row for rid in "pqcd"] == [1, 2, 3, 4]
    node.land("p", "q")
    batch = node.plan()
    node.expect(batch, p=(1, 1, True), q=(2, 2, True))
    node.round(batch, {"p": 3, "q": 2})
    # Host-decided finishes leave the permits' records counted.
    node.finish("p", "q")
    assert node.counts[1:3] == [3, 2]

    node.land("c", "d")
    batch = node.plan()
    node.expect(batch, c=(3, 1, True), d=(4, 2, True))
    node.round(batch, {"c": 2, "d": 3})
    node.submit("a", "b")
    # Each admission resets its request row (1, then 2) while the decoders
    # keep their records in execution slots 1 and 2.
    for rid, row, accepted in (("a", 1, {"c": 1, "d": 2}), ("b", 2, {"c": 3, "d": 1})):
        batch = node.admit(rid)
        assert node.requests[rid].row == row
        node.expect(batch, c=(3, 1, False), d=(4, 2, False))
        node.round(batch, accepted)
    node.land("a", "b")
    landed = {rid: node.dense_states(node.requests[rid].landing) for rid in "ab"}
    batch = node.plan()
    node.expect(batch, c=(3, 1, False), d=(4, 2, False))
    node.round(batch, {"c": 2, "d": 2}, ending=("c", "d"))
    node.finish("c", "d")
    for rid in "ab":
        _assert_bitwise(
            node.dense_states(node.requests[rid].landing), landed[rid], f"{rid} landed"
        )

    batch = node.plan()
    node.expect(batch, a=(1, {1, 2}, True), b=(2, {1, 2}, True))
    node.round(batch, {"a": 2, "b": 3})
    batch = node.plan()
    node.expect(batch, a=(1, {1, 2}, False), b=(2, {1, 2}, False))
    node.round(batch, {"a": 3, "b": 1})


@pytest.mark.parametrize("end", ["device_finish", "host_finish", "abort"])
def test_a_released_permit_starts_its_next_owner_from_zero(end):
    """One permit, two residents. The permit's owner ends: on the device (its
    ending commit folded its records, count 0), or on the host by a finish
    or an abort (its records stay counted). The next plan hands the permit to
    the waiting request as a fresh slot, whose reset clears any count left,
    so it decodes exactly from its landed state."""
    node = _Node(Role.D, max_bs=1, residents=2)
    node.submit("first", "second")
    node.admit("first")
    node.admit("second")
    node.land("first", "second")
    batch = node.plan()
    node.expect(batch, first=(1, 1, True))
    node.round(batch, {"first": 2})
    batch = node.plan()
    node.expect(batch, first=(1, 1, False))
    node.round(batch, {"first": 3}, ending=("first",) if end == "device_finish" else ())
    first = node.requests["first"]
    if end == "device_finish":
        # The ending commit left a dense live page and no record counted.
        assert node.counts[1] == 0
        _assert_states(
            node.dense_states(_live_pages(batch, "first")),
            first.states[first.consumed],
            "ended live page",
        )
        node.finish("first")
    else:
        assert node.counts[1] == 5
        if end == "host_finish":
            node.finish("first")
        else:
            node.abort("first")
        _assert_stale_records_corrupt(node, node.requests["second"], 1)

    batch = node.plan()
    node.expect(batch, second=(2, 1, True))
    node.round(batch, {"second": 1})
    batch = node.plan()
    node.expect(batch, second=(2, 1, False))
    node.round(batch, {"second": 3})


def test_a_retraction_folds_its_permit_before_the_same_plan_regrants_it():
    """One permit, two residents, a forced retraction at the fifth plan. The
    plan that retracts the decoding victim (request row 2 on permit 1) hands
    permit 1 to the other request: the fold applies the victim's records to
    its live pages by the slot the victim held, the new owner's reset then
    clears the slot and it decodes from zero. The image the store took
    restores, and the victim resumes on its permit after the new owner's
    host-decided finish left records counted there, exactly where it
    stopped."""
    node = _Node(Role.D, max_bs=1, residents=2, retraction_interval=5)
    try:
        node.submit("next", "victim")
        node.admit("next")  # plan 1
        node.admit("victim")  # plan 2
        node.land("victim")
        batch = node.plan()  # 3
        node.expect(batch, victim=(2, 1, True))
        node.round(batch, {"victim": 2})
        batch = node.plan()  # 4
        node.expect(batch, victim=(2, 1, False))
        node.round(batch, {"victim": 1})
        victim = node.requests["victim"]
        live = _live_pages(batch, "victim")
        node.land("next")

        plan = node.scheduler.next_execution_plan()  # 5
        assert node.scheduler.retracted_size() == 1
        assert _folds(plan) == {group: [[2, page, 1]] for group, page in live.items()}
        assert node.counts[1] == 3
        # The fold leaves the victim's endpoint dense in its live pages, which
        # the store images, and clears permit 1 before its new owner's reset.
        batch = node.execute(
            plan,
            after_folds=lambda: _assert_states(
                node.dense_states(live), victim.states[victim.consumed], "folded"
            ),
        )
        assert node.counts[1] == 0
        assert [
            op.request_id for op in node.last_operations if isinstance(op, SnapshotOp)
        ] == ["victim"]
        node.expect(batch, next=(1, 1, True))
        node.round(batch, {"next": 3})
        batch = node.plan()  # 6
        node.expect(batch, next=(1, 1, False))
        node.round(batch, {"next": 2})
        node.finish("next")
        assert node.counts[1] == 5

        batch = node.plan()  # 7: the restore
        assert batch is None
        assert [
            op.request_id for op in node.last_operations if isinstance(op, RestoreOp)
        ] == ["victim"]
        assert node.scheduler.retracted_size() == 0
        batch = node.plan()  # 8
        node.expect(batch, victim=(2, 1, True))
        node.round(batch, {"victim": 2})
        batch = node.plan()  # 9
        node.expect(batch, victim=(2, 1, False))
        node.round(batch, {"victim": 3})
    finally:
        node.close()


def test_a_restored_request_resumes_exactly_on_a_different_permit():
    """Two permits, three residents, a forced retraction at the seventh plan.
    The victim decodes on permit 1 beside the holder on permit 2; its
    retraction hands permit 1 to a late landing in the same plan. The holder
    finishes on the host, leaving its records counted in permit 2, and the
    victim restores onto permit 2: folded into its image, its state does not
    depend on permit 1, and the fresh reset of permit 2 drops the holder's
    records."""
    node = _Node(Role.D, max_bs=2, residents=3, retraction_interval=7)
    try:
        node.submit("victim", "holder", "late")
        for rid in ("victim", "holder", "late"):
            node.admit(rid)  # plans 1-3
        node.land("victim")
        batch = node.plan()  # 4
        node.expect(batch, victim=(1, 1, True))
        node.round(batch, {"victim": 2})
        node.land("holder")
        batch = node.plan()  # 5
        node.expect(batch, victim=(1, 1, False), holder=(2, 2, True))
        node.round(batch, {"victim": 1, "holder": 2})
        batch = node.plan()  # 6
        node.expect(batch, victim=(1, 1, False), holder=(2, 2, False))
        node.round(batch, {"victim": 3, "holder": 1})
        victim = node.requests["victim"]
        live = _live_pages(batch, "victim")
        node.land("late")

        plan = node.scheduler.next_execution_plan()  # 7
        assert node.scheduler.retracted_size() == 1
        assert _folds(plan) == {group: [[1, page, 1]] for group, page in live.items()}
        batch = node.execute(
            plan,
            after_folds=lambda: _assert_states(
                node.dense_states(live), victim.states[victim.consumed], "folded"
            ),
        )
        assert [
            op.request_id for op in node.last_operations if isinstance(op, SnapshotOp)
        ] == ["victim"]
        node.expect(batch, holder=(2, 2, False), late=(3, 1, True))
        node.round(batch, {"holder": 2, "late": 2})
        batch = node.plan()  # 8
        node.expect(batch, holder=(2, 2, False), late=(3, 1, False))
        node.round(batch, {"holder": 1, "late": 3})
        node.finish("holder")
        assert node.counts[2] > 0

        batch = node.plan()  # 9: the victim's restore beside late's decode
        assert [
            op.request_id for op in node.last_operations if isinstance(op, RestoreOp)
        ] == ["victim"]
        node.expect(batch, late=(3, 1, False))
        node.round(batch, {"late": 1})
        batch = node.plan()  # 10
        node.expect(batch, victim=(1, 2, True), late=(3, 1, False))
        node.round(batch, {"victim": 2, "late": 2})
        batch = node.plan()  # 11
        node.expect(batch, victim=(1, 2, False), late=(3, 1, False))
        node.round(batch, {"victim": 1, "late": 3})
    finally:
        node.close()


def test_fused_mixed_batch_keys_records_by_request_row():
    """On the fused role an execution slot is the request row: a prompt's
    prefill, its decodes and a mixed round that prefills a second prompt
    ahead of the first's decode row keep each row's records in its own slot,
    with the oracle's numerics for both."""
    node = _Node(Role.Fused, max_bs=4, residents=4, mixed=True)
    node.submit("old", prompt=8)
    batch = node.plan()
    node.expect(batch, old=(1, 1, True))
    assert batch.num_extends() == 1
    node.round(batch)
    batch = node.plan()
    node.expect(batch, old=(1, 1, False))
    node.round(batch, {"old": 2})
    node.submit("new", prompt=7)
    batch = node.plan()
    assert list(batch.request_ids) == ["new", "old"]
    assert batch.num_extends() == 1
    node.expect(batch, new=(2, 2, True), old=(1, 1, False))
    node.round(batch, {"old": 3})
    for accepted in ({"old": 1, "new": 2}, {"old": 2, "new": 3}):
        batch = node.plan()
        node.expect(batch, old=(1, 1, False), new=(2, 2, False))
        assert list(batch.execution_slots) == list(batch.request_pool_indices)
        node.round(batch, accepted)


def test_graph_replay_pads_rows_onto_the_execution_padding_slot():
    """A decode graph captured at three rows replays two decoders from request
    rows 4 and 5 on permits 1 and 2. The padding row reads execution slot
    max_bs + 1, whose records and count no replay or commit touches, and its
    inputs never reach a live row. After a host-decided finish, two landings
    fill the bucket on permit 3 and on the finished request's permit 1, which
    the replayed graph reads once its fresh reset dropped the stale records;
    a device-seen finish then leaves a padding row again."""
    node = _Node(Role.D, max_bs=3, residents=5)
    node.submit("a", "b", "c", "d", "e")
    for rid in ("a", "b", "c", "d", "e"):
        node.admit(rid)
    node.land("d", "e")
    batch = node.plan()
    node.expect(batch, d=(4, 1, True), e=(5, 2, True))
    node.round(batch, {"d": 2, "e": 3})

    records = node.backend._retained_records
    counts = node.backend._replay_length

    def null_and_padding_rows():
        # Copies: advanced indexing gathers the rows.
        rows = [0, node.padding]
        return {"records": records[rows], "counts": counts[rows]}

    # Neither the warmup nor the capture, whose rows all read the padding
    # slot, writes a record or a count of the null or padding row.
    untouched = null_and_padding_rows()
    graph = node.capture(3)
    torch.cuda.synchronize()
    _assert_bitwise(null_and_padding_rows(), untouched, "graph warmup and capture")
    node.assert_counts("graph warmup and capture")
    # A live row that read the padding row's records would turn NaN.
    records[node.padding].fill_(float("nan"))
    untouched = null_and_padding_rows()

    def replay(batch, accepted, **kwargs):
        node.round(batch, accepted, graph=graph, **kwargs)
        _assert_bitwise(null_and_padding_rows(), untouched, "graph replay")

    for accepted in ({"d": 1, "e": 3}, {"d": 3, "e": 2}):
        batch = node.plan()
        node.expect(batch, d=(4, 1, False), e=(5, 2, False))
        replay(batch, accepted)
    node.finish("d")
    node.land("a", "b")
    _assert_stale_records_corrupt(node, node.requests["b"], 1)
    batch = node.plan()
    node.expect(batch, a=(1, 3, True), b=(2, 1, True), e=(5, 2, False))
    replay(batch, {"a": 2, "b": 1, "e": 3})
    batch = node.plan()
    node.expect(batch, a=(1, 3, False), b=(2, 1, False), e=(5, 2, False))
    replay(batch, {"a": 3, "b": 2, "e": 1}, ending=("e",))
    node.finish("e")
    batch = node.plan()
    node.expect(batch, a=(1, 3, False), b=(2, 1, False))
    replay(batch, {"a": 1, "b": 3})


def test_retained_records_take_max_bs_plus_two_rows_with_extra_slots():
    """Two permits and four extra resident slots: the records and counts take
    RequestSlotLayout(max_bs).capacity rows -- the null row, the permits and
    the padding row -- not one per resident. Requests decoding from request
    rows 5 and 6 run on permits 1 and 2 with the oracle's numerics."""
    max_bs, residents = 2, 6
    node = _Node(Role.D, max_bs=max_bs, residents=residents)
    rows = RequestSlotLayout(max_bs).capacity
    assert rows == max_bs + 2 < RequestSlotLayout(residents).capacity
    assert node.backend._record_slot_count == rows
    assert node.backend._retained_records.shape[0] == rows
    assert node.backend._replay_length.shape == (rows, 1)
    assert node.backend._commit_request_slots.shape == (max_bs,)
    node.submit("a", "b", "c", "d", "e", "f")
    for rid in ("a", "b", "c", "d", "e", "f"):
        node.admit(rid)
    node.land("e", "f")
    fresh = True
    for accepted in ({"e": 2, "f": 3}, {"e": 3, "f": 1}, {"e": 1, "f": 2}):
        batch = node.plan()
        node.expect(batch, e=(5, {1, 2}, fresh), f=(6, {1, 2}, fresh))
        assert min(batch.request_pool_indices) > max_bs
        node.round(batch, accepted)
        fresh = False


def test_permuted_permits_and_request_rows_never_mix_records():
    """Three permits, five residents. Landings in reverse row order put
    request rows 1, 3 and 5 on permits 3, 2 and 1; a host-decided finish and
    a device-seen one hand permits 1 and 2 to rows 4 and 2. Every round's
    rows hold distinct slots, and every request matches its own oracle."""
    node = _Node(Role.D, max_bs=3, residents=5)
    node.submit("a", "b", "c", "d", "e")
    for rid in ("a", "b", "c", "d", "e"):
        node.admit(rid)
    node.land("e")
    batch = node.plan()
    node.expect(batch, e=(5, 1, True))
    node.round(batch, {"e": 2})
    node.land("c")
    batch = node.plan()
    node.expect(batch, c=(3, 2, True), e=(5, 1, False))
    node.round(batch, {"c": 3, "e": 1})
    node.land("a")
    batch = node.plan()
    node.expect(batch, a=(1, 3, True), c=(3, 2, False), e=(5, 1, False))
    node.round(batch, {"a": 1, "c": 2, "e": 3})
    batch = node.plan()
    node.expect(batch, a=(1, 3, False), c=(3, 2, False), e=(5, 1, False))
    node.round(batch, {"a": 3, "c": 1, "e": 2})
    node.finish("e")
    node.land("d")
    _assert_stale_records_corrupt(node, node.requests["d"], 1)
    batch = node.plan()
    node.expect(batch, a=(1, 3, False), c=(3, 2, False), d=(4, 1, True))
    node.round(batch, {"a": 2, "c": 3, "d": 2})
    batch = node.plan()
    node.expect(batch, a=(1, 3, False), c=(3, 2, False), d=(4, 1, False))
    node.round(batch, {"a": 1, "c": 2, "d": 3}, ending=("c",))
    node.finish("c")
    node.land("b")
    batch = node.plan()
    node.expect(batch, a=(1, 3, False), b=(2, 2, True), d=(4, 1, False))
    node.round(batch, {"a": 3, "b": 2, "d": 1})
    batch = node.plan()
    node.expect(batch, a=(1, 3, False), b=(2, 2, False), d=(4, 1, False))
    node.round(batch, {"a": 2, "b": 3, "d": 2})


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
