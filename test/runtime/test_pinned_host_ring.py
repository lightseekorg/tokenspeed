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

"""Tests for PinnedHostRing, InputBuffers ring staging, and in-place graph padding."""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=15, suite="runtime-1gpu")

from tokenspeed.runtime.execution.context import ForwardContext  # noqa: E402
from tokenspeed.runtime.execution.forward_batch_info import (  # noqa: E402
    ForwardMode,
    ForwardOutputLayout,
)
from tokenspeed.runtime.execution.forward_step import ForwardStepRunner  # noqa: E402
from tokenspeed.runtime.execution.input_buffer import InputBuffers  # noqa: E402
from tokenspeed.runtime.execution.runtime_states import RuntimeStates  # noqa: E402
from tokenspeed.runtime.sampling.backends.base import (  # noqa: E402
    CUDA_GRAPH_VARIANT_DEFAULT,
)
from tokenspeed.runtime.utils.pinned_host_ring import (  # noqa: E402
    PinnedHostRing,
    copy_host_values,
)


class _FakeEvent:
    def __init__(self, events: list[_FakeEvent]) -> None:
        self.recorded = False
        self.complete = True
        events.append(self)

    def record(self) -> None:
        self.recorded = True
        self.complete = False

    def query(self) -> bool:
        return self.complete


def test_pinned_host_ring_respects_event_and_weakref_fences() -> None:
    events: list[_FakeEvent] = []
    ring = PinnedHostRing(
        "cpu",
        max_slots=4,
        min_capacity_bytes=64,
        event_factory=lambda: _FakeEvent(events),
    )

    with ring.staging():
        a0, b0 = ring.acquire_views([(3, torch.int64), (2, torch.int32)])
        a0.copy_(torch.tensor([10, 20, 30], dtype=torch.int64))
        b0.copy_(torch.tensor([1, 2], dtype=torch.int32))
        ptr0 = a0.data_ptr()
        # A second acquire while still inside staging must not overwrite slot 0.
        (a1,) = ring.acquire_views([(3, torch.int64)])
        assert a1.data_ptr() != ptr0
        del a1
    assert ring.alloc_count == 2
    assert len(events) == 2 and all(ev.recorded for ev in events)

    # Even once views drop, incomplete DMA events keep slots locked.
    del a0, b0
    assert not events[0].query() and not events[1].query()
    with ring.staging():
        (a2,) = ring.acquire_views([(3, torch.int64)])
        assert a2.data_ptr() != ptr0
    assert ring.alloc_count == 3

    # Completing all prior events allows zero-allocation reuse of slot 0.
    del a2
    for ev in events:
        ev.complete = True
    with ring.staging():
        (a3,) = ring.acquire_views([(3, torch.int64)])
        assert a3.data_ptr() == ptr0
    assert ring.alloc_count == 3

    # copy_packed_d2h holds the slot locked while either event OR returned view is live.
    del a3
    for ev in events:
        ev.complete = True
    src = torch.tensor([7, 8, 9, 4, 5, 6], dtype=torch.int32)
    held_t, held_l = ring.copy_packed_d2h(
        src, [(0, (2, 2), torch.int32), (4, (2,), torch.int32)]
    )
    assert held_t.tolist() == [[7, 8], [9, 4]]
    assert held_l.tolist() == [5, 6]
    # Mark event complete while Python caller still holds held_t/held_l.
    for ev in events:
        ev.complete = True
    held_ptr = held_t.data_ptr()
    src2 = torch.tensor([99, 98, 97, 96, 1, 2], dtype=torch.int32)
    other_t, other_l = ring.copy_packed_d2h(
        src2, [(0, (2, 2), torch.int32), (4, (2,), torch.int32)]
    )
    assert other_t.data_ptr() != held_ptr
    assert held_t.tolist() == [[7, 8], [9, 4]]
    assert other_l.tolist() == [1, 2]
    del held_t, held_l, other_t, other_l


def test_pinned_host_ring_grows_and_falls_back_when_all_slots_busy() -> None:
    ring = PinnedHostRing("cpu", max_slots=2, min_capacity_bytes=64)
    held = []
    for val in (1, 2, 3):
        (v,) = ring.acquire_views([(4, torch.int32)])
        v.fill_(val)
        ring.seal_pending()
        held.append(v)
    assert ring.alloc_count == 3
    assert ring.fallback_alloc_count == 1
    assert [t.tolist() for t in held] == [[1] * 4, [2] * 4, [3] * 4]
    held.clear()

    # Reclaiming a slot with larger size grows it once, then steady reuse has 0 allocs.
    with ring.staging():
        (big,) = ring.acquire_views([(64, torch.int32)])
        big.fill_(11)
        del big
    grown_allocs = ring.alloc_count
    for step in range(25):
        with ring.staging():
            (v,) = ring.acquire_views([(64, torch.int32)])
            v.fill_(step)
            del v
    assert ring.alloc_count == grown_allocs


def test_copy_host_values_handles_empty_list_array_and_tensor() -> None:
    for dtype, np_dtype in ((torch.int32, np.int32), (torch.int64, np.int64)):
        empty_dst = torch.empty(0, dtype=dtype)
        copy_host_values(empty_dst, [])
        assert empty_dst.tolist() == []

        dst = torch.empty(3, dtype=dtype)
        copy_host_values(dst, [3, 1, 4])
        assert dst.tolist() == [3, 1, 4]
        copy_host_values(dst, (9, 8, 7))
        assert dst.tolist() == [9, 8, 7]
        copy_host_values(dst, np.array([5, 6, 7], dtype=np_dtype))
        assert dst.tolist() == [5, 6, 7]
        copy_host_values(dst, torch.tensor([2, 0, 1], dtype=dtype))
        assert dst.tolist() == [2, 0, 1]


def test_input_buffers_reuses_pinned_ring_and_forward_step_pads_in_place() -> None:
    ib = InputBuffers(
        max_bs=8,
        max_num_tokens=64,
        state_write_padding_pool_index=9,
        device="cpu",
        reads_state_checkpoints=False,
    )
    runtime = RuntimeStates(
        req_pool_size=9,
        input_vocab_size=128,
        vocab_size=128,
        output_length=1,
        device="cpu",
    )

    for step in range(12):
        forward_op = SimpleNamespace(
            request_ids=[0, 1],
            request_pool_indices=[2 + (step % 3), 5],
            input_lengths=[1, 1],
            decode_input_ids=None,
            extend_prefix_lens=[],
            prefill_lengths=[],
            extend_replay_lens=[],
            extend_state_checkpoint_lens=[],
            num_extends=lambda: 0,
        )
        ib.fill_input_buffers(forward_op, runtime, total_tokens=2)
    assert ib._pinned_ring.alloc_count == 1
    assert ib._pinned_ring.fallback_alloc_count == 0
    assert ib.req_pool_indices_buf[:2].tolist() == [4, 5]
    assert ib.input_lengths_buf[:2].tolist() == [1, 1]

    # Verify ForwardStepRunner.__call__ pads seq_lens and req_pool_indices in place
    # without calling torch.cat/F.pad or losing backing buffer pointers.
    for spec_algo, expected_pad in ((None, 0), ("DSPARK", 9)):
        ib.seq_lens_buf.fill_(99)
        ib.req_pool_indices_buf.fill_(88)
        ib.req_pool_indices_buf[:2].copy_(torch.tensor([3, 6], dtype=torch.int64))
        ib.seq_lens_buf[:2].copy_(torch.tensor([11, 12], dtype=torch.int32))
        seq_ptr = ib.seq_lens_buf.data_ptr()
        req_ptr = ib.req_pool_indices_buf.data_ptr()

        observed = {}

        def _record_meta(padded_bs, bs, req_pool_indices, seq_lens, **_kwargs) -> None:
            observed["req"] = req_pool_indices.tolist()
            observed["seq"] = seq_lens.tolist()
            observed["req_ptr"] = req_pool_indices.data_ptr()
            observed["seq_ptr"] = seq_lens.data_ptr()

        runner = ForwardStepRunner.__new__(ForwardStepRunner)
        runner.config = SimpleNamespace(
            max_req_pool_size=9,
            spec_algo=spec_algo,
            spec_draft_tree=False,
        )
        runner.input_buffers = ib
        runner.runtime_states = None
        runner.attn_backend = SimpleNamespace(
            commit_state_after_acceptance=lambda *_a, **_kw: None
        )
        runner.max_tokens_per_req = 1
        runner.device = "cpu"
        runner.drafter = None
        runner.deepep_adapter = SimpleNamespace(replay=lambda: None)
        runner.token_to_kv_pool = SimpleNamespace(
            arena=SimpleNamespace(cache_group_specs=[])
        )
        runner.moe_source_rows = ()
        runner._expert_load_rows = None
        runner._mega_moe_live_requests = None
        runner._reported_query_graphs = set()
        runner._graph_debug = False
        runner.graphs = {
            (CUDA_GRAPH_VARIANT_DEFAULT, 6): SimpleNamespace(replay=lambda: None)
        }
        runner.output_buffers = {
            (CUDA_GRAPH_VARIANT_DEFAULT, 6): (
                torch.arange(6, dtype=torch.int32),
                torch.ones(6, dtype=torch.int32),
                None,
                None,
                None,
            )
        }
        runner._can_use_graph = lambda *_a: True
        runner._padded_bs = lambda *_a: 6
        runner._prepare_request_token_history_graph_inputs = lambda **_kw: None
        runner._prepare_decode_metadata = _record_meta
        runner._cuda_graph_key = lambda bs: (CUDA_GRAPH_VARIANT_DEFAULT, bs)

        ctx = ForwardContext(
            attn_backend=None,
            token_to_kv_pool=None,
            bs=2,
            num_extends=0,
            output_layout=ForwardOutputLayout(0, 0, 2, 1),
            input_num_tokens=2,
            forward_mode=ForwardMode.DECODE,
        )
        out_tokens, out_lengths, *_ = runner(
            2,
            ctx,
            SimpleNamespace(),
            extend_with_prefix=False,
            extend_prefix_lens=torch.empty(0, dtype=torch.int32),
            extend_prefix_lens_cpu=torch.empty(0, dtype=torch.int32),
            extend_seq_lens=torch.empty(0, dtype=torch.int32),
            extend_seq_lens_cpu=torch.empty(0, dtype=torch.int32),
            extend_replay_lens_cpu=torch.empty(0, dtype=torch.int32),
            extend_prompt_lens_cpu=torch.empty(0, dtype=torch.int32),
            block_tables_cpu={},
            settle=lambda: None,
        )
        assert observed["req"] == [3, 6] + [expected_pad] * 4
        assert observed["seq"] == [11, 12, 1, 1, 1, 1]
        assert observed["req_ptr"] == req_ptr
        assert observed["seq_ptr"] == seq_ptr
        assert ctx.bs == 2 and ctx.output_layout == ForwardOutputLayout(0, 0, 2, 1)
        assert out_tokens.tolist() == [0, 1]
        assert out_lengths.tolist() == [1, 1]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_pinned_host_ring_cuda_real_event_and_zero_gpu_alloc() -> None:
    torch.cuda.synchronize()
    before_bytes = torch.cuda.memory_allocated()

    ring = PinnedHostRing("cuda", max_slots=4, min_capacity_bytes=1024)
    gpu_dst = torch.empty(16, dtype=torch.int32, device="cuda")

    for step in range(16):
        with ring.staging():
            (cpu_src,) = ring.acquire_views([(16, torch.int32)])
            assert cpu_src.is_pinned()
            copy_host_values(cpu_src, list(range(step, step + 16)))
            gpu_dst.copy_(cpu_src, non_blocking=True)
            del cpu_src
        cpu_out, cpu_tail = ring.copy_packed_d2h(
            gpu_dst, [(0, (12,), torch.int32), (12, (4,), torch.int32)]
        )
        assert cpu_out.is_pinned() and cpu_tail.is_pinned()
        torch.cuda.synchronize()
        assert cpu_out.tolist() == list(range(step, step + 12))
        assert cpu_tail.tolist() == list(range(step + 12, step + 16))
        del cpu_out, cpu_tail

    torch.cuda.synchronize()
    after_bytes = torch.cuda.memory_allocated()
    assert after_bytes == before_bytes
    assert ring.alloc_count <= 4
    assert ring.fallback_alloc_count == 0
