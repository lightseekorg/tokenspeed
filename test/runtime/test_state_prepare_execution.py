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

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from tokenspeed.runtime.execution.model_executor import ModelExecutor
from tokenspeed.runtime.execution.runtime_states import RuntimeStates
from tokenspeed.runtime.execution.types import StopCriteria
from tokenspeed.runtime.layers.attention.backends.base import CachePoolBinding
from tokenspeed.runtime.layers.attention.kv_cache.recipes.cache_runtime import (
    CacheRuntimeContract,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import CacheGroupSpec


class DeviceEvent:
    def __init__(self):
        self.stream = None

    def record(self, stream):
        self.stream = stream


class DeviceStream:
    def wait_stream(self, stream):
        pass


class CpuDevice:
    """Only the device stream/event API is absent on the host."""

    Event = DeviceEvent

    @staticmethod
    def stream(stream):
        return nullcontext()


class StateOwner(CachePoolBinding):
    def __init__(self, success):
        self.success = success
        self.states = torch.zeros(4, dtype=torch.int64)

    def deferred_state_group_ids(self):
        return ("state",)

    def prepare_state_checkpoint(
        self, *, request_id, request_pool_index, num_computed_tokens, state_blocks
    ):
        # A CPU implementation of a state writer; repeated invocation changes
        # the value, so duplicate target/draft traversal is observable.
        self.states[state_blocks["state"]] += num_computed_tokens
        return self.success


def executor_fixture(*, rank, success):
    executor = ModelExecutor.__new__(ModelExecutor)
    executor.runtime_states = RuntimeStates(4, 32, 8, device="cpu")
    executor.runtime_states.bind_stop_criteria(["r"], [2], [StopCriteria((), 256)])
    executor.attn_backend = StateOwner(success)
    executor.draft_attn_backend = executor.attn_backend
    executor.default_stream = DeviceStream()
    executor.execution_stream = DeviceStream()
    executor.device_module = CpuDevice()
    executor._cache_kvp_rank = rank
    executor._cache_runtime_contract = CacheRuntimeContract(
        prefix_granularity=128,
        num_lcm_blocks=3,
        token_capacity=384,
        group_specs=(
            CacheGroupSpec(
                group_id="state", attn_kind="state", replayable=False, shard_count=2
            ),
        ),
        group_page_counts={"state": 4},
        group_packing={"state": 1},
    )
    return executor


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("success", [True, False])
def test_prepare_translates_owned_blocks_and_deduplicates_backends(rank, success):
    executor = executor_fixture(rank=rank, success=success)
    # The binding exposes this immutable metadata with no Python constructor.
    op = SimpleNamespace(
        request_id="r",
        request_pool_index=2,
        num_computed_tokens=130,
        state_blocks={"state": 3},
    )
    event, ready = executor.prepare_state_checkpoint(op)
    assert event.stream is executor.execution_stream
    assert ready is (success if rank == 0 else True)
    expected = torch.zeros(4, dtype=torch.int64)
    if rank == 0:
        expected[2] = 130
    torch.testing.assert_close(executor.attn_backend.states, expected)


def test_prepare_rejects_stale_request_after_slot_reuse_without_writes():
    executor = executor_fixture(rank=0, success=True)
    executor.runtime_states.bind_cache_request_slots(["replacement"], [2])
    op = SimpleNamespace(
        request_id="r",
        request_pool_index=2,
        num_computed_tokens=130,
        state_blocks={"state": 3},
    )
    event, ready = executor.prepare_state_checkpoint(op)
    assert not ready
    assert event.stream is executor.execution_stream
    assert torch.count_nonzero(executor.attn_backend.states) == 0


def test_slot_restore_rebinds_identity_before_any_new_forward():
    states = RuntimeStates(4, 32, 8, device="cpu")
    states.bind_cache_request_slots(["old"], [2])
    states.valid_cache_lengths[1] = 130
    image = torch.empty(states.slot_state_bytes(), dtype=torch.uint8)
    states.export_slot_state(1, image, None, request_id="r")
    states.import_slot_state(2, image, None, request_id="r")
    assert states.owns_cache_request_slot("r", 2)
    assert not states.owns_cache_request_slot("old", 2)
    assert states.valid_cache_lengths[2] == 130


@pytest.mark.parametrize("slots", [[2, 2], [-1, 2], [2, 4]])
def test_invalid_slot_binding_is_atomic(slots):
    states = RuntimeStates(4, 32, 8, device="cpu")
    states.bind_cache_request_slots(["old"], [2])
    with pytest.raises(ValueError):
        states.bind_cache_request_slots(["r", "s"], slots)
    assert states.owns_cache_request_slot("old", 2)
