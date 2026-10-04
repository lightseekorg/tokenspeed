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

"""Fixed hot reservation and the existing scheduler admission gate."""

from dataclasses import replace

import pytest
import tokenspeed_scheduler as ts

from tokenspeed.runtime.layers.attention.kv_cache.offload_config import KVOffloadPolicy
from tokenspeed.runtime.layers.attention.kv_cache.recipes.plan import (
    CacheFieldSpec,
    pack,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import CacheGroupSpec
from tokenspeed.runtime.layers.attention.kv_cache.recipes.storage import (
    compute_offload_capacity,
    plan_cache_storage,
)


def geometry():
    group = CacheGroupSpec(
        group_id="history",
        retention="full_history",
        rows_per_page=8,
        entry_stride_tokens=1,
        transfer_policy="full_suffix",
        replayable=False,
    )
    layout = pack(
        (
            (
                group,
                (
                    CacheFieldSpec("full", "full", (8, 1, 32), "bfloat16"),
                    CacheFieldSpec("hot", "hot", (8, 1, 32), "bfloat16"),
                ),
            ),
        ),
        prefix_granularity=8,
        cache_blocks_per_lcm_block={"history": 1},
        alignment=256,
        max_padding_fraction=0.25,
    )
    config = KVOffloadPolicy(("hot",), 64, 4, 16, 1, 1 << 20, True, 0, ()).bind(
        request_slots=8, device_rows=1024, max_extend_tokens=32
    )
    return layout, (group,), config


def solve(budget, *, cap, probe=None):
    layout, groups, config = geometry()
    parents, actual = compute_offload_capacity(
        layout,
        groups,
        offload=config,
        device_budget_bytes=budget,
        max_lcm_blocks=cap,
        probe_lcm_blocks=probe,
    )
    plan = plan_cache_storage(layout.bind(parents), groups, offload=actual)
    return actual, plan


def test_spare_memory_does_not_grow_pool_past_configured_concurrency():
    a, p = solve(1 << 20, cap=32)
    b, q = solve((1 << 20) + 4096, cap=32)
    assert a.device_rows == b.device_rows == 544
    assert p.device_history_bytes == q.device_history_bytes == 33 * 8 * 64
    assert a.device_rows == a.hot_rows
    assert p.device_budget_bytes == q.device_budget_bytes < 1 << 20
    assert p.device_budget_bytes - p.fixed_device_bytes == p.device_history_bytes


def test_host_cap_leaves_spare_device_budget_unallocated():
    layout, groups, cfg = geometry()
    cfg = replace(cfg, host_budget_bytes=10 * 8 * 64)
    parents, actual = compute_offload_capacity(
        layout,
        groups,
        offload=cfg,
        device_budget_bytes=1 << 20,
        max_lcm_blocks=128,
        probe_lcm_blocks=None,
    )
    plan = plan_cache_storage(layout.bind(parents), groups, offload=actual)
    assert parents == 9
    assert plan.host_bytes == cfg.host_budget_bytes
    assert actual.device_rows == 544
    assert plan.device_budget_bytes < 1 << 20


@pytest.mark.parametrize("budget_rows", [208, 215, 256, 543, 544, 1000])
@pytest.mark.parametrize("probe", [None, 1])
def test_fixed_pool_must_fit_without_reducing_concurrency(budget_rows, probe):
    layout, groups, cfg = geometry()
    unit = plan_cache_storage(layout.bind(1), groups, offload=cfg)
    overhead = unit.fixed_device_bytes - unit.workspaces[0].device_rows * 64
    budget = overhead + unit.device_history_bytes + budget_rows * 64
    if budget_rows < 544:
        with pytest.raises(ValueError, match="fixed hot pool"):
            solve(budget, cap=1, probe=probe)
    else:
        actual, plan = solve(budget, cap=1, probe=probe)
        assert actual.device_rows == 544
        assert plan.device_budget_bytes <= budget


@pytest.mark.parametrize("request_slots,expected_rows", [(3, 208), (4, 272), (8, 544)])
def test_concurrency_sets_pool_demand(request_slots, expected_rows):
    layout, groups, cfg = geometry()
    cfg = replace(cfg, request_slots=request_slots)
    _, actual = compute_offload_capacity(
        layout,
        groups,
        offload=cfg,
        device_budget_bytes=1 << 20,
        max_lcm_blocks=32,
        probe_lcm_blocks=None,
    )
    assert actual.device_rows == expected_rows


@pytest.mark.parametrize("probe", [None, 2])
def test_extend_chunk_can_exceed_concurrency_demand(probe):
    layout, groups, cfg = geometry()
    cfg = replace(cfg, max_extend_tokens=1024, device_rows=1032)
    _, actual = compute_offload_capacity(
        layout,
        groups,
        offload=cfg,
        device_budget_bytes=1 << 20,
        max_lcm_blocks=32,
        probe_lcm_blocks=probe,
    )
    assert actual.device_rows == 1032  # 1024 writes + null row, rounded to 8.
    assert actual.device_rows > actual.hot_rows


def test_bound_config_rejects_incomplete_request_partitions():
    _, _, cfg = geometry()
    with pytest.raises(ValueError, match="all request slots"):
        replace(cfg, device_rows=cfg.hot_rows - 1)


def test_history_uses_only_the_balance_after_fixed_costs():
    layout, groups, cfg = geometry()
    actual, unit = solve(1 << 20, cap=1)
    budget = unit.device_budget_bytes + 3 * 512
    parents, bound = compute_offload_capacity(
        layout,
        groups,
        offload=cfg,
        device_budget_bytes=budget,
        max_lcm_blocks=100,
        probe_lcm_blocks=None,
    )
    assert parents == 4
    assert bound.device_rows == actual.device_rows
    plan = plan_cache_storage(layout.bind(parents), groups, offload=bound)
    assert plan.device_budget_bytes == budget


def test_probe_does_not_allocate_the_serving_pool_twice():
    cfg, plan = solve(1 << 30, cap=1024, probe=2)
    assert cfg.device_rows == 544
    assert plan.device_budget_bytes < 1 << 20


def test_remote_pd_landing_holds_the_only_hot_request_partition():
    cfg = ts.SchedulerConfig()
    cfg.role = ts.SchedulerConfig.Role.D
    cfg.prefix_granularity = 64
    cfg.num_device_pages = 129
    cfg.max_batch_size = 1  # The configured B is also the fixed hot partition count.
    cfg.max_scheduled_tokens = 64
    cfg.decode_input_tokens = 1
    cfg.disable_l2_cache = True
    cfg.disable_prefix_cache = True
    cfg.cache_groups = [
        ts.CacheGroupConfig(
            group_id="history",
            block_granularity=64,
            total_pages=129,
            retention=ts.CacheRetention.FullHistory,
            family=ts.CacheGroupFamily.History,
            transfer_policy=ts.CacheTransferPolicy.FullSuffix,
        )
    ]
    scheduler = ts.Scheduler(cfg)
    for rid in ("first", "second"):
        spec = ts.RequestSpec()
        spec.request_id, spec.tokens, spec.max_new_tokens = rid, list(range(65)), 16
        scheduler.submit_requests([spec])
        scheduler.advance(ts.ExecutionEvent().add_event(ts.PD.BootstrappedEvent(rid)))
    first = scheduler.next_execution_plan()
    assert list(first.remote_prefill.request_ids) == ["first"]
    assert scheduler.next_execution_plan().remote_prefill is None
    scheduler.advance(
        ts.ExecutionEvent().add_event(ts.PD.RemotePrefillDoneEvent("first", 100))
    )
    scheduler.next_execution_plan()
    event = ts.ForwardEvent.Abort()
    event.request_id = "first"
    scheduler.advance(ts.ExecutionEvent().add_event(event))
    for _ in range(4):
        plan = scheduler.next_execution_plan()
        if plan.remote_prefill is not None:
            assert list(plan.remote_prefill.request_ids) == ["second"]
            break
    else:
        pytest.fail("released hot partition was not readmitted")


def test_all_host_history_uses_the_joint_row_cost():
    layout, groups, cfg = geometry()
    cfg = replace(cfg, field_ids=("full", "hot"))
    unit = plan_cache_storage(layout.bind(1), groups, offload=cfg)
    overhead = unit.fixed_device_bytes - cfg.device_rows * 128
    budget = overhead + cfg.hot_rows * 128
    parents, actual = compute_offload_capacity(
        layout,
        groups,
        offload=cfg,
        device_budget_bytes=budget,
        max_lcm_blocks=32,
        probe_lcm_blocks=None,
    )
    plan = plan_cache_storage(layout.bind(parents), groups, offload=actual)
    assert parents == 32
    assert plan.device_history_bytes == 0
    assert sum(w.row_bytes for w in plan.workspaces) == 128
    assert actual.device_rows == 544
    assert plan.device_budget_bytes <= budget
    assert {w.device_rows for w in plan.workspaces} == {actual.device_rows}


def test_large_query_hash_scratch_is_prepaid():
    layout, groups, config = geometry()
    config = replace(
        config,
        hot_tokens=4096,
        reserved_tokens=8,
        topk=2048,
        queries=8,
        device_rows=8 * (4096 + 8),
    )
    plan = plan_cache_storage(layout.bind(1), groups, offload=config)
    metadata = dict(plan.workspaces[0].metadata)
    assert metadata["entry_dest"] == 8 * 8 * 2048
    assert metadata["hash_keys"] == metadata["hash_owners"] == 8 * 32768
    smaller = replace(config, queries=4)
    small_metadata = smaller.metadata_counts()
    assert small_metadata["hash_keys"] == small_metadata["hash_owners"] == 0
    # The allocator consumes exactly this list, so the global-table mode cannot
    # allocate a separate unbudgeted per-request hash after admission.
    assert metadata == config.metadata_counts()
    assert config.temporary_bytes() == smaller.temporary_bytes()
