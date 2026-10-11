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

"""Retraction snapshot pool sizing: the server knobs and their resolution.

``ServerArgs`` settles the knobs (role, request cap, the abort-only ratio 0);
``cache/l2/sizing.py`` turns them into LCM blocks against the rank's layout,
with the tail unit the scheduler's ``CapacityModel`` answers.
"""

import logging

import pytest

ts = pytest.importorskip("tokenspeed_scheduler")

from tokenspeed.runtime.cache.l2.sizing import (  # noqa: E402
    DEFAULT_DEVICE_RATIO,
    NO_POOL,
    RetractionPoolRequest,
    RetractionPoolSizing,
    resolve_retraction_pool,
    tail_lcm_blocks_per_request,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.scheduler_bridge import (  # noqa: E402
    SchedulerLimits,
    capacity_model,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import (  # noqa: E402
    CacheGroupSpec,
)
from tokenspeed.runtime.utils.server_args import ServerArgs  # noqa: E402

PREFETCH = dict(
    kvstore_prefetch_min_pages=2,
    kvstore_prefetch_timeout_base_s=1.0,
    kvstore_prefetch_timeout_per_page_s=0.0,
    kvstore_prefetch_batch_pages=128,
)


# ----------------------------------------------------------------------
# ServerArgs: the knobs as the operator states them
# ----------------------------------------------------------------------


def test_default_is_a_derived_pool_with_the_request_cap_from_max_num_seqs():
    args = ServerArgs(model="x", max_num_seqs=64)
    assert args.retraction_snapshot_host_gb == 0.0
    assert args.retraction_snapshot_ratio is None
    assert args.retraction_snapshot_max_requests == 64
    assert not args.retraction_snapshot_pool_disabled
    # Per attention-DP rank: the scheduler's batch bound is rank-local.
    args = ServerArgs(model="x", world_size=2, data_parallel_size=2, max_num_seqs=10)
    assert args.mapping.attn.dp_size == 2
    assert args.retraction_snapshot_max_requests == 5
    # An explicit cap is kept as is.
    args = ServerArgs(model="x", max_num_seqs=64, retraction_snapshot_max_requests=3)
    assert args.retraction_snapshot_max_requests == 3
    # Fewer requests than attention-DP ranks derive to no rows: refused by
    # name (resolve_cache runs before the general max_num_seqs check).
    with pytest.raises(
        ValueError, match=r"--max-num-seqs \(1\).*--retraction-snapshot-max-requests"
    ):
        ServerArgs(model="x", world_size=2, data_parallel_size=2, max_num_seqs=1)


def test_explicit_size_and_ratio_pass_through_and_each_is_enough_alone():
    args = ServerArgs(model="x", retraction_snapshot_host_gb=4.0)
    assert (args.retraction_snapshot_host_gb, args.retraction_snapshot_ratio) == (
        4.0,
        None,
    )
    args = ServerArgs(model="x", retraction_snapshot_ratio=0.5)
    assert (args.retraction_snapshot_host_gb, args.retraction_snapshot_ratio) == (
        0.0,
        0.5,
    )
    assert args.retraction_snapshot_max_requests == args.max_num_seqs
    for bad in (
        dict(retraction_snapshot_host_gb=-1.0),
        dict(retraction_snapshot_ratio=-0.1),
        dict(retraction_snapshot_max_requests=-1),
    ):
        with pytest.raises(ValueError, match="non-negative"):
            ServerArgs(model="x", **bad)


def test_ratio_zero_is_the_one_way_to_run_abort_only():
    args = ServerArgs(model="x", retraction_snapshot_ratio=0.0)
    assert args.retraction_snapshot_pool_disabled
    assert args.retraction_snapshot_max_requests == 0
    # A size override still builds a pool; the rows have something to size.
    args = ServerArgs(
        model="x", retraction_snapshot_ratio=0.0, retraction_snapshot_host_gb=1.0
    )
    assert not args.retraction_snapshot_pool_disabled
    with pytest.raises(ValueError, match="has no rows to size"):
        ServerArgs(
            model="x", retraction_snapshot_ratio=0.0, retraction_snapshot_max_requests=4
        )


@pytest.mark.parametrize("role", ["prefill", "encode"])
def test_non_retracting_roles_resolve_to_no_pool_with_a_log(role, caplog):
    with caplog.at_level(logging.INFO):
        args = ServerArgs(
            model="x", disaggregation_mode=role, retraction_snapshot_host_gb=2.0
        )
    assert args.retraction_snapshot_pool_disabled
    assert (args.retraction_snapshot_host_gb, args.retraction_snapshot_ratio) == (
        0.0,
        0.0,
    )
    assert args.retraction_snapshot_max_requests == 0
    assert any("never retracts" in record.message for record in caplog.records)
    # The forced-retraction test knob is dropped too (the scheduler refuses
    # it on a pool-less engine), and the log says so.
    caplog.clear()
    with caplog.at_level(logging.INFO):
        args = ServerArgs(
            model="x", disaggregation_mode=role, debug_force_retraction_interval=3
        )
    assert args.debug_force_retraction_interval == 0
    assert any(
        "never retracts" in record.message
        and "--debug-force-retraction-interval" in record.message
        for record in caplog.records
    )
    # Nothing to say when nothing was asked for.
    caplog.clear()
    with caplog.at_level(logging.INFO):
        args = ServerArgs(model="x", disaggregation_mode=role)
    assert args.retraction_snapshot_pool_disabled
    assert not any("never retracts" in record.message for record in caplog.records)


def test_the_pool_is_independent_of_the_kvstore():
    args = ServerArgs(model="x", disable_kvstore=True, retraction_snapshot_host_gb=1.0)
    assert args.enable_kvstore is False
    assert args.retraction_snapshot_host_gb == 1.0
    args = ServerArgs(model="x", disable_kvstore=True)
    assert not args.retraction_snapshot_pool_disabled


def test_kvp_takes_both_host_tiers_but_not_l3():
    # The scheduler allocates every Host block in its Device block's residue
    # class and the executor translates ownership on both ends of every row,
    # so a KV-page-sharded engine may run the Host KVStore and the snapshot
    # pool. L3 keys have no owner-stable form under sharding and stay refused.
    args = ServerArgs(
        model="x",
        world_size=2,
        kv_parallel_size=2,
        retraction_snapshot_host_gb=1.0,
    )
    assert args.enable_kvstore is True and args.retraction_snapshot_host_gb == 1.0
    with pytest.raises(ValueError, match="L3.*kv-parallel-size"):
        ServerArgs(
            model="x",
            world_size=2,
            kv_parallel_size=2,
            kvstore_storage_backend="mooncake",
            **PREFETCH,
        )


def test_forced_retraction_is_a_test_knob_that_needs_a_pool():
    assert ServerArgs(model="x").debug_force_retraction_interval == 0
    with pytest.raises(ValueError, match="debug-force-retraction-interval.*ratio 0"):
        ServerArgs(
            model="x", retraction_snapshot_ratio=0.0, debug_force_retraction_interval=3
        )
    for interval in (3, -2):
        args = ServerArgs(model="x", debug_force_retraction_interval=interval)
        assert args.debug_force_retraction_interval == interval


def test_the_l3_prefetch_knobs_are_explicit_with_a_store_and_absent_without():
    # No silent default: the threshold, the deadline and the batch all come
    # with the store; without one they have nothing to size.
    with pytest.raises(
        ValueError, match="needs the L3 prefetch knobs: --kvstore-prefetch-min-pages"
    ):
        ServerArgs(model="x", kvstore_storage_backend="memory")
    with pytest.raises(ValueError, match="need --kvstore-storage-backend"):
        ServerArgs(model="x", kvstore_prefetch_min_pages=2)
    args = ServerArgs(model="x", kvstore_storage_backend="memory", **PREFETCH)
    assert args.kvstore_prefetch_min_pages == 2
    for bad in (
        dict(kvstore_prefetch_min_pages=0),
        dict(kvstore_prefetch_timeout_base_s=0.0),
        dict(kvstore_prefetch_timeout_per_page_s=-1.0),
        dict(kvstore_prefetch_batch_pages=0),
    ):
        with pytest.raises(ValueError, match=next(iter(bad)).replace("_", "-")):
            ServerArgs(
                model="x", kvstore_storage_backend="memory", **{**PREFETCH, **bad}
            )


# ----------------------------------------------------------------------
# sizing.py: the knobs against a rank's layout
# ----------------------------------------------------------------------

BLOCK = 1_000_000  # 1 MB Host LCM blocks
DEVICE = 40  # Device LCM blocks


def _request(host_gb=0.0, ratio=None, rows=8, tail=3):
    return RetractionPoolRequest(
        host_gb=host_gb,
        ratio=ratio,
        max_retracted_requests=rows,
        tail_lcm_blocks_per_request=tail,
    )


def _resolve(request, *, l2_tier):
    return resolve_retraction_pool(
        request, l2_tier=l2_tier, device_lcm_blocks=DEVICE, host_lcm_block_bytes=BLOCK
    )


@pytest.mark.parametrize("l2_tier", [False, True])
def test_resolution_order_size_then_ratio_then_derived(l2_tier):
    # An explicit size wins over everything, whole blocks only.
    sizing = _resolve(_request(host_gb=0.0105, ratio=0.5), l2_tier=l2_tier)
    assert (sizing.lcm_blocks, sizing.max_retracted_requests) == (10, 8)
    assert "--retraction-snapshot-host-gb" in sizing.source
    # Then the ratio of this rank's Device KV.
    sizing = _resolve(_request(ratio=0.25), l2_tier=l2_tier)
    assert sizing.lcm_blocks == 10
    assert "--retraction-snapshot-ratio" in sizing.source
    # Neither: a tenth of the Device KV without L2 (whole images, kept cheap
    # to pin), the image tails with it.
    sizing = _resolve(_request(), l2_tier=l2_tier)
    assert DEFAULT_DEVICE_RATIO == 0.1
    assert sizing.lcm_blocks == (8 * 3 if l2_tier else int(DEVICE * 0.1))
    assert sizing.source.startswith("derived")
    if not l2_tier:
        # Explicit ratio 1 is how a deployment makes every resident suspendable.
        assert _resolve(_request(ratio=1.0), l2_tier=False).lcm_blocks == DEVICE


def test_ratio_zero_resolves_to_no_pool_and_too_small_sizes_are_refused():
    assert _resolve(_request(ratio=0.0, rows=0), l2_tier=True) is NO_POOL
    assert NO_POOL == RetractionPoolSizing(0, 0, "--retraction-snapshot-ratio 0")
    assert _request(ratio=0.0, rows=0).disabled
    assert not _request(ratio=0.0, host_gb=1.0).disabled
    with pytest.raises(ValueError, match="no whole LCM block"):
        _resolve(_request(host_gb=0.0005), l2_tier=True)
    with pytest.raises(ValueError, match="no whole LCM block"):
        _resolve(_request(ratio=0.01), l2_tier=True)
    with pytest.raises(ValueError, match="slot-state rows"):
        _resolve(_request(ratio=0.5, rows=0), l2_tier=True)


def test_describe_states_the_pool_or_that_capacity_blocks_abort():
    line = NO_POOL.describe(host_lcm_block_bytes=BLOCK, blob_bytes=0, l2_tier=False)
    assert "none" in line and "aborts its victim" in line
    sizing = _resolve(_request(ratio=0.25), l2_tier=True)
    line = sizing.describe(host_lcm_block_bytes=BLOCK, blob_bytes=250_000, l2_tier=True)
    assert "0.01 GB (10 LCM blocks" in line
    assert "up to 8 retracted requests (2.00 MB slot-state arena)" in line
    assert "tails to the pool" in line


# ----------------------------------------------------------------------
# The tail unit, from the scheduler's capacity model
# ----------------------------------------------------------------------

PAGE = 64


def _model(specs, packing):
    return capacity_model(
        specs,
        prefix_granularity=PAGE,
        virtual_packing=packing,
        limits=SchedulerLimits(
            role=ts.SchedulerConfig.Role.Fused,
            max_live_requests=4,
            max_scheduled_tokens=2048,
            max_context_len=4096,
            decode_input_tokens=1,
            overlap_schedule_depth=0,
            disable_prefix_cache=False,
        ),
    )


FULL = CacheGroupSpec(
    group_id="full",
    retention="full_history",
    rows_per_page=PAGE,
    entry_stride_tokens=1,
    replayable=False,
)
# Never published to L2: every page of it is tail.
SLIDING = CacheGroupSpec(
    group_id="swa",
    retention="sliding_window",
    rows_per_page=PAGE,
    entry_stride_tokens=1,
    sliding_window_tokens=512,
    replayable=True,
)
# Publishes its checkpoints: only the live block is tail.
STATE = CacheGroupSpec(
    group_id="state",
    retention="full_history",
    replayable=False,
    family="state",
    checkpoint_granularity=PAGE,
)


def test_tail_is_one_page_per_published_group_plus_a_replayable_groups_worst_case():
    specs = (FULL, SLIDING)
    model = _model(specs, {"full": 1, "swa": 1})
    worst_case = model.single_request_group_pages(4096)
    assert worst_case[0] == 4096 // PAGE  # the full group's whole context
    # The replayable group's worst case is the scheduler's (window + chunk).
    assert tail_lcm_blocks_per_request(
        model, specs, 4096
    ) == model.lcm_blocks_needed_for([1, worst_case[1]])
    # The fold follows the packing: four sliding pages per LCM block.
    model = _model(specs, {"full": 2, "swa": 4})
    assert tail_lcm_blocks_per_request(
        model, specs, 4096
    ) == model.lcm_blocks_needed_for([1, worst_case[1]])
    # A state group checkpoints to L2 like the full group: one page each.
    specs = (FULL, STATE)
    model = _model(specs, {"full": 1, "state": 1})
    assert tail_lcm_blocks_per_request(
        model, specs, 4096
    ) == model.lcm_blocks_needed_for([1, 1])
    with pytest.raises(ValueError, match="answers 2 groups for 1 specs"):
        tail_lcm_blocks_per_request(model, (FULL,), 4096)


# ----------------------------------------------------------------------
# The device build hands the executor the knobs plus the tail unit
# ----------------------------------------------------------------------


def _device_request(**overrides):
    from types import SimpleNamespace

    from tokenspeed.runtime.execution.device import _retraction_pool_request

    knobs = dict(
        enable_kvstore=True,
        retraction_snapshot_host_gb=0.0,
        retraction_snapshot_ratio=None,
        retraction_snapshot_max_requests=8,
        disaggregation_mode="null",
        chunked_prefill_size=2048,
        enable_prefix_caching=True,
    )
    knobs.update(overrides)
    specs = (FULL, SLIDING)
    contract = SimpleNamespace(
        group_specs=specs,
        prefix_granularity=PAGE,
        virtual_packing={"full": 1, "swa": 1},
    )
    return _retraction_pool_request(
        SimpleNamespace(**knobs),
        contract,
        max_batch_size=4,
        max_context_len=4096,
        decode_input_tokens=1,
        overlap_schedule_depth=0,
    )


def test_device_build_counts_the_tail_only_for_the_derived_pool_beside_l2():
    request = _device_request()
    assert (request.host_gb, request.ratio, request.max_retracted_requests) == (
        0.0,
        None,
        8,
    )
    specs = (FULL, SLIDING)
    model = _model(specs, {"full": 1, "swa": 1})
    assert request.tail_lcm_blocks_per_request == tail_lcm_blocks_per_request(
        model, specs, 4096
    )
    # Settled by a knob, or sized off the Device KV alone: no model is built.
    for overrides in (
        dict(retraction_snapshot_host_gb=2.0),
        dict(retraction_snapshot_ratio=0.5),
        dict(retraction_snapshot_ratio=0.0, retraction_snapshot_max_requests=0),
        dict(enable_kvstore=False),
    ):
        assert _device_request(**overrides).tail_lcm_blocks_per_request == 0
    assert _device_request(
        retraction_snapshot_ratio=0.0, retraction_snapshot_max_requests=0
    ).disabled
