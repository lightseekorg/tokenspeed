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

"""The local Disk tier's runtime side (``docs/rfcs/0006-disk-tier.md``): its
server arguments and tuning, the engine host's settings over the Host pool,
and the scheduler's Disk operations as the plan adapter routes them."""

from __future__ import annotations

from test.ci_system.ci_register import register_cuda_ci
from test.runtime.scheduler_driver import (
    UNBOUNDED_CACHED_PREFIX_TOKENS,
    advance_scheduler,
    make_extend_result_event,
    make_finish_event,
    make_spec,
)

import pytest
import torch

register_cuda_ci(est_time=10, suite="runtime-1gpu")

ts = pytest.importorskip("tokenspeed_engine.scheduler")

from tokenspeed.runtime.cache.disk import (  # noqa: E402
    DIRECT_IO_ALIGNMENT,
    DiskTierTuning,
    disk_scheduler_config,
    disk_settings,
    host_block_alignment,
)
from tokenspeed.runtime.cache.host_storage import HostCacheStorage  # noqa: E402
from tokenspeed.runtime.cache.transfer.layout import (  # noqa: E402
    CacheField,
    CacheGroupLayout,
    CacheTransferLayout,
)
from tokenspeed.runtime.cache.transfer.ownership import (  # noqa: E402
    BlockOwnerTranslation,
)
from tokenspeed.runtime.distributed.mapping import Mapping  # noqa: E402
from tokenspeed.runtime.engine.scheduler_utils import (  # noqa: E402
    cache_ops_from_plan,
    make_config,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import (  # noqa: E402
    CacheGroupSpec,
)
from tokenspeed.runtime.utils.server_args import ServerArgs  # noqa: E402


def _disk_args(**overrides) -> ServerArgs:
    """Resolved arguments of a two-rank replica with a Disk tier on two
    drives, as the Disk tier's validation and settings read them."""
    args = object.__new__(ServerArgs)
    args.disk_tier_paths = ["/nvme0/ts", "/nvme1/ts"]
    args.disk_tier_size_gb = 1.0
    args.disk_tier_io = "io_uring"
    args.disk_tier_write_mbps = 500
    args.disk_tier_config = None
    args.enable_kvstore = True
    args.kvstore_write_policy = "write_back"
    args.enable_prefix_caching = True
    args.kvstore_storage_backend = None
    args.pipeline_parallel_size = 1
    args.port = 30000
    args.mapping = Mapping(rank=1, world_size=2, attn_tp_size=2, attn_dp_size=1)
    for name, value in overrides.items():
        setattr(args, name, value)
    return args


def test_the_disk_tier_runs_under_a_write_back_host_tier():
    _disk_args().validate_disk_tier_options()
    # Off, a stray setting is refused rather than ignored.
    _disk_args(
        disk_tier_paths=None, disk_tier_io=None, disk_tier_write_mbps=None
    ).validate_disk_tier_options()
    with pytest.raises(ValueError, match="--disk-tier-io requires"):
        _disk_args(
            disk_tier_paths=None, disk_tier_write_mbps=None
        ).validate_disk_tier_options()


@pytest.mark.parametrize(
    "overrides,refused",
    [
        ({"enable_kvstore": False}, "--disable-kvstore"),
        ({"kvstore_write_policy": "write_through"}, "write_back"),
        ({"enable_prefix_caching": False}, "--disable-prefix-caching"),
        ({"kvstore_storage_backend": "mooncake"}, "--kvstore-storage-backend"),
        ({"pipeline_parallel_size": 2}, "--pipeline-parallel-size"),
        ({"disk_tier_paths": []}, "at least one directory"),
        ({"disk_tier_size_gb": 0.0}, "--disk-tier-size-gb"),
        ({"disk_tier_io": None}, "--disk-tier-io"),
        ({"disk_tier_write_mbps": None}, "--disk-tier-write-mbps"),
        ({"disk_tier_config": '{"queue_depht": 4}'}, "unknown"),
    ],
)
def test_the_disk_tier_refuses_what_it_does_not_run_with(overrides, refused):
    with pytest.raises(ValueError, match=refused):
        _disk_args(**overrides).validate_disk_tier_options()


def test_only_a_disk_tier_aligns_the_host_layout():
    assert host_block_alignment(_disk_args()) == DIRECT_IO_ALIGNMENT
    assert host_block_alignment(_disk_args(disk_tier_paths=None)) == 1


def test_tuning_takes_each_key_over_its_default():
    assert DiskTierTuning.from_json(None) == DiskTierTuning()
    tuning = DiskTierTuning.from_json('{"queue_depth": 64, "write_queue_depth": 4}')
    assert (tuning.queue_depth, tuning.write_queue_depth) == (64, 4)
    assert tuning.max_operations == DiskTierTuning().max_operations


@pytest.mark.parametrize(
    "text,refused",
    [
        ("[1]", "JSON object"),
        ('{"write_queue_depth": 64}', "write_queue_depth"),
        ('{"write_queue_depth": 0}', "write_queue_depth"),
        ('{"max_io_bytes": 1000}', "max_io_bytes"),
        ('{"host_blocks_fraction": 0}', "host_blocks_fraction"),
        ('{"write_host_blocks_fraction": 1.5}', "write_host_blocks_fraction"),
    ],
)
def test_tuning_refuses_values_the_executor_cannot_run(text, refused):
    with pytest.raises(ValueError, match=refused):
        DiskTierTuning.from_json(text)


def test_the_scheduler_limits_take_their_shares_of_the_host_pool():
    config = disk_scheduler_config(
        DiskTierTuning(host_blocks_fraction=0.25, write_host_blocks_fraction=0.5),
        num_host_blocks=100,
    )
    assert (config.max_host_blocks, config.max_write_host_blocks) == (25, 12)
    assert (config.wait_timeout_us, config.ready_timeout_us) == (10_000_000, 5_000_000)
    # A tiny pool still lends one block to each.
    config = disk_scheduler_config(DiskTierTuning(), num_host_blocks=2)
    assert (config.max_host_blocks, config.max_write_host_blocks) == (1, 1)


def _host_storage(monkeypatch) -> HostCacheStorage:
    """A two-group Host pool laid out for direct I/O: a 6000-byte KV page
    and a 10000-byte State checkpoint."""
    allocate = torch.empty

    def empty(*args, **kwargs):
        # Pinned allocation needs a CUDA device; the settings read only the
        # pool's address and geometry.
        kwargs["pin_memory"] = False
        return allocate(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", empty)
    field = lambda name, size: CacheField(name, 0, 0, size, size)
    layout = CacheTransferLayout(
        num_lcm_blocks=2,
        groups=(
            CacheGroupLayout("full", 1, (field("k", 3000), field("v", 3000))),
            CacheGroupLayout("state", 1, (field("state", 10000),)),
        ),
        buffers=(object(),),
        consumers=(("k", "v"), ("state",)),
    )
    specs = (
        CacheGroupSpec(
            "full",
            "full_history",
            rows_per_page=1,
            entry_stride_tokens=1,
            replayable=False,
        ),
        CacheGroupSpec("state", "state", replayable=False),
    )
    return HostCacheStorage(
        layout,
        group_specs=specs,
        num_host_lcm_blocks=8,
        block_alignment=DIRECT_IO_ALIGNMENT,
    )


def test_settings_lay_the_file_out_as_the_host_pool(monkeypatch):
    storage = _host_storage(monkeypatch)
    parent = storage.host_lcm_block_bytes
    args = _disk_args(disk_tier_size_gb=(10 * parent + 1) / 1e9)
    ownership = BlockOwnerTranslation(
        shard_counts=(2, 1),
        device_virtual_counts=(17, 17),
        host_virtual_counts=(
            1 + 8 * storage.group_packing[0] * 2,
            1 + 8 * storage.group_packing[1],
        ),
        rank=1,
    )

    settings = disk_settings(
        args,
        tuning=DiskTierTuning(queue_depth=16, write_queue_depth=2),
        host_storage=storage,
        ownership=ownership,
        global_rank=3,
    )

    # Ranks spread over the drives; the port keeps engines apart.
    assert settings.file == "/nvme1/ts/tokenspeed-disk-p30000-r3.bin"
    # The size holds ten whole parents, each a Host parent's bytes.
    assert settings.num_disk_lcm_blocks == 10
    assert settings.file_bytes == 10 * parent
    assert settings.lcm_block_bytes == parent
    assert settings.file_bytes % DIRECT_IO_ALIGNMENT == 0
    assert settings.groups == (
        (storage.group_packing[0], 8192, 2),
        (storage.group_packing[1], 12288, 1),
    )
    assert (settings.host_buffer, settings.host_buffer_bytes) == (
        storage.host_buffer.data_ptr(),
        8 * parent,
    )
    assert settings.num_host_lcm_blocks == 8
    assert (settings.owner_rank, settings.tp_rank) == (1, 1)
    assert (settings.backend, settings.write_bytes_per_second) == (
        "io_uring",
        500_000_000,
    )
    assert (settings.queue_depth, settings.write_queue_depth) == (16, 2)

    with pytest.raises(ValueError, match="holds no"):
        disk_settings(
            _disk_args(disk_tier_size_gb=(parent - 1) / 1e9),
            tuning=DiskTierTuning(),
            host_storage=storage,
            ownership=ownership,
            global_rank=0,
        )


def _disk_scheduler():
    """A real scheduler with eight usable Device blocks over a write-back
    Host tier and a Disk tier."""
    scheduler = ts.Scheduler(
        make_config(
            num_device_pages=9,
            max_scheduled_tokens=64,
            max_batch_size=8,
            max_resident_requests=8,
            prefix_granularity=4,
            num_host_pages=65,
            disable_l2_cache=False,
            enable_l3_storage=False,
            role="fused",
            num_snapshot_pages=9,
            host_group_packing=(1,),
            max_retracted_requests=8,
            storage_config=None,
            num_disk_pages=33,
            disk_config=disk_scheduler_config(
                DiskTierTuning(min_prefetch_tokens=1), num_host_blocks=64
            ),
            host_write_policy="write_back",
            prefill_decode_interval=0,
            prefill_decode_min_interval=0,
            cache_groups=[
                ts.CacheGroupConfig(
                    group_id="full",
                    block_granularity=4,
                    total_pages=9,
                    attn_kind=ts.AttnKind.Full,
                )
            ],
        )
    )
    advance_scheduler(
        scheduler, [ts.Cache.StorageTick(1, True, True, ts.StoragePressure(0, 0, 0))]
    )
    return scheduler


def _run(scheduler, request_id: str, tokens: list[int]):
    """Admit, prefill and finish one request; its admission plan."""
    scheduler.submit_requests(
        [
            make_spec(
                request_id,
                tokens,
                max_cached_prefix_tokens=UNBOUNDED_CACHED_PREFIX_TOKENS,
                max_new_tokens=1,
            )
        ]
    )
    plan = scheduler.next_execution_plan()
    advance_scheduler(scheduler, [make_extend_result_event(request_id, [900])])
    advance_scheduler(scheduler, [make_finish_event(request_id)])
    return plan


def test_a_written_back_page_is_cleaned_to_disk_by_the_host_executors():
    scheduler = _disk_scheduler()
    _run(scheduler, "seed", list(range(1, 9)))
    # The 23-token prompt evicts seed's second page, which goes to Host.
    pressure = _run(scheduler, "pressure", list(range(101, 124)))
    (write_back,) = [
        op for op in pressure.cache if isinstance(op, ts.Cache.WriteBackOp)
    ]
    acks = []
    for op_id in write_back.op_ids:
        ack = ts.Cache.WriteBackDoneEvent()
        ack.op_id = op_id
        acks.append(ack)
    advance_scheduler(scheduler, acks)
    assert scheduler.disk_stats().queued_cleanings == 1

    operations = cache_ops_from_plan(
        scheduler.next_execution_plan(), request_admission={}.__getitem__
    )

    # The engine host's Disk executor runs the write; no Device copy does.
    assert operations.device == ()
    (write,) = operations.storage
    assert isinstance(write, ts.Cache.DiskOp)
    assert write.kind == ts.Cache.DiskOpKind.Write
    assert len(write.group_ids) == len(write.host_blocks) == len(write.disk_blocks) == 1
    assert scheduler.disk_stats().operations == 1
    advance_scheduler(scheduler, [ts.Cache.DiskDone(write.op_id, True)])
    stats = scheduler.disk_stats()
    assert (stats.operations, stats.disk_blocks, stats.written_blocks) == (0, 1, 1)
