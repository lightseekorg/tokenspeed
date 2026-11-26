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

"""Peer prefix sharing's startup values (``tokenspeed.runtime.cache.peer``):
the geometry a peer reads this rank's cache blocks by, checked against the
Host pool's own offsets and the Device layout's own, and the server
arguments it runs with."""

from __future__ import annotations

import os
import sys

import pytest
import torch

# CI registration (AST-parsed, runtime no-op).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

from tokenspeed.runtime.cache.host_storage import HostCacheStorage  # noqa: E402
from tokenspeed.runtime.cache.peer import (  # noqa: E402
    build_peer_geometry,
    peer_host_regions,
    peer_incarnation,
    peer_scheduler_config,
    peer_semantic_id,
)
from tokenspeed.runtime.cache.transfer.layout import (  # noqa: E402
    CacheField,
    CacheGroupLayout,
    CacheTransferLayout,
)
from tokenspeed.runtime.distributed.mapping import Mapping  # noqa: E402
from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import (  # noqa: E402
    CacheGroupSpec,
)
from tokenspeed.runtime.utils.server_args import ServerArgs  # noqa: E402

register_cuda_ci(est_time=5, suite="runtime-1gpu")

NUM_LCM_BLOCKS = 6


def test_peer_identity_binds_the_generation_and_effective_state_encoding():
    abi = {"state": "native-kda-v1", "index_rope": False}
    base = peer_semantic_id("model", None, "7", abi)
    assert len(base) == 32
    assert peer_semantic_id("model", "main", "7", dict(reversed(abi.items()))) == base
    assert peer_semantic_id("another-model", None, "7", abi) != base
    assert peer_semantic_id("model", None, "8", abi) != base
    assert peer_semantic_id("model", None, "7", {**abi, "index_rope": True}) != base
    assert peer_semantic_id("model", "main", "default", abi) == b""
    assert len(peer_semantic_id("model", "a" * 40, "default", abi)) == 32
    assert peer_semantic_id("model", None, "7", None) == b""


def test_peer_host_region_plans_cover_the_pool_without_splitting_a_parent():
    assert peer_host_regions(100, 42, 6, None) == ((100, 42),)
    assert peer_host_regions(100, 42, 6, lambda length, parent: (24, 18)) == (
        (100, 24),
        (124, 18),
    )


@pytest.mark.parametrize("lengths", [(12,), (18, 25), (17, 25), (0, 42), (-6, 48)])
def test_invalid_peer_host_registration_plans_are_refused(lengths):
    with pytest.raises(ValueError, match="(region|regions)"):
        peer_host_regions(100, 42, 6, lambda *_: lengths)


def test_release_only_loaded_checkpoint_file_pages(tmp_path, monkeypatch):
    from tokenspeed.runtime.cache.l2.executor import release_checkpoint_file_cache

    shard = tmp_path / "model.safetensors"
    shard.write_bytes(b"weights")
    (tmp_path / "config.json").write_text("{}")
    advised = []
    monkeypatch.setattr(
        os,
        "posix_fadvise",
        lambda fd, start, length, advice: advised.append(
            (os.fstat(fd).st_ino, start, length, advice)
        ),
    )
    release_checkpoint_file_cache([str(tmp_path)], node_leader=False, cpu_group=None)
    assert not advised
    release_checkpoint_file_cache([str(tmp_path)] * 2, node_leader=True, cpu_group=None)
    assert advised == [(shard.stat().st_ino, 0, 0, os.POSIX_FADV_DONTNEED)]
    assert shard.read_bytes() == b"weights"


def _spec(group_id: str, attn_kind: str) -> CacheGroupSpec:
    return CacheGroupSpec(
        group_id,
        attn_kind,
        rows_per_page=1,
        entry_stride_tokens=2,
        replayable=False,
    )


def _setup():
    """A Device buffer of two groups, packed differently on the Host: a full
    attention group of two fields and a smaller one of one field."""
    groups = (
        CacheGroupLayout(
            "full",
            1,
            (
                # Padded pages: the stride exceeds the payload.
                CacheField("layer.0.k", 0, 0, 16, 12),
                CacheField("layer.0.v", 0, 16 * (NUM_LCM_BLOCKS + 1), 16, 12),
            ),
        ),
        CacheGroupLayout(
            "swa",
            1,
            (CacheField("layer.1.k", 0, 32 * (NUM_LCM_BLOCKS + 1), 4, 4),),
        ),
    )
    device = torch.zeros(36 * (NUM_LCM_BLOCKS + 1), dtype=torch.uint8)
    layout = CacheTransferLayout(
        num_lcm_blocks=NUM_LCM_BLOCKS,
        groups=groups,
        buffers=(device,),
        consumers=(("layer.0.k", "layer.0.v"), ("layer.1.k",)),
    )
    storage = HostCacheStorage(
        layout,
        group_specs=(_spec("full", "full_history"), _spec("swa", "full_history")),
        num_host_lcm_blocks=NUM_LCM_BLOCKS,
        block_alignment=1,
        align_host_buffer=False,
    )
    field_addresses = {
        field.field_id: device.data_ptr() + field.device_block_zero_offset_bytes
        for group in groups
        for field in group.fields
    }
    return groups, device, storage, field_addresses


@pytest.mark.skipif(not torch.cuda.is_available(), reason="pinned Host memory")
def test_the_geometry_addresses_the_bytes_the_cache_holds():
    groups, device, storage, field_addresses = _setup()
    geometry = build_peer_geometry(
        scheduler_group_ids=("full", "swa"),
        groups=groups,
        field_addresses=field_addresses,
        device_regions=((device.data_ptr(), device.numel()),),
        host_storage=storage,
    )

    host_base = storage.host_buffer.data_ptr()
    for field in geometry.device:
        group = geometry.host.groups[field.group]
        index = [f.field_id for f in groups[field.group].fields].index(field.field_id)
        offsets = dict(group.field_offsets)
        for block in range(1, NUM_LCM_BLOCKS * group.packing + 1):
            # The Host formula the reader computes, against the pool's own.
            zero_based = block - 1
            host = (
                geometry.host.address
                + zero_based // group.packing * group.parent_bytes
                + zero_based % group.packing * group.block_bytes
                + offsets[field.field_id]
            )
            assert host == host_base + storage.host_field_offset(
                field.group, block, index
            )
        for page in range(NUM_LCM_BLOCKS + 1):
            # The Device page the reader writes, against the layout's own.
            layout_field = groups[field.group].fields[index]
            expected = (
                device.data_ptr()
                + layout_field.device_block_zero_offset_bytes
                + page * layout_field.block_stride_bytes
            )
            assert field.address + page * field.page_stride == expected
            assert field.bytes == layout_field.payload_bytes
    # Both memories are registered for the transfer engine.
    assert geometry.regions == (
        (device.data_ptr(), device.numel()),
        (host_base, storage.host_buffer_bytes),
    )

    split_first = 2 * storage.host_lcm_block_bytes
    split_rest = storage.host_buffer_bytes - split_first
    split_geometry = build_peer_geometry(
        scheduler_group_ids=("full", "swa"),
        groups=groups,
        field_addresses=field_addresses,
        device_regions=((device.data_ptr(), device.numel()),),
        host_storage=storage,
        host_region_planner=lambda length, parent: (2 * parent, length - 2 * parent),
    )
    assert split_geometry.regions == (
        (device.data_ptr(), device.numel()),
        (host_base, split_first),
        (host_base + split_first, split_rest),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="pinned Host memory")
def test_a_host_layout_out_of_the_schedulers_order_is_refused():
    groups, device, storage, field_addresses = _setup()
    with pytest.raises(ValueError, match="scheduler's order"):
        build_peer_geometry(
            scheduler_group_ids=("swa", "full"),
            groups=groups,
            field_addresses=field_addresses,
            device_regions=((device.data_ptr(), device.numel()),),
            host_storage=storage,
        )


def test_an_incarnation_fits_the_schedulers_lease_ids():
    for _ in range(64):
        assert 0 < peer_incarnation() < 1 << 32


def _peer_args(**overrides) -> ServerArgs:
    """Resolved arguments of a replica with peer sharing and two
    attention-DP ranks, as ``validate_peer_prefix_options`` reads them."""
    args = object.__new__(ServerArgs)
    args.peer_prefix_port = 7100
    args.peer_prefix_answer_timeout_ms = 50
    args.peer_prefix_min_fetch_tokens = 256
    args.peer_prefix_landing_tier = "device"
    args.disaggregation_mode = "null"
    args.enable_kvstore = True
    args.enable_prefix_caching = True
    args.kvstore_storage_backend = None
    args.pipeline_parallel_size = 1
    args.query_context_parallel_size = 1
    args.enable_memory_saver = False
    args.mapping = Mapping(rank=0, world_size=2, attn_tp_size=1, attn_dp_size=2)
    for name, value in overrides.items():
        setattr(args, name, value)
    return args


def test_peer_sharing_runs_with_the_host_tier_and_prefix_caching():
    _peer_args().validate_peer_prefix_options()
    # A decode replica holds the history it decoded (RFC 0005).
    _peer_args(disaggregation_mode="decode").validate_peer_prefix_options()
    # Off, nothing is checked but a stray landing tier.
    _peer_args(
        peer_prefix_port=None, enable_kvstore=False, peer_prefix_landing_tier=None
    ).validate_peer_prefix_options()
    with pytest.raises(ValueError, match="requires --peer-prefix-port"):
        _peer_args(peer_prefix_port=None).validate_peer_prefix_options()


@pytest.mark.parametrize("tier", [None, "disk"])
def test_peer_sharing_needs_an_explicit_landing_tier(tier):
    with pytest.raises(ValueError, match="--peer-prefix-landing-tier"):
        _peer_args(peer_prefix_landing_tier=tier).validate_peer_prefix_options()


def test_host_landings_take_a_share_of_the_host_blocks():
    ts = pytest.importorskip("tokenspeed_engine.scheduler")
    common = dict(
        name="10.0.0.2:7100",
        incarnation=3,
        answer_timeout_ms=50,
        min_fetch_tokens=256,
        num_host_blocks=100,
        versioned_shards=False,
    )
    host = peer_scheduler_config(landing_tier="host", **common)
    assert host.landing_tier == ts.PeerLandingTier.Host
    assert host.max_host_landing_blocks == 50
    assert host.host_ready_timeout_us > 0
    device = peer_scheduler_config(landing_tier="device", **common)
    assert device.landing_tier == ts.PeerLandingTier.Device
    assert (device.max_host_landing_blocks, device.host_ready_timeout_us) == (0, 0)


def test_peer_scheduler_config_scales_soft_pin_and_host_ready_with_answer_timeout():
    pytest.importorskip("tokenspeed_engine.scheduler")
    base = {
        "name": "10.0.0.2:7100",
        "incarnation": 3,
        "min_fetch_tokens": 256,
        "num_host_blocks": 100,
        "versioned_shards": False,
    }
    # 50 ms: soft_pin = max(2.0, 2*0.05 + 2.0) = 2.1s, host_ready = max(5.0, 0.05 + 5.0) = 5.05s
    low = peer_scheduler_config(landing_tier="host", answer_timeout_ms=50, **base)
    assert low.soft_pin_us == 2_100_000
    assert low.host_ready_timeout_us == 5_050_000
    assert low.lookup_timeout_us == 50_000

    # 1000 ms: soft_pin = 4.0s, host_ready = 6.0s
    mid = peer_scheduler_config(landing_tier="host", answer_timeout_ms=1000, **base)
    assert mid.soft_pin_us == 4_000_000
    assert mid.host_ready_timeout_us == 6_000_000
    assert mid.lookup_timeout_us == 1_000_000

    # 5000 ms: soft_pin = 12.0s (> 2.0s hardcoded floor!), host_ready = 10.0s
    high = peer_scheduler_config(landing_tier="host", answer_timeout_ms=5000, **base)
    assert high.soft_pin_us == 12_000_000
    assert high.host_ready_timeout_us == 10_000_000
    assert high.lookup_timeout_us == 5_000_000

    # Device landing scales soft_pin without dropping below the 2 s SOFT_PIN_S
    # floor or exceeding HARD_PIN_S, while leaving host_ready_timeout_us == 0.
    dev_low = peer_scheduler_config(landing_tier="device", answer_timeout_ms=50, **base)
    assert dev_low.soft_pin_us == 2_200_000
    assert dev_low.host_ready_timeout_us == 0

    dev_mid = peer_scheduler_config(
        landing_tier="device", answer_timeout_ms=1000, **base
    )
    assert dev_mid.soft_pin_us == 6_000_000
    assert dev_mid.host_ready_timeout_us == 0

    dev_high = peer_scheduler_config(
        landing_tier="device", answer_timeout_ms=5000, **base
    )
    assert dev_high.soft_pin_us == 22_000_000
    assert dev_high.host_ready_timeout_us == 0

    # Large answer timeouts clamp soft_pin to HARD_PIN_S (30 s) so a Pin never
    # shortens a lease and RECORD_LIFETIME_S always outlives it.
    dev_clamped = peer_scheduler_config(
        landing_tier="device", answer_timeout_ms=12000, **base
    )
    assert dev_clamped.soft_pin_us == 30_000_000 == dev_clamped.hard_pin_us
    host_clamped = peer_scheduler_config(
        landing_tier="host", answer_timeout_ms=20000, **base
    )
    assert host_clamped.soft_pin_us == 30_000_000 == host_clamped.hard_pin_us


@pytest.mark.parametrize(
    "overrides,refused",
    [
        ({"enable_kvstore": False}, "--disable-kvstore"),
        ({"enable_prefix_caching": False}, "--disable-prefix-caching"),
        ({"kvstore_storage_backend": "mooncake"}, "--kvstore-storage-backend"),
        ({"pipeline_parallel_size": 2}, "--pipeline-parallel-size"),
        ({"query_context_parallel_size": 2}, "--query-context-parallel-size"),
        ({"enable_memory_saver": True}, "--enable-memory-saver.*Mooncake MRs"),
    ],
)
def test_peer_sharing_refuses_what_it_does_not_run_with(overrides, refused):
    with pytest.raises(ValueError, match=refused):
        _peer_args(**overrides).validate_peer_prefix_options()


def test_attention_dp_replicas_share_one_public_port():
    from tokenspeed.runtime.cache.peer import peer_endpoint_name

    _peer_args(peer_prefix_port=65535).validate_peer_prefix_options()
    assert peer_endpoint_name("::1", 65535, 0) == "[::1]:65535"
    assert peer_endpoint_name("::1", 65535, 31) == "[::1]:65535/31"
    for port in (0, 65536):
        with pytest.raises(ValueError, match="1..65535"):
            _peer_args(peer_prefix_port=port).validate_peer_prefix_options()


@pytest.mark.parametrize(
    "overrides",
    [
        {"peer_prefix_answer_timeout_ms": 0},
        {"peer_prefix_min_fetch_tokens": 0},
    ],
)
def test_peer_limits_must_be_positive(overrides):
    with pytest.raises(ValueError, match="must be positive"):
        _peer_args(**overrides).validate_peer_prefix_options()
