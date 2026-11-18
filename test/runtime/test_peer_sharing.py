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
    peer_incarnation,
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
    args.disaggregation_mode = "null"
    args.enable_kvstore = True
    args.enable_prefix_caching = True
    args.kvstore_storage_backend = None
    args.pipeline_parallel_size = 1
    args.query_context_parallel_size = 1
    args.mapping = Mapping(rank=0, world_size=2, attn_tp_size=1, attn_dp_size=2)
    for name, value in overrides.items():
        setattr(args, name, value)
    return args


def test_peer_sharing_runs_with_the_host_tier_and_prefix_caching():
    _peer_args().validate_peer_prefix_options()
    # Off, nothing is checked.
    _peer_args(
        peer_prefix_port=None, enable_kvstore=False
    ).validate_peer_prefix_options()


@pytest.mark.parametrize(
    "overrides,refused",
    [
        ({"enable_kvstore": False}, "--disable-kvstore"),
        ({"enable_prefix_caching": False}, "--disable-prefix-caching"),
        ({"disaggregation_mode": "decode"}, "--disaggregation-mode decode"),
        ({"kvstore_storage_backend": "mooncake"}, "--kvstore-storage-backend"),
        ({"pipeline_parallel_size": 2}, "--pipeline-parallel-size"),
        ({"query_context_parallel_size": 2}, "--query-context-parallel-size"),
    ],
)
def test_peer_sharing_refuses_what_it_does_not_run_with(overrides, refused):
    with pytest.raises(ValueError, match=refused):
        _peer_args(**overrides).validate_peer_prefix_options()


def test_every_attention_dp_rank_needs_a_valid_port():
    # Rank 1 binds 65535.
    _peer_args(peer_prefix_port=65534).validate_peer_prefix_options()
    with pytest.raises(ValueError, match="valid ports"):
        _peer_args(peer_prefix_port=65535).validate_peer_prefix_options()
    with pytest.raises(ValueError, match="valid ports"):
        _peer_args(peer_prefix_port=0).validate_peer_prefix_options()


@pytest.mark.parametrize(
    "overrides",
    [{"peer_prefix_answer_timeout_ms": 0}, {"peer_prefix_min_fetch_tokens": 0}],
)
def test_peer_limits_must_be_positive(overrides):
    with pytest.raises(ValueError, match="must be positive"):
        _peer_args(**overrides).validate_peer_prefix_options()
