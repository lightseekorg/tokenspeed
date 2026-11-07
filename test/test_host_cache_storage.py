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

from __future__ import annotations

import pytest
import torch

from tokenspeed.runtime.cache.host_storage import (
    HostCacheStorage,
    compute_host_group_packing,
    compute_host_lcm_block_bytes,
    validate_host_group_packing,
)
from tokenspeed.runtime.cache.transfer.layout import (
    CacheField,
    CacheGroupLayout,
    CacheTransferLayout,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import CacheGroupSpec


@pytest.fixture(autouse=True)
def _cpu_pinned_allocation(monkeypatch):
    allocate = torch.empty

    def empty(*args, **kwargs):
        # Pinned allocation needs a CUDA device; storage behavior uses real
        # CPU tensors so these tests do not acquire a serving GPU.
        if kwargs.get("pin_memory", False):
            kwargs["pin_memory"] = False
        return allocate(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", empty)


def _kv(group_id, *, rows_per_page=1, entry_stride_tokens=1):
    return CacheGroupSpec(
        group_id,
        "full_history",
        rows_per_page=rows_per_page,
        entry_stride_tokens=entry_stride_tokens,
        replayable=False,
    )


def _sliding(group_id, *, rows_per_page, window, replayable):
    return CacheGroupSpec(
        group_id,
        "sliding_window",
        rows_per_page=rows_per_page,
        entry_stride_tokens=1,
        sliding_window_tokens=window,
        replayable=replayable,
    )


def _state(group_id):
    return CacheGroupSpec(group_id, "state", replayable=False)


# The groups of ``_layout()``: a 32-byte KV page and an 80-byte State.
_SPECS = (_kv("full"), _state("state"))


def _layout():
    field = lambda name, size: CacheField(name, 0, 0, size, size)
    return CacheTransferLayout(
        num_lcm_blocks=2,
        groups=(
            CacheGroupLayout("full", 4, (field("k", 32),)),
            CacheGroupLayout("state", 1, (field("state", 80),)),
        ),
        buffers=(object(),),
        consumers=(("k",), ("state",)),
    )


def _payload_layout(groups):
    """A layout with one field per ``(spec, payload_bytes)``, and its specs."""
    layout = CacheTransferLayout(
        num_lcm_blocks=1,
        groups=tuple(
            CacheGroupLayout(
                spec.group_id,
                1,
                (CacheField(f"{spec.group_id}.0", 0, 0, payload, payload),),
            )
            for spec, payload in groups
        ),
        buffers=(object(),),
        consumers=tuple((f"{spec.group_id}.0",) for spec, _ in groups),
    )
    return layout, tuple(spec for spec, _ in groups)


def test_block_offsets_follow_group_packing_without_overlap():
    layout = _layout()
    storage = HostCacheStorage(layout, group_specs=_SPECS, num_host_lcm_blocks=2)

    # The 80-byte State rounds the parent up to three whole KV pages.
    assert storage.host_lcm_block_bytes == 96
    assert storage.host_block_offset(0, 1) == 0
    assert storage.host_block_offset(0, 2) == storage.host_cache_block_bytes[0]
    assert storage.host_block_offset(0, 4) == storage.host_lcm_block_bytes
    assert storage.host_block_offset(1, 1) == 0
    assert storage.host_block_offset(1, 2) == storage.host_lcm_block_bytes


@pytest.mark.parametrize("group_index,block_id", [(0, 0), (0, 7), (1, 3)])
def test_null_and_out_of_range_blocks_are_rejected(group_index, block_id):
    layout = _layout()
    storage = HostCacheStorage(layout, group_specs=_SPECS, num_host_lcm_blocks=2)

    with pytest.raises(IndexError):
        storage.host_block_offset(group_index, block_id)


def test_host_fields_are_packed_without_padding():
    field = lambda name, stride, payload: CacheField(name, 0, 0, stride, payload)
    layout = CacheTransferLayout(
        num_lcm_blocks=10,
        groups=(
            CacheGroupLayout(
                "full",
                16,
                (
                    field("full.k", 96, 40),
                    field("full.v", 96, 40),
                ),
            ),
            CacheGroupLayout(
                "state",
                1,
                (
                    field("state.ssm", 2048, 1000),
                    field("state.conv", 1024, 200),
                ),
            ),
        ),
        buffers=(object(),),
        consumers=(("full.k", "full.v"), ("state.ssm", "state.conv")),
    )

    storage = HostCacheStorage(layout, group_specs=_SPECS, num_host_lcm_blocks=1)

    assert storage.host_cache_block_bytes == (80, 1200)
    assert storage.host_field_offsets == ((0, 40), (0, 1000))
    # 1200 bytes is 15 whole KV pages, so the State payload is the parent.
    assert storage.host_lcm_block_bytes == 1200


def test_host_storage_does_not_retain_device_buffers():
    import dataclasses
    import gc
    import weakref

    class DeviceBuffer:
        pass

    buffer = DeviceBuffer()
    reference = weakref.ref(buffer)
    layout = dataclasses.replace(_layout(), buffers=(buffer,))
    storage = HostCacheStorage(layout, group_specs=_SPECS, num_host_lcm_blocks=2)
    del layout, buffer
    gc.collect()
    assert reference() is None
    assert storage.host_block_range(0, 1) == (0, 32)
    assert len(storage.host_buffer) == 2 * 96


def test_shared_host_storage_packs_payloads_independently_of_device_strides():
    layout = CacheTransferLayout(
        num_lcm_blocks=2,
        groups=(
            CacheGroupLayout("full", 4, (CacheField("kv", 0, 0, 32, 32),)),
            CacheGroupLayout("state", 1, (CacheField("ssm", 0, 0, 80, 80),)),
            CacheGroupLayout("aux", 2, (CacheField("aux", 0, 0, 40, 5),)),
        ),
        buffers=(object(),),
        consumers=(("kv",), ("ssm",), ("aux",)),
    )
    storage = HostCacheStorage(
        layout, group_specs=(*_SPECS, _kv("aux")), num_host_lcm_blocks=2
    )

    # Three 32-byte KV pages make the 96-byte parent: 19 aux payloads fit it.
    assert storage.host_buffer.numel() == 2 * 96
    assert storage.host_block_range(0, 2) == (32, 32)
    assert storage.host_block_range(2, 19) == (90, 5)
    assert storage.host_block_range(2, 20) == (96, 5)
    assert storage.host_block_range(1, 2) == (96, 80)
    # One shared parent can hold history or State. These two occupants use
    # different parents and retain their bytes across the compact boundary.
    storage.host_buffer[32:64].fill_(17)
    storage.host_buffer[96:176].fill_(29)
    assert torch.all(storage.host_buffer[32:64] == 17)
    assert torch.all(storage.host_buffer[96:176] == 29)


# Qwen3.8-Flash-Next's compact Host payloads with FP8 KV, 256-token pages and
# attention TP 1, in scheduler group order.
_QWEN38_GROUPS = (
    (_kv("full_attention", rows_per_page=256), 3_407_872),
    (_state("linear_attention_0"), 29_417_472),
    (_state("linear_attention_1"), 29_417_472),
    (_kv("qwen4_exp_qsa", rows_per_page=64, entry_stride_tokens=4), 212_992),
    (
        _sliding("qwen4_exp_qsa_recent", rows_per_page=64, window=4, replayable=False),
        13_624,
    ),
    (_state("qwen4_exp_ple"), 184_336),
)


def test_parent_packs_whole_kv_pages_beside_a_larger_state_checkpoint():
    layout, specs = _payload_layout(_QWEN38_GROUPS)
    storage = HostCacheStorage(layout, group_specs=specs, num_host_lcm_blocks=2)

    # The State checkpoint is 8.63 full-attention pages: 8 pages leave a 7.3%
    # gap in every full-attention parent. The parent rounds up to 9 pages,
    # which is also 144 QSA pages, so neither KV group leaves a gap, and each
    # State checkpoint still takes one parent.
    parent = 9 * 3_407_872
    assert storage.host_lcm_block_bytes == parent == 144 * 212_992 == 30_670_848
    assert storage.group_packing == (9, 1, 1, 144, 2_251, 166)
    assert compute_host_lcm_block_bytes(layout, group_specs=specs) == parent
    assert compute_host_group_packing(layout, group_specs=specs) == (
        storage.group_packing
    )
    assert storage.host_buffer_bytes == 2 * parent
    # The ninth page ends the first parent; the tenth opens the second.
    assert storage.host_block_range(0, 9) == (8 * 3_407_872, 3_407_872)
    assert storage.host_block_range(0, 10) == (parent, 3_407_872)
    assert storage.host_block_range(1, 2) == (parent, 29_417_472)
    assert storage.host_block_range(3, 145) == (parent, 212_992)


def _glm53_flash_groups(*, first_state, other_state):
    """GLM-5.3-Flash's compact Host payloads at attention TP 8 with FP8 KV and
    MTP: three KDA checkpoints, sized by the state dtype, and the 64-token DSA
    page."""
    return (
        (_state("linear_attention_0"), first_state),
        (_state("linear_attention_1"), other_state),
        (_state("linear_attention_2"), other_state),
        (_kv("full_attention", rows_per_page=64), 418_560),
    )


@pytest.mark.parametrize(
    "groups,parent",
    [
        # FP32 KDA state: 15 pages leave a 234,240-byte gap in every KV
        # parent, and 16 pages close it for 184,320 more bytes of slack in
        # every State parent.
        pytest.param(
            _glm53_flash_groups(first_state=6_512_640, other_state=5_969_920),
            16 * 418_560,
            id="fp32-state-rounds-up",
        ),
        # BF16 KDA state: 8 pages leave an 18,432-byte gap; 9 pages would add
        # 400,128 bytes of slack to every State parent to close it, which
        # costs more than it saves, so the parent stays the largest payload.
        pytest.param(
            _glm53_flash_groups(first_state=3_366_912, other_state=3_086_336),
            3_366_912,
            id="bf16-state-keeps-the-largest-payload",
        ),
    ],
)
def test_parent_rounds_up_only_when_that_lowers_host_bytes_per_token(groups, parent):
    layout, specs = _payload_layout(groups)

    assert compute_host_lcm_block_bytes(layout, group_specs=specs) == parent
    assert compute_host_group_packing(layout, group_specs=specs) == tuple(
        parent // payload for _, payload in groups
    )


@pytest.mark.parametrize(
    "groups,parent",
    [
        pytest.param(_QWEN38_GROUPS[::-1], 9 * 3_407_872, id="largest-kv-page-last"),
        # A compressed page larger than the full-attention page but spanning
        # four times the tokens: full attention stores more Host bytes per
        # token, so the parent takes 9 of its pages, not 6 compressed pages.
        pytest.param(
            (
                (_kv("compressed", rows_per_page=64, entry_stride_tokens=4), 1_600),
                (_state("state"), 8_600),
                (_kv("full", rows_per_page=64), 1_000),
            ),
            9_000,
            id="larger-page-with-fewer-bytes-per-token-first",
        ),
    ],
)
def test_parent_rounds_to_the_kv_group_with_the_most_host_bytes_per_token(
    groups, parent
):
    layout, specs = _payload_layout(groups)

    assert compute_host_lcm_block_bytes(layout, group_specs=specs) == parent


def test_a_replayable_group_neither_picks_nor_prices_the_rounding():
    # Host L2 never stores a replayable sliding window, so although the window
    # has the most bytes per token, the parent rounds to whole full-attention
    # pages, and the window's unchanged packing does not count against that.
    groups = (
        (_sliding("swa", rows_per_page=64, window=128, replayable=True), 2_300),
        (_state("state"), 8_600),
        (_kv("full", rows_per_page=64), 1_000),
    )
    layout, specs = _payload_layout(groups)

    assert compute_host_lcm_block_bytes(layout, group_specs=specs) == 9_000


def test_a_replayable_window_as_the_largest_payload_rounds_like_a_state_checkpoint():
    # Pure attention whose largest payload is a replayable sliding window beside
    # compressed KV and compressor tails. Host L2 never stores the window, so it
    # only sizes the parent: 1,351,680 bytes would hold 19 global_r2 and 59
    # global_r1 pages; 20 global_r2 pages hold exactly 60 global_r1 pages too.
    groups = (
        (_sliding("swa", rows_per_page=64, window=128, replayable=True), 1_351_680),
        (_kv("global_r2", rows_per_page=64, entry_stride_tokens=2), 68_352),
        (_kv("global_r1", rows_per_page=64), 22_784),
        (_sliding("tail", rows_per_page=2, window=2, replayable=True), 24_576),
    )
    layout, specs = _payload_layout(groups)

    assert compute_host_lcm_block_bytes(layout, group_specs=specs) == 20 * 68_352
    assert compute_host_group_packing(layout, group_specs=specs) == (1, 20, 60, 55)


@pytest.mark.parametrize(
    "groups",
    [
        pytest.param(
            ((_kv("full_attention", rows_per_page=256), 3_407_872),),
            id="one-kv-group",
        ),
        # The cached KV page is the largest payload; the replayable window only
        # packs beside it.
        pytest.param(
            (
                (_kv("full_attention", rows_per_page=256), 3_407_872),
                (
                    _sliding("swa", rows_per_page=64, window=128, replayable=True),
                    1_351_680,
                ),
            ),
            id="cached-kv-page-largest-beside-a-replayable-window",
        ),
        pytest.param(
            (
                (_state("linear_attention"), 3_000_000),
                (_kv("full_attention", rows_per_page=256), 3_407_872),
            ),
            id="kv-page-larger-than-state",
        ),
        pytest.param(
            (
                (_kv("full_attention", rows_per_page=256), 3_407_872),
                (_state("linear_attention"), 8 * 3_407_872),
            ),
            id="state-of-whole-kv-pages",
        ),
        pytest.param(
            ((_state("linear_attention_0"), 8_600), (_state("conv"), 3_000)),
            id="state-only",
        ),
        # Host L2 never stores the replayable window, so no KV page is cached.
        pytest.param(
            (
                (_state("state"), 8_600),
                (_sliding("swa", rows_per_page=64, window=128, replayable=True), 1_000),
            ),
            id="state-beside-a-replayable-window-only",
        ),
    ],
)
def test_parent_stays_the_largest_payload(groups):
    layout, specs = _payload_layout(groups)
    largest = max(payload for _, payload in groups)

    assert compute_host_lcm_block_bytes(layout, group_specs=specs) == largest
    assert compute_host_group_packing(layout, group_specs=specs) == tuple(
        largest // payload for _, payload in groups
    )


def _check_host_packing_agreement(rank, rendezvous, peer_packing):
    from datetime import timedelta

    import torch.distributed as dist

    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=60),
    )
    try:
        packing = (2, 1) if rank == 0 else peer_packing
        if peer_packing == (2, 1):
            validate_host_group_packing(packing, cpu_group=dist.group.WORLD)
        else:
            with pytest.raises(ValueError, match="Host packing must match"):
                validate_host_group_packing(packing, cpu_group=dist.group.WORLD)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("peer_packing", [(2, 1), (3, 1), (2, 1, 1)])
def test_mirrored_ranks_accept_only_identical_host_packing(tmp_path, peer_packing):
    import torch.multiprocessing as mp

    mp.spawn(
        _check_host_packing_agreement,
        args=(f"file://{tmp_path}/gloo", peer_packing),
        nprocs=2,
        join=True,
    )
