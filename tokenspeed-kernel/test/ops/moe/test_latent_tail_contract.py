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

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from tokenspeed_kernel.ops.moe.latent_tail import (
    KimiK3LatentTailOp,
    _allocator_identity,
    _tail_pool_slot,
)


def test_initialize_constructs_a_fresh_op(monkeypatch: pytest.MonkeyPatch) -> None:
    constructed = []

    def fake_init(self, *args, **kwargs) -> None:
        constructed.append((self, args, kwargs))

    monkeypatch.setattr(KimiK3LatentTailOp, "__init__", fake_init)
    monkeypatch.setattr(
        "tokenspeed_kernel.ops.moe.latent_tail.dist.get_world_size", lambda group: 8
    )
    kwargs = {
        "group": SimpleNamespace(group_name="test-group"),
        "hidden_size": 7168,
        "latent_size": 3584,
        "rms_eps": 1e-6,
        "device": torch.device("cuda", 0),
        "layer_index": 1,
        "model_scope": "model.layers",
    }

    first = KimiK3LatentTailOp.initialize(**kwargs)
    second = KimiK3LatentTailOp.initialize(**kwargs)

    assert first is not second
    assert len(constructed) == 2


def test_slot_binding_is_rank_identical_for_same_layer() -> None:
    rank_zero_slot = _tail_pool_slot(31, 2)
    rank_one_slot = _tail_pool_slot(31, 2)

    assert rank_zero_slot == rank_one_slot


def test_adjacent_layers_use_different_slots_at_depth_two() -> None:
    assert _tail_pool_slot(31, 2) != _tail_pool_slot(32, 2)


def test_k3_first_moe_layer_starts_at_logical_layer_one() -> None:
    assert _tail_pool_slot(1, 2) == 1
    assert _tail_pool_slot(2, 2) == 0
    assert _tail_pool_slot(3, 2) == _tail_pool_slot(1, 2)


@pytest.mark.parametrize("depth", [1, 2, 3, 17])
def test_slot_binding_has_exact_depth_reuse_spacing(depth: int) -> None:
    slots = [_tail_pool_slot(layer, depth) for layer in range(depth * 2)]

    assert slots[:depth] == slots[depth:]
    assert len(set(slots[:depth])) == depth


def test_base_and_draft_layer_namespaces_use_different_slot_bundles() -> None:
    base_scope = "model.layers"
    draft_scope = "model.decoder"
    layer_index = 0

    assert _tail_pool_slot(layer_index, 2) == _tail_pool_slot(layer_index, 2)
    assert (base_scope, _tail_pool_slot(layer_index, 2)) != (
        draft_scope,
        _tail_pool_slot(layer_index, 2),
    )


def test_slot_binding_rejects_invalid_depth() -> None:
    with pytest.raises(ValueError, match="greater than zero"):
        _tail_pool_slot(0, 0)


def test_allocator_identity_handles_builtin_bound_methods() -> None:
    owner: list[object] = []

    assert _allocator_identity(owner.append) is owner


def test_lamport_copy_releases_successors_before_rearming() -> None:
    """The gather must publish its result before re-arming sentinels.

    Overlapping the sentinel re-arm with successor kernels is where the
    barrier-free gather earns its latency; the pool depth, not this ordering,
    is what keeps a peer from writing a slot that is still being cleaned.
    """
    source = (
        Path(__file__).parents[3]
        / "python/tokenspeed_kernel/thirdparty/cute_dsl/latent_moe_tail/lamport_copy.py"
    ).read_text()
    release = source.index("griddepcontrol_launch_dependents()")
    cleanup = source.index("store_lamport_sentinel_128(source")
    assert release < cleanup


def test_only_the_down_projection_moved_off_the_upstream_sentinel() -> None:
    """These mailboxes keep 0x80000000; only the down projection's overrides it.

    The word is a compile-time parameter of primitives two paths share, and the
    other path is the down projection's, which owns a different one. Nothing on
    one GPU can run the early exit, so what is checked is that the tail and the
    early exit still reach these primitives without an argument, and that the
    argument they therefore get is the upstream word.
    """
    package = Path(__file__).parents[3] / "python/tokenspeed_kernel"
    primitives = (
        package / "thirdparty/cute_dsl/latent_moe_tail/primitives.py"
    ).read_text()
    assert "NEG_ZERO_F32_BITS = 0x80000000" in primitives
    assert primitives.count("sentinel: cutlass.Constexpr[int] = NEG_ZERO_F32_BITS") == 2

    early_exit = (
        package
        / "thirdparty/cute_dsl/latent_moe_tail/allreduce_rmsnorm_reduce_scatter_early_exit.py"
    ).read_text()
    assert early_exit.count("store_lamport_sentinel_128(clear_ptr)") == 2
    assert early_exit.count("fragment_is_dirty(remote)") == 2
    assert early_exit.count("fill_(-0x80000000)") == 1

    tail = (package / "ops/moe/latent_tail.py").read_text()
    assert "sentinel=" not in tail


@pytest.mark.parametrize(
    "ranks,per_host,probed",
    [
        # Whole group inside one host-sized window: no probe needed, and a
        # probe would decline it on a machine that has no fabric at all.
        ([0, 1, 2, 3], 4, False),
        ([4, 5, 6, 7], 4, False),
        ([1, 2], 8, False),
        ([0, 2], 4, False),
        # Spanning the window, in the four shapes that reach this differently:
        # strided, contiguous-but-unaligned, self-aligned, and simply too wide.
        ([0, 8], 4, True),
        ([3, 4, 5], 4, True),
        ([6, 7, 8], 8, True),
        ([7, 8, 9, 10, 11, 12, 13], 8, True),
        (list(range(8)), 4, True),
        # The same eight ranks on either side of the divisor: node-local at
        # eight a host, spanning at four. An implementation that ignored the
        # divisor would pass every row above and fail this one.
        (list(range(8)), 8, False),
    ],
)
def test_the_probe_is_skipped_only_within_one_host_window(
    ranks, per_host, probed
) -> None:
    """Neither size nor self-alignment establishes node-locality.

    ``[6, 7, 8]`` at eight devices a host is contiguous and starts on a
    multiple of its own width while still living on two hosts. What decides it
    is whether every rank falls in the same host-sized window. Skipping the
    probe on a spanning group admits one the fabric may not map, and such a
    group hangs inside the rendezvous rather than falling back; probing a
    node-local one declines a group that works over plain NVLink.
    """
    from unittest import mock

    import tokenspeed_kernel.ops.communication.fabric as fabric
    import tokenspeed_kernel.ops.moe.latent_tail as tail

    seen: list[list[int]] = []
    asked = mock.Mock()

    def only_this_group(group):
        # An implementation that ignored its argument and always tested the
        # world would otherwise pass every row here while declining a
        # node-local subgroup that works over plain NVLink.
        assert group is asked, f"queried {group!r} instead of the caller's group"
        return ranks

    with (
        mock.patch.object(tail.dist, "is_initialized", return_value=True),
        mock.patch.object(
            tail.dist, "get_process_group_ranks", side_effect=only_this_group
        ),
        mock.patch.object(torch.cuda, "device_count", return_value=per_host),
        mock.patch.object(torch.cuda, "current_device", return_value=0),
        mock.patch.object(
            fabric,
            "group_has_fabric",
            side_effect=lambda group_ranks: seen.append(list(group_ranks)) or False,
        ),
    ):
        assert tail.multicast_reachable(asked) is not probed
    assert bool(seen) is probed
    if probed:
        assert seen == [ranks]


@pytest.mark.parametrize("per_host,probed", [(4, True), (8, False)])
def test_the_default_group_is_tested_like_any_other(per_host, probed) -> None:
    """``None`` means the world group, not "assume reachable".

    Short-circuiting it drops the only size term the test has, so a world that
    spans hosts is admitted with no probe -- and the reachability vote its
    callers take cannot catch that, because every rank agrees.
    """
    from unittest import mock

    import tokenspeed_kernel.ops.communication.fabric as fabric
    import tokenspeed_kernel.ops.moe.latent_tail as tail

    seen: list[list[int]] = []
    with (
        mock.patch.object(tail.dist, "is_initialized", return_value=True),
        mock.patch.object(
            tail.dist, "get_process_group_ranks", return_value=list(range(8))
        ),
        mock.patch.object(torch.cuda, "device_count", return_value=per_host),
        mock.patch.object(torch.cuda, "current_device", return_value=0),
        mock.patch.object(
            fabric,
            "group_has_fabric",
            side_effect=lambda ranks: seen.append(list(ranks)) or False,
        ),
    ):
        assert tail.multicast_reachable() is not probed
    assert bool(seen) is probed


def test_no_visible_device_declines_rather_than_dividing_by_zero() -> None:
    """Both directions matter: the permissive answer also passed before."""
    from unittest import mock

    import tokenspeed_kernel.ops.moe.latent_tail as tail

    with (
        mock.patch.object(tail.dist, "is_initialized", return_value=True),
        mock.patch.object(torch.cuda, "device_count", return_value=0),
    ):
        assert tail.multicast_reachable(mock.Mock()) is False
