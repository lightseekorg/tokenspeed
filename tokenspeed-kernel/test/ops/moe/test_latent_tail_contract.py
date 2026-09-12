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

import ast
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


def _parameters_with_defaults(args: ast.arguments) -> list:
    """Every parameter of one signature, paired with its default or None."""
    positional = list(args.posonlyargs) + list(args.args)
    padding = [None] * (len(positional) - len(args.defaults))
    return list(zip(positional, padding + list(args.defaults))) + list(
        zip(args.kwonlyargs, args.kw_defaults)
    )


def test_only_the_down_projection_moved_off_the_upstream_sentinel() -> None:
    """These mailboxes keep 0x80000000; only the down projection's overrides it.

    The word is a compile-time parameter of primitives two paths share, and the
    other path is the down projection's, which owns a different one. Nothing on
    one GPU can run the early exit, so what is checked is that neither
    primitive defaults the word, and that the tail and the early exit name the
    upstream one at every call site.
    """
    package = Path(__file__).parents[3] / "python/tokenspeed_kernel"
    primitives = (
        package / "thirdparty/cute_dsl/latent_moe_tail/primitives.py"
    ).read_text()
    assert "NEG_ZERO_F32_BITS = 0x80000000" in primitives
    sentinels = {
        node.name: default
        for node in ast.walk(ast.parse(primitives))
        if isinstance(node, ast.FunctionDef)
        for arg, default in _parameters_with_defaults(node.args)
        if arg.arg == "sentinel"
    }
    assert set(sentinels) == {"store_lamport_sentinel_128", "fragment_is_dirty"}
    assert all(default is None for default in sentinels.values())

    early_exit = (
        package
        / "thirdparty/cute_dsl/latent_moe_tail/allreduce_rmsnorm_reduce_scatter_early_exit.py"
    ).read_text()
    armed = "store_lamport_sentinel_128(clear_ptr, sentinel=NEG_ZERO_F32_BITS)"
    assert early_exit.count(armed) == 2
    early_imports = {
        alias.asname or alias.name
        for node in ast.walk(ast.parse(early_exit))
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    assert "NEG_ZERO_F32_BITS" in early_imports
    checked = "fragment_is_dirty(remote, sentinel=NEG_ZERO_F32_BITS)"
    assert early_exit.count(checked) == 2
    assert early_exit.count("fill_(-0x80000000)") == 1

    # Each mailbox names its own: the down projection's producers cannot all
    # sanitize -0, so it needs the pattern that takes two coincidences to spell.
    tail = (package / "ops/moe/latent_tail.py").read_text()
    assert "sentinel=NEG_ZERO_F32_BITS" in tail
    # Naming it is not enough: the text check passes while the import is absent,
    # which is a NameError at the first tail construction and not at import.
    tail_imports = {
        alias.asname or alias.name
        for node in ast.walk(ast.parse(tail))
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    assert "NEG_ZERO_F32_BITS" in tail_imports
    down = (package / "ops/moe/latent_down.py").read_text()
    assert "sentinel=_DOWN_SENTINEL" in down
    assert "_DOWN_SENTINEL = 0x80008000" in down


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
        # The same eight ranks on either side of the boundary: node-local at
        # eight a host, spanning at four. Every row runs with eight visible
        # devices, so a gate dividing by the device count answers "one host"
        # for this row and the one above alike, and only one of them is right.
        (list(range(8)), 8, False),
    ],
)
def test_the_probe_is_skipped_only_within_one_host_window(
    ranks, per_host, probed
) -> None:
    """Neither size nor self-alignment establishes node-locality.

    ``[6, 7, 8]`` at eight devices a host is contiguous and starts on a
    multiple of its own width while still living on two hosts. What decides it
    is which host each rank sits on, which the gathered map records. Skipping
    the probe on a spanning group admits one the fabric may not map, and such a
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

    saved = fabric._host_map
    fabric._host_map = [rank // per_host for rank in range(max(ranks) + 1)]
    try:
        with (
            mock.patch.object(tail.dist, "is_initialized", return_value=True),
            mock.patch.object(
                tail.dist, "get_process_group_ranks", side_effect=only_this_group
            ),
            mock.patch.object(torch.cuda, "device_count", return_value=8),
            mock.patch.object(torch.cuda, "current_device", return_value=0),
            mock.patch.object(
                fabric,
                "group_has_fabric",
                side_effect=lambda group_ranks: seen.append(list(group_ranks)) or False,
            ),
        ):
            assert tail.multicast_reachable(asked) is not probed
    finally:
        fabric._host_map = saved
    assert bool(seen) is probed
    if probed:
        assert seen == [ranks]


@pytest.mark.parametrize("per_host,probed", [(4, True), (8, False)])
def test_the_world_group_is_tested_like_any_other(per_host, probed) -> None:
    """The world group earns no exemption from the probe.

    Short-circuiting it drops the only size term the test has, so a world that
    spans hosts is admitted with no probe -- and the reachability vote its
    callers take cannot catch that, because every rank agrees.
    """
    from unittest import mock

    import tokenspeed_kernel.ops.communication.fabric as fabric
    import tokenspeed_kernel.ops.moe.latent_tail as tail

    seen: list[list[int]] = []
    saved = fabric._host_map
    fabric._host_map = [rank // per_host for rank in range(8)]
    try:
        # Uninitialised, ``dist.group.WORLD`` is None, so a gate that forwards
        # None unchanged is indistinguishable from one that resolves the world
        # group. ``WORLD`` is a property on the metaclass, so give it a value
        # there and the two become different arguments.
        world = mock.Mock(name="WORLD")
        with (
            mock.patch.object(type(tail.dist.group), "WORLD", world),
            mock.patch.object(tail.dist, "is_initialized", return_value=True),
            mock.patch.object(
                tail.dist, "get_process_group_ranks", return_value=list(range(8))
            ) as ranks_of,
            mock.patch.object(torch.cuda, "device_count", return_value=8),
            mock.patch.object(torch.cuda, "current_device", return_value=0),
            mock.patch.object(
                fabric,
                "group_has_fabric",
                side_effect=lambda ranks: seen.append(list(ranks)) or False,
            ),
        ):
            assert tail.multicast_reachable(world) is not probed
            ranks_of.assert_called_once_with(world)
    finally:
        fabric._host_map = saved
    assert bool(seen) is probed


def test_an_ungathered_map_declines_rather_than_guessing_placement() -> None:
    """Both directions matter: the permissive answer also passed before.

    Without the map there is nothing to place ranks with, and the gate cannot
    reach for the device count instead -- that is the divisor this replaced.
    Declining costs a fallback; admitting costs the rendezvous it hangs in.
    """
    from unittest import mock

    import tokenspeed_kernel.ops.communication.fabric as fabric
    import tokenspeed_kernel.ops.moe.latent_tail as tail

    saved = fabric._host_map
    fabric._host_map = None
    try:
        with (
            mock.patch.object(tail.dist, "is_initialized", return_value=True),
            mock.patch.object(
                tail.dist, "get_process_group_ranks", return_value=list(range(8))
            ),
            mock.patch.object(torch.cuda, "device_count", return_value=8),
        ):
            assert tail.multicast_reachable(mock.Mock()) is False
    finally:
        fabric._host_map = saved


def test_a_non_nvidia_platform_fills_the_map_without_a_collective() -> None:
    """Fabric handles are NVIDIA-only, so the answer needs no exchange.

    The device type does not settle it: ROCm reports "cuda" too. Gathering
    anyway would cost a collective whose answer is known, and skipping the
    gather at the call site instead would leave the map empty for a gate to
    fill in lazily -- putting the collective back in the dispatch path.
    """
    from unittest import mock

    import tokenspeed_kernel.ops.communication.fabric as fabric

    fabric._fabric_map = None
    try:
        with (
            mock.patch.object(
                fabric,
                "current_platform",
                return_value=mock.Mock(is_nvidia=False),
            ),
            mock.patch.object(
                fabric.torch.distributed, "get_world_size", return_value=8
            ),
            mock.patch.object(
                fabric.torch.distributed,
                "all_gather",
                side_effect=AssertionError("no collective off NVIDIA"),
            ),
        ):
            assert fabric.gather_fabric_map() == [False] * 8
            assert fabric.group_has_fabric([0, 1]) is False
            # Distinct hosts, so every span declines rather than being admitted
            # as node-local on a platform that has no fabric to map.
            assert fabric.group_host_span([0, 1]) == 2
    finally:
        fabric._fabric_map = None
        fabric._host_map = None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the gather needs a device")
def test_the_gather_keeps_every_rank_its_own_host() -> None:
    """Losing the host id in transit makes the whole world look like one host.

    That is the permissive direction: every group then reads node-local, skips
    the fabric test, and the rendezvous hangs -- the failure the map exists to
    prevent. Setting the map by hand, as the gate tests do, cannot see this.
    """
    from unittest import mock

    import tokenspeed_kernel.ops.communication.fabric as fabric

    def two_hosts(gathered, local, group=None):
        # Rank zero gets its own payload back, so what the gather *packs* is
        # checked and not only what it decodes: a call site packing a constant
        # leaves every decoded host identical and nothing else would see it.
        assert local[1].item() == 100, "the host id never reached the payload"
        gathered[0].copy_(local)
        for rank, slot in enumerate(gathered[1:], start=1):
            slot.copy_(torch.tensor([1, 100 + rank // 2], dtype=torch.int64))

    fabric._fabric_map = None
    fabric._host_map = None
    try:
        with (
            mock.patch.object(
                fabric, "current_platform", return_value=mock.Mock(is_nvidia=True)
            ),
            mock.patch.object(
                fabric.torch.distributed, "get_world_size", return_value=4
            ),
            mock.patch.object(
                fabric.torch.distributed, "all_gather", side_effect=two_hosts
            ),
            mock.patch.object(fabric, "fabric_allocation_supported", return_value=True),
            mock.patch.object(
                fabric, "_host_identity", return_value=100
            ) as host_identity,
            mock.patch.object(torch.cuda, "current_device", return_value=0),
        ):
            assert fabric.gather_fabric_map() == [True] * 4
            # The value alone is satisfied by a literal 100 at the call site;
            # this pins that the packed word came from the function.
            host_identity.assert_called_once_with()
            assert fabric.group_host_span([0, 1]) == 1
            assert fabric.group_host_span([0, 2]) == 2
    finally:
        fabric._fabric_map = None
        fabric._host_map = None


def test_the_host_id_is_stable_across_processes() -> None:
    """A salted hash gives each rank a different id for the same host.

    ``hash()`` is seeded per process, so ranks on one host would look like as
    many hosts as there are ranks. Every group would then span, the path would
    decline everywhere instead of hanging anywhere, and nothing would fail --
    which is the failure that hides longest.
    """
    import os
    import subprocess
    import sys

    from tokenspeed_kernel.ops.communication.fabric import _host_identity

    here = _host_identity()
    # int64 is what the gather carries it in; a wider id would wrap to another
    # host's value and merge two machines.
    assert 0 <= here < 2**63

    elsewhere = subprocess.run(
        [
            sys.executable,
            "-c",
            "from tokenspeed_kernel.ops.communication.fabric import _host_identity;"
            "print(_host_identity())",
        ],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "PYTHONHASHSEED": "12345"},
    ).stdout.strip()
    assert int(elsewhere) == here, "the host id depends on the process hash seed"


def test_two_hosts_do_not_share_one_identity() -> None:
    """Stability alone is satisfied by a constant, which merges every host.

    A constant id is the permissive direction again: one host, no fabric test,
    and the rendezvous the map exists to keep the group out of.
    """
    from unittest import mock

    import tokenspeed_kernel.ops.communication.fabric as fabric

    identities = []
    for boot in ("11111111-1111-1111-1111-111111111111", "22222222-2222"):
        with mock.patch.object(fabric.Path, "read_text", return_value=boot):
            identities.append(fabric._host_identity())
    assert len(set(identities)) == 2


def test_a_high_digest_bit_still_fits_the_gathered_int64() -> None:
    """Asserting the range on this host's own digest is a coin flip.

    Half of all digests leave the top bit clear, so dropping the reduction
    passes wherever the test happens to run and fails later on a host whose
    digest does not.
    """
    from unittest import mock

    import tokenspeed_kernel.ops.communication.fabric as fabric

    with mock.patch.object(
        fabric.hashlib, "blake2b", return_value=mock.Mock(digest=lambda: b"\xff" * 8)
    ):
        assert fabric._host_identity() < 2**63


def test_a_missing_map_raises_instead_of_gathering_from_dispatch() -> None:
    """The lazy gather would be a world collective from a group-scoped call.

    This is asked at dispatch, where the ranks present are the group's -- a
    stage under pipeline parallelism, or a data-parallel subset. Gathering
    there would block on world ranks that never arrive, so a missing map has
    to be loud instead: it means the initialization hook did not run.
    """
    from unittest import mock

    import tokenspeed_kernel.ops.communication.fabric as fabric

    saved = fabric._fabric_map
    fabric._fabric_map = None
    try:
        with (
            mock.patch.object(
                fabric.torch.distributed,
                "all_gather",
                side_effect=AssertionError("must not gather from dispatch"),
            ),
            mock.patch.object(
                fabric.torch.distributed, "get_world_size", return_value=8
            ),
        ):
            with pytest.raises(RuntimeError, match="never gathered"):
                fabric.group_has_fabric([0, 1])
    finally:
        fabric._fabric_map = saved
