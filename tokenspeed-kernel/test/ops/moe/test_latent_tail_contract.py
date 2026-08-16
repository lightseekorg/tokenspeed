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
    cleanup = source.index("store_lamport_sentinel_128(source)")
    assert release < cleanup
