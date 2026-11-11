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

"""``Mapping.replicated_for_draft`` (cheap, no GPU).

A block draft's MLP holds whole weights on every rank so its forward carries no
dense collective. The derivation must collapse the dense TP width to 1, keep the
ATTENTION width (the draft writes K/V at the target's cache locations), and
leave the caller's own layout untouched.
"""

import pytest

from tokenspeed.runtime.distributed.mapping import Mapping


def _decode_layout(rank: int = 3) -> Mapping:
    """An attention-DP decode layout with a head-TP tail and a dense group."""
    return Mapping(
        rank=rank,
        world_size=8,
        attn_tp_size=1,
        attn_dp_size=8,
        attn_head_tp_size=4,
        lm_head_tp_size=4,
        dense_tp_size=4,
        moe_ep_size=8,
    )


def _prefill_layout(rank: int = 3) -> Mapping:
    """A tensor-parallel prefill layout: the draft's cache rows are sharded by
    attention TP there, so that width must survive the derivation."""
    return Mapping(
        rank=rank,
        world_size=8,
        attn_tp_size=8,
        dense_tp_size=1,
        moe_ep_size=8,
    )


@pytest.mark.parametrize("rank", range(8))
def test_draft_dense_group_is_this_rank_alone(rank: int) -> None:
    """The group a draft MLP would reduce over holds only this rank, which is
    what makes its ``all_reduce`` a no-op instead of a hang."""
    draft = _decode_layout(rank).replicated_for_draft()

    assert draft.dense.tp_size == 1
    assert draft.dense.tp_group == (rank,)
    assert not draft.dense.has_tp


@pytest.mark.parametrize("layout", [_decode_layout, _prefill_layout])
def test_draft_keeps_the_attention_width(layout) -> None:
    """The draft's q/k/v must keep producing the heads its cache rows hold:
    ``MhaSpec.cache_cell_size`` divides by the attention TP width, and the
    GQA prologue rejects a k/v that is not dense rows of the cache heads."""
    target = layout()
    draft = target.replicated_for_draft()

    assert draft.attn.tp_size == target.attn.tp_size
    assert draft.attn.tp_group == target.attn.tp_group
    assert draft.attn.head_tp_size == target.attn.head_tp_size
    assert draft.attn.kvp_size == target.attn.kvp_size
    assert draft.attn.qcp_size == target.attn.qcp_size


def test_draft_layout_keeps_every_other_axis() -> None:
    target = _decode_layout()
    draft = target.replicated_for_draft()

    assert draft.rank == target.rank
    assert draft.world_size == target.world_size
    assert draft.attn.dp_size == target.attn.dp_size
    assert draft.lm_head.tp_size == target.lm_head.tp_size
    assert draft.moe.ep_size == target.moe.ep_size
    assert draft.moe.tp_ep_size == target.moe.tp_ep_size
    assert draft.pp_size == target.pp_size
    assert draft.nprocs_per_node == target.nprocs_per_node
    assert draft.nnodes == target.nnodes
    assert draft.base_gpu_id == target.base_gpu_id
    assert draft.gpu_id_step == target.gpu_id_step


def test_deriving_a_draft_layout_leaves_the_target_layout_alone() -> None:
    """The target keeps the parallelism it was launched with: the draft-side
    fix must cost the target nothing."""
    target = _decode_layout()

    # Read the groups first so any cached_property is already resolved.
    assert target.dense.tp_group == (0, 1, 2, 3)
    target.replicated_for_draft()

    assert target.attn.head_tp_size == 4
    assert target.dense.tp_size == 4
    assert target.dense.tp_group == (0, 1, 2, 3)
    assert target.lm_head.tp_size == 4


def test_draft_layout_under_pipeline_parallelism_keeps_its_stage() -> None:
    target = Mapping(
        rank=5,
        world_size=8,
        pp_size=2,
        attn_tp_size=1,
        attn_dp_size=4,
        dense_tp_size=4,
    )
    draft = target.replicated_for_draft()

    assert draft.pp_size == target.pp_size
    assert draft.pp_rank == target.pp_rank
    assert draft.attn.dp_size == target.attn.dp_size
    assert draft.dense.tp_size == 1


def test_a_layout_without_dense_parallelism_is_already_a_draft_layout() -> None:
    target = Mapping(rank=0, world_size=1)
    draft = target.replicated_for_draft()

    assert draft.dense.tp_size == 1
    assert draft.attn.tp_size == target.attn.tp_size
    assert draft.world_size == 1
    assert draft.rank == 0
