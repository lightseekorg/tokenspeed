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

"""The slot-state image: every per-slot owner exports what it must, exactly.

CPU-only. Two guards per owner (``RuntimeStates``, the drafters with
cross-round state, the Inkling ring, the DSA KPool tail, the executor's
aggregate):

* a round trip -- export one slot's rows into a uint8 image, wipe them, import
  into another slot, compare byte for byte;
* the completeness test -- every tensor an owner allocates with a slot-sized
  dimension is either reachable from ``slot_state_rows`` or named in
  ``token_derived_slot_state``. A per-slot buffer added without an exporter
  fails here instead of silently breaking a restored request.
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

# CPU-only; scheduled in runtime-1gpu because it imports the full runtime.
register_cuda_ci(est_time=20, suite="runtime-1gpu")

from tokenspeed.runtime.execution.drafter.deepseek_v4_dspark import (  # noqa: E402
    DeepseekV4DSpark,
)
from tokenspeed.runtime.execution.drafter.eagle import Eagle  # noqa: E402
from tokenspeed.runtime.execution.drafter.mtp import Mtp  # noqa: E402
from tokenspeed.runtime.execution.model_executor import ModelExecutor  # noqa: E402
from tokenspeed.runtime.execution.runtime_states import RuntimeStates  # noqa: E402
from tokenspeed.runtime.execution.slot_state import (  # noqa: E402
    SLOT_STATE_ALIGNMENT,
    SlotStateExporter,
    pack_slot_rows,
    slot_state_image_bytes,
    unpack_slot_rows,
)
from tokenspeed.runtime.layers.attention.backends.base import (  # noqa: E402
    AttentionBackend,
)
from tokenspeed.runtime.layers.attention.backends.paged.dsa import (  # noqa: E402
    DSABackend,
)
from tokenspeed.runtime.layers.attention.backends.specific.inkling import (  # noqa: E402
    InklingAttnBackend,
    InklingConvStatePool,
)
from tokenspeed.runtime.layers.attention.kv_cache.hybrid_glm53_flash import (  # noqa: E402
    HybridGlm53FlashTokenToKVPool,
    _KPoolTailWorkspace,
)

# One request-pool geometry for every owner: max_bs decode rows, one sink row
# (max_req_pool_size = max_bs + 1) and the graph-padding row past it.
MAX_BS = 9
MAX_REQ_POOL_SIZE = MAX_BS + 1
POOL_ROWS = MAX_REQ_POOL_SIZE + 1  # 11: RuntimeStates, drafters, Inkling (max_bs + 2)
WINDOW_SLOTS = MAX_REQ_POOL_SIZE + MAX_BS  # 19: DSpark's live + padding slots
SPEC_TOKENS = 4
VOCAB = 32
HIDDEN = 8


# ----------------------------------------------------------------------
# Builders: real owner classes over fakes for the runners they never touch here.
# ----------------------------------------------------------------------


def _runtime_states(*, draft_probs: bool, trees: bool, history: bool) -> RuntimeStates:
    states = RuntimeStates(
        req_pool_size=MAX_REQ_POOL_SIZE,
        vocab_size=VOCAB,
        output_length=SPEC_TOKENS,
        device="cpu",
    )
    if draft_probs:
        states.init_draft_probs(spec_num_tokens=SPEC_TOKENS, reject_threshold=2.0)
    if trees:
        states.init_draft_trees(SPEC_TOKENS)
    if history:
        states.init_ngram_state(3)
        states.init_request_token_history(16)
        states.init_draft_request_token_history(16)
    return states


def _input_buffers() -> SimpleNamespace:
    return SimpleNamespace(
        max_bs=MAX_BS,
        state_write_padding_pool_index=MAX_REQ_POOL_SIZE,
        seq_lens_buf=torch.zeros(MAX_BS, dtype=torch.int32),
        req_pool_indices_buf=torch.zeros(MAX_BS, dtype=torch.int64),
        extend_prefix_lens_cpu=torch.zeros(MAX_BS, dtype=torch.int64),
    )


def _mtp(states: RuntimeStates) -> Mtp:
    class _MtpModel:
        num_mtp_layers = 3

    runner = SimpleNamespace(
        device="cpu",
        forward_accepts_spec_step_idx=True,
        model=_MtpModel(),
        model_config=SimpleNamespace(hidden_size=HIDDEN, dtype=torch.bfloat16),
    )
    return Mtp(
        spec_num_tokens=SPEC_TOKENS,
        spec_num_steps=3,
        draft_model_runner=runner,
        runtime_states=states,
        input_buffers=_input_buffers(),
        vocab_size=VOCAB,
    )


def _eagle(states: RuntimeStates) -> Eagle:
    runner = SimpleNamespace(
        device="cpu",
        model=SimpleNamespace(get_hot_token_id=lambda: None),
        mapping=SimpleNamespace(attn=SimpleNamespace(dp_size=1), world_size=1),
        model_config=SimpleNamespace(requires_request_token_history=True),
    )
    return Eagle(
        spec_num_tokens=SPEC_TOKENS,
        spec_num_steps=3,
        draft_model_runner=runner,
        runtime_states=states,
        input_buffers=_input_buffers(),
        vocab_size=VOCAB,
    )


def _dspark(states: RuntimeStates) -> DeepseekV4DSpark:
    block_size = 4
    inner = SimpleNamespace(
        block_size=block_size,
        num_stages=2,
        window_size=3,
        attention_params={"head_dim": HIDDEN},
        target_layer_ids=[0, 1],
        hidden_size=HIDDEN,
    )
    draft_model = SimpleNamespace(
        model=inner,
        mapping=SimpleNamespace(attn=SimpleNamespace(tp_size=1)),
        lm_head=SimpleNamespace(num_embeddings_per_partition=VOCAB),
    )
    runner = SimpleNamespace(
        device="cpu",
        model=draft_model,
        mapping=SimpleNamespace(attn=SimpleNamespace(dp_size=1, tp_size=1)),
    )
    # DSPARK: block_size == num_steps, num_draft_tokens == num_steps + 1.
    return DeepseekV4DSpark(
        spec_num_tokens=block_size + 1,
        spec_num_steps=block_size,
        draft_model_runner=runner,
        runtime_states=states,
        input_buffers=_input_buffers(),
        vocab_size=VOCAB,
    )


class _Leaf(AttentionBackend):
    """A stateless leaf: the composite default images nothing for it."""

    def __init__(self) -> None:
        self._init_pool_binding()


def _inkling() -> InklingAttnBackend:
    conv_pool = InklingConvStatePool(
        num_layers=2,
        num_slots=MAX_BS + 2,
        conv_dim=HIDDEN,
        ring_size=5,
        dtype=torch.bfloat16,
        device="cpu",
    )
    return InklingAttnBackend(_Leaf(), conv_pool)


def _kpool_pool() -> HybridGlm53FlashTokenToKVPool:
    pool = HybridGlm53FlashTokenToKVPool.__new__(HybridGlm53FlashTokenToKVPool)
    pool._kpool_tail_workspace = _KPoolTailWorkspace(
        options=None,
        storage=torch.zeros((2, 2, MAX_BS + 2, 3, HIDDEN), dtype=torch.bfloat16),
        row_by_layer={0: 0, 1: 1},
    )
    return pool


def _dsa(pool, *, is_draft: bool) -> DSABackend:
    backend = DSABackend.__new__(DSABackend)
    backend._init_pool_binding()
    backend.cache_pool = pool
    backend.is_draft = is_draft
    backend.kpool_runtime = object()
    backend._dense_backend = _Leaf()
    return backend


def _randomize(rows: list[torch.Tensor]) -> None:
    for row in rows:
        if row.dtype == torch.bool:
            row.copy_(torch.rand(row.shape) > 0.5)
        elif row.dtype.is_floating_point:
            row.copy_(torch.randn(row.shape).to(row.dtype))
        else:
            row.copy_(torch.randint(1, 100, row.shape).to(row.dtype))


def _round_trip(exporter: SlotStateExporter, *, src_slot: int, dst_slot: int) -> None:
    """Export ``src_slot``, wipe it, import into ``dst_slot``; both must match."""
    torch.manual_seed(0)
    source_rows = exporter.slot_state_rows(src_slot)
    _randomize(source_rows)
    expected = [row.clone() for row in source_rows]
    assert exporter.slot_state_bytes() == slot_state_image_bytes(source_rows)
    assert exporter.slot_state_bytes() % SLOT_STATE_ALIGNMENT == 0
    image = torch.full((exporter.slot_state_bytes(),), 0xEE, dtype=torch.uint8)

    exporter.export_slot_state(src_slot, image, None)
    for row in source_rows:
        row.zero_()
    exporter.import_slot_state(dst_slot, image, None, request_id="req-restored")

    for restored, original in zip(exporter.slot_state_rows(dst_slot), expected):
        assert restored.shape == original.shape
        assert torch.equal(restored, original)


# ----------------------------------------------------------------------
# The packing helpers
# ----------------------------------------------------------------------


def test_pack_aligns_every_row_and_refuses_a_short_image():
    rows = [
        torch.tensor(7, dtype=torch.int32),
        torch.ones(3, dtype=torch.bool),
        torch.arange(6, dtype=torch.float32).view(2, 3),
    ]
    nbytes = slot_state_image_bytes(rows)
    # 4 B, 3 B and 24 B, each padded to the 16 B alignment.
    assert (
        nbytes == SLOT_STATE_ALIGNMENT + SLOT_STATE_ALIGNMENT + 2 * SLOT_STATE_ALIGNMENT
    )
    image = torch.zeros(nbytes, dtype=torch.uint8)
    pack_slot_rows(rows, image, None)
    out = [torch.zeros_like(row) for row in rows]
    unpack_slot_rows(out, image, None)
    assert all(torch.equal(a, b) for a, b in zip(out, rows))
    with pytest.raises(ValueError, match="cannot hold"):
        pack_slot_rows(rows, image[:-1], None)
    with pytest.raises(ValueError, match="1-D uint8"):
        pack_slot_rows(rows, image.view(torch.int8), None)


def test_pack_copies_a_non_contiguous_slot_slice():
    ring = torch.arange(2 * 4 * 3, dtype=torch.bfloat16).view(2, 4, 3)
    row = ring[:, 2]  # the slot dimension is not the leading one
    assert not row.is_contiguous()
    image = torch.zeros(slot_state_image_bytes([row]), dtype=torch.uint8)
    pack_slot_rows([row], image, None)
    target = torch.zeros(2, 4, 3, dtype=torch.bfloat16)
    unpack_slot_rows([target[:, 1]], image, None)
    assert torch.equal(target[:, 1], ring[:, 2])
    assert target[:, 0].abs().sum() == 0


# ----------------------------------------------------------------------
# Round trips, one owner at a time
# ----------------------------------------------------------------------


@pytest.mark.parametrize("draft_probs", [False, True])
@pytest.mark.parametrize("trees", [False, True])
def test_runtime_states_round_trip(draft_probs, trees):
    states = _runtime_states(draft_probs=draft_probs, trees=trees, history=True)
    _round_trip(states, src_slot=3, dst_slot=7)
    # Token-derived rows stay out of the image and untouched by it.
    assert states.request_token_history_ids.abs().sum() == 0


def test_mtp_stash_round_trip():
    states = _runtime_states(draft_probs=False, trees=False, history=False)
    _round_trip(_mtp(states), src_slot=1, dst_slot=5)


def test_eagle_history_frontier_round_trip():
    states = _runtime_states(draft_probs=False, trees=False, history=True)
    eagle = _eagle(states)
    assert eagle.draft_history_lengths_buf is not None
    _round_trip(eagle, src_slot=2, dst_slot=8)


def test_dspark_windows_round_trip_and_claim_the_slot():
    states = _runtime_states(draft_probs=False, trees=False, history=False)
    dspark = _dspark(states)
    _round_trip(dspark, src_slot=4, dst_slot=6)
    # The import claims the new slot for the request, so the next prologue
    # does not reset the imported windows as a previous occupant's.
    assert dspark._request_by_pool_slot[6] == "req-restored"
    with pytest.raises(ValueError, match="persistent state domain"):
        dspark.slot_state_rows(MAX_REQ_POOL_SIZE)


def test_inkling_ring_round_trip_through_the_wrapper():
    backend = _inkling()
    _round_trip(backend, src_slot=3, dst_slot=9)


def test_dsa_kpool_tail_round_trip_once_per_arena():
    pool = _kpool_pool()
    target = _dsa(pool, is_draft=False)
    draft = _dsa(pool, is_draft=True)
    assert draft.slot_state_bytes() == 0, "the shared tail is imaged by the target"
    _round_trip(target, src_slot=2, dst_slot=5)


def test_a_leaf_without_slot_state_images_nothing():
    leaf = _Leaf()
    assert leaf.slot_state_bytes() == 0
    leaf.export_slot_state(0, torch.zeros(0, dtype=torch.uint8), None)
    leaf.import_slot_state(0, torch.zeros(0, dtype=torch.uint8), None, request_id="r")


def test_model_executor_concatenates_its_owners_once_each():
    states = _runtime_states(draft_probs=True, trees=False, history=False)
    executor = ModelExecutor.__new__(ModelExecutor)
    executor.runtime_states = states
    executor.attn_backend = _inkling()
    executor.drafter = _mtp(states)
    # A shared draft tree is listed once; a distinct one adds its own segment.
    executor.draft_attn_backend = executor.attn_backend
    shared = executor.slot_state_bytes()
    assert shared == (
        states.slot_state_bytes()
        + executor.attn_backend.slot_state_bytes()
        + executor.drafter.slot_state_bytes()
    )
    executor.draft_attn_backend = _inkling()
    assert (
        executor.slot_state_bytes()
        == shared + executor.draft_attn_backend.slot_state_bytes()
    )

    rows = lambda slot: [  # noqa: E731
        row
        for owner in executor.slot_state_exporters()
        for row in owner.slot_state_rows(slot)
    ]
    torch.manual_seed(1)
    _randomize(rows(2))
    expected = [row.clone() for row in rows(2)]
    image = torch.zeros(executor.slot_state_bytes(), dtype=torch.uint8)
    executor.export_slot_state(2, image, None)
    for row in rows(2):
        row.zero_()
    executor.import_slot_state(6, image, None, request_id="r6")
    assert all(torch.equal(a, b) for a, b in zip(rows(6), expected))
    with pytest.raises(ValueError, match="need"):
        executor.export_slot_state(2, image[:-SLOT_STATE_ALIGNMENT], None)


# ----------------------------------------------------------------------
# Completeness: a per-slot buffer must be exported or declared token-derived.
# ----------------------------------------------------------------------


def _tensor_attributes(obj) -> dict[str, torch.Tensor]:
    """``name -> tensor`` over the object's own attributes and listed sub-owners."""
    found: dict[str, torch.Tensor] = {}
    for name, value in vars(obj).items():
        if isinstance(value, torch.Tensor):
            found[name] = value
        elif name in ("conv_pool", "_kpool_tail_workspace"):
            for sub_name, sub_value in vars(value).items():
                if isinstance(sub_value, torch.Tensor):
                    found[f"{name}.{sub_name}"] = sub_value
    return found


def _per_slot_tensors(obj, slot_domains: set[int]) -> dict[str, torch.Tensor]:
    return {
        name: tensor
        for name, tensor in _tensor_attributes(obj).items()
        if any(dim in slot_domains for dim in tensor.shape)
    }


def _storage_ids(rows) -> set[int]:
    return {row.untyped_storage().data_ptr() for row in rows}


def _assert_slot_state_complete(owner, *, exporter, slot_domains: set[int]) -> None:
    exported = _storage_ids(exporter.slot_state_rows(0))
    token_derived = set(getattr(type(owner), "token_derived_slot_state", ()))
    uncovered = sorted(
        name
        for name, tensor in _per_slot_tensors(owner, slot_domains).items()
        if tensor.untyped_storage().data_ptr() not in exported
        and name not in token_derived
    )
    assert not uncovered, (
        f"{type(owner).__name__} keeps per-slot state with no exporter: {uncovered}. "
        "List it in slot_state_rows (restored byte for byte) or in "
        "token_derived_slot_state (reseeded from the request's tokens)."
    )


def test_runtime_states_slot_state_is_complete():
    states = _runtime_states(draft_probs=True, trees=True, history=True)
    _assert_slot_state_complete(states, exporter=states, slot_domains={POOL_ROWS})
    # The declared token-derived rows exist and really are per-slot.
    per_slot = _per_slot_tensors(states, {POOL_ROWS})
    assert set(RuntimeStates.token_derived_slot_state) <= set(per_slot)


def test_drafter_slot_state_is_complete():
    states = _runtime_states(draft_probs=False, trees=False, history=True)
    mtp = _mtp(states)
    _assert_slot_state_complete(mtp, exporter=mtp, slot_domains={POOL_ROWS})
    eagle = _eagle(states)
    _assert_slot_state_complete(eagle, exporter=eagle, slot_domains={POOL_ROWS})
    dspark = _dspark(states)
    assert dspark.kv_windows.shape[0] == WINDOW_SLOTS
    _assert_slot_state_complete(
        dspark, exporter=dspark, slot_domains={POOL_ROWS, WINDOW_SLOTS}
    )


def test_backend_slot_state_is_complete():
    inkling = _inkling()
    _assert_slot_state_complete(inkling, exporter=inkling, slot_domains={POOL_ROWS})
    pool = _kpool_pool()
    target = _dsa(pool, is_draft=False)
    _assert_slot_state_complete(pool, exporter=target, slot_domains={POOL_ROWS})


def test_completeness_check_catches_an_unexported_buffer():
    states = _runtime_states(draft_probs=False, trees=False, history=False)
    states.rogue_per_slot = torch.zeros(POOL_ROWS, dtype=torch.int32)
    with pytest.raises(AssertionError, match="rogue_per_slot"):
        _assert_slot_state_complete(states, exporter=states, slot_domains={POOL_ROWS})
