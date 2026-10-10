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
from unittest.mock import patch

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
    PREPARED_MARKER_BYTES,
    SLOT_STATE_ALIGNMENT,
    SlotStateExporter,
    SlotStateLayout,
    pack_slot_rows,
    read_prepared_marker,
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
from tokenspeed.runtime.sampling.backends.base import (  # noqa: E402
    SamplingBackend,
    SamplingBackendConfig,
)
from tokenspeed.runtime.sampling.backends.flashinfer import (  # noqa: E402
    FlashInferSamplingBackend,
)
from tokenspeed.runtime.sampling.backends.flashinfer_full import (  # noqa: E402
    FlashInferFullSamplingBackend,
)
from tokenspeed.runtime.sampling.backends.greedy import (  # noqa: E402
    GreedySamplingBackend,
)
from tokenspeed.runtime.sampling.backends.triton import (  # noqa: E402
    TritonSamplingBackend,
)
from tokenspeed.runtime.sampling.backends.triton_full import (  # noqa: E402
    TritonFullSamplingBackend,
)
from tokenspeed.runtime.sampling.sampling_params import SamplingParams  # noqa: E402
from tokenspeed.runtime.sampling.utils import coin_eps  # noqa: E402

SAMPLING_BACKENDS = (
    GreedySamplingBackend,
    FlashInferSamplingBackend,
    FlashInferFullSamplingBackend,
    TritonSamplingBackend,
    TritonFullSamplingBackend,
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


def _sampling_backend(cls) -> SamplingBackend:
    """A sampling backend over the test pool geometry, on the CPU."""
    config = SamplingBackendConfig(
        enable_speculative_sampling=cls is not GreedySamplingBackend,
        sampling_stream="batch",
        logprob_order="torch",
        max_bs=MAX_BS,
        max_draft_tokens_per_req=SPEC_TOKENS,
        max_req_pool_size=MAX_REQ_POOL_SIZE,
        vocab_size=VOCAB,
        device="cpu",
    )
    # The Triton backends size a scratch buffer by the SM count.
    with patch.object(
        torch.cuda,
        "get_device_properties",
        return_value=SimpleNamespace(multi_processor_count=4),
    ):
        return cls(config)


def _sampling_params(rid: str, **overrides) -> SamplingParams:
    fields = dict(
        temperature=0.7,
        top_k=5,
        top_p=0.9,
        min_p=0.1,
        frequency_penalty=0.5,
        presence_penalty=0.25,
        repetition_penalty=1.5,
    )
    fields.update(overrides)
    params = SamplingParams(**fields)
    params.resolve_seed(rid)
    params.normalize(None)
    return params


def _randomize(rows: list[torch.Tensor]) -> None:
    for row in rows:
        if row.dtype == torch.bool:
            row.copy_(torch.rand(row.shape) > 0.5)
        elif row.dtype.is_floating_point:
            row.copy_(torch.randn(row.shape).to(row.dtype))
        else:
            row.copy_(torch.randint(1, 100, row.shape).to(row.dtype))


RESTORED = "req-restored"


def _round_trip(
    exporter: SlotStateExporter,
    *,
    src_slot: int,
    dst_slot: int,
    rows_rewritten_on_import: frozenset[int] = frozenset(),
    request_keyed: bool = False,
) -> None:
    """Export ``src_slot`` as ``RESTORED``'s, wipe it, import into ``dst_slot``;
    both must match except the rows an owner deliberately rewrites on import.
    A request-keyed owner images the prepared marker ahead of its rows."""
    torch.manual_seed(0)
    source_rows = exporter.slot_state_rows(src_slot)
    _randomize(source_rows)
    expected = [row.clone() for row in source_rows]
    marker = PREPARED_MARKER_BYTES if request_keyed else 0
    assert exporter.slot_state_bytes() == marker + slot_state_image_bytes(source_rows)
    assert exporter.slot_state_bytes() % SLOT_STATE_ALIGNMENT == 0
    image = torch.full((exporter.slot_state_bytes(),), 0xEE, dtype=torch.uint8)

    exporter.export_slot_state(src_slot, image, None, request_id=RESTORED)
    if request_keyed:
        assert read_prepared_marker(image)
    for row in source_rows:
        row.zero_()
    exporter.import_slot_state(dst_slot, image, None, request_id=RESTORED)

    for index, (restored, original) in enumerate(
        zip(exporter.slot_state_rows(dst_slot), expected)
    ):
        assert restored.shape == original.shape
        if index not in rows_rewritten_on_import:
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


# RuntimeStates.slot_state_rows order: cache length, next-step inputs,
# candidate readiness, then the optional draft distributions and tree parents.
_CANDIDATE_READY_ROW = 2


@pytest.mark.parametrize("draft_probs", [False, True])
@pytest.mark.parametrize("trees", [False, True])
def test_runtime_states_round_trip(draft_probs, trees):
    states = _runtime_states(draft_probs=draft_probs, trees=trees, history=True)
    _round_trip(
        states,
        src_slot=3,
        dst_slot=7,
        rows_rewritten_on_import=frozenset({_CANDIDATE_READY_ROW}),
    )
    # Token-derived rows stay out of the image and untouched by it.
    assert states.request_token_history_ids.abs().sum() == 0


def test_runtime_states_import_marks_the_imaged_candidates_ready():
    """A restored request's first decode op carries its token explicitly (no
    forward of its own is in flight), and the prologue treats an explicit id
    on a row whose candidates are not marked ready as a bootstrap row to
    verify single-token. The imaged candidates are the ones the victim's last
    forward drafted, so the import marks them ready and the first verify
    after the restore consumes them as the unretracted step would have."""
    states = _runtime_states(draft_probs=False, trees=False, history=False)
    states.remote_spec_candidate_ready[3] = False  # a fused victim's row
    states.future_input_map[3] = torch.arange(1, states.future_input_map.shape[1] + 1)
    image = torch.empty((states.slot_state_bytes(),), dtype=torch.uint8)
    states.export_slot_state(3, image, None, request_id="restored")
    states.import_slot_state(7, image, None, request_id="restored")
    assert bool(states.remote_spec_candidate_ready[7])
    assert torch.equal(states.future_input_map[7], states.future_input_map[3])


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
    # The victim ran a forward in slot 4: its windows are its own.
    dspark._request_by_pool_slot[4] = RESTORED
    _round_trip(dspark, src_slot=4, dst_slot=6, request_keyed=True)
    # The import claims the new slot for the request, so the next prologue
    # does not reset the imported windows as a previous occupant's.
    assert dspark._request_by_pool_slot[6] == RESTORED
    with pytest.raises(ValueError, match="persistent state domain"):
        dspark.slot_state_rows(MAX_REQ_POOL_SIZE)


def test_dspark_images_an_unprepared_slot_as_nobodys():
    """A PD decode role's request retracted before its first decode never ran
    ``prepare_request_state`` here: the slot's windows are a previous
    occupant's, so the image carries no windows and the restore leaves the
    new slot for the first forward to prepare, as it would have unretracted."""
    states = _runtime_states(draft_probs=False, trees=False, history=False)
    dspark = _dspark(states)
    dspark._request_by_pool_slot[4] = "previous-occupant"
    dspark.kv_windows[4].fill_(3.0)
    dspark.context_lengths[4] = 11
    image = torch.full((dspark.slot_state_bytes(),), 0xEE, dtype=torch.uint8)
    dspark.export_slot_state(4, image, None, request_id=RESTORED)
    assert not read_prepared_marker(image)
    assert torch.all(image[PREPARED_MARKER_BYTES:] == 0xEE), "no windows imaged"

    dspark._request_by_pool_slot[6] = "stale-occupant"
    dspark.kv_windows[6].fill_(5.0)
    dspark.context_lengths[6] = 13
    dspark.import_slot_state(6, image, None, request_id=RESTORED)
    assert dspark._request_by_pool_slot[6] is None
    assert float(dspark.kv_windows[6].sum()) != 0, "the import wrote nothing"
    # The first forward prepares the slot: a flip, so the windows are reset.
    dspark.prepare_request_state([RESTORED], [6], num_extends=0)
    assert dspark._request_by_pool_slot[6] == RESTORED
    assert float(dspark.kv_windows[6].abs().sum()) == 0
    assert int(dspark.context_lengths[6]) == 0


def test_inkling_ring_round_trip_through_the_wrapper():
    backend = _inkling()
    _round_trip(backend, src_slot=3, dst_slot=9)
    # The wrapper images its own ring; its leaf is a separate (empty) node.
    assert backend.slot_state_exporters() == (backend, backend.child_backends()[0])


def test_dsa_kpool_tail_round_trip_once_per_arena():
    pool = _kpool_pool()
    target = _dsa(pool, is_draft=False)
    draft = _dsa(pool, is_draft=True)
    assert draft.slot_state_bytes() == 0, "the shared tail is imaged by the target"
    _round_trip(target, src_slot=2, dst_slot=5)


@pytest.mark.parametrize("cls", SAMPLING_BACKENDS, ids=lambda cls: cls.__name__)
def test_sampling_backend_round_trip_continues_where_the_victim_stopped(cls):
    """A restored request samples as the unretracted run would: its scalars,
    penalty history, bias and coin generator land in the new slot, and the
    next prepare_step sees no rid flip there -- so nothing is re-initialised."""
    backend = _sampling_backend(cls)
    victim, params = "req-victim", _sampling_params("req-victim")
    params.logit_bias = {"3": 2.0, "7": -1.5}
    src_slot, dst_slot = 4, 8
    # The victim's steps so far: scattered scalars (and a generator) on its
    # first step, then some accumulated history.
    backend.prepare_step([victim], [src_slot], [params], num_tokens_per_req=2)
    rows = backend.slot_state_rows(src_slot)
    for row in rows:
        if row.ndim == 1 and row.dtype == torch.int32:  # the penalty counts
            row[::3] = 2
    expected = [row.clone() for row in rows]
    # The victim's own generator stays where its last step left it: it is
    # the oracle for what an unretracted next step would have drawn.
    generator = (
        backend._cpu_generator_per_slot[src_slot]
        if isinstance(backend, FlashInferSamplingBackend)
        else None
    )

    assert backend.slot_state_bytes() % SLOT_STATE_ALIGNMENT == 0
    image = torch.full((backend.slot_state_bytes(),), 0xEE, dtype=torch.uint8)
    backend.export_slot_state(src_slot, image, None, request_id=victim)
    for row in rows:
        row.zero_()
    backend.import_slot_state(dst_slot, image, None, request_id=victim)
    assert all(
        torch.equal(a, b) for a, b in zip(backend.slot_state_rows(dst_slot), expected)
    )

    # The restored slot is the victim's: the next step must not reset it.
    with patch.object(
        type(backend), "_reset_slot", side_effect=AssertionError("slot was reset")
    ):
        backend.prepare_step([victim], [dst_slot], [params], num_tokens_per_req=2)
    if cls is GreedySamplingBackend:
        assert backend.slot_state_bytes() == 0
        return
    assert backend._last_rid_per_slot[dst_slot] == victim
    assert all(
        torch.equal(a, b) for a, b in zip(backend.slot_state_rows(dst_slot), expected)
    )
    if generator is not None:
        # The coin stream continues from the victim's last step: the step
        # above refilled the restored slot's coins (n = 2, plus the final
        # coin) from the restored generator, and they are exactly what the
        # unretracted generator draws next.
        assert backend._cpu_generator_per_slot[dst_slot] is not generator
        lo = coin_eps(torch.float32)
        unretracted_coins = torch.empty((1, 2), dtype=torch.float32)
        unretracted_coins[0, :2].uniform_(lo, 1.0, generator=generator)
        unretracted_final = torch.empty((1,), dtype=torch.float32)
        unretracted_final[0].uniform_(lo, 1.0, generator=generator)
        assert torch.equal(backend._coins_buf[0, :2], unretracted_coins[0])
        assert torch.equal(backend._final_coins_buf[0], unretracted_final[0])


@pytest.mark.parametrize("cls", SAMPLING_BACKENDS[1:], ids=lambda cls: cls.__name__)
@pytest.mark.parametrize("previous_occupant", [None, "req-previous"])
def test_sampling_backend_images_an_unprepared_slot_as_nobodys(cls, previous_occupant):
    """A PD decode role's request retracted between its landing and its first
    decode never went through prepare_step: the slot is either untouched or
    the previous occupant's (scalars, seed, coin generator). The image says
    so and carries no payload; the restore leaves the new slot unclaimed, and
    the request's first prepare_step resets it from its own SamplingParams,
    exactly as the unretracted first decode would have."""
    backend = _sampling_backend(cls)
    victim, params = "req-victim", _sampling_params("req-victim", temperature=0.3)
    src_slot, dst_slot = 4, 8
    if previous_occupant is not None:
        backend.prepare_step(
            [previous_occupant], [src_slot], [_sampling_params(previous_occupant)]
        )
    image = torch.full((backend.slot_state_bytes(),), 0xEE, dtype=torch.uint8)
    backend.export_slot_state(src_slot, image, None, request_id=victim)
    assert not read_prepared_marker(image)
    assert torch.all(image[PREPARED_MARKER_BYTES:] == 0xEE), "no payload imaged"

    # The new slot held someone else; the restore makes it nobody's.
    backend.prepare_step(["req-stale"], [dst_slot], [_sampling_params("req-stale")])
    backend.import_slot_state(dst_slot, image, None, request_id=victim)
    assert backend._last_rid_per_slot[dst_slot] is None
    with patch.object(type(backend), "_reset_slot", wraps=backend._reset_slot) as reset:
        backend.prepare_step([victim], [dst_slot], [params], num_tokens_per_req=2)
    reset.assert_called_once()
    assert reset.call_args.args[:2] == (dst_slot, params)
    assert backend._last_rid_per_slot[dst_slot] == victim
    assert float(backend._temperature_pool[dst_slot]) == pytest.approx(0.3)
    assert int(backend._seed_pool[dst_slot]) == int(params.seed)
    if isinstance(backend, FlashInferSamplingBackend):
        # The coin stream starts from the request's own seed: the step above
        # drew its coins from a generator seeded by the reset.
        expected = torch.Generator(device="cpu")
        expected.manual_seed(int(params.seed))
        lo = coin_eps(torch.float32)
        coins = torch.empty((1, 2), dtype=torch.float32)
        coins[0, :2].uniform_(lo, 1.0, generator=expected)
        assert torch.equal(backend._coins_buf[0, :2], coins[0])


def test_a_leaf_without_slot_state_images_nothing():
    leaf = _Leaf()
    assert leaf.slot_state_bytes() == 0
    leaf.export_slot_state(0, torch.zeros(0, dtype=torch.uint8), None, request_id="r")
    leaf.import_slot_state(0, torch.zeros(0, dtype=torch.uint8), None, request_id="r")


def test_model_executor_lists_every_owner_once_and_the_layout_is_fixed():
    states = _runtime_states(draft_probs=True, trees=False, history=False)
    executor = ModelExecutor.__new__(ModelExecutor)
    executor.runtime_states = states
    # A composite tree is flattened: the wrapper and its leaf are separate
    # segments, each imaging its own rows.
    executor.attn_backend = _inkling()
    leaf = executor.attn_backend.child_backends()[0]
    executor.drafter = _mtp(states)
    executor.sampling_backend = _sampling_backend(FlashInferFullSamplingBackend)
    # A shared draft tree is listed once; a distinct one adds its own nodes.
    executor.draft_attn_backend = executor.attn_backend
    assert executor.slot_state_exporters() == (
        states,
        executor.attn_backend,
        leaf,
        executor.drafter,
        executor.sampling_backend,
    )
    executor.draft_attn_backend = _inkling()
    exporters = executor.slot_state_exporters()
    assert exporters == (
        states,
        executor.attn_backend,
        leaf,
        executor.draft_attn_backend,
        executor.draft_attn_backend.child_backends()[0],
        executor.drafter,
        executor.sampling_backend,
    )
    # Slot 2 of the sampling backend must have been prepared to be imaged.
    executor.sampling_backend.prepare_step(["r2"], [2], [_sampling_params("r2")])

    # The layout measures every owner once at construction and only slices
    # afterwards: no owner is asked its size again on export or import.
    layout = SlotStateLayout(exporters)
    assert layout.nbytes == sum(owner.slot_state_bytes() for owner in exporters)
    assert [owner for owner, _, _ in layout.segments] == list(exporters)
    assert leaf.slot_state_bytes() == 0
    rows = lambda slot: [  # noqa: E731
        row for owner in exporters for row in owner.slot_state_rows(slot)
    ]
    torch.manual_seed(1)
    _randomize(rows(2))
    expected = [row.clone() for row in rows(2)]
    image = torch.zeros(layout.nbytes, dtype=torch.uint8)
    with patch.object(
        RuntimeStates, "slot_state_bytes", side_effect=AssertionError("re-measured")
    ):
        layout.export(2, image, None, request_id="r2")
        for row in rows(2):
            row.zero_()
        layout.import_(6, image, None, request_id="r6")
    assert all(torch.equal(a, b) for a, b in zip(rows(6), expected))
    with pytest.raises(ValueError, match="need"):
        layout.export(2, image[:-SLOT_STATE_ALIGNMENT], None, request_id="r2")

    class _Unpadded:
        def slot_state_bytes(self):
            return 3

    with pytest.raises(ValueError, match="not padded"):
        SlotStateLayout([_Unpadded()])


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
    declared = set(getattr(type(owner), "token_derived_slot_state", ())) | set(
        getattr(type(owner), "constant_slot_state", ())
    )
    uncovered = sorted(
        name
        for name, tensor in _per_slot_tensors(owner, slot_domains).items()
        if tensor.untyped_storage().data_ptr() not in exported and name not in declared
    )
    assert not uncovered, (
        f"{type(owner).__name__} keeps per-slot state with no exporter: {uncovered}. "
        "List it in slot_state_rows (restored byte for byte), in "
        "token_derived_slot_state (reseeded from the request's tokens) or in "
        "constant_slot_state (identical for every slot)."
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


@pytest.mark.parametrize("cls", SAMPLING_BACKENDS, ids=lambda cls: cls.__name__)
def test_sampling_backend_slot_state_is_complete(cls):
    backend = _sampling_backend(cls)
    _assert_slot_state_complete(backend, exporter=backend, slot_domains={POOL_ROWS})
    # The declared constant pools exist, are per-slot, and really are constant.
    per_slot = _per_slot_tensors(backend, {POOL_ROWS})
    for name in SamplingBackend.constant_slot_state:
        assert name in per_slot
        assert per_slot[name].abs().sum() == 0
    # Every per-slot thing the rid-flip reset writes is in the image.
    if cls is not GreedySamplingBackend:
        exported = _storage_ids(backend.slot_state_rows(0))
        written = []
        for name, tensor in per_slot.items():
            if name in SamplingBackend.constant_slot_state:
                continue
            tensor.fill_(0)
            backend._reset_slot(2, _sampling_params("r2"))
            if tensor[2].abs().sum() != 0:
                written.append(name)
            assert tensor.untyped_storage().data_ptr() in exported, name
        assert "_temperature_pool" in written and "_seed_pool" in written


def test_completeness_check_catches_an_unexported_buffer():
    states = _runtime_states(draft_probs=False, trees=False, history=False)
    states.rogue_per_slot = torch.zeros(POOL_ROWS, dtype=torch.int32)
    with pytest.raises(AssertionError, match="rogue_per_slot"):
        _assert_slot_state_complete(states, exporter=states, slot_domains={POOL_ROWS})
    backend = _sampling_backend(TritonFullSamplingBackend)
    backend._rogue_pool = torch.zeros(POOL_ROWS, VOCAB, dtype=torch.int32)
    with pytest.raises(AssertionError, match="_rogue_pool"):
        _assert_slot_state_complete(backend, exporter=backend, slot_domains={POOL_ROWS})
