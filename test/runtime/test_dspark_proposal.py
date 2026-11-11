"""DSpark's semi-autoregressive proposal and the verify-side state commit.

The draft's backbone pass is non-autoregressive: all block positions are
computed at once from one anchor plus mask tokens. The only thing carrying
token-level dependence inside the block is the Markov head, which adds a
learned bigram bias conditioned on the *previously proposed* token. These tests
pin that chaining, the vocab-shard handling of the bias, and the fact that K3's
recurrent KDA state still gets committed after a DSpark verify.
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest
import torch
from tokenspeed_kernel.ops.sampling.cute_dsl import MarkovArgmaxWorkspace

from tokenspeed.runtime.execution.drafter import dspark as dspark_module
from tokenspeed.runtime.execution.drafter.dspark import DSpark
from tokenspeed.runtime.execution.input_buffer import InputBuffers
from tokenspeed.runtime.models.dspark import VanillaMarkov

VOCAB = 32
RANK = 4
HIDDEN = 6


def _drafter(spec_num_tokens: int = 8, vocab: int = VOCAB) -> DSpark:
    """A DSpark drafter shell carrying only the proposal state."""
    drafter = DSpark.__new__(DSpark)
    drafter.spec_num_tokens = spec_num_tokens
    drafter.lm_head = None
    drafter.draft_sampler = None  # the greedy walk
    # Resolved without the fused step: the default walk is the unfused one.
    drafter._markov_argmax_resolved = True
    drafter._markov_argmax_workspace = None
    torch.manual_seed(0)
    head = VanillaMarkov(vocab_size=vocab, markov_rank=RANK)
    drafter.markov_head = head
    return drafter


# --------------------------------------------------------------------------
# Draft prefill capture gate
# --------------------------------------------------------------------------


@pytest.mark.parametrize("declares", [False, True])
@pytest.mark.parametrize("prefill_disabled", [False, True])
@pytest.mark.parametrize("decode_disabled", [False, True])
def test_draft_prefill_capture_uses_the_resolved_prefill_gate(
    monkeypatch, prefill_disabled, decode_disabled, declares
):
    from tokenspeed.runtime.execution import model_executor
    from tokenspeed.runtime.execution.memory_delta import (
        NULL_MEMORY_DELTA_OBSERVER,
    )

    monkeypatch.setattr(model_executor, "workspace_pool", Mock())
    # A real instance, so the gate resolves the production property.
    executor = model_executor.ModelExecutor.__new__(model_executor.ModelExecutor)
    executor.device = "cuda"
    # Without --enable-drafter-prefill-graph: the declaration alone decides.
    executor.config = Mock(enable_drafter_prefill_graph=False)
    executor.forward_step = Mock(disable=decode_disabled)
    executor.prefill_graph = Mock(disable=prefill_disabled)
    # Explicit: a bare Mock answers truthily, so the gate below would hold anyway.
    executor.drafter = Mock(captures_prefill_graph=declares)
    model_executor.ModelExecutor.capture_graphs(
        executor, entries=None, observer=NULL_MEMORY_DELTA_OBSERVER
    )
    executor.drafter.capture_extend_graphs.assert_not_called()
    if prefill_disabled or not declares:
        executor.drafter.capture_prefill_graph.assert_not_called()
    else:
        # The drafter's own graph is a measured ladder of its own.
        executor.drafter.capture_prefill_graph.assert_called_once()
        (stream, _observer), _ = executor.drafter.capture_prefill_graph.call_args
        assert stream is executor.forward_step.stream


# --------------------------------------------------------------------------
# The Markov bias
# --------------------------------------------------------------------------


def test_bias_matches_the_heads_own_step_bias() -> None:
    drafter = _drafter()
    prev = torch.tensor([3, 17])
    bias_fn = drafter._make_step_bias_fn(prev)

    produced = bias_fn(0, VOCAB)
    expected = drafter.markov_head.compute_step_bias(prev)
    torch.testing.assert_close(produced, expected)


def test_bias_slices_match_the_full_bias() -> None:
    """A vocab-parallel shard must see exactly its own slice of the bias."""
    drafter = _drafter()
    prev = torch.tensor([5, 5, 9])
    bias_fn = drafter._make_step_bias_fn(prev)
    full = bias_fn(0, VOCAB)

    for start, count in ((0, 8), (8, 8), (24, 8)):
        torch.testing.assert_close(
            bias_fn(start, count), full[:, start : start + count]
        )


def test_out_of_range_previous_tokens_do_not_index_past_the_embedding() -> None:
    """The anchor is the target's last output and each step's input is a
    vocab-parallel argmax; both can be out of range before the loop's own
    clamp runs, and the Markov head embeds them.
    """
    drafter = _drafter()
    for prev in (
        torch.tensor([-1, 0]),
        torch.tensor([VOCAB, VOCAB + 500]),
        torch.tensor([-7, VOCAB * 3]),
    ):
        bias = drafter._make_step_bias_fn(prev)(0, VOCAB)
        assert bias.shape == (2, VOCAB)
        assert torch.isfinite(bias).all()


def test_clamping_maps_to_the_boundary_rows() -> None:
    drafter = _drafter()
    clamped = drafter._make_step_bias_fn(torch.tensor([-4, VOCAB + 9]))(0, VOCAB)
    expected = drafter._make_step_bias_fn(torch.tensor([0, VOCAB - 1]))(0, VOCAB)
    torch.testing.assert_close(clamped, expected)


def test_bias_is_zero_beyond_the_markov_vocabulary() -> None:
    """The target lm_head's added-vocab shard has no Markov row to read.

    Biasing it with whatever memory follows markov_w2 would let padding tokens
    win the argmax, so the added shard must compete unbiased.
    """
    drafter = _drafter()
    bias_fn = drafter._make_step_bias_fn(torch.tensor([1, 2]))

    beyond = bias_fn(VOCAB, 4)
    assert beyond.shape == (2, 4)
    assert torch.all(beyond == 0)


def test_bias_straddling_the_vocab_edge_is_partly_real_partly_zero() -> None:
    drafter = _drafter()
    prev = torch.tensor([7])
    bias_fn = drafter._make_step_bias_fn(prev)
    full = bias_fn(0, VOCAB)

    straddling = bias_fn(VOCAB - 3, 6)
    torch.testing.assert_close(straddling[:, :3], full[:, VOCAB - 3 :])
    assert torch.all(straddling[:, 3:] == 0)


# --------------------------------------------------------------------------
# The proposal chain
# --------------------------------------------------------------------------


def _install_recording_argmax(drafter: DSpark, lm_head_weight: torch.Tensor) -> list:
    """Replace the vocab-parallel argmax with a recording torch reference."""
    seen: list[tuple[int, torch.Tensor]] = []

    def fake_argmax(hidden, out=None, bias_fn=None):
        logits = hidden.float() @ lm_head_weight.float().T
        if bias_fn is not None:
            logits = logits + bias_fn(0, logits.shape[-1]).float()
        argmax = torch.argmax(logits, dim=-1)
        seen.append((len(seen), argmax.clone()))
        if out is not None:
            out.copy_(argmax.view_as(out))
            return out
        return argmax

    drafter._greedy_argmax_vocab_parallel = fake_argmax
    return seen


def test_anchor_is_copied_and_drafts_fill_the_rest_of_the_block() -> None:
    spec = 8
    drafter = _drafter(spec_num_tokens=spec)
    torch.manual_seed(1)
    lm_head = torch.randn(VOCAB, HIDDEN)
    _install_recording_argmax(drafter, lm_head)

    bs = 2
    draft_hidden = torch.randn(bs, spec - 1, HIDDEN)
    block_ids = torch.full((bs, spec - 1), 11, dtype=torch.int32)
    block_ids[:, 0] = torch.tensor([4, 9], dtype=torch.int32)
    next_tokens = torch.zeros((bs, spec), dtype=torch.int32)

    out = drafter._sample_block(draft_hidden, block_ids, next_tokens)

    # Column 0 is the anchor verbatim; the remaining 7 are proposals.
    assert out[:, 0].tolist() == [4, 9]
    assert out.shape == (bs, spec)


def test_each_step_reads_the_hidden_one_position_back() -> None:
    """Block position k is proposed from draft_hidden[k-1], not [k].

    This is the off-by-one PR #829's last commit fixed; getting it wrong shifts
    every draft by one position and quietly halves acceptance.
    """
    spec = 4
    drafter = _drafter(spec_num_tokens=spec)
    lm_head = torch.zeros(VOCAB, HIDDEN)
    # Make position p's hidden select token p deterministically.
    for p in range(spec - 1):
        lm_head[p, p % HIDDEN] = 1.0
    drafter.markov_head.markov_w2.weight.data.zero_()

    draft_hidden = torch.zeros(1, spec - 1, HIDDEN)
    for p in range(spec - 1):
        draft_hidden[0, p, p % HIDDEN] = 10.0

    _install_recording_argmax(drafter, lm_head)
    next_tokens = torch.zeros((1, spec), dtype=torch.int32)
    block_ids = torch.zeros((1, spec - 1), dtype=torch.int32)

    out = drafter._sample_block(draft_hidden, block_ids, next_tokens)
    # Token at column k came from hidden row k-1, which selects token k-1.
    assert out[0, 1:].tolist() == [0, 1, 2]


def test_the_chain_conditions_on_the_previous_proposal() -> None:
    """Changing only the anchor must change the downstream proposals.

    If the Markov bias were dropped, the block would be a pure function of the
    hidden states and the anchor would not propagate at all.
    """
    spec = 5
    torch.manual_seed(3)
    lm_head = torch.randn(VOCAB, HIDDEN)
    draft_hidden = torch.randn(1, spec - 1, HIDDEN)

    proposals = []
    for anchor in (2, 21):
        drafter = _drafter(spec_num_tokens=spec)
        # A strong, token-dependent bias so the chain is visible.
        drafter.markov_head.markov_w1.weight.data.normal_(0.0, 3.0)
        drafter.markov_head.markov_w2.weight.data.normal_(0.0, 3.0)
        _install_recording_argmax(drafter, lm_head)
        block_ids = torch.zeros((1, spec - 1), dtype=torch.int32)
        block_ids[0, 0] = anchor
        next_tokens = torch.zeros((1, spec), dtype=torch.int32)
        proposals.append(
            drafter._sample_block(draft_hidden, block_ids, next_tokens).clone()
        )

    assert proposals[0][0, 1:].tolist() != proposals[1][0, 1:].tolist()


def test_proposals_are_valid_token_ids() -> None:
    """clamp_(min=0) guards the vocab-parallel all-shards-lost case."""
    spec = 6
    drafter = _drafter(spec_num_tokens=spec)
    _install_recording_argmax(drafter, torch.randn(VOCAB, HIDDEN))
    out = drafter._sample_block(
        torch.randn(1, spec - 1, HIDDEN),
        torch.zeros((1, spec - 1), dtype=torch.int32),
        torch.zeros((1, spec), dtype=torch.int32),
    )
    assert int(out.min()) >= 0


class _ShardIndices:
    def __init__(self, num_org: int, start: int) -> None:
        self.num_org_elements = num_org
        self.org_vocab_start_index = start


class _ShardedHead:
    """Enough of a vocab-parallel head for the block projection to run.

    The shard of an ``org_vocab_size``-token vocabulary starts at ``start``.
    """

    def __init__(
        self, weight: torch.Tensor, start: int = 0, org_vocab_size: int = VOCAB
    ) -> None:
        self.weight = weight
        self.shard_indices = _ShardIndices(weight.shape[0], start)
        self.org_vocab_size = org_vocab_size


def test_the_walk_feeds_each_round_its_own_positions_hoisted_slice() -> None:
    """The whole proposal walk must hand round k the hoisted projection of
    position k-1, allowing float32 GEMM rounding across batch shapes."""
    torch.manual_seed(11)
    drafter = _drafter()
    weight = torch.randn(VOCAB, HIDDEN)
    drafter.lm_head = _ShardedHead(weight)
    received: list[torch.Tensor] = []

    def fake_argmax(hidden, out=None, bias_fn=None, base_logits=None):
        assert base_logits is not None, "the walk must use the hoisted slice"
        assert base_logits.is_contiguous()
        received.append(base_logits.clone())
        logits = base_logits.float()
        if bias_fn is not None:
            logits = logits + bias_fn(0, logits.shape[-1]).float()
        argmax = torch.argmax(logits, dim=-1)
        if out is not None:
            out.copy_(argmax.view_as(out))
            return out
        return argmax

    drafter._greedy_argmax_vocab_parallel = fake_argmax
    hidden = torch.randn(3, drafter.spec_num_tokens, HIDDEN)
    ids = torch.zeros(3, drafter.spec_num_tokens, dtype=torch.long)
    block = torch.randint(0, VOCAB, (3, drafter.spec_num_tokens))
    drafter._sample_block(hidden, block, ids)

    assert len(received) == drafter.spec_num_tokens - 1
    for k, got in enumerate(received, start=1):
        want = hidden[:, k - 1, :].to(weight.dtype) @ weight.T
        torch.testing.assert_close(got, want)


def test_the_block_projection_matches_projecting_each_position() -> None:
    """The walk is semi-autoregressive, so the projection is hoisted out of it
    while the bias and the argmax stay behind. A layout slip here would feed
    each position another position's logits and go unnoticed: every proposal
    would still be a valid token id."""
    rows, positions = 3, 8
    drafter = _drafter(spec_num_tokens=positions)
    torch.manual_seed(1)
    weight = torch.randn(VOCAB, HIDDEN)
    drafter.lm_head = _ShardedHead(weight)
    draft_hidden = torch.randn(rows, positions, HIDDEN)

    block = drafter._block_base_logits(draft_hidden)

    assert block is not None
    assert block.shape == (positions, rows, VOCAB)
    for k in range(positions):
        expected = torch.matmul(draft_hidden[:, k, :], weight.T)
        assert torch.allclose(block[k], expected, atol=1e-5), f"position {k}"


def test_the_block_projection_stands_down_without_a_sharded_head() -> None:
    """Drafters whose head carries no shard metadata keep the per-step path."""
    drafter = _drafter(spec_num_tokens=4)
    assert drafter._block_base_logits(torch.randn(2, 4, HIDDEN)) is None

    drafter.lm_head = _ShardedHead(torch.randn(0, HIDDEN))
    assert drafter._block_base_logits(torch.randn(2, 4, HIDDEN)) is None


# --------------------------------------------------------------------------
# The fused step
# --------------------------------------------------------------------------


def _eighths(*shape: int, seed: int) -> torch.Tensor:
    """Draw BF16 values in {-1/8, 0, 1/8}.

    Base logits and Markov biases are then small multiples of 1/64, exact in
    BF16 in any accumulation order and of one magnitude, so both decide
    proposals.
    """
    gen = torch.Generator().manual_seed(seed)
    return (torch.randint(-1, 2, shape, generator=gen) / 8).to(torch.bfloat16)


def _kernel_contract(workspace, base, prev, embedding, projection, start, out):
    """markov_argmax_step on one rank, written in torch: the kernel needs a GPU.

    Clamp the previous token, round the FP32 bias to the logits dtype, add in
    that dtype and write the global id of the argmax.
    """
    assert workspace.vocab_per_rank == base.shape[-1] == projection.shape[0]
    latent = embedding[prev.long().clamp(0, embedding.shape[0] - 1)]
    bias = (latent.float() @ projection.float().T).to(base.dtype)
    out.copy_(torch.argmax((base + bias).float(), dim=-1) + start)


def _shard_walk(fused: bool, monkeypatch) -> torch.Tensor:
    """Walk a rank whose head shard is tokens [8, 24) of the 32-token vocab."""
    spec, rows, start, width = 6, 3, 8, 16
    drafter = _drafter(spec_num_tokens=spec)
    markov = drafter.markov_head.to(torch.bfloat16)
    markov.markov_w1.weight.data = _eighths(VOCAB, RANK, seed=1)
    markov.markov_w2.weight.data = _eighths(VOCAB, RANK, seed=2)
    drafter.lm_head = _ShardedHead(_eighths(width, HIDDEN, seed=3), start=start)
    hidden = _eighths(rows, spec, HIDDEN, seed=4)
    block_ids = torch.tensor([[5], [31], [-3]], dtype=torch.int32).expand(
        rows, spec - 1
    )
    next_tokens = torch.full((rows, spec), -1, dtype=torch.int32)

    def unfused_argmax(hidden, out=None, bias_fn=None, base_logits=None):
        assert not fused, "the fused walk must not run the unfused step"
        assert hidden.shape[0] == base_logits.shape[0] == rows
        bias = bias_fn(start, base_logits.shape[-1]).to(base_logits.dtype)
        out.copy_(torch.argmax((base_logits + bias).float(), dim=-1) + start)
        return out

    drafter._greedy_argmax_vocab_parallel = unfused_argmax
    if fused:
        drafter._markov_argmax_workspace = MarkovArgmaxWorkspace(
            state=None,
            device=torch.device("cpu"),
            max_rows=rows,
            vocab_per_rank=width,
            rank_dim=RANK,
            dtype=torch.bfloat16,
            keys=torch.empty(0, dtype=torch.int64),
            ctas_done=torch.empty(0, dtype=torch.int32),
            kernels={},
        )
        monkeypatch.setattr(dspark_module, "markov_argmax_step", _kernel_contract)
    return drafter._sample_block(hidden, block_ids, next_tokens).clone()


def test_the_fused_walk_proposes_what_the_unfused_walk_does(monkeypatch) -> None:
    """The fused walk proposes exactly what the unfused walk does.

    Each fused position must read its own base logits, the token just written
    for the previous position (clamped), this rank's projection rows, and its
    vocabulary offset.
    """
    unfused = _shard_walk(False, monkeypatch)
    fused = _shard_walk(True, monkeypatch)
    assert unfused[:, 0].tolist() == [5, 31, 0]
    assert ((unfused[:, 1:] >= 8) & (unfused[:, 1:] < 24)).all()
    assert torch.equal(fused, unfused)


def _gate_drafter(markov_dtype, markov_vocab: int, state) -> DSpark:
    drafter = _drafter()
    drafter._markov_argmax_resolved = False
    drafter.markov_head = VanillaMarkov(vocab_size=markov_vocab, markov_rank=128).to(
        markov_dtype
    )
    drafter.lm_head = _ShardedHead(
        torch.zeros(16, HIDDEN, dtype=torch.bfloat16), start=16, org_vocab_size=32
    )
    drafter._dist_argmax_state = state
    drafter.input_buffers = InputBuffers(
        max_bs=12, max_num_tokens=96, state_write_padding_pool_index=0, device="cpu"
    )
    return drafter


@pytest.mark.parametrize(
    "markov_dtype, markov_vocab, has_state, fused",
    [
        (torch.bfloat16, 32, True, True),
        (torch.bfloat16, 32, False, False),  # no distributed-argmax slots
        (torch.float32, 32, True, False),  # Markov head in another dtype
        (torch.bfloat16, 24, True, False),  # a shard without Markov rows
    ],
)
def test_the_fused_step_resolves_once_where_the_distributed_argmax_runs(
    monkeypatch, markov_dtype, markov_vocab, has_state, fused
) -> None:
    created = []

    def create(state, **kwargs):
        created.append((state, kwargs))
        return "workspace"

    monkeypatch.setattr(dspark_module, "create_markov_argmax_workspace", create)
    state = object() if has_state else None
    drafter = _gate_drafter(markov_dtype, markov_vocab, state)
    block_logits = torch.zeros(3, 16, dtype=torch.bfloat16)

    for _ in range(2):
        got = drafter._resolve_markov_argmax(block_logits)
        assert got == ("workspace" if fused else None)
    assert drafter._markov_argmax_resolved
    if fused:
        assert created == [
            (
                state,
                dict(
                    device=block_logits.device,
                    max_rows=12,
                    vocab_per_rank=16,
                    rank_dim=128,
                    dtype=torch.bfloat16,
                ),
            )
        ]
    else:
        assert created == []


def test_the_fused_step_is_never_resolved_inside_a_capture(monkeypatch) -> None:
    """A first walk inside a capture keeps the unfused step for now.

    Allocation and compilation belong to warmup, so the verdict waits for an
    eager walk.
    """
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    monkeypatch.setattr(
        dspark_module,
        "create_markov_argmax_workspace",
        lambda *a, **k: pytest.fail("created inside a capture"),
    )
    drafter = _gate_drafter(torch.bfloat16, 32, object())
    assert drafter._resolve_markov_argmax(torch.zeros(3, 16)) is None
    assert not drafter._markov_argmax_resolved
