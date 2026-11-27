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

"""A DFLASH draft under ``--enable-draft-forward-batch-invariant``.

Checks the two parts of the draft whose result depends on how many rows ride
together: the context K|V write must run the batch-invariant GEMM leaf its plan
selects (per layer and stacked), and the block proposal's tokens and recorded
``q`` must depend only on the request's seed, frontier and draft step. A
reference GEMM provider that sums K in index order stands in for the platform's
leaf. The attention leaf is tested in ``tokenspeed-kernel``.
"""

import os
import sys
from types import SimpleNamespace

import pytest
import torch
from tokenspeed_kernel.registry import KernelRegistry, KernelSpec, Priority
from tokenspeed_kernel.selection import NoKernelFoundError
from tokenspeed_kernel.signature import format_signatures
from transformers import PretrainedConfig

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=60, suite="runtime-1gpu")

from tokenspeed.runtime.distributed.mapping import Mapping  # noqa: E402
from tokenspeed.runtime.execution.drafter import dflash as drafter_module  # noqa: E402
from tokenspeed.runtime.execution.drafter.dflash import DFlash  # noqa: E402
from tokenspeed.runtime.execution.drafter.speculative_sampling import (  # noqa: E402
    DraftProposalSampler,
)
from tokenspeed.runtime.execution.input_buffer import InputBuffers  # noqa: E402
from tokenspeed.runtime.execution.runtime_states import RuntimeStates  # noqa: E402
from tokenspeed.runtime.layers.attention.backends.paged.mha import (  # noqa: E402
    MHAAttnBackend,
)
from tokenspeed.runtime.layers.attention.configs.base import AttnConfig  # noqa: E402
from tokenspeed.runtime.layers.attention.configs.mha import MHAConfig  # noqa: E402
from tokenspeed.runtime.layers.logits_processor import LogitsProcessor  # noqa: E402
from tokenspeed.runtime.layers.paged_attention import PagedAttention  # noqa: E402
from tokenspeed.runtime.layers.vocab_parallel_embedding import (  # noqa: E402
    ParallelLMHead,
)
from tokenspeed.runtime.model_loader.loader import (  # noqa: E402
    post_load_transformed_modules,
    set_default_torch_dtype,
)
from tokenspeed.runtime.models import dflash as model_module  # noqa: E402
from tokenspeed.runtime.models.dflash import (  # noqa: E402
    DFlashAttention,
    DFlashDecoderLayer,
    DFlashDraftModel,
)
from tokenspeed.runtime.sampling.backends.base import (  # noqa: E402
    SpeculativeSamplingPools,
)
from tokenspeed.runtime.utils.env import global_server_args_dict  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")

PROVIDER = "test_row_ordered_mm"
HEADS = 32
KV_HEADS = 8
HEAD_DIM = 128


def row_ordered_mm(a, b, a_scales, b_scales, out_dtype, *, alpha, block_size, out=None):
    """``a @ b.T`` with the sum over K taken in index order in fp32, which makes
    a row's bits independent of the other rows."""
    total = torch.zeros(a.shape[0], b.shape[0], dtype=torch.float32, device=a.device)
    for k in range(a.shape[1]):
        total.add_(a[:, k, None].float() * b[None, :, k].float())
    result = total.to(out_dtype)
    if out is not None:
        out.copy_(result)
        return out
    return result


@pytest.fixture
def flag(monkeypatch):
    """Run under the bitwise envelope with ``--enable-draft-forward-batch-invariant`` on;
    ``flag(False)`` is the same launch without it. The reference GEMM provider
    is registered at the lowest priority, so only a plan that requires
    ``batch_invariant`` reaches it."""
    registry = KernelRegistry.get()
    snapshot = registry.snapshot()
    registry.register(
        KernelSpec(
            name=PROVIDER,
            family="gemm",
            mode="mm",
            solution="test_reference",
            format_signatures=format_signatures(("a", "b"), "dense", {torch.bfloat16}),
            features=frozenset({"batch_invariant"}),
            priority=int(Priority.REFERENCE),
        ),
        row_ordered_mm,
    )
    monkeypatch.setitem(global_server_args_dict, "numerics", "rl-bitwise")

    def select(enabled: bool) -> None:
        monkeypatch.setitem(
            global_server_args_dict, "enable_draft_forward_batch_invariant", enabled
        )

    select(True)
    yield select
    registry.restore(snapshot)


# ---------------------------------------------------------------------------
# Context K|V write
# ---------------------------------------------------------------------------

HIDDEN = 2048
ROWS = (1, 2, 8, 16, 64, 128, 256, 512)


def _config() -> PretrainedConfig:
    return PretrainedConfig(
        hidden_size=HIDDEN,
        num_hidden_layers=2,
        num_attention_heads=HEADS,
        num_key_value_heads=KV_HEADS,
        head_dim=HEAD_DIM,
        intermediate_size=5120,
        hidden_act="silu",
        rms_norm_eps=1e-5,
        rope_theta=10000.0,
        max_position_embeddings=4096,
        layer_types=["sliding_attention"] * 2,
        sliding_window=1024,
        sliding_window_sink=16,
        dflash_config={"target_layer_ids": [1, 2, 3, 4], "block_size": 8},
    )


def _build(config=None, **kwargs) -> DFlashDraftModel:
    """Built as the model loader builds it: in the checkpoint's dtype, on the
    device, so the plans made at construction name the dtype the weights hold."""
    with set_default_torch_dtype(torch.bfloat16), torch.device("cuda"):
        return DFlashDraftModel(
            _config() if config is None else config,
            Mapping(rank=0, world_size=1),
            **kwargs,
        )


def _model() -> DFlashDraftModel:
    torch.manual_seed(0)
    model = _build()
    # Constructed weights are uninitialised and may be zero.
    for layer in model.layers:
        layer.self_attn.qkv_proj.weight.data.normal_(0, 0.02)
    return model


def _attention() -> DFlashAttention:
    return _model().layers[0].self_attn


def _drafter() -> DFlash:
    """A DFlash holding a real two-layer draft and a pool of the right shape."""
    model = _model()

    class Pool:
        def get_key_buffer(self, layer_id):
            return torch.zeros(
                4, 128, KV_HEADS, HEAD_DIM, device="cuda", dtype=torch.bfloat16
            )

        get_value_buffer = get_key_buffer

    drafter = DFlash.__new__(DFlash)
    drafter.draft_model_runner = SimpleNamespace(
        model=model, mapping=SimpleNamespace(attn=SimpleNamespace(kvp_size=1))
    )
    drafter.token_to_kv_pool = Pool()
    drafter.device = torch.device("cuda")
    drafter.spec_num_tokens = 8
    drafter.input_buffers = InputBuffers(
        max_bs=max(ROWS) // drafter.spec_num_tokens,
        max_num_tokens=max(ROWS),
        state_write_padding_pool_index=0,
        device="cuda",
    )
    return drafter


def _rows() -> torch.Tensor:
    generator = torch.Generator(device="cuda").manual_seed(1)
    return torch.randn(
        max(ROWS), HIDDEN, generator=generator, device="cuda", dtype=torch.bfloat16
    )


def test_the_flag_plans_the_kv_gemm_and_off_does_not(flag) -> None:
    assert _attention().kv_proj_plan.kernel.name == PROVIDER
    flag(False)
    assert _attention().kv_proj_plan is None


@pytest.mark.parametrize("count", ROWS)
def test_the_per_layer_write_projects_with_the_planned_leaf(flag, count) -> None:
    layer = _attention()
    rows = _rows()[:count]
    k, v = layer.kv_proj_only(rows)
    weight = layer.qkv_proj.weight[layer.q_size : layer.q_size + 2 * layer.kv_size]
    expected = row_ordered_mm(
        rows, weight, None, None, torch.bfloat16, alpha=None, block_size=None
    )
    assert torch.equal(torch.cat([k, v], dim=-1), expected)


def _stacked_projections(drafter: DFlash, monkeypatch) -> list[torch.Tensor]:
    """What ``_write_native_cache_fused`` hands its scatter for each row count.

    The scatter needs a real paged pool, so it is replaced; the projection it
    receives is the object under test.
    """
    seen = []
    monkeypatch.setattr(
        drafter_module,
        "_fused_norm_rope_stacked_scatter",
        lambda proj, *args, **kwargs: seen.append(proj.clone()),
    )
    rows = _rows()
    positions = torch.zeros(max(ROWS), dtype=torch.int64, device="cuda")
    locs = torch.zeros(max(ROWS), dtype=torch.int32, device="cuda")
    for count in ROWS:
        drafter._write_native_cache_fused(
            rows[:count].contiguous(), positions[:count], locs[:count]
        )
    return seen


def test_the_fused_write_projects_with_the_planned_leaf(flag, monkeypatch) -> None:
    drafter = _drafter()
    drafter._init_fused_kv_helper()
    assert drafter._fused_kv_enabled
    assert drafter._fused_kv_plan.kernel.name == PROVIDER
    stacked = drafter._fused_kv_flat_weight
    rows = _rows()
    for count, seen in zip(
        ROWS, _stacked_projections(drafter, monkeypatch), strict=True
    ):
        expected = row_ordered_mm(
            rows[:count],
            stacked,
            None,
            None,
            torch.bfloat16,
            alpha=None,
            block_size=None,
        )
        assert torch.equal(seen.reshape(count, -1), expected), count


def test_the_fused_write_is_unplanned_without_the_flag(flag) -> None:
    flag(False)
    drafter = _drafter()
    drafter._init_fused_kv_helper()
    assert drafter._fused_kv_enabled
    assert drafter._fused_kv_plan is None


def test_no_batch_invariant_leaf_stops_the_launch_instead_of_slowing_it(
    flag, monkeypatch
) -> None:
    """``_init_fused_kv_helper`` degrades to the per-layer write on any error
    inside it, which is right for a pool or a layout the fusion cannot serve.
    A flag whose GEMM has no batch-invariant leaf is not that: serving on
    would break what the flag promises, so the planning error must reach the
    launch."""

    def no_leaf(**kwargs):
        raise NoKernelFoundError("no batch-invariant leaf")

    drafter = _drafter()
    monkeypatch.setattr(drafter_module, "plan_dense_gemm", no_leaf)
    with pytest.raises(NoKernelFoundError):
        drafter._init_fused_kv_helper()


def test_the_draft_stays_reloadable_in_place(flag) -> None:
    """Planning at construction leaves no post-load hook, so the draft is not
    counted among the modules a checkpoint reload cannot rewrite."""
    assert post_load_transformed_modules(_model()) == []


# ---------------------------------------------------------------------------
# Block attention
# ---------------------------------------------------------------------------

CONTEXT = 2048
PAGE = 128
ATTN_BATCHES = (1, 2, 8, 16)


class _Cache:
    """KV storage for one layer; the backend and the kernels stay real."""

    def __init__(self, pages: int):
        generator = torch.Generator(device="cuda").manual_seed(3)
        self.k, self.v = (
            torch.randn(
                pages,
                PAGE,
                KV_HEADS,
                HEAD_DIM,
                generator=generator,
                device="cuda",
                dtype=torch.bfloat16,
            )
            for _ in range(2)
        )

    def get_key_buffer(self, layer_id):
        return self.k

    def get_value_buffer(self, layer_id):
        return self.v


def _backend(*, is_draft: bool) -> MHAAttnBackend:
    spec = MHAConfig(
        backend_name="triton",
        num_attention_heads=HEADS,
        num_kv_heads=KV_HEADS,
        head_dim=HEAD_DIM,
        attn_tp_size=1,
    )
    config = AttnConfig(
        device="cuda",
        dtype=torch.bfloat16,
        kv_cache_dtype=torch.bfloat16,
        kv_cache_quant_method="none",
        context_len=CONTEXT,
        max_bs=max(ATTN_BATCHES),
        max_resident_requests=max(ATTN_BATCHES),
        speculative_num_draft_tokens=8,
        is_draft=is_draft,
        draft_block_decode=is_draft,
        components=(spec,),
    )
    return MHAAttnBackend(config, spec, kernel_page_size=PAGE)


def _block_decode(bs: int) -> torch.Tensor:
    """The block rows of the first ``bs`` requests, decoded through the real
    attention backend of a block draft with a sliding window and a sink."""
    spec = MHAConfig(
        backend_name="triton",
        num_attention_heads=HEADS,
        num_kv_heads=KV_HEADS,
        head_dim=HEAD_DIM,
        attn_tp_size=1,
    )
    config = AttnConfig(
        device="cuda",
        dtype=torch.bfloat16,
        kv_cache_dtype=torch.bfloat16,
        kv_cache_quant_method="none",
        context_len=CONTEXT,
        max_bs=max(ATTN_BATCHES),
        max_resident_requests=max(ATTN_BATCHES),
        speculative_num_draft_tokens=8,
        is_draft=True,
        draft_block_decode=True,
        components=(spec,),
    )
    backend = MHAAttnBackend(config, spec, kernel_page_size=PAGE)
    backend.init_cuda_graph_state(max(ATTN_BATCHES))
    layer = PagedAttention(
        HEADS,
        HEAD_DIM,
        HEAD_DIM**-0.5,
        KV_HEADS,
        0,
        sliding_window_size=1024,
        sliding_window_sink=16,
        rotary_emb=None,
        qk_norm=None,
    )
    pages = CONTEXT // PAGE
    cache = _Cache(max(ATTN_BATCHES) * pages + 1)
    table = torch.arange(
        1, max(ATTN_BATCHES) * pages + 1, device="cuda", dtype=torch.int32
    ).view(max(ATTN_BATCHES), pages)
    generator = torch.Generator(device="cuda").manual_seed(4)
    q = torch.randn(
        max(ATTN_BATCHES) * 8,
        HEADS,
        HEAD_DIM,
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )
    lengths = torch.full((bs,), CONTEXT, dtype=torch.int32, device="cuda")
    backend.refresh_decode_metadata(bs, bs, lengths, table[:bs])
    return backend.forward_decode(q[: bs * 8], None, None, layer, None, cache, bs)


def test_the_backend_decodes_a_draft_block_the_same_alone_and_batched(flag) -> None:
    """The draft's attention backend selects the batch-invariant leaf under the
    flag, so request 0's block rows are the same whatever rides with them. A
    backend that left the flag out of its selection would take the adaptive
    leaf, whose key split follows the batch."""
    alone = _block_decode(1)
    for bs in ATTN_BATCHES[1:]:
        assert torch.equal(_block_decode(bs)[: alone.shape[0]], alone), bs


def test_without_the_flag_the_draft_keeps_the_adaptive_leaf(flag) -> None:
    flag(False)
    assert not _backend(is_draft=True).batch_invariant


def test_the_flag_reaches_the_draft_backend_only(flag) -> None:
    """The target's backend keeps the adaptive leaf: the flag is the draft's."""
    assert _backend(is_draft=True).batch_invariant
    assert not _backend(is_draft=False).batch_invariant


def test_a_batch_invariant_draft_builds(flag) -> None:
    """The structure the flag supports: a GQA draft whose layers all slide."""
    assert _model().layers


@pytest.mark.parametrize(
    "change,reason",
    [
        (dict(layer_types=["full_attention", "sliding_attention"]), "layer_types"),
        (dict(layer_types=None), "layer_types"),
        (dict(attention_method="mla"), "mla=True"),
    ],
    ids=["a full-attention layer", "no layer types", "mla"],
)
def test_another_draft_structure_stops_the_launch(flag, change, reason) -> None:
    config = _config()
    for name, value in change.items():
        setattr(config, name, value)
    with pytest.raises(ValueError, match=reason):
        _build(config)
    # Without the flag the same draft builds: the limit belongs to the flag.
    flag(False)
    _build(config)


def test_a_quantized_draft_stops_the_launch(flag) -> None:
    with pytest.raises(ValueError, match="quantized=True"):
        _build(quant_config=object())


def test_a_subclass_keeping_the_layer_class_builds(flag) -> None:
    """A deployment's own subclass of the draft model, which builds the same
    GQA layers, is covered: the limit is on the layers, not on the class."""

    class Subclass(DFlashDraftModel):
        pass

    with set_default_torch_dtype(torch.bfloat16), torch.device("cuda"):
        assert Subclass(_config(), Mapping(rank=0, world_size=1)).layers


def test_another_layer_class_stops_the_launch(flag) -> None:
    """DFlash2 builds its own layer class, which the flag does not cover."""

    class OtherLayer(DFlashDecoderLayer):
        pass

    class Other(DFlashDraftModel):
        decoder_layer_cls = OtherLayer

    with pytest.raises(ValueError, match="OtherLayer"):
        with set_default_torch_dtype(torch.bfloat16), torch.device("cuda"):
            Other(_config(), Mapping(rank=0, world_size=1))


# ---------------------------------------------------------------------------
# Block proposal
# ---------------------------------------------------------------------------

SAMPLER_HIDDEN = 512
VOCAB = 32768
N = 8  # the draft's verify width: an anchor plus seven sampled rows
STEPS = N - 1
POOL = 72
MAX_BS = 64
# Requests sharing one round. Row 0 is the request under test.
BATCHES = (1, 2, 4, 8, 16, 32, 64)
SEED = 1234


def _sampling_drafter():
    """A DFlash holding the state its block proposal runs on."""
    torch.manual_seed(0)
    states = RuntimeStates(
        req_pool_size=POOL,
        input_vocab_size=VOCAB,
        vocab_size=VOCAB,
        output_length=N,
        device="cuda",
    )
    states.init_draft_probs(spec_num_tokens=N, reject_threshold=1.0)
    buffers = InputBuffers(
        max_bs=MAX_BS,
        max_num_tokens=MAX_BS * N,
        state_write_padding_pool_index=0,
        device="cuda",
    )
    pools = SpeculativeSamplingPools(
        # Temperature 1 with top_k disabled: every row samples, so the
        # proposal is the stochastic path rather than a hidden argmax.
        temperature=torch.ones(POOL + 1, dtype=torch.float32, device="cuda"),
        top_k=torch.zeros(POOL + 1, dtype=torch.int32, device="cuda"),
        top_p=torch.ones(POOL + 1, dtype=torch.float32, device="cuda"),
        # A distinct seed per slot, so a row that reads the wrong slot draws
        # visibly different tokens.
        seed=(
            SEED + torch.arange(POOL + 1, dtype=torch.int64, device="cuda")
        ).contiguous(),
    )
    sampler = DraftProposalSampler(
        pools=pools,
        runtime_states=states,
        input_buffers=buffers,
        spec_num_tokens=N,
        sampling_stream="per-request",
        vocab_map=None,
        device="cuda",
    )
    with set_default_torch_dtype(torch.bfloat16), torch.device("cuda"):
        head = ParallelLMHead(VOCAB, SAMPLER_HIDDEN, bias=False)
    head.weight.data.normal_(0, 0.02)
    head.quant_method.process_weights_after_loading(head)

    drafter = DFlash.__new__(DFlash)
    drafter.hidden_size = SAMPLER_HIDDEN
    drafter.spec_num_tokens = N
    drafter.lm_head = head
    drafter.logits_processor = LogitsProcessor(
        PretrainedConfig(vocab_size=VOCAB),
        dp_lm_head_tp=False,
        attn_tp_size=1,
    )
    drafter.draft_sampler = sampler
    return drafter, states, sampler, buffers


def _slots(bs: int) -> torch.Tensor:
    """Pool slots of a round: row 0 keeps slot 1 whatever rides with it."""
    return torch.arange(1, bs + 1, dtype=torch.int64, device="cuda")


def _propose(drafter, states, sampler, buffers, hidden, bs, *, frontier=64):
    """One round's block proposal for ``bs`` requests.

    Returns row 0's drafted tokens and the ``q`` rows recorded for its slot.
    """
    slots = _slots(bs)
    buffers.req_pool_indices_buf[:bs].copy_(slots)
    buffers.state_write_req_pool_indices_buf[:bs].copy_(slots)
    # Every request sits at the same frontier, so a row that read a peer's
    # offset would still be caught by its seed.
    buffers.seq_lens_buf[:bs].fill_(frontier)
    states.valid_cache_lengths[slots] = frontier
    sampler.prepare_round(
        torch.zeros(bs, dtype=torch.int32, device="cuda"), num_extends=0
    )
    tokens = torch.zeros(bs, N, dtype=torch.int32, device="cuda")
    drafter._sample_block_from_q(hidden[:bs], sampler, tokens)
    return tokens[0].clone(), states.draft_probs[int(slots[0])].clone()


def _hidden(rows: int) -> torch.Tensor:
    """Generated once at the largest batch and sliced: CUDA ``randn`` values
    depend on the tensor's element count, so generating per batch size would
    change row 0's input along with its co-batch."""
    generator = torch.Generator(device="cuda").manual_seed(7)
    return torch.randn(
        rows,
        N,
        SAMPLER_HIDDEN,
        generator=generator,
        device="cuda",
        dtype=torch.bfloat16,
    )


@pytest.fixture
def rounds(flag):
    """Row 0's proposal at each batch size, from one set of hidden states."""
    drafter, states, sampler, buffers = _sampling_drafter()
    hidden = _hidden(max(BATCHES))
    return {
        bs: _propose(drafter, states, sampler, buffers, hidden, bs) for bs in BATCHES
    }


def test_the_drafted_tokens_do_not_move_with_the_batch(rounds) -> None:
    """The tokens a request drafts are the same alone and in a batch of 64."""
    expected, _ = rounds[1]
    for bs in BATCHES[1:]:
        tokens, _ = rounds[bs]
        assert torch.equal(tokens, expected), bs


def test_the_recorded_q_does_not_move_with_the_batch(rounds) -> None:
    """The verifier reads ``q`` bit for bit, so it must not move either."""
    _, expected = rounds[1]
    for bs in BATCHES[1:]:
        _, recorded = rounds[bs]
        assert torch.equal(recorded, expected), bs


def test_every_step_drafted_a_token_from_its_own_distribution(rounds) -> None:
    """The comparisons above are not passing on empty rows: each of the seven
    sampled columns holds a token whose recorded ``q`` is a distribution, and
    the tokens are not all the same column. A step that proposes under another
    step's index draws from the wrong one."""
    tokens, recorded = rounds[1]
    drafted = tokens[1:]
    assert drafted.min() >= 0 and drafted.max() < VOCAB
    assert len(set(drafted.tolist())) > 1
    rows = recorded[:STEPS]
    assert torch.allclose(rows.sum(dim=-1), torch.ones(STEPS, device="cuda"), atol=1e-3)
    # Sampled rows, not a one-hot greedy record.
    assert rows.max() < 1.0


def test_a_different_seed_drafts_different_tokens(flag) -> None:
    """What the invariance above is worth: the seed is what fixes the draw, so
    a request with another seed draws another trajectory. Without this, rows
    could match for want of any randomness at all."""
    drafter, states, sampler, buffers = _sampling_drafter()
    hidden = _hidden(1)
    first, _ = _propose(drafter, states, sampler, buffers, hidden, 1)
    sampler._pools.seed.fill_(SEED + 9999)
    second, _ = _propose(drafter, states, sampler, buffers, hidden, 1)
    assert not torch.equal(first, second)


def test_the_same_round_drafts_the_same_tokens_twice(flag) -> None:
    """Run invariance: the same request at the same frontier redraws its own
    trajectory, which is what lets a replay compare against a recorded run."""
    drafter, states, sampler, buffers = _sampling_drafter()
    hidden = _hidden(4)
    first, first_q = _propose(drafter, states, sampler, buffers, hidden, 4)
    second, second_q = _propose(drafter, states, sampler, buffers, hidden, 4)
    assert torch.equal(first, second)
    assert torch.equal(first_q, second_q)


def test_a_later_frontier_drafts_a_different_round(flag) -> None:
    """The draw is keyed by the committed frontier, so the next round of the
    same request is a fresh draw rather than a repeat of this one."""
    drafter, states, sampler, buffers = _sampling_drafter()
    hidden = _hidden(1)
    here, _ = _propose(drafter, states, sampler, buffers, hidden, 1, frontier=64)
    later, _ = _propose(drafter, states, sampler, buffers, hidden, 1, frontier=65)
    assert not torch.equal(here, later)
