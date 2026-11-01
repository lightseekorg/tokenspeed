"""Contracts of the multicast down projection that hold without a fabric."""

import contextlib
import os
import sys
from unittest import mock

import pytest
import tokenspeed_kernel.ops.moe._cute_dsl.latent_down as latent_down
import torch
from tokenspeed_kernel.platform import current_platform

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=10, suite="runtime-1gpu")


@pytest.fixture(autouse=True)
def _clean_class_state():
    """Both caches live on the class, so one test can answer for the next."""
    for cache in (
        latent_down.KimiK3LatentDownOp._verdicts,
        latent_down.KimiK3LatentDownOp._pools,
        latent_down.KimiK3LatentDownOp._ceilings,
        latent_down.KimiK3LatentDownOp._reasons,
        latent_down._DECLINED,
    ):
        cache.clear()
    yield
    for cache in (
        latent_down.KimiK3LatentDownOp._verdicts,
        latent_down.KimiK3LatentDownOp._pools,
        latent_down.KimiK3LatentDownOp._ceilings,
        latent_down.KimiK3LatentDownOp._reasons,
        latent_down._DECLINED,
    ):
        cache.clear()


def _stub_slot():
    """A slot object for tests that mock the build to avoid allocating one.

    Returning None from a mocked ``_build_slot`` now means the rendezvous gave
    no multicast address, which raises; these tests only want to skip the
    allocation, so they have to hand back something slot-shaped.
    """
    return latent_down._MailboxSlot(
        torch.zeros(1, 8, 64, dtype=torch.bfloat16), _rendezvous_stub(), 0, {}, {}
    )


def _rendezvous_stub():
    """Stand-in for the symmetric-memory handle the slot has to outlive."""
    return mock.Mock(name="symm_mem_handle")


@contextlib.contextmanager
def _eligible(
    *,
    reachable: bool = True,
    initialized: bool = True,
    nvidia: bool = True,
    peer_ceiling: int | None = None,
):
    """Hold every eligibility term true but the one under test.

    ``peer_ceiling`` makes the last peer report that many mailbox rows instead
    of this rank's, which is the disagreement the ceiling vote exists to catch.
    """

    def gather_ceilings(output, value, group=None):
        output.fill_(int(value.item()))
        if peer_ceiling is not None:
            output[-1] = peer_ceiling

    with (
        mock.patch.object(
            latent_down,
            "multicast_backend_unavailable_reason",
            return_value=None if reachable else "fabric unreachable",
        ),
        mock.patch.object(
            latent_down,
            "current_platform",
            return_value=mock.Mock(is_nvidia=nvidia, arch_version=mock.Mock(major=10)),
        ),
        mock.patch.object(latent_down.dist, "is_initialized", return_value=initialized),
        mock.patch.object(
            latent_down.dist, "all_gather_single", side_effect=gather_ceilings
        ),
        # Without this the vote reaches the real collective with a Mock group,
        # which no-ops: the tensor keeps its local value and the test passes
        # having exercised nothing. Tests that assert on the vote patch it
        # again inside this, and the inner patch wins.
        mock.patch.object(latent_down.dist, "all_reduce", side_effect=lambda t, **k: t),
    ):
        yield


def _library_gemm(x, weight, out):
    return x @ weight.t()


def _passing_vote(tensor, **kwargs):
    """A vote nothing refuses: the tensor keeps the value this rank cast."""
    return tensor


@contextlib.contextmanager
def _building(*, rank: int, tp_size: int, multicast_ptr: int):
    """Stand in for the rendezvous and the kernels ``_build_slot`` compiles.

    Yields the record of what the body asked for: the mailbox it allocated, the
    tensor and group it rendezvoused, the GEMMs by static width, the gathers,
    and the multicast views the wide producer took.
    """
    if not current_platform().is_nvidia:
        pytest.skip("the slot builders import NVIDIA CuTe DSL kernels")
    made: dict = {
        "gemms": {},
        "gathers": [],
        "views": [],
        "multicast_ptr": multicast_ptr,
    }

    def empty(shape, dtype, device):
        return made.setdefault("mailbox", torch.zeros(shape, dtype=dtype))

    def rendezvous(mailbox, group):
        made["rendezvoused"] = mailbox
        made["group"] = group
        return mock.Mock(multicast_ptr=multicast_ptr)

    with (
        mock.patch("torch.distributed._symmetric_memory.empty", empty),
        mock.patch("torch.distributed._symmetric_memory.rendezvous", rendezvous),
        mock.patch.object(latent_down.dist, "get_rank", side_effect=lambda group: rank),
        mock.patch.object(
            latent_down.dist, "get_world_size", side_effect=lambda group: tp_size
        ),
        mock.patch.object(latent_down.dist, "all_reduce"),
        mock.patch(
            "tokenspeed_kernel.thirdparty.cute_dsl.latent_moe_tail"
            ".fused_multicast_latent_down_gemm.FusedMulticastLatentDownGemmKernel",
            side_effect=lambda **kw: made["gemms"].setdefault(kw["num_rows"], kw),
        ),
        mock.patch(
            "tokenspeed_kernel.thirdparty.cute_dsl.latent_moe_tail"
            ".lamport_copy.LamportCopyKernel",
            side_effect=lambda **kw: made["gathers"].append(kw) or kw,
        ),
        mock.patch.object(
            latent_down,
            "bf16_tensor_on_pointer",
            side_effect=lambda *a: made["views"].append(a) or torch.zeros(1),
        ),
    ):
        yield made


def test_pool_slot_alternates_over_layers() -> None:
    """Consecutive layers must not land on the same mailbox."""
    slots = [latent_down._pool_slot(i) for i in range(6)]
    assert slots == [0, 1, 0, 1, 0, 1]
    assert len(set(slots)) == latent_down._DOWN_POOL_DEPTH


def test_arm_mailbox_writes_the_sentinel_the_gather_spins_on() -> None:
    """This mailbox's sentinel is -0 in both BF16 lanes, not the tail's (+0, -0).

    A producer that cannot be made to sanitize -0 away spells (+0, -0) with one
    coincidence, because the +0 half is a value real results carry all the time;
    spelling (-0, -0) takes two independent ones.
    """
    mailbox = torch.zeros(1, 4, 64, dtype=torch.bfloat16)
    latent_down.arm_mailbox(mailbox, latent_down._DOWN_SENTINEL)
    words = mailbox.view(torch.int32)
    assert torch.equal(words, torch.full_like(words, -0x7FFF8000))
    pairs = mailbox.view(torch.int16).view(-1, 2)
    assert torch.equal(pairs[:, 0], torch.full_like(pairs[:, 0], -0x8000))
    assert torch.equal(pairs[:, 1], torch.full_like(pairs[:, 1], -0x8000))


@pytest.mark.parametrize(
    ("hidden", "latent", "tp", "expected"),
    [
        (7168, 3584, 8, True),
        (7168, 3584, 1, False),
        (7168, 3584, 3, False),
        (7168, 3584, 256, False),
        (4096, 3584, 8, False),
        # 3585 is indivisible by the group but its block would still be by 8.
        (7168, 3585, 8, False),
        # Divisible by the group, but the block itself is 28 -- not a multiple of 8.
        (7168, 3584, 128, False),
    ],
)
def test_availability_needs_a_divisible_multi_rank_group(
    hidden, latent, tp, expected
) -> None:
    """Every term of the eligibility test has to be visible on its own."""
    with _eligible():
        assert (
            latent_down.KimiK3LatentDownOp.available(
                hidden, latent, tp, 92, _voting_group()
            )
            is expected
        )


def test_the_fabric_is_probed_for_the_group_that_will_rendezvous() -> None:
    """A world spanning hosts cannot answer for an intra-host subgroup."""
    group = _voting_group()
    asked = []
    with (
        _eligible(),
        mock.patch.object(
            latent_down,
            "multicast_backend_unavailable_reason",
            side_effect=lambda g: asked.append(g) or None,
        ),
    ):
        latent_down.KimiK3LatentDownOp.available(7168, 3584, 8, 92, group)

    assert asked == [group]


@pytest.mark.parametrize("term", ["reachable", "nvidia", "initialized"])
def test_availability_needs_every_eligibility_term(term: str) -> None:
    """Each term stands alone: another one would refuse the same shapes anyway."""
    with _eligible(**{term: False}):
        assert (
            latent_down.KimiK3LatentDownOp.available(7168, 3584, 8, 92, _voting_group())
            is False
        )


def test_initialize_remembers_that_a_slot_could_not_be_built() -> None:
    """A failed build must not rendezvous again on the next layer.

    It now raises rather than returning None -- a rendezvous that yields no
    multicast address is the fabric failing a group that expected it -- so what
    this pins is that the remembered failure is re-raised without rebuilding.
    """
    group = mock.Mock(group_name="g")
    calls: list = []
    with _eligible(), _voting_ranks(calls, _passing_vote, lambda: None):
        for layer in range(8):
            with pytest.raises(
                RuntimeError, match="rendezvous returned no multicast address"
            ):
                _initialize(group=group, block_index=layer)
    assert len(calls) == latent_down._DOWN_POOL_DEPTH
    # The widths the op was initialized with are the widths the slot is built
    # for; transposing them compiles kernels that fail on the first batch.
    assert calls[0] == (group, 7168, 3584, torch.device("cpu"), 8, None)


def test_pool_key_separates_process_groups() -> None:
    """Two groups of the same size must not share a rendezvoused mailbox."""
    built: list = []
    with _eligible(), _voting_ranks(built, _passing_vote, _stub_slot):
        for name in ("base", "draft"):
            _initialize(group=mock.Mock(group_name=name))
    assert [call[0].group_name for call in built] == ["base", "draft"]


@pytest.mark.parametrize("ceiling", [8, 9, 128, 512])
def test_the_width_gate_is_the_ceiling_it_was_built_with(ceiling: int) -> None:
    """Everything past the claimed width belongs to the replicated projection.

    The bound is per op now, not a module constant, so an op built for a wide
    decode ladder must claim the widths past the fused kernel's ceiling and an
    op built narrow must not.
    """
    op = latent_down.KimiK3LatentDownOp(
        latent_down._MailboxSlot(torch.zeros(1), _rendezvous_stub(), 0, {}, {}),
        448,
        0,
        ceiling,
    )
    # The lower bound is the shape the multicast split wins by the most, so it
    # needs a witness of its own rather than riding on the upper one.
    assert op.handles(1)
    assert op.handles(latent_down._FUSED_MAX_M)
    assert op.handles(ceiling)
    assert not op.handles(ceiling + 1)
    assert not op.handles(0)


def _require(condition: bool, what: str) -> None:
    """Fail like the real kernel does, so the stub cannot hide a caller bug."""
    if not condition:
        raise AssertionError(f"kernel contract violated: {what}")


def _stub_op(rank: int, shard_dim: int, latent: int, seen: dict, max_m: int = 8):
    """An op whose kernels are stubs, so the call's own arithmetic is visible."""

    def gather(mailbox, m):
        _require(mailbox.shape[1] == max_m, "full capacity")
        seen.update(mailbox=mailbox)
        # The real gather allocates its output, then releases its dependents
        # and only afterwards re-arms the rows it read.
        return mailbox[:, :m, :].clone()

    slot = latent_down._MailboxSlot(
        mailbox=torch.zeros(1, max_m, latent, dtype=torch.bfloat16),
        handle=_rendezvous_stub(),
        multicast_ptr=1234,
        gemm_by_m={
            # The real kernel is compiled for a static M and checks it, and the
            # real gather refuses a mailbox short of its full capacity.
            m: (
                lambda h, w, out, ptr, m=m: (
                    seen.update(weight=w, tokens=h.shape[0], ptr=ptr, out=out, m=m),
                    _require(h.shape[0] == m, "static M"),
                )
            )
            for m in range(1, max_m + 1)
        },
        gather_by_m={m: gather for m in range(1, max_m + 1)},
    )
    return latent_down.KimiK3LatentDownOp(slot, shard_dim, rank, max_m)


@pytest.mark.parametrize("rank", [0, 3, 7])
def test_call_publishes_the_block_it_was_handed(rank: int) -> None:
    """The op takes this rank's rows; it no longer carves them itself.

    The caller slices, because a projection that narrowed its storage has no
    full width left to slice from -- so the op must pass the weight through
    untouched rather than indexing into it by rank.
    """
    hidden_size, latent, tp = 7168, 3584, 8
    shard_dim = latent // tp
    seen: dict = {}
    op = _stub_op(rank, shard_dim, latent, seen)
    block_in = torch.zeros(shard_dim, hidden_size, dtype=torch.bfloat16)
    out = op(
        torch.zeros(4, hidden_size, dtype=torch.bfloat16), block_in, output_scale=None
    )
    assert out.shape == (4, latent)
    assert (
        out.untyped_storage().data_ptr() != seen["mailbox"].untyped_storage().data_ptr()
    )
    assert seen["tokens"] == 4
    assert seen["weight"] is block_in
    assert seen["ptr"] == 1234
    # The kernel's capacity guard bounds a raw-pointer write, so it has to see
    # the whole mailbox; a slice sized to the batch makes it always hold.
    assert seen["out"].shape[1] == 8
    assert seen["out"].untyped_storage().data_ptr() == (
        seen["mailbox"].untyped_storage().data_ptr()
    )


def test_pool_key_separates_model_scopes() -> None:
    """A draft model must not inherit the base model's mailbox."""
    built: list = []
    group = mock.Mock(group_name="g")
    with _eligible(), _voting_ranks(built, _passing_vote, _stub_slot):
        for scope in ("model.layers", "model.decoder"):
            _initialize(group=group, model_scope=scope)
    assert len(built) == 2


def _voting_group(name="g"):
    """A group stub whose all_reduce we can drive rank by rank."""
    return mock.Mock(group_name=name)


@pytest.mark.parametrize("peer_says", [True, False])
def test_eligibility_is_agreed_across_the_group(peer_says: bool) -> None:
    """A rank must not rendezvous on a verdict its peers do not share."""
    built: list = []

    def min_vote(tensor, op=None, group=None):
        tensor.fill_(min(int(tensor.item()), int(peer_says)))

    with _eligible(), _voting_ranks(built, min_vote, _stub_slot):
        _initialize(device=torch.device("cuda"))
    assert bool(built) is peer_says


def test_the_group_is_polled_once_not_once_per_layer() -> None:
    """92 layers must not cast 92 votes."""
    votes, built = [], []
    with _eligible(), _voting_ranks(built, lambda t, **k: votes.append(1), _stub_slot):
        group = _voting_group()
        for layer in range(8):
            _initialize(group=group, block_index=layer, device=torch.device("cuda"))
    assert len(votes) == 1


@pytest.mark.parametrize("layers", [1, 2, 3, 4, 23, 31, 46, 92])
def test_availability_needs_a_whole_number_of_rotations(layers: int) -> None:
    """A stage's blocks must wrap onto a different slot, not the same one."""
    with _eligible():
        available = latent_down.KimiK3LatentDownOp.available(
            7168, 3584, 8, layers, _voting_group()
        )
    depth = latent_down._DOWN_POOL_DEPTH
    assert available is (layers >= depth and layers % depth == 0)
    if not available:
        # PP3 and PP4 give 31 and 23 local blocks; both wrap onto themselves.
        slots = [latent_down._pool_slot(b) for b in range(layers)]
        assert slots[-1] == slots[0], "this is the case the gate must refuse"


def test_the_vote_takes_the_minimum_so_one_refusal_is_decisive() -> None:
    """A MAX vote would let one willing rank drag the others into a rendezvous."""
    ops, built = [], []
    with (
        _eligible(),
        _voting_ranks(built, lambda t, op=None, group=None: ops.append(op), _stub_slot),
    ):
        _initialize(device=torch.device("cuda"))
    assert ops == [latent_down.dist.ReduceOp.MIN]


def test_a_pooled_slot_is_re_armed_before_it_is_handed_out() -> None:
    """A reload must not inherit whatever an abandoned round left behind."""
    mailbox = torch.zeros(1, latent_down._FUSED_MAX_M, 64, dtype=torch.bfloat16)
    slot = latent_down._MailboxSlot(mailbox, _rendezvous_stub(), 1, {}, {})
    built: list = []
    with _eligible(), _voting_ranks(built, _passing_vote, lambda: slot):
        _initialize()
        mailbox.fill_(7.0)
        _initialize()
    words = mailbox.view(torch.int32)
    assert torch.equal(words, torch.full_like(words, -0x7FFF8000))


@pytest.mark.parametrize("tokens", [1, 4, 8, 9, 64])
def test_call_selects_the_kernel_compiled_for_this_width(tokens: int) -> None:
    """A batch must reach the kernel built for it and gather exactly its rows."""
    hidden_size, latent, tp = 7168, 3584, 8
    seen: dict = {}
    op = _stub_op(0, latent // tp, latent, seen, max_m=64)
    out = op(
        torch.zeros(tokens, hidden_size, dtype=torch.bfloat16),
        torch.zeros(latent, hidden_size, dtype=torch.bfloat16),
        output_scale=None,
    )
    assert out.shape == (tokens, latent)
    assert seen["tokens"] == tokens


def test_build_slot_arms_and_binds_what_the_kernels_need() -> None:
    """The rendezvous body is where a mistake corrupts a peer's heap."""
    hidden_size, latent, tp = 7168, 3584, 8
    group = _voting_group()
    with _building(rank=3, tp_size=tp, multicast_ptr=4096) as made:
        slot = latent_down.KimiK3LatentDownOp._build_slot(
            group,
            hidden_size,
            latent,
            torch.device("cpu"),
            latent_down._FUSED_MAX_M,
            None,
        )
    assert slot is not None
    assert slot.mailbox.shape == (1, latent_down._FUSED_MAX_M, latent)
    words = slot.mailbox.view(torch.int32)
    assert torch.equal(words, torch.full_like(words, -0x7FFF8000))
    assert made["group"] is group
    assert sorted(made["gemms"]) == list(range(1, latent_down._FUSED_MAX_M + 1))
    for kw in made["gemms"].values():
        assert (kw["rank"], kw["tp_size"]) == (3, tp)
        assert (kw["in_dim"], kw["latent_dim"]) == (hidden_size, latent)
    gather = made["gathers"][0]
    assert gather["hidden_dim"] == latent
    assert gather["max_m"] == latent_down._FUSED_MAX_M
    # Pin that the tuned launch geometry reaches the kernel, not its value.
    assert gather["ctas"] == latent_down._LAMPORT_CTAS
    assert gather["threads"] == latent_down._LAMPORT_THREADS
    # The gather must spin on the word the arming above wrote, not the default
    # one every other Lamport buffer keeps.
    assert gather["sentinel"] == latent_down._DOWN_SENTINEL == 0x80008000
    assert words[0, 0, 0].item() == -0x7FFF8000
    # The address the rendezvous returned is the one every peer is published
    # through; binding anything else stores through a pointer nobody owns.
    assert slot.multicast_ptr == made["multicast_ptr"]
    # The address bound above belongs to the tensor that was rendezvoused, not
    # to some other buffer the peers know nothing about.
    assert made["rendezvoused"] is slot.mailbox
    # A ceiling at the fused width claims nothing past it, so no wide producer
    # is built and no multicast view is taken.
    assert made["views"] == []
    assert sorted(slot.gemm_by_m) == list(range(1, latent_down._FUSED_MAX_M + 1))


@pytest.mark.parametrize("peer_has_pointer", [True, False])
def test_the_pointer_is_agreed_after_the_rendezvous(peer_has_pointer: bool) -> None:
    """Multicast support is per rank, so a live pointer here is not enough."""
    seen_ops = []

    def min_vote(tensor, op=None, group=None):
        seen_ops.append(op)
        tensor.fill_(min(int(tensor.item()), int(peer_has_pointer)))

    with mock.patch.object(latent_down.dist, "all_reduce", side_effect=min_vote):
        agreed = latent_down.KimiK3LatentDownOp._agree_on_pointer(
            _voting_group(), 4096, torch.device("cpu")
        )
    assert agreed is peer_has_pointer
    assert seen_ops == [latent_down.dist.ReduceOp.MIN]


def test_a_null_pointer_here_refuses_regardless_of_peers() -> None:
    """This rank having no address is decisive on its own."""
    with mock.patch.object(
        latent_down.dist, "all_reduce", side_effect=lambda t, **k: None
    ):
        assert not latent_down.KimiK3LatentDownOp._agree_on_pointer(
            _voting_group(), 0, torch.device("cpu")
        )


def test_build_slot_declines_when_the_pointer_vote_fails() -> None:
    """A refused pointer must leave no slot behind."""
    with (
        _building(rank=0, tp_size=8, multicast_ptr=4096),
        mock.patch.object(
            latent_down.KimiK3LatentDownOp, "_agree_on_pointer", return_value=False
        ),
    ):
        slot = latent_down.KimiK3LatentDownOp._build_slot(
            _voting_group(), 7168, 3584, torch.device("cpu"), 8, None
        )
    assert slot is None


def test_the_verdict_does_not_answer_for_another_rotation() -> None:
    """A second model on the same group has its own block count."""
    group = mock.Mock(group_name="g")
    votes, built = [], []
    # Record what is cast, not that a cast happened: the eligibility answer
    # reaches production only through this tensor.
    with (
        _eligible(),
        _voting_ranks(built, lambda t, **k: votes.append(int(t.item())), _stub_slot),
    ):
        _initialize(group=group, model_scope="base")
        # 23 blocks is not a whole number of rotations, so this one is refused
        # -- but only if the remembered verdict was not asked the wrong question.
        assert _initialize(group=group, layer_count=23, model_scope="draft") is None
    # The second rotation is refused because a 0 was cast, not because the
    # build failed: without this the guards `available` pins go unconsulted.
    assert votes == [1, 0]


def test_the_verdict_is_per_group() -> None:
    """Two groups of one shape still each need their own vote."""
    votes, built = [], []
    with _eligible(), _voting_ranks(built, lambda t, **k: votes.append(1), _stub_slot):
        for name in ("g", "h"):
            _initialize(group=mock.Mock(group_name=name))
    assert len(votes) == 2


def test_the_vote_asks_about_the_widths_in_the_right_order() -> None:
    """A latent narrower than the k-tile is eligible; a hidden that narrow is not."""
    votes, built = [], []
    with (
        _eligible(),
        _voting_ranks(built, lambda t, **k: votes.append(int(t.item())), _stub_slot),
    ):
        _initialize(group=mock.Mock(group_name="g"), latent_size=1792)
    assert votes == [1]


def test_initialize_hands_back_an_op_bound_to_its_slot_and_rank() -> None:
    """The refusing half has nine witnesses; this is the succeeding one.

    Without it, returning None on success -- the whole multicast shard disabled
    everywhere -- is indistinguishable from the shard working, because refusing
    is a supported answer the projection accepts in silence.
    """
    group = mock.Mock(group_name="g")
    slot = mock.Mock(name="slot")
    with (
        _eligible(),
        mock.patch.object(latent_down.dist, "get_rank", side_effect=lambda group: 5),
        mock.patch.object(
            latent_down.dist, "get_world_size", side_effect=lambda group: 8
        ),
        mock.patch.object(latent_down.dist, "all_reduce", lambda t, **k: None),
        mock.patch.object(
            latent_down.KimiK3LatentDownOp, "_build_slot", side_effect=lambda *a: slot
        ),
        mock.patch.object(latent_down, "arm_mailbox", lambda mailbox, sentinel: None),
    ):
        op = _initialize(group=group, model_scope="base")

    assert op is not None
    assert op._slot is slot
    assert op.rank == 5
    assert op.shard_dim == 3584 // 8


def test_build_slot_votes_about_the_address_it_actually_got() -> None:
    """A rank whose rendezvous yields nothing must not vote as if it had."""
    group = _voting_group()
    # _agree_on_pointer stays unpatched: it must be asked about the real address.
    with _building(rank=0, tp_size=8, multicast_ptr=0):
        slot = latent_down.KimiK3LatentDownOp._build_slot(
            group, 7168, 3584, torch.device("cpu"), 8, None
        )

    assert slot is None


def test_build_slot_sizes_everything_to_the_ceiling_it_is_given() -> None:
    """The mailbox, the gather and the dispatch must agree on one width."""
    hidden_size, latent, tp, ceiling = 7168, 3584, 8, 32
    group = _voting_group()
    with _building(rank=3, tp_size=tp, multicast_ptr=4096) as made:
        slot = latent_down.KimiK3LatentDownOp._build_slot(
            group, hidden_size, latent, torch.device("cpu"), ceiling, None
        )
    assert slot is not None
    assert slot.mailbox.shape == (1, ceiling, latent)
    # The fused kernel compiles one static M and has none past its ceiling, so
    # the widths above it are a different producer behind the same dispatch.
    assert sorted(made["gemms"]) == list(range(1, latent_down._FUSED_MAX_M + 1))
    assert sorted(slot.gemm_by_m) == list(range(1, ceiling + 1))
    wide = {
        id(slot.gemm_by_m[m]) for m in range(latent_down._FUSED_MAX_M + 1, ceiling + 1)
    }
    assert len(wide) == 1
    producer = slot.gemm_by_m[ceiling]
    assert isinstance(producer, latent_down._MulticastVaGemm)
    assert producer is not slot.gemm_by_m[latent_down._FUSED_MAX_M]
    # Every width must have a gather, and one geometry must not build many.
    assert sorted(slot.gather_by_m) == list(range(1, ceiling + 1))
    assert len(made["gathers"]) == 1
    assert made["gathers"][0]["max_m"] == ceiling
    assert len({id(g) for g in slot.gather_by_m.values()}) == 1
    # The view is this rank's column block of the mailbox, addressed through
    # the multicast pointer: a wrong base publishes over a peer's columns.
    assert len(made["views"]) == 1
    pointer, shape, strides, _ = made["views"][0]
    assert pointer == made["multicast_ptr"] + 3 * (latent // tp) * 2
    assert shape == (ceiling, latent // tp)
    assert strides == (latent, 1)


def test_the_ceiling_never_falls_below_the_fused_kernel_s_width() -> None:
    """Those widths are compiled either way; claiming fewer only loses them."""
    built: list = []
    with _eligible(), _voting_ranks(built, _passing_vote, _stub_slot):
        _initialize(max_m=1)
    assert built[0][4] == latent_down._FUSED_MAX_M


def test_the_pool_key_separates_two_ceilings() -> None:
    """A wider op must not inherit a mailbox that is too short for it."""
    built: list = []
    group = mock.Mock(group_name="g")
    with _eligible(), _voting_ranks(built, _passing_vote, _stub_slot):
        for ceiling in (8, 64):
            _initialize(group=group, max_m=ceiling)
    assert [call[4] for call in built] == [8, 64]


def _wide_producer_over(backing: torch.Tensor, seen: dict, **kwargs):
    """A wide producer whose multicast view is a tensor the test can read."""

    def fake_view(pointer, shape, strides, device_index):
        seen.update(pointer=pointer, shape=shape, strides=strides, device=device_index)
        return backing

    with mock.patch.object(latent_down, "bf16_tensor_on_pointer", fake_view):
        return latent_down._wide_producer(**kwargs)


def test_the_wide_producer_publishes_through_the_multicast_view() -> None:
    """The view the GEMM stores into: this rank's columns, at the right stride.

    The pointer, shape and stride assertions are the load-bearing half -- they
    catch a block aimed at the wrong rank's columns or at a row pitch that
    would stage through a copy. That the store reaches peers is fixture here,
    not evidence; the multicast address is a plain tensor in this test.
    """
    latent, shard_dim, rank, ceiling, k = 3584, 448, 3, 32, 128
    torch.manual_seed(7)
    stale = torch.randn(ceiling, shard_dim)
    backing = stale.clone()
    seen: dict = {}
    producer = _wide_producer_over(
        backing,
        seen,
        multicast_ptr=4096,
        rank=rank,
        shard_dim=shard_dim,
        latent_size=latent,
        max_m=ceiling,
        device=torch.device("cpu"),
        publish_gemm=None,
    )
    assert seen["pointer"] == 4096 + rank * shard_dim * 2
    assert seen["shape"] == (ceiling, shard_dim)
    assert seen["strides"] == (latent, 1)
    torch.manual_seed(11)
    hidden = torch.randn(9, k)
    weight_block = torch.randn(shard_dim, k)
    mailbox = object()
    assert producer(hidden, weight_block, mailbox, 4096) is mailbox
    torch.testing.assert_close(backing[:9], hidden @ weight_block.T)
    # Only the batch's rows may be touched: the rest still hold sentinels.
    assert torch.equal(backing[9:], stale[9:])


def _initialize(**overrides):
    """Drive initialize with the shapes every rank agrees on, past the overrides."""
    kwargs = dict(
        group=_voting_group(),
        hidden_size=7168,
        latent_size=3584,
        device=torch.device("cpu"),
        block_index=0,
        layer_count=92,
        model_scope="s",
        max_m=8,
        publish_gemm=None,
    )
    kwargs.update(overrides)
    return latent_down.KimiK3LatentDownOp.initialize(**kwargs)


@contextlib.contextmanager
def _voting_ranks(built, all_reduce, make_slot):
    """Rank 0 of eight, recording _build_slot's args and handing back make_slot()."""
    with (
        mock.patch.object(latent_down.dist, "get_rank", side_effect=lambda group: 0),
        mock.patch.object(
            latent_down.dist, "get_world_size", side_effect=lambda group: 8
        ),
        mock.patch.object(latent_down.dist, "all_reduce", side_effect=all_reduce),
        mock.patch.object(
            latent_down.KimiK3LatentDownOp,
            "_build_slot",
            side_effect=lambda *a: built.append(a) or make_slot(),
        ),
    ):
        yield


def test_a_disagreed_ceiling_is_refused_before_anything_is_allocated() -> None:
    """Differently sized mailboxes hang the rendezvous with no traceback.

    So the disagreement has to be an error raised where the cause is, not a
    hang later that looks like any other boot hang on this fabric.
    """
    built: list[int] = []
    with _eligible(peer_ceiling=1024), _voting_ranks(built, _passing_vote, _stub_slot):
        with pytest.raises(ValueError, match="size the down mailbox alike"):
            _initialize(max_m=128)
    assert built == []


def test_the_disagreement_names_what_each_rank_derived() -> None:
    """A bare "mismatch" would leave the next person nothing to look at."""
    built: list[int] = []
    with _eligible(peer_ceiling=1024), _voting_ranks(built, _passing_vote, _stub_slot):
        with pytest.raises(ValueError) as excinfo:
            _initialize(max_m=128)
    message = str(excinfo.value)
    assert "rank 0 derived 128" in message
    assert "7: 1024" in message


def test_an_agreed_ceiling_is_polled_once_not_once_per_layer() -> None:
    """The vote sits on the boot path of every MoE block."""
    built: list[int] = []
    polls: list[int] = []

    def counted(output, value, group=None):
        polls.append(int(value.item()))
        output.fill_(int(value.item()))

    with _eligible(), _voting_ranks(built, _passing_vote, _stub_slot):
        with mock.patch.object(
            latent_down.dist, "all_gather_single", side_effect=counted
        ):
            for block in range(4):
                _initialize(block_index=block, max_m=128)
    assert polls == [128]
    assert len(built) == latent_down._DOWN_POOL_DEPTH


@pytest.mark.parametrize(
    "kwargs,expected",
    [
        ({"tp_size": 1}, "a single rank"),
        ({"layer_count": 91}, "not whole 2-slot rotations"),
        ({"hidden_size": 7000}, "k-tile"),
        ({"latent_size": 3580}, "does not split"),
    ],
)
def test_the_decline_reason_names_the_condition_that_failed(kwargs, expected) -> None:
    """A silent fallback is the failure mode; the reason is the whole point."""
    args = dict(
        hidden_size=7168,
        latent_size=3584,
        tp_size=8,
        layer_count=92,
        group=_voting_group(),
        fused_producer=True,
    )
    args.update(kwargs)
    with _eligible():
        reason = latent_down.KimiK3LatentDownOp._unavailable_reason(**args)
    assert reason is not None and expected in reason


def test_the_reason_predicate_still_backs_availability() -> None:
    """available() must stay exactly the negation, or the two can drift apart."""
    args = dict(
        hidden_size=7168,
        latent_size=3584,
        tp_size=8,
        layer_count=92,
        group=_voting_group(),
    )
    with _eligible():
        assert latent_down.KimiK3LatentDownOp.available(**args)
        assert (
            latent_down.KimiK3LatentDownOp._unavailable_reason(
                **args, fused_producer=True
            )
            is None
        )
    with _eligible(reachable=False):
        assert not latent_down.KimiK3LatentDownOp.available(**args)
        assert "fabric" in latent_down.KimiK3LatentDownOp._unavailable_reason(
            **args, fused_producer=True
        )


def test_two_configurations_on_one_group_keep_their_own_reasons(caplog) -> None:
    """The verdict and its reason must be cached under the same identity.

    Keyed on the group alone, a second configuration overwrites the first's
    reason, and a later decline names an unrelated condition -- which is worse
    than no reason at all, because it sends the reader somewhere true-looking.
    """
    built: list[int] = []
    with (
        _eligible(),
        _voting_ranks(built, _passing_vote, _stub_slot),
        caplog.at_level("INFO", logger=latent_down.logger.name),
    ):
        assert _initialize(latent_size=3580) is None
        assert _initialize(layer_count=91) is None
        # This third call hits the verdict cache, so it can only answer from
        # the stored reason; the once-per-reason dedup would hide that.
        latent_down._DECLINED.clear()
        assert _initialize(latent_size=3580) is None
    said = [
        r.getMessage()
        for r in caplog.records
        if "down mailbox unavailable" in r.message
    ]
    assert len(said) == 3
    assert "does not split" in said[0]
    assert "rotations" in said[1]
    assert "does not split" in said[2]


def test_a_decline_is_reported_once_not_once_per_block(caplog) -> None:
    """92 MoE blocks decline together; one line is the news, 92 is noise."""
    with caplog.at_level("INFO", logger=latent_down.logger.name):
        for _ in range(4):
            assert latent_down._decline("no process group") is None
    lines = [r for r in caplog.records if "down mailbox unavailable" in r.message]
    assert len(lines) == 1
    assert "no process group" in lines[0].getMessage()


def test_an_ungrouped_rank_says_so_rather_than_falling_back_silently(caplog) -> None:
    with mock.patch.object(latent_down.dist, "is_initialized", return_value=False):
        with caplog.at_level("INFO", logger=latent_down.logger.name):
            assert _initialize() is None
    assert any("no process group" in r.getMessage() for r in caplog.records)


def test_the_fault_is_claimed_only_on_the_validated_architecture() -> None:
    """The arch gate and the fault condition are deliberately different numbers.

    The gate admits sm100 and up so a later architecture can attempt the path.
    The fault must not follow it there: on hardware nobody here has run, a
    rendezvous that cannot deliver more likely means the path was never really
    supported, and the message names Blackwell specifics that may not apply. So
    above the validated architecture the same failure declines quietly, which is
    what the shipped gate asks for and what this pins.
    """
    for major, expect_fault in ((10, True), (11, False), (12, False)):
        with mock.patch.object(
            latent_down,
            "current_platform",
            return_value=mock.Mock(is_nvidia=True, arch_version=mock.Mock(major=major)),
        ):
            assert latent_down._rendezvous_failure_is_a_fault() is expect_fault


@pytest.mark.parametrize(
    "arch_major,publish_gemm", [(11, None), (10, lambda x, w, o: o)]
)
def test_a_rendezvous_failure_above_the_validated_arch_declines(
    caplog, arch_major: int, publish_gemm
) -> None:
    """Above the validated arch, or with a caller fallback, it declines and says why."""
    built: list = []
    with (
        _eligible(),
        # None from the build is the rendezvous yielding no multicast address.
        _voting_ranks(built, _passing_vote, lambda: None),
        mock.patch.object(
            latent_down,
            "current_platform",
            return_value=mock.Mock(
                is_nvidia=True, arch_version=mock.Mock(major=arch_major)
            ),
        ),
        caplog.at_level("INFO", logger=latent_down.logger.name),
    ):
        assert _initialize(publish_gemm=publish_gemm) is None
    assert any("no multicast address" in r.getMessage() for r in caplog.records)


def test_the_library_producer_keeps_its_ceiling_and_builds_no_fused_kernel() -> None:
    """Without the fused GEMM the ceiling is the caller's and no width is raised."""
    built: list = []
    args = dict(hidden_size=8192, latent_size=4096, layer_count=62)
    reason = latent_down.KimiK3LatentDownOp._unavailable_reason
    with _eligible(), _voting_ranks(built, _passing_vote, _stub_slot):
        _initialize(**args, max_m=4, publish_gemm=_library_gemm)
        # A width off the fused k-tile is eligible for the library producer only.
        assert "k-tile" in reason(**args, tp_size=8, group=None, fused_producer=True)
        assert reason(**args, tp_size=8, group=None, fused_producer=False) is None
    assert built[0][4:] == (4, _library_gemm)


@pytest.mark.skipif(
    not current_platform().is_nvidia, reason="requires a NVIDIA CUDA pointer view"
)
@pytest.mark.parametrize("writes_out", [False, True])
def test_the_library_producer_publishes_the_weight_it_is_handed(
    writes_out: bool,
) -> None:
    """Pooled across layers, it must never keep a weight, and may write rows itself."""
    latent, shard, hidden, max_m = 32, 8, 16, 4
    mailbox = torch.zeros(max_m, latent, dtype=torch.bfloat16, device="cuda")
    latent_down.arm_mailbox(mailbox, latent_down.HALF_POLLED_SENTINEL)
    armed = mailbox.view(torch.int16).clone()
    writes = []

    def gemm(x, weight, out):
        if not writes_out:
            return x @ weight.t()
        writes.append(out.data_ptr())
        return out.copy_(x @ weight.t())

    producer = latent_down._MulticastVaGemm(
        multicast_ptr=mailbox.data_ptr(),
        rank=1,
        shard_dim=shard,
        latent_size=latent,
        max_m=max_m,
        device=mailbox.device,
        publish_gemm=gemm,
    )
    torch.manual_seed(2)
    x = torch.randn(2, hidden, dtype=torch.bfloat16, device="cuda")
    x[1] = 0.0
    published = []
    for seed in (3, 4):
        torch.manual_seed(seed)
        weight = -torch.rand(shard, hidden, dtype=torch.bfloat16, device="cuda")
        producer(x, weight, mailbox, mailbox.data_ptr())
        torch.cuda.synchronize()
        published.append(mailbox[:2, shard : 2 * shard].clone())
        assert torch.equal(
            published[-1].view(torch.int16), (x @ weight.t()).view(torch.int16)
        )
    assert not torch.equal(published[0][0], published[1][0])
    # Every other rank's column and every row past the batch keeps the sentinel.
    words = mailbox.view(torch.int16).clone()
    words[:2, shard : 2 * shard] = armed[:2, shard : 2 * shard]
    assert torch.equal(words, armed)
    assert writes == [mailbox[0, shard:].data_ptr()] * (2 if writes_out else 0)
    for tokens in (0, max_m + 1):
        with pytest.raises(ValueError, match="do not fit"):
            producer(
                torch.zeros(tokens, hidden, dtype=torch.bfloat16, device="cuda"),
                weight,
                mailbox,
                mailbox.data_ptr(),
            )


@pytest.mark.parametrize(
    "publish_gemm,sentinel,polls",
    [
        (None, latent_down._DOWN_SENTINEL, False),
        (_library_gemm, latent_down.HALF_POLLED_SENTINEL, True),
    ],
)
def test_only_the_library_producer_asks_the_gather_to_poll_halves(
    publish_gemm, sentinel: int, polls: bool
) -> None:
    """K3's -0 slot keeps word compares; the NaN slot arms, polls halves, warms up."""
    group = mock.Mock()
    group.group_name = "halves"
    producer: dict = {}
    published: list = []
    with (
        _building(rank=1, tp_size=4, multicast_ptr=0x1000) as made,
        mock.patch.object(
            latent_down,
            "_MulticastVaGemm",
            side_effect=lambda **kw: producer.setdefault("kwargs", kw),
        ),
        mock.patch.object(
            latent_down,
            "multicast_publish",
            side_effect=lambda block, out: published.append((block, out)),
        ),
    ):
        slot = latent_down.KimiK3LatentDownOp._build_slot(
            group, 8192, 4096, torch.device("cpu"), 4, publish_gemm
        )
    assert slot is not None and slot.sentinel == sentinel
    words = slot.mailbox.view(torch.int32)
    expected = sentinel - 0x1_0000_0000 if sentinel >= 0x8000_0000 else sentinel
    assert torch.equal(words, torch.full_like(words, expected))
    assert made["gathers"][0]["sentinel"] == sentinel
    assert made["gathers"][0]["poll_halves"] is polls
    assert sorted(slot.gemm_by_m) == [1, 2, 3, 4]
    if publish_gemm is not None:
        # The library producer covers every width; no fused kernel is built.
        assert len({id(p) for p in slot.gemm_by_m.values()}) == 1
        assert producer["kwargs"]["multicast_ptr"] == 0x1000
        assert producer["kwargs"]["shard_dim"] == 1024
        assert producer["kwargs"]["max_m"] == 4
        # The publish kernel is warmed on local scratch at the mailbox's row pitch.
        [(block, destination)] = published
        assert block.shape == destination.shape == (1, 1024)
        assert destination.stride() == (4096, 1)
        end = slot.mailbox.data_ptr() + slot.mailbox.numel() * 2
        assert not slot.mailbox.data_ptr() <= destination.data_ptr() < end
    else:
        assert published == []


def test_publish_hands_out_the_arriving_rows_for_a_consumer_to_gather() -> None:
    """The consumer kernel polls the rows itself, so no gather runs here."""
    hidden_size, latent, tp = 7168, 3584, 8
    seen: dict = {}
    op = _stub_op(0, latent // tp, latent, seen)
    rows = op.publish(
        torch.zeros(4, hidden_size, dtype=torch.bfloat16),
        torch.zeros(latent // tp, hidden_size, dtype=torch.bfloat16),
    )
    assert rows.shape == (4, latent) and rows.is_contiguous()
    assert rows.data_ptr() == seen["out"].data_ptr()
    assert seen["tokens"] == 4
    assert "mailbox" not in seen
    assert op.sentinel == latent_down._DOWN_SENTINEL


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
