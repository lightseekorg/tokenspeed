"""Contracts of the multicast down projection that hold without a fabric."""

import contextlib
from unittest import mock

import pytest
import torch
from tokenspeed_kernel.ops.moe import latent_down


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
        torch.zeros(1, 8, 64, dtype=torch.bfloat16), 0, {}, {}
    )


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
            latent_down, "current_platform", return_value=mock.Mock(is_nvidia=nvidia)
        ),
        mock.patch.object(latent_down.dist, "is_initialized", return_value=initialized),
        mock.patch.object(
            latent_down.dist, "all_gather_into_tensor", side_effect=gather_ceilings
        ),
    ):
        yield


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
    latent_down.arm_mailbox(mailbox)
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
            latent_down.KimiK3LatentDownOp.available(hidden, latent, tp, 92) is expected
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


def test_availability_requires_a_reachable_fabric() -> None:
    """An unreachable fabric must be refused before any rendezvous."""
    with _eligible(reachable=False):
        assert latent_down.KimiK3LatentDownOp.available(7168, 3584, 8, 92) is False


def test_availability_requires_an_nvidia_platform() -> None:
    """A term the fabric probe would also refuse still has to stand alone."""
    with _eligible(nvidia=False):
        assert latent_down.KimiK3LatentDownOp.available(7168, 3584, 8, 92) is False


def test_availability_requires_an_initialized_process_group() -> None:
    """Without distributed there is no group to rendezvous over."""
    with _eligible(initialized=False):
        assert latent_down.KimiK3LatentDownOp.available(7168, 3584, 8, 92) is False


def test_initialize_remembers_that_a_slot_could_not_be_built() -> None:
    """A failed build must not rendezvous again on the next layer.

    It now raises rather than returning None -- a rendezvous that yields no
    multicast address is the fabric failing a group that expected it -- so what
    this pins is that the remembered failure is re-raised without rebuilding.
    """
    latent_down.KimiK3LatentDownOp._pools.clear()
    group = mock.Mock(group_name="g")
    calls = []
    with (
        _eligible(),
        mock.patch.object(latent_down.dist, "get_rank", side_effect=lambda group: 0),
        mock.patch.object(
            latent_down.dist, "get_world_size", side_effect=lambda group: 8
        ),
        mock.patch.object(
            latent_down.KimiK3LatentDownOp,
            "_build_slot",
            side_effect=lambda *a: calls.append(a) or None,
        ),
    ):
        for layer in range(8):
            with pytest.raises(
                RuntimeError, match="rendezvous returned no multicast address"
            ):
                latent_down.KimiK3LatentDownOp.initialize(
                    group=group,
                    hidden_size=7168,
                    latent_size=3584,
                    device=torch.device("cpu"),
                    block_index=layer,
                    layer_count=92,
                    model_scope="s",
                    max_m=8,
                )
    assert len(calls) == latent_down._DOWN_POOL_DEPTH
    # The widths the op was initialized with are the widths the slot is built
    # for; transposing them compiles kernels that fail on the first batch.
    assert calls[0] == (group, 7168, 3584, torch.device("cpu"), 8)
    latent_down.KimiK3LatentDownOp._pools.clear()


def test_pool_key_separates_process_groups() -> None:
    """Two groups of the same size must not share a rendezvoused mailbox."""
    latent_down.KimiK3LatentDownOp._pools.clear()
    built = []
    with (
        _eligible(),
        mock.patch.object(latent_down.dist, "get_rank", side_effect=lambda group: 0),
        mock.patch.object(
            latent_down.dist, "get_world_size", side_effect=lambda group: 8
        ),
        mock.patch.object(
            latent_down.KimiK3LatentDownOp,
            "_build_slot",
            side_effect=lambda g, *a: built.append(g.group_name) or _stub_slot(),
        ),
    ):
        for name in ("base", "draft"):
            latent_down.KimiK3LatentDownOp.initialize(
                group=mock.Mock(group_name=name),
                hidden_size=7168,
                latent_size=3584,
                device=torch.device("cpu"),
                block_index=0,
                layer_count=92,
                model_scope="s",
                max_m=8,
            )
    assert built == ["base", "draft"]
    latent_down.KimiK3LatentDownOp._pools.clear()


@pytest.mark.parametrize("ceiling", [8, 9, 128, 512])
def test_the_width_gate_is_the_ceiling_it_was_built_with(ceiling: int) -> None:
    """Everything past the claimed width belongs to the replicated projection.

    The bound is per op now, not a module constant, so an op built for a wide
    decode ladder must claim the widths past the fused kernel's ceiling and an
    op built narrow must not.
    """
    op = latent_down.KimiK3LatentDownOp(
        latent_down._MailboxSlot(torch.zeros(1), 0, {}, {}), 448, 0, ceiling
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
    out = op(torch.zeros(4, hidden_size, dtype=torch.bfloat16), block_in)
    assert out.shape == (4, latent)
    assert (
        out.untyped_storage().data_ptr() != seen["mailbox"].untyped_storage().data_ptr()
    )
    assert seen["tokens"] == 4
    assert seen["weight"] is block_in
    assert seen["ptr"] == 1234
    # The kernel's capacity guard only protects a peer's heap if it is handed
    # rows for the batch; a narrower slice raises on every batch past the first.
    assert seen["out"].shape[1] == 4
    assert seen["out"].untyped_storage().data_ptr() == (
        seen["mailbox"].untyped_storage().data_ptr()
    )


def test_pool_key_separates_model_scopes() -> None:
    """A draft model must not inherit the base model's mailbox."""
    latent_down.KimiK3LatentDownOp._pools.clear()
    built = []
    group = mock.Mock(group_name="g")
    with (
        _eligible(),
        mock.patch.object(latent_down.dist, "get_rank", side_effect=lambda group: 0),
        mock.patch.object(
            latent_down.dist, "get_world_size", side_effect=lambda group: 8
        ),
        mock.patch.object(
            latent_down.KimiK3LatentDownOp,
            "_build_slot",
            side_effect=lambda *a: built.append(1) or _stub_slot(),
        ),
    ):
        for scope in ("model.layers", "model.decoder"):
            latent_down.KimiK3LatentDownOp.initialize(
                group=group,
                hidden_size=7168,
                latent_size=3584,
                device=torch.device("cpu"),
                block_index=0,
                layer_count=92,
                model_scope=scope,
                max_m=8,
            )
    assert len(built) == 2
    latent_down.KimiK3LatentDownOp._pools.clear()


def _voting_group(name="g"):
    """A group stub whose all_reduce we can drive rank by rank."""
    return mock.Mock(group_name=name)


@pytest.mark.parametrize("peer_says", [True, False])
def test_eligibility_is_agreed_across_the_group(peer_says: bool) -> None:
    """A rank must not rendezvous on a verdict its peers do not share."""
    latent_down.KimiK3LatentDownOp._verdicts.clear()
    latent_down.KimiK3LatentDownOp._pools.clear()
    built = []

    def min_vote(tensor, op=None, group=None):
        tensor.fill_(min(int(tensor.item()), int(peer_says)))

    with (
        _eligible(),
        mock.patch.object(latent_down.dist, "get_rank", side_effect=lambda group: 0),
        mock.patch.object(
            latent_down.dist, "get_world_size", side_effect=lambda group: 8
        ),
        mock.patch.object(latent_down.dist, "all_reduce", side_effect=min_vote),
        mock.patch.object(
            latent_down.KimiK3LatentDownOp,
            "_build_slot",
            side_effect=lambda *a: built.append(1) or _stub_slot(),
        ),
    ):
        latent_down.KimiK3LatentDownOp.initialize(
            group=_voting_group(),
            hidden_size=7168,
            latent_size=3584,
            device=torch.device("cuda"),
            block_index=0,
            layer_count=92,
            model_scope="s",
            max_m=8,
        )
    assert bool(built) is peer_says
    latent_down.KimiK3LatentDownOp._verdicts.clear()
    latent_down.KimiK3LatentDownOp._pools.clear()


def test_the_group_is_polled_once_not_once_per_layer() -> None:
    """92 layers must not cast 92 votes."""
    latent_down.KimiK3LatentDownOp._verdicts.clear()
    latent_down.KimiK3LatentDownOp._pools.clear()
    votes = []
    with (
        _eligible(),
        mock.patch.object(latent_down.dist, "get_rank", side_effect=lambda group: 0),
        mock.patch.object(
            latent_down.dist, "get_world_size", side_effect=lambda group: 8
        ),
        mock.patch.object(
            latent_down.dist, "all_reduce", side_effect=lambda t, **k: votes.append(1)
        ),
        mock.patch.object(
            latent_down.KimiK3LatentDownOp, "_build_slot", return_value=_stub_slot()
        ),
    ):
        group = _voting_group()
        for layer in range(8):
            latent_down.KimiK3LatentDownOp.initialize(
                group=group,
                hidden_size=7168,
                latent_size=3584,
                device=torch.device("cuda"),
                block_index=layer,
                layer_count=92,
                model_scope="s",
                max_m=8,
            )
    assert len(votes) == 1
    latent_down.KimiK3LatentDownOp._verdicts.clear()
    latent_down.KimiK3LatentDownOp._pools.clear()


@pytest.mark.parametrize("layers", [1, 2, 3, 4, 23, 31, 46, 92])
def test_availability_needs_a_whole_number_of_rotations(layers: int) -> None:
    """A stage's blocks must wrap onto a different slot, not the same one."""
    with _eligible():
        available = latent_down.KimiK3LatentDownOp.available(7168, 3584, 8, layers)
    depth = latent_down._DOWN_POOL_DEPTH
    assert available is (layers >= depth and layers % depth == 0)


def test_a_stage_that_wraps_onto_its_own_slot_is_refused() -> None:
    """PP3 and PP4 give 31 and 23 local blocks; both wrap onto themselves."""
    depth = latent_down._DOWN_POOL_DEPTH
    for local in (23, 31):
        slots = [latent_down._pool_slot(b, depth) for b in range(local)]
        assert slots[-1] == slots[0], "this is the case the gate must refuse"
        with _eligible():
            assert not latent_down.KimiK3LatentDownOp.available(7168, 3584, 8, local)


def test_the_vote_takes_the_minimum_so_one_refusal_is_decisive() -> None:
    """A MAX vote would let one willing rank drag the others into a rendezvous."""
    latent_down.KimiK3LatentDownOp._verdicts.clear()
    ops = []
    with (
        _eligible(),
        mock.patch.object(latent_down.dist, "get_rank", side_effect=lambda group: 0),
        mock.patch.object(
            latent_down.dist, "get_world_size", side_effect=lambda group: 8
        ),
        mock.patch.object(
            latent_down.dist,
            "all_reduce",
            side_effect=lambda t, op=None, group=None: ops.append(op),
        ),
        mock.patch.object(
            latent_down.KimiK3LatentDownOp, "_build_slot", return_value=_stub_slot()
        ),
    ):
        latent_down.KimiK3LatentDownOp.initialize(
            group=_voting_group(),
            hidden_size=7168,
            latent_size=3584,
            device=torch.device("cuda"),
            block_index=0,
            layer_count=92,
            model_scope="s",
            max_m=8,
        )
    assert ops == [latent_down.dist.ReduceOp.MIN]
    latent_down.KimiK3LatentDownOp._verdicts.clear()


def test_a_pooled_slot_is_re_armed_before_it_is_handed_out() -> None:
    """A reload must not inherit whatever an abandoned round left behind."""
    latent_down.KimiK3LatentDownOp._pools.clear()
    latent_down.KimiK3LatentDownOp._verdicts.clear()
    mailbox = torch.zeros(1, latent_down._FUSED_MAX_M, 64, dtype=torch.bfloat16)
    slot = latent_down._MailboxSlot(mailbox, 1, {}, {})
    with (
        _eligible(),
        mock.patch.object(latent_down.dist, "get_rank", side_effect=lambda group: 0),
        mock.patch.object(
            latent_down.dist, "get_world_size", side_effect=lambda group: 8
        ),
        mock.patch.object(latent_down.dist, "all_reduce"),
        mock.patch.object(
            latent_down.KimiK3LatentDownOp, "_build_slot", return_value=slot
        ),
    ):
        group = _voting_group()
        kwargs = dict(
            group=group,
            hidden_size=7168,
            latent_size=3584,
            device=torch.device("cpu"),
            block_index=0,
            layer_count=92,
            model_scope="s",
            max_m=8,
        )
        latent_down.KimiK3LatentDownOp.initialize(**kwargs)
        mailbox.fill_(7.0)
        latent_down.KimiK3LatentDownOp.initialize(**kwargs)
    words = mailbox.view(torch.int32)
    assert torch.equal(words, torch.full_like(words, -0x7FFF8000))
    latent_down.KimiK3LatentDownOp._pools.clear()
    latent_down.KimiK3LatentDownOp._verdicts.clear()


@pytest.mark.parametrize("tokens", [1, 4, 8, 9, 64])
def test_call_selects_the_kernel_compiled_for_this_width(tokens: int) -> None:
    """A batch must reach the kernel built for it and gather exactly its rows."""
    hidden_size, latent, tp = 7168, 3584, 8
    seen: dict = {}
    op = _stub_op(0, latent // tp, latent, seen, max_m=64)
    out = op(
        torch.zeros(tokens, hidden_size, dtype=torch.bfloat16),
        torch.zeros(latent, hidden_size, dtype=torch.bfloat16),
    )
    assert out.shape == (tokens, latent)
    assert seen["tokens"] == tokens


def test_zero_tokens_are_not_claimed() -> None:
    """There is no kernel compiled for an empty batch."""
    op = _stub_op(0, 448, 3584, {})
    assert not op.handles(0)


def test_build_slot_arms_and_binds_what_the_kernels_need() -> None:
    """The rendezvous body is where a mistake corrupts a peer's heap."""
    hidden_size, latent, tp = 7168, 3584, 8
    group = _voting_group()
    made = {}

    class FakeHandle:
        multicast_ptr = 4096

    fake_symm = mock.Mock()
    fake_symm.empty.side_effect = lambda shape, dtype, device: made.setdefault(
        "mailbox", torch.zeros(shape, dtype=dtype)
    )

    def record_rendezvous(mailbox, group):
        made["group"] = group
        made["rendezvoused"] = mailbox
        return FakeHandle()

    fake_symm.rendezvous.side_effect = record_rendezvous

    gemms = {}
    with (
        mock.patch("torch.distributed._symmetric_memory.empty", fake_symm.empty),
        mock.patch(
            "torch.distributed._symmetric_memory.rendezvous", fake_symm.rendezvous
        ),
        mock.patch.object(latent_down.dist, "get_rank", side_effect=lambda group: 3),
        mock.patch.object(
            latent_down.dist, "get_world_size", side_effect=lambda group: tp
        ),
        mock.patch.object(latent_down.dist, "all_reduce"),
        mock.patch(
            "tokenspeed_kernel.thirdparty.cute_dsl.latent_moe_tail"
            ".fused_multicast_latent_down_gemm.FusedMulticastLatentDownGemmKernel",
            side_effect=lambda **kw: gemms.setdefault(kw["num_rows"], kw),
        ),
        mock.patch(
            "tokenspeed_kernel.thirdparty.cute_dsl.latent_moe_tail"
            ".lamport_copy.LamportCopyKernel",
            side_effect=lambda **kw: made.setdefault("gather", kw),
        ),
    ):
        slot = latent_down.KimiK3LatentDownOp._build_slot(
            group, hidden_size, latent, torch.device("cpu"), 8
        )
    assert slot is not None
    assert slot.mailbox.shape == (1, latent_down._FUSED_MAX_M, latent)
    words = slot.mailbox.view(torch.int32)
    assert torch.equal(words, torch.full_like(words, -0x7FFF8000))
    assert made["group"] is group
    assert sorted(gemms) == list(range(1, latent_down._FUSED_MAX_M + 1))
    for kw in gemms.values():
        assert (kw["rank"], kw["tp_size"]) == (3, tp)
        assert (kw["in_dim"], kw["latent_dim"]) == (hidden_size, latent)
    assert made["gather"]["hidden_dim"] == latent
    assert made["gather"]["max_m"] == latent_down._FUSED_MAX_M
    # Pin that the tuned launch geometry reaches the kernel, not its value.
    assert made["gather"]["ctas"] == latent_down._LAMPORT_CTAS
    assert made["gather"]["threads"] == latent_down._LAMPORT_THREADS
    # The gather must spin on the word the arming above wrote, not the default
    # one every other Lamport buffer keeps.
    assert made["gather"]["sentinel"] == latent_down._DOWN_SENTINEL == 0x80008000
    assert words[0, 0, 0].item() == -0x7FFF8000
    # The address the rendezvous returned is the one every peer is published
    # through; binding anything else stores through a pointer nobody owns.
    assert slot.multicast_ptr == FakeHandle.multicast_ptr
    # The address bound above belongs to the tensor that was rendezvoused, not
    # to some other buffer the peers know nothing about.
    assert made["rendezvoused"] is slot.mailbox


def test_pool_slots_alternate_over_moe_blocks_not_decoder_layers() -> None:
    """A MoE frequency above one must not collapse the pool onto one slot."""
    for freq in (1, 2, 4):
        blocks = [layer // freq for layer in range(0, 12 * freq, freq)]
        slots = {latent_down._pool_slot(b) for b in blocks}
        assert len(slots) == latent_down._DOWN_POOL_DEPTH


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
    fake_symm = mock.Mock()
    fake_symm.empty.side_effect = lambda shape, dtype, device: torch.zeros(
        shape, dtype=dtype
    )
    fake_symm.rendezvous.side_effect = lambda mb, g: mock.Mock(multicast_ptr=4096)
    with (
        mock.patch("torch.distributed._symmetric_memory.empty", fake_symm.empty),
        mock.patch(
            "torch.distributed._symmetric_memory.rendezvous", fake_symm.rendezvous
        ),
        mock.patch.object(
            latent_down.KimiK3LatentDownOp, "_agree_on_pointer", return_value=False
        ),
    ):
        slot = latent_down.KimiK3LatentDownOp._build_slot(
            _voting_group(), 7168, 3584, torch.device("cpu"), 8
        )
    assert slot is None


def test_the_verdict_does_not_answer_for_another_rotation() -> None:
    """A second model on the same group has its own block count."""
    group = mock.Mock(group_name="g")
    votes, built = [], []
    with (
        _eligible(),
        mock.patch.object(latent_down.dist, "get_rank", side_effect=lambda group: 0),
        mock.patch.object(
            latent_down.dist, "get_world_size", side_effect=lambda group: 8
        ),
        mock.patch.object(
            latent_down.dist,
            "all_reduce",
            # Record what is cast, not that a cast happened: the eligibility
            # answer reaches production only through this tensor.
            side_effect=lambda t, **k: votes.append(int(t.item())),
        ),
        mock.patch.object(
            latent_down.KimiK3LatentDownOp,
            "_build_slot",
            side_effect=lambda *a: built.append(a) or _stub_slot(),
        ),
    ):
        latent_down.KimiK3LatentDownOp.initialize(
            group=group,
            hidden_size=7168,
            latent_size=3584,
            device=torch.device("cpu"),
            block_index=0,
            layer_count=92,
            model_scope="base",
            max_m=8,
        )
        # 23 blocks is not a whole number of rotations, so this one is refused
        # -- but only if the remembered verdict was not asked the wrong question.
        assert (
            latent_down.KimiK3LatentDownOp.initialize(
                group=group,
                hidden_size=7168,
                latent_size=3584,
                device=torch.device("cpu"),
                block_index=0,
                layer_count=23,
                model_scope="draft",
                max_m=8,
            )
            is None
        )
    # The second rotation is refused because a 0 was cast, not because the
    # build failed: without this the guards `available` pins go unconsulted.
    assert votes == [1, 0]


def test_the_pool_is_keyed_by_the_depth_it_rotates_over() -> None:
    """Two depths over one group are two rotations, not one."""
    group = mock.Mock(group_name="g")
    built = []
    with (
        _eligible(),
        mock.patch.object(latent_down.dist, "get_rank", side_effect=lambda group: 0),
        mock.patch.object(
            latent_down.dist, "get_world_size", side_effect=lambda group: 8
        ),
        mock.patch.object(latent_down.dist, "all_reduce", lambda t, **k: None),
        mock.patch.object(
            latent_down.KimiK3LatentDownOp,
            "_build_slot",
            side_effect=lambda *a: built.append(a) or _stub_slot(),
        ),
    ):
        for depth in (2, 4):
            latent_down.KimiK3LatentDownOp.initialize(
                group=group,
                hidden_size=7168,
                latent_size=3584,
                device=torch.device("cpu"),
                block_index=0,
                layer_count=92,
                model_scope="s",
                max_m=8,
                pool_depth=depth,
            )
    assert len(built) == 2


def test_the_eight_rank_check_program_still_binds() -> None:
    """That program is the only exerciser of the real path; it drifted once."""
    import inspect

    inspect.signature(latent_down.KimiK3LatentDownOp.initialize).bind(
        group=object(),
        hidden_size=7168,
        latent_size=3584,
        device=torch.device("cpu"),
        block_index=0,
        layer_count=92,
        model_scope="check",
        max_m=8,
        pool_depth=2,
    )


def test_the_verdict_is_per_group() -> None:
    """Two groups of one shape still each need their own vote."""
    votes = []
    with (
        _eligible(),
        mock.patch.object(latent_down.dist, "get_rank", side_effect=lambda group: 0),
        mock.patch.object(
            latent_down.dist, "get_world_size", side_effect=lambda group: 8
        ),
        mock.patch.object(
            latent_down.dist, "all_reduce", side_effect=lambda t, **k: votes.append(1)
        ),
        mock.patch.object(
            latent_down.KimiK3LatentDownOp,
            "_build_slot",
            side_effect=lambda *a: _stub_slot(),
        ),
    ):
        for name in ("g", "h"):
            latent_down.KimiK3LatentDownOp.initialize(
                group=mock.Mock(group_name=name),
                hidden_size=7168,
                latent_size=3584,
                device=torch.device("cpu"),
                block_index=0,
                layer_count=92,
                model_scope="s",
                max_m=8,
            )
    assert len(votes) == 2


def test_the_vote_asks_about_the_widths_in_the_right_order() -> None:
    """A latent narrower than the k-tile is eligible; a hidden that narrow is not."""
    votes = []
    with (
        _eligible(),
        mock.patch.object(latent_down.dist, "get_rank", side_effect=lambda group: 0),
        mock.patch.object(
            latent_down.dist, "get_world_size", side_effect=lambda group: 8
        ),
        mock.patch.object(
            latent_down.dist,
            "all_reduce",
            side_effect=lambda t, **k: votes.append(int(t.item())),
        ),
        mock.patch.object(
            latent_down.KimiK3LatentDownOp,
            "_build_slot",
            side_effect=lambda *a: _stub_slot(),
        ),
    ):
        latent_down.KimiK3LatentDownOp.initialize(
            group=mock.Mock(group_name="g"),
            hidden_size=7168,
            latent_size=1792,
            device=torch.device("cpu"),
            block_index=0,
            layer_count=92,
            model_scope="s",
            max_m=8,
        )
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
        mock.patch.object(latent_down, "arm_mailbox", lambda mailbox: None),
    ):
        op = latent_down.KimiK3LatentDownOp.initialize(
            group=group,
            hidden_size=7168,
            latent_size=3584,
            device=torch.device("cpu"),
            block_index=0,
            layer_count=92,
            model_scope="base",
            max_m=8,
        )

    assert op is not None
    assert op._slot is slot
    assert op.rank == 5
    assert op.shard_dim == 3584 // 8


def test_build_slot_votes_about_the_address_it_actually_got() -> None:
    """A rank whose rendezvous yields nothing must not vote as if it had."""
    group = _voting_group()

    class NullHandle:
        multicast_ptr = 0

    with (
        mock.patch(
            "torch.distributed._symmetric_memory.empty",
            lambda shape, dtype, device: torch.zeros(shape, dtype=dtype),
        ),
        mock.patch(
            "torch.distributed._symmetric_memory.rendezvous",
            lambda mailbox, group: NullHandle(),
        ),
        mock.patch.object(latent_down.dist, "get_rank", side_effect=lambda group: 0),
        mock.patch.object(
            latent_down.dist, "get_world_size", side_effect=lambda group: 8
        ),
        # Not patched: _agree_on_pointer must be asked about the real address.
        mock.patch.object(latent_down.dist, "all_reduce", lambda t, **k: None),
    ):
        slot = latent_down.KimiK3LatentDownOp._build_slot(
            group, 7168, 3584, torch.device("cpu"), 8
        )

    assert slot is None


def test_build_slot_sizes_everything_to_the_ceiling_it_is_given() -> None:
    """The mailbox, the gather and the dispatch must agree on one width."""
    hidden_size, latent, tp, ceiling = 7168, 3584, 8, 32
    group = _voting_group()
    made: dict = {}

    class FakeHandle:
        multicast_ptr = 4096

    gemms, gathers, views = {}, [], []
    with (
        mock.patch(
            "torch.distributed._symmetric_memory.empty",
            lambda shape, dtype, device: made.setdefault(
                "mailbox", torch.zeros(shape, dtype=dtype)
            ),
        ),
        mock.patch(
            "torch.distributed._symmetric_memory.rendezvous",
            lambda mailbox, group: FakeHandle(),
        ),
        mock.patch.object(latent_down.dist, "get_rank", side_effect=lambda group: 3),
        mock.patch.object(
            latent_down.dist, "get_world_size", side_effect=lambda group: tp
        ),
        mock.patch.object(latent_down.dist, "all_reduce"),
        mock.patch(
            "tokenspeed_kernel.thirdparty.cute_dsl.latent_moe_tail"
            ".fused_multicast_latent_down_gemm.FusedMulticastLatentDownGemmKernel",
            side_effect=lambda **kw: gemms.setdefault(kw["num_rows"], kw),
        ),
        mock.patch(
            "tokenspeed_kernel.thirdparty.cute_dsl.latent_moe_tail"
            ".lamport_copy.LamportCopyKernel",
            side_effect=lambda **kw: gathers.append(kw) or kw,
        ),
        mock.patch.object(
            latent_down,
            "bf16_tensor_on_pointer",
            side_effect=lambda *a: views.append(a) or torch.zeros(1),
        ),
    ):
        slot = latent_down.KimiK3LatentDownOp._build_slot(
            group, hidden_size, latent, torch.device("cpu"), ceiling
        )
    assert slot is not None
    assert slot.mailbox.shape == (1, ceiling, latent)
    # The fused kernel compiles one static M and has none past its ceiling, so
    # the widths above it are a different producer behind the same dispatch.
    assert sorted(gemms) == list(range(1, latent_down._FUSED_MAX_M + 1))
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
    assert len(gathers) == 1
    assert gathers[0]["max_m"] == ceiling
    assert len({id(g) for g in slot.gather_by_m.values()}) == 1
    # The view is this rank's column block of the mailbox, addressed through
    # the multicast pointer: a wrong base publishes over a peer's columns.
    assert len(views) == 1
    pointer, shape, strides, _ = views[0]
    assert pointer == FakeHandle.multicast_ptr + 3 * (latent // tp) * 2
    assert shape == (ceiling, latent // tp)
    assert strides == (latent, 1)


def test_a_ceiling_at_the_fused_width_builds_no_wide_producer() -> None:
    """Nothing past the fused kernel is claimed, so nothing else is built."""
    views = []
    with (
        mock.patch(
            "torch.distributed._symmetric_memory.empty",
            lambda shape, dtype, device: torch.zeros(shape, dtype=dtype),
        ),
        mock.patch(
            "torch.distributed._symmetric_memory.rendezvous",
            lambda mailbox, group: mock.Mock(multicast_ptr=4096),
        ),
        mock.patch.object(latent_down.dist, "get_rank", side_effect=lambda group: 0),
        mock.patch.object(
            latent_down.dist, "get_world_size", side_effect=lambda group: 8
        ),
        mock.patch.object(latent_down.dist, "all_reduce"),
        mock.patch(
            "tokenspeed_kernel.thirdparty.cute_dsl.latent_moe_tail"
            ".fused_multicast_latent_down_gemm.FusedMulticastLatentDownGemmKernel",
            side_effect=lambda **kw: kw,
        ),
        mock.patch(
            "tokenspeed_kernel.thirdparty.cute_dsl.latent_moe_tail"
            ".lamport_copy.LamportCopyKernel",
            side_effect=lambda **kw: kw,
        ),
        mock.patch.object(
            latent_down,
            "bf16_tensor_on_pointer",
            side_effect=lambda *a: views.append(a) or torch.zeros(1),
        ),
    ):
        slot = latent_down.KimiK3LatentDownOp._build_slot(
            _voting_group(), 7168, 3584, torch.device("cpu"), latent_down._FUSED_MAX_M
        )
    assert views == []
    assert sorted(slot.gemm_by_m) == list(range(1, latent_down._FUSED_MAX_M + 1))


def test_the_ceiling_never_falls_below_the_fused_kernel_s_width() -> None:
    """Those widths are compiled either way; claiming fewer only loses them."""
    built = []
    with (
        _eligible(),
        mock.patch.object(latent_down.dist, "get_rank", side_effect=lambda group: 0),
        mock.patch.object(
            latent_down.dist, "get_world_size", side_effect=lambda group: 8
        ),
        mock.patch.object(latent_down.dist, "all_reduce", lambda t, **k: None),
        mock.patch.object(
            latent_down.KimiK3LatentDownOp,
            "_build_slot",
            side_effect=lambda *a: built.append(a) or _stub_slot(),
        ),
    ):
        latent_down.KimiK3LatentDownOp.initialize(
            group=_voting_group(),
            hidden_size=7168,
            latent_size=3584,
            device=torch.device("cpu"),
            block_index=0,
            layer_count=92,
            model_scope="s",
            max_m=1,
        )
    assert built[0][-1] == latent_down._FUSED_MAX_M


def test_a_ceiling_below_one_row_is_refused() -> None:
    """A zero-row mailbox would hand the gather a width it cannot run."""
    with _eligible():
        with pytest.raises(ValueError, match="at least one row"):
            latent_down.KimiK3LatentDownOp.initialize(
                group=_voting_group(),
                hidden_size=7168,
                latent_size=3584,
                device=torch.device("cpu"),
                block_index=0,
                layer_count=92,
                model_scope="s",
                max_m=0,
            )


def test_the_pool_key_separates_two_ceilings() -> None:
    """A wider op must not inherit a mailbox that is too short for it."""
    latent_down.KimiK3LatentDownOp._pools.clear()
    latent_down.KimiK3LatentDownOp._verdicts.clear()
    built = []
    group = mock.Mock(group_name="g")
    with (
        _eligible(),
        mock.patch.object(latent_down.dist, "get_rank", side_effect=lambda group: 0),
        mock.patch.object(
            latent_down.dist, "get_world_size", side_effect=lambda group: 8
        ),
        mock.patch.object(latent_down.dist, "all_reduce", lambda t, **k: None),
        mock.patch.object(
            latent_down.KimiK3LatentDownOp,
            "_build_slot",
            side_effect=lambda *a: built.append(a) or _stub_slot(),
        ),
    ):
        for ceiling in (8, 64):
            latent_down.KimiK3LatentDownOp.initialize(
                group=group,
                hidden_size=7168,
                latent_size=3584,
                device=torch.device("cpu"),
                block_index=0,
                layer_count=92,
                model_scope="s",
                max_m=ceiling,
            )
    assert [call[-1] for call in built] == [8, 64]
    latent_down.KimiK3LatentDownOp._pools.clear()
    latent_down.KimiK3LatentDownOp._verdicts.clear()


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


def test_the_wide_producer_overwrites_rather_than_accumulates() -> None:
    """The publish overwrites its destination rather than accumulating onto it.

    What this pins is narrow: the producer runs over an ordinary CPU tensor, so
    it holds the call to ``torch.mm(out=)`` semantics and would still pass with
    the multicast view, the mailbox and cuBLAS all replaced.

    Nothing covers beta on the real path, and the obvious candidate cannot.
    The mailbox is armed with ``0x80008000``, both BF16 lanes negative zero,
    and ``x + (-0.0) == x`` bitwise for every value -- so an accumulating
    publish onto an armed mailbox is byte-identical to an overwriting one, and
    no comparison of final values can tell them apart. A witness would have to
    observe the transient: a peer reading these columns while the GEMM is still
    accumulating into them. That test does not exist.
    """
    latent, shard_dim, ceiling, k = 3584, 448, 16, 128
    torch.manual_seed(3)
    garbage = torch.randn(ceiling, shard_dim) * 1e3
    backing = garbage.clone()
    producer = _wide_producer_over(
        backing,
        {},
        multicast_ptr=4096,
        rank=0,
        shard_dim=shard_dim,
        latent_size=latent,
        max_m=ceiling,
        device=torch.device("cpu"),
    )
    hidden = torch.randn(ceiling, k)
    weight_block = torch.randn(shard_dim, k)
    producer(hidden, weight_block, None, 4096)
    torch.testing.assert_close(backing, hidden @ weight_block.T)


def test_the_gather_geometry_is_chosen_in_one_place() -> None:
    """A per-width table lands in _lamport_geometry and nowhere else."""
    geometries = {latent_down._lamport_geometry(m) for m in (1, 8, 9, 64, 512)}
    assert geometries == {(latent_down._LAMPORT_CTAS, latent_down._LAMPORT_THREADS)}


def _initialize(**overrides):
    """Drive initialize with the arguments every rank agrees on but the ceiling."""
    kwargs = dict(
        group=_voting_group(),
        hidden_size=7168,
        latent_size=3584,
        device=torch.device("cuda"),
        block_index=0,
        layer_count=92,
        model_scope="ceiling",
        max_m=128,
    )
    kwargs.update(overrides)
    return latent_down.KimiK3LatentDownOp.initialize(**kwargs)


@contextlib.contextmanager
def _voting_ranks(built):
    """Rank 0 of eight, with the availability vote passing and no allocation."""
    with (
        mock.patch.object(latent_down.dist, "get_rank", side_effect=lambda group: 0),
        mock.patch.object(
            latent_down.dist, "get_world_size", side_effect=lambda group: 8
        ),
        mock.patch.object(latent_down.dist, "all_reduce", side_effect=lambda t, **k: t),
        mock.patch.object(
            latent_down.KimiK3LatentDownOp,
            "_build_slot",
            side_effect=lambda *a: built.append(1) or _stub_slot(),
        ),
    ):
        yield


def test_a_disagreed_ceiling_is_refused_before_anything_is_allocated() -> None:
    """Differently sized mailboxes hang the rendezvous with no traceback.

    So the disagreement has to be an error raised where the cause is, not a
    hang later that looks like any other boot hang on this fabric.
    """
    built: list[int] = []
    with _eligible(peer_ceiling=1024), _voting_ranks(built):
        with pytest.raises(ValueError, match="size the down mailbox alike"):
            _initialize(max_m=128)
    assert built == []


def test_the_disagreement_names_what_each_rank_derived() -> None:
    """A bare "mismatch" would leave the next person nothing to look at."""
    built: list[int] = []
    with _eligible(peer_ceiling=1024), _voting_ranks(built):
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

    with _eligible(), _voting_ranks(built):
        with mock.patch.object(
            latent_down.dist, "all_gather_into_tensor", side_effect=counted
        ):
            for block in range(4):
                _initialize(block_index=block)
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
    args = dict(hidden_size=7168, latent_size=3584, tp_size=8, layer_count=92)
    args.update(kwargs)
    with _eligible():
        reason = latent_down.KimiK3LatentDownOp._unavailable_reason(**args)
    assert reason is not None and expected in reason


def test_the_reason_predicate_still_backs_availability() -> None:
    """available() must stay exactly the negation, or the two can drift apart."""
    args = dict(hidden_size=7168, latent_size=3584, tp_size=8, layer_count=92)
    with _eligible():
        assert latent_down.KimiK3LatentDownOp.available(**args)
        assert latent_down.KimiK3LatentDownOp._unavailable_reason(**args) is None
    with _eligible(reachable=False):
        assert not latent_down.KimiK3LatentDownOp.available(**args)
        assert "fabric" in latent_down.KimiK3LatentDownOp._unavailable_reason(**args)


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
