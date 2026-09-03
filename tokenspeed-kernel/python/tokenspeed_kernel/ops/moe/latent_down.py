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

"""Multicast down-projection for Kimi-K3's latent MoE.

The replicated projection streams the whole 51.4 MB weight on every rank. Here
each rank owns a contiguous block of the output columns, publishes that block
straight into every peer's symmetric mailbox through the NVLS multicast
address, and a Lamport gather turns the mailbox into the full latent without a
barrier. The weight block is a view, so nothing is duplicated.

Two producers write that block, split at the width the fused kernel supports.
Up to eight rows a fused SIMT GEMM computes and publishes in one launch. Wider
batches hand the multicast address to cuBLAS as its output tensor, which
broadcasts the same way for the price of the store -- but skips the epilogue
that keeps the fused path's output off the Lamport sentinel. Those widths stay
off it for two other reasons: the mailbox is armed with a word both BF16 lanes
spell -0, which a product would have to reach twice over rather than once, and
the publish overwrites rather than accumulates onto rows still holding
sentinels. Which producer serves them is chosen in one place, ``_wide_producer``.

Every rank must construct in lockstep: the mailbox rendezvous is collective.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.distributed as dist
from tokenspeed_kernel.ops.moe.latent_tail import (
    multicast_backend_unavailable_reason,
)
from tokenspeed_kernel.ops.moe.multicast_view import bf16_tensor_on_pointer
from tokenspeed_kernel.platform import current_platform

# Same rotation depth as the up-projection tail, for the same reason.
_DOWN_POOL_DEPTH = 2
# Widest batch the fused kernel compiles a static-M instance for.
_FUSED_MAX_M = 8
# Widest k-tile the fused kernel's configs use; a hidden width off it cannot compile.
_K_TILE = 448 * 8
# Ties or beats 304 at every measured width, and wins at 1, 2 and 256 upward.
_LAMPORT_CTAS = 608
_LAMPORT_THREADS = 512
# Both BF16 lanes -0, so an unsanitized producer needs two coincidences, not one.
_DOWN_SENTINEL = 0x80008000
_BF16_BYTES = 2

logger = logging.getLogger(__name__)
# Reasons already reported; the decline repeats per MoE block and says nothing new.
_DECLINED: set[str] = set()
# The architecture whose fabric semantics are validated here, and so the only
# one where failing to rendezvous means a broken machine rather than an
# unsupported one. Deliberately NOT latent_tail's _MULTICAST_MIN_ARCH: that
# decides whether to attempt the path and admits later architectures, this
# decides whether to fail a boot over the attempt and must not. They agree
# today and are separate numbers because they answer different questions.
_MULTICAST_VALIDATED_ARCH = 10


def _rendezvous_failure_is_a_fault() -> bool:
    """Whether a rendezvous that yielded nothing means the machine is broken.

    On hardware nobody here has run, a rendezvous that cannot deliver is far
    likelier to mean the path was never really supported than that the fabric
    failed -- and the message we would print names Blackwell specifics that may
    not even apply there. So above the validated architecture this declines,
    which is the pre-existing behaviour and what the shipped gate asks for.
    """
    platform = current_platform()
    return (
        platform.is_nvidia and platform.arch_version.major == _MULTICAST_VALIDATED_ARCH
    )


def _fabric_fault_message(reason: str) -> str:
    """What to look at when a rendezvous this group committed to yields nothing.

    Raised only here, after every probe said the fabric could map the group:
    that is the one signal that cannot also mean hardware which never carried
    the feature, so it is the one place a boot may fail. The alternative is not
    a working server but one that quietly gives back 3.85 GiB of weight per GPU
    and 20-40 percent of the projection, found months later from a throughput
    number if at all.
    """
    return (
        f"Kimi-K3 down mailbox: the fabric said it could map this group and "
        f"then the rendezvous returned no multicast address ({reason}). Check "
        f"that /dev/nvidia-caps-imex-channels is populated on every node, and "
        f"that CUDA_VISIBLE_DEVICES is not set externally, which detaches the "
        f"mnnvl lane while leaving every other probe passing."
    )


def _decline(reason: str) -> None:
    """Report why the mailbox is unavailable, once per reason.

    The replicated fallback is correct, so nothing fails, which is exactly the
    problem: the whole measured win silently does not happen and the only other
    symptom is a throughput number nobody can attribute months later.
    """
    if reason not in _DECLINED:
        _DECLINED.add(reason)
        logger.info(
            "Kimi-K3 down mailbox unavailable (%s); the projection stays replicated",
            reason,
        )
    return None


def _pool_slot(block_index: int, depth: int = _DOWN_POOL_DEPTH) -> int:
    """Map an ordinal among the MoE blocks to its rank-identical slot.

    The ordinal, not the decoder layer id: with a MoE frequency above one every
    MoE layer has the same parity, which would collapse the pool to one slot and
    leave consecutive rounds sharing a mailbox with nothing in between.
    """
    return block_index % depth


def _lamport_geometry(m: int) -> tuple[int, int]:
    """The ``(ctas, threads)`` the gather launches at for a batch this wide.

    One pair serves every width today. The pair that wins is measured per M, so
    a width-keyed table drops in here alone: the mailbox, the producers and the
    dispatch are all indifferent to which geometry a width picks, and kernels
    are built once per distinct pair rather than once per width.
    """
    return _LAMPORT_CTAS, _LAMPORT_THREADS


@dataclass
class _MailboxSlot:
    """One symmetric mailbox plus the kernels bound to it."""

    mailbox: torch.Tensor
    multicast_ptr: int
    gemm_by_m: dict[int, object]
    gather_by_m: dict[int, object]


class _MulticastVaGemm:
    """cuBLAS writing this rank's column block straight through the multicast VA.

    The block is a strided view over the mailbox's multicast address at this
    rank's column base, so the GEMM's own store is the publish: no staging
    buffer and no copy kernel behind it. Rows are ``latent_size`` apart, which
    is a leading dimension cuBLAS takes without staging the output itself.

    One instance covers every width past the fused kernel's ceiling: the view
    is built once at full capacity and a batch takes its first rows.

    A library GEMM exposes no way to release its dependents, so unlike the
    fused producer it does not let the gather become resident early.
    """

    def __init__(
        self,
        *,
        multicast_ptr: int,
        rank: int,
        shard_dim: int,
        latent_size: int,
        max_m: int,
        device: torch.device,
    ) -> None:
        with torch.inference_mode(False), torch.no_grad():
            self._block = bf16_tensor_on_pointer(
                multicast_ptr + rank * shard_dim * _BF16_BYTES,
                (max_m, shard_dim),
                (latent_size, 1),
                device.index,
            )

    def __call__(
        self,
        hidden_states: torch.Tensor,
        weight_block: torch.Tensor,
        mailbox: torch.Tensor,
        multicast_ptr: int,
    ) -> torch.Tensor:
        """Publish ``hidden_states @ weight_block.T`` into every peer's mailbox.

        Args:
            hidden_states: Contiguous ``[tokens, hidden_size]`` activation.
            weight_block: This rank's ``[shard_dim, hidden_size]`` weight rows.
            mailbox: The local mailbox, returned for signature parity with the
                fused producer; the store goes through the multicast address.
            multicast_ptr: Unused here, folded into the view at construction.

        Returns:
            The mailbox, as the fused producer returns it.
        """
        published = self._block[: hidden_states.shape[0]]
        # Overwrites rather than accumulates, so a word transitions once a round.
        # Untested, and not cheaply testable: the mailbox is armed with negative
        # zero in both BF16 lanes, and adding that is the identity bitwise, so an
        # accumulating publish is byte-identical to this one. A witness would
        # have to catch a peer reading these columns mid-accumulation.
        torch.mm(hidden_states, weight_block.t(), out=published)
        return mailbox


def _wide_producer(
    *,
    multicast_ptr: int,
    rank: int,
    shard_dim: int,
    latent_size: int,
    max_m: int,
    device: torch.device,
) -> object:
    """Choose the producer for widths past the fused kernel's ceiling.

    One producer today, and this is the only place a second one is chosen.
    Every producer is called as
    ``(hidden_states, weight_block, mailbox, multicast_ptr)`` and leaves this
    rank's column block in every peer's mailbox, so a swap is this function
    plus whatever the new producer needs to be constructed with.

    The alternative this seam exists for is ``AdaptiveUpProjectionKernel`` in
    ``thirdparty/cute_dsl/latent_moe_tail/fused_add_multicast_gemm.py``: a
    tcgen05 persistent multicast GEMM that runs the tensor-core path at every
    width with ``skinny_max_m=0`` and releases the gather early, which a
    library GEMM gives no way to do. It owns the mailbox it publishes into, so
    adopting it means the slot borrowing that mailbox rather than this one.
    """
    return _MulticastVaGemm(
        multicast_ptr=multicast_ptr,
        rank=rank,
        shard_dim=shard_dim,
        latent_size=latent_size,
        max_m=max_m,
        device=device,
    )


class KimiK3LatentDownOp:
    """Project hidden states to the latent through a multicast column shard."""

    _pools: dict[tuple, _MailboxSlot] = {}
    _verdicts: dict[tuple, bool] = {}
    _ceilings: dict[tuple, bool] = {}
    _reasons: dict[tuple, str] = {}

    def __init__(
        self, slot: _MailboxSlot, shard_dim: int, rank: int, max_m: int
    ) -> None:
        self._slot = slot
        self.shard_dim = shard_dim
        self.rank = rank
        self.max_m = max_m

    @classmethod
    def available(
        cls,
        hidden_size: int,
        latent_size: int,
        tp_size: int,
        layer_count: int,
        group: dist.ProcessGroup | None = None,
        pool_depth: int = _DOWN_POOL_DEPTH,
    ) -> bool:
        """Whether this rank can host the multicast down projection.

        The reachability probe has to run before the rendezvous, not after: a
        cross-host group without fabric reports multicast support locally and
        then hangs inside the rendezvous rather than failing. So does the width
        test, which the kernel would otherwise raise on after allocating.

        ``layer_count`` -- the blocks this stage actually runs, not the model's
        total -- must be a whole number of rotations. The gather releases its
        dependents before re-arming, so a slot needs a layer of work between
        rounds, and that includes the wrap from a stage's last block to its
        first on the next pass: an odd count under pipeline parallelism lands
        both on the same slot with nothing in between.
        """
        return (
            cls._unavailable_reason(
                hidden_size, latent_size, tp_size, layer_count, group, pool_depth
            )
            is None
        )

    @classmethod
    def _unavailable_reason(
        cls,
        hidden_size: int,
        latent_size: int,
        tp_size: int,
        layer_count: int,
        group: dist.ProcessGroup | None = None,
        pool_depth: int = _DOWN_POOL_DEPTH,
    ) -> str | None:
        """The first condition this rank fails, or None when it can host.

        Every one of these is a quiet decline, because none can be told apart
        from hardware that simply does not carry the feature: an H100 fails on
        capability, and an unreachable fabric reads identically whether IMEX is
        broken or the rack has no NVLink domain. What is worth failing a boot
        over is the rendezvous, which runs only after all of this has passed.

        Same order as the predicate it backs, which matters: the reachability
        probe stays last, after the width tests the kernel would otherwise
        raise on, and after the rotation test.
        """
        if not dist.is_initialized():
            return "no process group"
        if not current_platform().is_nvidia:
            return "not an NVIDIA platform"
        if tp_size <= 1:
            return "a single rank has nothing to gather"
        if layer_count < pool_depth or layer_count % pool_depth:
            return f"{layer_count} blocks is not whole {pool_depth}-slot rotations"
        if hidden_size % _K_TILE:
            return f"hidden {hidden_size} is not a multiple of the {_K_TILE} k-tile"
        if latent_size % tp_size or (latent_size // tp_size) % 8:
            return f"latent {latent_size} does not split into {tp_size} blocks of 8"
        return multicast_backend_unavailable_reason(group)

    @classmethod
    def initialize(
        cls,
        *,
        group: dist.ProcessGroup,
        hidden_size: int,
        latent_size: int,
        device: torch.device,
        block_index: int,
        layer_count: int,
        model_scope: str,
        max_m: int,
        pool_depth: int = _DOWN_POOL_DEPTH,
    ) -> "KimiK3LatentDownOp | None":
        """Bind this layer to a pooled mailbox, or return None if unsupported.

        Args:
            group: Process group every rank constructs with, in lockstep. Its
                name is part of the pool key, so two groups of the same size
                cannot share a mailbox one of them never rendezvoused on.
            hidden_size: Contraction width of the projection.
            latent_size: Full output width, split across the group.
            device: Device owning the mailbox.
            block_index: This layer's ordinal among the MoE blocks.
            layer_count: MoE blocks this stage runs; a whole number of rotations.
            model_scope: Separates concurrently executing models.
            max_m: Widest batch this op claims, and the mailbox's row capacity.
                It is the gate itself, not a property of the capture ladder: a
                width above it takes the column gather, never the replicated
                projection, because a caller that has a mailbox necessarily has
                the column group too -- both need the latent width to divide the
                group. Raised to the fused kernel's ceiling when smaller: those
                widths are compiled either way and cost eight mailbox rows.

        Returns:
            The op, or None when the platform or shapes cannot support it.
        """
        if not dist.is_initialized():
            return _decline("no process group")
        if max_m < 1:
            raise ValueError("max_m must be at least one row")
        max_m = max(max_m, _FUSED_MAX_M)
        rank = dist.get_rank(group)
        tp_size = dist.get_world_size(group)
        verdict = cls._agreed(
            group,
            hidden_size,
            latent_size,
            tp_size,
            layer_count,
            pool_depth,
            max_m,
            device,
        )
        if not verdict:
            key = cls._verdict_key(
                group, hidden_size, latent_size, tp_size, layer_count, pool_depth, max_m
            )
            return _decline(cls._reasons.get(key, "the group declined"))
        cls._agree_on_ceiling(group, max_m, tp_size, device)
        shard_dim = latent_size // tp_size
        key = (
            hidden_size,
            latent_size,
            tp_size,
            device.index,
            model_scope,
            _pool_slot(block_index, pool_depth),
            pool_depth,
            max_m,
            group.group_name,
        )
        if key not in cls._pools:
            # A failure is remembered too: repeating it per layer would leak.
            cls._pools[key] = cls._build_slot(
                group, hidden_size, latent_size, device, max_m
            )
        slot = cls._pools[key]
        if slot is None:
            reason = "the rendezvous returned no multicast address"
            if not _rendezvous_failure_is_a_fault():
                return _decline(reason)
            raise RuntimeError(_fabric_fault_message(reason))
        arm_mailbox(slot.mailbox)
        return cls(slot, shard_dim, rank, max_m)

    @staticmethod
    def _verdict_key(
        group: dist.ProcessGroup,
        hidden_size: int,
        latent_size: int,
        tp_size: int,
        layer_count: int,
        pool_depth: int,
        max_m: int,
    ) -> tuple:
        """The identity a verdict and its reason are both cached under.

        Built in one place so the two cannot drift: keyed on the group alone,
        a second model on the same group would overwrite the first's reason
        and a later decline would name an unrelated one.
        """
        return (
            group.group_name,
            hidden_size,
            latent_size,
            tp_size,
            layer_count,
            pool_depth,
            max_m,
        )

    @classmethod
    def _agreed(
        cls,
        group: dist.ProcessGroup,
        hidden_size: int,
        latent_size: int,
        tp_size: int,
        layer_count: int,
        pool_depth: int,
        max_m: int,
        device: torch.device,
    ) -> bool:
        """One group-wide answer to a question every rank answers locally.

        Reachability is a local probe -- visible device count, this host's
        fabric -- so ranks can disagree, and the ones that say yes would block
        in the rendezvous while the ones that say no walk away. The vote is the
        agreement point, and the one thing above it that does not return a
        reason is the fabric lookup, which raises on a map that was never
        gathered. Ranks that raise there never reach the reduction, so if the
        map is missing on only some of them the rest wait here. Gathering
        happens once at distributed init, which is why that state is a harness
        bug rather than something to negotiate.
        """
        key = cls._verdict_key(
            group, hidden_size, latent_size, tp_size, layer_count, pool_depth, max_m
        )
        if key not in cls._verdicts:
            reason = cls._unavailable_reason(
                hidden_size, latent_size, tp_size, layer_count, group, pool_depth
            )
            # A peer's reason is not knowable here, only that it voted no.
            cls._reasons[key] = reason or "a peer declined"
            vote = torch.tensor([int(reason is None)], dtype=torch.int32, device=device)
            dist.all_reduce(vote, op=dist.ReduceOp.MIN, group=group)
            cls._verdicts[key] = bool(vote.item())
        return cls._verdicts[key]

    @classmethod
    def _agree_on_ceiling(
        cls,
        group: dist.ProcessGroup,
        max_m: int,
        tp_size: int,
        device: torch.device,
    ) -> None:
        """Refuse a group whose ranks sized the mailbox differently.

        ``_build_slot`` allocates ``(1, max_m, latent_size)`` and rendezvouses
        it, so ranks that disagree map symmetric buffers of different sizes and
        the rendezvous hangs with no traceback -- which on this fabric is
        indistinguishable from an unrelated boot hang. The pointer vote cannot
        catch it either: its key carries ``max_m``, so disagreeing ranks vote
        on different keys and each one agrees with itself.

        The check gathers rather than reduces so the error can name what every
        rank derived, which is the part that says where to look.
        """
        key = (group.group_name, max_m)
        if key in cls._ceilings:
            return
        gathered = torch.empty(tp_size, dtype=torch.int64, device=device)
        dist.all_gather_into_tensor(
            gathered,
            torch.tensor([max_m], dtype=torch.int64, device=device),
            group=group,
        )
        derived = [int(v) for v in gathered.tolist()]
        if len(set(derived)) > 1:
            raise ValueError(
                "every rank must size the down mailbox alike or the rendezvous "
                f"hangs: rank {dist.get_rank(group)} derived {max_m} rows and "
                f"the group derived {dict(enumerate(derived))} (rank in group "
                "-> rows)"
            )
        cls._ceilings[key] = True

    @classmethod
    def _agree_on_pointer(
        cls, group: dist.ProcessGroup, multicast_ptr: int, device: torch.device
    ) -> bool:
        """Whether every rank came out of the rendezvous with an address.

        Multicast support is decided per rank, so one rank holding null while
        its peers hold a live mailbox would deadlock the first narrow batch
        rather than the boot.
        """
        agreed = torch.tensor(
            [int(bool(multicast_ptr))], dtype=torch.int32, device=device
        )
        dist.all_reduce(agreed, op=dist.ReduceOp.MIN, group=group)
        return bool(agreed.item())

    @classmethod
    def _build_slot(
        cls,
        group: dist.ProcessGroup,
        hidden_size: int,
        latent_size: int,
        device: torch.device,
        max_m: int,
    ) -> _MailboxSlot | None:
        from tokenspeed_kernel.thirdparty.cute_dsl.latent_moe_tail.fused_multicast_latent_down_gemm import (  # noqa: E501
            FusedMulticastLatentDownGemmKernel,
        )
        from tokenspeed_kernel.thirdparty.cute_dsl.latent_moe_tail.lamport_copy import (
            LamportCopyKernel,
        )
        from torch.distributed import _symmetric_memory as symm_mem

        with torch.inference_mode(False), torch.no_grad():
            mailbox = symm_mem.empty(
                (1, max_m, latent_size), dtype=torch.bfloat16, device=device
            )
        handle = symm_mem.rendezvous(mailbox, group)
        multicast_ptr = handle.multicast_ptr
        if not cls._agree_on_pointer(group, multicast_ptr, device):
            return None
        arm_mailbox(mailbox)
        rank = dist.get_rank(group)
        tp_size = dist.get_world_size(group)
        shard_dim = latent_size // tp_size
        gemm_by_m: dict[int, object] = {
            m: FusedMulticastLatentDownGemmKernel(
                rank=rank,
                tp_size=tp_size,
                in_dim=hidden_size,
                latent_dim=latent_size,
                num_rows=m,
            )
            for m in range(1, min(max_m, _FUSED_MAX_M) + 1)
        }
        if max_m > _FUSED_MAX_M:
            wide = _wide_producer(
                multicast_ptr=multicast_ptr,
                rank=rank,
                shard_dim=shard_dim,
                latent_size=latent_size,
                max_m=max_m,
                device=device,
            )
            gemm_by_m.update({m: wide for m in range(_FUSED_MAX_M + 1, max_m + 1)})
        gathers: dict[tuple[int, int], object] = {}
        gather_by_m: dict[int, object] = {}
        for m in range(1, max_m + 1):
            geometry = _lamport_geometry(m)
            if geometry not in gathers:
                ctas, threads = geometry
                gathers[geometry] = LamportCopyKernel(
                    hidden_dim=latent_size,
                    max_m=max_m,
                    ctas=ctas,
                    threads=threads,
                    sentinel=_DOWN_SENTINEL,
                )
            gather_by_m[m] = gathers[geometry]
        return _MailboxSlot(mailbox, multicast_ptr, gemm_by_m, gather_by_m)

    def handles(self, num_tokens: int) -> bool:
        """Whether this op covers a batch of this width."""
        return 1 <= num_tokens <= self.max_m

    def __call__(
        self, hidden_states: torch.Tensor, weight: torch.Tensor
    ) -> torch.Tensor:
        """Project ``hidden_states`` through this rank's column block.

        Args:
            hidden_states: Contiguous ``[tokens, hidden_size]`` activation.
            weight: This rank's ``[shard_dim, hidden_size]`` block. The caller
                slices it, because a projection that narrowed its storage has
                no full width left to slice from.

        Returns:
            The full ``[tokens, latent_size]`` latent, gathered from the mailbox.
        """
        tokens = hidden_states.shape[0]
        slot = self._slot
        # Every aligned 32-bit word must transition exactly once a round, not in halves.
        # The publishing GEMM's beta must be 0, never accumulating into the mailbox.
        # Full mailbox, not a slice: the capacity guard bounds a raw-pointer
        # write and cannot see what it protects if handed only the batch.
        slot.gemm_by_m[tokens](
            hidden_states,
            weight,
            slot.mailbox,
            slot.multicast_ptr,
        )
        return slot.gather_by_m[tokens](slot.mailbox, m=tokens)[0]


def arm_mailbox(mailbox: torch.Tensor) -> None:
    """Fill a mailbox with the empty sentinel the Lamport gather spins on.

    The sentinel is four ``_DOWN_SENTINEL`` words per 128-bit fragment, which
    as BF16 pairs is (-0, -0). The tail's mailboxes keep the upstream
    0x80000000, (+0, -0): a producer that cannot be made to sanitize -0 away
    spells that one with a single coincidence, since the +0 half is a value
    real results carry all the time.
    """
    # int32 cannot hold the sentinel's top bit; fill_ takes its signed image.
    mailbox.view(torch.int32).fill_(_DOWN_SENTINEL - 0x1_0000_0000)


__all__ = ["KimiK3LatentDownOp", "arm_mailbox"]
