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

"""L3 admission probes and prefetch recovery for the scheduler event loop.

The scheduler and DeviceHandle are explicit dependencies; no live loop state
is needed. DeviceHandle owns storage operations and per-plan prefetch result
capture. These hooks only coordinate Host-tier probes and their replica-wide
outcomes, returning recovery events for the loop's single tail advance.
``device=None`` disables L3 work while preserving ordinary request admission.
"""

from __future__ import annotations

import logging

import torch
import torch.distributed as dist

from tokenspeed.runtime.engine.scheduler_utils import make_retract_event

logger = logging.getLogger(__name__)


class L3CacheHooks:
    """Coordinate L3 prefix admission and recovery without a loop reference."""

    def __init__(
        self,
        scheduler,
        device,
        *,
        attn_tp_size: int,
        attn_tp_cpu_group,
        attn_cp_size: int,
        attn_cp_cpu_group,
        pp_size: int,
        pp_cpu_group,
    ) -> None:
        self._scheduler = scheduler
        self._device = device
        # Same replica order as L2 completion tracking. DP ranks have different
        # requests and never participate in these per-prefix decisions.
        self._replica_groups = [
            group
            for size, group in (
                (attn_tp_size, attn_tp_cpu_group),
                (attn_cp_size, attn_cp_cpu_group),
                (pp_size, pp_cpu_group),
            )
            if size > 1 and group is not None
        ]

    def submit_requests(self, specs) -> None:
        """Register reusable L3 pages, then submit specs through the scheduler."""
        self._register_l3_storage_hits(specs)
        self._scheduler.submit_requests(specs)

    def revalidate_queued_hits(self) -> None:
        """Drop L3 keys that vanished while a request waited for capacity.

        ``_register_l3_storage_hits`` runs at submit. A queued request can
        sit past a later ``batch_exists`` miss (delete, eviction, lost
        object). Re-probe admission candidates immediately before
        ``next_execution_plan`` so Admit cannot treat a stale scheduler key
        as a Host hit and then ``batch_get_into`` a missing object.
        ``waiting_prefix_hashes`` is only the Submitted/Retracted work that
        can take a batch slot and Device pages this round, so a long waiter
        is not hashed and remotely probed on every decode step.
        """

        if self._device is None:
            return
        self._sync_l3_storage_keys(self._scheduler.waiting_prefix_hashes())

    def _register_l3_storage_hits(self, specs) -> None:
        """Tell the scheduler which prefix pages already live in L3.

        Cross-instance reuse cannot see Mooncake objects through the Host
        index. Probe them with the same content hashes the scheduler will
        use, then register only keys every cache-owning rank in the replica
        agrees exist.

        Skipped when L3 is unset: hashing the full token list is not free,
        and --disable-kvstore admit still goes through this helper.
        """

        if not specs or self._device is None:
            return
        hashes = []
        seen: set[str] = set()
        for spec in specs:
            tokens = spec.tokens
            if not isinstance(tokens, list):
                tokens = list(tokens)
            for content_hash in self._scheduler.prefix_hashes_for_tokens(tokens):
                if content_hash in seen:
                    continue
                seen.add(content_hash)
                hashes.append(content_hash)
        self._sync_l3_storage_keys(hashes)

    def _sync_l3_storage_keys(self, hashes: list[str]) -> None:
        if not hashes:
            return
        group_ids, content_hashes, page_offsets = self._scheduler.expand_prefix_keys(
            hashes
        )
        pages = [
            (int(group_id), 0, content_hash, int(page_offset))
            for group_id, content_hash, page_offset in zip(
                group_ids, content_hashes, page_offsets
            )
        ]
        local_exists = self._l3_exists_or_miss(pages, expected_len=len(group_ids))
        local_readable = [
            present
            and not self._device.l3_key_is_unread(
                group_id=int(group_id),
                content_hash=str(content_hash),
                page_offset=int(page_offset),
            )
            for group_id, content_hash, page_offset, present in zip(
                group_ids, content_hashes, page_offsets, local_exists
            )
        ]
        exists = self._converge_l3_exists(local_readable)
        hit_groups = []
        hit_hashes = []
        hit_offsets = []
        miss_groups = []
        miss_hashes = []
        miss_offsets = []
        for group_id, content_hash, page_offset, present in zip(
            group_ids, content_hashes, page_offsets, exists
        ):
            target = (
                (hit_groups, hit_hashes, hit_offsets)
                if present
                else (miss_groups, miss_hashes, miss_offsets)
            )
            target[0].append(int(group_id))
            target[1].append(content_hash)
            target[2].append(int(page_offset))
        if miss_groups:
            self._scheduler.unregister_storage_keys(
                miss_groups, miss_hashes, miss_offsets
            )
        if hit_groups:
            self._scheduler.register_storage_keys(hit_groups, hit_hashes, hit_offsets)

    def prepare_forward(self, execution_plan, forward_op) -> tuple:
        """Return the safe forward and retract events for this execution plan.

        A replica-wide prefetch miss suppresses the whole model batch. The
        caller must still execute the plan's cache ops so LoadBackDone unpins
        the destinations without publishing empty Host pages, but withhold
        remote prefills when retracts are returned. Commit any older in-flight
        forwards before applying these retracts at the loop's tail advance.
        """
        retracts = self._recover_if_l3_prefetch_failed(execution_plan, forward_op)
        return (None if retracts else forward_op), retracts

    def _recover_if_l3_prefetch_failed(self, execution_plan, forward_op) -> list:
        """Drop vanished L3 keys and retract the batch so the next admit computes.

        ``batch_exists`` is not a lease. After Admit, ``batch_get_into`` can
        still miss. Prefetch on this control-plane turn, MIN-reduce across
        the replica, then skip H2D / skip publishing empty Host pages and
        retract (snapshot-less) so the next admit recomputes those tokens.
        A backend exception or malformed result is a local miss so every
        rank still enters the MIN-reduce; raising would hang healthy peers.
        Failed keys stay unread: a later ``batch_exists`` hit must not
        re-register them and retry the same prefetch. Only pages whose
        replica-converged ``batch_get_into`` missed are blacklisted;
        successfully restored leading pages stay readable. Replica admission
        MIN-reduces local readability so one rank cannot re-register a key
        while a peer still blacklists it. A later Host backup forgets an
        unread entry only when this put created a missing object. The whole
        forward is skipped so ranks stay aligned; mixed prefill/decode
        partners retract rather than finish with an error. D-role admit
        rides ``plan.remote_prefill`` with no local forward: those request
        ids retract on the same path, and the loop withholds that stream
        from ``DeviceHandle.execute`` so the peer does not land suffix-only
        KV on empty prefix pages.
        """

        if self._device is None:
            return []
        if not self._device.plan_has_l3_prefetch(execution_plan):
            return []
        groups, hashes, offsets = self._device.l3_prefetch_storage_keys(execution_plan)
        local_ok = self._l3_prefetch_ok_or_miss(
            execution_plan, expected_len=len(groups)
        )
        ok = self._converge_l3_exists(local_ok)
        if all(ok):
            return []
        self._device.invalidate_l3_prefetch()
        failed_groups = []
        failed_hashes = []
        failed_offsets = []
        for group_id, content_hash, page_offset, present in zip(
            groups, hashes, offsets, ok
        ):
            if present:
                continue
            failed_groups.append(int(group_id))
            failed_hashes.append(str(content_hash))
            failed_offsets.append(int(page_offset))
        if failed_groups:
            self._device.mark_l3_keys_unread(
                groups=failed_groups, hashes=failed_hashes, offsets=failed_offsets
            )
            self._scheduler.unregister_storage_keys(
                failed_groups, failed_hashes, failed_offsets
            )
        retracted = []
        seen = set()

        def _retract(request_ids) -> None:
            for rid in request_ids:
                if rid in seen:
                    continue
                seen.add(rid)
                retracted.append(make_retract_event(rid))

        if forward_op is not None:
            _retract(forward_op.request_ids)
        remote_prefill = execution_plan.remote_prefill
        if remote_prefill is not None:
            _retract(remote_prefill.request_ids)
        logger.warning(
            f"L3 prefetch missed after admit; unregistered {len(failed_groups)} "
            f"key(s) and retracted {len(retracted)} request(s) for recompute"
        )
        return retracted

    def _l3_exists_or_miss(self, pages, *, expected_len: int) -> list[bool]:
        """Probe L3 without skipping the replica MIN-reduce on a local fault.

        A backend exception or a malformed result becomes an all-miss vector
        of ``expected_len`` so every cache-owning rank still enters
        ``_converge_l3_exists``. Raising here would leave peers blocked in
        that collective.
        """

        try:
            exists = self._device.query_l3_storage(pages)
        except Exception:
            logger.exception(
                "L3 existence probe failed; treating keys as misses so replica "
                "ranks can converge"
            )
            return [False] * expected_len
        if exists is None or len(exists) != expected_len:
            if exists is not None:
                logger.error(
                    "L3 existence result is not aligned with cache keys: "
                    f"ok_flags={len(exists)} keys={expected_len}"
                )
            return [False] * expected_len
        return exists

    def _l3_prefetch_ok_or_miss(
        self, execution_plan, *, expected_len: int
    ) -> list[bool]:
        """Prefetch Host pages from L3, or miss every key if the RPC faults.

        Peers must still enter ``_converge_l3_exists``. A raised
        ``batch_get_into`` on one rank would otherwise hang the replica.
        A malformed per-page vector becomes an all-miss of ``expected_len``.
        """

        try:
            flags = self._device.prefetch_l3_load_backs(execution_plan)
        except Exception:
            logger.exception(
                "L3 prefetch RPC failed; treating as a miss so replica ranks "
                "can converge"
            )
            return [False] * expected_len
        if flags is None or len(flags) != expected_len:
            if flags is not None:
                logger.error(
                    "L3 prefetch result is not aligned with cache keys: "
                    f"ok_flags={len(flags)} keys={expected_len}"
                )
            return [False] * expected_len
        return [bool(flag) for flag in flags]

    def _converge_l3_exists(self, exists: list[bool]) -> list[bool]:
        """MIN-reduce L3 exists across every cache-owning rank in this replica.

        Cache-owning ranks share a DP replica (attention TP × CP × PP). They
        must admit the same prefix pages or later PP/CP collectives hang.
        DP ranks hold different sequences and are not reduced. Order is
        TP, then CP, then PP so every rank enters the same sequence of
        groups.
        """

        if not self._replica_groups:
            return exists
        flags = torch.tensor(
            [1 if present else 0 for present in exists], dtype=torch.int32
        )
        for group in self._replica_groups:
            dist.all_reduce(flags, op=dist.ReduceOp.MIN, group=group)
        return [bool(flag) for flag in flags.tolist()]
