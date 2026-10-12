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

"""L3 probes and prefetch convergence for the scheduler event loop.

Two touch points, both replica-wide decisions and nothing else:

* at submit, ``batch_exists`` tells the scheduler which prefix keys L3 holds
  (``_register_l3_storage_hits``), MIN-reduced across the replica so every
  mirrored scheduler registers the same keys;
* each round, the in-flight L3 prefetch ops -- a waiting request's Host pages
  being filled from L3 by the device's lane, before the request is admitted --
  are MIN-reduced across the replica (done flag and landed prefix), and a
  converged op is completed on the device, which then acknowledges it once as
  ``PrefetchDoneEvent(op_id, landed_pages)`` through the ordinary cache poll.

Only the L3-to-Host leg can miss, and it runs before admission, so nothing is
ever retracted or skipped because of L3: a short landing is a shorter Host
hit. The scheduler and DeviceHandle are explicit dependencies; no live loop
state is needed. ``device=None`` disables L3 work while preserving ordinary
request admission.
"""

from __future__ import annotations

import logging

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

# "Not landed yet" in the MIN-reduced landed vector: any rank still fetching
# keeps the replica's landed prefix unknown, so the sentinel dominates.
_UNLANDED = (1 << 31) - 1


class L3CacheHooks:
    """Coordinate the replica-wide L3 decisions without a loop reference."""

    def __init__(
        self,
        scheduler,
        device,
        *,
        attn_tp_size: int,
        attn_tp_cpu_group,
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
                (pp_size, pp_cpu_group),
            )
            if size > 1 and group is not None
        ]

    def submit_requests(self, specs) -> None:
        """Register reusable L3 pages, then submit specs through the scheduler."""
        self._register_l3_storage_hits(specs)
        self._scheduler.submit_requests(specs)

    def _register_l3_storage_hits(self, specs) -> None:
        """Tell the scheduler which prefix pages already live in L3.

        Cross-instance reuse cannot see L3 objects through the Host index.
        Probe them with the same content hashes the scheduler will use, then
        register only keys every cache-owning rank in the replica agrees
        exist. The scheduler prefetches those keys into Host pages before it
        admits the request; a key that vanished in between lands short and
        is forgotten by the scheduler then (no blacklist: a later probe may
        find it again).

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
        exists = self._converge_l3_exists(local_exists)
        hit_groups = []
        hit_hashes = []
        hit_offsets = []
        for group_id, content_hash, page_offset, present in zip(
            group_ids, content_hashes, page_offsets, exists
        ):
            if not present:
                continue
            hit_groups.append(int(group_id))
            hit_hashes.append(content_hash)
            hit_offsets.append(int(page_offset))
        if hit_groups:
            self._scheduler.register_storage_keys(hit_groups, hit_hashes, hit_offsets)

    def converge_prefetches(self) -> None:
        """Agree across the replica on every in-flight L3 prefetch, once a round.

        The set of in-flight op ids is mirrored (every rank submitted the same
        plans), so every rank reduces the same vector in op-id order: per op
        its done flag and, once done, the pages it landed -- a prefix length,
        so the replica MIN is the common prefix every rank holds. A rank whose
        lane raised reports done with nothing landed. An op converged done is
        completed on the device with the replica's landed count; the device
        acknowledges it once on the next cache poll. With no op in flight
        anywhere the round makes no collective.
        """

        if self._device is None:
            return
        progress = self._device.l3_prefetch_progress()
        if not progress:
            return
        op_ids = sorted(progress)
        local: list[int] = []
        for op_id in op_ids:
            done, landed = progress[op_id]
            local.extend((1 if done else 0, int(landed) if done else _UNLANDED))
        reduced = self._converge_min(local)
        for index, op_id in enumerate(op_ids):
            done, landed = reduced[2 * index], reduced[2 * index + 1]
            if not done:
                continue
            self._device.complete_l3_prefetch(op_id, landed)
            if landed < progress[op_id][1]:
                logger.warning(
                    f"L3 prefetch {op_id}: a replica peer landed {landed} pages, this "
                    f"rank {progress[op_id][1]}; the common prefix is admitted"
                )

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

    def _converge_l3_exists(self, exists: list[bool]) -> list[bool]:
        """MIN-reduce L3 exists across every cache-owning rank in this replica.

        Cache-owning ranks share a DP replica (attention TP x CP x PP). They
        must admit the same prefix pages or later PP/CP collectives hang.
        DP ranks hold different sequences and are not reduced.
        """

        return [
            bool(flag)
            for flag in self._converge_min([1 if present else 0 for present in exists])
        ]

    def _converge_min(self, values: list[int]) -> list[int]:
        """MIN-reduce an int32 vector over the replica groups, TP then CP then PP.

        Every rank enters the same sequence of groups with a vector of the
        same length, which the callers guarantee by deriving it from mirrored
        scheduler state.
        """

        if not self._replica_groups:
            return list(values)
        flags = torch.tensor(values, dtype=torch.int32)
        for group in self._replica_groups:
            dist.all_reduce(flags, op=dist.ReduceOp.MIN, group=group)
        return flags.tolist()
