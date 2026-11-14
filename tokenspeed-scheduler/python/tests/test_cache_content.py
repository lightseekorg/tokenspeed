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

"""Randomized closed loops that check what every forward reads.

The scheduler's tables name the blocks each forward row reads and writes;
they say nothing about what those blocks hold. These loops replay every plan
on a model of the device, in the order the runtime runs it -- the plan's Host
stores and snapshots, its page copies and zeroing, its loads and restores,
a PD peer's landing, then its forwards -- and track what each block holds: a
State block the state after some token prefix (a zeroed one the state before
any token), each row of a History block the KV of the token at its position.
Every forward row must read its own prefix: the State its tokens continue
from (slot 0 when it holds a block, else slot 1, as
``compute_state_block_indices`` resolves it; the zero state before the first
token, as the scan always loads its initial state) and the KV of every
position below its frontier. A row the device already
stopped runs as padding and touches nothing (``masked_state_tables``).

The PD roles add what a transfer needs: the pages a Prefill rank's transfer
reads (each chunk's KV as its forward wrote it, the prompt end's State) and
the pages a Decode rank's peer writes must not change under the transfer, and
a landing must cover the prompt from the block holding its hit on.

Tokens are drawn when a request is submitted, so the device model knows every
token a forward consumes, including those of rows dispatched before the
previous result landed (overlap depth 1). Speculative rows feed several
tokens and keep a random accepted prefix; the KV of the rejected rest is
garbage.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import pytest

ts = pytest.importorskip("tokenspeed_scheduler")

P = 8
HISTORY = "full_attention"
STATES = ("linear_attention_0", "linear_attention_1")
ROUNDS = 400


class ContentError(AssertionError):
    """A forward read, or a PD transfer saw, contents other than its own."""


def _state_after(tokens) -> tuple:
    """A State block's contents: the state after `tokens`."""
    return ("state", tuple(tokens))


def _kv_of(tokens) -> tuple:
    """A History row's contents: the KV of the last of `tokens`."""
    return ("kv", tuple(tokens))


ZERO_STATE = _state_after(())


def _copied(group: str, value):
    """A block's contents as a copy leaves them in another block.

    A History block's rows by offset, or a State.
    """
    return dict(value or {}) if group == HISTORY else value


@dataclass
class _Request:
    prompt: list[int]
    # Every token it samples, drawn at submission.
    outputs: list[int]
    # Whether the PD peer bootstrapped it (a fused request needs none).
    bootstrapped: bool
    # Tokens the scheduler was told of.
    landed: int = 0
    finished: bool = False

    @property
    def max_new(self) -> int:
        return len(self.outputs)

    def tokens(self, count: int) -> list[int]:
        return (self.prompt + self.outputs)[:count]


@dataclass
class _Transfer:
    """A PD transfer's pages, by group, and when it stops using them."""

    pages: set[tuple[str, int]] = field(default_factory=set)
    # The round after which its readers or writers stopped; None while the
    # transfer runs.
    stops_after: int | None = None


class _Device:
    """What each block holds, as the runtime would leave it."""

    def __init__(self, group_names: list[str], history_block: int):
        self.group_names = group_names
        self.history_block = history_block
        self.device = {group: {} for group in group_names}
        self.host = {group: {} for group in group_names}
        self.snapshot = {group: {} for group in group_names}
        self.transfers: dict[str, _Transfer] = {}
        self.state_reads = 0
        self.kv_reads = 0

    # ---- writes, each checked against the transfers using the page ----
    def _guard(self, group: str, page: int, writer: str | None) -> None:
        for request_id, transfer in self.transfers.items():
            if request_id != writer and (group, page) in transfer.pages:
                raise ContentError(
                    f"{writer or 'a cache op'} writes {group} page {page} while "
                    f"{request_id}'s PD transfer uses it"
                )

    def put_state(self, group: str, page: int, value, writer: str | None) -> None:
        self._guard(group, page, writer)
        self.device[group][page] = value

    def put_kv(self, page: int, offset: int, value, writer: str | None) -> None:
        self._guard(HISTORY, page, writer)
        self.device[HISTORY].setdefault(page, {})[offset] = value

    def put_block(self, group: str, page: int, value) -> None:
        self._guard(group, page, None)
        self.device[group][page] = _copied(group, value)

    # ---- the plan's cache work, in the runtime's order ----
    def run_cache_ops(self, plan) -> None:
        stores = [
            op
            for op in plan.cache
            if isinstance(op, (ts.Cache.WriteBackOp, ts.Cache.SnapshotOp))
        ]
        loads = [
            op
            for op in plan.cache
            if isinstance(op, (ts.Cache.LoadBackOp, ts.Cache.RestoreOp))
        ]
        for op in stores:
            target = (
                self.host if isinstance(op, ts.Cache.WriteBackOp) else self.snapshot
            )
            for group, source, destination in self._rows(op):
                target[group][destination] = _copied(
                    group, self.device[group].get(source)
                )
        for group, pairs in plan.pages_to_copy_arrays().items():
            for source, destination in pairs:
                self.put_block(
                    group, int(destination), self.device[group].get(int(source))
                )
        for group, pages in plan.pages_to_zero.items():
            for page in pages:
                self.put_block(group, int(page), ZERO_STATE if group in STATES else {})
        for op in loads:
            if isinstance(op, ts.Cache.RestoreOp):
                rows = zip(op.group_ids, op.src_pages, op.dst_pages, op.source_tiers)
                for groups, sources, destinations, tiers in rows:
                    for group, source, destination, tier in zip(
                        groups, sources, destinations, tiers
                    ):
                        name = self.group_names[group]
                        pool = (
                            self.host if tier == ts.Cache.HostTier.L2 else self.snapshot
                        )
                        self.put_block(
                            name, int(destination), pool[name].get(int(source))
                        )
            else:
                for group, source, destination in self._rows(op):
                    self.put_block(group, destination, self.host[group].get(source))

    def _rows(self, op):
        for groups, sources, destinations in zip(
            op.group_ids, op.src_pages, op.dst_pages
        ):
            for group, source, destination in zip(groups, sources, destinations):
                yield self.group_names[group], int(source), int(destination)

    # ---- reads ----
    def check_state(self, request_id: str, group: str, row, want) -> int:
        checkpoint, live = int(row[0]), int(row[1])
        block = checkpoint if checkpoint > 0 else live
        got = self.device[group].get(block)
        self.state_reads += 1
        if got != want:
            raise ContentError(
                f"{request_id} reads {group} block {block} "
                f"(row {[int(x) for x in row]}) holding {_describe(got)}; "
                f"it continues from {_describe(want)}"
            )
        return block

    def check_kv(self, request_id: str, table, position: int, want) -> int:
        bg = self.history_block
        block = int(table[position // bg])
        got = self.device[HISTORY].get(block, {}).get(position % bg)
        self.kv_reads += 1
        if got != want:
            raise ContentError(
                f"{request_id} reads the KV of position {position} from block "
                f"{block}: {_describe(got)}; it wants {_describe(want)}"
            )
        return block


def _describe(value) -> str:
    if value is None:
        return "nothing a forward or copy put there"
    kind, tokens = value
    if kind == "state":
        return f"the state after {len(tokens)} tokens {list(tokens)[-3:]}"
    if kind == "rejected":
        request_id, position = tokens
        return f"the KV of {request_id}'s rejected draft token at position {position}"
    return f"the KV of a token at position {len(tokens) - 1} {list(tokens)[-3:]}"


def _config(rng, role, *, host_pages, policy, snapshot_pages, mixed, width):
    pages = rng.choice([24, 40, 64])
    history_block = rng.choice([P, P // 2])
    cfg = ts.SchedulerConfig()
    cfg.prefix_granularity = P
    cfg.num_device_pages = pages
    cfg.num_host_pages = host_pages
    cfg.num_snapshot_pages = snapshot_pages
    cfg.max_retracted_requests = 4 if snapshot_pages > 1 else 0
    cfg.debug_force_retraction_interval = 3 if snapshot_pages > 1 else 0
    cfg.max_scheduled_tokens = 48
    cfg.max_batch_size = 6
    cfg.enable_l3_storage = False
    cfg.disable_l2_cache = host_pages == 0
    cfg.host_write_policy = policy
    cfg.disable_prefix_cache = False
    cfg.decode_input_tokens = width
    cfg.overlap_schedule_depth = 1
    cfg.enable_mixed_prefill_decode = mixed
    cfg.role = role
    groups = [
        ts.CacheGroupConfig(
            group_id=HISTORY,
            block_granularity=history_block,
            total_pages=pages * (P // history_block),
            attn_kind=ts.AttnKind.Full,
            cache_blocks_per_lcm_block=1,
            transfer_policy=ts.CacheTransferPolicy.FullSuffix,
            shard_count=1,
        )
    ]
    groups += [
        ts.CacheGroupConfig(
            group_id=group_id,
            block_granularity=0,
            total_pages=pages,
            attn_kind=ts.AttnKind.State,
            transfer_policy=ts.CacheTransferPolicy.LatestSnapshot,
        )
        for group_id in STATES
    ]
    # K3 declares a State group before its history group.
    rng.shuffle(groups)
    cfg.cache_groups = groups
    if host_pages > 0 or snapshot_pages > 1:
        cfg.host_group_packing = [group.cache_blocks_per_lcm_block for group in groups]
    return cfg, history_block


def _event(kind: str, request_id: str, **fields):
    event = getattr(ts.ForwardEvent, kind)()
    event.request_id = request_id
    for name, value in fields.items():
        setattr(event, name, value)
    return event


class _Loop:
    """The event loop's calls, a client and a PD peer around one scheduler.

    Every plan is replayed on the device model.
    """

    def __init__(
        self,
        seed: int,
        role: str,
        *,
        host_pages: int,
        policy: str,
        snapshot_pages: int,
        mixed: bool,
        width: int,
        aborts: bool,
        abort_transfers: bool,
        short_turns: bool,
    ):
        self.rng = random.Random(seed)
        # The device's accepted counts and the client's aborts draw from
        # their own streams.
        self.accept_rng = random.Random(seed + 7)
        self.abort_rng = random.Random(seed + 1) if aborts else None
        self.role = role
        self.width = width
        # Whether the client aborts requests whose PD transfer is in flight.
        self.abort_transfers = abort_transfers
        # Whether some prompts are shorter than a page, so that a later turn
        # can hit a tail inside the first block.
        self.short_turns = short_turns
        cfg, history_block = _config(
            self.rng,
            getattr(ts.SchedulerConfig.Role, role),
            host_pages=host_pages,
            policy=getattr(ts.SchedulerConfig.HostWritePolicy, policy),
            snapshot_pages=snapshot_pages,
            mixed=mixed,
            width=width,
        )
        self.scheduler = ts.Scheduler(cfg)
        self.device = _Device(
            [group.group_id for group in cfg.cache_groups], history_block
        )
        self.requests: dict[str, _Request] = {}
        self.live: set[str] = set()
        self.done: list[list[int]] = []
        self.systems = [
            [self.rng.randrange(2, 500) for _ in range(self.rng.randrange(3, 40))]
            for _ in range(3)
        ]
        self.next_id = 0
        self.round = 0
        # Device facts per request: tokens computed, and whether the device
        # stopped it on its token limit.
        self.computed: dict[str, int] = {}
        self.stopped: set[str] = set()
        # The tokens each dispatched decode row keeps, decided as it runs.
        self.decided: dict[tuple[int, int, int], list[int]] = {}
        self.pending: list = []
        self.acks: list[tuple[int, object]] = []
        # D role: remote prefills the peer writes, and when each lands.
        self.landing: dict[str, tuple[int, list, int]] = {}
        # P role: requests whose remote decode went out, until the transfer
        # concludes.
        self.transferring: set[str] = set()

    # ---- the client ----
    def submit(self, count: int) -> None:
        specs = []
        for _ in range(count):
            user = [
                self.rng.randrange(2, 500) for _ in range(self.rng.randrange(1, 30))
            ]
            draw = self.rng.random()
            if draw < 0.15 and self.live:
                tokens = list(self.requests[self.rng.choice(sorted(self.live))].prompt)
            elif draw < 0.45 and self.done:
                tokens = list(self.rng.choice(self.done))
                if self.rng.random() < 0.5:
                    tokens += user
            elif draw < 0.6 and self.short_turns:
                tokens = user[: self.rng.randrange(1, P)]
            else:
                tokens = self.rng.choice(self.systems) + user
            max_new = self.rng.randrange(1, 10)
            limit = (
                self.scheduler.max_single_request_tokens()
                - max_new
                - 1
                - 3 * self.width
            )
            tokens = tokens[: max(1, limit)]
            request_id = f"r{self.next_id}"
            self.next_id += 1
            self.requests[request_id] = _Request(
                prompt=tokens,
                outputs=[self.rng.randrange(5000, 95000) for _ in range(max_new)],
                bootstrapped=self.role == "Fused",
            )
            self.live.add(request_id)
            spec = ts.RequestSpec()
            spec.request_id = request_id
            spec.tokens = tokens
            spec.max_new_tokens = max_new
            specs.append(spec)
        self.scheduler.submit_requests(specs)

    # ---- the device ----
    def run_device(self, plan, plan_id: int) -> None:
        self.device.run_cache_ops(plan)
        if plan.remote_prefill is not None and plan.remote_prefill.request_ids:
            self.start_landings(plan.remote_prefill)
        for index, batch in enumerate(plan.forward):
            if batch.request_ids:
                self.forward(batch, (plan_id, index))

    def forward(self, batch, key) -> None:
        tables = batch.block_tables_arrays()
        num_extends = batch.num_extends()
        bg = self.device.history_block
        state_writes: dict[tuple[str, int], str] = {}
        kv_writes: dict[tuple[int, int], str] = {}
        state_reads: list[tuple[str, int, str]] = []
        kv_reads: list[tuple[int, int, str]] = []
        pending_writes = []
        for i, request_id in enumerate(batch.request_ids):
            request = self.requests[request_id]
            extend = i < num_extends
            if not extend and request_id in self.stopped:
                continue  # a padding row
            if extend:
                before = int(batch.extend_prefix_lens[i])
                count = int(batch.input_lengths[i])
                window = count
            else:
                before = self.computed[request_id]
                window = int(batch.input_lengths[i])
                produced = before - len(request.prompt) + 1
                count = min(
                    self.accept_rng.randint(1, window), request.max_new - produced
                )
                if count <= 0:
                    raise ContentError(f"{request_id} decodes past its token limit")
                self.decided[(*key, i)] = request.outputs[produced : produced + count]
            sequence = request.tokens(before + count)
            for group in STATES:
                row = tables[group][i]
                want = _state_after(request.tokens(before))
                block = self.device.check_state(request_id, group, row, want)
                state_reads.append((group, block, request_id))
                live = int(row[1])
                if live <= 0:
                    raise ContentError(
                        f"{request_id} has no live {group} block: "
                        f"{[int(x) for x in row]}"
                    )
                owner = state_writes.setdefault((group, live), request_id)
                if owner != request_id:
                    raise ContentError(
                        f"{owner} and {request_id} write {group} block {live}"
                    )
                pending_writes.append(
                    ("state", group, live, _state_after(sequence), request_id)
                )
            table = tables[HISTORY][i]
            for position in range(before):
                block = self.device.check_kv(
                    request_id, table, position, _kv_of(request.tokens(position + 1))
                )
                kv_reads.append((block, position % bg, request_id))
            for position in range(before, before + window):
                slot = position // bg
                block = int(table[slot]) if slot < table.shape[0] else 0
                if block <= 0:
                    if position >= before + count:
                        continue  # a rejected draft past the extent
                    raise ContentError(
                        f"{request_id} writes the KV of position {position} to "
                        f"block {block}"
                    )
                owner = kv_writes.setdefault((block, position % bg), request_id)
                if owner != request_id:
                    raise ContentError(
                        f"{owner} and {request_id} write block {block} row "
                        f"{position % bg}"
                    )
                value = (
                    _kv_of(request.tokens(position + 1))
                    if position < before + count
                    else ("rejected", (request_id, position))
                )
                pending_writes.append(
                    ("kv", HISTORY, (block, position % bg), value, request_id)
                )
            if self.role == "P":
                self.export_chunk(request_id, tables, i, before, count)
            self.computed[request_id] = before + count
            emits = not extend or before + count >= len(request.prompt)
            if emits and before + count - len(request.prompt) + 1 >= request.max_new:
                self.stopped.add(request_id)
        for group, block, request_id in state_reads:
            owner = state_writes.get((group, block))
            if owner is not None and owner != request_id:
                raise ContentError(
                    f"{request_id} reads {group} block {block} as {owner} writes it"
                )
        for block, offset, request_id in kv_reads:
            owner = kv_writes.get((block, offset))
            if owner is not None and owner != request_id:
                raise ContentError(
                    f"{request_id} reads block {block} row {offset} as {owner} "
                    f"writes it"
                )
        for kind, group, where, value, request_id in pending_writes:
            if kind == "state":
                self.device.put_state(group, where, value, request_id)
            else:
                self.device.put_kv(*where, value, request_id)

    def export_chunk(
        self, request_id: str, tables, i: int, before: int, count: int
    ) -> None:
        """P role: adds the pages the transfer reads to it.

        The transfer reads each chunk's KV once its forward wrote it
        (layerwise), and the State at the prompt end.
        """
        transfer = self.device.transfers.setdefault(request_id, _Transfer())
        bg = self.device.history_block
        table = tables[HISTORY][i]
        for slot in range(before // bg, (before + count - 1) // bg + 1):
            transfer.pages.add((HISTORY, int(table[slot])))
        if before + count >= len(self.requests[request_id].prompt):
            for group in STATES:
                transfer.pages.add((group, int(tables[group][i][1])))

    # ---- the PD peer ----
    def start_landings(self, batch) -> None:
        """D role: the peer starts writing a remote prefill's pages.

        It lands zero to two rounds later.
        """
        tables = batch.block_tables_arrays()
        cached = list(batch.state_prefix_cached) or [False] * len(batch.request_ids)
        bg = self.device.history_block
        for i, request_id in enumerate(batch.request_ids):
            request = self.requests[request_id]
            hit = int(batch.extend_prefix_lens[i])
            table = tables[HISTORY][i]
            transfer = _Transfer()
            writes = []
            for position in range((hit // bg) * bg, len(request.prompt)):
                block = int(table[position // bg])
                if block <= 0:
                    raise ContentError(
                        f"{request_id}'s landing has no block for the KV of position "
                        f"{position} (hit {hit}, row {[int(x) for x in table]})"
                    )
                transfer.pages.add((HISTORY, block))
                writes.append(
                    ("kv", block, position % bg, _kv_of(request.tokens(position + 1)))
                )
            for group in STATES:
                row = tables[group][i]
                if cached[i]:
                    want = _state_after(request.prompt)
                    self.device.check_state(request_id, group, row, want)
                else:
                    transfer.pages.add((group, int(row[1])))
                    writes.append(
                        ("state", group, int(row[1]), _state_after(request.prompt))
                    )
            self.device.transfers[request_id] = transfer
            self.landing[request_id] = (
                self.round + self.rng.randrange(3),
                writes,
                len(request.prompt),
            )

    def peer(self) -> None:
        events = ts.ExecutionEvent()
        count = 0
        for request_id, transfer in list(self.device.transfers.items()):
            if transfer.stops_after is not None and transfer.stops_after < self.round:
                del self.device.transfers[request_id]
        for request_id in sorted(self.live):
            request = self.requests[request_id]
            if not request.bootstrapped and self.rng.random() < 0.7:
                events.add_event(ts.PD.BootstrappedEvent(request_id))
                request.bootstrapped = True
                count += 1
        for request_id, (due, writes, prompt_len) in sorted(self.landing.items()):
            if due > self.round:
                continue
            del self.landing[request_id]
            del self.device.transfers[request_id]
            for write in writes:
                if write[0] == "kv":
                    self.device.put_kv(write[1], write[2], write[3], request_id)
                else:
                    self.device.put_state(write[1], write[2], write[3], request_id)
            self.computed[request_id] = prompt_len
            request = self.requests[request_id]
            events.add_event(
                ts.PD.RemotePrefillDoneEvent(request_id, request.outputs[0])
            )
            request.landed = 1
            count += 1
            if request.max_new == 1:
                self.stopped.add(request_id)
        for request_id in sorted(self.transferring):
            if self.rng.random() < 0.5:
                self.transferring.discard(request_id)
                failed = self.rng.random() < 0.1
                events.add_event(
                    (ts.PD.FailedEvent if failed else ts.PD.SucceededEvent)(request_id)
                )
                self.requests[request_id].finished = True
                if failed:
                    # A failed transfer's readers stop a little later.
                    self.device.transfers[request_id].stops_after = (
                        self.round + self.rng.randrange(3)
                    )
                else:
                    del self.device.transfers[request_id]
                count += 1
        if count:
            self.scheduler.advance(events)

    # ---- results ----
    def land(self, plan_id: int, plan) -> None:
        events = ts.ExecutionEvent()
        finishes = []
        for index, batch in enumerate(plan.forward):
            num_extends = batch.num_extends()
            for i, request_id in enumerate(batch.request_ids):
                request = self.requests[request_id]
                decode = i >= num_extends
                if not decode and (
                    batch.extend_prefix_lens[i] + batch.input_lengths[i]
                    < batch.prefill_lengths[i]
                ):
                    events.add_event(_event("ExtendResult", request_id, tokens=[]))
                    continue
                if decode:
                    tokens = self.decided.pop((plan_id, index, i), [])
                else:
                    tokens = [request.outputs[0]] if request.landed == 0 else []
                events.add_event(_event("ExtendResult", request_id, tokens=tokens))
                request.landed += len(tokens)
                if not tokens or request.finished or request_id not in self.live:
                    continue
                if self.role == "P":
                    continue  # the transfer ends a P request
                if request.landed >= request.max_new or self.rng.random() < 0.03:
                    request.finished = True
                    finishes.append(request_id)
                elif decode:
                    events.add_event(
                        _event(
                            "UpdateReserveNumTokens",
                            request_id,
                            reserve_num_tokens_in_next_schedule_event=self.width,
                        )
                    )
        for request_id in finishes:
            events.add_event(
                _event("Abort" if self.rng.random() < 0.1 else "Finish", request_id)
            )
        self.scheduler.advance(events)

    def queue_acks(self, plan) -> None:
        kinds = (
            (ts.Cache.WriteBackOp, ts.Cache.WriteBackDoneEvent),
            (ts.Cache.LoadBackOp, ts.Cache.LoadBackDoneEvent),
            (ts.Cache.SnapshotOp, ts.Cache.SnapshotDoneEvent),
            (ts.Cache.RestoreOp, ts.Cache.RestoreDoneEvent),
        )
        for op in plan.cache:
            for kind, done in kinds:
                if isinstance(op, kind):
                    for op_id in op.op_ids:
                        ack = done()
                        ack.op_id = op_id
                        self.acks.append((self.round + self.rng.randrange(4), ack))

    def acknowledge(self) -> None:
        due = [ack for when, ack in self.acks if when <= self.round]
        self.acks = [(when, ack) for when, ack in self.acks if when > self.round]
        self.rng.shuffle(due)
        for ack in due:
            events = ts.ExecutionEvent()
            events.add_event(ack)
            self.scheduler.advance(events)

    def end(self, request_id: str) -> None:
        """Drops a request that was aborted.

        Its PD transfer, if any, learns of the abort and stops a little later.
        """
        self.live.discard(request_id)
        self.transferring.discard(request_id)
        self.landing.pop(request_id, None)
        transfer = self.device.transfers.get(request_id)
        if transfer is not None and transfer.stops_after is None:
            transfer.stops_after = self.round + self.rng.randrange(3)

    def abort_one(self) -> None:
        abortable = sorted(
            r
            for r in self.live
            if not self.requests[r].finished
            and (self.abort_transfers or r not in self.device.transfers)
        )
        if not abortable or self.abort_rng.random() >= 0.05:
            return
        request_id = self.abort_rng.choice(abortable)
        events = ts.ExecutionEvent()
        events.add_event(_event("Abort", request_id))
        self.scheduler.advance(events)
        self.end(request_id)

    def run(self, rounds: int = ROUNDS) -> None:
        self.submit(self.rng.randrange(2, 6))
        for plan_id in range(rounds):
            self.round = plan_id
            if self.rng.random() < 0.3:
                self.submit(self.rng.randrange(1, 4))
            if self.role != "Fused":
                self.peer()
            plan = self.scheduler.next_execution_plan()
            for abort in plan.aborts:
                self.end(abort.request_id)
            self.run_device(plan, plan_id)
            if plan.remote_decode is not None:
                self.transferring.update(plan.remote_decode.request_ids)
            self.queue_acks(plan)
            self.pending.append((plan_id, plan))
            if self.abort_rng is not None:
                self.abort_one()
            if len(self.pending) > 1:
                self.land(*self.pending.pop(0))
            if self.rng.random() < 0.1:
                while self.pending:
                    self.land(*self.pending.pop(0))
            for request_id in sorted(r for r in self.live if self.requests[r].finished):
                if self.scheduler.request_token_size(request_id) == -1:
                    self.live.discard(request_id)
                    request = self.requests[request_id]
                    sequence = request.tokens(len(request.prompt) + request.landed - 1)
                    if len(self.done) < 20:
                        self.done.append(sequence)
                    else:
                        self.done[self.rng.randrange(20)] = sequence
            self.acknowledge()


def _run(role: str, seeds: int, **knobs) -> None:
    """Runs one loop per seed; the first wrong read fails the test."""
    state_reads = kv_reads = 0
    for seed in range(seeds):
        loop = _Loop(seed, role, **knobs)
        try:
            loop.run()
        except ContentError as error:
            raise ContentError(f"seed {seed}, round {loop.round}: {error}") from None
        state_reads += loop.device.state_reads
        kv_reads += loop.device.kv_reads
    assert state_reads > 0 and kv_reads > 0, "the loops read cached contents"


# Host L2 under both policies, retraction into a snapshot pool, mixed
# batches, speculative verify of two and three tokens, and short turns.
@pytest.mark.parametrize(
    "host_pages, policy, snapshot_pages, mixed, width, short_turns",
    [
        (0, "WriteThrough", 1, False, 1, True),
        (0, "WriteThrough", 1, True, 3, False),
        (48, "WriteThrough", 1, True, 1, True),
        (48, "WriteBack", 1, False, 2, False),
        (48, "WriteBack", 24, True, 1, True),
        (0, "WriteThrough", 24, False, 3, False),
    ],
)
def test_fused_forwards_read_their_own_prefixes(
    host_pages: int,
    policy: str,
    snapshot_pages: int,
    mixed: bool,
    width: int,
    short_turns: bool,
) -> None:
    _run(
        "Fused",
        20,
        host_pages=host_pages,
        policy=policy,
        snapshot_pages=snapshot_pages,
        mixed=mixed,
        width=width,
        aborts=True,
        abort_transfers=True,
        short_turns=short_turns,
    )


@pytest.mark.parametrize(
    "abort_transfers, short_turns",
    [
        (False, False),
        pytest.param(
            True,
            False,
            marks=pytest.mark.xfail(
                strict=True,
                raises=ContentError,
                reason="Known bug: an abort frees a RemotePrefilling request's "
                "pages while the peer still writes them.",
            ),
            id="aborts-during-landing",
        ),
        pytest.param(
            False,
            True,
            marks=pytest.mark.xfail(
                strict=True,
                raises=ContentError,
                reason="Known bug: a remote admission whose hit is a tail inside "
                "the history table's first block exports the null block for it.",
            ),
            id="first-block-tails",
        ),
    ],
)
def test_decode_role_landings_and_forwards_read_their_own_prefixes(
    abort_transfers: bool, short_turns: bool
) -> None:
    _run(
        "D",
        60,
        host_pages=0,
        policy="WriteThrough",
        snapshot_pages=24,
        mixed=True,
        width=2,
        aborts=True,
        abort_transfers=abort_transfers,
        short_turns=short_turns,
    )


@pytest.mark.xfail(
    strict=True,
    raises=ContentError,
    reason="Known bugs: prefix publication dedupes a chunk's pages onto another "
    "request's blocks, and an abort or a failed transfer frees the request's "
    "pages, while the PD transfer still reads them.",
)
def test_prefill_role_transfers_keep_their_pages() -> None:
    _run(
        "P",
        20,
        host_pages=0,
        policy="WriteThrough",
        snapshot_pages=1,
        mixed=False,
        width=1,
        aborts=True,
        abort_transfers=True,
        short_turns=True,
    )


def _scripted(role: str, groups, block_granularity: int) -> "ts.Scheduler":
    cfg = ts.SchedulerConfig()
    cfg.prefix_granularity = P
    cfg.num_device_pages = 24
    cfg.num_host_pages = 0
    cfg.num_snapshot_pages = 1
    cfg.max_retracted_requests = 0
    cfg.max_scheduled_tokens = 48
    cfg.max_batch_size = 6
    cfg.enable_l3_storage = False
    cfg.disable_l2_cache = True
    cfg.host_write_policy = ts.SchedulerConfig.HostWritePolicy.WriteThrough
    cfg.disable_prefix_cache = False
    cfg.decode_input_tokens = 1
    cfg.overlap_schedule_depth = 1
    cfg.enable_mixed_prefill_decode = True
    cfg.role = getattr(ts.SchedulerConfig.Role, role)
    cfg.cache_groups = [
        ts.CacheGroupConfig(
            group_id=group,
            block_granularity=block_granularity if group == HISTORY else 0,
            total_pages=24 * (P // block_granularity) if group == HISTORY else 24,
            attn_kind=ts.AttnKind.Full if group == HISTORY else ts.AttnKind.State,
            transfer_policy=(
                ts.CacheTransferPolicy.FullSuffix
                if group == HISTORY
                else ts.CacheTransferPolicy.LatestSnapshot
            ),
        )
        for group in groups
    ]
    return ts.Scheduler(cfg)


def _submit(scheduler, request_id: str, tokens: list[int], max_new: int) -> None:
    spec = ts.RequestSpec()
    spec.request_id = request_id
    spec.tokens = tokens
    spec.max_new_tokens = max_new
    scheduler.submit_requests([spec])


def test_a_finish_behind_a_masked_first_decode_publishes_the_prompt_end_state() -> None:
    """B's finish behind its padded first decode publishes the prompt-end state.

    Two requests prefill the same prompt; B stops on its first token, so its
    first decode, already dispatched behind the prefill, runs as padding and
    writes nothing. The prompt-end checkpoint B publishes at its finish must
    hold the prompt-end state even after A's copy of it was evicted, as C
    resumes from it.
    """
    scheduler = _scripted("Fused", [STATES[1], HISTORY, STATES[0]], P)
    prompt = [8, 245, 288, 266, 420, 388, 449, 50, 287, 25, 80, 185]
    holds = {group: {} for group in STATES}
    stopped = set()

    def run_device(plan) -> dict:
        for group, pages in plan.pages_to_zero.items():
            for page in pages if group in holds else ():
                holds[group][int(page)] = ZERO_STATE
        for group, pairs in plan.pages_to_copy_arrays().items():
            for source, destination in pairs if group in holds else ():
                holds[group][int(destination)] = holds[group].get(int(source))
        reads = {}
        for batch in plan.forward:
            tables = batch.block_tables_arrays()
            for i, request_id in enumerate(batch.request_ids):
                extend = i < batch.num_extends()
                if not extend and request_id in stopped:
                    continue  # a padding row
                for group in STATES:
                    checkpoint, live = (int(x) for x in tables[group][i][:2])
                    before = int(batch.extend_prefix_lens[i]) if extend else len(prompt)
                    if before > 0:
                        reads[(request_id, group)] = holds[group].get(
                            checkpoint if checkpoint > 0 else live
                        )
                    after = before + (int(batch.input_lengths[i]) if extend else 1)
                    holds[group][live] = _state_after((prompt + [0] * after)[:after])
        return reads

    def land(*results, finish=None) -> None:
        events = ts.ExecutionEvent()
        for request_id, tokens in results:
            events.add_event(_event("ExtendResult", request_id, tokens=tokens))
        if finish is not None:
            events.add_event(_event("Finish", finish))
        scheduler.advance(events)

    _submit(scheduler, "A", prompt, 7)
    _submit(scheduler, "B", prompt, 1)
    run_device(scheduler.next_execution_plan())
    stopped.add("B")  # its first token reaches its limit on the device
    run_device(scheduler.next_execution_plan())
    land(("A", [5001]), ("B", [5002]), finish="B")
    # A 41-token prompt evicts what nothing pins.
    _submit(scheduler, "R", list(range(100, 141)), 1)
    run_device(scheduler.next_execution_plan())
    land(("A", [5003]), ("B", []), ("R", [5004]))  # B's padding row keeps nothing
    _submit(scheduler, "C", prompt + [219], 1)
    reads = run_device(scheduler.next_execution_plan())
    for group in STATES:
        assert reads.get(("C", group)) == _state_after(prompt), (
            f"C resumes at the prompt end from a {group} block holding "
            f"{_describe(reads.get(('C', group)))}"
        )


@pytest.mark.xfail(
    strict=True,
    reason="Known bug: the D role's first-block tail hole is exported as the "
    "null block (reclaimed_prefix_blocks_ is not lowered when the slot is "
    "filled).",
)
@pytest.mark.parametrize(
    "block_granularity, first_len, extra",
    [(8, 6, 1), (8, 3, 5), (4, 3, 2)],
)
def test_a_tail_hit_in_the_first_block_lands_in_a_real_block(
    block_granularity: int, first_len: int, extra: int
) -> None:
    """A landing whose hit is a tail inside the first block names that block.

    A D request continues a sequence whose earlier turn finished inside the
    history table's first block: the peer writes the block holding the hit
    whole, so the landing table must name a real block for it.
    """
    scheduler = _scripted("D", [HISTORY, STATES[0]], block_granularity)
    prompt = list(range(100, 100 + first_len))

    def remote_admission(request_id: str, tokens: list[int], max_new: int):
        _submit(scheduler, request_id, tokens, max_new)
        events = ts.ExecutionEvent()
        events.add_event(ts.PD.BootstrappedEvent(request_id))
        scheduler.advance(events)
        plan = scheduler.next_execution_plan()
        assert plan.remote_prefill is not None
        assert list(plan.remote_prefill.request_ids) == [request_id]
        return plan.remote_prefill

    # A lands and finishes on its bootstrap token, publishing a tail inside
    # the first page.
    remote_admission("A", prompt, 1)
    events = ts.ExecutionEvent()
    events.add_event(ts.PD.RemotePrefillDoneEvent("A", 7))
    scheduler.advance(events)
    events = ts.ExecutionEvent()
    events.add_event(_event("Finish", "A"))
    scheduler.advance(events)
    scheduler.next_execution_plan()
    remote = remote_admission("B", prompt + list(range(900, 900 + extra)), 4)
    hit = int(remote.extend_prefix_lens[0])
    assert 0 < hit < block_granularity, "the hit lies in the first block"
    table = [int(x) for x in remote.block_tables_arrays()[HISTORY][0]]
    assert table[hit // block_granularity] > 0, f"the landing table is {table}"
