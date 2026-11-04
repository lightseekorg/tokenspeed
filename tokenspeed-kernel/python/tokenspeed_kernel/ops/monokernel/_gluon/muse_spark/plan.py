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

"""The Muse Spark layer's roles, as layer.py composes them, as a plan.

:func:`muse_plan` builds the step sequence of every workgroup ``(xcd, slot)``
for one :class:`PlanKey`, following ``gluon_monokernel_muse_spark_gfx950``
role for role; :func:`expected_counts` is the per-XCD role table written
apart from the builder, so the checker compares two descriptions. Role names
and exchanges (SENDER, AR_OWNER, ..., AR, POOLED_AG, ...) are defined in
layer.py's module docstring. Keys name the hand-offs (``AR.rs[n]``,
``ids[g]``, ...); cross-GPU keys are modelled on one rank (as rank 0). Waits
whose producers depend on the routes or on the balanced attention schedule
list every possible producer.

Per XCD, in launch order:

- pre-MoE: slots 0-15 SENDER (o_proj at entry, or at full layers after the
  first attention item; the router weights after the push, on slot 0 after
  AR_OWNER), then ROUTER; slot 0 also runs AR_OWNER between them, and slot 1
  publishes h'[:, part xcd]; slots 16-23 POOLER (slot 16 + T on XCD T sums
  tile T); sliding attention block b on XCD b, slots 24-31 (split s on slot
  24 + s, merged by split 0); full layers run balanced items cta and cta +
  256 everywhere and merge task t < requests x 16 on (slot 24 + t // 8, xcd
  t % 8); TOPK of 8-row group g on (xcd 7 - g, slot 24); POOLED_NORM of row
  r on (xcd r % 8, slot 25 + r // 8);
- MoE: every workgroup, after the ids and xq tag lines (scalar polls);
- post-MoE: the QKV task's weights at the MoE end (not when the next layer
  is regular; POST_SENDER workgroups issue the relay's weights, then the QKV
  task's first window, and the rest of it after POST_SENDER and REDUCER);
  then ROUTE_OWNER of every row, then FINISHER, on the tail workgroups
  (slots 0 and 24-30, tail index i, tokens i + 8 r); POST_SENDER on 16-23
  (slot 16 also REDUCER); then the QKV task (part 2 xcd + slot % 2, group
  slot // 2) on every workgroup; then PROLOGUE i * 8 + xcd < 34 on the tail
  workgroups, or BOUNDARY there when the next layer is regular.
"""

from __future__ import annotations

from tokenspeed_kernel.ops.monokernel import PlanKey
from tokenspeed_kernel.ops.monokernel._gluon.plan import (
    GRID,
    SLOTS,
    XCDS,
    Clear,
    LayerPlan,
    Load,
    Publish,
    Region,
    Wait,
)

ROWS_PER_REQUEST = 8
REGION_ROWS = 32
HEAD_TASKS = 34
QKV_GROUPS = 16
QKV_PARTS = 16
UNPS = 64
FULL_MERGE_TASKS_PER_BLOCK = 8
TAIL_SLOTS = (0, 24, 25, 26, 27, 28, 29, 30)
_ALL = "all"
# Why some vector polls may sit behind loads in flight (plan.Wait.queued).
_IWC = "issue-wait-compute"
# The tail workgroups' QKV weights go out at their MoE end and land before
# their first post-MoE poll (an assumption the phase stamps check).
_LANDED = "landed"

# Two-copy cross-GPU receive areas and the local attention output area
# (``att_box``): re-armed by the o_proj senders at entry. Tagged words
# (``S``: the routed sums): never cleared.
_CLEARED = ("att_box", "AR", "POOLED_AG", "ROUTE_PUSH_own", "routes", "POST_AG")
_TAGGED = ("att_rec", "AR_sums", "rtr", "pool", "exchange", "S", "hprime",
           "reducer_in", "POST_AG_sums", "qkv")  # fmt: skip
REGIONS = tuple(Region(name, 2, REGION_ROWS) for name in _CLEARED) + tuple(
    Region(name, 1, REGION_ROWS) for name in _TAGGED
)


def _per_xcd(value):
    return (value,) * XCDS


def expected_counts(key: PlanKey) -> dict[str, tuple[int, ...]]:
    """The composer's role table per XCD for ``key``."""
    requests = key.requests
    if key.layer_kind == "full":
        attention = _per_xcd(SLOTS)
    else:
        attention = tuple(8 if xcd < 2 * requests else 0 for xcd in range(XCDS))
    counts = {
        "ATT": attention,
        "SENDER": _per_xcd(16),
        "AR_OWNER": _per_xcd(1),
        "ROUTER": _per_xcd(16),
        "POOLER": _per_xcd(8),
        "TOPK": tuple(1 if xcd >= XCDS - requests else 0 for xcd in range(XCDS)),
        "POOLED_NORM": _per_xcd(requests),
        "MOE": _per_xcd(SLOTS),
        "ROUTE_OWNER": _per_xcd(8),
        "FINISHER": _per_xcd(8),
        "POST_SENDER": _per_xcd(8),
        "REDUCER": _per_xcd(1),
    }
    if key.layer_kind == "full":
        counts["MERGE"] = _per_xcd(2 * requests)
    if key.next_kind == "regular":
        counts["BOUNDARY"] = _per_xcd(8)
    else:
        counts["QKV"] = _per_xcd(SLOTS)
        counts["PROLOGUE"] = tuple(
            sum(1 for i in range(8) if i * 8 + xcd < HEAD_TASKS) for xcd in range(XCDS)
        )
    return counts


def _front(key: PlanKey, xcd: int, slot: int) -> list:
    requests = key.requests
    rows = requests * ROWS_PER_REQUEST
    full = key.layer_kind == "full"
    blocks = 2 * requests
    merges = requests * 2 * FULL_MERGE_TASKS_PER_BLOCK
    cta = slot * XCDS + xcd
    steps = []
    ag = tuple(f"AR.ag[{j}]" for j in range(XCDS))
    boxes = tuple(f"att_box[{b}]" for b in range(merges if full else blocks))

    def items(weights=()):
        """Full layers: balanced items cta and cta + 256; a weight-holding
        role issues its weights after the first item's partial store."""
        out = [Load("ATT", _ALL), Publish("ATT", _ALL, (f"att_rec[{cta}]",))]
        out += [Load(role, _ALL) for role in weights]
        out += [Load("ATT", _ALL), Publish("ATT", _ALL, (f"att_rec[{cta + GRID}]",))]
        return out

    if slot < 16:
        sender = xcd * 16 + slot
        # o_proj at entry (paced), or at full layers after the attention
        # items; the router's hi / lo tiles after the push and AR_OWNER.
        if full:
            steps += items(("SENDER",))
        else:
            steps.append(Load("SENDER", _ALL))
        steps += [
            Clear("SENDER", _ALL, _CLEARED),
            Wait("SENDER", _ALL, boxes, "vector", _IWC),
            Publish("SENDER", _ALL, (f"AR.rs[{sender}]",)),
        ]
        if slot == 0:
            rs = (f"AR.rs[{2 * xcd}]", f"AR.rs[{2 * xcd + 1}]")
            steps += [
                Wait("AR_OWNER", _ALL, rs, "vector", ""),
                Publish("AR_OWNER", _ALL, (f"AR.ag[{xcd}]",)),
            ]
        steps.append(Load("ROUTER", _ALL))  # paced
        # The router weights may still be in flight here; they feed the
        # K-part compute after this wait.
        steps.append(Wait("ROUTER", _ALL, ag, "vector", _IWC))
        if slot == 1:
            steps.append(Publish("ROUTER", _ALL, (f"hprime[{xcd}]",)))
        steps.append(Publish("ROUTER", _ALL, (f"rtr[{xcd}][{slot}]",)))
        return steps
    if slot < 24:
        tile = slot - 16
        if full:
            steps += items(("POOLER",))
        else:
            steps.append(Load("POOLER", _ALL))
        steps += [
            Wait("POOLER", _ALL, ag, "vector", _IWC),
            Publish("POOLER", _ALL, (f"pool[{xcd}][{tile}]",)),
        ]
        if xcd == tile:
            parts = tuple(f"pool[{x}][{tile}]" for x in range(XCDS))
            steps += [
                Wait("POOLER", _ALL, parts, "vector", ""),
                Publish("POOLER", _ALL, (f"POOLED_AG[{tile}]",)),
            ]
        return steps
    if not full and xcd < blocks:
        split = slot - 24
        steps += [
            Load("ATT", _ALL),
            Publish("ATT", _ALL, (f"att_rec[{xcd}][{split}]",)),
        ]
        if split == 0:
            records = tuple(f"att_rec[{xcd}][{s}]" for s in range(8))
            steps += [
                Wait("ATT", _ALL, records, "smem", ""),
                Publish("ATT", _ALL, (f"att_box[{xcd}]",)),
            ]
    elif full:
        steps += items()
        merge = (slot - 24) * XCDS + xcd
        if merge < merges:
            records = tuple(f"att_rec[{c}]" for c in range(2 * GRID))
            steps += [
                Wait("MERGE", _ALL, records, "vector", ""),
                Publish("MERGE", _ALL, (f"att_box[{merge}]",)),
            ]
    if slot == 24 and xcd >= XCDS - requests:
        group = 7 - xcd
        logits = tuple(f"rtr[{x}][{s}]" for x in range(XCDS) for s in range(16))
        steps += [
            Wait("TOPK", _ALL, logits, "vector", ""),
            Publish("TOPK", _ALL, (f"ids[{group}]",)),
        ]
    if 25 <= slot < 25 + requests:
        row = xcd + 8 * (slot - 25)
        steps += [
            Wait(
                "POOLED_NORM",
                _ALL,
                tuple(f"POOLED_AG[{t}]" for t in range(XCDS)),
                "vector",
                "",
            ),
            Publish("POOLED_NORM", _ALL, (f"xq[{row}]",)),
        ]
    return steps


def _moe(key: PlanKey, xcd: int, slot: int) -> list:
    rows = key.requests * ROWS_PER_REQUEST
    ids = tuple(f"ids[{g}]" for g in range(key.requests))
    return [
        Wait("MOE", _ALL, ids, "smem", ""),
        Load("MOE", _ALL),  # the ring's first weights, before the xq rows
        Wait("MOE", _ALL, tuple(f"xq[{r}]" for r in range(rows)), "smem", ""),
        # The down units' epilogue pushes their outputs (ROUTE_PUSH).
        Publish("MOE", _ALL, (f"ROUTE_PUSH[{xcd}][{slot}]",)),
    ]


def _tail(key: PlanKey, xcd: int, slot: int) -> list:
    requests = key.requests
    rows = requests * ROWS_PER_REQUEST
    regular = key.next_kind == "regular"
    steps = []
    tail = slot in TAIL_SLOTS
    post_sender = 16 <= slot < 24
    all_route_push = tuple(
        f"ROUTE_PUSH[{x}][{s}]" for x in range(XCDS) for s in range(SLOTS)
    )
    reducer = tuple(f"reducer[{x}]" for x in range(XCDS))
    if not regular and not post_sender:
        steps.append(Load("QKV", _ALL))  # paced, at this workgroup's MoE end
    if tail:
        index = TAIL_SLOTS.index(slot)
        tokens = [index + 8 * r for r in range(requests)]
        # Every row's route-owner push before any finisher wait; one publish
        # of the routed sums.
        for token in tokens:
            steps += [
                Wait("ROUTE_OWNER", _ALL, all_route_push, "vector", _LANDED),
                Publish("ROUTE_OWNER", _ALL, (f"ROUTE_AG[{token}][{xcd}]",)),
            ]
        for token in tokens:
            route_ag = tuple(f"ROUTE_AG[{token}][{k}]" for k in range(XCDS))
            steps.append(Wait("FINISHER", _ALL, route_ag, "vector", _LANDED))
        steps.append(
            Publish("FINISHER", _ALL, tuple(f"S[{xcd}][{token}]" for token in tokens))
        )
    elif post_sender:
        sender = xcd * 8 + slot - 16
        s_lines = tuple(f"S[{p}][{t}]" for p in range(XCDS) for t in range(rows))
        steps.append(Load("POST_SENDER", _ALL))  # the relay's post-expert weights
        if not regular:
            # The QKV task's first window of tiles, paced; the rest follow
            # the relay's (and REDUCER's) work.
            steps.append(Load("QKV", _ALL))
        steps += [
            Wait("POST_SENDER", _ALL, s_lines, "smem", ""),
            Publish("POST_SENDER", _ALL, (f"POST_AG[{sender}]",)),
            Wait("POST_SENDER", _ALL, ("hprime[0]",), "smem", ""),
            Publish("POST_SENDER", _ALL, (f"reducer_in[{xcd}][{slot - 16}]",)),
        ]
        if slot == 16:
            partials = tuple(f"reducer_in[{xcd}][{b}]" for b in range(8))
            steps += [
                Wait("REDUCER", _ALL, partials, "vector", _LANDED),
                Publish("REDUCER", _ALL, (f"reducer[{xcd}]",)),
            ]
        if not regular:
            steps.append(Load("QKV", _ALL))  # the rest of the QKV tiles, paced
    if regular:
        if tail:
            post_ag = tuple(f"POST_AG[{u}]" for u in range(UNPS))
            steps += [
                Wait("BOUNDARY", _ALL, reducer + post_ag, "vector", ""),
                Wait("BOUNDARY", _ALL, (f"hprime[{xcd}]",), "smem", ""),
                Publish("BOUNDARY", _ALL, (f"rows[{xcd}][{slot}]",)),
            ]
        return steps
    part = 2 * xcd + slot % 2
    group = slot // 2
    task = part * QKV_GROUPS + group
    half = part % 2
    post_ag = tuple(f"POST_AG[{u}]" for u in range(32 * half, 32 * half + 32))
    steps += [
        Wait("QKV", _ALL, reducer + post_ag, "vector", _IWC),
        Wait("QKV", _ALL, (f"hprime[{part // 2}]",), "smem", ""),
        Publish("QKV", _ALL, (f"qkv[{task}]",)),
    ]
    if tail:
        head = TAIL_SLOTS.index(slot) * 8 + xcd
        if head < HEAD_TASKS:
            # Every task's partials (the head's columns span groups).
            parts = tuple(f"qkv[{t}]" for t in range(QKV_PARTS * QKV_GROUPS))
            steps += [
                Wait("PROLOGUE", _ALL, parts, "vector", ""),
                Publish("PROLOGUE", _ALL, (f"kv[{head}]",)),
            ]
    return steps


def muse_plan(key: PlanKey) -> LayerPlan:
    """The plan of one Muse Spark monokernel launch for ``key``."""
    programs = {}
    for xcd in range(XCDS):
        for slot in range(SLOTS):
            steps = (
                _front(key, xcd, slot) + _moe(key, xcd, slot) + _tail(key, xcd, slot)
            )
            programs[(xcd, slot)] = tuple(steps)
    return LayerPlan(
        name=f"muse_spark{key}",
        rows=key.requests * ROWS_PER_REQUEST,
        groups=(_ALL,),
        programs=programs,
        regions=REGIONS,
        all_to_all="AR.rs",
        expected_counts=expected_counts(key),
    )
