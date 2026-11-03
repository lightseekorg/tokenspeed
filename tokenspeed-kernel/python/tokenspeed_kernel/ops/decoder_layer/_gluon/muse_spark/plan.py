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

"""The Muse Spark layer's role table, as layer.py composes it, as a plan.

:func:`muse_plan` builds the step sequence of every workgroup ``(xcd, slot)``
for one :class:`PlanKey`, following ``gluon_decoder_layer_muse_spark_gfx950``
role for role; :func:`expected_counts` is the per-XCD role table written apart
from the builder, so the checker compares two descriptions. Keys name the
hand-offs (``H1.rs[n]``, ``ids[g]``, ...); cross-GPU keys are modelled on one
rank (as rank 0). Waits whose producers depend on the routes or on the
balanced attention schedule list every possible producer.

Per XCD, in launch order:

- front: slots 0-15 SND (o_proj at entry, or at full layers after the
  first attention item; the router weights after the push, on slot 0 after
  OWN1) then RTR, slot 0 also OWN1 between them and
  slot 1 publishing h'[:, part xcd]; slots 16-23 POOL (slot 16 + T on XCD T
  reduces tile T); sliding attention blocks b on XCD b, slots 24-31 (split s
  on slot 24 + s, merger split 0); full layers run balanced items cta and cta
  + 256 everywhere and merge task t < requests x 16 on (slot 24 + t // 8,
  xcd t % 8); TOPK of 8-row group g on (xcd 7 - g, slot 24); XN of row r on
  (xcd r % 8, slot 25 + r // 8);
- MoE: every workgroup, after the ids and xq tag lines (scalar polls);
- tail: the QKV task's weights at the MoE end (not when the next layer is
  regular; UNP workgroups issue the relay's weights, then the QKV task's
  first window, and the rest of it after UNP, before RED's scalar polls),
  then OWN4 of every row, then FIN, on slots 0 and 24-30 (tail index i,
  tokens i + 8 r), UNP on 16-23 (slot 16 also RED), then the QKV task (part
  2 xcd + slot % 2, group slot // 2) on every workgroup, then HEAD i * 8 +
  xcd < 34 on tail workgroups; or BND2 on tail workgroups when the next
  layer is regular.
"""

from __future__ import annotations

from tokenspeed_kernel.ops.decoder_layer import PlanKey
from tokenspeed_kernel.ops.decoder_layer._gluon.plan import (
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
# The tail roles' QKV weights go out at their MoE end and land before the
# first tail poll (an assumption the phase stamps check).
_LANDED = "landed"

# Two-copy cross-GPU receive areas and the local attention box: re-armed by
# the SND workgroups at entry. Tagged words: never cleared.
_CLEARED = ("att_box", "H1", "H3", "H4_own", "routes", "H6")
_TAGGED = ("att_rec", "H1_sums", "rtr", "pool", "exchange", "S", "hprime",
           "red_in", "H6_sums", "qkv")  # fmt: skip
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
        "SND": _per_xcd(16),
        "OWN1": _per_xcd(1),
        "RTR": _per_xcd(16),
        "POOL": _per_xcd(8),
        "TOPK": tuple(1 if xcd >= XCDS - requests else 0 for xcd in range(XCDS)),
        "XN": _per_xcd(requests),
        "MOE": _per_xcd(SLOTS),
        "OWN4": _per_xcd(8),
        "FIN": _per_xcd(8),
        "UNP": _per_xcd(8),
        "RED": _per_xcd(1),
    }
    if key.layer_kind == "full":
        counts["MERGE"] = _per_xcd(2 * requests)
    if key.next_kind == "regular":
        counts["BND2"] = _per_xcd(8)
    else:
        counts["QKV"] = _per_xcd(SLOTS)
        counts["HEAD"] = tuple(
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
    ag = tuple(f"H1.ag[{j}]" for j in range(XCDS))
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
        # o_proj (SND) at entry (paced), or at full layers at the attention
        # items' hook; the router's hi / lo (RTR) after the push and OWN1.
        if full:
            steps += items(("SND",))
        else:
            steps.append(Load("SND", _ALL))
        steps += [
            Clear("SND", _ALL, _CLEARED),
            Wait("SND", _ALL, boxes, "vector", _IWC),
            Publish("SND", _ALL, (f"H1.rs[{sender}]",)),
        ]
        if slot == 0:
            rs = (f"H1.rs[{2 * xcd}]", f"H1.rs[{2 * xcd + 1}]")
            steps += [
                Wait("OWN1", _ALL, rs, "vector", ""),
                Publish("OWN1", _ALL, (f"H1.ag[{xcd}]",)),
            ]
        steps.append(Load("RTR", _ALL))  # paced
        # The router weights may still be in flight here; they feed the
        # K-part compute after this wait.
        steps.append(Wait("RTR", _ALL, ag, "vector", _IWC))
        if slot == 1:
            steps.append(Publish("RTR", _ALL, (f"hprime[{xcd}]",)))
        steps.append(Publish("RTR", _ALL, (f"rtr[{xcd}][{slot}]",)))
        return steps
    if slot < 24:
        tile = slot - 16
        if full:
            steps += items(("POOL",))
        else:
            steps.append(Load("POOL", _ALL))
        steps += [
            Wait("POOL", _ALL, ag, "vector", _IWC),
            Publish("POOL", _ALL, (f"pool[{xcd}][{tile}]",)),
        ]
        if xcd == tile:
            parts = tuple(f"pool[{x}][{tile}]" for x in range(XCDS))
            steps += [
                Wait("POOL", _ALL, parts, "vector", ""),
                Publish("POOL", _ALL, (f"H3[{tile}]",)),
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
            Wait("XN", _ALL, tuple(f"H3[{t}]" for t in range(XCDS)), "vector", ""),
            Publish("XN", _ALL, (f"xq[{row}]",)),
        ]
    return steps


def _moe(key: PlanKey, xcd: int, slot: int) -> list:
    rows = key.requests * ROWS_PER_REQUEST
    ids = tuple(f"ids[{g}]" for g in range(key.requests))
    return [
        Wait("MOE", _ALL, ids, "smem", ""),
        Load("MOE", _ALL),  # the ring's first weights, before the xq rows
        Wait("MOE", _ALL, tuple(f"xq[{r}]" for r in range(rows)), "smem", ""),
        # dn epilogue: H4 pushes from this workgroup's units (producer-direct).
        Publish("MOE", _ALL, (f"H4[{xcd}][{slot}]",)),
    ]


def _tail(key: PlanKey, xcd: int, slot: int) -> list:
    requests = key.requests
    rows = requests * ROWS_PER_REQUEST
    regular = key.next_kind == "regular"
    steps = []
    tail = slot in TAIL_SLOTS
    unp = 16 <= slot < 24
    all_h4 = tuple(f"H4[{x}][{s}]" for x in range(XCDS) for s in range(SLOTS))
    red = tuple(f"red[{x}]" for x in range(XCDS))
    if not regular and not unp:
        steps.append(Load("QKV", _ALL))  # paced, at this workgroup's MoE end
    if tail:
        index = TAIL_SLOTS.index(slot)
        tokens = [index + 8 * r for r in range(requests)]
        # Every row's OWN4 push before any FIN wait; one S publish.
        for token in tokens:
            steps += [
                Wait("OWN4", _ALL, all_h4, "vector", _LANDED),
                Publish("OWN4", _ALL, (f"H5[{token}][{xcd}]",)),
            ]
        for token in tokens:
            h5 = tuple(f"H5[{token}][{k}]" for k in range(XCDS))
            steps.append(Wait("FIN", _ALL, h5, "vector", _LANDED))
        steps.append(
            Publish("FIN", _ALL, tuple(f"S[{xcd}][{token}]" for token in tokens))
        )
    elif unp:
        sender = xcd * 8 + slot - 16
        s_lines = tuple(f"S[{p}][{t}]" for p in range(XCDS) for t in range(rows))
        steps.append(Load("UNP", _ALL))  # the relay's post-expert weights
        if not regular:
            # The QKV task's first window of tiles, paced; the rest follow
            # the relay's (and RED's) work.
            steps.append(Load("QKV", _ALL))
        steps += [
            Wait("UNP", _ALL, s_lines, "smem", ""),
            Publish("UNP", _ALL, (f"H6[{sender}]",)),
            Wait("UNP", _ALL, ("hprime[0]",), "smem", ""),
            Publish("UNP", _ALL, (f"red_in[{xcd}][{slot - 16}]",)),
        ]
        if not regular:
            steps.append(Load("QKV", _ALL))  # the rest of the QKV tiles, paced
        if slot == 16:
            # Scalar polls: the tiles just issued stay in flight.
            partials = tuple(f"red_in[{xcd}][{b}]" for b in range(8))
            steps += [
                Wait("RED", _ALL, partials, "smem", ""),
                Publish("RED", _ALL, (f"red[{xcd}]",)),
            ]
    if regular:
        if tail:
            h6 = tuple(f"H6[{u}]" for u in range(UNPS))
            steps += [
                Wait("BND2", _ALL, red + h6, "vector", ""),
                Wait("BND2", _ALL, (f"hprime[{xcd}]",), "smem", ""),
                Publish("BND2", _ALL, (f"rows[{xcd}][{slot}]",)),
            ]
        return steps
    part = 2 * xcd + slot % 2
    group = slot // 2
    task = part * QKV_GROUPS + group
    half = part % 2
    h6 = tuple(f"H6[{u}]" for u in range(32 * half, 32 * half + 32))
    steps += [
        Wait("QKV", _ALL, red + h6, "vector", _IWC),
        Wait("QKV", _ALL, (f"hprime[{part // 2}]",), "smem", ""),
        Publish("QKV", _ALL, (f"qkv[{task}]",)),
    ]
    if tail:
        head = TAIL_SLOTS.index(slot) * 8 + xcd
        if head < HEAD_TASKS:
            # Every task's partials (the head's columns span groups).
            parts = tuple(f"qkv[{t}]" for t in range(QKV_PARTS * QKV_GROUPS))
            steps += [
                Wait("HEAD", _ALL, parts, "vector", ""),
                Publish("HEAD", _ALL, (f"kv[{head}]",)),
            ]
    return steps


def muse_plan(key: PlanKey) -> LayerPlan:
    """The plan of one Muse Spark fused-layer launch for ``key``."""
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
        all_to_all="H1.rs",
        expected_counts=expected_counts(key),
    )
