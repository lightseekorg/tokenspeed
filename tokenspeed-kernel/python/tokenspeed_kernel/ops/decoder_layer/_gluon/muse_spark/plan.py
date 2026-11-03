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

"""The Muse Spark layer's role table (NATIVE-DESIGN v0.6 §4.1) as a plan.

:func:`muse_plan` builds the step sequence of every workgroup ``(xcd, slot)``
for one :class:`PlanKey`; :data:`EXPECTED_COUNTS` is the per-XCD role table
the design states, kept apart from the builder so the checker compares two
independent descriptions. Keys name the hand-offs of NATIVE-DESIGN §3.1
(``H1.rs[n]``, ``ids[row]``, ...); cross-GPU keys are modelled on one rank.

Roles, per XCD (8 rows; 16 rows where different):

- front: slots 0-15 SND then RTR (slot 0 is also OWN1 between them); 16-23
  POOL (slot 16 + T on XCD T reduces tile T); attention blocks
  ``b = (request, KV head)`` on XCDs ``0 .. 2 * requests - 1``, slots 24-31
  (split s on slot 24 + s, merger = split 0; a full layer runs attention on
  every workgroup); TOPK on slot 24 [24-25], XN on slot 25 [26-27];
- MoE: every workgroup;
- tail: OWN4 → FIN → HEAD on slots 0 and 24-30 (one pick each [two]),
  UNP on 16-23 (slot 16 is also RED), one QKV task on every workgroup; when
  the next layer is regular, BND2 on the OWN4 workgroups replaces QKV and
  HEAD.
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
HEADS = 36
QKV_GROUPS = 16
QKV_PARTS = 16
TAIL_SLOTS = (0, 24, 25, 26, 27, 28, 29, 30)
_ALL = "all"

# Two-copy cross-GPU receive areas and the local attention box: re-armed by
# the SND workgroups at entry (NATIVE-DESIGN §7.1). Tagged words: never cleared.
_CLEARED = ("att_box", "H1.rs", "H1.ag", "H3", "H4", "H5", "H6")
_TAGGED = ("att_rec", "rtr", "pool", "ids", "xq", "S", "hprime", "redsum", "qkv")
REGIONS = tuple(Region(name, 2, REGION_ROWS) for name in _CLEARED) + tuple(
    Region(name, 1, REGION_ROWS) for name in _TAGGED
)


def _per_xcd(value):
    return (value,) * XCDS


def expected_counts(key: PlanKey) -> dict[str, tuple[int, ...]]:
    """The design's role table per XCD for ``key`` (NATIVE-DESIGN §4.1)."""
    blocks = 2 * key.requests
    if key.layer_kind == "full":
        attention = _per_xcd(SLOTS)
    else:
        attention = tuple(8 if xcd < blocks else 0 for xcd in range(XCDS))
    counts = {
        "ATT": attention,
        "SND": _per_xcd(16),
        "OWN1": _per_xcd(1),
        "RTR": _per_xcd(16),
        "POOL": _per_xcd(8),
        "TOPK": _per_xcd(key.requests),
        "XN": _per_xcd(key.requests),
        "MOE": _per_xcd(SLOTS),
        "OWN4": _per_xcd(8),
        "FIN": _per_xcd(8),
        "UNP": _per_xcd(8),
        "RED": _per_xcd(1),
    }
    if key.next_kind == "regular":
        counts["BND2"] = _per_xcd(8)
    else:
        counts["QKV"] = _per_xcd(SLOTS)
        counts["HEAD"] = tuple(
            sum(1 for head in range(HEADS) if head % XCDS == xcd) for xcd in range(XCDS)
        )
    return counts


def _attention(key: PlanKey, xcd: int, slot: int) -> list:
    """Attention splits and mergers of this workgroup."""
    blocks = 2 * key.requests
    if key.layer_kind == "full":
        cta = slot * XCDS + xcd
        block, split = cta % blocks, cta // blocks
        splits = GRID // blocks
        load = "vector"  # records from other XCDs: global-tagged pairs
    elif xcd < blocks and slot >= 24:
        block, split, splits, load = xcd, slot - 24, 8, "smem"
    else:
        return []
    steps = [
        Load("ATT", _ALL),
        Publish("ATT", _ALL, (f"att_rec[{block}][{split}]",)),
    ]
    if split == 0:
        records = tuple(f"att_rec[{block}][{s}]" for s in range(splits))
        steps += [
            Wait("ATT", _ALL, records, load, False),
            Publish("ATT", _ALL, (f"att_box[{block}]",)),
        ]
    return steps


def _front(key: PlanKey, xcd: int, slot: int) -> list:
    rows = key.requests * ROWS_PER_REQUEST
    blocks = 2 * key.requests
    steps = _attention(key, xcd, slot)
    boxes = tuple(f"att_box[{b}]" for b in range(blocks))
    ag = tuple(f"H1.ag[{j}]" for j in range(XCDS))
    if slot < 16:
        sender = xcd * 16 + slot
        steps += [
            Load("SND", _ALL),
            Clear("SND", _ALL, _CLEARED),
            Wait("SND", _ALL, boxes, "vector", True),
            Publish("SND", _ALL, (f"H1.rs[{sender}]",)),
        ]
        if slot == 0:
            steps += [
                Wait("OWN1", _ALL, (f"H1.rs[{2 * xcd}]", f"H1.rs[{2 * xcd + 1}]"),
                     "vector", False),
                Publish("OWN1", _ALL, (f"H1.ag[{xcd}]", f"hprime[{xcd}]")),
            ]  # fmt: skip
        steps += [
            Load("RTR", _ALL),
            Wait("RTR", _ALL, ag, "vector", True),
            Publish("RTR", _ALL, (f"rtr[{xcd}][{slot}]",)),
        ]
    elif slot < 24:
        tile = slot - 16
        steps += [
            Load("POOL", _ALL),
            Wait("POOL", _ALL, ag, "vector", True),
            Publish("POOL", _ALL, (f"pool[{xcd}][{tile}]",)),
        ]
        if xcd == tile:
            parts = tuple(f"pool[{x}][{tile}]" for x in range(XCDS))
            steps += [
                Wait("POOL", _ALL, parts, "vector", False),
                Publish("POOL", _ALL, (f"H3[{tile}]",)),
            ]
    topk_slots = range(24, 24 + key.requests)
    xn_slots = range(24 + key.requests, 24 + 2 * key.requests)
    if slot in topk_slots:
        row = xcd + XCDS * (slot - 24)
        logits = tuple(f"rtr[{x}][{s}]" for x in range(XCDS) for s in range(16))
        steps += [
            Wait("TOPK", _ALL, logits, "vector", False),
            Publish("TOPK", _ALL, (f"ids[{row}]",)),
        ]
    if slot in xn_slots:
        row = xcd + XCDS * (slot - xn_slots.start)
        steps += [
            Wait("XN", _ALL, tuple(f"H3[{t}]" for t in range(XCDS)), "vector", False),
            Publish("XN", _ALL, (f"xq[{row}]",)),
        ]
    return steps


def _moe(key: PlanKey, xcd: int, slot: int) -> list:
    rows = key.requests * ROWS_PER_REQUEST
    return [
        Wait("MOE", _ALL, tuple(f"ids[{r}]" for r in range(rows)), "smem", False),
        Wait("MOE", _ALL, tuple(f"xq[{r}]" for r in range(rows)), "smem", False),
        Load("MOE", _ALL),
        # dn epilogue: H4 pushes from this workgroup's units (producer-direct).
        Publish("MOE", _ALL, (f"H4[{xcd}][{slot}]",)),
    ]


def _tail(key: PlanKey, xcd: int, slot: int) -> list:
    rows = key.requests * ROWS_PER_REQUEST
    steps = []
    all_h4 = tuple(f"H4[{x}][{s}]" for x in range(XCDS) for s in range(SLOTS))
    h6 = tuple(f"H6[{b}][{x}]" for b in range(8) for x in range(XCDS))
    h6_sums = tuple(f"H6sum[{x}]" for x in range(XCDS))
    if slot in TAIL_SLOTS:
        index = TAIL_SLOTS.index(slot)
        my_rows = tuple(range(index, rows, XCDS))
        # OWN4: pick (row, k = xcd) for each of its rows; H4 sources depend on
        # the routes, so every dn workgroup is listed.
        steps += [
            Wait("OWN4", _ALL, all_h4, "vector", False),
            Publish("OWN4", _ALL, tuple(f"H5[{r}][{xcd}]" for r in my_rows)),
            Wait("FIN", _ALL, tuple(f"H5[{r}][{k}]" for r in my_rows for k in range(XCDS)),
                 "vector", False),
            Publish("FIN", _ALL, tuple(f"S[{r}][{xcd}]" for r in my_rows)),
        ]  # fmt: skip
    elif 16 <= slot < 24:
        sender = slot - 16
        steps += [
            Load("UNP", _ALL),
            Wait("UNP", _ALL, tuple(f"S[{r}][{xcd}]" for r in range(rows)), "vector", True),
            Publish("UNP", _ALL, (f"H6[{sender}][{xcd}]", f"redsum[{sender}][{xcd}]")),
        ]  # fmt: skip
        if slot == 16:
            steps += [
                Wait("RED", _ALL, tuple(f"redsum[{b}][{xcd}]" for b in range(8)),
                     "vector", False),
                Publish("RED", _ALL, (f"H6sum[{xcd}]",)),
            ]  # fmt: skip
    if key.next_kind == "regular":
        if slot in TAIL_SLOTS:
            steps += [
                Wait("BND2", _ALL, h6 + h6_sums, "vector", False),
                Publish("BND2", _ALL, (f"rows[{xcd}][{slot}]",)),
            ]
        return steps
    task = xcd * SLOTS + slot
    group, part = task // QKV_PARTS, task % QKV_PARTS
    half = part % 2
    post = tuple(f"H6[{u // 8}][{u % 8}]" for u in range(32 * half, 32 * half + 32))
    steps += [
        Load("QKV", _ALL),
        Wait("QKV", _ALL, post + h6_sums + (f"hprime[{xcd}]",), "vector", True),
        Publish("QKV", _ALL, (f"qkv[{group}][{part}]",)),
    ]
    if slot in TAIL_SLOTS:
        index = TAIL_SLOTS.index(slot)
        for head in range(index * XCDS + xcd, HEADS, XCDS * len(TAIL_SLOTS)):
            parts = tuple(f"qkv[{head % QKV_GROUPS}][{j}]" for j in range(QKV_PARTS))
            steps += [
                Wait("HEAD", _ALL, parts, "vector", False),
                Publish("HEAD", _ALL, (f"kv[{head}]",)),
            ]
    return steps


def muse_plan(key: PlanKey) -> LayerPlan:
    """The plan of one Muse Spark fused-layer launch for ``key``."""
    programs = {}
    for xcd in range(XCDS):
        for slot in range(SLOTS):
            steps = _front(key, xcd, slot) + _moe(key, xcd, slot) + _tail(key, xcd, slot)
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
