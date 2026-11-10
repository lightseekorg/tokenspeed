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

"""Stuck-poll records of Muse Spark layer launches, decoded under the plans
the launches ran (``monokernel.stuck``).

The serving seam (and the layer bench's ``--stuck-dir``) describes its
launches in a sidecar next to a file-backed record buffer (``<file>.json``):
each launching layer's kind, the ``(rows, chain)`` of every launch it
compiled, whether the launches check the peer parity (``check_parity``), and,
for launches that do not follow serving's rule (the layer bench), their
``layer_plan`` flags (``flags``).
Then, after a run that trapped::

    python -m tokenspeed_kernel_amd.ops.gfx950.monokernel.muse_spark.stuck_report FILE...

The report rebuilds each launch's plan from the sidecar's ``flags``, or by
serving's rule (:func:`served_flags`) when it has none.
"""

from __future__ import annotations

import json
import sys

import torch
from tokenspeed_kernel_amd.ops.gfx950.monokernel import stuck
from tokenspeed_kernel_amd.ops.gfx950.monokernel.muse_spark.layer import layer_plan


def served_flags(layer: int, rows: int, layers: dict[int, dict]) -> dict[str, bool]:
    """``layer_plan``'s ``warm_next``, ``own_qkvg`` and ``next_qkvg`` of
    serving's launch of layer ``layer`` at ``rows`` rows (``layers``: the
    launching layers). Past one row a launch whose next layer launches ends
    with that layer's q | k | v projection (qnext), and a launch whose
    previous layer launches reads its own from there; at one row a launch
    whose next layer launches warms that layer's QKVG weights instead
    (``projection_hand_off`` and ``warm_next`` in the seam,
    ``python/tokenspeed/runtime/models/muse_spark_monokernel.py``)."""
    prev, nxt = layer - 1 in layers, layer + 1 in layers
    return {
        "warm_next": rows == 1 and nxt,
        "own_qkvg": rows == 1 or not prev,
        "next_qkvg": rows > 1 and nxt,
    }


def plans_of(
    layers: dict[int, dict],
    launches: list[tuple[int, int]],
    check_parity: bool,
    flags: dict[str, bool] | None,
):
    """``plan_of(layer)`` for :func:`stuck.decode`: the plan of each
    compiled ``(rows, chain)`` launch of layer ``layer`` (its kind from
    ``layers``: ``{"full": bool}``), with ``flags`` (``layer_plan``'s
    ``warm_next``, ``own_qkvg`` and ``next_qkvg`` of every launch), or with
    :func:`served_flags` when ``flags`` is None. ``check_parity`` (the layer
    bench's launches check it; serving's do not) moves the regions."""

    def plan_of(layer: int):
        kind = layers.get(layer)
        if kind is None:
            return []
        return [
            layer_plan(
                full=kind["full"],
                rows=rows,
                chain=chain,
                check_parity=check_parity,
                **(served_flags(layer, rows, layers) if flags is None else flags),
            )
            for rows, chain in launches
        ]

    return plan_of


def describe(
    t: torch.Tensor,
    layers: dict[int, dict],
    launches,
    check_parity: bool,
    flags: dict[str, bool] | None,
) -> list[str]:
    """The records of buffer ``t`` (by CTA and wave) decoded under the
    launches' plans (:func:`plans_of`), one group a wait, the first stalled
    wait first (:func:`stuck.summarize`)."""
    return stuck.summarize(
        t, plans_of(layers, [tuple(x) for x in launches], check_parity, flags)
    )


def describe_file(path: str) -> list[str]:
    """The records of a file-backed buffer, with its sidecar's plans."""
    with open(f"{path}.json") as f:
        side = json.load(f)
    layers = {int(k): v for k, v in side["layers"].items()}
    # Only a writer whose launches do not follow serving's rule sets flags.
    return describe(
        stuck.read_file(path),
        layers,
        side["launches"],
        side["check_parity"],
        side.get("flags"),
    )


def main(paths: list[str]) -> int:
    for path in paths:
        for line in describe_file(path) or [f"{path}: no stuck poll"]:
            print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
