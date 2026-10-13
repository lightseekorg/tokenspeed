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

"""Full garbage collections of a scheduler process, out of the way of serving.

Startup leaves millions of long-lived Python objects in a scheduler process:
imported modules, the model's modules and parameters, captured graphs, kernel
and compile caches. A full (generation 2) collection walks all of them, which
takes hundreds of milliseconds, and CPython starts one whenever enough objects
have survived the young generations -- on each rank at its own moment. Every
round waits on the whole parallel group, so each rank's pause stalls every
rank.

``freeze_startup_heap`` runs once, at the end of startup. It moves the startup
objects into the permanent generation (``gc.freeze``), which no collection
walks, and raises the generation-2 threshold far enough that CPython does not
start full collections by itself while the loop runs its own. ``GcHooks``
runs one every ``FULL_COLLECTION_INTERVAL`` forwarded rounds. That count is
the same on every rank, so all ranks collect in the same round, and each
collection walks only the objects serving keeps alive. The young generations
keep their automatic collections.
"""

from __future__ import annotations

import gc
import logging

__all__ = ["FULL_COLLECTION_INTERVAL", "GcHooks", "freeze_startup_heap"]

logger = logging.getLogger(__name__)

# Forwarded rounds between two full collections: about 15 s of decoding at a
# 15 ms step. With the startup objects frozen, one collection walks only what
# serving keeps alive, mostly the requests' token lists: a few milliseconds
# for 128 requests of 9K tokens.
FULL_COLLECTION_INTERVAL = 1024

# The generation-2 threshold while serving: how many generation-1 collections
# CPython lets pass before it starts a full collection (10 by default). The
# interval above resets that count long before it gets here; this bound only
# matters for a process that keeps allocating without forwarding.
_SERVING_GEN2_THRESHOLD = 10_000


def freeze_startup_heap() -> None:
    """End startup for the garbage collector; call once, before serving.

    Collects first, so that no startup garbage is frozen, then freezes what
    is left and leaves full collections to ``GcHooks``.
    """
    gc.collect()
    gc.freeze()
    threshold0, threshold1, _ = gc.get_threshold()
    gc.set_threshold(threshold0, threshold1, _SERVING_GEN2_THRESHOLD)
    logger.info(
        f"Froze {gc.get_freeze_count():d} startup objects out of garbage "
        "collection; full collections now run between rounds"
    )


class GcHooks:
    """Runs a full garbage collection every N forwarded rounds.

    Self-contained: it depends only on its interval. The loop calls
    ``note_round`` after it submits each round's forward, with the same
    rank-identical flag the expert rebalance counts (a real or a DP-idle
    forward), so every rank collects in the same round.
    """

    def __init__(self, *, full_collection_interval: int) -> None:
        if full_collection_interval < 1:
            raise ValueError(
                f"full_collection_interval must be positive, got "
                f"{full_collection_interval:d}"
            )
        self._full_collection_interval = full_collection_interval
        self._forwarded_rounds = 0

    def note_round(self, *, forwarded: bool) -> None:
        """Count a round that forwarded; collect when the count is due."""
        if not forwarded:
            return
        self._forwarded_rounds += 1
        if self._forwarded_rounds % self._full_collection_interval == 0:
            gc.collect()
