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

"""Cache-group geometry learned from the pool's published specs.

One immutable value object answers every "what shape is this group?"
question a backend asks — group block granularities, every group's family
(the positive-claim vocabulary ``cache_consumer_families`` filters against),
and which group is the full history the draft chain writes along. Learned
exactly once, at ``set_cache_pool`` (the arena's published specs are the
only source, so the eager and CUDA-graph arms can never answer
differently).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CacheGroupGeometry:
    """Frozen per-pool group geometry.

    Attributes:
        granularities: ``group_id -> block_granularity`` for every non-state
            group (rows-per-page span; equals the group's page size for the
            row-geometry groups kept here).
        families: ``group_id -> family`` for every published group. The
            positive-claim vocabulary: a backend keeps exactly the delivered
            groups whose family it declared in ``cache_consumer_families``;
            the rest (state blocks for the mamba backend, wrapper-owned conv
            groups) ride the same dict to their own consumers. Empty when
            no pool is bound (unit fixtures, pre-contract draft pools).
        full_history_group_id: The first ``family="history"`` group with
            ``retention="full_history"`` — the table the router's draft
            write locations ride — or None when the pool publishes no such
            group (a single-group pool's sole group then serves as the
            history).
    """

    granularities: dict[str, int] = field(default_factory=dict)
    families: dict[str, str] = field(default_factory=dict)
    full_history_group_id: str | None = None

    def granularity_of(self, group_id: str) -> int:
        """This group's block granularity; an unknown id is a contract bug.

        Every id reaching here must name a learned row-geometry group —
        layer group ids are validated against the pool's published specs at
        startup (``validate_cache_group_ids``), and table dicts are keyed by
        contract ids. No fallback: a miss means the geometry was never
        learned (pool not bound) or the id belongs to a state group.
        """
        try:
            return self.granularities[group_id]
        except KeyError:
            raise KeyError(
                f"cache group {group_id!r} has no learned block granularity "
                f"(learned row-geometry groups: {sorted(self.granularities)})"
            ) from None


def learn_cache_group_geometry(cache_group_specs) -> CacheGroupGeometry:
    """Build the geometry from the pool's published group specs.

    Args:
        cache_group_specs: The arena's ``cache_group_specs`` tuple.

    Returns:
        The frozen geometry.
    """
    full_history = next(
        (
            spec
            for spec in cache_group_specs
            if spec.family == "history" and spec.retention == "full_history"
        ),
        None,
    )
    return CacheGroupGeometry(
        granularities={
            str(spec.group_id): spec.block_granularity
            for spec in cache_group_specs
            if spec.family != "state"
        },
        families={str(spec.group_id): str(spec.family) for spec in cache_group_specs},
        full_history_group_id=(
            str(full_history.group_id) if full_history is not None else None
        ),
    )
