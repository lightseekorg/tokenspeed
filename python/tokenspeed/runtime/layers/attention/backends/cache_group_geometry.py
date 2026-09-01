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
question a backend asks — group block granularities, which groups are
state-family (shed from attention tables), and the full-history grain that
batch-ordered draft tables carry. Learned exactly once, at
``set_cache_pool`` (the arena's published specs are the only source, so the
eager and CUDA-graph arms can never answer differently).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from tokenspeed.runtime.layers.attention.page_table import expand_page_table


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
            the rest ride the same dict to their own consumers. Empty when
            no pool is bound (unit fixtures, pre-contract draft pools).
        state_group_ids: ``family="state"`` group ids (GDN/mamba state
            blocks, consumed by the mamba backend and shed from every
            attention table).
        full_history_group_id: The first ``family="history"`` group with
            ``retention="full_history"``, or None when no pool bound (unit
            fixtures). Same selection rule as the executor's staging.
        history_block_granularity: That group's grain — the unit of the
            batch-ordered draft page table; falls back to the backend's
            kernel page size when no pool bound.
    """

    granularities: dict[str, int] = field(default_factory=dict)
    families: dict[str, str] = field(default_factory=dict)
    state_group_ids: frozenset[str] = frozenset()
    full_history_group_id: str | None = None
    history_block_granularity: int = 0

    def granularity_of(self, group_id: str, default: int) -> int:
        """This group's block granularity, or ``default`` for unknown ids."""
        return self.granularities.get(group_id, default)


def learn_cache_group_geometry(
    cache_group_specs, default_granularity: int
) -> CacheGroupGeometry:
    """Build the geometry from the pool's published group specs.

    Args:
        cache_group_specs: The arena's ``cache_group_specs`` tuple.
        default_granularity: Fallback history grain when the pool publishes
            no full-history group (the backend's kernel page size).

    Returns:
        The frozen geometry.
    """
    full_history = next(
        (
            spec
            for spec in cache_group_specs
            if spec.family == "history"
            and getattr(spec, "retention", "full_history") == "full_history"
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
        state_group_ids=frozenset(
            str(spec.group_id) for spec in cache_group_specs if spec.family == "state"
        ),
        full_history_group_id=(
            str(full_history.group_id) if full_history is not None else None
        ),
        history_block_granularity=(
            int(full_history.block_granularity)
            if full_history is not None
            else default_granularity
        ),
    )


def resolve_full_history_table(
    block_tables,
    geometry: CacheGroupGeometry,
    bs: int,
    *,
    kernel_page_size: int,
    max_kernel_pages: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """This forward's full-attention table in the backend's kernel pages.

    Expands the wrapper-delivered raw group table (block-granularity page
    ids, batch-ordered rows) into kernel pages of width ``max_kernel_pages``;
    -1 holes clamp into the null page 0's kernel range. Returns None when no
    table was delivered (warmup placeholders, idle before binding) — the
    caller falls back to ``page_table``. ``out`` writes the expansion into a
    persistent buffer (its rows past the raw table's are the caller's to
    null).
    """
    if not block_tables or geometry.full_history_group_id is None:
        return None
    raw = block_tables.get(geometry.full_history_group_id)
    if raw is None:
        return None
    if raw.shape[0] < bs:
        raise RuntimeError(
            f"full-attention table has {raw.shape[0]} rows but the "
            f"batch has {bs} requests"
        )
    return expand_history_table(
        raw,
        history_block_granularity=geometry.history_block_granularity
        or kernel_page_size,
        kernel_page_size=kernel_page_size,
        max_kernel_pages=max_kernel_pages,
        out=out,
    )


def expand_history_table(
    raw: torch.Tensor,
    history_block_granularity: int,
    kernel_page_size: int,
    max_kernel_pages: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Expand a batch-ordered raw table (scheduler pages of the full-history
    grain) into kernel pages, ``max_kernel_pages`` wide.

    The staged draft page table and the wrapper's group tables share this
    one mapping (the write-location math ``table[i, pos // P] * P + pos % P``
    is page-size invariant, so the expansion is the only grain-sensitive
    step).
    """
    return expand_page_table(
        raw,
        block_granularity=history_block_granularity,
        kernel_page_size=kernel_page_size,
        max_kernel_pages=max_kernel_pages,
        out=out,
    )
