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

"""Sink retention declarations and absolute decode graph table widths."""

from dataclasses import replace

import pytest

from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import (
    CacheGroupSpec,
    compute_max_logical_pages_for_capture,
)


def sliding_spec(*, sink_tokens=0, replayable=False):
    return CacheGroupSpec(
        group_id="swa",
        attn_kind="sliding_window",
        rows_per_page=64,
        entry_stride_tokens=1,
        sliding_window_tokens=128,
        sink_tokens=sink_tokens,
        replayable=replayable,
    )


@pytest.mark.parametrize("sink_tokens", [-1, True, 1.5, None])
def test_sink_requires_a_nonnegative_integer(sink_tokens):
    with pytest.raises(ValueError, match="nonnegative integer"):
        sliding_spec(sink_tokens=sink_tokens)


@pytest.mark.parametrize("attn_kind", ["full_history", "state"])
def test_sink_requires_sliding_storage(attn_kind):
    with pytest.raises(ValueError, match="requires a sliding-window group"):
        CacheGroupSpec(
            group_id="other", attn_kind=attn_kind, sink_tokens=1, replayable=False
        )


def test_sink_cannot_be_regenerated_by_bounded_replay():
    with pytest.raises(ValueError, match="bounded replay"):
        sliding_spec(sink_tokens=16, replayable=True)
    assert sliding_spec(replayable=True).sink_tokens == 0


@pytest.mark.parametrize(
    "overlap_depth,verify_width,window_columns,absolute_columns",
    [(0, 1, 4, 17), (1, 17, 4, 17), (1, 64, 5, 18), (1, 65, 6, 19)],
)
def test_sink_graph_tables_cover_absolute_rows(
    overlap_depth, verify_width, window_columns, absolute_columns
):
    ordinary = sliding_spec()
    sink = replace(ordinary, sink_tokens=16)
    context = 1024
    kwargs = dict(
        max_context_len=context,
        max_tokens_per_req=verify_width,
        overlap_schedule_depth=overlap_depth,
    )
    assert compute_max_logical_pages_for_capture(ordinary, **kwargs) == window_columns
    assert compute_max_logical_pages_for_capture(sink, **kwargs) == absolute_columns


@pytest.mark.parametrize("sink_tokens", [0, 16])
def test_sink_retention_survives_the_scheduler_bridge(sink_tokens):
    from tokenspeed.runtime.layers.attention.kv_cache.recipes.scheduler_bridge import (
        cache_group_config,
    )

    config = cache_group_config(
        sliding_spec(sink_tokens=sink_tokens),
        total_pages=64,
        cache_blocks_per_lcm_block=1,
    )
    assert config.sink_tokens == sink_tokens
    assert config.sliding_window_tokens == 128
