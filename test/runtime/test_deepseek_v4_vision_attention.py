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

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from tokenspeed.runtime.engine.scheduler_utils import validate_atomic_spans_flat
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.execution.prefill_graph import PrefillGraph
from tokenspeed.runtime.layers.attention.backends.specific.deepseek_v4 import (
    DeepseekV4AttentionBackend,
)
from tokenspeed.runtime.layers.attention.deepseek_v4.metadata import (
    DeepseekV4ForwardMetadata,
)
from tokenspeed.runtime.layers.attention.kv_cache.hybrid_deepseek_v4 import (
    DeepseekV4CacheMetadata,
)


def _backend() -> DeepseekV4AttentionBackend:
    backend = object.__new__(DeepseekV4AttentionBackend)
    backend.vision_enabled = True
    backend.vision_max_n_token = 384
    return backend


def _metadata(
    *,
    seq_len: int = 20,
    query_len: int = 10,
    atomic_spans: list[tuple[int, int]] | None = None,
    visibility_spans: list[tuple[int, int]] | None = None,
    contained_atomic: list[tuple[int, int]] | None = None,
    contained_visibility: list[tuple[int, int]] | None = None,
) -> DeepseekV4ForwardMetadata:
    atomic_spans = atomic_spans if atomic_spans is not None else [(12, 18)]
    visibility_spans = visibility_spans if visibility_spans is not None else [(15, 18)]
    contained_atomic = (
        contained_atomic if contained_atomic is not None else list(atomic_spans)
    )
    contained_visibility = (
        contained_visibility
        if contained_visibility is not None
        else list(visibility_spans)
    )
    return DeepseekV4ForwardMetadata(
        seq_lens=torch.tensor([seq_len], dtype=torch.int32),
        query_lens=torch.tensor([query_len], dtype=torch.int32),
        query_start_loc=torch.tensor([0, query_len], dtype=torch.int32),
        token_to_req_indices=torch.zeros(query_len, dtype=torch.int32),
        cache=DeepseekV4CacheMetadata(
            page_size=64,
            page_table=torch.zeros((1, 1), dtype=torch.int32),
            swa_page_table=torch.zeros((1, 1), dtype=torch.int32),
        ),
        seq_lens_cpu=torch.tensor([seq_len], dtype=torch.int32),
        query_lens_cpu=torch.tensor([query_len], dtype=torch.int32),
        num_prefill_reqs=1,
        num_prefill_tokens=query_len,
        forward_mode=ForwardMode.EXTEND,
        vision_left=torch.zeros(query_len, dtype=torch.int32),
        vision_right=torch.zeros(query_len, dtype=torch.int32),
        vision_atomic_spans=[atomic_spans],
        vision_visibility_spans=[visibility_spans],
        vision_atomic_spans_in_chunk=[contained_atomic],
        vision_visibility_spans_in_chunk=[contained_visibility],
    )


def test_visible_attention_marks_only_non_pad_image_rows() -> None:
    metadata = _metadata()
    left, right = _backend()._prepare_prefill_visibility(
        metadata, device=torch.device("cpu")
    )
    assert left.tolist() == [0, 0, 0, 0, 0, 0, 1, 2, 3, 0]
    assert right.tolist() == [0, 0, 0, 0, 0, 3, 2, 1, 0, 0]
    assert metadata.vision_prefill_validated


def test_partial_gather_keeps_visible_indices_inside_absolute_window() -> None:
    metadata = _metadata(
        seq_len=1024,
        query_len=16,
        atomic_spans=[(1008, 1023)],
        visibility_spans=[(1011, 1023)],
    )
    left, right = _backend()._prepare_prefill_visibility(
        metadata, device=torch.device("cpu")
    )
    assert left.tolist()[:3] == [0, 0, 0]
    assert left.tolist()[3:] == list(range(13))
    assert right.tolist()[3:] == list(reversed(range(13)))


def test_prefill_graph_remains_available_only_without_multimodal_context() -> None:
    graph = PrefillGraph.__new__(PrefillGraph)
    graph._multimodal_graph_safe = False
    graph._multimodal_input_embeds = lambda *_: None
    graph._replay_bucket = lambda _: 64
    assert graph.can_run(SimpleNamespace(), None)
    assert not graph.can_run(SimpleNamespace(), SimpleNamespace())


def test_failure_prefix_hit_inside_span_is_actionable() -> None:
    metadata = _metadata(
        atomic_spans=[(8, 12)],
        visibility_spans=[(11, 12)],
        contained_atomic=[(8, 12)],
        contained_visibility=[(11, 12)],
    )
    with pytest.raises(RuntimeError, match="atomic span .* partially overlaps") as exc:
        _backend()._prepare_prefill_visibility(metadata, device=torch.device("cpu"))
    print(f"WP06_FAILURE case=prefix_hit_inside_span outcome={exc.value}")


def test_failure_chunk_boundary_inside_span_is_actionable() -> None:
    metadata = _metadata(
        seq_len=16,
        query_len=8,
        atomic_spans=[(6, 10)],
        visibility_spans=[(9, 10)],
        contained_atomic=[(6, 10)],
        contained_visibility=[(9, 10)],
    )
    with pytest.raises(RuntimeError, match="atomic span .* partially overlaps") as exc:
        _backend()._prepare_prefill_visibility(metadata, device=torch.device("cpu"))
    print(f"WP06_FAILURE case=chunk_boundary_inside_span outcome={exc.value}")


def test_failure_image_block_larger_than_budget_is_actionable() -> None:
    with pytest.raises(ValueError, match="length 9.*--chunked-prefill-size 8") as exc:
        validate_atomic_spans_flat([4, 12], num_tokens=16, max_scheduled_tokens=8)
    print(f"WP06_FAILURE case=block_larger_than_budget outcome={exc.value}")
