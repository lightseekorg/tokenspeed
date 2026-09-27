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
"""Prefill-graph admission: a side-effect-free query and the command it mirrors.

The capture loop used to learn whether a shape could be captured from the
return value of ``prepare_prefill_metadata``, which writes backend metadata.
It now asks ``admits_prefill_graph`` first and calls the command only for
shapes that query admits; a command that then refuses is an error.
"""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=10, suite="runtime-1gpu")

from tokenspeed.runtime.execution.forward_batch_info import ForwardMode  # noqa: E402
from tokenspeed.runtime.execution.memory_delta import (  # noqa: E402
    NULL_MEMORY_DELTA_OBSERVER,
)
from tokenspeed.runtime.execution.prefill_graph import (  # noqa: E402
    PrefillGraph,
    dummy_batch_size,
)
from tokenspeed.runtime.layers.attention.backends.base import (  # noqa: E402
    AttentionBackend,
)
from tokenspeed.runtime.layers.attention.backends.hybrid.linear import (  # noqa: E402
    HybridLinearAttnBackend,
)
from tokenspeed.runtime.layers.attention.backends.state.kda import (  # noqa: E402
    KDA_PREFILL_BACKENDS,
    KdaAttnBackend,
)


def _graph(buckets, *, sizes, query, command, dp_size=1, narrowing=False):
    """A PrefillGraph carrying only what ``_capture_all_buckets`` reads.

    ``query`` and ``command`` are separate so a test can make them disagree;
    every command call is recorded as ``(bucket, bs, capture)``.
    """
    graph = PrefillGraph.__new__(PrefillGraph)
    graph.disable = False
    graph.decoder_buckets = []
    graph.capture_buckets = list(buckets)
    graph._narrowing = SimpleNamespace() if narrowing else None
    graph.config = SimpleNamespace(
        global_rank=1,
        device="cpu",
        gpu_id=0,
        context_len=4096,
        max_num_seqs=8,
        data_parallel_size=dp_size,
        prefill_graph_capture_batch_sizes=sizes,
    )
    graph.dp_size = dp_size
    graph.commands = []

    def prepare(bucket, bs, _mode, *, capture):
        graph.commands.append((bucket, bs, capture))
        return command(bucket, bs)

    graph.attn_backend = SimpleNamespace(
        admits_prefill_graph=lambda bucket, bs, mode: (
            mode.is_extend() and query(bucket, bs)
        ),
        prepare_prefill_metadata=prepare,
    )
    graph.input_buffers = SimpleNamespace(
        input_ids_buf=torch.zeros(max(buckets), dtype=torch.int32)
    )
    graph._embed_tokens = lambda ids: ids
    graph._land_input_embeds = lambda _embeds, _bucket: None
    graph.make_dummy_batch = lambda _bucket, _bs: SimpleNamespace(
        capture_hidden_mode=None, forward_mode=ForwardMode.EXTEND
    )
    graph._captures = {}
    graph._encoders = {}
    graph._capture_bucket = lambda bucket, _wrapper, _observer: ("graph", bucket)
    graph._capture_encoder = lambda bucket, _wrapper, _observer: ("encoder", bucket)
    return graph


def test_the_request_count_is_the_ceil_it_replaced() -> None:
    """One helper for three copies of the same division; zero included."""
    for context_len in (0, 1, 7, 1024, 4096):
        for num_tokens in range(0, 20000, 37):
            assert dummy_batch_size(num_tokens, context_len) == -(
                -num_tokens // max(1, int(context_len))
            ), (num_tokens, context_len)


def test_only_admitted_shapes_reach_the_command() -> None:
    """A refused shape is skipped before anything is written for it."""
    graph = _graph(
        [16, 64, 32],
        sizes=[1, 2, 4],
        query=lambda bucket, bs: bs != 2 and bucket >= 32,
        command=lambda _bucket, _bs: True,
    )

    graph._capture_all_buckets(None, None, NULL_MEMORY_DELTA_OBSERVER)

    inline = {key for key in graph._captures if key[1] is not None}
    assert inline == {(64, 1), (64, 4), (32, 1), (32, 4)}
    assert {(64, None), (32, None), (16, None)} <= set(graph._captures)
    written = {(b, bs) for b, bs, capture in graph.commands if capture}
    assert written == inline


def test_a_command_that_refuses_an_admitted_shape_is_an_error() -> None:
    graph = _graph(
        [64],
        sizes=[1],
        query=lambda _bucket, _bs: True,
        command=lambda _bucket, bs: bs != 1,
    )

    with pytest.raises(RuntimeError, match=r"admitted \(64, 1\)"):
        graph._capture_all_buckets(None, None, NULL_MEMORY_DELTA_OBSERVER)


@pytest.mark.parametrize(
    "dp_size, narrowing, sizes",
    [(2, False, [5]), (2, False, [0]), (1, True, [0]), (1, True, [99])],
)
def test_invalid_capture_counts_still_fail_under_dp_and_narrowing(
    dp_size, narrowing, sizes
) -> None:
    """The counts are validated for every bucket, whatever the model or DP size.

    Mutation-checked: validating only where variants are captured lets an
    invalid ``--prefill-graph-capture-batch-sizes`` boot silently here.
    """
    graph = _graph(
        [64, 32],
        sizes=sizes,
        query=lambda _bucket, _bs: True,
        command=lambda _bucket, _bs: True,
        dp_size=dp_size,
        narrowing=narrowing,
    )

    with pytest.raises(ValueError, match="capture batch sizes"):
        graph._capture_all_buckets(None, None, NULL_MEMORY_DELTA_OBSERVER)


def test_dp_and_narrowing_capture_no_inline_shapes() -> None:
    for dp_size, narrowing in ((2, False), (1, True)):
        graph = _graph(
            [64, 32],
            sizes=[1, 2],
            query=lambda _bucket, _bs: True,
            command=lambda _bucket, _bs: True,
            dp_size=dp_size,
            narrowing=narrowing,
        )

        graph._capture_all_buckets(None, None, NULL_MEMORY_DELTA_OBSERVER)

        assert not any(capture for _b, _bs, capture in graph.commands)


def test_the_base_backend_admits_nothing_and_its_command_agrees() -> None:
    extend = ForwardMode.EXTEND
    assert AttentionBackend.admits_prefill_graph(object(), 8, 1, extend) is False
    for admits in (True, False):
        stub = SimpleNamespace(admits_prefill_graph=lambda *_a, **_k: admits)
        assert (
            AttentionBackend.prepare_prefill_metadata(stub, 8, 1, extend, capture=True)
            is admits
        )


def test_the_hybrid_wrapper_needs_its_own_counter_and_the_inner_answer() -> None:
    hybrid = HybridLinearAttnBackend.__new__(HybridLinearAttnBackend)
    for step_counter, inner, expected in (
        (None, True, True),
        (None, False, False),
        (object(), True, False),
    ):
        calls = []

        def _inner_admits(*args, _inner=inner):
            calls.append(("admits", *args))
            return _inner

        hybrid.step_counter = step_counter
        hybrid.linear_attn_backend = SimpleNamespace(
            admits_prefill_graph=_inner_admits,
            prepare_prefill_metadata=lambda *args, capture: calls.append(
                ("prepare", *args, capture)
            )
            or True,
        )
        decode = ForwardMode.DECODE
        assert hybrid.admits_prefill_graph(8, 1, decode) is expected
        assert hybrid.prepare_prefill_metadata(8, 1, decode, capture=True) is expected
        # The inner backend is asked the very same question, arguments untouched.
        asked = [c for c in calls if c[0] == "admits"]
        assert all(c == ("admits", 8, 1, decode) for c in asked)
        assert any(c[0] == "prepare" for c in calls) is expected


def test_kda_admits_only_the_enabled_cutedsl_extend_path() -> None:
    extend = ForwardMode.EXTEND
    backend = KdaAttnBackend.__new__(KdaAttnBackend)
    backend._prefill_graph_enabled = True
    backend.kda_backend = "cutedsl_kda"
    backend.step_counter = None
    assert backend.admits_prefill_graph(8, 1, extend) is True
    assert backend.admits_prefill_graph(8, 1, ForwardMode.DECODE) is False

    refusals = [("step_counter", object()), ("_prefill_graph_enabled", False)]
    refusals += [
        ("kda_backend", name) for name in KDA_PREFILL_BACKENDS if name != "cutedsl_kda"
    ]
    for attr, value in refusals:
        restore = getattr(backend, attr)
        setattr(backend, attr, value)
        assert backend.admits_prefill_graph(8, 1, extend) is False, (attr, value)
        assert backend.prepare_prefill_metadata(8, 1, extend, capture=True) is False, (
            attr,
            value,
        )
        setattr(backend, attr, restore)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
