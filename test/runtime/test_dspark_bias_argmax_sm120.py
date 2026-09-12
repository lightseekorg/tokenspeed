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

"""Source-method consumer boundaries; dependency fixtures are not model E2E.

Set TOKENSPEED_BIAS_ARGMAX_BASELINE_DIR to the SHA-verified original file
snapshot used by the companion benchmark. CPU tests check host integration;
GPU-marked cases additionally execute the registered candidate kernels.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tokenspeed-kernel/benchmarks"))
from dspark_bias_argmax_support import (  # noqa: E402
    copy_consumer_parameters,
    load_consumers,
    make_consumer,
)

GPU = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0),
    reason="requires NVIDIA SM120",
)


@pytest.fixture(scope="module")
def classes():
    baseline_dir = os.environ.get("TOKENSPEED_BIAS_ARGMAX_BASELINE_DIR")
    if baseline_dir is None:
        pytest.skip("requires explicit original baseline snapshot directory")
    if torch.cuda.is_available():
        from tokenspeed_kernel.ops.sampling import (
            create_bias_argmax_workspace,
            try_bias_argmax,
        )
    else:
        # CPU host-branch evidence: explicit unavailable-kernel dependency;
        # no GPU package import or fabricated platform/capability.
        def create_bias_argmax_workspace(*args, **kwargs):
            return None

        def try_bias_argmax(*args, **kwargs):
            raise AssertionError("CPU boundary cannot execute GPU candidate")

    result, provenance = load_consumers(
        ROOT, baseline_dir, create_bias_argmax_workspace, try_bias_argmax
    )
    assert provenance["A"]["dflash"]["sha256"] != provenance["P"]["dflash"]["sha256"]
    return result


def _consumer(classes, arm, device):
    return make_consumer(classes, arm, 2, 5, 4097, 16, 8, torch.float32, device)


def test_bias_closure_is_once_when_candidate_rejects(classes):
    consumer = _consumer(classes, "P", "cpu")
    consumer._bias_argmax_workspace = object()
    hidden = torch.zeros((2, 16))
    logits = torch.randn((2, 4097))
    bias = torch.randn_like(logits)
    output = torch.full((2, 5), -37, dtype=torch.int32)
    events = []
    consumer._ensure_dist_argmax_state = lambda *args: events.append("probe")

    def bias_fn(start, count):
        events.append("bias")
        assert (start, count) == (0, 4097)
        return bias

    def reject(**kwargs):
        events.append("try")
        return False

    namespace = consumer._greedy_argmax_vocab_parallel.__func__.__globals__
    with mock.patch.dict(namespace, try_bias_argmax=reject):
        result = consumer._greedy_argmax_vocab_parallel(
            hidden, out=output[:, 2], bias_fn=bias_fn, base_logits=logits
        )
    assert events == ["probe", "bias", "try"]
    assert result.data_ptr() == output[:, 2].data_ptr()
    torch.testing.assert_close(result, (logits + bias).argmax(-1).to(torch.int32))
    assert torch.all(output[:, [0, 1, 3, 4]] == -37)


def test_out_none_keeps_fresh_result_and_once_bias(classes):
    consumer = _consumer(classes, "P", "cpu")
    consumer._bias_argmax_workspace = object()
    logits = torch.randn((2, 4097))
    bias = torch.randn_like(logits)
    hidden = torch.zeros((2, 16))
    count = []

    def bias_fn(start, width):
        count.append((start, width))
        return bias

    namespace = consumer._greedy_argmax_vocab_parallel.__func__.__globals__
    with mock.patch.dict(
        namespace,
        try_bias_argmax=mock.Mock(
            side_effect=AssertionError("must not try without out")
        ),
    ):
        first = consumer._greedy_argmax_vocab_parallel(
            hidden, out=None, bias_fn=bias_fn, base_logits=logits
        )
        before = first.clone()
        logits[:, 23] = 100
        second = consumer._greedy_argmax_vocab_parallel(
            hidden, out=None, bias_fn=bias_fn, base_logits=logits
        )
    assert count == [(0, 4097), (0, 4097)]
    assert first.dtype == torch.int32 and first.data_ptr() != second.data_ptr()
    torch.testing.assert_close(first, before)


def test_collective_probe_precedes_empty_shard_branch(classes):
    consumer = _consumer(classes, "P", "cpu")
    consumer.lm_head.shard_indices.num_org_elements = 0
    events = []
    consumer._ensure_dist_argmax_state = lambda *args: events.append("probe")

    def unused_bias(*args):
        events.append("bias")
        raise AssertionError("empty shard cannot call bias")

    result = consumer._greedy_argmax_vocab_parallel(
        torch.zeros((2, 16)), out=None, bias_fn=unused_bias, base_logits=None
    )
    assert events == ["probe"]
    assert torch.equal(result, torch.zeros((2,), dtype=torch.int32))


@pytest.mark.parametrize(
    "guard", ["no_bias", "no_workspace", "added_vocab", "dist_state", "tp2"]
)
def test_unsupported_scope_never_calls_candidate(classes, guard):
    consumer = _consumer(classes, "P", "cpu")
    consumer._bias_argmax_workspace = None if guard == "no_workspace" else object()
    if guard == "added_vocab":
        consumer.lm_head.shard_indices.num_added_elements = 1
        consumer.lm_head.weight = torch.cat(
            (consumer.lm_head.weight, torch.zeros((1, 16)))
        )
    if guard == "tp2":
        consumer.logits_processor.tp_size = 2
        # Stop at the existing gather branch after the guarded candidate check.
        consumer._ensure_greedy_gather_buffers = mock.Mock(
            side_effect=RuntimeError("reached TP fallback")
        )
    if guard == "dist_state":
        consumer._dist_argmax_state = object()
    bias = torch.zeros((2, 4097))

    def bias_fn(start, count):
        return bias[:, :count]

    namespace = consumer._greedy_argmax_vocab_parallel.__func__.__globals__
    dist = lambda state, logits: (None, logits.argmax(-1))
    with mock.patch.dict(
        namespace,
        try_bias_argmax=mock.Mock(side_effect=AssertionError("unsupported scope")),
        _dist_argmax=dist,
    ):
        call = lambda: consumer._greedy_argmax_vocab_parallel(
            torch.zeros((2, 16)),
            out=torch.empty((2,), dtype=torch.int32),
            bias_fn=None if guard == "no_bias" else bias_fn,
            base_logits=torch.randn((2, 4097)),
        )
        if guard == "tp2":
            with pytest.raises(RuntimeError, match="reached TP fallback"):
                call()
        else:
            call()


def test_wire_target_allocates_after_binding_and_checks(classes):
    consumer = classes["P"][0]()
    consumer.spec_algorithm = "DSPARK"
    consumer.target_layer_ids = [2, 7]
    consumer.input_buffers = SimpleNamespace(max_bs=16)
    events = []
    head = SimpleNamespace(
        weight=torch.empty((4097, 16)),
        shard_indices=SimpleNamespace(num_org_elements=4097, num_added_elements=0),
    )
    target = SimpleNamespace(
        lm_head=head,
        logits_processor=SimpleNamespace(tp_size=1),
        get_input_embeddings=lambda: object(),
        set_dflash_layers_to_capture=lambda layers: events.append(
            ("capture_layers", layers)
        ),
    )
    consumer._wire_aux_hidden_stream = lambda bound: events.append(
        ("aux", bound is target)
    )
    marker = object()

    def factory(**kwargs):
        assert consumer.lm_head is head and consumer.target_model is target
        events.append(("allocate", kwargs))
        return marker

    namespace = consumer.wire_target.__func__.__globals__
    with mock.patch.dict(namespace, create_bias_argmax_workspace=factory):
        consumer.wire_target(target)
    assert [event[0] for event in events] == ["capture_layers", "aux", "allocate"]
    assert events[-1][1] == {
        "max_rows": 16,
        "vocab_size": 4097,
        "device": head.weight.device,
    }
    assert consumer._bias_argmax_workspace is marker


@pytest.mark.parametrize("failure", ["missing_capture", "aux_failure"])
def test_invalid_target_cannot_allocate_workspace(classes, failure):
    consumer = classes["P"][0]()
    consumer.spec_algorithm = "DSPARK"
    consumer.target_layer_ids = [2]
    consumer.input_buffers = SimpleNamespace(max_bs=16)
    target = SimpleNamespace(
        lm_head=object(),
        logits_processor=SimpleNamespace(tp_size=1),
        get_input_embeddings=lambda: object(),
    )
    if failure == "aux_failure":
        target.set_dflash_layers_to_capture = lambda layers: None
    consumer._wire_aux_hidden_stream = mock.Mock(
        side_effect=ValueError("invalid aux stream")
    )
    namespace = consumer.wire_target.__func__.__globals__
    factory = mock.Mock(side_effect=AssertionError("must validate first"))
    with mock.patch.dict(namespace, create_bias_argmax_workspace=factory):
        with pytest.raises(ValueError):
            consumer.wire_target(target)
    factory.assert_not_called()


@GPU
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_actual_proposal_walk_matches_original_with_changing_anchors(classes, dtype):
    consumers = {
        arm: make_consumer(classes, arm, 2, 5, 4097, 32, 8, dtype, "cuda")
        for arm in ("A", "P")
    }
    copy_consumer_parameters(consumers["A"], consumers["P"])
    hidden = torch.randn((2, 4, 32), dtype=dtype, device="cuda")
    block_ids = torch.zeros((2, 5), dtype=torch.int32, device="cuda")
    outputs = {
        arm: torch.full((2, 5), -37, dtype=torch.int32, device="cuda")
        for arm in ("A", "P")
    }
    with torch.no_grad():
        for anchor in (-3, 0, 173, 5000):
            block_ids[:, 0] = anchor
            for arm in ("A", "P"):
                consumers[arm]._sample_block(hidden, block_ids, outputs[arm])
            torch.testing.assert_close(outputs["A"], outputs["P"], rtol=0, atol=0)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        graphs = {}
        with torch.cuda.stream(stream):
            for arm in ("A", "P"):
                graphs[arm] = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graphs[arm], stream=stream):
                    consumers[arm]._sample_block(hidden, block_ids, outputs[arm])
        torch.cuda.current_stream().wait_stream(stream)
        for anchor in (24, 89, 0):
            block_ids[:, 0] = anchor
            hidden.normal_()
            for graph in graphs.values():
                graph.replay()
            torch.testing.assert_close(outputs["A"], outputs["P"], rtol=0, atol=0)
