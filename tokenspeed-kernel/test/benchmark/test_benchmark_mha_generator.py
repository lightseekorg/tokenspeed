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

import pytest
import tokenspeed_kernel.benchmark.generators.mha as mha_generator
import torch
from tokenspeed_kernel.benchmark import (
    GraphBenchmarkConfig,
    GraphTimer,
    KernelBenchmarkHarness,
)
from tokenspeed_kernel.benchmark.harness import (
    BenchmarkCaseError,
    BenchmarkRequest,
    BenchmarkStatus,
)
from tokenspeed_kernel.platform import current_platform
from tokenspeed_kernel.profiling import ShapeCapture

_HEADS = {"num_q_heads": 4, "num_kv_heads": 2, "head_dim": 64, "dtype": "bfloat16"}
_CASES = {
    "mha_decode_with_kvcache": {
        **_HEADS,
        "batch": 3,
        "q_len": 1,
        "context_tokens": 200,
        "max_seqlen_k": 1024,
        "page_size": 64,
        "window_left": 127,
    },
    "mha_prefill": {
        **_HEADS,
        "sequences": 2,
        "tokens_per_sequence": 96,
        "window_left": -1,
    },
    "mha_extend_with_kvcache": {
        **_HEADS,
        "sequences": 2,
        "extend_tokens": 64,
        "prefix_tokens": 100,
        "max_seqlen_k": 1024,
        "page_size": 64,
        "window_left": 127,
    },
    "gqa_prologue": {
        "num_q_heads": 4,
        "num_kv_heads": 2,
        "head_dim": 64,
        "dtype": "bfloat16",
        "tokens": 5,
        "qkv_row_stride": 640,
        "cache_slots": 64,
        "max_position": 256,
        "has_norm": True,
        "rope": "gptj",
        "return_kv": False,
    },
}
_PREPARE = {
    "mha_decode_with_kvcache": mha_generator.prepare_mha_decode,
    "mha_prefill": mha_generator.prepare_mha_prefill,
    "mha_extend_with_kvcache": mha_generator.prepare_mha_extend,
    "gqa_prologue": mha_generator.prepare_gqa_prologue,
}


def _request(mode: str, **updates: object) -> BenchmarkRequest:
    return BenchmarkRequest(
        family="attention",
        mode=mode,
        parameters={**_CASES[mode], **updates},
        solution=None,
        registration=None,
        cold_cache=True,
        seed=7,
    )


@pytest.mark.parametrize(
    ("mode", "updates", "match"),
    [
        ("mha_decode_with_kvcache", {"num_kv_heads": 3}, "divide"),
        ("mha_decode_with_kvcache", {"window_left": -2}, "window_left"),
        ("mha_decode_with_kvcache", {"context_tokens": 2048}, "max_seqlen_k"),
        ("mha_prefill", {"dtype": "float16"}, "dtype"),
        ("mha_extend_with_kvcache", {"prefix_tokens": 1000}, "max_seqlen_k"),
        ("gqa_prologue", {"rope": "yarn"}, "rope"),
        ("gqa_prologue", {"qkv_row_stride": 400}, "qkv_row_stride"),
        ("gqa_prologue", {"tokens": 65}, "cache_slots"),
    ],
)
def test_attention_generators_reject_invalid_parameters(
    mode: str, updates: dict[str, object], match: str
) -> None:
    with pytest.raises(BenchmarkCaseError, match=match):
        _PREPARE[mode](_request(mode, **updates), None)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
@pytest.mark.parametrize("mode", sorted(_CASES))
def test_attention_generator_times_the_registration_the_operation_selects(
    mode: str,
) -> None:
    request = _request(mode)
    prepared = _PREPARE[mode](request, current_platform())

    capture = ShapeCapture.get()
    capture.clear()
    capture.enabled = True
    try:
        prepared.invocation.invoke()
        torch.cuda.synchronize()
        dispatched = [record.kernel_name for record in capture._records]
    finally:
        capture.enabled = False
        capture.clear()
    assert dispatched == [prepared.registration.name]

    result = KernelBenchmarkHarness(
        GraphTimer(
            GraphBenchmarkConfig(eager_warmup_iterations=1, replay_warmup_iterations=1)
        ),
        platform_provider=current_platform,
    ).run(request, measurement_blocks=3)
    assert result.status is BenchmarkStatus.SUCCESS, result.to_dict()
    assert result.registration_name == prepared.registration.name
