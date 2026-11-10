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

import json
import time
from collections.abc import Iterator
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from tokenspeed_kernel import selection
from tokenspeed_kernel.ops.tuning import kernel_warmup, set_autotune_process_group
from tokenspeed_kernel.registry import KernelRegistry, Priority, register_kernel
from tokenspeed_kernel.selection import (
    AutotuneCache,
    AutotuneParams,
    SelectionPolicy,
    SelectionStrategy,
    autotune_decide,
    autotune_key,
    default_autotune_cache,
    register_autotune_benchmark,
    select_kernel,
    set_selection_policy,
    source_digest,
)
from tokenspeed_kernel.signature import format_signatures

pytestmark = pytest.mark.usefixtures("fresh_registry")

INPUT_BF16 = next(iter(format_signatures("input", "dense", {torch.bfloat16})))
ENVIRONMENT = {"schema": 1, "device": "test", "compute_units": 256}


@pytest.fixture
def cache_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    monkeypatch.setenv("TOKENSPEED_KERNEL_AUTOTUNE_CACHE_DIR", str(tmp_path))
    default_autotune_cache.cache_clear()
    yield tmp_path
    default_autotune_cache.cache_clear()


@pytest.fixture
def autotune_policy() -> Iterator[None]:
    set_selection_policy(
        SelectionPolicy(
            op_strategies={("toy", "forward"): SelectionStrategy.AUTOTUNE},
            autotune_params=AutotuneParams(
                warmup_iters=1, bench_iters=3, use_cuda_events=False
            ),
        )
    )
    yield
    set_selection_policy(SelectionPolicy())
    selection._autotune_benchmarks.pop(("toy", "forward"), None)


class _Measure:
    def __init__(self, value: object) -> None:
        self.value = value
        self.calls = 0

    def __call__(self) -> object:
        self.calls += 1
        return self.value


def test_cache_persists_per_environment(tmp_path: Path) -> None:
    cache = AutotuneCache.for_environment(tmp_path, ENVIRONMENT)
    cache.put("op|n=1|abc", {"winner": "a"}, persist=True)

    reread = AutotuneCache.for_environment(tmp_path, ENVIRONMENT)
    other_gpu = AutotuneCache.for_environment(
        tmp_path, ENVIRONMENT | {"compute_units": 304}
    )

    assert reread.get("op|n=1|abc") == {"winner": "a"}
    assert other_gpu.get("op|n=1|abc") is None
    assert other_gpu.path != reread.path
    assert json.loads(reread.path.read_text())["environment"] == ENVIRONMENT


def test_cache_writers_merge_entries(tmp_path: Path) -> None:
    first = AutotuneCache.for_environment(tmp_path, ENVIRONMENT)
    second = AutotuneCache.for_environment(tmp_path, ENVIRONMENT)

    first.put("a", 1, persist=True)
    second.put("b", 2, persist=True)

    merged = AutotuneCache.for_environment(tmp_path, ENVIRONMENT)
    assert (merged.get("a"), merged.get("b")) == (1, 2)


def test_key_names_op_shape_and_sources() -> None:
    sources = source_digest(json, Path)

    key = autotune_key("gemm.mm", {"n": 5120, "k": 6144}, sources)

    assert key == f"gemm.mm|k=6144,n=5120|{sources}"
    assert source_digest(Path, json) == sources
    assert source_digest(json) != sources


def test_decide_measures_only_during_warmup(tmp_path: Path) -> None:
    cache = AutotuneCache.for_environment(tmp_path, ENVIRONMENT)
    measure = _Measure("fast")

    assert autotune_decide(cache, "k", measure) is None
    with kernel_warmup(enabled=True):
        assert autotune_decide(cache, "k", measure) == "fast"
        assert autotune_decide(cache, "k", measure) == "fast"
    assert autotune_decide(cache, "k", measure) == "fast"

    assert measure.calls == 1
    assert AutotuneCache.for_environment(tmp_path, ENVIRONMENT).get("k") == "fast"


def test_decide_broadcasts_from_rank_zero(tmp_path: Path) -> None:
    dist.init_process_group(
        "gloo", init_method=f"file://{tmp_path / 'store'}", rank=0, world_size=1
    )
    group = dist.new_group(backend="gloo")
    set_autotune_process_group(group)
    try:
        cache = AutotuneCache.for_environment(tmp_path, ENVIRONMENT)
        measure = _Measure({"rows": [2, 4]})
        with kernel_warmup(enabled=True):
            first = autotune_decide(cache, "k", measure)
            second = autotune_decide(cache, "k", measure)
    finally:
        set_autotune_process_group(None)
        dist.destroy_process_group()

    assert first == second == {"rows": [2, 4]}
    assert measure.calls == 1
    assert AutotuneCache.for_environment(tmp_path, ENVIRONMENT).get("k") == first


def _register_toy_kernels() -> None:
    @register_kernel(
        "toy",
        "forward",
        name="toy_ranked_first",
        solution="python",
        signatures={INPUT_BF16},
        priority=Priority.SPECIALIZED,
    )
    def ranked_first(delay: float) -> None:
        time.sleep(delay * 4)

    @register_kernel(
        "toy",
        "forward",
        name="toy_faster",
        solution="python",
        signatures={INPUT_BF16},
        priority=Priority.PORTABLE,
    )
    def faster(delay: float) -> None:
        time.sleep(delay)


def test_autotune_strategy_selects_the_measured_fastest(
    cache_dir: Path, autotune_policy: None
) -> None:
    _register_toy_kernels()
    launches: list[str] = []

    def benchmark(impl, traits):
        launches.append(impl.__name__)
        return lambda: impl(1e-3)

    register_autotune_benchmark("toy", "forward", benchmark)

    before = select_kernel("toy", "forward", INPUT_BF16)
    with kernel_warmup(enabled=True):
        during = select_kernel("toy", "forward", INPUT_BF16)
    after = select_kernel("toy", "forward", INPUT_BF16)

    # A miss outside warmup keeps the heuristic and stays uncached.
    assert before.name == "toy_ranked_first"
    assert during.name == after.name == "toy_faster"
    assert sorted(launches) == ["faster", "ranked_first"]

    # A new process reads the decision without measuring.
    KernelRegistry.get().clear_cache()
    default_autotune_cache.cache_clear()
    assert select_kernel("toy", "forward", INPUT_BF16).name == "toy_faster"
    assert len(launches) == 2


def test_autotune_strategy_without_benchmark_ranks(
    cache_dir: Path, autotune_policy: None
) -> None:
    _register_toy_kernels()

    with kernel_warmup(enabled=True):
        selected = select_kernel("toy", "forward", INPUT_BF16)

    assert selected.name == "toy_ranked_first"
    assert not list(cache_dir.iterdir())
