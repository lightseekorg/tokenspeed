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

from dataclasses import replace

import pytest
import tokenspeed_kernel.benchmark.generators.gdn as gdn_generator
import tokenspeed_kernel.ops.attention.gdn as gdn_ops
import torch
from tokenspeed_kernel.benchmark.harness import (
    BenchmarkCaseError,
    BenchmarkRequest,
    BenchmarkStatus,
)
from tokenspeed_kernel.platform import current_platform
from tokenspeed_kernel.registry import KernelSpec

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")


def _request(mode: str, **parameters: object) -> BenchmarkRequest:
    return BenchmarkRequest(
        family="attention",
        mode=mode,
        parameters={
            "model_profile": "qwen38_flash_tp2",
            "dtype": "bfloat16",
            "state_dtype": "float32",
            **parameters,
        },
        solution=None,
        registration=None,
        cold_cache=True,
        seed=42,
    )


def _capture_selections(monkeypatch: pytest.MonkeyPatch) -> list[tuple]:
    captured: list[tuple] = []

    def select(family, mode, signature, **kwargs):
        captured.append((family, mode, signature, kwargs["traits"]))
        raise RuntimeError("selection captured")

    monkeypatch.setattr(gdn_ops, "select_kernel", select)
    monkeypatch.setattr(gdn_generator, "select_kernel", select)
    monkeypatch.setattr(gdn_generator, "load_builtin_kernels", lambda: None)
    return captured


def _decode_args(batch: int, tokens: int) -> dict[str, torch.Tensor]:
    def empty(*shape: int, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
        return torch.empty(shape, dtype=dtype, device="cuda")

    return {
        "q": empty(batch, tokens, 8, 128),
        "k": empty(batch, tokens, 8, 128),
        "v": empty(batch, tokens, 24, 128),
        "A_log": empty(24, dtype=torch.float32),
        "a": empty(batch, tokens, 24),
        "dt_bias": empty(24, dtype=torch.float32),
        "b": empty(batch, tokens, 24),
        "initial_state": empty(batch, 24, 128, 128, dtype=torch.float32),
        "initial_state_indices": torch.arange(batch, device="cuda", dtype=torch.int32),
    }


def test_gdn_decode_generators_select_like_the_operations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _capture_selections(monkeypatch)
    with pytest.raises(RuntimeError, match="selection captured"):
        gdn_ops.gdn_decode_step(**_decode_args(4, 1))
    with pytest.raises(RuntimeError, match="selection captured"):
        gdn_generator.prepare_gdn_decode_step(
            _request("gdn_decode_step", batch=4), None
        )
    with pytest.raises(RuntimeError, match="selection captured"):
        gdn_ops.gdn_decode_mtp(**_decode_args(4, 4), tree_ancestors=None)
    with pytest.raises(RuntimeError, match="selection captured"):
        gdn_generator.prepare_gdn_decode_mtp(
            _request("gdn_decode_mtp", batch=4, tokens=4, state_writes="none"), None
        )

    assert captured[0] == captured[1]
    assert captured[2] == captured[3]


def test_gdn_chunk_prefill_generator_selects_like_the_operation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _capture_selections(monkeypatch)
    q = torch.empty((1, 64, 8, 128), dtype=torch.bfloat16, device="cuda")
    v = torch.empty((1, 64, 24, 128), dtype=torch.bfloat16, device="cuda")
    with pytest.raises(RuntimeError, match="selection captured"):
        gdn_ops.gdn_chunk_prefill(
            q,
            q,
            v,
            torch.empty((1, 64, 24), dtype=torch.float32, device="cuda"),
            torch.empty((1, 64, 24), dtype=torch.bfloat16, device="cuda"),
            scale=None,
            initial_state=torch.empty((2, 24, 128, 128), device="cuda"),
            cu_seqlens=torch.tensor([0, 32, 64], dtype=torch.int32, device="cuda"),
            qk_l2norm=True,
        )
    with pytest.raises(RuntimeError, match="selection captured"):
        gdn_generator.prepare_gdn_chunk_prefill(
            _request("gdn_chunk_prefill", sequences=2, tokens_per_sequence=32), None
        )

    assert captured[0] == captured[1]


def test_gdn_staged_verify_writes_each_position_to_its_own_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = KernelSpec.__new__(KernelSpec)
    monkeypatch.setattr(gdn_generator, "_select_registration", lambda *_args: spec)
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(gdn_ops, "gdn_decode_mtp", lambda *_a, **kw: calls.append(kw))

    prepared = gdn_generator.prepare_gdn_decode_mtp(
        _request("gdn_decode_mtp", batch=3, tokens=4, state_writes="per_token"), None
    )
    prepared.invocation.invoke()

    (call,) = calls
    rows = call["output_state_indices"]
    assert call["disable_state_update"] is False
    assert rows.shape == (3, 4)
    assert torch.unique(rows).numel() == 12
    assert int(rows.min()) >= 3
    assert int(rows.max()) < call["initial_state"].shape[0]


@pytest.mark.parametrize("state_dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("state_writes", [None, "none", "per_token"])
def test_gdn_decode_benchmark_runs_with_explicit_state_dtype(state_dtype, state_writes):
    if state_writes is None:
        prepare = gdn_generator.prepare_gdn_decode_step
        request = _request("gdn_decode_step", batch=2, state_dtype=state_dtype)
        tokens = 1
    else:
        prepare = gdn_generator.prepare_gdn_decode_mtp
        tokens = 4
        request = _request(
            "gdn_decode_mtp",
            batch=2,
            tokens=tokens,
            state_writes=state_writes,
            state_dtype=state_dtype,
        )
    prepared = prepare(replace(request, solution="triton"), current_platform())
    output = prepared.invocation.invoke()
    assert output.shape == (2, tokens, 24, 128)
    assert torch.isfinite(output).all()
    assert prepared.parameters["state_dtype"] == state_dtype


@pytest.mark.parametrize("state_dtype", [None, "float16"])
@pytest.mark.parametrize("mode", ["gdn_decode_step", "gdn_decode_mtp"])
def test_gdn_decode_benchmark_rejects_missing_or_unsupported_state_dtype(
    mode, state_dtype
):
    request = _request(mode, batch=2, tokens=4, state_writes="none")
    parameters = dict(request.parameters)
    if state_dtype is None:
        del parameters["state_dtype"]
    else:
        parameters["state_dtype"] = state_dtype
    prepare = (
        gdn_generator.prepare_gdn_decode_step
        if mode == "gdn_decode_step"
        else gdn_generator.prepare_gdn_decode_mtp
    )
    with pytest.raises(BenchmarkCaseError, match="state_dtype") as error:
        prepare(replace(request, parameters=parameters), current_platform())
    assert error.value.status == BenchmarkStatus.INVALID_CASE
