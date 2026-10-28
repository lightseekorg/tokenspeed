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

"""Check that live selected lengths leave sparse launch parameters unchanged."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
import torch

pytest.importorskip("tokenspeed_kernel_amd")

from tokenspeed_kernel_amd.ops.gfx950.attention.dsa import attention
from tokenspeed_kernel_amd.ops.gfx950.attention.dsa.attention import (
    launch_gluon_dsa_decode_gfx950,
    launch_gluon_dsa_prefill_gfx950,
)


@pytest.mark.parametrize(
    ("launch", "tokens"),
    (
        pytest.param(launch_gluon_dsa_decode_gfx950, 8, id="decode"),
        pytest.param(launch_gluon_dsa_prefill_gfx950, 16, id="prefill"),
    ),
)
@pytest.mark.parametrize(
    "q_dtype", (torch.bfloat16, torch.float8_e4m3fn, torch.float8_e5m2)
)
@pytest.mark.parametrize(
    "kv_dtype", (torch.bfloat16, torch.float8_e4m3fn, torch.float8_e5m2)
)
@pytest.mark.parametrize("rope_dim", (0, 64))
def test_dense_launch_parameters_ignore_live_length(
    monkeypatch: pytest.MonkeyPatch,
    launch: Callable[..., torch.Tensor],
    tokens: int,
    q_dtype: torch.dtype,
    kv_dtype: torch.dtype,
    rope_dim: int,
) -> None:
    launches: list[tuple[torch.Size, tuple[int, int, int], dict[str, Any]]] = []

    class CaptureKernel:
        def __getitem__(self, grid: Any) -> Any:
            def launch(*args: Any, **kwargs: Any) -> None:
                launches.append(
                    (
                        args[2].shape,
                        grid({"BLOCK_H": kwargs["BLOCK_H"]}),
                        kwargs,
                    )
                )

            return launch

    class IgnoreReduce:
        def __getitem__(self, grid: Any) -> Any:
            del grid
            return lambda *args, **kwargs: None

    monkeypatch.setattr(attention, "_dsa_dense_mfma_kv_kernel", CaptureKernel())
    monkeypatch.setattr(attention, "_dsa_dense_mfma_reduce_kernel", IgnoreReduce())

    q = torch.empty((tokens, 16, 512 + rope_dim), dtype=q_dtype)
    kv = torch.empty((2051, 512 + rope_dim), dtype=kv_dtype)
    slots = torch.zeros((tokens, 2051), dtype=torch.int32)
    lengths = torch.empty((tokens,), dtype=torch.int32)

    for live_length, max_seqlen_k in (
        (0, 0),
        (13, 13),
        (63, 63),
        (64, 64),
        (65, 65),
        (2048, 50_000),
        (2051, 50_000),
    ):
        lengths.fill_(live_length)
        launch(
            q=q,
            kv_cache=kv,
            sparse_kv_cache=None,
            topk_slots=slots,
            topk_lens=lengths,
            max_seqlen_k=max_seqlen_k,
            qk_nope_head_dim=256,
            kv_lora_rank=512,
            qk_rope_head_dim=rope_dim,
            softmax_scale=1.0,
            page_size=64,
        )

    assert len(launches) == 7
    assert all(launch == launches[0] for launch in launches[1:])
    width, grid, constants = launches[0]
    assert width == torch.Size((tokens, 2051))
    assert grid[0] == tokens
    assert grid[2] == constants["NUM_KV_SPLITS"]
    assert "PHASED_E4M3" not in constants
    assert "COLUMN_PARALLEL" not in constants
    assert constants["CONVERT_INPUTS"] == (q_dtype != kv_dtype)
    assert constants["NATIVE_FP8"] == (
        q_dtype == kv_dtype and q_dtype in attention._FP8_DTYPES
    )
    assert constants["NATIVE_E5M2"] == (q_dtype == kv_dtype == torch.float8_e5m2)


def test_scalar_rank128_still_trims_selected_width(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    widths: list[int] = []

    def fake_dense(
        q: torch.Tensor,
        kv_cache: torch.Tensor,
        topk_slots: torch.Tensor,
        topk_lens: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        del kv_cache, topk_lens, kwargs
        widths.append(topk_slots.shape[1])
        return torch.empty((q.shape[0], q.shape[1], 512), dtype=torch.bfloat16)

    monkeypatch.setattr(attention, "_run_dense_kv_scalar", fake_dense)
    q = torch.empty((1, 16, 128), dtype=torch.bfloat16)
    kv = torch.empty((2051, 128), dtype=torch.bfloat16)
    slots = torch.zeros((1, 2051), dtype=torch.int32)
    lengths = torch.ones((1,), dtype=torch.int32)
    for max_seqlen_k in (511, 2051):
        attention.launch_gluon_dsa_decode_gfx950(
            q=q,
            kv_cache=kv,
            sparse_kv_cache=None,
            topk_slots=slots,
            topk_lens=lengths,
            max_seqlen_k=max_seqlen_k,
            qk_nope_head_dim=256,
            kv_lora_rank=128,
            qk_rope_head_dim=0,
            softmax_scale=1.0,
            page_size=64,
        )
    assert widths == [512, 2051]
