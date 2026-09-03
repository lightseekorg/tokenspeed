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

"""FlashInfer NoPE sparse MLA tests for the GLM-5.3-Flash KPool handoff."""

from __future__ import annotations

import pytest
import torch
from tokenspeed_kernel import dsa_decode, dsa_prefill
from tokenspeed_kernel.registry import KernelRegistry

PAGE_SIZE = 64
LATENT_DIM = 512
TOPK_WIDTH = 2051
KV_SLOTS = 4096
ATOL = RTOL = 2.0e-2


def _flashinfer_nope_available() -> bool:
    return (
        torch.cuda.is_available()
        and torch.cuda.get_device_capability()[0] >= 10
        and KernelRegistry.get().get_by_name("flashinfer_trtllm_nope_dsa_decode")
        is not None
    )


pytestmark = pytest.mark.skipif(
    not _flashinfer_nope_available(),
    reason="requires Blackwell and FlashInfer native NoPE sparse MLA",
)


def _make_selection(
    lengths: list[int], generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    slots = torch.full(
        (len(lengths), TOPK_WIDTH),
        -1,
        dtype=torch.int32,
        device="cuda",
    )
    for row, length in enumerate(lengths):
        slots[row, :length] = torch.randperm(
            KV_SLOTS, generator=generator, device="cuda", dtype=torch.int64
        )[:length].to(torch.int32)
    return slots, torch.tensor(lengths, dtype=torch.int32, device="cuda")


def _reference(
    q: torch.Tensor,
    kv: torch.Tensor,
    slots: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.Tensor:
    output = torch.empty(q.shape, dtype=torch.bfloat16, device=q.device)
    scale = LATENT_DIM**-0.5
    for row, length in enumerate(lengths.tolist()):
        selected = kv.index_select(0, slots[row, :length].long()).float()
        scores = q[row].float() @ selected.T * scale
        output[row] = torch.softmax(scores, dim=-1) @ selected
    return output


def _inputs(
    lengths: list[int], dtype: torch.dtype, seed: int
) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    q = (
        torch.randn((len(lengths), 16, LATENT_DIM), device="cuda", generator=generator)
        * 0.1
    ).to(dtype)
    kv = (
        torch.randn((KV_SLOTS, LATENT_DIM), device="cuda", generator=generator) * 0.1
    ).to(dtype)
    return q, kv, *_make_selection(lengths, generator)


def _run(
    mode: str,
    q: torch.Tensor,
    kv: torch.Tensor,
    slots: torch.Tensor,
    lengths: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    api = dsa_decode if mode == "decode" else dsa_prefill
    return api(
        q=q,
        kv_cache=kv,
        sparse_kv_cache=None,
        topk_slots=slots,
        topk_lens=lengths,
        kv_seq_lens=torch.full_like(lengths, KV_SLOTS),
        max_seqlen_k=KV_SLOTS,
        qk_nope_head_dim=256,
        kv_lora_rank=LATENT_DIM,
        qk_rope_head_dim=0,
        softmax_scale=LATENT_DIM**-0.5,
        page_size=PAGE_SIZE,
        out=out,
        solution="flashinfer_trtllm",
    )


@pytest.mark.parametrize(
    ("mode", "dtype"),
    [
        pytest.param("decode", torch.bfloat16, id="decode-bf16"),
        pytest.param("prefill", torch.bfloat16, id="prefill-bf16"),
        pytest.param("decode", torch.float8_e4m3fn, id="decode-fp8"),
    ],
)
def test_flashinfer_nope_dsa_matches_reference(mode: str, dtype: torch.dtype) -> None:
    q, kv, slots, lengths = _inputs([TOPK_WIDTH, TOPK_WIDTH - 3], dtype, seed=17)
    out = torch.empty(q.shape, device="cuda", dtype=torch.bfloat16)
    result = _run(mode, q, kv, slots, lengths, out=out)

    assert result.data_ptr() == out.data_ptr()
    assert result.dtype == torch.bfloat16
    torch.testing.assert_close(
        result.float(), _reference(q, kv, slots, lengths).float(), rtol=RTOL, atol=ATOL
    )


def test_flashinfer_nope_dsa_cuda_graph_replay_tracks_inputs() -> None:
    q, kv, slots, lengths = _inputs([TOPK_WIDTH, TOPK_WIDTH - 1], torch.bfloat16, 41)
    out = torch.empty_like(q)

    _run("decode", q, kv, slots, lengths, out=out)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = _run("decode", q, kv, slots, lengths, out=out)

    next_inputs = _inputs([TOPK_WIDTH - 2, TOPK_WIDTH], torch.bfloat16, 51)
    for target, source in zip((q, kv, slots, lengths), next_inputs, strict=True):
        target.copy_(source)
    graph.replay()
    torch.cuda.synchronize()

    assert result.data_ptr() == out.data_ptr()
    torch.testing.assert_close(
        out.float(),
        _reference(q, kv, slots, lengths).float(),
        rtol=RTOL,
        atol=ATOL,
    )
