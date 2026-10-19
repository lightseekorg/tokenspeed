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
import tokenspeed_kernel.ops.attention.qsa as qsa
import tokenspeed_kernel.ops.attention.qsa.cute_dsl as cute_ops
import torch
from tokenspeed_kernel.platform import ArchVersion, current_platform
from tokenspeed_kernel.selection import NoKernelFoundError, kernel_override

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and current_platform().is_blackwell),
    reason="QSA specialized selection requires NVIDIA Blackwell",
)


@pytest.mark.parametrize(
    ("rows", "pages", "width", "path", "budget"),
    [
        (256, 256, 256, "auto", 256 << 20),
        (512, 255, 512, "auto", 256 << 20),
        (512, 256, 512, "auto", 256 << 20),
        (512, 256, 8, "auto", 256 << 20),
        (8184, 128, 8, "auto", 256 << 20),
        (8192, 128, 8192, "auto", 256 << 20),
        (8192, 128, 4, "auto", 256 << 20),
        (8192, 128, None, "auto", 256 << 20),
        (256, 128, 256, "stream", 256 << 20),
        (8192, 128, 8192, "logits", 0),
        (8, 128, 1, "auto", 0),
        (512, 2049, 512, "auto", 1 << 30),
    ],
)
def test_block_topk_policy_matches_fp32_reference(
    device, rows, pages, width, path, budget
) -> None:
    torch.manual_seed(307)
    capacity = pages * 64
    query = torch.randn(rows, 4, 128, device=device, dtype=torch.bfloat16)
    keys = torch.randn((pages + 1) * 64, 1, 128, device=device, dtype=query.dtype)
    requests = torch.arange(rows // (width or 1), device=device, dtype=torch.int32)
    table = torch.arange(1, pages + 1, device=device, dtype=torch.int32).repeat(
        requests.numel(), 1
    )
    requests = requests.repeat_interleave(width or 1)
    frontiers = torch.tensor(
        [0, 1, 511, 512, 513, capacity // 2, capacity - 1, capacity],
        device=device,
        dtype=torch.int32,
    )
    complete = frontiers[torch.arange(rows, device=device) % frontiers.numel()]
    actual = qsa.qwen4_exp_qsa_block_topk(
        query,
        keys,
        table,
        requests,
        complete,
        page_size=64,
        block_topk=512,
        queries_per_request=width,
        max_partial_bytes=32 << 20,
        max_logits_bytes=budget,
        path=path,
        enable_pdl=False,
        solution=None,
        override=None,
    )
    sample = torch.arange(8, device=device) * (rows // 8) + torch.arange(
        8, device=device
    )
    sample.clamp_max_(rows - 1)
    scores = torch.einsum("rhd,nd->rhn", query[sample].float(), keys[64:, 0].float())
    scores = scores.clamp_min_(0).sum(1)
    scores.masked_fill_(
        torch.arange(capacity, device=device)[None] >= complete[sample, None],
        -float("inf"),
    )
    selected = actual[sample]
    selected_scores = scores.gather(1, selected.clamp_min(0).long())
    selected_scores.masked_fill_(selected < 0, -float("inf"))
    torch.testing.assert_close(
        selected_scores.sort(1, descending=True).values,
        scores.topk(512, dim=1).values,
        atol=1e-4,
        rtol=1e-5,
    )
    torch.testing.assert_close((actual >= 0).sum(1), complete.clamp_max(512).long())
    ordered = actual.sort(1).values
    assert ((ordered[:, 1:] != ordered[:, :-1]) | (ordered[:, 1:] == -1)).all()


@pytest.mark.parametrize(
    "control", ["solution", "override", "context", "env", "hopper"]
)
def test_block_topk_respects_backend_selection(monkeypatch, device, control) -> None:
    query = torch.zeros(8, 4, 128, device=device, dtype=torch.bfloat16)
    keys = torch.zeros(64, 1, 128, device=device, dtype=query.dtype)
    table = torch.zeros(1, 1, device=device, dtype=torch.int32)
    requests = torch.zeros(8, device=device, dtype=torch.int32)
    complete = torch.full_like(requests, 64)
    names = []
    select = qsa.select_kernel

    def record_selection(*args, **kwargs):
        if control == "hopper":
            kwargs["platform"] = replace(
                current_platform(), arch_version=ArchVersion(9, 0)
            )
        kernel = select(*args, **kwargs)
        names.append(kernel.name)
        return kernel

    monkeypatch.setattr(qsa, "select_kernel", record_selection)
    if control == "hopper":
        monkeypatch.setattr(cute_ops, "_IS_NVIDIA_BLACKWELL", False)
    if control == "env":
        monkeypatch.setenv(
            "TOKENSPEED_KERNEL_OVERRIDE_ATTENTION_QSA_BLOCK_TOPK_STREAM", "triton"
        )

    def invoke():
        return qsa.qwen4_exp_qsa_block_topk(
            query,
            keys,
            table,
            requests,
            complete,
            page_size=64,
            block_topk=512,
            queries_per_request=8,
            max_partial_bytes=32 << 20,
            max_logits_bytes=256 << 20,
            path="stream",
            enable_pdl=False,
            solution="triton" if control == "solution" else None,
            override="triton" if control == "override" else None,
        )

    if control == "context":
        with kernel_override("attention", "qsa_block_topk_stream", "triton"):
            selected = invoke()
    else:
        selected = invoke()
    assert names == ["triton_qsa_block_topk_stream"]
    torch.testing.assert_close(
        (selected >= 0).sum(1), torch.full_like(requests, 64).long()
    )


@pytest.mark.parametrize(
    ("path", "override"),
    [
        ("logits", "cute_dsl_qwen4_exp_qsa_block_topk"),
        ("stream", "triton_qsa_block_scores"),
    ],
)
def test_block_topk_override_cannot_change_path(device, path, override) -> None:
    query = torch.empty(8, 4, 128, device=device, dtype=torch.bfloat16)
    keys = torch.empty(64, 1, 128, device=device, dtype=query.dtype)
    table = torch.zeros(1, 1, device=device, dtype=torch.int32)
    requests = torch.zeros(8, device=device, dtype=torch.int32)
    with pytest.raises(NoKernelFoundError, match="feature"):
        qsa.qwen4_exp_qsa_block_topk(
            query,
            keys,
            table,
            requests,
            requests,
            page_size=64,
            block_topk=512,
            queries_per_request=8,
            max_partial_bytes=32 << 20,
            max_logits_bytes=256 << 20,
            path=path,
            enable_pdl=False,
            solution=None,
            override=override,
        )


@pytest.mark.parametrize("rows", [256, 512])
def test_auto_block_topk_graph_refreshes_frontiers(device, rows) -> None:
    torch.manual_seed(313)
    pages, page_size, topk = 256, 64, 512
    capacity = pages * page_size
    query = torch.randn(rows, 5, 128, device=device, dtype=torch.bfloat16)[:, :4]
    keys = torch.randn(capacity, 1, 128, device=device, dtype=query.dtype)
    table = torch.arange(pages, device=device, dtype=torch.int32)[None]
    requests = torch.zeros(rows, device=device, dtype=torch.int32)
    complete = torch.full_like(requests, capacity)

    def invoke():
        return qsa.qwen4_exp_qsa_block_topk(
            query,
            keys,
            table,
            requests,
            complete,
            page_size=page_size,
            block_topk=topk,
            queries_per_request=rows,
            max_partial_bytes=32 << 20,
            max_logits_bytes=256 << 20,
            path="auto",
            enable_pdl=False,
            solution=None,
            override=None,
        )

    invoke()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        selected = invoke()
    sample = torch.tensor([0, 1, 2, 3, 4, 5, rows - 2, rows - 1], device=device)
    frontiers = torch.tensor(
        [0, 1, 511, 512, 513, capacity, capacity - 4, 0],
        device=device,
        dtype=torch.int32,
    )
    for step in range(2):
        complete.copy_(frontiers[(torch.arange(rows, device=device) + step) % 8])
        query.normal_()
        keys.normal_()
        graph.replay()
        scores = torch.einsum("rhd,nd->rhn", query[sample].float(), keys[:, 0].float())
        scores = scores.clamp_min_(0).sum(1)
        scores.masked_fill_(
            torch.arange(capacity, device=device)[None] >= complete[sample, None],
            -float("inf"),
        )
        ids = selected[sample]
        actual = scores.gather(1, ids.clamp_min(0).long()).masked_fill(
            ids < 0, -float("inf")
        )
        expected = scores.topk(topk, dim=1).values
        torch.testing.assert_close(
            actual.sort(1, descending=True).values, expected, atol=1e-4, rtol=1e-5
        )
        torch.testing.assert_close(
            (ids >= 0).sum(1), complete[sample].clamp(0, topk).long()
        )
