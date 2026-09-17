# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""GFX950/GFX1250 gluon DeepSeek V4.1 CSA2 indexer checks."""

from __future__ import annotations

import os
import sys
from datetime import timedelta
from pathlib import Path

import pytest
import torch
from tokenspeed_kernel.platform import current_platform
from tokenspeed_kernel.selection import select_kernel
from tokenspeed_kernel.signature import dense_tensor_format, format_signature

pytest.importorskip("tokenspeed_triton")
pytest.importorskip("tokenspeed_kernel_amd", reason="AMD kernel package is optional")

_OPS = Path(__file__).resolve().parents[3] / "ops"
if str(_OPS) not in sys.path:
    sys.path.insert(0, str(_OPS))

from test_attention_dsv41_index_scan import (  # noqa: E402
    run_index_scan_graph_oracle,
    run_index_topk_full_and_reindex,
)


def _index_name() -> str:
    kernel = select_kernel(
        "attention",
        "dsv41_index_topk",
        format_signature(x=dense_tensor_format(torch.bfloat16)),
        traits={"native_indexer": False},
        solution="gluon",
    )
    return kernel.name


def test_gluon_index_topk_is_selected_on_supported_amd():
    platform = current_platform()
    if platform.is_cdna4:
        assert _index_name() == "gluon_dsv41_index_topk_gfx950"
    elif platform.is_cdna5:
        assert _index_name() == "gluon_dsv41_index_topk_gfx1250"
    else:
        pytest.skip("AMD gluon DSV4.1 indexer")


@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA/ROCm")
    return torch.device("cuda:0")


@pytest.fixture(scope="module")
def tp_group():
    if (
        os.environ.get("TOKENSPEED_TEST_TP4") != "1"
        or os.environ.get("WORLD_SIZE") != "4"
    ):
        yield None
        return
    device = torch.device(f"cuda:{os.environ['LOCAL_RANK']}")
    torch.cuda.set_device(device)
    torch.distributed.init_process_group(
        "nccl", timeout=timedelta(seconds=120), device_id=device
    )
    try:
        yield torch.distributed.group.WORLD
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.parametrize("shards", [1, 4], ids=["local", "tp4"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_index_scan_graph_oracle_gluon(device, shards, dtype, tp_group, require):
    run_index_scan_graph_oracle(
        device, shards, dtype, tp_group, require, "gluon", rtol=0.1, atol=0.1
    )


@pytest.mark.parametrize("heads", [1, 8, 32])
@pytest.mark.parametrize("candidate_topk", [0, 17])
@pytest.mark.parametrize("topk", [65, 512])
def test_index_topk_full_and_reindex_gluon(
    device, heads, candidate_topk, topk, require
):
    run_index_topk_full_and_reindex(
        device, "gluon", heads, candidate_topk, topk, require, match_triton=False
    )
