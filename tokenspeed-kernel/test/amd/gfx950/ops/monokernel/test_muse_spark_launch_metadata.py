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

"""Proton launch metadata of the Muse Spark monokernel entry kernels: each
launch reports its rows and the weight bytes it reads (host-only)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

# The CPU-side helpers come from the AMD kernel package, which other vendors
# do not install.
pytest.importorskip("tokenspeed_kernel_amd")

from tokenspeed_kernel_amd.ops.gfx950.monokernel.muse_spark import (  # noqa: E402
    draft_context,
    draft_head,
    draft_layer,
    layer,
)
from tokenspeed_kernel_amd.ops.gfx950.monokernel.muse_spark.weights import (  # noqa: E402
    LayerWeights,
)

BF16, FP32 = 2, 4
# The rank's MXFP4 expert banks (as the regular path's gluon MoE prepares
# them) and their scales.
W13, W13S, W2, W2S = (
    (256, 2048, 1024),
    (256, 64, 512),
    (256, 256, 4096),
    (256, 16, 1024),
)
EXPERT_BYTES = sum(a * b * c for a, b, c in (W13, W13S, W2, W2S)) // 256
# The dense weights every layer launch reads: o, router, pooler and
# post-expert projection shards.
DENSE_BYTES = (8192 * 1024 + 512 * 8192 + 1024 * 4096) * BF16 + 1024 * 256 * FP32
# The fused projection's q | k | v rows and its gate rows.
QKV_BYTES, GATE_BYTES = 1280 * 8192 * BF16, 1024 * 8192 * BF16
QKVG_BYTES = QKV_BYTES + GATE_BYTES


def _meta(*shape, dtype=torch.bfloat16):
    """A tensor of ``shape`` with no storage: metadata reads only its size."""
    return torch.empty(shape, dtype=dtype, device="meta")


def _kernel(fn):
    """The launch's compiled kernel as launch metadata sees it: its name
    (a real one needs a GPU compile)."""
    return SimpleNamespace(name=fn.__name__)


def _layer_weights() -> LayerWeights:
    f32 = torch.float32
    return LayerWeights(
        rank=0,
        full=False,
        qkvg=_meta(2304, 8192),
        o=_meta(8192, 1024),
        alpha1=_meta(1024, dtype=f32),
        beta1=_meta(1024, dtype=f32),
        pre_gain=_meta(1024, dtype=f32),
        router=_meta(256, 1024, dtype=f32),
        router_bias=_meta(256, dtype=f32),
        pool=_meta(512, 8192),
        w13=_meta(*W13, dtype=torch.uint8),
        w13s=_meta(*W13S, dtype=torch.uint8),
        w2=_meta(*W2, dtype=torch.uint8),
        w2s=_meta(*W2S, dtype=torch.uint8),
        pre_exp_gain=_meta(4096, dtype=f32),
        pen_gain=_meta(512, dtype=f32),
        post=_meta(1024, 4096),
        post_gain=_meta(8192, dtype=f32),
        alpha2=_meta(1024, dtype=f32),
        beta2=_meta(8192, dtype=f32),
        in_gain=_meta(8192, dtype=f32),
    )


@pytest.mark.parametrize(
    "rows,chain,own_qkvg,warm_quads,next_qkvg,expected",
    [
        # One decode row: its own projection, 8 experts.
        (1, 1, True, 0, False, QKVG_BYTES + DENSE_BYTES + 8 * EXPERT_BYTES),
        # Warming the next layer's weights reads them too.
        (8, 8, True, 4096, False, QKVG_BYTES + DENSE_BYTES + 64 * EXPERT_BYTES + 65536),
        # The previous launch projected q | k | v: only the gate rows are
        # left.
        (16, 8, False, 0, False, GATE_BYTES + DENSE_BYTES + 128 * EXPERT_BYTES),
        # ... and this one projects the next layer's q | k | v.
        (
            16,
            8,
            False,
            0,
            True,
            GATE_BYTES + DENSE_BYTES + 128 * EXPERT_BYTES + QKV_BYTES,
        ),
    ],
)
def test_layer_reports_its_rows_and_the_weight_bytes_it_reads(
    rows, chain, own_qkvg, warm_quads, next_qkvg, expected
):
    plan = layer.layer_plan(
        full=False,
        rows=rows,
        chain=chain,
        warm_next=warm_quads > 0,
        own_qkvg=own_qkvg,
        next_qkvg=next_qkvg,
        check_parity=False,
    )
    weights = _layer_weights()
    attn_ops, _, moe_ops = weights.kernel_args(
        h=_meta(rows, 1024, dtype=torch.float32),
        a=_meta(rows, 8192),
        seq_len=_meta(rows, dtype=torch.int32),
        slot=_meta(rows, dtype=torch.int32),
        page_table=_meta(2, 4096, dtype=torch.int32),
        k_cache=_meta(4096, 2, 64),
        v_cache=_meta(4096, 2, 64),
        cos_sin=_meta(4096, 64, dtype=torch.float32),
        h_out=_meta(rows, 1024, dtype=torch.float32),
        a_out=_meta(rows, 8192),
    )
    fn = layer.gluon_muse_spark_layer_gfx950
    args = {
        "PLAN": plan.key,
        "attn_ops": attn_ops,
        "moe_ops": moe_ops,
        "warm_quads": warm_quads,
        # The next layer's fused weight (qnext), the same shape as qkvg.
        "w_next": _meta(2304, 8192),
    }
    metadata = fn.launch_metadata((plan.grid,), _kernel(fn), args)
    assert metadata == {
        "name": f"gluon_muse_spark_layer_gfx950 (rows={rows})",
        "bytes": expected,
    }


@pytest.mark.parametrize("rows", [7, 14])
def test_draft_head_reports_its_rows_and_the_weight_bytes_it_reads(rows):
    plan = draft_head.head_plan(rows)
    fn = draft_head.gluon_muse_spark_draft_head_gfx950
    args = {
        "PLAN": plan.key,
        "x": _meta(rows, 8192),
        "lmh_w": _meta(1579 * 16, 8192),
        "hproj_w": _meta(256, 8192),
        "pred_cb": _meta(202048, 256),
        "succ_cb": _meta(202048, 256),
    }
    metadata = fn.launch_metadata((plan.grid,), _kernel(fn), args)
    assert metadata == {
        "name": f"gluon_muse_spark_draft_head_gfx950 (rows={rows})",
        "bytes": (1579 * 16 + 256) * 8192 * BF16,
    }


@pytest.mark.parametrize("rows", [8, 16])
def test_draft_context_reports_its_rows_and_the_weight_bytes_it_reads(rows):
    plan = draft_context.context_plan(rows)
    fn = draft_context.gluon_muse_spark_draft_context_gfx950
    args = {
        "PLAN": plan.key,
        "x": _meta(rows, 40960),
        "fc_w": _meta(1024, 40960),
        "gain": _meta(1024, dtype=torch.float32),
        "kv_w": _meta(1280, 8192),
        "k_norm": _meta(5, 128),
    }
    metadata = fn.launch_metadata((plan.grid,), _kernel(fn), args)
    assert metadata == {
        "name": f"gluon_muse_spark_draft_context_gfx950 (rows={rows})",
        "bytes": (1024 * 40960 + 1280 * 8192) * BF16,
    }


@pytest.mark.parametrize("rows,last", [(8, False), (16, True)])
def test_draft_layer_reports_its_rows_and_the_weight_bytes_it_reads(rows, last):
    plan = draft_layer.layer_plan(rows, last)
    fn = draft_layer.gluon_muse_spark_draft_layer_gfx950
    args = {
        "PLAN": plan.key,
        "x": _meta(rows, 8192),
        "in_gain": _meta(1024, dtype=torch.float32),
        "a_coef": _meta(2048, 8192),
        "qkv_w": _meta(1280, 8192),
        "o_w": _meta(8192, 1024),
        "post_gain": _meta(1024, dtype=torch.float32),
        "m_coef": _meta(2048, 8192),
        "gu_w": _meta(7168, 8192),
        "down_w": _meta(8192, 3584),
    }
    metadata = fn.launch_metadata((plan.grid,), _kernel(fn), args)
    assert metadata == {
        "name": f"gluon_muse_spark_draft_layer_gfx950 (rows={rows})",
        "bytes": ((2048 + 1280 + 2048 + 7168) * 8192 + 8192 * (1024 + 3584)) * BF16,
    }
