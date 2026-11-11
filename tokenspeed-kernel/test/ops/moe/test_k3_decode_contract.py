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

"""The Kimi-K3 decode chain's declared envelope."""

import ast
from pathlib import Path

import pytest
import torch

SERVED = dict(
    tp_size=8,
    hidden_size=7168,
    latent_size=3584,
    num_experts=896,
    top_k=16,
    num_groups=1,
    topk_groups=1,
    scoring_func="sigmoid",
    linear_beta=25.0,
    router_dtype=torch.bfloat16,
    expert_dtype="mxfp4",
    pdl=True,
    batch_invariant=False,
)


def _supports(**overrides) -> bool:
    from tokenspeed_kernel.ops.moe._cute_dsl.k3_decode_support import supports

    return supports(**{**SERVED, **overrides})


def test_the_kimi_k3_tp8_mxfp4_layer_is_served():
    assert _supports()


@pytest.mark.parametrize(
    "overrides",
    [
        {"tp_size": 16},
        {"hidden_size": 6144},
        {"latent_size": 3072},
        {"num_experts": 384},
        {"top_k": 8},
        {"num_groups": 8},
        {"topk_groups": 4},
        {"scoring_func": "softmax"},
        {"linear_beta": None},
        {"router_dtype": torch.float32},
        {"expert_dtype": "nvfp4"},
        {"pdl": False},
        {"batch_invariant": True},
    ],
    ids=lambda overrides: "-".join(f"{k}={v}" for k, v in overrides.items()),
)
def test_anything_else_is_declined(overrides):
    assert not _supports(**overrides)


def test_the_envelope_imports_no_cutlass():
    """Platforms without nvidia-cutlass-dsl import it to decline the chain."""
    source = (
        Path(__file__).parents[3]
        / "python/tokenspeed_kernel/ops/moe/_cute_dsl/k3_decode_support.py"
    ).read_text()
    imported = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module.split(".")[0])
    assert imported <= {"__future__", "torch"}


ATTENTION_SERVED = dict(
    projection_rows=6288,
    o_proj_width=1536,
    projection_dtype=torch.bfloat16,
    o_proj_dtype=torch.bfloat16,
)


def _attention_supports(**overrides) -> bool:
    from tokenspeed_kernel.ops.moe._cute_dsl.k3_decode_support import (
        attention_supports,
    )

    return attention_supports(**{**ATTENTION_SERVED, **overrides})


@pytest.mark.parametrize("rows", [6288, 3648, 3808], ids=["kda", "mla", "dsa"])
def test_the_kimi_k3_tp8_attention_layers_are_served(rows):
    assert _attention_supports(projection_rows=rows)


@pytest.mark.parametrize(
    "overrides",
    [
        {"projection_rows": 4096},
        {"o_proj_width": 1024},
        {"projection_dtype": torch.float8_e4m3fn},
        {"o_proj_dtype": torch.float32},
    ],
    ids=lambda overrides: "-".join(f"{k}={v}" for k, v in overrides.items()),
)
def test_other_attention_layers_are_declined(overrides):
    assert not _attention_supports(**overrides)
