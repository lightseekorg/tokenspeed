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

"""A GDN verify returns the gated-RMSNormed output, fused where the kernel can,
in the raw output's shape either way."""

from __future__ import annotations

import os
import sys

import pytest
import torch

# CI Registration (parsed via AST, runtime no-op)
_TEST_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _TEST_DIR)
sys.path.insert(0, os.path.dirname(_TEST_DIR))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=20, suite="runtime-1gpu")

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

from test.runtime.test_gdn_state_paging import _ContractPool, _mamba_config_pair

from tokenspeed_kernel.ops.attention.gdn import (
    gdn_replay_commit_supported,
    gdn_verify_fuses_output_norm,
)
from tokenspeed_kernel.ops.attention.kda import KdaOutputNorm, apply_kda_output_norm

from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.attention.backends.state.gdn import GdnAttnBackend

DEVICE = "cuda"
BATCH, TOKENS, HEADS, HEAD_DIM = 2, 4, 2, 128
CONV_DIM = 3 * HEADS * HEAD_DIM
EPS = 1e-6


def _backend():
    if not gdn_replay_commit_supported(torch.bfloat16):
        pytest.skip("GDN ReplaySSM kernel unavailable on this platform")
    gen = torch.Generator(device=DEVICE).manual_seed(5)
    conv = torch.randn(8, CONV_DIM, 3, device=DEVICE, generator=gen) * 0.02
    recurrent = (
        torch.randn(8, HEADS, HEAD_DIM, HEAD_DIM, device=DEVICE, generator=gen) * 0.05
    )
    pool = _ContractPool(
        {0: ("linear_attention", conv.bfloat16(), recurrent.bfloat16())}
    )
    backend = GdnAttnBackend(
        *_mamba_config_pair(
            torch,
            heads=HEADS,
            head_dim=HEAD_DIM,
            spec_tokens=TOKENS,
            device=DEVICE,
            replay_ssm=True,
        ),
        writes_state_checkpoints_in_forward=False,
        defer_recurrent_state=False,
    )
    backend.set_kv_pool(pool)
    backend.init_cuda_graph_state(BATCH)
    seq_len = 4 + TOKENS
    backend.refresh_decode_metadata(
        BATCH,
        BATCH,
        torch.tensor([0, 1], dtype=torch.int32, device=DEVICE),
        torch.tensor([seq_len, seq_len], dtype=torch.int32, device=DEVICE),
        forward_mode=ForwardMode.DECODE,
        block_tables={
            "linear_attention": torch.tensor(
                [[1, 5], [2, 6]], dtype=torch.int32, device=DEVICE
            )
        },
    )
    return backend, pool


def _verify(backend, pool, inputs, output_norm):
    return backend.forward_decode(
        None,
        None,
        None,
        layer=None,
        out_cache_loc=None,
        token_to_kv_pool=pool,
        bs=BATCH,
        mixed_qkv=inputs["mixed_qkv"].clone(),
        conv_weights=inputs["conv_weights"],
        bias=None,
        activation="silu",
        key_dim=HEADS * HEAD_DIM,
        value_dim=HEADS * HEAD_DIM,
        attention_tp_size=1,
        head_k_dim=HEAD_DIM,
        head_v_dim=HEAD_DIM,
        a=inputs["a"],
        b=inputs["b"],
        A_log=inputs["A_log"],
        dt_bias=inputs["dt_bias"],
        layer_id=0,
        seq_len=BATCH * TOKENS,
        output_norm=output_norm,
    )


@pytest.mark.parametrize("fused", [True, False], ids=["fused", "separate"])
def test_verify_applies_the_gated_output_norm(fused):
    backend, pool = _backend()
    assert backend._verify_fuses_output_norm == gdn_verify_fuses_output_norm(
        torch.bfloat16,
        state_dtype=torch.bfloat16,
        traits={
            "head_dim": HEAD_DIM,
            "value_head_dim": HEAD_DIM,
            "strided_state": False,
            "verify_tokens": TOKENS,
            "tree": False,
            "output_only": True,
        },
        solution=None,
    )
    if fused and not backend._verify_fuses_output_norm:
        pytest.skip("the selected GDN verify kernel applies no output norm")
    # A platform without a fusing kernel takes the separate norm.
    backend._verify_fuses_output_norm = fused
    gen = torch.Generator(device=DEVICE).manual_seed(9)
    rows = BATCH * TOKENS
    inputs = {
        "mixed_qkv": torch.randn(rows, CONV_DIM, device=DEVICE, generator=gen),
        "a": torch.randn(rows, HEADS, device=DEVICE, generator=gen),
        "b": torch.randn(rows, HEADS, device=DEVICE, generator=gen),
        "conv_weights": torch.randn(CONV_DIM, 4, device=DEVICE, generator=gen) * 0.1,
    }
    inputs = {name: value.bfloat16() for name, value in inputs.items()}
    inputs["A_log"] = torch.randn(HEADS, device=DEVICE, generator=gen) * 0.1
    inputs["dt_bias"] = torch.randn(HEADS, device=DEVICE, generator=gen) * 0.1
    # z is a column view of a wider projection row, as in the model.
    projection = torch.randn(
        rows, 2 * HEADS * HEAD_DIM, device=DEVICE, generator=gen
    ).bfloat16()
    output_norm = KdaOutputNorm(
        projection[:, HEADS * HEAD_DIM :],
        (1 + 0.1 * torch.randn(HEAD_DIM, device=DEVICE, generator=gen)).bfloat16(),
        EPS,
    )

    raw = _verify(backend, pool, inputs, None)
    normed = _verify(backend, pool, inputs, output_norm)
    expected = apply_kda_output_norm(raw, output_norm)
    # Both the fused and the separate norm keep the raw output's shape.
    assert raw.shape == (1, rows, HEADS, HEAD_DIM)
    assert normed.shape == raw.shape
    # A fused norm reads the FP32 output, the separate one its BF16 rounding.
    torch.testing.assert_close(
        normed.reshape(rows, -1).float(), expected.float(), rtol=2e-2, atol=2e-2
    )
