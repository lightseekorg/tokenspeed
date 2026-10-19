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

"""gfx1250 AttnRes candidate-tile and batch-size coverage."""

import pytest
import torch
from tokenspeed_kernel.ops.residual.torch import torch_attn_res_fwd
from utils import assert_no_triton_compile, is_cdna5

if not is_cdna5():
    pytest.skip("AMD CDNA5 is required", allow_module_level=True)

from tokenspeed_kernel_amd.ops.gfx1250.attention.kda.attn_res import (
    attn_res_rmsnorm_gfx1250,
    gluon_attn_res_fwd_gfx1250,
)

HIDDEN = 7168


def _inputs(tokens, valid_blocks, *, delta, write_block, seed):
    generator = torch.Generator(device="cuda").manual_seed(seed)

    def randn(*shape, scale=1.0):
        return (torch.randn(*shape, device="cuda", generator=generator) * scale).to(
            torch.bfloat16
        )

    stored_blocks = max(valid_blocks + 1, 8) if write_block else max(valid_blocks, 1)
    return {
        "layer": randn(tokens, HIDDEN),
        # Block-major storage, as the model keeps it.
        "blocks": randn(stored_blocks, tokens, HIDDEN),
        "delta": randn(tokens, HIDDEN) if delta else None,
        "res_weight": randn(HIDDEN, scale=0.05),
        "score_weight": 1 + randn(HIDDEN, scale=0.1),
        "output_weight": 1 + randn(HIDDEN, scale=0.1),
    }


def _reference(inputs, valid_blocks, write_block):
    layer = inputs["layer"].clone()
    blocks = inputs["blocks"].clone()
    output = torch_attn_res_fwd(
        layer_residual=layer,
        block_residual=blocks,
        res_weight=inputs["res_weight"],
        rms_weight=inputs["score_weight"],
        eps=1e-5,
        out_norm_weight=inputs["output_weight"],
        out_norm_eps=1e-5,
        delta=inputs["delta"],
        num_valid_blocks=valid_blocks,
        block_write_idx=valid_blocks if write_block else -1,
    )
    return output, layer, blocks


def _run(inputs, valid_blocks, write_block):
    blocks = inputs["blocks"] if write_block else inputs["blocks"][:valid_blocks]
    return attn_res_rmsnorm_gfx1250(
        layer_residual=inputs["layer"],
        block_residual=blocks.transpose(0, 1),
        res_weight=inputs["res_weight"],
        score_rms_weight=inputs["score_weight"],
        score_eps=1e-5,
        output_rms_weight=inputs["output_weight"],
        output_eps=1e-5,
        delta=inputs["delta"],
        num_valid_blocks=valid_blocks,
        block_write_idx=valid_blocks if write_block else -1,
    )


# Snapshot counts cover a power-of-two tile (prefix scored separately), tiles
# with a spare row for the prefix, and the 16-row tile.
@pytest.mark.parametrize("valid_blocks", [0, 1, 3, 4, 6, 8, 11])
@pytest.mark.parametrize("tokens", [1, 4, 300])
@pytest.mark.parametrize(
    "delta,write_block", [(False, False), (True, False), (False, True)]
)
def test_attn_res_candidate_tiles(valid_blocks, tokens, delta, write_block):
    if valid_blocks == 11 and write_block:
        pytest.skip("at most 11 snapshots exist")
    inputs = _inputs(
        tokens, valid_blocks, delta=delta, write_block=write_block, seed=valid_blocks
    )
    expected, expected_layer, expected_blocks = _reference(
        inputs, valid_blocks, write_block
    )
    actual = _run(inputs, valid_blocks, write_block)
    torch.testing.assert_close(actual, expected, rtol=1.6e-2, atol=1.6e-2)
    torch.testing.assert_close(inputs["layer"], expected_layer, rtol=0, atol=0)
    torch.testing.assert_close(inputs["blocks"], expected_blocks, rtol=0, atol=0)


def test_attn_res_token_counts_reuse_compilation():
    # Token counts and the snapshot-row stride (T * H) vary per batch; neither
    # may specialize the kernel. Warm the two decode warp counts first.
    valid_blocks = 5
    for tokens in (4, 64):
        inputs = _inputs(tokens, valid_blocks, delta=False, write_block=False, seed=7)
        _run(inputs, valid_blocks, False)
    with assert_no_triton_compile(gluon_attn_res_fwd_gfx1250):
        for tokens in (1, 3, 4, 16, 17, 33, 64, 255):
            sliced = _inputs(
                tokens, valid_blocks, delta=False, write_block=False, seed=tokens
            )
            expected, _, _ = _reference(sliced, valid_blocks, False)
            actual = _run(sliced, valid_blocks, False)
            torch.testing.assert_close(actual, expected, rtol=1.6e-2, atol=1.6e-2)


@pytest.mark.parametrize("weight_view", ["odd_offset", "strided", "fp32"])
def test_attn_res_rejects_weights_it_cannot_read_as_pairs(weight_view):
    # Weights are read as 4-byte BF16 pairs; views that would misread every
    # pair must be refused rather than silently mixed.
    inputs = _inputs(4, 3, delta=False, write_block=False, seed=0)
    weight = inputs["res_weight"]
    if weight_view == "odd_offset":
        storage = torch.empty(HIDDEN + 1, device="cuda", dtype=torch.bfloat16)
        inputs["res_weight"] = storage[1:].copy_(weight)
    elif weight_view == "strided":
        storage = torch.empty(2 * HIDDEN, device="cuda", dtype=torch.bfloat16)
        inputs["res_weight"] = storage[::2].copy_(weight)
    else:
        inputs["res_weight"] = weight.float()
    with pytest.raises(ValueError, match="AttnRes"):
        _run(inputs, 3, False)
