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

"""Native DSA rejects old projection contracts before any device operation."""

import pytest
import torch
from transformers import PretrainedConfig

from tokenspeed.runtime.distributed import Mapping
from tokenspeed.runtime.layers.attention.dsa.ascend import AscendDSAAttention


def make_attention(*, nope, separate, independent):
    config = PretrainedConfig(
        rms_norm_eps=1e-6,
        mla_use_nope=nope,
        mla_use_output_gate=False,
        cli_factor=1,
        index_n_heads=2,
        index_head_dim=4,
        index_topk=4,
        index_init_tokens=0,
        index_local_tokens=0,
    )
    return AscendDSAAttention(
        config=config,
        mapping=Mapping(rank=0),
        hidden_size=8,
        num_heads=2,
        qk_nope_head_dim=4,
        qk_rope_head_dim=4,
        v_head_dim=4,
        q_lora_rank=4,
        kv_lora_rank=4,
        max_position_embeddings=16,
        layer_id=0,
        lora_norm_eps=1e-6,
        separate_mla_projections=separate,
        independent_selection=independent,
    )


def test_rotary_model_preserves_requested_positions():
    attention = make_attention(nope=False, separate=True, independent=True)
    rotary = attention.rotary_emb
    positions = torch.tensor([0, 7], dtype=torch.int64)
    # Exercise the actual rotary module's table construction with two positions.
    coeff = rotary.cos_sin_cache[positions]
    assert torch.equal(coeff[0, :2], torch.ones(2))
    assert torch.equal(coeff[0, 2:], torch.zeros(2))
    assert not torch.equal(coeff[0], coeff[1])
    inverse = 1 / (rotary.base ** (torch.arange(0, 4, 2).double() / 4))
    angles = positions.double()[:, None] * inverse
    expected = torch.cat((angles.cos(), angles.sin()), -1)
    torch.testing.assert_close(coeff.double(), expected, atol=1e-6, rtol=1e-6)


def test_rejects_old_projection_layout():
    with pytest.raises(ValueError, match="separate checkpoint"):
        make_attention(nope=True, separate=False, independent=True)


def test_uninitialized_native_forward_never_uses_standard_projections():
    attention = make_attention(nope=True, separate=True, independent=True)
    with pytest.raises(RuntimeError, match="native DSA resources"):
        attention(None, torch.zeros((64, 4096)), None, None)


def test_rejects_shared_cross_layer_selection():
    with pytest.raises(ValueError, match="independent per-layer selection"):
        make_attention(nope=True, separate=True, independent=False)
