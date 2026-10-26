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

from dataclasses import dataclass

import pytest
import torch
from transformers import PretrainedConfig

from tokenspeed.runtime.distributed import Mapping
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.attention.dsa.ascend import AscendDSAAttention


def make_attention(*, nope, separate, independent, native_prefill=False):
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
        native_prefill=native_prefill,
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


def test_unprepared_weights_fail_before_device_execution():
    attention = make_attention(nope=True, separate=True, independent=True)

    @dataclass
    class Context:
        forward_mode: ForwardMode = ForwardMode.DECODE

    with pytest.raises(RuntimeError, match="Prepare DSA weights"):
        attention(None, torch.zeros((64, 4096)), Context(), None)


def test_rejects_shared_cross_layer_selection():
    with pytest.raises(ValueError, match="independent per-layer selection"):
        make_attention(nope=True, separate=True, independent=False)


def test_failed_native_initialization_releases_its_window(monkeypatch):
    from tokenspeed_kernel.ops.attention.dsa import DSAResources

    from tokenspeed.runtime.distributed.process_group_manager import (
        process_group_manager as pg,
    )
    from tokenspeed.runtime.layers.attention.dsa import ascend

    # Substitute only communication and its native window. The real fused
    # constructor rejects this small model after the resource has been acquired.
    class DeviceGroup:
        def _get_backend(self, device):
            return self

        def get_hccl_comm_name(self, rank):
            return "test-window"

    @dataclass
    class Window:
        storage: torch.Tensor | None

        def release(self, window):
            assert window is self
            self.storage = None

    group = DeviceGroup()
    window = Window(torch.ones(16))
    resources = DSAResources(window, {}, {}, window.release)
    monkeypatch.setattr(pg, "get_dedicated_device_group", lambda *args: group)
    monkeypatch.setattr(pg, "init_process_group", lambda *args, **kwargs: None)
    monkeypatch.setattr(pg, "get_process_group", lambda *args: group)
    monkeypatch.setattr(pg, "get_device_process_group", lambda *args: group)
    monkeypatch.setattr(ascend.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(ascend.dist, "all_reduce", lambda *args, **kwargs: None)
    monkeypatch.setattr(ascend.dist, "barrier", lambda *args, **kwargs: None)
    monkeypatch.setattr(torch.npu, "synchronize", lambda: None)
    monkeypatch.setattr(ascend, "prepare_dsa_resources", lambda **kwargs: resources)
    layer = make_attention(nope=True, separate=True, independent=True)
    with pytest.raises(ValueError, match="eight local and 64 global heads"):
        ascend.AscendDSARuntime([layer], Mapping(rank=0, world_size=8, attn_tp_size=8))
    assert window.storage is None
    assert layer._fused_decode_runtime is None


@pytest.mark.parametrize("role", [None, "prefill", "decode"])
@pytest.mark.parametrize("native_prefill", [False, True])
def test_prefill_weight_preparation_follows_model_opt_in(
    monkeypatch, role, native_prefill
):
    from tokenspeed.runtime.utils.env import global_server_args_dict

    monkeypatch.setitem(global_server_args_dict, "disaggregation_mode", role)
    attention = make_attention(
        nope=True, separate=True, independent=True, native_prefill=native_prefill
    )
    with torch.no_grad():
        for index, parameter in enumerate(attention.parameters()):
            parameter.copy_(
                torch.sin(torch.arange(parameter.numel()).float() + index).reshape(
                    parameter.shape
                )
            )
    attention.process_weights_after_loading()
    if not native_prefill:
        assert attention._prefill_weights is None
    else:
        weights = attention._prefill_weights
        assert weights is not None
        for actual, expected in zip(
            weights,
            (
                attention.q_b_proj.weight,
                attention.w_kc,
                attention.w_vc,
                attention.o_proj.weight,
            ),
        ):
            torch.testing.assert_close(actual, expected)
        attention.process_weights_after_loading()
        assert attention._prefill_weights is weights
