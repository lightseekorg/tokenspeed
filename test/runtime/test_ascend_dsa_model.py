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


@pytest.mark.parametrize("rows", [1, 1025, 2049])
def test_model_router_leaves_cache_readiness_to_the_index_writer(monkeypatch, rows):
    """The prologue's KV write alone cannot release a DSA producer step."""
    from tokenspeed_kernel.ops.attention.prologue.types import (
        LatentKVCache,
        MLAPrologueOutput,
    )

    from tokenspeed.runtime.layers import paged_attention
    from tokenspeed.runtime.layers.attention.backends.paged.ascend_dsa import (
        AscendDSABackend,
    )
    from tokenspeed.runtime.layers.attention.backends.paged.cache_group_geometry import (
        CacheGroupGeometry,
    )
    from tokenspeed.runtime.layers.attention.backends.paged.router import (
        CacheGroupRouter,
    )
    from tokenspeed.runtime.layers.dense import unquant
    from tokenspeed.runtime.layers.layernorm import RMSNorm
    from tokenspeed.runtime.pd.utils import StepCounter

    class CpuStepCounter(StepCounter):
        # Substitute only device/pinned allocation, retaining record_cache's
        # arithmetic and host publication.
        def __init__(self):
            self.d_ready_cache_step = torch.tensor(0, dtype=torch.int64)
            self.h_ready_cache_step = torch.tensor(0, dtype=torch.int64)

    class CpuIndexWriter(AscendDSABackend):
        # The device SFA/index kernels are unavailable. This leaf writes a real
        # index-cache tensor and publishes when its device work would finish.
        def __init__(self):
            self._prefill_options = None
            self.step_counter = None
            self.index_cache = torch.zeros(rows, 1, 4, dtype=torch.bfloat16)
            self.counter_before_index = None

        def cache_placement(self, layer):
            return None

        def forward_extend(self, q, k, v, layer, slots, pool, bs, **kwargs):
            self.counter_before_index = int(self.step_counter.h_ready_cache_step)
            self.index_cache.index_copy_(0, slots.long(), kwargs["index_key"])
            self.step_counter.record_cache()
            return q[..., :4].transpose(0, 1).contiguous()

    @dataclass
    class Pool:
        latent: torch.Tensor

        def kv_write_target(self, layer_id, slots, owned):
            return LatentKVCache(self.latent, False, slots, owned)

    @dataclass
    class Context:
        attn_backend: CacheGroupRouter
        token_to_kv_pool: Pool
        forward_mode: ForwardMode = ForwardMode.EXTEND
        bs: int = 1

    def cpu_prologue(query, q_pe, latent, *, cache, **kwargs):
        cache.kv_cache.index_copy_(0, cache.slots.long(), latent.unsqueeze(1))
        return MLAPrologueOutput(query=query, key=None, value=None, latent=latent)

    def cpu_rmsnorm(x, weight, eps, **kwargs):
        return torch.nn.functional.rms_norm(
            x.float(), (x.shape[-1],), weight.float(), eps
        ).to(x.dtype)

    monkeypatch.setattr(paged_attention, "mla_prologue", cpu_prologue)
    monkeypatch.setattr(
        unquant,
        "kernel_mm",
        lambda x, weight, *, bias=None, **kwargs: torch.nn.functional.linear(
            x, weight, bias
        ),
    )
    attention = make_attention(nope=True, separate=True, independent=True)
    attention.to(dtype=torch.bfloat16)
    attention.indexer.weights_proj.to(dtype=torch.float32)
    with torch.no_grad():
        for index, parameter in enumerate(attention.parameters()):
            parameter.copy_(
                torch.sin(torch.arange(parameter.numel()).float() + index + 1).reshape(
                    parameter.shape
                )
            )
    for module in attention.modules():
        if isinstance(module, RMSNorm):
            monkeypatch.setattr(module, "kernel", cpu_rmsnorm)
    attention.process_weights_after_loading()
    group = "full_history"
    attention.attn_mqa.bind_cache_group(group)
    leaf = CpuIndexWriter()
    router = CacheGroupRouter(
        lambda *args: leaf,
        is_draft=False,
        spec_num_tokens=1,
        device="cpu",
        consumed_group_ids=None,
    )
    router.bind(
        CacheGroupGeometry(
            granularities={group: 128},
            attn_kinds={group: "full_history"},
            full_history_group_id=group,
            row_geometry={group: (None, None)},
            windows={group: None},
        ),
        {group: leaf},
    )
    router._extend_write_locations = {group: torch.arange(rows)}
    counter = CpuStepCounter()
    router.register_step_counter(counter)
    pool = Pool(torch.zeros(rows, 1, 8, dtype=torch.bfloat16))
    hidden = torch.sin(torch.arange(rows * 8).float()).reshape(rows, 8).bfloat16()
    output = attention(torch.arange(rows), hidden, Context(router, pool), None)

    assert leaf.counter_before_index == 0
    assert int(counter.d_ready_cache_step) == int(counter.h_ready_cache_step) == 1
    assert torch.isfinite(output).all() and output.shape == hidden.shape
    assert torch.count_nonzero(pool.latent) > 0
    assert torch.count_nonzero(leaf.index_cache) > 0
