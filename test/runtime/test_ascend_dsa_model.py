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

from contextlib import nullcontext
from dataclasses import dataclass

import pytest
import torch
from transformers import PretrainedConfig

from tokenspeed.runtime.distributed import Mapping
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.attention.dsa.ascend import AscendDSAAttention


def make_attention(
    *, nope, separate, independent, prefill_enabled=False, scale_q=False
):
    config = PretrainedConfig(
        rms_norm_eps=1e-6,
        mla_use_nope=nope,
        index_use_rope=False,
        mla_use_output_gate=False,
        mla_scale_q_lora=scale_q,
        cli_factor=1,
        index_n_heads=16,
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
        layer_id=0,
        lora_norm_eps=1e-6,
        prefill_enabled=prefill_enabled,
    )


def test_rotary_model_rejected_at_initialization():
    with pytest.raises(ValueError, match="RoPE disabled"):
        make_attention(nope=False, separate=True, independent=True)


def test_unprepared_weights_fail_before_device_execution():
    attention = make_attention(nope=True, separate=True, independent=True)

    @dataclass
    class Context:
        forward_mode: ForwardMode = ForwardMode.DECODE

    with pytest.raises(RuntimeError, match="Prepare DSA weights"):
        attention(None, torch.zeros((64, 4096)), Context(), None)


def test_failed_native_initialization_releases_its_window(monkeypatch):
    from tokenspeed_kernel.ops.attention.dsa import DSAResources

    from tokenspeed.runtime.distributed.process_group_manager import (
        process_group_manager as pg,
    )
    from tokenspeed.runtime.layers.attention.backends.paged import (
        ascend_dsa_tp_kvp_decode as ascend,
    )

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
    resources = DSAResources(window, {}, window.release)
    monkeypatch.setattr(pg, "get_dedicated_device_group", lambda *args: group)
    monkeypatch.setattr(pg, "init_process_group", lambda *args, **kwargs: None)
    monkeypatch.setattr(pg, "get_process_group", lambda *args: group)
    monkeypatch.setattr(pg, "get_device_process_group", lambda *args: group)
    monkeypatch.setattr(ascend.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(ascend.dist, "all_reduce", lambda *args, **kwargs: None)
    monkeypatch.setattr(ascend.dist, "barrier", lambda *args, **kwargs: None)

    class DeviceRuntime:
        @staticmethod
        def synchronize():
            pass

    # CPU CI has no torch.npu module; only the unavailable device fence is mocked.
    monkeypatch.setattr(torch, "npu", DeviceRuntime(), raising=False)
    monkeypatch.setattr(ascend, "prepare_dsa_resources", lambda **kwargs: resources)
    layer = make_attention(nope=True, separate=True, independent=True)
    with pytest.raises(ValueError, match="Unsupported fused DSA projection geometry"):
        ascend.AscendDSATPKVPRuntime(
            [layer], Mapping(rank=0, world_size=8, attn_tp_size=8)
        )
    assert window.storage is None
    assert layer._fused_tp_kvp_decode_runtime is None


@pytest.mark.parametrize("role", [None, "prefill", "decode"])
@pytest.mark.parametrize("prefill_enabled", [False, True])
def test_prefill_weight_preparation_follows_model_opt_in(
    monkeypatch, role, prefill_enabled
):
    from tokenspeed.runtime.utils.env import global_server_args_dict

    monkeypatch.setitem(global_server_args_dict, "disaggregation_mode", role)
    attention = make_attention(
        nope=True, separate=True, independent=True, prefill_enabled=prefill_enabled
    )
    with torch.no_grad():
        for index, parameter in enumerate(attention.parameters()):
            parameter.copy_(
                torch.sin(torch.arange(parameter.numel()).float() + index).reshape(
                    parameter.shape
                )
            )
    attention.process_weights_after_loading()
    assert attention.q_b_proj.weight.shape == (16, 4)
    assert attention.o_proj.weight.shape == (8, 8)
    pointers = (attention.w_kc.data_ptr(), attention.w_vc.data_ptr())
    attention.process_weights_after_loading()
    assert pointers == (attention.w_kc.data_ptr(), attention.w_vc.data_ptr())
    if prefill_enabled:
        _, kc, vc, _ = attention._prefill_weights
        assert kc is attention.w_kc and vc is attention.w_vc


@pytest.mark.parametrize("rows", [1, 1025, 2049])
@pytest.mark.parametrize("scale_q", [False, True])
def test_model_router_leaves_cache_readiness_to_the_index_writer(
    monkeypatch, rows, scale_q
):
    """The prologue's KV write alone cannot release a DSA producer step."""
    from tokenspeed_kernel.ops.attention.prologue.types import (
        LatentKVCache,
    )

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

    @dataclass
    class QuerySpan:
        row_start: int = 0
        row_end: int = rows

    class CpuIndexWriter(AscendDSABackend):
        # The device SFA/index kernels are unavailable. This leaf writes a real
        # index-cache tensor and publishes when its device work would finish.
        def __init__(self):
            self._prefill_layout = None
            self._prefill = QuerySpan()
            self.step_counter = None
            self.index_cache = torch.zeros(rows, 1, 4, dtype=torch.bfloat16)
            self.counter_before_index = None
            self.query = None
            self.index_query = None
            self.index_weights = None

        def cache_placement(self, layer):
            return None

        def forward_extend(self, q, k, v, layer, slots, pool, bs, **kwargs):
            self.counter_before_index = int(self.step_counter.h_ready_cache_step)
            self.query = q.clone()
            self.index_query = kwargs["index_query"].clone()
            self.index_weights = kwargs["index_weights"].clone()
            pool.latent.index_copy_(
                0, slots.long(), kwargs["prefill_latent"].unsqueeze(1)
            )
            self.index_cache.index_copy_(0, slots.long(), kwargs["index_key"])
            self.step_counter.record_cache()
            return q[..., :4].transpose(0, 1).contiguous()

    @dataclass
    class Pool:
        latent: torch.Tensor
        latent_write_sanitizes: bool = False

        def kv_write_target(self, layer_id, slots, owned):
            return LatentKVCache(self.latent, False, slots, owned)

    @dataclass
    class Context:
        attn_backend: CacheGroupRouter
        token_to_kv_pool: Pool
        forward_mode: ForwardMode = ForwardMode.EXTEND
        bs: int = 1

    def cpu_rmsnorm(x, weight, eps, **kwargs):
        return torch.nn.functional.rms_norm(
            x.float(), (x.shape[-1],), weight.float(), eps
        ).to(x.dtype)

    monkeypatch.setattr(
        unquant,
        "kernel_mm",
        lambda x, weight, *, bias=None, **kwargs: torch.nn.functional.linear(
            x, weight, bias
        ),
    )
    attention = make_attention(
        nope=True,
        separate=True,
        independent=True,
        scale_q=scale_q,
        prefill_enabled=True,
    )
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
    q_gamma = attention.q_a_layernorm.weight.detach().clone()
    attention.process_weights_after_loading()
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

    latent = torch.nn.functional.linear(hidden, attention.q_a_proj.weight)
    latent = cpu_rmsnorm(latent, q_gamma, 1e-6)
    main_q = torch.nn.functional.linear(latent, attention.q_b_proj.weight).view(
        rows, 2, 8
    )
    if scale_q:
        main_q = main_q * (8 / 4) ** 0.5
    expected_query = torch.cat(
        (
            torch.bmm(main_q[..., :4].transpose(0, 1), attention.w_kc).transpose(0, 1),
            main_q[..., 4:],
        ),
        -1,
    )
    expected_index_q = torch.nn.functional.linear(
        latent, attention.indexer.wq_b.weight
    ).view(rows, 16, 4)
    expected_weights = torch.nn.functional.linear(
        hidden.float(), attention.indexer.weights_proj.weight
    ) * (4**-0.5 * 16**-0.5)
    torch.testing.assert_close(leaf.query, expected_query, atol=0, rtol=0)
    torch.testing.assert_close(leaf.index_query, expected_index_q, atol=0, rtol=0)
    torch.testing.assert_close(leaf.index_weights, expected_weights, atol=0, rtol=0)
    assert leaf.index_weights.dtype == torch.float32

    assert leaf.counter_before_index == 0
    assert int(counter.d_ready_cache_step) == int(counter.h_ready_cache_step) == 1
    assert torch.isfinite(output).all() and output.shape == hidden.shape
    assert torch.count_nonzero(pool.latent) > 0
    assert torch.count_nonzero(leaf.index_cache) > 0


@pytest.mark.parametrize(
    "hidden,heads,query_rank,index_heads,supported",
    [
        (4096, 64, 1536, 16, False),
        (4096, 64, 1536, 32, True),
        (4096, 64, 1536, 64, False),
        (2048, 32, 768, 16, True),
        (2048, 32, 768, 32, False),
        (2048, 32, 768, 64, False),
    ],
)
def test_fused_support_uses_real_loaded_indexer_geometry(
    hidden, heads, query_rank, index_heads, supported
):
    from tokenspeed.runtime.layers.attention.backends.paged.ascend_dsa_tp_kvp_decode import (
        supports_fused_dsa_tp_kvp_decode,
    )

    config = PretrainedConfig(
        rms_norm_eps=1e-5,
        mla_use_nope=True,
        mla_use_output_gate=False,
        mla_scale_q_lora=True,
        cli_factor=1,
        index_n_heads=index_heads,
        index_head_dim=128,
        index_topk=2048,
        index_init_tokens=16,
        index_local_tokens=1024,
        index_use_rope=False,
    )
    rejection = (
        pytest.raises(NotImplementedError, match="supports only Indexer16/32")
        if index_heads == 64
        else nullcontext()
    )
    with torch.device("meta"), rejection:
        layer = AscendDSAAttention(
            config=config,
            mapping=Mapping(rank=0, world_size=8, attn_tp_size=8, attn_kvp_size=8),
            hidden_size=hidden,
            num_heads=heads,
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            v_head_dim=128,
            q_lora_rank=query_rank,
            kv_lora_rank=512,
            layer_id=0,
            lora_norm_eps=1e-5,
            prefill_enabled=False,
            decode_enabled=True,
        ).bfloat16()
        layer.indexer.weights_proj.float()
    if index_heads == 64:
        return
    assert supports_fused_dsa_tp_kvp_decode(layer) == supported
    layer.indexer.weights_proj.bfloat16()
    assert not supports_fused_dsa_tp_kvp_decode(layer)


@pytest.mark.parametrize("index_heads", [16, 32])
def test_runtime_shares_projection_buffers_and_releases_bindings(
    monkeypatch, index_heads
):
    """Real model resource owner with only native projection/communication replaced."""
    from types import SimpleNamespace

    from tokenspeed_kernel.ops.attention.dsa import DSAResources

    from tokenspeed.runtime.distributed.process_group_manager import (
        process_group_manager as pg,
    )
    from tokenspeed.runtime.layers.attention.backends.paged import (
        ascend_dsa_tp_kvp_decode as ascend,
    )

    class Group:
        def _get_backend(self, device):
            return self

        def get_hccl_comm_name(self, rank):
            return "cpu-window"

    class Layer:
        def __init__(self):
            self.num_heads = 32 if index_heads == 16 else 64
            self.num_local_heads = self.num_heads // 8
            self.indexer = SimpleNamespace(index_n_heads=index_heads)
            self.q_a_proj = SimpleNamespace(weight=torch.zeros(1))
            self.q_a_layernorm = self.kv_a_layernorm = SimpleNamespace(
                variance_epsilon=1e-6
            )
            self.runtime = None

        def configure_fused_tp_kvp_decode(self, runtime):
            self.runtime = runtime

    class CpuProjection:
        # Device weight preparation/native execution are absent; this producer
        # writes the actual buffers supplied by the real model resource owner.
        def __init__(self, layer, *, outputs, **kwargs):
            self.outputs = outputs
            self.window = None

    group = Group()
    releases = []
    window = object()
    heads = 32 if index_heads == 16 else 64
    resources = DSAResources(
        window,
        {
            1: (
                torch.empty(heads, 1, 512),
                torch.empty(1, 1, heads),
                torch.empty(1, 1, heads),
            )
        },
        releases.append,
    )
    monkeypatch.setattr(pg, "get_dedicated_device_group", lambda *a: group)
    monkeypatch.setattr(pg, "init_process_group", lambda *a, **k: None)
    monkeypatch.setattr(pg, "get_process_group", lambda *a: group)
    monkeypatch.setattr(pg, "get_device_process_group", lambda *a: group)
    monkeypatch.setattr(ascend.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(ascend.dist, "all_reduce", lambda *a, **k: None)
    monkeypatch.setattr(ascend.dist, "barrier", lambda *a, **k: None)
    monkeypatch.setattr(
        torch, "npu", SimpleNamespace(synchronize=lambda: None), raising=False
    )
    monkeypatch.setattr(ascend, "prepare_dsa_resources", lambda **kw: resources)
    monkeypatch.setattr(ascend, "_LayerDSATPKVPDecode", CpuProjection)
    layers = [Layer() for _ in range(3)]
    runtime = ascend.AscendDSATPKVPRuntime(
        layers, Mapping(rank=0, world_size=8, attn_tp_size=8)
    )
    addresses = {
        rows: tuple(t.data_ptr() for t in values)
        for rows, values in runtime.projection_outputs.items()
    }
    unique_storage = set()
    for number, layer in enumerate(layers):
        for rows in (1, 8, 64, 512):
            values = layer.runtime.outputs[rows]
            assert tuple(t.data_ptr() for t in values) == addresses[rows]
            for value in values:
                value.fill_(number + rows)
                unique_storage.add(value.untyped_storage().data_ptr())
            torch.testing.assert_close(
                layers[0].runtime.outputs[rows][0],
                torch.full_like(values[0], number + rows),
                rtol=0,
                atol=0,
            )
            assert values[2].dtype == torch.float32
    assert len(unique_storage) == 4 * 3

    # A routing wrapper can disappear while its cache leaf is still referenced.
    # Teardown must clear every successfully bound leaf, not just live roots.
    class Router:
        def __init__(self, children):
            self.children = children

        def child_backends(self):
            return self.children

    from tokenspeed.runtime.layers.attention.backends.paged.ascend_dsa import (
        AscendDSABackend,
    )

    leaf = object.__new__(AscendDSABackend)
    from tokenspeed.runtime.layers.attention.backends.paged import (
        ascend_dsa_tp_kvp_decode as ascend_kvp_decode,
    )
    from tokenspeed.runtime.layers.attention.backends.paged.ascend_dsa_tp_kvp_decode import (
        AscendDSATPKVPDecode,
    )

    monkeypatch.setattr(
        ascend_kvp_decode, "validate_dsa_context_parallel", lambda **kw: None
    )
    leaf._kvp_ranks = tuple(range(8))
    leaf._decode = object.__new__(AscendDSATPKVPDecode)
    leaf._decode._dcp = SimpleNamespace(ranks=tuple(range(8)), degree=8)
    leaf._decode._indexer_spec = SimpleNamespace(
        num_attention_heads=heads, kv_lora_rank=512
    )
    leaf._decode._fused_decode = None
    leaf._dense_backend = Router(())
    router = Router((leaf,))
    runtime.bind_backend(router)
    assert leaf._decode._fused_decode.group == "cpu-window"
    del router
    borrowed = layers[0].runtime
    runtime.close()
    runtime.close()
    assert releases == [window]
    assert all(layer.runtime is None for layer in layers)
    assert leaf._decode._fused_decode is None
    assert borrowed.window is None
    assert not borrowed.outputs


@pytest.mark.parametrize("index_heads", [0, 8, 48, 64])
def test_indexer_head_count_is_rejected_at_initialization(index_heads):
    config = PretrainedConfig(index_n_heads=index_heads)
    with pytest.raises(NotImplementedError, match="supports only Indexer16/32"):
        AscendDSAAttention(
            config=config,
            mapping=Mapping(rank=0),
            hidden_size=8,
            num_heads=2,
            qk_nope_head_dim=4,
            qk_rope_head_dim=4,
            v_head_dim=4,
            q_lora_rank=4,
            kv_lora_rank=4,
            lora_norm_eps=1e-6,
            prefill_enabled=True,
            decode_enabled=False,
        )


@pytest.mark.parametrize(
    "kvp,prefill,decode,local_heads,can_decode",
    [
        (2, True, False, 4, False),
        (2, False, True, 2, True),
        (2, True, True, 2, True),
        (2, True, None, 4, False),
        (2, False, None, 2, True),
        (1, True, None, 4, True),
        (1, True, True, 4, True),
    ],
)
def test_native_dsa_role_projection_layouts(
    kvp, prefill, decode, local_heads, can_decode, monkeypatch
):
    config = make_attention(nope=True, separate=True, independent=True).config
    # Omitting decode_enabled exercises plugins using the original constructor.
    from tokenspeed.runtime.utils.env import global_server_args_dict

    options = {} if decode is None else {"decode_enabled": decode}
    if kvp > 1 and prefill and decode is None:
        monkeypatch.setitem(global_server_args_dict, "disaggregation_mode", "null")
        with pytest.raises(ValueError, match="Unified DSA.*decode_enabled"):
            AscendDSAAttention(
                config=config,
                mapping=Mapping(
                    rank=0, world_size=2, attn_tp_size=2, attn_kvp_size=kvp
                ),
                hidden_size=8,
                num_heads=4,
                qk_nope_head_dim=4,
                qk_rope_head_dim=4,
                v_head_dim=4,
                q_lora_rank=4,
                kv_lora_rank=4,
                lora_norm_eps=1e-6,
                prefill_enabled=True,
            )
        monkeypatch.setitem(global_server_args_dict, "disaggregation_mode", "prefill")

    with torch.device("meta"):
        attention = AscendDSAAttention(
            config=config,
            mapping=Mapping(rank=0, world_size=2, attn_tp_size=2, attn_kvp_size=kvp),
            hidden_size=8,
            num_heads=4,
            qk_nope_head_dim=4,
            qk_rope_head_dim=4,
            v_head_dim=4,
            q_lora_rank=4,
            kv_lora_rank=4,
            lora_norm_eps=1e-6,
            prefill_enabled=prefill,
            **options,
        )
    assert attention.q_b_proj.weight.shape == (local_heads * 8, 4)
    assert attention.kv_b_proj.weight.shape == (local_heads * 8, 4)
    assert attention.o_proj.weight.shape == (8, local_heads * 4)
    assert attention.num_local_heads == local_heads
    assert attention.decode_enabled is can_decode


def test_unified_dsa_prefill_views_preserve_decode_shards(monkeypatch):
    from tokenspeed.runtime.layers.attention.dsa import ascend

    # Only communication is simulated. Rank one's shard has twice the
    # coefficients of rank zero's shard for each projection.
    def gather(value, group, dim):
        assert tuple(group) == (0, 1)
        return torch.cat((value, value * 2), dim=dim)

    monkeypatch.setattr(ascend, "all_gather", gather)
    config = make_attention(nope=True, separate=True, independent=True).config
    attention = AscendDSAAttention(
        config=config,
        mapping=Mapping(rank=0, world_size=2, attn_tp_size=2, attn_kvp_size=2),
        hidden_size=8,
        num_heads=4,
        qk_nope_head_dim=4,
        qk_rope_head_dim=4,
        v_head_dim=4,
        q_lora_rank=4,
        kv_lora_rank=4,
        lora_norm_eps=1e-6,
        prefill_enabled=True,
        decode_enabled=True,
    ).bfloat16()
    addresses = None
    for generation in (1, 3):
        with torch.no_grad():
            for weight in (
                attention.q_b_proj.weight,
                attention.kv_b_proj.weight,
                attention.o_proj.weight,
            ):
                weight.fill_(generation)
        attention.process_weights_after_loading()
        qb, kc, vc, output = attention._prefill_weights
        assert qb.shape == (32, 4)
        assert kc.shape == vc.shape == (4, 4, 4)
        assert output.shape == (8, 16)
        for rank in range(2):
            for actual in (
                qb[rank * 16 : (rank + 1) * 16],
                kc[rank * 2 : (rank + 1) * 2],
                vc[rank * 2 : (rank + 1) * 2],
                output[:, rank * 8 : (rank + 1) * 8],
            ):
                torch.testing.assert_close(
                    actual,
                    torch.full_like(actual, generation * (rank + 1)),
                    atol=0,
                    rtol=0,
                )
        for weight in (
            attention.q_b_proj.weight,
            attention.w_kc,
            attention.w_vc,
            attention.o_proj.weight,
        ):
            torch.testing.assert_close(
                weight, torch.full_like(weight, generation), atol=0, rtol=0
            )
        current = tuple(
            t.data_ptr() for t in (qb, kc, vc, output, attention.w_kc, attention.w_vc)
        )
        if addresses is not None:
            assert current == addresses
        addresses = current


def test_native_projection_initializes_from_indexer_rotary_descriptor(monkeypatch):
    from tokenspeed.runtime.layers.attention.backends.paged import (
        ascend_dsa_tp_kvp_decode as decode,
    )

    # Substitute only absent accelerator transfer, HCCL, and the native kernel
    # provider. Use the real attention, indexer, coefficient packing and owner.
    monkeypatch.setattr(torch.Tensor, "npu", lambda value: value, raising=False)
    monkeypatch.setattr(decode, "validate_dsa_prolog", lambda **kwargs: None)

    class CpuWeightCarrier:
        def prepare_weight(self, value):
            return value

    monkeypatch.setattr(decode, "select_dsa_projection_kernels", CpuWeightCarrier)

    def gather(pieces, value, group):
        for piece in pieces:
            piece.copy_(value)

    monkeypatch.setattr(decode.dist, "all_gather", gather)
    config = PretrainedConfig(
        rms_norm_eps=1e-6,
        mla_use_nope=True,
        index_use_rope=False,
        mla_use_output_gate=False,
        mla_scale_q_lora=False,
        cli_factor=1,
        index_n_heads=16,
        index_head_dim=128,
        index_topk=2048,
        index_init_tokens=0,
        index_local_tokens=0,
    )
    attention = AscendDSAAttention(
        config=config,
        mapping=Mapping(rank=0, world_size=8, attn_tp_size=8, attn_kvp_size=8),
        hidden_size=2048,
        num_heads=32,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        q_lora_rank=768,
        kv_lora_rank=512,
        layer_id=0,
        lora_norm_eps=1e-6,
        prefill_enabled=False,
        decode_enabled=True,
    ).bfloat16()
    attention.indexer.weights_proj.float()
    with torch.no_grad():
        for parameter in attention.parameters():
            parameter.zero_()
    attention.process_weights_after_loading()
    projection = decode.AscendDSATPKVPProjection(
        attention, rank=0, group=None, eps_q=1e-6, eps_kv=1e-6, outputs={}
    )
    assert projection.rotary_mode == 0
    assert projection.rotary_emb is None and projection.cos_sin is None
    torch.testing.assert_close(projection.w_vc, attention.w_vc, atol=0, rtol=0)
    torch.testing.assert_close(projection.w_o, attention.o_proj.weight, atol=0, rtol=0)
