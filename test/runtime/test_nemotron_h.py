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

"""Nemotron-H: cache-layer view, Mamba2 loaders and backend seams, non-gated experts."""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci
from tokenspeed_kernel.ops.gemm.fp8_utils import static_quant_fp8

from tokenspeed.runtime.configs.nemotron_h_config import NemotronHConfig
from tokenspeed.runtime.layers.attention.configs.linear_attn import Mamba2Config
from tokenspeed.runtime.layers.moe.loader import (
    MoECheckpointLoader,
    _build_default_expert_plan,
    build_moe_checkpoint_loader,
)
from tokenspeed.runtime.layers.moe.schema import ExpertCheckpointSchema
from tokenspeed.runtime.layers.moe.types import MoELayerSpec
from tokenspeed.runtime.layers.moe.weights.nvfp4 import create_nvfp4_weight_pair

register_cuda_ci(
    est_time=60,
    suite="runtime-1gpu",
    disabled_on_runners=["amd-*"],
    disabled_on_runners_reason="the Mamba2 kernels are NVIDIA-only",
)

# M = Mamba2, * = attention, E = MoE.
_PATTERN = "M*EMM*E"
# GDN-only inputs the shared linear-attention seam takes, unused by Mamba2.
_UNUSED = dict(
    g_raw=None, f_a_out=None, f_b_weight=None, beta_raw=None, lower_bound=None
)


def _config(**overrides) -> NemotronHConfig:
    fields = dict(
        hybrid_override_pattern=_PATTERN,
        hidden_size=64,
        mamba_num_heads=8,
        mamba_head_dim=4,
        n_groups=2,
        ssm_state_size=16,
        conv_kernel=4,
        chunk_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
    )
    fields.update(overrides)
    return NemotronHConfig(**fields)


def test_cache_layers_are_the_mamba_and_attention_blocks_in_order():
    config = _config()
    assert config.cache_layer_ids == [0, 1, None, 2, 3, 4, None]
    assert config.cache_layer_types == [
        "linear_attention",
        "full_attention",
        "linear_attention",
        "linear_attention",
        "full_attention",
    ]
    assert config.linear_layer_ids == [0, 2, 3]
    assert config.full_attention_layer_ids == [1, 4]


def test_mamba2_component_maps_ssd_geometry_onto_the_linear_fields():
    config = _config()
    model_config = SimpleNamespace(hf_text_config=config)
    server_args = SimpleNamespace(
        mapping=SimpleNamespace(linear_attn=SimpleNamespace(tp_size=2))
    )
    mamba2 = Mamba2Config.generate(server_args, model_config)
    assert mamba2.layer_ids == (0, 2, 3)
    assert mamba2.chunk_size == 128
    assert mamba2.dt_limit == (0.0, float("inf"))
    # conv over [C | B | x]: 2 * groups * state + heads * head_dim.
    assert mamba2.conv_dim == 2 * 2 * 16 + 8 * 4
    assert mamba2.conv_state_shape == (mamba2.conv_dim // 2, 3)
    assert mamba2.temporal_state_shape == (4, 4, 16)


def test_non_gated_expert_plan_loads_up_proj_as_all_of_w13():
    schema = ExpertCheckpointSchema(gate_proj_name=None)
    plan = _build_default_expert_plan(schema, num_experts=4, ep_rank=1, ep_size=2)
    assert [
        (e.checkpoint_weight_name, e.shard_id, e.local_expert_id) for e in plan
    ] == [
        ("experts.2.up_proj.", "w13", 0),
        ("experts.2.down_proj.", "w2", 0),
        ("experts.3.up_proj.", "w13", 1),
        ("experts.3.down_proj.", "w2", 1),
    ]
    with pytest.raises(KeyError, match="no gate_proj"):
        schema.make_expert_weight_name(0, "gate_proj")

    # An expert-parallel rank recognizes, and so can skip, experts it does not own.
    loader = build_moe_checkpoint_loader(
        params_dict={}, expert_schema=schema, num_experts=4, ep_rank=1, ep_size=2
    )
    assert not loader.matches("model.layers.1.mixer.experts.0.up_proj.weight")
    assert loader.is_expert_checkpoint_weight(
        "model.layers.1.mixer.experts.0.up_proj.weight"
    )
    assert not loader.is_expert_checkpoint_weight(
        "model.layers.1.mixer.shared_experts.up_proj.weight"
    )


def _spec(activation: str) -> MoELayerSpec:
    return MoELayerSpec(
        top_k=2,
        num_experts=2,
        num_local_experts=2,
        hidden_size=64,
        intermediate_size=32,
        activation=activation,
        tp_rank=0,
        tp_size=1,
        ep_rank=0,
        ep_size=1,
    )


def test_nvfp4_weights_hold_one_projection_for_relu2_experts():
    gated, relu2 = torch.nn.Module(), torch.nn.Module()
    create_nvfp4_weight_pair(_spec("swiglu"), gated, group_size=16)
    create_nvfp4_weight_pair(_spec("relu2"), relu2, group_size=16)
    assert gated.w13_weight.shape == (2, 64, 32)
    assert gated.w13_weight_scale_2.shape == (2, 2)
    assert relu2.w13_weight.shape == (2, 32, 32)
    assert relu2.w13_weight_scale.shape == (2, 32, 4)
    assert relu2.w13_weight_scale_2.shape == (2,)

    experts = torch.nn.Module()
    experts.experts = relu2
    params = dict(experts.named_parameters())
    loader = MoECheckpointLoader(
        params_dict=params,
        expert_plan=_build_default_expert_plan(
            ExpertCheckpointSchema(gate_proj_name=None),
            num_experts=2,
            ep_rank=0,
            ep_size=1,
        ),
    )
    up = torch.randint(0, 255, (32, 32), dtype=torch.uint8)
    loader.load("experts.1.up_proj.weight", up)
    loader.load("experts.1.up_proj.weight_scale_2", torch.tensor(0.25))
    assert torch.equal(relu2.w13_weight[1], up)
    assert relu2.w13_weight_scale_2[1].item() == 0.25


def _mixer(tp_rank: int, tp_size: int):
    from tokenspeed.runtime.models.nemotron_h import NemotronHMamba2Mixer

    mapping = SimpleNamespace(
        linear_attn=SimpleNamespace(
            tp_rank=tp_rank, tp_size=tp_size, tp_group=tuple(range(tp_size))
        )
    )
    return NemotronHMamba2Mixer(_config(), mapping, 0, None, "backbone.layers.0.mixer")


def test_mamba2_loaders_reorder_to_cbx_and_shard_each_segment():
    mixer = _mixer(tp_rank=1, tp_size=2)
    # Tag every checkpoint row with its index: [x 32 | B 32 | C 32].
    conv = torch.arange(96, dtype=torch.float32).view(96, 1, 1).expand(96, 1, 4)
    mixer._load_conv(mixer.conv_weight, conv.to(mixer.conv_weight.dtype))
    rows = mixer.conv_weight[:, 0].float().tolist()
    x_r, b_r, c_r = range(16, 32), range(48, 64), range(80, 96)
    assert rows == [*c_r, *b_r, *x_r]

    # In-projection rows: [z 32 | x 32 | B 32 | C 32 | dt 8].
    in_proj = torch.arange(136, dtype=torch.float32).view(136, 1).expand(136, 64)
    mixer.load_in_proj(mixer.in_proj.weight, in_proj.to(mixer.in_proj.weight.dtype))
    rows = mixer.in_proj.weight[:, 0].float().tolist()
    z_r, dt_r = range(16, 32), range(132, 136)
    x_r, b_r, c_r = range(48, 64), range(80, 96), range(112, 128)
    assert rows == [*z_r, *c_r, *b_r, *x_r, *dt_r]


def _ssd_oracle(x, dt, A, B, C, D, dt_bias, state):
    """Token-by-token fp32 SSD recurrence for one sequence."""
    heads_per_group = x.shape[1] // B.shape[1]
    s = state.float().clone()
    ys = []
    for t in range(x.shape[0]):
        step = F.softplus(dt[t].float() + dt_bias)
        b = B[t].float().repeat_interleave(heads_per_group, dim=0)
        c = C[t].float().repeat_interleave(heads_per_group, dim=0)
        xt = x[t].float()
        s = (
            s * torch.exp(step * A)[:, None, None]
            + (step[:, None] * xt)[:, :, None] * b[:, None, :]
        )
        ys.append(torch.einsum("hpn,hn->hp", s, c) + D[:, None] * xt)
    return torch.stack(ys), s


def _relative(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def _backend(
    heads: int,
    head_dim: int,
    groups: int,
    d_state: int,
    *,
    verify_width: int = 1,
    replay_ssm: bool = False,
):
    from tokenspeed.runtime.layers.attention.backends.state.mamba2 import (
        Mamba2AttnBackend,
    )
    from tokenspeed.runtime.layers.attention.configs.base import AttnConfig
    from tokenspeed.runtime.layers.attention.configs.mha import MHAConfig

    spec = MHAConfig(
        num_attention_heads=4, num_kv_heads=2, head_dim=128, attn_tp_size=1
    )
    mamba2 = Mamba2Config(
        num_k_heads=groups,
        num_v_heads=heads,
        head_k_dim=d_state,
        head_v_dim=head_dim,
        conv_kernel_size=4,
        layer_ids=(0,),
        tp_size=1,
        chunk_size=128,
        dt_limit=(0.0, float("inf")),
        replay_ssm=replay_ssm,
    )
    config = AttnConfig(
        device="cuda",
        dtype=torch.bfloat16,
        kv_cache_dtype=torch.bfloat16,
        kv_cache_quant_method="none",
        prefix_granularity=128,
        context_len=4096,
        max_bs=8,
        is_draft=False,
        speculative_num_draft_tokens=verify_width,
        components=(spec, mamba2),
    )
    return Mamba2AttnBackend(config, spec)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_mamba2_seams_match_the_sequential_recurrence():
    heads, head_dim, groups, d_state = 8, 64, 2, 128
    backend = _backend(heads, head_dim, groups, d_state)
    g = torch.Generator(device="cuda").manual_seed(0)
    lengths = [130, 7]
    total = sum(lengths)
    x = torch.randn(total, heads, head_dim, generator=g, device="cuda").bfloat16()
    B = torch.randn(total, groups, d_state, generator=g, device="cuda").bfloat16()
    C = torch.randn(total, groups, d_state, generator=g, device="cuda").bfloat16()
    dt = torch.randn(total, heads, generator=g, device="cuda").bfloat16()
    A_log = torch.log(torch.arange(1, heads + 1, device="cuda", dtype=torch.float32))
    D = torch.rand(heads, generator=g, device="cuda")
    dt_bias = torch.rand(heads, generator=g, device="cuda") - 4
    initial = torch.randn(2, heads, head_dim, d_state, generator=g, device="cuda") * 0.1
    bounds = torch.tensor([0, 130, 137])

    out, final = backend._prefill_scan(
        C.unsqueeze(0),
        B.unsqueeze(0),
        x.unsqueeze(0),
        initial,
        bounds.to("cuda", torch.int32),
        A_log=A_log,
        dt_bias=dt_bias,
        D=D,
        a=dt,
        b=None,
        seq_len=total,
        num_real_tokens=total,
        inputs_packed=False,
        cu_seqlens_cpu=bounds,
        **_UNUSED,
    )
    A = -torch.exp(A_log)
    for i, (start, end) in enumerate([(0, 130), (130, 137)]):
        ref_y, ref_s = _ssd_oracle(
            x[start:end],
            dt[start:end],
            A,
            B[start:end],
            C[start:end],
            D,
            dt_bias,
            initial[i],
        )
        assert _relative(out[0, start:end], ref_y) < 1e-2
        assert _relative(final[i], ref_s) < 2e-3

    # One decode step for each request continues from its final state.
    pool = torch.zeros(4, heads, head_dim, d_state, device="cuda")
    pool[1], pool[2] = final[0], final[1]
    step_x, step_B, step_C, step_dt = (
        x[:3].clone(),
        B[:3].clone(),
        C[:3].clone(),
        dt[:3].clone(),
    )
    decoded = backend._decode_scan(
        step_C.unsqueeze(1),
        step_B.unsqueeze(1),
        step_x.unsqueeze(1),
        pool,
        torch.tensor([1, 2, -1], device="cuda", dtype=torch.int32),
        torch.tensor([3, 2, -1], device="cuda", dtype=torch.int32),
        A_log=A_log,
        dt_bias=dt_bias,
        D=D,
        a=step_dt,
        b=None,
        output_gate=None,
        norm_weight=None,
        norm_eps=None,
        **_UNUSED,
    )
    assert decoded.shape == (1, 3, heads, head_dim)
    for row, (src, dst) in enumerate([(1, 3), (2, 2)]):
        ref_y, ref_s = _ssd_oracle(
            step_x[row : row + 1],
            step_dt[row : row + 1],
            A,
            step_B[row : row + 1],
            step_C[row : row + 1],
            D,
            dt_bias,
            final[row],
        )
        assert _relative(decoded[0, row], ref_y[0]) < 1e-2
        assert _relative(pool[dst], ref_s) < 1e-4
    torch.testing.assert_close(pool[1], final[0])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_seams_refuse_inputs_of_the_other_recurrence():
    from tokenspeed.runtime.layers.attention.backends.state.mamba import (
        _reject_skip_term,
    )

    with pytest.raises(ValueError, match="no D skip term"):
        _reject_skip_term(torch.ones(1))
    backend = _backend(8, 64, 2, 128)
    empty = torch.empty(1, 1, 1, 1, device="cuda")
    common = dict(
        A_log=torch.zeros(1),
        dt_bias=torch.zeros(1),
        a=torch.zeros(1, 1),
        seq_len=1,
        num_real_tokens=1,
        inputs_packed=False,
        cu_seqlens_cpu=torch.tensor([0, 1]),
        **_UNUSED,
    )
    with pytest.raises(ValueError, match="must pass their D"):
        backend._prefill_scan(
            empty, empty, empty, empty, empty, D=None, b=None, **common
        )
    with pytest.raises(ValueError, match="take no b"):
        backend._prefill_scan(
            empty,
            empty,
            empty,
            empty,
            empty,
            D=torch.ones(1),
            b=torch.ones(1),
            **common,
        )


def _super_cache_layer_types() -> tuple[str, ...]:
    """Nemotron-3 Super's 48 cache layers: Mamba2 runs of 4 or 5 between attention."""
    labels: list[str] = []
    for run in (4, 4, 4, 5, 5, 5, 5, 4):
        labels += ["linear_attention"] * run + ["full_attention"]
    return (*labels, *["linear_attention"] * 4)


def _super_attn_config(tp: int, kv_dtype: torch.dtype, *, is_draft: bool, device: str):
    """Nemotron-3 Super's per-rank attention config; a draft has one MTP attention layer."""
    from tokenspeed.runtime.layers.attention.configs.base import AttnConfig
    from tokenspeed.runtime.layers.attention.configs.mha import MHAConfig

    layer_types = () if is_draft else _super_cache_layer_types()
    state_ids = tuple(i for i, t in enumerate(layer_types) if t == "linear_attention")
    components = [
        MHAConfig(
            backend_name="fa2",
            num_attention_heads=32,
            num_kv_heads=2,
            head_dim=128,
            attn_tp_size=tp,
            cache_layer_types=layer_types,
        )
    ]
    if not is_draft:
        components.append(
            Mamba2Config(
                num_k_heads=8,
                num_v_heads=128,
                head_k_dim=128,
                head_v_dim=64,
                conv_kernel_size=4,
                layer_ids=state_ids,
                tp_size=tp,
                chunk_size=128,
                dt_limit=(0.0, float("inf")),
            )
        )
    return AttnConfig(
        components=tuple(components),
        device=device,
        dtype=torch.bfloat16,
        kv_cache_dtype=kv_dtype,
        kv_cache_quant_method="none",
        kv_cache_mxfp8=False,
        prefix_granularity=128,
        kernel_page_size=64,
        context_len=16384,
        max_bs=2,
        is_draft=is_draft,
    )


def _super_cache_recipe(tp: int, kv_dtype: torch.dtype, *, draft_tokens: int):
    """The Mamba2 recipe, with one MTP draft layer when ``draft_tokens`` is set."""
    from tokenspeed.runtime.layers.attention.kv_cache.recipes.setup import (
        cache_recipe,
    )

    with_draft = draft_tokens > 0
    # ReplaySSM is a CUDA path; the plain plan needs no device.
    device = "cuda" if with_draft else "cpu"
    return cache_recipe(
        "mamba2",
        server_args=SimpleNamespace(
            prefix_granularity=128,
            max_total_tokens=None,
            speculative_num_draft_tokens=draft_tokens,
            enable_replay_ssm=with_draft,
        ),
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(text_config=SimpleNamespace()),
            num_attention_layers=48,
        ),
        attn_config=_super_attn_config(tp, kv_dtype, is_draft=False, device=device),
        draft_model_config=(
            SimpleNamespace(num_attention_layers=1) if with_draft else None
        ),
        draft_attn_config=(
            _super_attn_config(tp, kv_dtype, is_draft=True, device=device)
            if with_draft
            else None
        ),
        cache_budget_bytes=1 << 31,
        probe_batch_rows=None,
        decode_input_tokens=draft_tokens or 1,
        overlap_schedule_depth=0,
    )


@pytest.mark.parametrize(
    ("tp", "kv_dtype"),
    [(1, torch.float8_e4m3fn), (2, torch.float8_e4m3fn), (1, torch.bfloat16)],
)
def test_mamba2_cache_plan_packs_state_and_kv_without_state_padding(tp, kv_dtype):
    """Nemotron-3 Super's per-rank geometry: every state page is fully used."""
    layer_types = _super_cache_layer_types()
    state_ids = tuple(i for i, t in enumerate(layer_types) if t == "linear_attention")
    full_ids = tuple(i for i, t in enumerate(layer_types) if t == "full_attention")
    assert len(state_ids) == 40 and len(full_ids) == 8
    recipe = _super_cache_recipe(tp, kv_dtype, draft_tokens=0)
    plan = recipe.setup().spec.memory_plan
    members: dict[str, list[int]] = {}
    for field in plan.fields:
        if field.field_id.endswith((".ssm", ".k")):
            members.setdefault(field.group_id, []).append(
                int(field.field_id.split(".")[1])
            )
    for k in range(5):
        assert members[f"linear_attention_{k}"] == list(state_ids[k::5])
    for k in range(2):
        assert members[f"full_attention_{k}"] == list(full_ids[k::2])

    ssm_bytes = (128 // tp) * 64 * 128 * 4
    conv_bytes = (10240 // tp) * 3 * 2
    k_bytes = 128 * max(2 // tp, 1) * 128 * kv_dtype.itemsize
    # A state page is exactly eight SSM states and eight conv windows.
    assert plan.lcm_block_bytes == 8 * (ssm_bytes + conv_bytes)
    packing = {g.group_id: g.cache_blocks_per_lcm_block for g in plan.groups}
    kv_packing = ssm_bytes // k_bytes
    assert packing == {
        **{f"linear_attention_{k}": 1 for k in range(5)},
        **{f"full_attention_{k}": kv_packing for k in range(2)},
    }
    # A KV page pads only by its share of the conv segments.
    kv_padding = (plan.lcm_block_bytes / kv_packing - 8 * k_bytes) / (8 * k_bytes)
    assert kv_padding < 0.02


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_mamba2_cache_plan_places_the_mtp_layer_and_replays_ssm():
    """The MTP attention layer joins the first KV group; verify stages only conv windows."""
    recipe = _super_cache_recipe(1, torch.float8_e4m3fn, draft_tokens=4)
    setup = recipe.setup()
    plan = setup.spec.memory_plan
    draft = {f.field_id: f for f in plan.fields if f.field_id.startswith("layer.48.")}
    assert sorted(draft) == ["layer.48.k", "layer.48.v"]
    assert {f.group_id for f in draft.values()} == {"full_attention_0"}
    assert recipe.replay_ssm and setup.num_draft_layers == 1
    # Replay keeps the five staged SSM states per layer and request out of the workspace.
    ssm_bytes = 128 * 64 * 128 * 4
    assert setup.fixed_workspace_bytes < 2 * 5 * 40 * ssm_bytes


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("tokens", [1, 7, 33, 200])
def test_router_matches_the_fp32_reference(tokens: int):
    from tokenspeed.runtime.models.nemotron_h import NemotronHRouter

    config = _config(hidden_size=4096, n_routed_experts=512)
    router = NemotronHRouter(config).cuda()
    g = torch.Generator(device="cuda").manual_seed(tokens)
    router.weight.data.copy_(
        torch.randn(512, 4096, generator=g, device="cuda").bfloat16() * 0.02
    )
    hidden = torch.randn(tokens, 4096, generator=g, device="cuda").bfloat16()
    logits = router(hidden)
    reference = F.linear(hidden.float(), router.weight.float())
    assert logits.dtype == torch.float32
    torch.testing.assert_close(logits, reference, atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_gated_norm_quantizes_for_a_static_fp8_out_proj():
    from tokenspeed.runtime.layers.attention.linear.layernorm_gated import rmsnorm_fn

    eps, group = 1e-5, 64
    g = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(9, 4 * group, generator=g, device="cuda").bfloat16()
    z = torch.randn(9, 4 * group, generator=g, device="cuda").bfloat16()
    weight = torch.rand(4 * group, generator=g, device="cuda").bfloat16() + 0.5
    scale = torch.tensor([4.0 / 448.0], device="cuda")
    gated = (x.float() * F.silu(z.float())).reshape(9, 4, group)
    ref = gated * torch.rsqrt(gated.pow(2).mean(-1, keepdim=True) + eps)
    ref = ref.reshape(9, -1) * weight.float()

    kwargs = dict(eps=eps, group_size=group, norm_before_gate=False)
    out = rmsnorm_fn(x, weight, z, **kwargs, weights_independent=True)
    out_fp8 = rmsnorm_fn(
        x, weight, z, **kwargs, weights_independent=True, fp8_scale=scale
    )

    torch.testing.assert_close(out.float(), ref, atol=2e-2, rtol=2e-2)
    expected, _ = static_quant_fp8(out, scale)
    assert torch.equal(out_fp8.view(torch.uint8), expected.view(torch.uint8))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("with_fp8", [False, True])
def test_add_norm_folds_both_moe_halves_into_the_residual(with_fp8: bool):
    from tokenspeed.runtime.distributed.mapping import Mapping
    from tokenspeed.runtime.models.nemotron_h import NemotronHNorm

    mapping = Mapping(rank=0, world_size=1)
    norm = NemotronHNorm(256, 1e-5, mapping, layer_index=1).cuda().to(torch.bfloat16)
    g = torch.Generator(device="cuda").manual_seed(1)
    norm.weight.data.copy_(torch.rand(256, generator=g, device="cuda") + 0.5)
    routed, shared, residual = (
        torch.randn(5, 256, generator=g, device="cuda").bfloat16() for _ in range(3)
    )
    scale = torch.tensor([4.0 / 448.0], device="cuda") if with_fp8 else None
    total = (routed + shared).float() + residual.float()
    ref = total * torch.rsqrt(total.pow(2).mean(-1, keepdim=True) + 1e-5)
    ref = ref * norm.weight.float()

    normed, normed_fp8, residual = norm.add_norm(
        (routed, shared), residual, scale, None
    )

    assert torch.equal(residual, total.bfloat16())
    torch.testing.assert_close(normed.float(), ref, atol=2e-2, rtol=2e-2)
    assert (normed_fp8 is not None) == with_fp8
    if with_fp8:
        expected, _ = static_quant_fp8(normed, scale)
        assert torch.equal(normed_fp8.view(torch.uint8), expected.view(torch.uint8))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    ("enabled", "tokens", "fused"),
    [(False, 5, False), (True, 5, True), (True, 9, False)],
)
def test_add_norm_follows_the_shared_fusion_policy_under_tp(
    monkeypatch: pytest.MonkeyPatch, enabled: bool, tokens: int, fused: bool
):
    """At TP2 the summed MoE halves take the fused all-reduce only when the server allows it."""
    from tokenspeed.runtime.distributed import comm_manager
    from tokenspeed.runtime.distributed.mapping import Mapping
    from tokenspeed.runtime.models.nemotron_h import NemotronHNorm
    from tokenspeed.runtime.utils.env import global_server_args_dict

    monkeypatch.setitem(global_server_args_dict, "enable_allreduce_fusion", enabled)
    monkeypatch.setitem(global_server_args_dict, "comm_fusion_max_num_tokens", 8)
    reduced, fused_inputs = [], []
    monkeypatch.setattr(
        comm_manager, "all_reduce", lambda x, group: reduced.append(x.clone()) or x
    )
    mapping = Mapping(rank=0, world_size=2)
    norm = NemotronHNorm(64, 1e-5, mapping, layer_index=1).cuda().to(torch.bfloat16)

    def fused_reduce_norm(rank, group, x, residual):
        fused_inputs.append(x.clone())
        return x, residual

    monkeypatch.setattr(norm, "forward_with_allreduce_fusion", fused_reduce_norm)
    g = torch.Generator(device="cuda").manual_seed(2)
    routed, shared, residual = (
        torch.randn(tokens, 64, generator=g, device="cuda").bfloat16() for _ in range(3)
    )

    _, normed_fp8, _ = norm.add_norm((routed, shared), residual, None, None)

    assert normed_fp8 is None
    assert len(fused_inputs) == int(fused) and len(reduced) == int(not fused)
    assert torch.equal((fused_inputs or reduced)[0], routed + shared)


@pytest.mark.parametrize("rank", [0, 1])
def test_quantized_fc2_sees_the_reduced_latent_and_counts_once(
    monkeypatch: pytest.MonkeyPatch, rank: int
):
    """At MoE TP2 a static-FP8 fc2 must quantize the summed latent, on one rank."""
    from tokenspeed.runtime.distributed.mapping import Mapping
    from tokenspeed.runtime.layers.dense.fp8 import Fp8LinearMethod
    from tokenspeed.runtime.layers.dense.unquant import UnquantizedLinearMethod
    from tokenspeed.runtime.layers.quantization.fp8 import Fp8Config
    from tokenspeed.runtime.models import nemotron_h

    mapping = Mapping(rank=rank, world_size=2)
    group = mapping.moe.tp_ep_group
    unquant = SimpleNamespace(quant_method=UnquantizedLinearMethod())
    fp8 = SimpleNamespace(
        quant_method=Fp8LinearMethod(
            Fp8Config(
                is_checkpoint_fp8_serialized=True,
                activation_scheme="static",
                weight_block_size=None,
            )
        )
    )
    assert nemotron_h._fc2_reduce_group(unquant, False, mapping) is None
    assert nemotron_h._fc2_reduce_group(unquant, True, mapping) == group
    assert nemotron_h._fc2_reduce_group(fp8, False, mapping) == group
    assert nemotron_h._fc2_reduce_group(fp8, False, Mapping(0, 1)) is None

    reduced, projected = [], []
    monkeypatch.setattr(
        nemotron_h, "all_reduce", lambda x, g: reduced.append(g) or x * 2
    )
    moe = SimpleNamespace(
        fc2_reduce_group=group,
        moe_rank=mapping.moe.tp_ep_rank,
        fc2_latent_proj=lambda x: (projected.append(x.clone()) or x + 1, None),
    )
    out = nemotron_h.NemotronHMoE._latent_to_hidden(moe, torch.ones(2, 4))
    assert reduced == [group]
    # Every rank runs fc2 on the reduced latent so startup tuning stays in lockstep.
    assert len(projected) == 1 and torch.equal(projected[0], torch.full((2, 4), 2.0))
    if rank == 0:
        assert torch.equal(out, torch.full((2, 4), 3.0))
    else:
        assert out is None


@pytest.mark.parametrize("graph_phase,capture", [(False, False), (True, True)])
def test_router_and_shared_expert_run_beside_fc1_only_in_decode_graphs(
    monkeypatch: pytest.MonkeyPatch, graph_phase: bool, capture: bool
):
    """Graph capture runs the router and shared expert on the aux stream beside fc1."""
    from tokenspeed.runtime.models import nemotron_h
    from tokenspeed.runtime.utils.cuda_stream import StreamFork

    monkeypatch.setattr(nemotron_h, "get_is_cuda_graph_phase", lambda: graph_phase)
    monkeypatch.setattr(nemotron_h, "get_is_capture_mode", lambda: capture)
    aux = torch.cuda.Stream()
    streams = {}

    def record(name, value):
        streams[name] = torch.cuda.current_stream()
        return value

    moe = SimpleNamespace(
        stream_fork=StreamFork(aux),
        gate=lambda x: record("gate", x * 3),
        shared_experts=lambda x: record("shared", x * 2),
        fc1_latent_proj=lambda x: (record("fc1", x - 1), None),
        _routed=lambda x, logits, latent, ctx: record("routed", logits + latent),
    )
    hidden = torch.arange(8.0, device="cuda").view(2, 4)
    out_routed, out_shared = nemotron_h.NemotronHMoE.forward(moe, hidden, None, None)
    torch.cuda.synchronize()
    assert torch.equal(out_routed, hidden * 4 - 1)
    assert torch.equal(out_shared, hidden * 2)
    main = torch.cuda.current_stream()
    side = aux if graph_phase else main
    assert streams == {"gate": side, "shared": side, "fc1": main, "routed": main}


@pytest.mark.parametrize(
    "blocks",
    [
        ["moe", "attention"],
        ["attention", "moe", "attention", "moe"],
        ["attention", "mamba"],
    ],
)
def test_mtp_head_refuses_patterns_its_narrowing_cannot_serve(blocks: list[str]):
    from tokenspeed.runtime.distributed.mapping import Mapping
    from tokenspeed.runtime.models.nemotron_h_nextn import NemotronHForCausalLMNextN

    config = _config(num_nextn_predict_layers=1, mtp_layers_block_type=blocks)
    assert list(config.mtp_layers_block_type) == blocks
    with pytest.raises(NotImplementedError, match="opens with its only attention"):
        NemotronHForCausalLMNextN(config, Mapping(rank=0, world_size=1))


def test_static_fp8_scale_names_only_per_tensor_static_linears():
    from tokenspeed.runtime.layers.dense.fp8 import Fp8LinearMethod
    from tokenspeed.runtime.layers.dense.unquant import UnquantizedLinearMethod
    from tokenspeed.runtime.layers.quantization.fp8 import Fp8Config
    from tokenspeed.runtime.models.nemotron_h import _static_fp8_scale

    scale = torch.ones(1)
    static = SimpleNamespace(
        quant_method=Fp8LinearMethod(
            Fp8Config(
                is_checkpoint_fp8_serialized=True,
                activation_scheme="static",
                weight_block_size=None,
            )
        ),
        input_scale=scale,
    )
    blocked = SimpleNamespace(
        quant_method=Fp8LinearMethod(Fp8Config()), input_scale=None
    )
    bf16 = SimpleNamespace(quant_method=UnquantizedLinearMethod(), input_scale=None)
    assert _static_fp8_scale(static) is scale
    assert _static_fp8_scale(blocked) is None
    assert _static_fp8_scale(bf16) is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_mamba2_decode_reads_the_projection_view_in_place():
    """The flag that skips the decode copy: conv and state update read strided rows."""
    from tokenspeed.runtime.layers.attention.linear.causal_conv1d import (
        causal_conv1d_update,
    )

    heads, head_dim, groups, d_state = 8, 64, 2, 128
    backend = _backend(heads, head_dim, groups, d_state)
    assert backend._decode_packed_qkv_views
    g = torch.Generator(device="cuda").manual_seed(2)
    conv_dim = 2 * groups * d_state + heads * head_dim
    # Projection rows are [z | C B x | dt]; the conv channels are a strided view.
    projected = torch.randn(
        3, head_dim * heads + conv_dim + heads, generator=g, device="cuda"
    ).bfloat16()
    view = projected[:, head_dim * heads : head_dim * heads + conv_dim]
    outside = torch.cat(
        [projected[:, : head_dim * heads], projected[:, head_dim * heads + conv_dim :]],
        dim=1,
    ).clone()
    weight = torch.randn(conv_dim, 4, generator=g, device="cuda").bfloat16()
    states = torch.randn(4, 3, conv_dim, generator=g, device="cuda").bfloat16()
    states = states.transpose(1, 2)
    compact_states = states.clone()
    indices = torch.tensor([1, 2, 3], device="cuda", dtype=torch.int32)

    compact = causal_conv1d_update(
        view.clone(), compact_states, weight, None, "silu", conv_state_indices=indices
    )
    strided = causal_conv1d_update(
        view, states, weight, None, "silu", conv_state_indices=indices
    )
    assert strided.data_ptr() == view.data_ptr()
    assert torch.equal(strided, compact)
    assert torch.equal(states, compact_states)
    after = torch.cat(
        [projected[:, : head_dim * heads], projected[:, head_dim * heads + conv_dim :]],
        dim=1,
    )
    assert torch.equal(after, outside)

    C, B, x = strided.split(
        [groups * d_state, groups * d_state, heads * head_dim], dim=-1
    )
    dt = projected[:, -heads:]
    scan = dict(
        A_log=torch.zeros(heads, device="cuda"),
        dt_bias=torch.zeros(heads, device="cuda"),
        D=torch.ones(heads, device="cuda"),
        a=dt,
        b=None,
        output_gate=None,
        norm_weight=None,
        norm_eps=None,
        **_UNUSED,
    )
    pools = [torch.zeros(4, heads, head_dim, d_state, device="cuda") for _ in range(2)]
    outputs = []
    for pool, (q, k, v) in zip(
        pools, [(C, B, x), (C.contiguous(), B.contiguous(), x.contiguous())]
    ):
        outputs.append(
            backend._decode_scan(
                q.view(3, 1, groups, d_state),
                k.view(3, 1, groups, d_state),
                v.view(3, 1, heads, head_dim),
                pool,
                indices,
                indices,
                **scan,
            )
        )
    assert torch.equal(outputs[0], outputs[1])
    assert torch.equal(pools[0], pools[1])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_mamba2_layers_of_one_forward_share_its_chunk_plan():
    backend = _backend(8, 64, 2, 128)
    device = torch.device("cuda")
    bounds = torch.tensor([0, 130, 137])
    plan = backend._chunk_plan(bounds, device)
    assert backend._chunk_plan(bounds, device) is plan
    body, tail = torch.tensor([0, 128]), torch.tensor([0, 2, 9])
    backend._chunk_plan(body, device)
    backend._chunk_plan(tail, device)
    assert backend._chunk_plan(bounds, device) is plan
    # The next forward brings fresh bounds, even when their values repeat.
    fresh = backend._chunk_plan(bounds.clone(), device)
    assert fresh is not plan
    assert torch.equal(fresh.cu_chunk_seqlens, plan.cu_chunk_seqlens)
    assert fresh.cu_chunk_seqlens.tolist() == [0, 128, 130, 137]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("replay", [True, False])
def test_mamba2_verify_scan_continues_each_request_from_its_committed_state(replay):
    heads, head_dim, groups, d_state, steps, batch = 8, 64, 2, 128, 4, 2
    backend = _backend(
        heads, head_dim, groups, d_state, verify_width=steps, replay_ssm=replay
    )
    g = torch.Generator(device="cuda").manual_seed(3)
    rows = batch * steps
    x = torch.randn(rows, heads, head_dim, generator=g, device="cuda").bfloat16()
    B = torch.randn(rows, groups, d_state, generator=g, device="cuda").bfloat16()
    C = torch.randn(rows, groups, d_state, generator=g, device="cuda").bfloat16()
    dt = torch.randn(rows, heads, generator=g, device="cuda").bfloat16()
    A_log = torch.log(torch.arange(1, heads + 1, device="cuda", dtype=torch.float32))
    D = torch.rand(heads, generator=g, device="cuda")
    dt_bias = torch.rand(heads, generator=g, device="cuda") - 4
    committed = 0.1 * torch.randn(
        batch, heads, head_dim, d_state, generator=g, device="cuda"
    )

    slab = torch.zeros(6, heads, head_dim, d_state, device="cuda")
    pages = torch.tensor([4, 1], dtype=torch.int32, device="cuda")
    slab[pages.long()] = committed
    scratch = torch.zeros(batch * (steps + 1), heads, head_dim, d_state, device="cuda")
    base = backend._verify_scratch_base_rows(batch, steps)
    scratch[base.long()] = committed
    grid = base[:, None] + torch.arange(1, steps + 1, device="cuda", dtype=torch.int32)
    before = slab.clone()

    out = backend._verify_scan(
        C.unsqueeze(0),
        B.unsqueeze(0),
        x.unsqueeze(0),
        slab,
        None if replay else scratch,
        pages,
        grid,
        A_log=A_log,
        dt_bias=dt_bias,
        D=D,
        a=dt,
        b=None,
        batch_size=batch,
        draft_token_num=steps,
        seq_len=rows,
        **_UNUSED,
    )
    assert out.shape == (1, rows, heads, head_dim)
    assert torch.equal(slab, before)
    A = -torch.exp(A_log)
    for req in range(batch):
        span = slice(req * steps, (req + 1) * steps)
        inputs = (x[span], dt[span], A, B[span], C[span], D, dt_bias, committed[req])
        ref_y, _ = _ssd_oracle(*inputs)
        assert _relative(out[0, span], ref_y) < 1e-2
        for t in range(steps if not replay else 0):
            prefix = [v[: t + 1] for v in inputs[:2]] + [A]
            prefix += [v[: t + 1] for v in inputs[3:5]] + [D, dt_bias, committed[req]]
            _, ref_s = _ssd_oracle(*prefix)
            assert _relative(scratch[int(grid[req, t])], ref_s) < 1e-4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("state_dtype", [torch.float32, torch.bfloat16])
def test_mamba2_replay_commit_matches_the_staged_verify_states(state_dtype):
    """Verify then commit through the backend: replayed pages equal the staged ones."""
    from test.runtime.test_gdn_state_paging import _ContractPool

    from tokenspeed.runtime.execution.forward_batch_info import ForwardMode

    heads, head_dim, groups, d_state, steps, batch = 8, 64, 2, 128, 3, 2
    key_dim, value_dim = groups * d_state, heads * head_dim
    conv_dim = 2 * key_dim + value_dim
    g = torch.Generator(device="cuda").manual_seed(5)
    conv = 0.02 * torch.randn(8, conv_dim, 3, generator=g, device="cuda")
    ssm = 0.02 * torch.randn(8, heads, head_dim, d_state, generator=g, device="cuda")
    inputs = dict(
        mixed_qkv=torch.randn(batch * steps, conv_dim, generator=g, device="cuda"),
        conv_weights=0.1 * torch.randn(conv_dim, 4, generator=g, device="cuda"),
        a=torch.randn(batch * steps, heads, generator=g, device="cuda"),
        A_log=torch.log(torch.arange(1, heads + 1, device="cuda").float()),
        dt_bias=torch.rand(heads, generator=g, device="cuda") - 4,
        D=torch.rand(heads, generator=g, device="cuda"),
    )
    for name in ("mixed_qkv", "conv_weights", "a"):
        inputs[name] = inputs[name].bfloat16()
    tables = torch.tensor([[1, 5], [2, 6]], dtype=torch.int32, device="cuda")
    outputs, pools = [], []
    for replay in (True, False):
        backend = _backend(
            heads, head_dim, groups, d_state, verify_width=steps, replay_ssm=replay
        )
        pool = _ContractPool(
            4,
            {0: ("linear_attention", conv.bfloat16(), ssm.to(state_dtype))},
        )
        backend.set_kv_pool(pool)
        backend.init_cuda_graph_state(batch)
        backend.refresh_decode_metadata(
            batch,
            batch,
            torch.tensor([0, 1], dtype=torch.int32, device="cuda"),
            torch.tensor([7, 7], dtype=torch.int32, device="cuda"),
            forward_mode=ForwardMode.DECODE,
            block_tables={"linear_attention": tables},
        )
        outputs.append(
            backend.forward_decode(
                None,
                None,
                None,
                layer=None,
                out_cache_loc=None,
                token_to_kv_pool=pool,
                bs=batch,
                mixed_qkv=inputs["mixed_qkv"].clone(),
                conv_weights=inputs["conv_weights"],
                bias=None,
                activation="silu",
                key_dim=key_dim,
                value_dim=value_dim,
                attention_tp_size=1,
                head_k_dim=d_state,
                head_v_dim=head_dim,
                a=inputs["a"],
                b=None,
                A_log=inputs["A_log"],
                dt_bias=inputs["dt_bias"],
                D=inputs["D"],
                layer_id=0,
                seq_len=batch * steps,
            )
        )
        backend.commit_verified_state(
            torch.tensor([1, 3], dtype=torch.int32, device="cuda")
        )
        pools.append(pool)
    torch.cuda.synchronize()

    assert torch.equal(outputs[0], outputs[1])
    committed = torch.tensor([5, 6], device="cuda")
    for component in ("conv_state", "recurrent_state"):
        replayed, staged = (p.get_component(0, component) for p in pools)
        assert torch.equal(replayed[committed], staged[committed]), component


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("shared_scale", [True, False])
def test_static_per_tensor_checkpoint_runs_a_merged_projection(
    shared_scale: bool,
) -> None:
    """ModelOpt FP8 exports: per-tensor weight and static input scales per shard."""
    from tokenspeed.runtime.layers.dense.fp8 import Fp8LinearMethod
    from tokenspeed.runtime.layers.quantization.fp8 import Fp8Config

    method = Fp8LinearMethod(
        Fp8Config(
            is_checkpoint_fp8_serialized=True,
            activation_scheme="static",
            weight_block_size=None,
        )
    )
    layer = torch.nn.Module()
    sizes = [64, 32]
    method.create_weights(
        layer, 128, sizes, 128, sum(sizes), torch.bfloat16, weight_loader=None
    )
    layer = layer.to("cuda")
    torch.manual_seed(0)
    reference = torch.randn(sum(sizes), 128, device="cuda") * 0.05
    shard_scales = torch.full((2,), reference.abs().amax().item() / 448.0)
    if not shared_scale:
        shard_scales[1] *= 2
    row_scales = shard_scales.repeat_interleave(torch.tensor(sizes)).cuda()[:, None]
    weight_fp8 = (reference / row_scales).to(torch.float8_e4m3fn)
    input_scale = 4.0 / 448.0
    layer.weight.data = weight_fp8.clone()
    layer.weight_scale.data.copy_(shard_scales)
    layer.input_scale.data.fill_(input_scale)
    method.process_weights_after_loading(layer)
    # A shared shard scale stays per-tensor; distinct ones become per-channel.
    assert layer.weight_scale.numel() == (1 if shared_scale else sum(sizes))

    x = torch.randn(8, 128, device="cuda", dtype=torch.bfloat16)
    out = method.apply(layer, x)
    x_fp8 = (x.float() / input_scale).clamp(-448, 448).to(torch.float8_e4m3fn)
    expected = (x_fp8.float() * input_scale) @ (weight_fp8.float() * row_scales).t()
    error = (out.float() - expected).norm() / expected.norm()
    assert error.item() < 1e-2

    # A producer that already quantized with the layer's scale gets the same GEMM.
    x_prequantized, _ = static_quant_fp8(x, layer.input_scale)
    out_prequantized = method.apply(layer, x_prequantized)
    assert out_prequantized.dtype == torch.bfloat16
    assert torch.equal(out_prequantized, out)


def test_mtp_checkpoint_names_map_onto_the_head():
    from tokenspeed.runtime.models.nemotron_h_nextn import _mtp_param_name

    assert _mtp_param_name("mtp.layers.0.eh_proj.weight") == "eh_proj.weight"
    assert _mtp_param_name("mtp.layers.0.enorm.weight") == "enorm.weight"
    assert _mtp_param_name("mtp.layers.0.hnorm.weight") == "hnorm.weight"
    assert _mtp_param_name("mtp.layers.0.norm.weight") == "layers.0.norm.weight"
    assert _mtp_param_name("mtp.layers.1.final_layernorm.weight") == (
        "final_layernorm.weight"
    )
    assert _mtp_param_name("mtp.layers.1.mixer.gate.weight") == (
        "layers.1.mixer.gate.weight"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
