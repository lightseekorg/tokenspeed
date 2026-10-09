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

"""Kimi checkpoint layout and model projection numerics at native layer widths."""

import os
import sys
from datetime import timedelta
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=120, suite="runtime-2gpu")


@pytest.mark.parametrize(
    ("sizes", "mapping_overrides", "message"),
    [
        ((1, 3), {}, "positive divisors"),
        ((0, 1), {}, "positive divisors"),
        ((2, 1), {"attn_tp_size": 2}, "attention TP1/DPworld"),
        ((2, 1), {"linear_attn_tp_size": 2}, "linear attention TP1"),
        ((2, 1), {"attn_head_tp_size": 2}, "attention TP1/DPworld"),
        ((1, 2), {"moe_ep_size": 2}, "MoE TP1/EPworld"),
        ((2, 2), {"pp_size": 2, "moe_ep_size": 2}, "PP1"),
    ],
)
def test_projection_settings_reject_invalid_configuration(
    sizes, mapping_overrides, message
):
    from tokenspeed.runtime.distributed.mapping import Mapping
    from tokenspeed.runtime.models.kimi_k3_projection import (
        validate_projection_settings,
    )

    mapping_args = dict(
        rank=0,
        world_size=4,
        attn_tp_size=1,
        linear_attn_tp_size=1,
        dense_tp_size=1,
        moe_tp_size=1,
        moe_ep_size=4,
        pp_size=1,
    )
    mapping_args.update(mapping_overrides)
    mapping = Mapping(**mapping_args)
    with pytest.raises(ValueError, match=message):
        validate_projection_settings(mapping, *sizes)


@torch.no_grad()
def test_checkpoint_shards():
    from tokenspeed.runtime.distributed.mapping import DenseLayerMapping
    from tokenspeed.runtime.models.kimi_k3_projection import (
        KimiKDAColumnProj,
        KimiMLAColumnProj,
        projection_storage_rows,
    )
    from tokenspeed.runtime.utils import ceil_div

    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA GPU")
    hidden, proj, head_dim, heads, tp = 7168, 12288, 128, 96, 4
    device = torch.device("cuda", 0)
    # Native K3 segment widths include a short beta row and a ragged MLA KV tail.
    for kind, widths in (
        ("kda", [proj] * 4 + [head_dim, heads]),
        ("mla", [12288, 1536, 576]),
    ):
        logical = sum(widths)
        for block_fp8 in (False, True):
            stored = projection_storage_rows(logical, tp, block_fp8)
            codes = (torch.arange(stored, device=device) % 31 - 15).to(torch.bfloat16)
            weight = codes[:, None].expand(-1, hidden).contiguous()
            weight[logical:].zero_()
            if block_fp8:
                weight = weight.to(torch.float8_e4m3fn)
                scales = (
                    (
                        torch.arange(stored // 128, device=device, dtype=torch.float32)
                        % 7
                        + 1
                    )[:, None]
                    .expand(-1, hidden // 128)
                    .contiguous()
                )
                scales[ceil_div(logical, 128) :].fill_(1)
            for rank in range(tp):
                parallel = DenseLayerMapping(
                    rank=rank, world_size=tp, tp_size=tp, dp_size=1
                )
                with torch.device(device):
                    if kind == "kda":
                        module = KimiKDAColumnProj(
                            hidden,
                            proj,
                            heads,
                            head_dim,
                            parallel,
                            block_fp8,
                            "qkvgb_proj",
                        )
                    else:
                        module = KimiMLAColumnProj(
                            hidden,
                            logical,
                            parallel,
                            block_fp8,
                            "fused_qkv_a_proj_with_mqa",
                        )
                if kind == "kda":
                    start = 0
                    for name, width in zip(("q", "k", "v", "g", "f_a", "b"), widths):
                        module.weight.weight_loader(
                            module.weight, weight[start : start + width], name
                        )
                        if block_fp8:
                            module.weight_scale_inv.weight_loader(
                                module.weight_scale_inv,
                                scales[start // 128 : ceil_div(start + width, 128)],
                                name,
                            )
                        start += width
                    module.verify_load_complete()
                elif block_fp8:
                    # Production MLA assembles gate/q_a/kv_a, then the parent shards it.
                    from tokenspeed.runtime.models.kimi_k3 import (
                        _assemble_fp8_fused_qkv_a,
                    )

                    starts = (0, 12288, 13824)
                    assembled, assembled_scales = _assemble_fp8_fused_qkv_a(
                        [
                            (
                                weight[start : start + rows],
                                scales[start // 128 : ceil_div(start + rows, 128)],
                            )
                            for start, rows in zip(starts, widths)
                        ],
                        total_rows=module.output_size,
                    )
                    module.weight.weight_loader(module.weight, assembled)
                    module.weight_scale_inv.weight_loader(
                        module.weight_scale_inv, assembled_scales
                    )
                else:
                    # Canonical BF16 order is q_a/kv_a/gate, unlike FP8's private order.
                    start = 0
                    for width in (1536, 576, 12288):
                        module.weight.weight_loader(
                            module.weight,
                            weight[start : start + width],
                            begin_size=start,
                        )
                        start += width
                width = stored // tp
                expected = weight[rank * width : (rank + 1) * width]
                torch.testing.assert_close(
                    module.weight.view(torch.uint8),
                    expected.view(torch.uint8),
                    rtol=0,
                    atol=0,
                )
                if block_fp8:
                    expected_scales = scales[
                        rank * width // 128 : (rank + 1) * width // 128
                    ]
                    torch.testing.assert_close(
                        module.weight_scale_inv, expected_scales, rtol=0, atol=0
                    )
                del module
            del weight


@torch.no_grad()
def _model_worker(rank, size, rendezvous):
    from tokenspeed_kernel.ops.activation.triton import rmsnorm_gated_sigmoid

    from tokenspeed.runtime.configs.kimi_k3_config import KimiLinearConfig
    from tokenspeed.runtime.distributed.comm_backend.auto import AutoBackend
    from tokenspeed.runtime.distributed.mapping import Mapping
    from tokenspeed.runtime.execution.context import ForwardContext
    from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
    from tokenspeed.runtime.execution.output_layout import ForwardOutputLayout
    from tokenspeed.runtime.layers.linear import prepare_dp_linear_communication
    from tokenspeed.runtime.models.kimi_k3 import KimiLinearKDA, KimiLinearMLAAttention
    from tokenspeed.runtime.models.kimi_k3_projection import (
        validate_projection_settings,
    )

    torch.cuda.set_device(rank)
    torch.set_default_dtype(torch.bfloat16)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "nccl",
        init_method=rendezvous,
        rank=rank,
        world_size=size,
        timeout=timedelta(seconds=240),
        device_id=device,
    )
    mapping = Mapping(
        rank=rank,
        world_size=size,
        attn_tp_size=1,
        attn_dp_size=size,
        linear_attn_tp_size=1,
        dense_tp_size=1,
        moe_tp_size=1,
        moe_ep_size=size,
    )
    parallel, _ = validate_projection_settings(mapping, size, size)
    hidden, proj, heads, head_dim = 7168, 12288, 96, 128
    config = KimiLinearConfig(
        hidden_size=hidden,
        num_attention_heads=heads,
        q_lora_rank=1536,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        mla_use_output_gate=True,
        linear_attn_config={
            "kda_layers": [1],
            "full_attn_layers": [2],
            "num_heads": heads,
            "head_dim": head_dim,
            "short_conv_kernel_size": 4,
            "gate_lower_bound": -5.0,
            "use_full_rank_gate": True,
        },
    )

    def sparse_weight(rows, cols):
        value = torch.zeros(rows, cols, device=device)
        index = torch.arange(rows, device=device)
        value[index, (index * 7) % cols] = 0.25
        return value

    # Isolate the projection integration from cache/attention numerics. This
    # deterministic local attention consumes restored rows on each owner; all
    # projection GEMMs and collectives below are the production implementations.
    def local_kda(**kwargs):
        value = kwargs["mixed_qkv"][:, 2 * proj : 3 * proj].contiguous()
        if kwargs["output_gate"] is not None:
            value = rmsnorm_gated_sigmoid(
                value,
                kwargs["output_gate"],
                kwargs["norm_weight"],
                kwargs["norm_eps"],
                heads,
                head_dim,
                enable_pdl=False,
            )
        return value

    comm = SimpleNamespace(pre_attn_comm=lambda value, ctx: value)
    for kind, gated in (("kda", True), ("mla", True), ("mla", False)):
        config.mla_use_output_gate = gated
        modules = []
        projection_mappings = (
            (None, None),
            (parallel, None),
            (None, parallel),
            (parallel, parallel),
        )
        if kind == "mla" and not gated:
            # Ungated MLA supports O-only TP, including empty token owners.
            projection_mappings = ((None, None), (None, parallel))
        for qkv_tp, output_tp in projection_mappings:
            with torch.device(device):
                if kind == "kda":
                    module = KimiLinearKDA(
                        config,
                        mapping,
                        0,
                        quant_config=None,
                        prefix="self_attn",
                        qkv_parallel=qkv_tp,
                        output_parallel=output_tp,
                    )
                    module.o_norm.weight.fill_(1)
                else:
                    module = KimiLinearMLAAttention(
                        config=config,
                        mapping=mapping,
                        hidden_size=hidden,
                        num_heads=heads,
                        qk_nope_head_dim=128,
                        qk_rope_head_dim=64,
                        v_head_dim=128,
                        q_lora_rank=1536,
                        kv_lora_rank=512,
                        quant_config=None,
                        prefix="self_attn",
                        qkv_parallel=qkv_tp,
                        output_parallel=output_tp,
                    )
                    module.fused_qk_layernorm.weight_q_a.fill_(1)
                    module.fused_qk_layernorm.weight_kv_a.fill_(1)
                    module.q_b_proj.weight.copy_(sparse_weight(heads * 192, 1536))
            modules.append(module)
        if kind == "kda":
            for name, width in zip(
                ("q", "k", "v", "g", "f_a", "b"), [proj] * 4 + [head_dim, heads]
            ):
                weight = sparse_weight(width, hidden)
                for module in modules:
                    linear = module.qkvgb_proj
                    linear.weight.weight_loader(linear.weight, weight, name)
        else:
            start = 0
            for width in ((1536, 576, proj) if gated else (1536, 576)):
                weight = sparse_weight(width, hidden)
                for module in modules:
                    linear = module.fused_qkv_a_proj_with_mqa
                    linear.weight.weight_loader(linear.weight, weight, begin_size=start)
                start += width
        weight = sparse_weight(hidden, proj)
        # Every input-channel shard contributes to each output channel.
        index = torch.arange(hidden, device=device)
        weight.zero_()
        for shard in range(size):
            weight[index, index % (proj // size) + shard * (proj // size)] = 0.25
        for module in modules:
            module.o_proj.weight.weight_loader(module.o_proj.weight, weight)
        model = torch.nn.ModuleList(modules)
        prepare_dp_linear_communication(model, 256, torch.bfloat16, AutoBackend())

        def forward(module, x, ctx):
            # MLA core attention is replaced only at its local computation
            # boundary; construction, Q-A/Q-B/norm/gate/O calls remain real.
            if kind == "mla":
                with (
                    mock.patch.object(
                        module, "_prefill_prologue_before_break", return_value=None
                    ),
                    mock.patch.object(
                        module,
                        "_attn",
                        side_effect=lambda positions, q, latent_cache, ctx, **kw: q[
                            :, :proj
                        ].contiguous(),
                    ),
                ):
                    return module(
                        torch.empty(x.shape[0], device=device),
                        x,
                        ctx,
                        comm,
                        projection_out=None,
                    )
            return module(
                torch.empty(x.shape[0], device=device),
                x,
                ctx,
                comm,
                projection_out=None,
            )

        for counts, mode in (
            ([32] * size, ForwardMode.DECODE),
            ([0, 64, 32, 1][:size], ForwardMode.DECODE),
            ([128] * size, ForwardMode.DECODE),
            ([0] * size, ForwardMode.DECODE),
            ([256] * size, ForwardMode.EXTEND),
        ):
            rows = counts[rank]
            x = (
                (
                    torch.arange(rows * hidden, device=device).view(rows, hidden) % 17
                    - 8
                    + rank
                )
                / 8
            ).to(torch.bfloat16)
            ctx = ForwardContext(
                attn_backend=SimpleNamespace(
                    forward=local_kda, supports_mla_projected_value_decode=False
                ),
                token_to_kv_pool=None,
                bs=rows,
                num_extends=int(mode != ForwardMode.DECODE),
                input_num_tokens=rows,
                forward_mode=mode,
                output_layout=ForwardOutputLayout(0, 0, rows, 1),
                global_num_tokens=counts,
            )
            expected = forward(modules[0], x, ctx)
            for index, module in enumerate(modules[1:], start=1):
                actual = forward(module, x, ctx)
                torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.008)
                forward(module, -x, ctx)
                torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.008)
                # A different layer may overwrite the same prepared scratch.
                following = modules[1 + index % (len(modules) - 1)]
                forward(following, -x, ctx)
                torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.008)
                if not max(counts):
                    continue
                for _ in range(2):
                    forward(module, x, ctx)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    captured = forward(module, x, ctx)
                original = x.clone()
                x.mul_(0.5).add_(0.125)
                refreshed = forward(module, x, ctx)
                graph.replay()
                torch.testing.assert_close(captured, refreshed, rtol=0, atol=0)
                torch.cuda.synchronize()
                x.copy_(original)
                del graph, captured
        # Shared communication is closed once after its final graph consumer.
        from tokenspeed.runtime.layers.linear import release_dp_linear_communication

        release_dp_linear_communication(model)
        del model, modules
    dist.destroy_process_group()


@pytest.mark.parametrize("size", [2, 4], ids=["tp2", "tp4"])
def test_model_projection_paths(size, tmp_path):
    if torch.cuda.device_count() < size:
        pytest.skip(f"requires {size} CUDA GPUs")
    mp.spawn(
        _model_worker,
        args=(size, f"file://{tmp_path / 'rendezvous'}"),
        nprocs=size,
        join=True,
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
