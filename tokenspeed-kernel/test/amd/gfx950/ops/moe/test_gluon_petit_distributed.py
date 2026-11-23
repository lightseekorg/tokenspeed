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

"""Four- and eight-rank numerical checks for the registered Petit

Run with torchrun --standalone --nproc_per_node=<4 or 8> -m pytest
--import-mode=importlib <path-to-this-file>. Optional PETIT_OUTPUT_DIR saves outputs for an exact
before/after comparison. Local fixtures are independent of benchmark timing
scope. Normal single-process pytest skips the GPU test.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import pytest
import tokenspeed_kernel
import torch
import torch.distributed as dist
from tokenspeed_kernel.ops.moe import moe_apply as kernel_moe_apply
from tokenspeed_kernel.ops.moe import moe_plan as kernel_moe_plan
from tokenspeed_kernel.ops.moe import moe_process_weights as kernel_moe_process_weights
from utils import is_cdna4


@dataclass(frozen=True)
class _BenchmarkProfile:
    name: str
    experts: int
    top_k: int
    hidden: int
    logical_intermediate: int
    compute_intermediate: int
    activation: str
    has_bias: bool
    clamp: float | None = None
    world_sizes: tuple[int, ...] = (8,)


_PROFILES = {
    "gpt_oss_120b": _BenchmarkProfile(
        name="gpt_oss_120b",
        experts=128,
        top_k=4,
        hidden=2880,
        logical_intermediate=2880,
        compute_intermediate=3072,
        activation="swiglu",
        has_bias=True,
    ),
    "dsv4": _BenchmarkProfile(
        name="dsv4",
        experts=384,
        top_k=6,
        hidden=7168,
        logical_intermediate=3072,
        compute_intermediate=3072,
        activation="swiglu",
        has_bias=False,
    ),
    "kimi_k3": _BenchmarkProfile(
        name="kimi_k3",
        experts=896,
        top_k=16,
        hidden=3584,
        logical_intermediate=3072,
        compute_intermediate=3072,
        activation="situ",
        has_bias=False,
    ),
    "muse_spark": _BenchmarkProfile(
        name="muse_spark",
        experts=256,
        top_k=8,
        hidden=4096,
        logical_intermediate=4096,
        compute_intermediate=4096,
        activation="silu",
        has_bias=False,
    ),
    **{
        name: _BenchmarkProfile(
            name=name,
            experts=experts,
            top_k=top_k,
            hidden=5120,
            logical_intermediate=2304,
            compute_intermediate=2560,
            activation="swiglu",
            has_bias=False,
            clamp=10.0,
            world_sizes=(4, 8),
        )
        for name, experts, top_k in (
            ("dsv41_flash", 384, 6),
            ("dsv41_flash_dspark", 128, 3),
        )
    },
}


def _parameter(shape: tuple[int, ...], device: torch.device) -> torch.nn.Parameter:
    return torch.nn.Parameter(
        torch.zeros(shape, dtype=torch.uint8, device=device),
        requires_grad=False,
    )


class _TestMoeLayer(torch.nn.Module):
    def __init__(
        self, profile: _BenchmarkProfile, ep_size: int, device: torch.device
    ) -> None:
        super().__init__()
        local_experts = profile.experts // ep_size
        self.num_experts = profile.experts
        self.top_k = profile.top_k
        self.hidden_size = profile.hidden
        self.intermediate_size = profile.logical_intermediate
        self.num_local_experts = local_experts
        self.ep_size = ep_size
        self.tp_size = 1
        self.activation = profile.activation
        self.activation_situ_beta = 4.0 if profile.activation == "situ" else None
        self.activation_situ_linear_beta = (
            25.0 if profile.activation == "situ" else None
        )
        self.swiglu_beta = 1.0 if profile.has_bias else None
        if profile.has_bias:
            self.swiglu_arg = argparse.Namespace(alpha=1.702, limit=7.0)
        elif profile.clamp is not None:
            self.swiglu_arg = argparse.Namespace(alpha=None, limit=profile.clamp)
        else:
            self.swiglu_arg = None
        self.w13_input_layout = "interleaved" if profile.has_bias else "concatenated"
        self.register_parameter(
            "w13_weight",
            _parameter(
                (
                    local_experts,
                    2 * profile.logical_intermediate,
                    profile.hidden // 2,
                ),
                device,
            ),
        )
        self.register_parameter(
            "w13_weight_scale",
            _parameter(
                (
                    local_experts,
                    2 * profile.logical_intermediate,
                    profile.hidden // 32,
                ),
                device,
            ),
        )
        self.register_parameter(
            "w2_weight",
            _parameter(
                (
                    local_experts,
                    profile.hidden,
                    profile.logical_intermediate // 2,
                ),
                device,
            ),
        )
        self.register_parameter(
            "w2_weight_scale",
            _parameter(
                (
                    local_experts,
                    profile.hidden,
                    profile.logical_intermediate // 32,
                ),
                device,
            ),
        )
        if profile.has_bias:
            self.register_parameter(
                "w13_weight_bias",
                torch.nn.Parameter(
                    torch.zeros(
                        (local_experts, 2 * profile.logical_intermediate),
                        dtype=torch.bfloat16,
                        device=device,
                    ),
                    requires_grad=False,
                ),
            )
            self.register_parameter(
                "w2_weight_bias",
                torch.nn.Parameter(
                    torch.zeros(
                        (local_experts, profile.hidden),
                        dtype=torch.bfloat16,
                        device=device,
                    ),
                    requires_grad=False,
                ),
            )
        else:
            self.register_parameter("w13_weight_bias", None)
            self.register_parameter("w2_weight_bias", None)


def _make_plan(profile: _BenchmarkProfile, ep_size: int) -> dict:
    return kernel_moe_plan(
        "mxfp4",
        input_dtype=torch.bfloat16,
        activation=profile.activation,
        requires_deferred_finalize=False,
        routing_mode="precomputed_topk",
        a2a_backend="gluon_petit",
        ep_size=ep_size,
        ispp=profile.logical_intermediate,
        fp8_scale_block_shape=None,
        internal_activation_dtype="mxfp4",
        with_bias=profile.has_bias,
        process_group=None,
        hidden=profile.hidden,
        swiglu_form=(
            "generalized"
            if profile.has_bias
            else "standard" if profile.activation == "swiglu" else None
        ),
        activation_clamped=profile.has_bias or profile.clamp is not None,
        expert_id_repeats=False,
        fast_math=True,
        combine_order="rank",
        features=frozenset(),
        deepep_mode=None,
        deepep_low_latency_max_num_tokens_per_gpu=None,
        solution="gluon",
    )


def _routing(
    tokens: int,
    rank: int,
    profile: _BenchmarkProfile,
    world_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    token = torch.arange(tokens, dtype=torch.int32, device=device)[:, None]
    route = torch.arange(profile.top_k, dtype=torch.int32, device=device)[None, :]
    local_experts = profile.experts // world_size
    destination = (rank + token + token // 16 + route) % world_size
    local = (token * profile.top_k + route) % local_experts
    ids = (destination * local_experts + local).to(torch.int32).contiguous()
    raw_weights = 1.0 + ((token + 2 * route) % 7).to(torch.float32)
    weights = (raw_weights / raw_weights.sum(dim=1, keepdim=True)).contiguous()
    return ids, weights


def _quantized_reference(
    value: torch.Tensor, *, input_quantization: bool
) -> torch.Tensor:
    """Reference input/intermediate scales and nearest-even E2M1 values."""
    shape = value.shape
    blocks = value.float().reshape(-1, 32)
    maximum = blocks.abs().amax(dim=1)
    if input_quantization:
        bits = (maximum * (1.0 / 6.0)).contiguous().view(torch.int32)
        exponent = (bits >> 23) + ((bits & 0x7FFFFF) != 0).int()
    else:
        bits = maximum.contiguous().view(torch.int32)
        exponent = (((bits + 0x00400000) & 0x7F800000) >> 23).clamp(min=2) - 2
    scale = (exponent << 23).view(torch.float32)[:, None]
    # A zero scale denotes an all-zero block; avoid division by zero in the oracle.
    scale = torch.where(scale == 0, 1.0, scale)
    normalized = blocks.abs() / scale
    # Tie priority is even E2M1 code, matching the hardware RNE conversion.
    levels = torch.tensor([0.0, 1.0, 2.0, 4.0, 0.5, 1.5, 3.0, 6.0], device=value.device)
    closest = (normalized[..., None] - levels).abs().argmin(dim=-1)
    return (levels[closest] * blocks.sign() * scale).reshape(shape)


def _weights(layer, profile, rank: int, world_size: int) -> None:
    """Sparse nonzero weights give an independent, inexpensive exact matmul oracle."""
    device = layer.w13_weight.device
    inter = profile.logical_intermediate
    hidden = profile.hidden
    rows = torch.arange(2 * inter, device=device)
    gate = rows % 2 == 0 if profile.has_bias else rows < inter
    channel = rows // 2 if profile.has_bias else rows % inter
    cols = (channel * 7 + torch.where(gate, 0, 3)) % hidden
    for local in range(profile.experts // world_size):
        expert = rank * (profile.experts // world_size) + local
        codes = torch.where(gate, 2 + expert % 3, 2 + (expert + 1) % 3).to(torch.uint8)
        layer.w13_weight[local, rows, cols // 2] = (codes << ((cols % 2) * 4)).to(
            torch.uint8
        )
        out_rows = torch.arange(hidden, device=device)
        out_cols = (out_rows * 11 + expert) % inter
        layer.w2_weight[local, out_rows, out_cols // 2] = (
            (2 + expert % 3) << ((out_cols % 2) * 4)
        ).to(torch.uint8)
        if profile.has_bias:
            layer.w13_weight_bias[local].copy_(
                torch.where(gate, 0.03125 * (expert % 3 + 1), -0.03125)
            )
            layer.w2_weight_bias[local].fill_(0.015625 * (expert % 5 - 2))
    layer.w13_weight_scale.data.fill_(123)
    layer.w2_weight_scale.data.fill_(123)


def _route_reference(x, ids, profile):
    """Each selected expert's BF16 output, ``[tokens, top_k, hidden]`` in FP32."""
    xq = _quantized_reference(x, input_quantization=True)
    inter = profile.logical_intermediate
    channel = torch.arange(inter, device=x.device)
    output_cols = torch.arange(profile.hidden, device=x.device)
    routes = torch.zeros(
        (*ids.shape, profile.hidden), dtype=torch.float32, device=x.device
    )
    # The sparse weights above select one input per matrix row. Both matrix
    # products are therefore exact FP32 multiplies, with no reduction ambiguity.
    levels = (1.0, 1.5, 2.0)
    for expert in range(profile.experts):
        token, slot = torch.where(ids == expert)
        if token.numel() == 0:
            continue
        gate = xq[token[:, None], (channel * 7) % profile.hidden] * (
            levels[expert % 3] / 16
        )
        up = xq[token[:, None], (channel * 7 + 3) % profile.hidden] * (
            levels[(expert + 1) % 3] / 16
        )
        if profile.has_bias:
            gate += 0.03125 * (expert % 3 + 1)
            up -= 0.03125
            gate = gate.clamp(max=7)
            hidden = gate * torch.sigmoid(1.702 * gate) * (up.clamp(-7, 7) + 1)
        elif profile.clamp is not None:
            hidden = torch.nn.functional.silu(gate.clamp(max=profile.clamp)) * (
                up.clamp(-profile.clamp, profile.clamp)
            )
        elif profile.activation == "situ":
            hidden = (4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate)) * (
                25.0 * torch.tanh(up / 25.0)
            )
        else:
            hidden = gate * torch.sigmoid(gate) * up
        hidden = _quantized_reference(hidden, input_quantization=False)
        result = hidden[:, (output_cols * 11 + expert) % inter] * (
            levels[expert % 3] / 16
        )
        if profile.has_bias:
            result += 0.015625 * (expert % 5 - 2)
        routes[token, slot] = result
    return routes


def _reference(x, ids, weights, profile):
    routes = _route_reference(x, ids, profile)
    # The kernel rounds each weighted route to BF16 before FP32 combine.
    routed = (routes * weights[..., None]).to(torch.bfloat16).float()
    return routed.sum(dim=1).to(torch.bfloat16)


_WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))


@pytest.mark.skipif(
    _WORLD_SIZE not in (4, 8) or not is_cdna4(),
    reason="requires torchrun with four or eight gfx950 GPUs",
)
@pytest.mark.parametrize("capacity", (1024, 4096))
@pytest.mark.parametrize("profile_name", tuple(_PROFILES))
def test_distributed_petit_reference(profile_name, capacity):
    profile = _PROFILES[profile_name]
    if _WORLD_SIZE not in profile.world_sizes:
        sizes = "/".join(map(str, profile.world_sizes))
        pytest.skip(f"{profile_name} is registered for EP{sizes} only")
    if capacity != 1024 and _WORLD_SIZE != 4:
        pytest.skip("Petit admits a per-rank capacity above 1024 rows at EP4 only")
    expected_root = os.environ.get("PETIT_EXPECTED_RUNTIME_ROOT")
    if expected_root is not None:
        assert (
            Path(tokenspeed_kernel.__file__)
            .resolve()
            .is_relative_to(Path(expected_root).resolve())
        ), "pytest imported a different runtime checkout"
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group("nccl", timeout=timedelta(minutes=5))
    rank = dist.get_rank()
    device = torch.device("cuda", local_rank)
    ws = _WORLD_SIZE
    layer = _TestMoeLayer(profile, ws, device)
    _weights(layer, profile, rank, ws)
    plan = _make_plan(profile, ws)
    if capacity != 1024:
        # The runtime passes ServerArgs.gluon_petit_max_tokens_per_rank here.
        plan["gluon_petit_max_tokens_per_rank"] = capacity
    kernel_moe_process_weights(plan, layer)
    assert layer._moe_backend_state.max_tokens_per_rank == capacity
    output_dir = os.environ.get("PETIT_OUTPUT_DIR")
    # Exercise empty source ranks, padding/tile boundaries and capacity, then
    # refresh both hidden states and destinations within the same captured graph.
    # Petit picks its producer geometry per rank from the token count; for the
    # DSpark draft it switches at 12 and 24 tokens. EP4 keeps the empty and
    # single-token ranks and starts one rank at each switch, so one call still
    # mixes all three geometries, as the EP8 counts do.
    cases = (
        (
            ("balanced", [16] * ws),
            ("uneven", {8: [0, 1, 15, 16, 31, 32, 63, 64], 4: [0, 1, 12, 24]}[ws]),
            ("partial_scales", [256] * ws),
            # For both DSv4.1 profiles, 640 tokens take the 4-warp 64-row stage-1
            # tile (Flash 480-1,007 tokens, DSpark 352-703).
            ("stage1_m64_w4", [640] * ws),
        )
        if capacity == 1024
        else (
            # A raised capacity still serves small and uneven calls.
            ("uneven", [0, 1, 12, 24]),
            # Every rank above the old 1024-row cap at once (EP4 2048-row chunks).
            ("above_1024", [2048] * ws),
            # Every rank at the raised capacity fills each destination's pool.
            ("full", [capacity] * ws),
        )
    ) + (("capacity", [capacity] + [1] * (ws - 1)),)
    for case, counts in cases:
        count = counts[rank]
        x = torch.empty((count, profile.hidden), device=device, dtype=torch.bfloat16)
        ids, weights = _routing(count, rank, profile, ws, device)

        def refresh(step):
            values = torch.arange(x.numel(), device=device).reshape(x.shape)
            # Clamped profiles need inputs large enough to cross the clamp.
            input_scale = (
                4.0
                if profile.activation == "situ" or profile.clamp is not None
                else 1.0 / 32
            )
            x.copy_(
                (((values * 13 + rank * 7 + step * 17) % 127) - 63).float()
                * input_scale
            )
            new_ids, new_weights = _routing(count, rank + step, profile, ws, device)
            if step:
                # Concentrate routes on one destination without duplicate experts.
                new_ids = (new_ids % (profile.experts // ws)) + (step % ws) * (
                    profile.experts // ws
                )
            ids.copy_(new_ids)
            weights.copy_(new_weights)

        def run():
            return kernel_moe_apply(
                plan,
                x,
                layer,
                x.new_empty((count, 0)),
                topk_weights=weights,
                topk_ids=ids,
                num_tokens_global=sum(counts),
                max_num_tokens_per_gpu=max(counts),
                do_finalize=True,
                low_latency=None,
                overlap_fn=None,
                shared_input=None,
                shared_weight=None,
                shared_out=None,
            )

        refresh(0)
        eager = run().clone()
        torch.cuda.synchronize()
        dist.barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = run()
        for step in (0, 1, 2):
            refresh(step)
            graph.replay()
            actual = captured.clone()
            expected = _reference(x, ids, weights, profile)
            # Sparse inputs keep the quantization-aware reference tight; BF16
            # route rounding permits one final BF16 rounding difference.
            torch.testing.assert_close(actual, expected, atol=2e-6, rtol=8e-3)
            torch.testing.assert_close(
                actual, eager if step == 0 else run(), atol=0, rtol=0
            )
            assert torch.isfinite(actual).all()
            if output_dir:
                directory = Path(output_dir)
                directory.mkdir(parents=True, exist_ok=True)
                torch.save(
                    actual.cpu(),
                    directory
                    / (
                        f"{profile_name}-{case}-{step}-rank{rank}.pt"
                        if capacity == 1024
                        else f"{profile_name}-cap{capacity}-{case}-{step}-rank{rank}.pt"
                    ),
                )
            dist.barrier()


_PETIT_KERNELS = frozenset({"MegaMoEStage1", "MegaMoEStage2", "MegaMoECombine"})


@pytest.mark.skipif(
    _WORLD_SIZE != 4 or not is_cdna4(),
    reason="requires torchrun with four gfx950 GPUs",
)
@pytest.mark.parametrize("capacity", (1024, 4096))
@pytest.mark.parametrize("profile_name", ("dsv41_flash", "dsv41_flash_dspark"))
def test_distributed_petit_startup_warmup_compiles_every_variant(
    profile_name, capacity
):
    """After one Petit call under startup warmup, no serving count compiles.

    The warmup call compiles (without launching) every kernel variant a rank
    can launch at this capacity; afterwards Triton's in-process cache must
    hit for every count, including the 848-row tail of a 50,000-token prompt
    that once compiled for 27-28 s inside a serving step.
    """
    import triton
    from tokenspeed_kernel.ops.moe import gluon as petit
    from tokenspeed_kernel.ops.tuning import kernel_warmup

    profile = _PROFILES[profile_name]
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group("nccl", timeout=timedelta(minutes=5))
    rank = dist.get_rank()
    device = torch.device("cuda", local_rank)
    ws = _WORLD_SIZE
    layer = _TestMoeLayer(profile, ws, device)
    _weights(layer, profile, rank, ws)
    plan = _make_plan(profile, ws)
    plan["gluon_petit_max_tokens_per_rank"] = capacity
    kernel_moe_process_weights(plan, layer)
    state = layer._moe_backend_state
    workspace = petit._get_workspace(device, state.profile, capacity)
    # Another test in this process may have compiled this workspace already.
    petit._compiled_variant_keys.discard(
        petit._workspace_key(device, state.profile, capacity)
    )

    def run(counts, do_finalize=True):
        count = counts[rank]
        x = torch.zeros((count, profile.hidden), device=device, dtype=torch.bfloat16)
        ids, weights = _routing(count, rank, profile, ws, device)
        out = kernel_moe_apply(
            plan,
            x,
            layer,
            x.new_empty((count, 0)),
            topk_weights=weights,
            topk_ids=ids,
            num_tokens_global=sum(counts),
            max_num_tokens_per_gpu=max(counts),
            do_finalize=do_finalize,
            low_latency=None,
            overlap_fn=None,
            shared_input=None,
            shared_weight=None,
            shared_out=None,
        )
        torch.cuda.synchronize()
        dist.barrier()
        return out

    with kernel_warmup(enabled=True):
        run([16] * ws)
    counts = workspace.config.variant_token_counts()
    expected = {
        "dsv41_flash": (0, 1, 2, 480, 481, 512, 513, 1008, 1009),
        "dsv41_flash_dspark": (0, 1, 2, 12, 16, 24, 32, 352, 353, 704, 705),
    }
    assert counts == expected[profile_name]

    compiled = []
    previous = triton.knobs.runtime.jit_cache_hook

    def cache_hook(**kwargs):
        # Triton calls this on every in-process cache miss, before it compiles
        # or loads a kernel from its disk cache.
        if kwargs["fn"].name in _PETIT_KERNELS:
            compiled.append((kwargs["fn"].name, kwargs["key"]))
        return previous(**kwargs) if previous is not None else None

    triton.knobs.runtime.jit_cache_hook = cache_hook
    try:
        for count in sorted({*counts, 848, 849, 1023, 1024, capacity - 1, capacity}):
            for do_finalize in (True, False):
                run([count] * ws, do_finalize)
        # Uneven ranks, as data-parallel attention produces them.
        run([848, 0, 1, capacity])
        run([3, 848, 1009, 0], do_finalize=False)
    finally:
        triton.knobs.runtime.jit_cache_hook = previous
    assert not compiled, f"rank {rank} compiled Petit kernels in serving: {compiled}"


@pytest.mark.skipif(
    int(os.environ.get("WORLD_SIZE", "1")) != 8 or not is_cdna4(),
    reason="requires torchrun with eight gfx950 GPUs",
)
@pytest.mark.parametrize("profile_name", sorted(_PROFILES))
def test_distributed_petit_deferred_routes(profile_name):
    """Deferred finalize returns every unweighted route to its source rank.

    Because finalized and deferred launches share one workspace, the test
    alternates them in eager execution and CUDA graph replay with changed
    inputs.
    """
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group("nccl", timeout=timedelta(minutes=5))
    rank = dist.get_rank()
    device = torch.device("cuda", local_rank)
    profile = _PROFILES[profile_name]
    layer = _TestMoeLayer(profile, 8, device)
    _weights(layer, profile, rank, 8)
    plan = _make_plan(profile, 8)
    assert plan["supports_deferred_finalize"]
    kernel_moe_process_weights(plan, layer)
    for counts in (
        [0, 1, 15, 16, 31, 32, 63, 64],
        [128] * 8,
        [1024] + [0] * 7,
    ):
        count = counts[rank]
        x = torch.empty((count, profile.hidden), device=device, dtype=torch.bfloat16)
        ids, weights = _routing(count, rank, profile, 8, device)

        def refresh(step):
            values = torch.arange(x.numel(), device=device).reshape(x.shape)
            input_scale = 4.0 if profile.activation == "situ" else 1.0 / 32
            x.copy_(
                (((values * 13 + rank * 7 + step * 17) % 127) - 63).float()
                * input_scale
            )
            new_ids, new_weights = _routing(count, rank + step, profile, 8, device)
            if step:
                new_ids = (new_ids % (profile.experts // 8)) + (step % 8) * (
                    profile.experts // 8
                )
            ids.copy_(new_ids)
            weights.copy_(new_weights)

        def run(do_finalize):
            return kernel_moe_apply(
                plan,
                x,
                layer,
                x.new_empty((count, 0)),
                topk_weights=weights,
                topk_ids=ids,
                num_tokens_global=sum(counts),
                max_num_tokens_per_gpu=max(counts),
                do_finalize=do_finalize,
                low_latency=None,
                overlap_fn=None,
                shared_input=None,
                shared_weight=None,
                shared_out=None,
            )

        refresh(0)
        routes, route_weights, route_map = run(False)
        assert routes.shape == (count, profile.top_k, profile.hidden)
        assert route_weights is weights and route_map is None
        eager_routes = routes.clone()
        # A finalized launch reuses the workspace and must restore the weights.
        eager_combined = run(True).clone()
        torch.cuda.synchronize()
        dist.barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured_routes, _, _ = run(False)
            captured_combined = run(True)
        for step in (0, 1, 2):
            refresh(step)
            graph.replay()
            # One exact nonzero per weight row: routes match the reference bits.
            expected = _route_reference(x, ids, profile).to(torch.bfloat16)
            torch.testing.assert_close(captured_routes, expected, atol=0, rtol=0)
            torch.testing.assert_close(
                captured_combined,
                _reference(x, ids, weights, profile),
                atol=2e-6,
                rtol=8e-3,
            )
            if step == 0:
                torch.testing.assert_close(
                    captured_routes, eager_routes, atol=0, rtol=0
                )
                torch.testing.assert_close(
                    captured_combined, eager_combined, atol=0, rtol=0
                )
            dist.barrier()


@pytest.mark.skipif(
    int(os.environ.get("WORLD_SIZE", "1")) != 8 or not is_cdna4(),
    reason="requires torchrun with eight gfx950 GPUs",
)
def test_distributed_petit_routes_follow_back_to_back_launches():
    """Back-to-back small launches never return an earlier launch's routes.

    Peers write each launch's routes into the same workspace rows; the combine
    copies them out with coherent loads, and an ordinary kernel reads every
    copy immediately, with no other traffic in between to evict cached lines.
    """
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group("nccl", timeout=timedelta(minutes=5))
    rank = dist.get_rank()
    device = torch.device("cuda", local_rank)
    profile = _PROFILES["muse_spark"]
    layer = _TestMoeLayer(profile, 8, device)
    _weights(layer, profile, rank, 8)
    plan = _make_plan(profile, 8)
    kernel_moe_process_weights(plan, layer)
    counts = [1, 2, 3, 4, 1, 2, 3, 4]
    count = counts[rank]
    iterations = 48
    inputs = []
    for step in range(iterations):
        values = torch.arange(count * profile.hidden, device=device).reshape(
            count, profile.hidden
        )
        x = ((((values * 13 + rank * 7 + step * 29) % 127) - 63).float() / 32).to(
            torch.bfloat16
        )
        ids, weights = _routing(count, rank + step, profile, 8, device)
        inputs.append((x, ids, weights))
    x = torch.empty_like(inputs[0][0])
    ids = torch.empty_like(inputs[0][1])
    weights = torch.empty_like(inputs[0][2])
    sums = torch.empty((iterations, count, profile.top_k), device=device)

    def run_all():
        for step, (step_x, step_ids, step_weights) in enumerate(inputs):
            x.copy_(step_x)
            ids.copy_(step_ids)
            weights.copy_(step_weights)
            routes, _, _ = kernel_moe_apply(
                plan,
                x,
                layer,
                x.new_empty((count, 0)),
                topk_weights=weights,
                topk_ids=ids,
                num_tokens_global=sum(counts),
                max_num_tokens_per_gpu=max(counts),
                do_finalize=False,
                low_latency=None,
                overlap_fn=None,
                shared_input=None,
                shared_weight=None,
                shared_out=None,
            )
            sums[step].copy_(routes.float().sum(-1))

    expected = torch.stack(
        [
            _route_reference(step_x, step_ids, profile)
            .to(torch.bfloat16)
            .float()
            .sum(-1)
            for step_x, step_ids, _ in inputs
        ]
    )
    run_all()
    torch.testing.assert_close(sums, expected, atol=0, rtol=0)
    torch.cuda.synchronize()
    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run_all()
    for _ in range(3):
        sums.zero_()
        graph.replay()
        torch.testing.assert_close(sums, expected, atol=0, rtol=0)
        dist.barrier()
