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

"""NVFP4 MegaMoE gated SiLU and SiTU correctness, including EP and graph replay.

Run directly with torchrun; WORLD_SIZE=16 uses all 896 K3 experts.
Pass --autotune to sweep all tactics before checking eager and graph results.
"""

from __future__ import annotations

import argparse
import os
import tempfile
from contextlib import nullcontext
from datetime import timedelta
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
from tokenspeed_kernel.numerics.torch.quantization import (
    dequantize_nvfp4_blocks as _dequant,
)
from utils import is_nvidia

if not is_nvidia():
    pytest.skip("NVIDIA GPU required", allow_module_level=True)

from flashinfer.autotuner import AutoTuner  # noqa: E402
from tokenspeed_kernel.ops.moe import (  # noqa: E402
    moe_apply,
    moe_plan,
    moe_process_weights,
)
from tokenspeed_kernel.ops.quantization.flashinfer import fp4_quantize  # noqa: E402
from tokenspeed_kernel.ops.tuning import (  # noqa: E402
    autotune,
    get_autotune_max_num_tokens,
    set_autotune_max_num_tokens,
    set_autotune_process_group,
)

HIDDEN = 3584
INTERMEDIATE = 192
LOCAL_EXPERTS = 56
TOP_K = 16


def _requant(x, scale):
    values = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device=x.device)
    blocks = x.reshape(*x.shape[:-1], -1, 16)
    sf = (blocks.abs().amax(-1) / (6 * scale)).to(torch.float8_e4m3fn).float()
    normalized = blocks / (sf.unsqueeze(-1) * scale).clamp_min(1e-30)
    # Ties select an even E2M1 code, matching round-to-nearest-even.
    distances = (normalized.abs().unsqueeze(-1) - values).abs()
    nearest = distances.argmin(-1)
    tie = (nearest % 2 == 1) & (nearest < 7)
    next_distance = distances.gather(
        -1, (nearest + 1).clamp_max(7).unsqueeze(-1)
    ).squeeze(-1)
    nearest = torch.where(
        tie & (next_distance == distances.amin(-1)), nearest + 1, nearest
    )
    return (values[nearest] * normalized.sign() * sf.unsqueeze(-1) * scale).reshape_as(
        x
    )


def _weights(plan, activation):
    w = torch.nn.Module()
    w.tp_size = 1
    w.num_experts = LOCAL_EXPERTS * dist.get_world_size()
    w.num_local_experts = LOCAL_EXPERTS
    w.hidden_size = HIDDEN
    w.intermediate_size = INTERMEDIATE
    w.top_k = TOP_K
    w.activation_situ_beta = 4.0 if activation == "situ" else None
    w.activation_situ_linear_beta = 25.0 if activation == "situ" else None
    one = torch.ones(1, device="cuda")
    reference = []
    for name, rows, cols, amplitude, global_scale in (
        ("w13", 2 * INTERMEDIATE, HIDDEN, 0.25, 0.8),
        ("w2", HIDDEN, INTERMEDIATE, 0.04, 0.6),
    ):
        raw = (
            torch.randn(LOCAL_EXPERTS * rows, cols, device="cuda") * amplitude
        ).bfloat16()
        data, scales = fp4_quantize(
            raw, one, is_sf_swizzled_layout=False, enable_pdl=False
        )
        data = data.reshape(LOCAL_EXPERTS, rows, cols // 2)
        scales = scales.view(torch.float8_e4m3fn).reshape(
            LOCAL_EXPERTS, rows, cols // 16
        )
        setattr(w, name + "_weight", torch.nn.Parameter(data, requires_grad=False))
        setattr(
            w, name + "_weight_scale", torch.nn.Parameter(scales, requires_grad=False)
        )
        s2_shape = (LOCAL_EXPERTS, 2) if name == "w13" else (LOCAL_EXPERTS,)
        setattr(
            w,
            name + "_weight_scale_2",
            torch.nn.Parameter(
                torch.full(s2_shape, global_scale, device="cuda"), requires_grad=False
            ),
        )
        reference.append((_dequant(data, scales), global_scale))
    w.w13_input_scale = torch.nn.Parameter(
        torch.tensor([0.75], device="cuda"), requires_grad=False
    )
    w.w2_input_scale = torch.nn.Parameter(
        torch.tensor([1.25], device="cuda"), requires_grad=False
    )
    moe_process_weights(plan, w)
    return w, reference


def _quantized(x):
    if isinstance(x, tuple):
        return x
    if x.shape[0]:
        q, sf = fp4_quantize(
            x,
            torch.tensor([1 / 0.75], device="cuda"),
            is_sf_swizzled_layout=False,
            enable_pdl=False,
        )
    else:
        q = torch.empty((0, HIDDEN // 2), dtype=torch.uint8, device="cuda")
        sf = torch.empty((0, HIDDEN // 16), dtype=torch.uint8, device="cuda")
    return q, sf


def _input_tensors(x):
    return x if isinstance(x, tuple) else (x,)


def _inputs(tokens, experts, *, prequantized):
    x = torch.randn(tokens, HIDDEN, dtype=torch.bfloat16, device="cuda")
    ids = (
        torch.arange(TOP_K, device="cuda") * 53
        + torch.arange(tokens, device="cuda")[:, None] * 17
        + dist.get_rank() * 13
    ) % experts
    weights = torch.randn(tokens, TOP_K, device="cuda").softmax(-1).bfloat16()
    return (_quantized(x) if prequantized else x), ids.int(), weights


def _reference(x, ids, weights, reference, capacity, activation):
    x = _quantized(x)
    tensors = [
        torch.nn.functional.pad(t, (0, 0, 0, capacity - t.shape[0]))
        for t in (x[0], x[1].view(torch.uint8), ids, weights)
    ]
    gathered = []
    for local in tensors:
        global_tensor = torch.empty(
            (capacity * dist.get_world_size(), local.shape[1]),
            dtype=local.dtype,
            device="cuda",
        )
        dist.all_gather_single(global_tensor, local.contiguous())
        gathered.append(global_tensor)
    q, sf, routes, scores = gathered
    hidden = _dequant(q, sf.view(torch.float8_e4m3fn))
    output = torch.zeros(hidden.shape, dtype=torch.float32, device="cuda")
    for local_expert in range(LOCAL_EXPERTS):
        rows, choices = torch.where(
            (routes == local_expert + dist.get_rank() * LOCAL_EXPERTS) & (scores != 0)
        )
        if rows.numel() == 0:
            continue
        # Global scales belong to the GEMM epilogues. Folding them into the
        # operands changes FP32 rounding before the intermediate FP4 quantizer.
        w1, scale1 = reference[0]
        w2, scale2 = reference[1]
        gate, up = ((hidden[rows] @ w1[local_expert].T) * (0.75 * scale1)).split(
            INTERMEDIATE, dim=-1
        )
        if activation == "silu":
            activated = torch.nn.functional.silu(gate) * up
        else:
            activated = (
                (4 * torch.tanh(gate / 4))
                * torch.sigmoid(gate)
                * (25 * torch.tanh(up / 25))
            )
        partial = (
            ((_requant(activated, 1.25) / 1.25) @ w2[local_expert].T)
            .mul(1.25 * scale2)
            .bfloat16()
            .float()
        )
        output.index_add_(0, rows, partial * scores[rows, choices].float()[:, None])
    dist.all_reduce(output)
    rank = dist.get_rank()
    return output[rank * capacity : rank * capacity + x[0].shape[0]].bfloat16()


def _check(output, expected, label):
    if output.numel() == 0:
        return
    error = (
        output.float() - expected.float()
    ).norm() / expected.float().norm().clamp_min(1e-6)
    maximum = (output.float() - expected.float()).abs().max()
    print(
        f"rank={dist.get_rank()} {label}: relative_l2={error.item():.6f} max_abs={maximum.item():.6f}",
        flush=True,
    )
    assert torch.isfinite(output).all()
    assert error < 0.005, (label, error.item())


def run_correctness(
    capacity: int, live_tokens: int, tune: bool, activation: str, *, prequantized: bool
):
    torch.manual_seed(713 + dist.get_rank())
    plan = moe_plan(
        "nvfp4",
        input_dtype=torch.bfloat16,
        activation=activation,
        requires_deferred_finalize=False,
        routing_mode="precomputed_topk",
        a2a_backend="none",
        ep_size=dist.get_world_size(),
        ispp=INTERMEDIATE,
        fp8_scale_block_shape=None,
        internal_activation_dtype="input",
        with_bias=False,
        process_group=dist.group.WORLD,
        deepep_mode=None,
        deepep_low_latency_max_num_tokens_per_gpu=None,
        solution="mega_moe",
        hidden=None,
        swiglu_form=None,
        activation_clamped=False,
        expert_id_repeats=False,
        fast_math=True,
        combine_order="rank",
        features=frozenset(),
    )
    w, ref = _weights(plan, activation)
    tokens = live_tokens if dist.get_rank() == 0 else max(1, live_tokens - 2)
    x, ids, weights = _inputs(tokens, w.num_experts, prequantized=prequantized)

    def apply(inputs):
        return moe_apply(
            plan,
            inputs,
            w,
            None,
            topk_weights=weights,
            topk_ids=ids,
            num_tokens_global=capacity * dist.get_world_size(),
            max_num_tokens_per_gpu=capacity,
            do_finalize=True,
            low_latency=None,
            overlap_fn=None,
            shared_input=None,
            shared_weight=None,
            shared_out=None,
        )

    def run():
        return apply(x)

    expected = _reference(x, ids, weights, ref, capacity, activation)
    cached_profiles = set(AutoTuner.get().profiling_cache)
    print(f"rank={dist.get_rank()} launching eager", flush=True)
    guard = (
        nullcontext()
        if tune
        else patch.object(
            AutoTuner.get(),
            "choose_one",
            side_effect=AssertionError("MegaMoE tuning requires explicit opt-in"),
        )
    )
    # Tuning initializes random profiling tensors. Keep those draws from
    # changing the later graph/idle test vectors between tuned and untuned runs.
    with guard, torch.random.fork_rng(devices=[torch.cuda.current_device()]), autotune(
        tune_mode=True, tuning_buckets=None, round_up=None
    ), torch.inference_mode():
        output = run()
    torch.cuda.synchronize()
    _check(output, expected, "eager")
    from tokenspeed_kernel.thirdparty.cute_dsl.mega_moe.runner import get_runner

    runner = (
        get_runner(plan, w) if prequantized else get_runner(plan, w, prequantized=False)
    )
    if prequantized:
        staged_x, staged_sf, staged_ids, staged_weights = runner.input_buffers(
            x[0].shape[0], capacity
        )
        staged_x.copy_(x[0])
        staged_sf[: x[0].shape[0]].copy_(x[1].view(torch.uint8))
        staged_ids.copy_(ids)
        staged_weights.copy_(weights)
        _check(
            runner.run(
                (staged_x, staged_sf),
                staged_ids,
                staged_weights,
                (
                    w.w13_weight,
                    w.w13_weight_scale,
                    w.w2_weight,
                    w.w2_weight_scale,
                    w.mega_fc1_alpha,
                    w.mega_fc2_alpha,
                    w.mega_fc1_norm,
                ),
                capacity,
            ),
            _reference(x, ids, weights, ref, capacity, activation),
            "direct producer input buffers",
        )
    if not prequantized and not tune:
        torch.testing.assert_close(output, apply(_quantized(x)), atol=0, rtol=0)
    if tune:
        tuner = AutoTuner.get()
        tactics = {
            key.nearest_profile: value[0]
            for key, value in tuner.profiling_cache.items()
            if key.custom_op == "trtllm_nvfp4_mega_moe" and key not in cached_profiles
        }
        assert len(tactics) == (capacity - 1).bit_length() + 1
        assert not tuner.stats.failed_tactics.get(
            "trtllm_nvfp4_mega_moe::MegaMoERunner"
        )
        all_tactics = [None] * dist.get_world_size()
        dist.all_gather_object(all_tactics, tactics)
        assert all(t == tactics for t in all_tactics)
    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run()
    replacement, replacement_ids, replacement_weights = _inputs(
        tokens, w.num_experts, prequantized=prequantized
    )
    for dst, src in zip(
        (*_input_tensors(x), ids, weights),
        (*_input_tensors(replacement), replacement_ids, replacement_weights),
    ):
        dst.copy_(src)
    expected = _reference(x, ids, weights, ref, capacity, activation)
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    _check(captured, expected, "graph replay")
    if not prequantized and not tune:
        torch.testing.assert_close(captured, apply(_quantized(x)), atol=0, rtol=0)
    first_graph = (graph, captured, expected, x, ids, weights)
    tokens = 0 if dist.get_rank() == 0 and dist.get_world_size() > 1 else 1
    x, ids, weights = _inputs(tokens, w.num_experts, prequantized=prequantized)
    expected = _reference(x, ids, weights, ref, capacity, activation)
    if tune:
        with patch.object(
            tuner, "_profile_single_kernel", wraps=tuner._profile_single_kernel
        ) as profile:
            with autotune(tune_mode=True, tuning_buckets=None, round_up=None):
                output = run()
            profile.assert_not_called()
    else:
        output = run()
    torch.cuda.synchronize()
    _check(output, expected, "uneven/idle rank")

    arena = runner.workspace
    assert arena.capacity == 1 << (get_autotune_max_num_tokens() - 1).bit_length()
    # Every candidate must be usable without growing the reserved arena,
    # including when the normal execution path has tactic tuning disabled.
    for tactic in runner.TACTICS:
        arena._kernel(arena.capacity, tactic)
    assert "fc2_output_workspace" not in arena.region_sizes["local"]
    with pytest.raises(ValueError, match="not reserved"):
        arena._kernel(4, (256, 512, "static", "reuse_dispatch_warps", 8))
    with pytest.raises(ValueError, match="not reserved"):
        arena._kernel(arena.capacity * 2, runner.TACTICS[0])
    pointers = (
        arena.storage.data_ptr(),
        arena.local.data_ptr(),
        arena.output.data_ptr(),
    )
    maximum = get_autotune_max_num_tokens()
    capacity = capacity // 2 if capacity == maximum else min(maximum, capacity * 2)
    tokens = min(capacity, live_tokens + 1)
    tokens = tokens if dist.get_rank() == 0 else max(1, tokens - 2)
    x, ids, weights = _inputs(tokens, w.num_experts, prequantized=prequantized)
    expected = _reference(x, ids, weights, ref, capacity, activation)
    _check(run(), expected, "alternate capacity eager")
    other_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(other_graph):
        other_output = run()
    second_graph = (other_graph, other_output, expected, x, ids, weights)
    outputs = []
    # Replay different capacities back-to-back on the same arena, without host resets.
    for saved in (first_graph, second_graph, first_graph, second_graph):
        saved[0].replay()
        outputs.append((saved[1].clone(), saved[2]))
    torch.cuda.synchronize()
    for actual, expected in outputs:
        _check(actual, expected, "alternating captured capacities")
    assert arena is runner.workspace
    assert pointers == (
        arena.storage.data_ptr(),
        arena.local.data_ptr(),
        arena.output.data_ptr(),
    )
    dist.barrier()
    if dist.get_rank() == 0:
        print(
            f"PASS NVFP4 {activation} MegaMoE eager, graph replay, idle ranks, and shared-arena capacity switching",
            flush=True,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("tune", [False, True])
@pytest.mark.parametrize("activation", ["situ", "silu"])
@pytest.mark.parametrize("prequantized", [False, True])
def test_nvfp4_megamoe(tune, activation, prequantized, monkeypatch):
    if tune:
        monkeypatch.setenv("MEGAMOE_TACTIC_AUTOTUNE", "1")
    else:
        monkeypatch.delenv("MEGAMOE_TACTIC_AUTOTUNE", raising=False)
    if not (10, 0) <= torch.cuda.get_device_capability() <= (10, 3):
        pytest.skip("requires SM100/SM103")
    with tempfile.TemporaryDirectory() as tmp:
        dist.init_process_group(
            "nccl",
            init_method=f"file://{tmp}/rendezvous",
            rank=0,
            world_size=1,
            timeout=timedelta(minutes=10),
            device_id=torch.cuda.current_device(),
        )
        try:
            from tokenspeed_kernel.thirdparty.cute_dsl.mega_moe.runner import (
                MegaMoERunner,
            )

            # Cover both tile widths and schedulers; torchrun uses the full sweep.
            with patch.object(
                MegaMoERunner,
                "TACTICS",
                (
                    (128, 512, "static", "epi_warps", 1),
                    (256, 1024, "atomic_counter", "epi_warps", 8),
                ),
            ):
                set_autotune_process_group(dist.group.WORLD)
                run_correctness(4, 4, tune, activation, prequantized=prequantized)
                if not tune and activation == "silu":
                    # Larger live batches, then idle ranks and alternating
                    # graphs that retain the full-capacity arena.
                    run_correctness(
                        512, 300, False, activation, prequantized=prequantized
                    )
        finally:
            set_autotune_process_group(None)
            dist.destroy_process_group()


if __name__ == "__main__":
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group(
        "nccl", timeout=timedelta(minutes=10), device_id=torch.cuda.current_device()
    )
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument("--capacity", type=int, default=4)
        parser.add_argument("--tokens", type=int, default=4)
        parser.add_argument("--autotune", action="store_true")
        parser.add_argument("--activation", choices=("situ", "silu"), required=True)
        parser.add_argument("--input-format", choices=("bf16", "nvfp4"), required=True)
        args = parser.parse_args()
        os.environ["MEGAMOE_TACTIC_AUTOTUNE"] = "1" if args.autotune else "0"
        set_autotune_max_num_tokens(args.capacity)
        set_autotune_process_group(dist.new_group(backend="gloo"))
        run_correctness(
            args.capacity,
            args.tokens,
            args.autotune,
            args.activation,
            prequantized=args.input_format == "nvfp4",
        )
    finally:
        set_autotune_process_group(None)
        dist.destroy_process_group()
