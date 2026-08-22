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

"""Offline tactic sweeper for the flashinfer TRTLLM-Gen SiTU MoE.

Why this exists: flashinfer's autotuner mispicks tactics for this op on two
counts -- its per-tactic profiling measurements do not track end-to-end kernel
time, and ``trtllm_get_valid_moe_configs`` gates tile_N eligibility on total
``num_tokens`` where the batched-GEMM workload actually depends on
tokens-per-expert (``num_tokens * top_k / num_experts``). For many-expert EP
deployments (e.g. Kimi-K3: 896 experts, EP8) both bites: at 2048 tokens the
enumeration offers only fat tiles and the tuner then picks the worst of them,
while the excluded narrow tiles measure fastest end to end.

This tool sweeps candidate tactics with real end-to-end CUDA-event timing on
representative random weights, writes the winners into flashinfer's autotuner,
and saves the result via ``AutoTuner.save_configs`` -- a JSON table whose
embedded metadata pins the GPU device name and FlashInfer/CUDA/cuDNN versions,
so a table can never be applied on a mismatched host. Name it vLLM-configs style
and ship it under ``ops/moe/flashinfer/tactics/`` to have it auto-load at
startup (see ops.tuning). By default the output filename is generated from the
model/layout and current GPU, FlashInfer, and cuDNN versions. Pass ``--output``
to choose the path explicitly.

Cost is bounded by a two-stage search: measured spreads within one tile_N
family are <3%, so stage one samples a few configs per family to pick the
family and stage two refines within it. A full run at Kimi-K3 shapes is a few
minutes on one idle GPU. Re-run whenever the FlashInfer or cuDNN version, GPU
model, or MoE shape changes (the metadata guard turns a stale table into a
logged fallback, never silent misuse).

Examples (Kimi-K3 on EP8, defaults):

    # Generate the environment-specific filename in the current directory.
    python -m tokenspeed_kernel.benchmark.moe_tactic_sweep

    # Or select the output path explicitly.
    python -m tokenspeed_kernel.benchmark.moe_tactic_sweep \\
        --output moe-tactics-kimi-k3-ep8.json
"""

from __future__ import annotations

import argparse
import sys
from importlib.metadata import version

import torch
from tokenspeed_kernel.ops.tuning import (
    flashinfer_tuning_cache_filename,
    get_autotune_max_num_tokens,
)

# Must match the tune_max the runtime keys its bucket ladder with: the
# runtime floors chunked_prefill_size at the ops.tuning default (8192), so
# the default sweep matches every deployment with chunked prefill <= 8192.
# Resweep with a matching value for larger prefill configurations.
SITU_TUNE_MAX_NUM_TOKENS = get_autotune_max_num_tokens()

SF_BLOCK = 32
# Buckets below this are left on the autotuner's own choice: the enumeration
# and profiling problems only bite once tokens-per-expert crosses tile
# boundaries, and measured decode-range picks were already optimal.
DEFAULT_SWEEP_MIN_BUCKET = 128
STAGE1_CONFIGS_PER_FAMILY = 3
STAGE2_MAX_CONFIGS = 64


def _make_weights(local_experts: int, hidden: int, ispp: int, device, seed: int):
    """Random weights in the prepared TRTLLM layout (perf-representative).

    Scale bytes are pinned to 127 (2^0) so garbage exponents cannot produce
    inf/nan; tactic ranking depends on shapes and routing, not weight values.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    w13 = torch.randint(
        0, 256, (local_experts, 2 * ispp, hidden // 2), generator=g, dtype=torch.uint8
    ).to(device)
    w13_scale = torch.full(
        (local_experts, 2 * ispp, hidden // SF_BLOCK),
        127,
        dtype=torch.uint8,
        device=device,
    ).view(torch.float8_e4m3fn)
    w2 = torch.randint(
        0, 256, (local_experts, hidden, ispp // 2), generator=g, dtype=torch.uint8
    ).to(device)
    w2_scale = torch.full(
        (local_experts, hidden, ispp // SF_BLOCK),
        127,
        dtype=torch.uint8,
        device=device,
    ).view(torch.float8_e4m3fn)
    return w13, w13_scale, w2, w2_scale


def _make_tokens(
    num_tokens: int, hidden: int, num_experts: int, top_k: int, device, seed: int
):
    from flashinfer import mxfp8_quantize

    g = torch.Generator(device="cpu").manual_seed(seed)
    x = (torch.randn(num_tokens, hidden, generator=g) * 0.05).bfloat16().to(device)
    topk_ids = torch.stack(
        [torch.randperm(num_experts, generator=g)[:top_k] for _ in range(num_tokens)]
    ).to(device=device, dtype=torch.int32)
    topk_weights = (
        torch.rand(num_tokens, top_k, generator=g).softmax(-1).bfloat16().to(device)
    )
    x_q, x_scale = mxfp8_quantize(x, False, alignment=hidden)
    x_scale = x_scale.view(torch.float8_e4m3fn).reshape(num_tokens, -1)
    return x_q, x_scale, topk_ids, topk_weights


def _time_call(fn, iters: int, warmup: int = 3) -> float:
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start.record()
    for _ in range(iters):
        fn()
    stop.record()
    torch.cuda.synchronize()
    return start.elapsed_time(stop) * 1000.0 / iters  # us


def _candidate_tactics(args) -> list[tuple[int, int]]:
    """All (tile_N, config) pairs across every tile family for this shape.

    Enumerated at several ``num_tokens`` values and unioned, because the
    upstream enumeration gates tile_N availability on num_tokens -- the whole
    reason narrow tiles need this sweeper to be reachable at large batches.
    """
    from flashinfer.fused_moe.core import gen_trtllm_gen_fused_moe_sm100_module
    from flashinfer.tllm_enums import (
        ActivationType,
        DtypeTrtllmGen,
        Fp8QuantizationType,
        WeightLayout,
    )

    moe_op = gen_trtllm_gen_fused_moe_sm100_module().build_and_load()
    seen: dict[tuple[int, int], None] = {}
    for probe_tokens in (64, 256, 512, SITU_TUNE_MAX_NUM_TOKENS):
        for tac in moe_op.trtllm_get_valid_moe_configs(
            DtypeTrtllmGen.MxE4m3,
            DtypeTrtllmGen.MxE2m1,
            Fp8QuantizationType.NoneFp8,
            args.top_k,
            args.hidden_size,
            args.intermediate_size,
            args.local_experts,
            ActivationType.Situ.value,
            True,  # use_shuffled_weight
            WeightLayout.MajorK.value,
            False,  # use_per_token_scaling
            probe_tokens,
            False,  # has_gemm1_lora_delta
        ):
            seen[(int(tac[0]), int(tac[1]))] = None
    return list(seen)


def _run(args, W, tokens, tactic_setter, tactic, iters, do_finalize=True):
    from flashinfer.fused_moe import trtllm_fp4_block_scale_routed_moe
    from flashinfer.tllm_enums import ActivationType

    x_q, x_scale, topk_ids, topk_weights = tokens
    w13, w13_scale, w2, w2_scale = W
    # The cache key carries the output shape, so the modes need separate sweeps.
    output = (
        torch.zeros(
            x_q.shape[0], args.hidden_size, dtype=torch.bfloat16, device=x_q.device
        )
        if do_finalize
        else None
    )
    alpha = torch.full(
        (args.local_experts,), args.situ_alpha, dtype=torch.float32, device=x_q.device
    )
    beta = torch.full(
        (args.local_experts,), args.situ_beta, dtype=torch.float32, device=x_q.device
    )

    def call():
        trtllm_fp4_block_scale_routed_moe(
            topk_ids=(topk_ids, topk_weights),
            routing_bias=None,
            hidden_states=x_q,
            hidden_states_scale=x_scale,
            gemm1_weights=w13,
            gemm1_weights_scale=w13_scale,
            gemm1_bias=None,
            gemm1_alpha=alpha,
            gemm1_beta=beta,
            gemm1_clamp_limit=None,
            gemm2_weights=w2,
            gemm2_weights_scale=w2_scale,
            gemm2_bias=None,
            output1_scale_scalar=None,
            output1_scale_gate_scalar=None,
            output2_scale_scalar=None,
            num_experts=args.num_experts,
            top_k=args.top_k,
            n_group=None,
            topk_group=None,
            intermediate_size=args.intermediate_size,
            local_expert_offset=0,
            local_num_experts=args.local_experts,
            routed_scaling_factor=None,
            routing_method_type=1,
            do_finalize=do_finalize,
            enable_pdl=False,
            activation_type=10,  # ActivationType.Situ; probed at startup
            tune_max_num_tokens=SITU_TUNE_MAX_NUM_TOKENS,
            output=output,
        )

    assert ActivationType.Situ.value == 10
    tactic_setter(tactic)
    return _time_call(call, iters=iters)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
    )
    parser.add_argument(
        "--output",
        help=(
            "Output JSON table path; defaults to an environment-specific "
            "filename in the current directory"
        ),
    )
    parser.add_argument(
        "--model",
        default="kimi-k3",
        help="Model slug used in an automatically generated output filename",
    )
    parser.add_argument(
        "--ep-size",
        type=int,
        default=8,
        help="Expert-parallel size used in an automatically generated filename",
    )
    parser.add_argument(
        "--tp-size",
        type=int,
        default=1,
        help="MoE tensor-parallel size used in an automatically generated filename",
    )
    parser.add_argument("--num-experts", type=int, default=896)
    parser.add_argument("--local-experts", type=int, default=112)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=3584,
        help="Expert hidden size (Kimi-K3: routed_expert_hidden_size)",
    )
    parser.add_argument("--intermediate-size", type=int, default=3072)
    parser.add_argument("--situ-alpha", type=float, default=4.0)
    parser.add_argument("--situ-beta", type=float, default=25.0)
    parser.add_argument(
        "--min-bucket",
        type=int,
        default=DEFAULT_SWEEP_MIN_BUCKET,
        help="Sweep buckets >= this; smaller buckets keep the autotuner's pick",
    )
    parser.add_argument(
        "--finalize-modes",
        default="both",
        choices=("both", "finalize", "deferred"),
        help=(
            "Which finalize modes to sweep. The autotuner keys on the output "
            "tensor's shape, so a table swept only with do_finalize=True "
            "misses every deferred-finalize call (K3 decode, where the MoE "
            "tail owns finalize)"
        ),
    )
    parser.add_argument("--coarse-iters", type=int, default=8)
    parser.add_argument("--fine-iters", type=int, default=50)
    args = parser.parse_args(argv)

    from flashinfer.autotuner import AutoTuner, autotune

    device = torch.device("cuda")
    output = args.output
    if output is None:
        cudnn_version = torch.backends.cudnn.version()
        if cudnn_version is None:
            parser.error("cannot generate an output filename without a cuDNN version")
        output = flashinfer_tuning_cache_filename(
            args.model,
            args.ep_size,
            args.tp_size,
            torch.cuda.get_device_name(),
            version("flashinfer-python"),
            cudnn_version,
        )
        print(f"generated output path: {output}")
    W = _make_weights(
        args.local_experts, args.hidden_size, args.intermediate_size, device, seed=1
    )

    modes = (
        (True, False)
        if args.finalize_modes == "both"
        else (True,) if args.finalize_modes == "finalize" else (False,)
    )
    tuner = AutoTuner.get()
    candidates = _candidate_tactics(args)
    families = sorted({t for t, _ in candidates})
    print(f"candidate tactics: {len(candidates)} across tile_N families {families}")

    for do_finalize in modes:
        label = "finalize" if do_finalize else "deferred"
        print(f"--- sweeping do_finalize={do_finalize} ({label}) ---")
        seen = set(tuner.profiling_cache)

        # One native pass materializes every bucket under the runtime's key layout.
        tokens_max = _make_tokens(
            SITU_TUNE_MAX_NUM_TOKENS,
            args.hidden_size,
            args.num_experts,
            args.top_k,
            device,
            seed=2,
        )
        with autotune():
            _run(
                args,
                W,
                tokens_max,
                lambda _: None,
                None,
                iters=1,
                do_finalize=do_finalize,
            )

        bucket_keys = {}
        for key, (_tac, profile) in tuner.profiling_cache.items():
            if profile is None or key in seen:
                continue
            bucket_keys[int(profile.get_opt_shapes()[0][0])] = (key, profile)
        print(f"materialized buckets ({label}): {sorted(bucket_keys)}")

        for bucket in sorted(b for b in bucket_keys if b >= args.min_bucket):
            key, profile = bucket_keys[bucket]
            native_tactic = tuner.profiling_cache[key][0]

            def set_tactic(tac, key=key, profile=profile):
                if tac is not None:
                    tuner.profiling_cache[key] = (tuple(tac), profile)

            tokens = _make_tokens(
                bucket,
                args.hidden_size,
                args.num_experts,
                args.top_k,
                device,
                seed=100 + bucket,
            )

            def run(tac, iters):
                return _run(
                    args, W, tokens, set_tactic, tac, iters, do_finalize=do_finalize
                )

            # Stage 1 picks the tile_N family, stage 2 refines within it.
            stage1: list[tuple[float, tuple[int, int]]] = []
            for family in families:
                for tac in [t for t in candidates if t[0] == family][
                    :STAGE1_CONFIGS_PER_FAMILY
                ]:
                    stage1.append((run(tac, args.coarse_iters), tac))
            best_family = min(stage1)[1][0]

            stage2 = [
                (run(tac, args.coarse_iters), tac)
                for tac in [t for t in candidates if t[0] == best_family][
                    :STAGE2_MAX_CONFIGS
                ]
            ]
            finalists = sorted(stage2)[:3] + [min(stage1)]
            timed = [(run(tac, args.fine_iters), tac) for _, tac in finalists]
            native_time = run(native_tactic, args.fine_iters)
            best_time, best_tactic = min(timed)
            if native_time <= best_time:
                best_time, best_tactic = native_time, tuple(native_tactic)
            set_tactic(best_tactic)
            print(
                f"bucket {bucket:>5} ({label}): tactic {best_tactic} "
                f"{best_time:7.1f}us (native {tuple(native_tactic)} "
                f"{native_time:7.1f}us)"
            )

    tuner.save_configs(output)
    # Round-trip guard: a table this process cannot re-load would be useless
    # at serving time; fail loudly here rather than at the first deploy.
    tuner.profiling_cache.clear()
    tuner._file_configs.clear()
    if not tuner.load_configs(output):
        print("ERROR: saved table failed to reload in-process", file=sys.stderr)
        return 1
    print(f"saved + reload-verified: {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
