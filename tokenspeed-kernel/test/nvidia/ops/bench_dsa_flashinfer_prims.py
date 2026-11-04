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

"""Compare registered Prims-TS and TRTLLM DSA with identical FP8 inputs.

Run on an isolated SM100/SM103 GPU. Times include the public adapter's GPU
work, including TRTLLM head padding. JIT and plan construction are excluded.
Cold-L2 samples flush 512 MiB before each timed call. CUDA events are captured
inside the graph so host submission and flush time are excluded.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import statistics
from pathlib import Path

import torch
from tokenspeed_kernel.ops.attention.dsa import dsa_decode, dsa_prefill
from tokenspeed_kernel.platform import pdl_enabled
from tokenspeed_kernel.workspace import capture_session


def _inputs(batch, width, heads, context, *, mode):
    rows = batch * width
    q = (torch.randn(rows, heads, 576, device="cuda") * 0.5).to(torch.float8_e4m3fn)
    kv = (
        torch.randn(batch * context, 576, device="cuda", dtype=torch.bfloat16) * 0.25
    ).to(q.dtype)
    base = torch.stack(
        [torch.randperm(context, device="cuda")[:2048] for _ in range(batch)]
    )
    offsets = torch.arange(width, device="cuda").view(1, width, 1) * 97
    slots = (base.unsqueeze(1) + offsets) % context
    slots += torch.arange(batch, device="cuda").view(batch, 1, 1) * context
    slots = slots.reshape(rows, 2048).int()
    lengths = torch.full((rows,), 2048, device="cuda", dtype=torch.int32)
    return q, kv, slots, lengths


def _call(q, kv, slots, lengths, out, *, case, solution):
    mode = case["mode"]
    kwargs = {"q_len_per_req": case["width"]} if mode == "decode" else {}
    return (dsa_decode if mode == "decode" else dsa_prefill)(
        q=q,
        kv_cache=kv,
        sparse_kv_cache=None,
        topk_slots=slots,
        topk_lens=lengths,
        max_seqlen_k=case["context"],
        qk_nope_head_dim=128,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        softmax_scale=192**-0.5,
        page_size=64,
        logit_cap=0.0,
        k_scale=1.0,
        return_lse=False,
        out=out,
        solution=solution,
        slot_order="selection",
        **kwargs,
    )


def _measure(fn, *, stream, cold):
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    flush = (
        torch.empty(512 * 1024 * 1024, dtype=torch.uint8, device="cuda")
        if cold
        else None
    )
    repeats = 1 if cold else 16
    starts = [torch.cuda.Event(enable_timing=True, external=True) for _ in range(12)]
    ends = [torch.cuda.Event(enable_timing=True, external=True) for _ in starts]
    with capture_session(graph), torch.cuda.graph(graph, stream=stream):
        for start, end in zip(starts, ends):
            if flush is not None:
                flush.zero_()
            start.record()
            for _ in range(repeats):
                fn()
            end.record()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    samples = []
    for _ in range(5):
        graph.replay()
        torch.cuda.synchronize()
        samples.extend(
            start.elapsed_time(end) * 1000 / repeats for start, end in zip(starts, ends)
        )
    return {
        "median_us": statistics.median(samples),
        "p10_us": sorted(samples)[len(samples) // 10],
        "p90_us": sorted(samples)[9 * len(samples) // 10],
        "samples_us": samples,
    }


def _metrics(actual, expected):
    a, b = actual.float().flatten(), expected.float().flatten()
    return {
        "max_abs": (a - b).abs().max().item(),
        "relative_rms": ((a - b).norm() / b.norm().clamp_min(1e-20)).item(),
        "cosine": torch.nn.functional.cosine_similarity(a, b, dim=0).item(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--heads", type=int, nargs="+", default=[12, 96])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(7984)
    torch.backends.cuda.matmul.allow_tf32 = False
    pdl_enabled(True)
    environment = {
        "gpu": torch.cuda.get_device_name(),
        "capability": torch.cuda.get_device_capability(),
        "sm_count": torch.cuda.get_device_properties(0).multi_processor_count,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "flashinfer": importlib.metadata.version("flashinfer-python"),
        "cutlass_dsl": importlib.metadata.version("nvidia-cutlass-dsl"),
        "timing": "CUDA graph with internal external CUDA events; adapter GPU work included",
        "cold_l2_flush_bytes": 512 * 1024 * 1024,
        "topk": 2048,
        "q_dtype": "E4M3",
        "kv_dtype": "E4M3",
        "pdl": True,
    }
    (args.output / "environment.json").write_text(json.dumps(environment, indent=2))
    print(json.dumps(environment), flush=True)
    cases = []
    for heads in args.heads:
        for context in ([8192] if args.quick else [8192, 131072]):
            for batch in ([1, 32] if args.quick else [1, 8, 32, 64]):
                for width in ([1, 6] if args.quick else [1, 6, 8]):
                    cases.append(
                        dict(
                            mode="decode",
                            heads=heads,
                            batch=batch,
                            width=width,
                            context=context,
                        )
                    )
        if not args.quick:
            for rows in (128, 2048, 8192):
                cases.append(
                    dict(
                        mode="prefill", heads=heads, batch=1, width=rows, context=131072
                    )
                )
    stream = torch.cuda.Stream()
    results = []
    for number, case in enumerate(cases):
        print(json.dumps({"starting": case}), flush=True)
        q, kv, slots, lengths = _inputs(
            case["batch"],
            case["width"],
            case["heads"],
            case["context"],
            mode=case["mode"],
        )
        outputs = {
            s: torch.empty(
                q.shape[0], q.shape[1], 512, device="cuda", dtype=torch.bfloat16
            )
            for s in ("flashinfer_trtllm", "flashinfer_prims")
        }
        functions = {
            s: (
                lambda s=s: _call(
                    q, kv, slots, lengths, outputs[s], case=case, solution=s
                )
            )
            for s in outputs
        }
        for fn in functions.values():
            fn()
        torch.cuda.synchronize()
        sample_rows = sorted({0, q.shape[0] // 2, q.shape[0] - 1})
        references = []
        for row in sample_rows:
            selected = kv.index_select(0, slots[row].long()).float()
            logits = q[row].float() @ selected.T * (192**-0.5)
            references.append(logits.softmax(-1) @ selected[:, :512])
        reference = torch.stack(references)
        result = {
            **case,
            "topk": 2048,
            "accuracy": {
                s: _metrics(out[sample_rows], reference) for s, out in outputs.items()
            },
            "prims_vs_trtllm": _metrics(
                outputs["flashinfer_prims"], outputs["flashinfer_trtllm"]
            ),
            "timings": {},
        }
        for s, out in outputs.items():
            if not torch.isfinite(out).all():
                raise AssertionError(f"{s} produced non-finite output for {case}")
            torch.testing.assert_close(
                out[sample_rows].float(), reference, rtol=2e-2, atol=1e-3
            )
        order = list(outputs)
        if number % 2:
            order.reverse()
        for cold in (False, True):
            cache = "cold_l2" if cold else "hot_l2"
            timing = {
                s: _measure(functions[s], stream=stream, cold=cold) for s in order
            }
            timing["trtllm_over_prims"] = (
                timing["flashinfer_trtllm"]["median_us"]
                / timing["flashinfer_prims"]["median_us"]
            )
            result["timings"][cache] = timing
        if (
            case["mode"] == "decode"
            and case["batch"] == 32
            and case["width"] == 6
            and case["context"] == 8192
        ):
            names = {}
            for s, fn in functions.items():
                with torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ]
                ) as profile:
                    fn()
                    torch.cuda.synchronize()
                names[s] = sorted(
                    {
                        e.name
                        for e in profile.events()
                        if e.device_type == torch.autograd.DeviceType.CUDA
                    }
                )
                profile.export_chrome_trace(
                    str(args.output / f"profile-h{case['heads']}-{s}.json")
                )
            result["cuda_kernels"] = names
        results.append(result)
        (args.output / "results.json").write_text(json.dumps(results, indent=2))
        compact = {
            **case,
            "accuracy": result["accuracy"],
            "timings": {
                cache: {
                    s: stats["median_us"] if isinstance(stats, dict) else stats
                    for s, stats in timing.items()
                }
                for cache, timing in result["timings"].items()
            },
        }
        print(json.dumps(compact), flush=True)
        del q, kv, slots, lengths, outputs, functions
    print(json.dumps({"completed_cases": len(results)}), flush=True)


if __name__ == "__main__":
    main()
