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

"""Paired source-method DSpark consumer benchmark; this is not model E2E.

Example shape syntax: B:N:V:H:R, where N includes the anchor. Both arms use
independent equal buffers, weights and CUDA graphs. A comes from original
fixed source files, not the candidate fallback. P must execute fused stages.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import time
import traceback
from functools import partial
from pathlib import Path

import torch
from dspark_bias_argmax_support import (
    BASELINE_SHA,
    copy_consumer_parameters,
    digest,
    load_consumers,
    make_consumer,
)
from tokenspeed_kernel._triton import triton
from tokenspeed_kernel.ops.sampling import bias_argmax as bias_ops
from tokenspeed_kernel.ops.sampling import create_bias_argmax_workspace, try_bias_argmax
from tokenspeed_kernel.ops.sampling.triton import bias_argmax as bias_kernels


def capture(fn, warmup):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(warmup):
            fn()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    return graph


def measure(fn, iterations):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / iterations


def path_evidence(fn):
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as profile:
        fn()
        torch.cuda.synchronize()
    return [
        event.name for event in profile.events() if event.device_type.name == "CUDA"
    ]


def _constant_bias(bias, start, count):
    assert start == 0 and count == bias.shape[1]
    return bias


def tensor_evidence(tensor):
    # Hash logical bytes outside timing; dtype byte-view supports bf16 too.
    byte_view = tensor.detach().contiguous().cpu().view(torch.uint8)
    return {
        "sha256": hashlib.sha256(memoryview(byte_view.numpy())).hexdigest(),
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "data_ptr": tensor.data_ptr(),
    }


def immutable_evidence(tensors):
    return {
        arm: {name: tensor_evidence(tensor) for name, tensor in named.items()}
        for arm, named in tensors.items()
    }


def make_workload(classes, shape, dtype, group):
    rows, steps, vocab, hidden, rank = map(int, shape.split(":"))
    assert rows > 0 and steps >= 2 and 0 < vocab <= 262144 and hidden > 0 and rank > 0
    consumers = {
        arm: make_consumer(
            classes, arm, rows, steps, vocab, hidden, rank, dtype, "cuda"
        )
        for arm in ("A", "P")
    }
    copy_consumer_parameters(consumers["A"], consumers["P"])
    hidden_a = torch.randn((rows, steps - 1, hidden), dtype=dtype, device="cuda")
    anchor_a = torch.randint(vocab, (rows, steps), dtype=torch.int32, device="cuda")
    inputs = {arm: (hidden_a.clone(), anchor_a.clone()) for arm in ("A", "P")}
    outputs = {
        arm: torch.full((rows, steps), -37, dtype=torch.int32, device="cuda")
        for arm in ("A", "P")
    }
    calls = {}
    retained = [consumers, inputs, outputs]
    if group == "proposal_walk":
        for arm in ("A", "P"):
            calls[arm] = partial(
                consumers[arm]._sample_block,
                inputs[arm][0],
                inputs[arm][1],
                outputs[arm],
            )
    else:
        logits_a = consumers["A"]._block_base_logits(inputs["A"][0])[0]
        bias_a = consumers["A"]._make_step_bias_fn(inputs["A"][1][:, 0])(0, vocab)
        logits = {arm: logits_a.clone() for arm in ("A", "P")}
        biases = {arm: bias_a.clone() for arm in ("A", "P")}
        retained.extend([logits, biases])
        for arm in ("A", "P"):
            calls[arm] = partial(
                consumers[arm]._greedy_argmax_vocab_parallel,
                inputs[arm][0][:, 0],
                out=outputs[arm][:, 1],
                bias_fn=partial(_constant_bias, biases[arm]),
                base_logits=logits[arm],
            )
    immutable = {
        arm: {
            "hidden": inputs[arm][0],
            "anchor": inputs[arm][1],
            "lm_head": consumers[arm].lm_head.weight,
            "markov_w1": consumers[arm].markov_head.markov_w1.weight,
            "markov_w2": consumers[arm].markov_head.markov_w2.weight,
        }
        for arm in ("A", "P")
    }
    if group == "postprocess_precomputed_bias":
        for arm in ("A", "P"):
            immutable[arm].update(base_logits=logits[arm], bias=biases[arm])
    evidence = immutable_evidence(immutable)
    for name in immutable["A"]:
        assert evidence["A"][name]["sha256"] == evidence["P"][name]["sha256"]
        assert evidence["A"][name]["data_ptr"] != evidence["P"][name]["data_ptr"]
    for call in calls.values():
        call()
    torch.testing.assert_close(outputs["A"], outputs["P"], rtol=0, atol=0)
    pointers = {
        arm: {
            "out": outputs[arm].data_ptr(),
            "hidden": inputs[arm][0].data_ptr(),
            "weight": consumers[arm].lm_head.weight.data_ptr(),
            "markov_w1": consumers[arm].markov_head.markov_w1.weight.data_ptr(),
            "markov_w2": consumers[arm].markov_head.markov_w2.weight.data_ptr(),
        }
        for arm in ("A", "P")
    }
    assert all(pointers["A"][key] != pointers["P"][key] for key in pointers["A"])
    return calls, outputs, retained, pointers, immutable, evidence


def run(args, record, counts):
    assert args.blocks > 0 and args.iterations > 0 and args.warmup > 0
    assert torch.cuda.get_device_capability() == (12, 0)
    torch.set_grad_enabled(False)
    torch.manual_seed(20260912)
    rng = random.Random(20260912)
    repository = Path(__file__).resolve().parents[2]
    classes, provenance = load_consumers(
        repository, args.baseline_dir, create_bias_argmax_workspace, try_bias_argmax
    )
    groups = ("postprocess_precomputed_bias", "proposal_walk")
    expected_arms = (
        len(args.shapes) * len(args.dtypes) * len(groups) * 2 * args.blocks * 4
    )
    expected_paths = len(args.shapes) * len(args.dtypes) * len(groups) * 2
    sources = [
        Path(__file__),
        Path(__file__).with_name("dspark_bias_argmax_support.py"),
        Path(bias_ops.__file__).with_name("__init__.py"),
        Path(bias_ops.__file__),
        Path(bias_kernels.__file__),
    ]
    properties = torch.cuda.get_device_properties(torch.cuda.current_device())
    selected_uuid = getattr(properties, "uuid", None)
    assert selected_uuid is not None, "selected GPU UUID unavailable"
    metadata = {
        "kind": "metadata",
        "pid": os.getpid(),
        "started_at": time.time(),
        "torch": torch.__version__,
        "triton": triton.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "selected_gpu_uuid": str(selected_uuid),
        "current_device": torch.cuda.current_device(),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "capability": torch.cuda.get_device_capability(),
        "driver": subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True,
        ).strip(),
        "baseline_sha": BASELINE_SHA,
        "consumer_sources": provenance,
        "source_sha256": {str(path.resolve()): digest(path) for path in sources},
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "expected_arm_records": expected_arms,
        "expected_path_records": expected_paths,
        "scope": "source-identical methods with TP1 head/comm/initialization fixtures; NVTX disabled via original wrapper; no model E2E",
        "timed_regions": {
            "postprocess_precomputed_bias": "actual DFlash method including metadata, once constant bias closure, native cast/add/max/offset/copy or fused selection and both launches; base and bias GEMMs excluded in both arms",
            "proposal_walk": "actual DSpark _sample_block including anchor copy, complete block base GEMM/layout, prev-token clamp/embedding, each Markov GEMM and DFlash method, final clamp; N includes anchor",
        },
        "immutable_contract": "Each arm hashes hidden/anchor/LM head/Markov parameters before invocation and after timing; precomputed-bias group also hashes base_logits and bias. Proposal walk computes base_logits and bias as ephemeral in-region intermediates, not persistent input buffers. Only output/workspace are mutable persistent tensors.",
        "statistics": "100*(1-exp(mean(block_log_ratio))); each block is mean(log P1,log P2)-mean(log A1,log A2); each run/shape/dtype/group/mode kept separate",
    }
    record(metadata)
    for shape in args.shapes:
        for dtype_name in args.dtypes:
            for group in groups:
                calls, outputs, retained, pointers, immutable, evidence = make_workload(
                    classes, shape, getattr(torch, dtype_name), group
                )
                context = {"shape": shape, "dtype": dtype_name, "group": group}
                record({"kind": "immutable_inputs", **context, "before": evidence})
                for arm, fn in calls.items():
                    kernels = path_evidence(fn)
                    for stage in ("_bias_argmax_partials", "_bias_argmax_finalize"):
                        hits = sum(stage in event for event in kernels)
                        expected = (
                            (
                                int(shape.split(":")[1]) - 1
                                if group == "proposal_walk"
                                else 1
                            )
                            if arm == "P"
                            else 0
                        )
                        assert hits == expected, (
                            arm,
                            stage,
                            hits,
                            expected,
                            kernels,
                        )
                    record(
                        {
                            "kind": "path",
                            **context,
                            "arm": arm,
                            "kernels": kernels,
                            "pointers": pointers[arm],
                        }
                    )
                graphs = {arm: capture(fn, args.warmup) for arm, fn in calls.items()}
                for mode in ("eager", "cuda_graph"):
                    timed = (
                        calls
                        if mode == "eager"
                        else {arm: graph.replay for arm, graph in graphs.items()}
                    )
                    for fn in timed.values():
                        measure(fn, args.iterations)
                    for block in range(args.blocks):
                        order = "APPA" if rng.randrange(2) == 0 else "PAAP"
                        for position, arm in enumerate(order):
                            latency = measure(timed[arm], args.iterations)
                            record(
                                {
                                    "kind": "arm",
                                    **context,
                                    "mode": mode,
                                    "block": block,
                                    "position": position,
                                    "arm": arm,
                                    "latency_us": latency,
                                    "iterations": args.iterations,
                                }
                            )
                    torch.testing.assert_close(
                        outputs["A"], outputs["P"], rtol=0, atol=0
                    )
                after = immutable_evidence(immutable)
                unchanged = after == evidence
                record(
                    {
                        "kind": "immutable_validation",
                        **context,
                        "after": after,
                        "unchanged": unchanged,
                    }
                )
                assert unchanged, "immutable input/weight changed during run"
                del calls, outputs, retained, graphs, immutable
    source_after = {str(path.resolve()): digest(path) for path in sources}
    consumer_after = {
        arm: {
            name: {"path": entry["path"], "sha256": digest(entry["path"])}
            for name, entry in entries.items()
        }
        for arm, entries in provenance.items()
    }
    sources_unchanged = (
        source_after == metadata["source_sha256"] and consumer_after == provenance
    )
    record(
        {
            "kind": "source_validation",
            "source_sha256": source_after,
            "consumer_sources": consumer_after,
            "unchanged": sources_unchanged,
        }
    )
    assert sources_unchanged, "source changed during run"
    assert counts == {"arm": expected_arms, "path": expected_paths}
    record(
        {
            "kind": "completion",
            "success": True,
            "completed_at": time.time(),
            "arm_records": counts["arm"],
            "path_records": counts["path"],
        }
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--blocks", type=int, required=True)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--warmup", type=int, required=True)
    parser.add_argument("--shapes", nargs="+", required=True)
    parser.add_argument(
        "--dtypes", nargs="+", choices=["float16", "bfloat16", "float32"], required=True
    )
    args = parser.parse_args()
    # Exclusive creation rejects previous/partial evidence before GPU work.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        counts = {"arm": 0, "path": 0}

        def record(row):
            if row["kind"] in counts:
                counts[row["kind"]] += 1
            line = json.dumps(row)
            output.write(line + "\n")
            output.flush()
            print(line, flush=True)

        record(
            {
                "kind": "run_start",
                "pid": os.getpid(),
                "started_at": time.time(),
                "output": str(args.output),
            }
        )
        try:
            run(args, record, counts)
        except BaseException as error:
            record(
                {
                    "kind": "completion",
                    "success": False,
                    "completed_at": time.time(),
                    "arm_records": counts["arm"],
                    "path_records": counts["path"],
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                }
            )
            raise


if __name__ == "__main__":
    main()
