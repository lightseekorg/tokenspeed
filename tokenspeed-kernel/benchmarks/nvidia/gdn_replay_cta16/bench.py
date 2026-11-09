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

"""Compare saved replay implementations with the same 16-CTA grid."""

import argparse
import hashlib
import importlib
import importlib.util
import json
import random
import statistics
import sys
import traceback
from pathlib import Path

import torch
from tokenspeed_kernel.platform import pdl_enabled


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def inputs(batch, layers, pattern):
    torch.manual_seed(731)
    h, hv, k, v, t = 4, 12, 128, 128, 4
    payload = torch.randn(
        layers, batch * t, h * k + hv * v + 2 * hv, device="cuda", dtype=torch.bfloat16
    )
    parameters = torch.randn(layers, 2, hv, device="cuda") * 0.1
    pool = (
        torch.randn(
            layers, 2 * batch + 1, hv, v, k, device="cuda", dtype=torch.bfloat16
        )
        * 0.02
    )
    read = torch.arange(1, batch + 1, device="cuda", dtype=torch.int32).repeat(
        layers, 1
    )
    lengths = (
        torch.arange(batch, device="cuda", dtype=torch.int32) % 4 + 1
        if pattern == "mixed"
        else torch.full((batch,), int(pattern[-1]), device="cuda", dtype=torch.int32)
    )
    addresses = torch.tensor(
        [pool[i].data_ptr() for i in range(layers)], device="cuda", dtype=torch.uint64
    )
    strides = torch.full((layers,), pool.stride(1), device="cuda", dtype=torch.int64)
    return payload, parameters, pool, read, lengths, addresses, strides, read + batch


def launch(module, settings, tensors):
    payload, parameters, pool, read, lengths, addresses, strides, writes = tensors
    return module._gdn_replay_commit_persistent_kernel[(16,)](
        payload,
        parameters,
        addresses,
        strides,
        read,
        writes,
        lengths,
        lengths.numel(),
        4,
        payload.shape[0],
        payload.stride(0),
        H=4,
        HV=12,
        K=128,
        V=128,
        TOKEN_BLOCK=4,
        GATE_BLOCK=32,
        STATE_DTYPE_CODE=0,
        NUM_BUFFERS=settings["buffers"],
        SPT_K=8,
        ENABLE_PDL=True,
        num_warps=settings["warps"],
        maxnreg=settings["registers"],
        launch_pdl=True,
    )


def run(args):
    torch.cuda.set_device(args.gpu)
    pdl_enabled(True)
    kernel_root = Path(__file__).resolve().parents[3]
    configurations = {
        "baseline": {
            "source": str(
                kernel_root
                / "python/tokenspeed_kernel/ops/attention/gdn/gluon_nvidia.py"
            ),
            "warps": 8,
            "registers": 168,
            "buffers": 2,
        },
        "scaled": {
            "source": str(Path(__file__).with_name("kernel.py")),
            "warps": 8,
            "registers": 168,
            "buffers": 2,
        },
        "stream": {
            "source": str(Path(__file__).with_name("kernel_stream.py")),
            "warps": 16,
            "registers": 96,
            "buffers": 3,
        },
    }
    unknown = set(args.names) - configurations.keys()
    if unknown:
        raise ValueError(f"Unknown candidate names: {sorted(unknown)}")
    configurations = {
        name: spec for name, spec in configurations.items() if name in args.names
    }
    if "baseline" not in configurations:
        raise ValueError("Include baseline for the correctness and timing comparison")
    modules = {
        "baseline": importlib.import_module(
            "tokenspeed_kernel.ops.attention.gdn.gluon_nvidia"
        )
    }
    for name, settings in configurations.items():
        if name != "baseline":
            modules[name] = load(Path(settings["source"]), f"gdn_cta16_saved_{name}")
    records = []
    for batch in args.batches:
        for pattern in args.patterns:
            tensors = inputs(batch, args.layers, pattern)
            pool = tensors[2]
            initial = pool.clone()
            launch(modules["baseline"], configurations["baseline"], tensors)
            torch.cuda.synchronize()
            expected = pool[:, batch + 1 :].clone()
            graphs = {}
            resources = {}
            for name, c in configurations.items():
                try:
                    pool[:, batch + 1 :].fill_(float("nan"))
                    kernel = launch(modules[name], c, tensors)
                    torch.cuda.synchronize()
                    torch.testing.assert_close(
                        pool[:, batch + 1 :], expected, rtol=2e-2, atol=1e-3
                    )
                    torch.testing.assert_close(
                        pool[:, : batch + 1], initial[:, : batch + 1], rtol=0, atol=0
                    )
                    resources[name] = {
                        "registers": kernel.n_regs,
                        "spills": kernel.n_spills,
                        "shared_bytes": kernel.metadata.shared,
                        "threads": kernel.metadata.num_warps * 32,
                    }
                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        launch(modules[name], c, tensors)
                    stream.synchronize()
                    graphs[name] = graph
                    print(
                        json.dumps(
                            {
                                "phase": "compiled",
                                "batch": batch,
                                "pattern": pattern,
                                "name": name,
                                "resources": resources[name],
                            }
                        ),
                        flush=True,
                    )
                except Exception as exc:
                    traceback.print_exc()
                    print(
                        json.dumps(
                            {
                                "phase": "candidate_error",
                                "name": name,
                                "batch": batch,
                                "pattern": pattern,
                                "error": str(exc)[:1200],
                            }
                        ),
                        flush=True,
                    )
                    raise
            if args.action == "profile":
                for name, graph in graphs.items():
                    for _ in range(3):
                        graph.replay()
                    torch.cuda.synchronize()
                    torch.cuda.cudart().cudaProfilerStart()
                    graph.replay()
                    torch.cuda.synchronize()
                    torch.cuda.cudart().cudaProfilerStop()
                continue
            rng = random.Random(1923)
            names = list(graphs)
            samples = {k: [] for k in graphs}
            for _ in range(args.blocks):
                rng.shuffle(names)
                for name in names:
                    graph = graphs[name]
                    for _ in range(4):
                        graph.replay()
                    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                        enable_timing=True
                    )
                    begin.record()
                    for _ in range(args.iterations):
                        graph.replay()
                    end.record()
                    end.synchronize()
                    samples[name].append(
                        begin.elapsed_time(end) * 1000 / args.iterations
                    )
            for name, values in samples.items():
                record = {
                    "batch": batch,
                    "layers": args.layers,
                    "pattern": pattern,
                    "name": name,
                    "cta_count": 16,
                    "median_us": statistics.median(values),
                    "samples_us": values,
                    "resources": resources[name],
                    "source_sha256": hashlib.sha256(
                        Path(configurations[name]["source"]).read_bytes()
                    ).hexdigest(),
                }
                records.append(record)
                print(json.dumps(record), flush=True)
            args.output.write_text(
                json.dumps(
                    {
                        "scope": "16CTA enforced; actual TP4 GDN geometry; fixed-work CUDA graphs; repository BF16 tolerances and untouched-row bitwise comparison. No full MTP.",
                        "results": records,
                    },
                    indent=2,
                )
                + "\n"
            )
    if args.action == "bench":
        assert records


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--action", choices=["bench", "profile"], required=True)
    p.add_argument(
        "--names",
        nargs="+",
        choices=["baseline", "scaled", "stream"],
        required=True,
    )
    p.add_argument("--batches", nargs="+", type=int, required=True)
    p.add_argument(
        "--patterns",
        nargs="+",
        choices=["full4", "mixed", "accept0", "accept1", "accept2", "accept3"],
        required=True,
    )
    p.add_argument("--layers", type=int, required=True)
    p.add_argument("--blocks", type=int, required=True)
    p.add_argument("--iterations", type=int, required=True)
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    run(p.parse_args())
