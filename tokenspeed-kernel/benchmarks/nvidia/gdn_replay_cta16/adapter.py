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

"""Real CUDA command backend; the adapter owns the fixed 16-CTA launch bound."""

import argparse
import importlib.util
import json
import sys
from functools import partial
from pathlib import Path
from types import ModuleType
from typing import Any

import torch
from tokenspeed_kernel._triton import triton
from tokenspeed_kernel.ops.attention.gdn import gdn_replay_commit
from tokenspeed_kernel.platform import max_active_ctas, pdl_enabled

sys.dont_write_bytecode = True
KERNEL_ROOT = Path(__file__).resolve().parents[3]
MAX_CTAS = 16


def _load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _candidate(path: Path) -> tuple[ModuleType, dict[str, int]]:
    manifest = json.loads((path / "candidate.json").read_text())
    settings = manifest["metadata"]["settings"]
    if set(settings) != {"max_ctas", "num_buffers", "num_warps", "max_registers"}:
        raise ValueError("candidate must supply every experiment setting explicitly")
    if settings["max_ctas"] != MAX_CTAS:
        raise ValueError("this experiment fixes the CTA bound at 16")
    if settings["num_buffers"] not in (1, 2, 3, 4):
        raise ValueError("num_buffers must be 1, 2, 3, or 4")
    return _load_module(path / "kernel.py", "gdn_kernel_candidate"), settings


def _inputs(params: dict[str, Any]):
    torch.manual_seed(params["seed"])
    b, l, t = params["batch"], params["layers"], params["tokens"]
    h, hv, k, v = params["geometry"]
    width = h * k + hv * v + 2 * hv
    payload = torch.randn(l, b * t, width, device="cuda", dtype=torch.bfloat16)
    parameters = torch.randn(l, 2, hv, device="cuda") * 0.1
    pool = (
        torch.randn(l, 2 * b + 1, hv, v, k, device="cuda", dtype=torch.bfloat16) * 0.02
    )
    read = torch.arange(1, b + 1, device="cuda", dtype=torch.int32).repeat(l, 1)
    pattern = params["accepted_lengths"]
    if not pattern or any(type(n) is not int or not 0 <= n <= t for n in pattern):
        raise ValueError(f"accepted_lengths must contain integers between 0 and {t}")
    accepted = torch.tensor(pattern, device="cuda", dtype=torch.int32)
    accepted = accepted.repeat(triton.cdiv(b, len(pattern)))[:b].clone()
    kwargs = dict(
        state_addresses=torch.tensor(
            [pool[i].data_ptr() for i in range(l)], device="cuda", dtype=torch.uint64
        ),
        state_row_strides=torch.full(
            (l,), pool.stride(1), device="cuda", dtype=torch.int64
        ),
        read_indices=read,
        write_indices=read + b,
        accepted_length=accepted,
        draft_token_num=t,
        geometry=(h, hv, k, v),
        state_dtype=pool.dtype,
    )
    return payload, parameters, pool, kwargs


def _launch(
    module,
    settings,
    payload,
    parameters,
    *,
    state_addresses,
    state_row_strides,
    read_indices,
    write_indices,
    accepted_length,
    draft_token_num,
    geometry,
    state_dtype,
):
    h, hv, k, v = geometry
    b = accepted_length.numel()
    l, t = payload.shape[0], draft_token_num
    codes = {torch.bfloat16: 0, torch.float16: 1, torch.float32: 2}
    pdl = pdl_enabled()
    # Candidate launcher code cannot enlarge the grid: launch geometry belongs
    # to the evaluator, independently of the candidate's optimization hypothesis.
    return module._gdn_replay_commit_persistent_kernel[(min(MAX_CTAS, l * b * hv),)](
        payload,
        parameters,
        state_addresses,
        state_row_strides,
        read_indices,
        write_indices,
        accepted_length,
        b,
        t,
        l,
        payload.stride(0),
        H=h,
        HV=hv,
        K=k,
        V=v,
        TOKEN_BLOCK=triton.next_power_of_2(t),
        GATE_BLOCK=triton.next_power_of_2(2 * hv),
        STATE_DTYPE_CODE=codes[state_dtype],
        NUM_BUFFERS=settings["num_buffers"],
        SPT_K=4 if state_dtype == torch.float32 else 8,
        ENABLE_PDL=pdl,
        num_warps=settings["num_warps"],
        maxnreg=settings["max_registers"],
        **({"launch_pdl": True} if pdl else {}),
    )


def _build(module, settings):
    payload, parameters, pool, kwargs = _inputs(
        dict(
            seed=731,
            batch=1,
            layers=1,
            tokens=4,
            geometry=[4, 12, 128, 128],
            accepted_lengths=[4],
        )
    )
    kernel = _launch(module, settings, payload, parameters, **kwargs)
    torch.cuda.synchronize()
    device = torch.cuda.get_device_properties(0)
    return dict(
        ok=True,
        message="compiled and launched on CUDA",
        max_ctas=MAX_CTAS,
        arch=f"sm{device.major}{device.minor}",
        sm_count=device.multi_processor_count,
        registers=kernel.n_regs,
        spills=kernel.n_spills,
        shared_bytes=kernel.metadata.shared,
        threads=kernel.metadata.num_warps * 32,
        compute_warps=settings["num_warps"],
        resident_ctas_per_sm=max_active_ctas(kernel),
        torch_version=torch.__version__,
        cuda_runtime=torch.version.cuda,
    )


def _check(module, settings, params):
    payload, parameters, pool, kwargs = _inputs(params)
    b, t = params["batch"], params["tokens"]
    initial = pool.clone()
    if params["correctness_contract"] == "bitwise":
        rtol, atol = 0, 0
    elif params["correctness_contract"] == "repository":
        rtol, atol = {
            torch.bfloat16: (2e-2, 1e-3),
            torch.float16: (1e-3, 1e-6),
            torch.float32: (1e-5, 1e-6),
        }[pool.dtype]
    else:
        raise ValueError(
            f"unknown correctness contract: {params['correctness_contract']}"
        )
    counts = [kwargs["accepted_length"].clone()] + [
        (torch.arange(b, device="cuda").to(torch.int32) + phase) % (t + 1)
        for phase in (0, t)
    ]
    for lengths in counts:
        kwargs["accepted_length"].copy_(lengths)
        kwargs["read_indices"][:, 0] = -1
        if b > 1:
            kwargs["write_indices"][:, 1] = kwargs["read_indices"][:, 1]
        if b > 2:
            kwargs["write_indices"][:, -1] = -1
        pool.copy_(initial)
        gdn_replay_commit(payload, parameters, solution="triton", **kwargs)
        expected = pool.clone()
        pool.copy_(initial)
        _launch(module, settings, payload, parameters, **kwargs)
        torch.cuda.synchronize()
        torch.testing.assert_close(pool, expected, rtol=rtol, atol=atol)
        # Numerical tolerances apply only to committed states. Every other row
        # must remain unchanged, including the null row and source-only rows.
        writes = kwargs["write_indices"]
        written = torch.zeros(pool.shape[:2], dtype=torch.int32, device=pool.device)
        written.scatter_add_(1, writes.clamp_min(0).long(), (writes >= 0).int())
        torch.testing.assert_close(
            pool[written == 0], initial[written == 0], rtol=0, atol=0
        )
    return dict(
        correct=True,
        message=f"{params['correctness_contract']} parity; exact untouched rows; configured and mixed acceptance, fresh, in-place, and padding rows",
    )


def _verify(module, settings):
    # The independent oracle uses PyTorch recurrence, not the registered replay
    # implementation used by the smoke check. Its odd row strides also exercise
    # the synchronous state-copy fallback.
    sys.path.insert(0, str(KERNEL_ROOT / "test"))
    oracle = _load_module(
        KERNEL_ROOT / "test/nvidia/ops/test_gdn_replay_pipeline.py", "gdn_replay_oracle"
    )
    for dtype in (torch.bfloat16, torch.float16, torch.float32):
        for padding in (0, 3):
            case = oracle._case(
                batch=5,
                tokens=4,
                layers=3,
                state_dtype=dtype,
                row_padding=padding,
                token_capacity=23,
            )
            kwargs = dict(
                state_addresses=case.addresses,
                state_row_strides=case.strides,
                read_indices=case.read,
                write_indices=case.write,
                accepted_length=case.accepted,
                draft_token_num=case.tokens,
                geometry=(2, 6, 128, 128),
                state_dtype=dtype,
            )
            for counts in ([0, 1, 7, 4, 2], [4, -1, 1, 2, 4]):
                case.accepted.copy_(
                    torch.tensor(counts, device="cuda", dtype=torch.int32)
                )
                initial = case.storage.clone()
                _launch(module, settings, case.payload, case.parameters, **kwargs)
                oracle._check_result(case, initial)
    return dict(
        correct=True,
        message="independent FP32 PyTorch recurrence; three state dtypes and aligned/odd-stride pools",
    )


def _graph(commit, params):
    graph = torch.cuda.CUDAGraph()
    main, side = torch.cuda.Stream(), torch.cuda.Stream()
    main.wait_stream(torch.cuda.current_stream())
    if params["mode"] == "commit":
        with torch.cuda.graph(graph, stream=main):
            commit()
        return graph, ()
    size, b, t = params["gemm_size"], params["batch"], params["tokens"]
    x = torch.randn(b * t, size, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(size, size, device="cuda", dtype=torch.bfloat16)
    y = torch.empty_like(x)

    def gemm():
        for _ in range(params["gemm_repeats"]):
            torch.mm(x, weight, out=y)

    for _ in range(4):
        gemm()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph, stream=main):
        side.wait_stream(main)
        if params["order"] == "draft_first":
            gemm()
        with torch.cuda.stream(side):
            commit()
        if params["order"] == "commit_first":
            gemm()
        main.wait_stream(side)
    return graph, (x, weight, y)


def _bench(module, settings, params, repeats):
    payload, parameters, pool, kwargs = _inputs(params)
    commit = partial(_launch, module, settings, payload, parameters, **kwargs)
    for _ in range(4):
        commit()
    torch.cuda.synchronize()
    graph, graph_inputs = _graph(commit, params)
    for _ in range(8):
        graph.replay()
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(params["iterations"]):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / params["iterations"])
    return dict(
        samples_us=samples,
        max_ctas=MAX_CTAS,
        settings=settings,
        accepted_lengths=kwargs["accepted_length"].tolist(),
        draft_token_num=params["tokens"],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action", choices=("build", "check", "verify", "bench", "profile")
    )
    parser.add_argument("candidate", type=Path)
    parser.add_argument("workload", type=Path, nargs="?")
    parser.add_argument("repeats", type=int, nargs="?")
    args = parser.parse_args()
    torch.cuda.set_device(0)
    module, settings = _candidate(args.candidate)
    if args.action == "build":
        result = _build(module, settings)
    elif args.action == "verify":
        result = _verify(module, settings)
    else:
        if args.workload is None:
            parser.error("the action requires a workload file")
        params = json.loads(args.workload.read_text())["params"]
        if args.action == "check":
            result = _check(module, settings, params)
        elif args.action == "bench":
            if args.repeats is None:
                parser.error("bench requires the number of samples")
            result = _bench(module, settings, params, args.repeats)
        else:
            payload, parameters, pool, kwargs = _inputs(params)
            kernel = _launch(module, settings, payload, parameters, **kwargs)
            torch.cuda.synchronize()
            _launch(module, settings, payload, parameters, **kwargs)
            torch.cuda.synchronize()
            result = dict(
                ok=True, registers=kernel.n_regs, shared_bytes=kernel.metadata.shared
            )
    print(json.dumps(result))


if __name__ == "__main__":
    main()
