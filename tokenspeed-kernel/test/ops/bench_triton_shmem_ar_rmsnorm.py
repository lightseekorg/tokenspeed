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

"""Multi-process microbenchmark for the fused AR + residual + RMSNorm backends.

Self-contained (``torch.distributed`` + ``mp.spawn``, no new deps) latency
microbench for the migrated ``triton_shmem`` backend. Times, per ``(world_size,
M, N)``, the full production op (input copy-in -> barrier -> fused kernel ->
barrier -> copy-out) for each backend selectable through the
``TS_ARNORM_BACKEND`` dispatch, plus production and explicit RCCL unfused
controls:

* ``production_unfused`` -- ordinary Iris all-reduce at or below the production
  512-KiB gate and RCCL above it, followed by TokenSpeed residual RMSNorm.
* ``rccl_unfused`` -- ``dist.all_reduce`` (native bf16 transport) + residual add
  + eager ``F.rms_norm`` diagnostic.
* ``triton_ar_unfused`` -- TokenSpeed's 512-KiB-gated Triton all-reduce +
  the same eager residual/norm. Unsupported rows are reported as NaN. This is a
  standalone small-message reference, not the crossover baseline.
* ``triton_shmem``   -- the migrated PyTorch symmetric-memory backend (this project).
* ``symm_mem``     -- the native TokenSpeed symm_mem fused kernel.
* ``iris``         -- the Iris backend.

For each config it CUDA-event-times every rank, takes the maximum rank for each
iteration before computing p50/p95/p99, retains raw samples, and reports
cross-rank skew. The default N axis
is modeled after the row widths encountered by the fused AR+RMSNorm / comm+norm
ops in the targeted large models (gpt-oss-120B, GLM-4.6, DeepSeek-V3/V4, Kimi-K2
-- hidden and MLA-compressed shapes; see the axis note below), and M spans
decode through the production chunked-prefill fusion cap.

Correctness is asserted (fp32 reference, 2e-2 tol) on the first iteration of
every config for every backend before timing, so a numerically-broken backend
fails loudly rather than reporting a fast-but-wrong latency.

Run (inside the ROCm container, from the tokenspeed repo root)::

    python tokenspeed-kernel/test/ops/bench_triton_shmem_ar_rmsnorm.py
    BENCH_WORLD_SIZES=8 \
      python tokenspeed-kernel/test/ops/bench_triton_shmem_ar_rmsnorm.py
    BENCH_BACKENDS=triton_shmem,production_unfused \
      python tokenspeed-kernel/test/ops/bench_triton_shmem_ar_rmsnorm.py
"""

from __future__ import annotations

import json
import os
import socket
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

_EPS = 1e-6
_TRITON_AR_MAX_BYTES = 512 * 1024
_DEFAULT_N_VALUES = [512, 1536, 2880, 5120, 7168]


def _env_list(name: str, default: List) -> List:
    raw = os.environ.get(name)
    if not raw:
        return default
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if default and isinstance(default[0], int):
        return [int(p) for p in parts]
    return parts


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(raw) if raw else default


# Axes. The N grid is modeled after the *actually encounterable* row widths of
# the AR+RMSNorm / fused comm+norm ops in the targeted large models rather than
# a synthetic power-of-two sweep:
#     512  -- DeepSeek-V3/V4 & Kimi-K2 kv_lora_rank (compressed-KV latent norm);
#             also GLM-4.x-lite kv_lora_rank. Power-of-two -> reaches
#             oneshot_wholerow at ws<=2.
#    1536  -- DeepSeek/Kimi q_lora_rank (compressed-Q latent norm); = GLM
#             moe_intermediate_size. Non-pow2 -> blocked kernels.
#    2880  -- gpt-oss-120B hidden size (dense residual stream).
#    5120  -- GLM-4.6 hidden size.
#    7168  -- DeepSeek-V3/V4 & Kimi-K2 hidden size (the big residual stream).
# The M grid spans representative inference token counts: decode/low-concurrency
# (1..128), chunked-prefill up to the production fusion cap (256..2048), and one
# point past the cap (4096) to show the trend beyond where fusion is served. All
# axes are env-overridable (BENCH_M_VALUES / BENCH_N_VALUES / BENCH_WORLD_SIZES /
# BENCH_BACKENDS); the triton_shmem kernels take arbitrary M and N (blocked
# variants mask N; two-shot cdiv-shards M), so there is no size gate to respect.
_DEFAULT_WORLD_SIZES: List[int] = [2, 4, 8]
_M_VALUES: List[int] = _env_list(
    "BENCH_M_VALUES", [1, 8, 32, 128, 256, 512, 1024, 2048, 4096]
)
_N_VALUES: List[int] = _env_list("BENCH_N_VALUES", _DEFAULT_N_VALUES)
_DEFAULT_BACKENDS: List[str] = [
    "production_unfused",
    "auto",
    "triton_shmem",
]

# Noise control uses a steady-state warmup and a per-rank synchronized sample
# distribution rather than relying on a single aggregate timer.
_N_WARMUP = _env_int("BENCH_N_WARMUP", 50)
_N_REPEAT = _env_int("BENCH_N_REPEAT", 300)


def _get_open_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def _reference(x, residual, weight, world_size, hidden, device):
    reduced = torch.full(
        (x.shape[0], hidden),
        world_size * (world_size + 1) // 2,
        dtype=torch.float32,
        device=device,
    )
    ref_residual = reduced + residual.float()
    ref_norm = ref_residual * torch.rsqrt(
        ref_residual.pow(2).mean(dim=-1, keepdim=True) + _EPS
    )
    ref_norm = ref_norm * weight.float()
    return ref_residual, ref_norm


def _make_inputs(tokens, hidden, rank, device):
    # rank contributes rank+1 (sum = ws*(ws+1)/2); residual deterministic and
    # replicated across ranks (as under TP) so a bug can't be masked.
    x = torch.full((tokens, hidden), rank + 1, dtype=torch.bfloat16, device=device)
    residual = (
        torch.arange(tokens * hidden, dtype=torch.float32, device=device)
        .reshape(tokens, hidden)
        .mul_(0.001)
        .to(torch.bfloat16)
    )
    return x, residual


def _prepare_backend(backend, x, scratch):
    """Reset an in-place unfused transport outside the timed region."""
    if backend in (
        "production_unfused",
        "rccl_unfused",
        "triton_ar_unfused",
    ):
        scratch.copy_(x)


def _run_backend(
    backend,
    x,
    residual,
    weight,
    rank,
    group,
    max_token_num,
    scratch,
    triton_ar_state,
):
    """Run one fused op for ``backend``; return (norm_out, residual_out)."""
    hidden = x.shape[1]
    if backend == "production_unfused":
        from tokenspeed_kernel.ops.communication.triton import all_reduce
        from tokenspeed_kernel.ops.layernorm.triton import rmsnorm

        if x.numel() * x.element_size() <= _TRITON_AR_MAX_BYTES:
            all_reduce(triton_ar_state, scratch)
        else:
            dist.all_reduce(scratch, group=group)
        return rmsnorm(scratch, weight, _EPS, residual=residual)
    if backend == "rccl_unfused":
        dist.all_reduce(scratch, group=group)
        residual_out = scratch + residual
        norm_out = F.rms_norm(residual_out, [hidden], weight, _EPS)
        return norm_out, residual_out
    if backend == "triton_ar_unfused":
        if x.numel() * x.element_size() > _TRITON_AR_MAX_BYTES:
            return None, None
        from tokenspeed_kernel.ops.communication.triton import all_reduce

        all_reduce(triton_ar_state, scratch)
        residual_out = scratch + residual
        norm_out = F.rms_norm(residual_out, [hidden], weight, _EPS)
        return norm_out, residual_out

    # All fused backends share the production dispatcher; TS_ARNORM_BACKEND
    # (set by the caller) selects which one runs.
    from tokenspeed_kernel.ops.communication.triton import (
        allreduce_residual_rmsnorm,
    )

    norm_out, residual_out, _, _ = allreduce_residual_rmsnorm(
        input_tensor=x,
        residual=residual,
        weight=weight,
        rank=rank,
        group=group,
        eps=_EPS,
        max_token_num=max_token_num,
    )
    return norm_out, residual_out


def _time_backend(
    backend,
    x,
    residual,
    weight,
    rank,
    group,
    max_token_num,
    scratch,
    triton_ar_state,
) -> list[float]:
    """Return this rank's per-iteration latency samples in milliseconds."""
    for _ in range(_N_WARMUP):
        _prepare_backend(backend, x, scratch)
        _run_backend(
            backend,
            x,
            residual,
            weight,
            rank,
            group,
            max_token_num,
            scratch,
            triton_ar_state,
        )
    torch.cuda.synchronize()
    dist.barrier(group=group)

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(_N_REPEAT)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(_N_REPEAT)]
    for i in range(_N_REPEAT):
        _prepare_backend(backend, x, scratch)
        starts[i].record()
        _run_backend(
            backend,
            x,
            residual,
            weight,
            rank,
            group,
            max_token_num,
            scratch,
            triton_ar_state,
        )
        ends[i].record()
    torch.cuda.synchronize()
    return [s.elapsed_time(e) for s, e in zip(starts, ends)]


def _presize_iris_heap(max_token_num: int) -> None:
    """Pre-create the process-global iris context sized for the ENTIRE sweep.

    The iris shim uses a module-level singleton context whose heap is fixed at
    first use and never grows; each distinct (max_token_num, hidden_dim) state
    allocates a buffer from it and is cached forever. Left to the per-config
    default sizing, the singleton is sized for the first (smallest) config and
    later configs overflow it -- the reason iris was previously excluded from the
    multi-config sweep. Sizing it up front for the sum of all N buffers removes
    that harness limitation (it is not an iris kernel/input-size limit).
    """
    from tokenspeed_kernel.ops.communication import iris as iris_mod

    itemsize = torch.tensor([], dtype=torch.bfloat16).element_size()
    sum_bytes = sum(max_token_num * n * itemsize for n in _N_VALUES)
    heap = max(1 << 28, 4 * sum_bytes + (256 << 20))
    iris_mod._get_or_create_iris_context(heap)


def _worker_fn(rank, world_size, port, backends, result_dict, error_dict):
    try:
        _worker_main(rank, world_size, port, backends, result_dict)
    except Exception:
        error_dict[rank] = traceback.format_exc()


def _worker_main(rank, world_size, port, backends, result_dict):
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://localhost:{port}",
        rank=rank,
        world_size=world_size,
    )
    try:
        group = dist.group.WORLD
        max_token_num = max(_M_VALUES)
        triton_ar_state = None
        if any(
            backend in backends
            for backend in ("production_unfused", "triton_ar_unfused")
        ):
            from tokenspeed_kernel.ops.communication.triton import create_state

            triton_ar_state = create_state(
                group=group,
                rank_in_group=rank,
                device=device,
                max_numel=_TRITON_AR_MAX_BYTES
                // torch.empty((), dtype=torch.bfloat16).element_size(),
            )
        if any(
            backend in backends for backend in ("production_unfused", "auto", "iris")
        ):
            _presize_iris_heap(max_token_num)
        for n in _N_VALUES:
            weight = torch.linspace(0.5, 1.5, n, dtype=torch.bfloat16, device=device)
            for m in _M_VALUES:
                x, residual = _make_inputs(m, n, rank, device)
                scratch = torch.empty_like(x)
                ref_residual, ref_norm = _reference(
                    x, residual, weight, world_size, n, device
                )
                for backend in backends:
                    if backend not in (
                        "production_unfused",
                        "rccl_unfused",
                        "triton_ar_unfused",
                    ):
                        os.environ["TS_ARNORM_BACKEND"] = backend
                    _prepare_backend(backend, x, scratch)
                    # Correctness gate before timing.
                    norm_out, residual_out = _run_backend(
                        backend,
                        x,
                        residual,
                        weight,
                        rank,
                        group,
                        max_token_num,
                        scratch,
                        triton_ar_state,
                    )
                    if norm_out is None:
                        result_dict[(rank, backend, m, n)] = float("nan")
                        continue
                    torch.testing.assert_close(
                        residual_out.float(), ref_residual, atol=2e-2, rtol=2e-2
                    )
                    torch.testing.assert_close(
                        norm_out.float(), ref_norm, atol=2e-2, rtol=2e-2
                    )
                    samples = _time_backend(
                        backend,
                        x,
                        residual,
                        weight,
                        rank,
                        group,
                        max_token_num,
                        scratch,
                        triton_ar_state,
                    )
                    result_dict[(rank, backend, m, n)] = samples
    finally:
        os.environ.pop("TS_ARNORM_BACKEND", None)
        dist.destroy_process_group()


def _aggregate(result_dict, world_size, backends) -> List[dict]:
    """Collapse rank samples after taking each iteration's slowest rank."""

    def percentile(samples: list[float], fraction: float) -> float:
        ordered = sorted(samples)
        return ordered[round((len(ordered) - 1) * fraction)]

    rows = []
    for n in _N_VALUES:
        for m in _M_VALUES:
            for backend in backends:
                rank_samples = [
                    result_dict.get((r, backend, m, n)) for r in range(world_size)
                ]
                if any(samples is None for samples in rank_samples) or any(
                    not isinstance(samples, list) for samples in rank_samples
                ):
                    rows.append(
                        {
                            "backend": backend,
                            "path": "unsupported",
                            "M": m,
                            "N": n,
                            "lat_ms": float("nan"),
                            "p95_ms": float("nan"),
                            "p99_ms": float("nan"),
                            "skew_pct": float("nan"),
                            "max_rank_samples_ms": [],
                            "rank_samples_ms": [],
                        }
                    )
                    continue
                sample_count = min(len(samples) for samples in rank_samples)
                max_rank_samples = [
                    max(samples[index] for samples in rank_samples)
                    for index in range(sample_count)
                ]
                rank_p50s = [percentile(samples, 0.5) for samples in rank_samples]
                lat = percentile(max_rank_samples, 0.5)
                lo = min(rank_p50s)
                skew = (lat - lo) / lat * 100.0 if lat > 0 else 0.0
                if backend == "production_unfused":
                    path = "iris" if m * n * 2 <= _TRITON_AR_MAX_BYTES else "rccl"
                elif backend == "auto":
                    path = "iris_fused"
                else:
                    path = backend
                rows.append(
                    {
                        "backend": backend,
                        "path": path,
                        "M": m,
                        "N": n,
                        "lat_ms": lat,
                        "p95_ms": percentile(max_rank_samples, 0.95),
                        "p99_ms": percentile(max_rank_samples, 0.99),
                        "skew_pct": skew,
                        "max_rank_samples_ms": max_rank_samples,
                        "rank_samples_ms": rank_samples,
                    }
                )
    return rows


def _run_one_world(world_size: int, backends: List[str]) -> List[dict]:
    manager = mp.Manager()
    result_dict = manager.dict()
    error_dict = manager.dict()
    port = _get_open_port()
    mp.spawn(
        _worker_fn,
        args=(world_size, port, backends, result_dict, error_dict),
        nprocs=world_size,
        join=True,
    )
    if error_dict:
        msg = "\n".join(f"Rank {r}: {e}" for r, e in error_dict.items())
        raise RuntimeError(f"worker failure at ws={world_size}:\n{msg}")
    return _aggregate(dict(result_dict), world_size, backends)


def _print_world(world_size: int, rows: List[dict], backends: List[str]) -> None:
    baseline = "production_unfused"
    by_cfg = {}
    for r in rows:
        by_cfg.setdefault((r["M"], r["N"]), {})[r["backend"]] = r
    print(f"\n===== world_size = {world_size} =====")
    hdr = (
        f"{'M':>6} {'N':>6} | "
        + " | ".join(f"{b:>16}" for b in backends)
        + " | "
        + " ".join(f"{b.split('_')[0]:>7}x" for b in backends if b != baseline)
    )
    print(hdr)
    print("-" * len(hdr))
    for n in _N_VALUES:
        for m in _M_VALUES:
            cfg = by_cfg.get((m, n), {})
            base = cfg.get(baseline, {}).get("lat_ms", float("nan"))
            lat_cells = []
            spd_cells = []
            for b in backends:
                lat = cfg.get(b, {}).get("lat_ms", float("nan"))
                lat_cells.append(f"{lat:16.4f}" if lat == lat else f"{'n/a':>16}")
                if b != baseline:
                    spd = base / lat if (lat == lat and lat > 0) else float("nan")
                    spd_cells.append(f"{spd:7.2f}x" if spd == spd else f"{'n/a':>8}")
            print(
                f"{m:>6} {n:>6} | "
                + " | ".join(lat_cells)
                + " | "
                + " ".join(spd_cells)
            )


def _write_csv(path: str, all_rows: List[Tuple[int, dict]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write("world_size,backend,path,M,N,lat_ms,p95_ms,p99_ms,skew_pct\n")
        for ws, r in all_rows:
            f.write(
                f"{ws},{r['backend']},{r['path']},{r['M']},{r['N']},"
                f"{r['lat_ms']:.6f},{r['p95_ms']:.6f},"
                f"{r['p99_ms']:.6f},{r['skew_pct']:.3f}\n"
            )


def _write_samples_json(path: str, all_rows: List[Tuple[int, dict]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "method": "per-iteration maximum rank before percentiles",
        "warmup": _N_WARMUP,
        "repeat": _N_REPEAT,
        "world_sizes": sorted({world_size for world_size, _ in all_rows}),
        "backends": sorted({row["backend"] for _, row in all_rows}),
        "M_values": list(_M_VALUES),
        "N_values": list(_N_VALUES),
        "torch_version": torch.__version__,
        "rocm_version": torch.version.hip,
        "devices": [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ],
        "code_identity": os.environ.get("BENCH_CODE_ID"),
        "image_identity": os.environ.get("BENCH_IMAGE_ID"),
        "rows": [{"world_size": world_size, **row} for world_size, row in all_rows],
    }
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA/ROCm required")
    world_sizes = _env_list("BENCH_WORLD_SIZES", _DEFAULT_WORLD_SIZES)
    backends = _env_list("BENCH_BACKENDS", _DEFAULT_BACKENDS)
    ndev = torch.cuda.device_count()
    print(
        f"torch={torch.__version__} rocm={torch.version.hip} "
        f"devices={[torch.cuda.get_device_name(i) for i in range(ndev)]} "
        f"warmup={_N_WARMUP} repeat={_N_REPEAT}"
    )
    csv_path = os.environ.get("BENCH_CSV")
    samples_json_path = os.environ.get("BENCH_SAMPLES_JSON")

    all_rows: List[Tuple[int, dict]] = []
    for ws in world_sizes:
        if ws > ndev:
            print(f"skip ws={ws}: only {ndev} GPUs")
            continue
        rows = _run_one_world(ws, backends)
        _print_world(ws, rows, backends)
        all_rows.extend((ws, r) for r in rows)

    if csv_path:
        _write_csv(csv_path, all_rows)
        print(f"\nwrote {csv_path}")
    if samples_json_path:
        _write_samples_json(samples_json_path, all_rows)
        print(f"wrote {samples_json_path}")


if __name__ == "__main__":
    main()
