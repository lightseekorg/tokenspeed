# MiniMax Sparse Attention (MSA) — CuTe-DSL kernel

This is the **CuTe-DSL** implementation of MSA, shipped inside the
`fmha_sm100` Python package. For the package overview, install steps, and
the dense csrc JIT path, see the
[top-level README](../../../../README.md).

The rest of this file documents the **sparse (CuTe-DSL)** surface only: CSR
metadata, schedules, sparse page attention, FP8 / NVFP4 / FP4 quantization,
the paged FP8 decode wrapper, and the FP4 indexer.

---

The public entrypoint is [`sparse_atten_func`](./interface.py), which accepts
CSR-style sparse metadata plus an optional prepared schedule and supports:

- sparse attention forward
- sparse page attention forward for inference
- causal and non-causal execution
- varlen-style metadata via `cu_seqlens_*`

For production sparse attention, the recommended flow is to build CSR metadata
and the schedule together with [`build_k2q_csr(..., return_schedule=True)`](./sparse_index_utils.py),
then pass the returned schedule into [`sparse_atten_func`](./interface.py).

## Current Support

The current public support contract is intentionally narrow:

| Feature | Current support |
|---|---|
| Head dimension | `D=128` |
| Input dtype | `torch.bfloat16`, `torch.float16` |
| Sparse attention forward | `qhead_per_kv` in `{1, 2, 4, 8, 16}` |
| CSR builder | `topK` in `{4, 8, 16, 32}`, `blk_kv=128` |
| Sparse page attention | Forward-only, `qhead_per_kv` in `{1, 2, 4, 8, 16}` |
| FP8 KV prefill | Forward-only, BF16 Q + FP8 e4m3 K/V -> BF16 attention/output, flat and paged KV |
| Mixed FP8 QKV prefill | Forward-only, FP8 e4m3 Q/K/V storage with FP8 QK and BF16 PV, flat and paged KV |
| NVFP4 KV prefill | Forward-only, BF16 or FP8 e4m3 Q + packed NVFP4 K/V, flat and paged KV |
| Paged FP8 decode | Forward-only, FP8 e4m3 Q/K/V → BF16 O, `qhead_per_kv=16`, `page_size=128`, SM100 |
| FP4 indexer | SM100 block-score API, MXFP4/NVFP4, `D=128`, paged K, `blk_kv=128` |
| Tests and benchmarks | CUDA required |

## Installation

Install a CUDA-enabled PyTorch build that matches your environment first. Then
install the repo-side Python requirements:

```bash
make setup
```

## Quick Start

The runtime API expects CUDA tensors for Q/K/V, `cu_seqlens_*`, CSR metadata,
and the sparse attention schedule. The simplest production flow is:

1. Build `q`, `k`, `v`
2. Build CUDA `q2k_indices` with shape `[Hkv, total_q, topK]`
3. Build CSR and schedule with [`build_k2q_csr`](./sparse_index_utils.py)
4. Call `sparse_atten_func(..., schedule=schedule)`

Example:

```python
import torch

from interface import sparse_atten_func
from sparse_index_utils import build_k2q_csr

device = "cuda"
dtype = torch.bfloat16

Sq = 4096
Skv = 4096
head_kv = 1
qhead_per_kv = 16
head_q = head_kv * qhead_per_kv
dim = 128
topK = 16
blk_kv = 128

q = torch.randn(Sq, head_q, dim, device=device, dtype=dtype)
k = torch.randn(Skv, head_kv, dim, device=device, dtype=dtype)
v = torch.randn(Skv, head_kv, dim, device=device, dtype=dtype)

# Shape: [head_kv, total_q, topK]. Values are batch-local KV block indices.
q2k_indices = (
    torch.arange(topK, device=device, dtype=torch.int32)
    .view(1, 1, topK)
    .expand(head_kv, Sq, topK)
    .contiguous()
)

cu_seqlens_q = torch.tensor([0, Sq], device=device, dtype=torch.int32)
cu_seqlens_k = torch.tensor([0, Skv], device=device, dtype=torch.int32)

k2q_row_ptr, k2q_q_indices, schedule = build_k2q_csr(
    q2k_indices,
    cu_seqlens_q,
    cu_seqlens_k,
    blk_kv,
    total_k=Skv,
    max_seqlen_k=Skv,
    max_seqlen_q=Sq,
    total_rows=(Skv + blk_kv - 1) // blk_kv,
    qhead_per_kv=qhead_per_kv,
    return_schedule=True,
)

out, lse, lse_temperature_out = sparse_atten_func(
    q,
    k,
    v,
    k2q_row_ptr,
    k2q_q_indices,
    topK,
    blk_kv=blk_kv,
    causal=False,
    # Temperature LSE uses qk logits scaled by softmax_scale / lse_temperature_scale.
    lse_temperature_scale=1.0,
    return_temperature_lse=True,
    return_softmax_lse=True,
    cu_seqlens_q=cu_seqlens_q,
    cu_seqlens_k=cu_seqlens_k,
    max_seqlen_q=Sq,
    max_seqlen_k=Skv,
    schedule=schedule,
)
```

For a complete customer-scale example with NVTX ranges around CSR build and
FWD, see [`example.py`](./example.py).

## Input Metadata

### Sparse Attention

- `q`: `[total_q, Hq, D]`
- `k`, `v`: `[total_k, Hkv, D]`
- `cu_seqlens_q`: `[B + 1]`
- `cu_seqlens_k`: `[B + 1]`
- `max_seqlen_q`: required keyword-only host int, normally `max(diff(cu_seqlens_q))`
- `max_seqlen_k`: required keyword-only host int, normally `max(diff(cu_seqlens_k))`;
  for CP all-gather dKV buffers, pass the gathered physical K span that must be
  covered by postprocess zero-fill
- `k2q_row_ptr`: `[Hkv, total_rows + 1]`
- `k2q_q_indices`: `[Hkv, total_q * topK]`
- `schedule`: `SparseAttentionSchedule`, preferably returned by
  `build_k2q_csr(..., return_schedule=True)`

For each CSR row `r`, the payload slice
`k2q_q_indices[h, k2q_row_ptr[h, r]:k2q_row_ptr[h, r + 1]]` must be sorted in
ascending batch-local `q_idx` order. This is part of the runtime metadata
contract, especially for causal execution. Prefer `build_k2q_csr(...)` to
construct valid CSR metadata instead of manually permuting row payloads.

The schedule contains `scheduler_metadata`, `work_count`, `qsplit_indices`, and
`split_counts`. Treat it as an opaque object owned by the CSR builder. If a
schedule object is passed to attention, it must be complete; incomplete schedule
objects raise an error instead of silently falling back.

### Sparse Page Attention

- `q`: `[total_q, Hq, D]`
- `k`, `v`: `[num_pages, page_size, Hkv, D]`
- `page_table`: `[B, max_num_pages_per_seq]`
- `seqused_k`: optional `[B]`, logical valid KV length per batch
- `cu_seqlens_q`: `[B + 1]`
- do not pass `cu_seqlens_k` together with `page_table`

Use `seqused_k` whenever logical KV length is smaller than the physical page
capacity. This is the normal way to represent partially used tail pages.

### Quantized KV Prefill

The SM100 sparse forward kernel exposes quantized and mixed-precision prefill
paths through the public sparse attention interface:

- `fp8_kv`: call `sparse_atten_func` with BF16 Q and FP8 e4m3 K/V. The kernel
  stages FP8 K/V with TMA and converts to BF16 MMA shared-memory layout before
  BF16 QK/PV attention.
- `mixed_fp8_qkv_pv_bf16`: call `sparse_atten_func` with FP8 e4m3 Q/K/V and
  pass `qk_dtype=torch.float8_e4m3fn`, `pv_dtype=torch.bfloat16`. QK runs with
  FP8 operands, while V is cast from FP8 storage to BF16 for PV MMA. This path
  does not apply KV scales or dequantization.
- `nvfp4_kv`: call [`sparse_atten_nvfp4_kv_func`](./interface.py) with BF16 or
  FP8 e4m3 Q and packed NVFP4 K/V bytes plus `scale_128x4` tensors. The kernel
  stages packed K/V with TMA and converts through the NVFP4 scale path before
  MMA.

Both paths support flat varlen K/V and paged K/V. Paged K/V uses the same
logical layout contract as sparse page attention. For packed NVFP4 K/V,
`k.shape[-1] == v.shape[-1] == D // 2` because two E2M1 values are packed per
byte; `D` is still taken from `q.shape[-1]`.

The canonical names use the KV format first:

- function: `sparse_atten_nvfp4_kv_func`
- benchmark flags: `--nvfp4-kv`, `--fp8-kv`, `--mixed-fp8-qkv-pv-bf16`
- benchmark labels: `bf16_q_nvfp4_kv`, `fp8_q_nvfp4_kv`, `bf16_q_fp8_kv`,
  `fp8_qkv_qk_fp8_pv_bf16`

`sparse_atten_func` also accepts optional `qk_dtype` and `pv_dtype` keyword
arguments. They select the compile-time QK/PV MMA operand dtypes. By default,
`qk_dtype` follows Q storage dtype and `pv_dtype` follows V storage dtype,
except the legacy BF16-Q + FP8 K/V cache path keeps BF16 compute operands for
both QK and PV.

Example mixed FP8 QKV call:

```python
out, lse = sparse_atten_func(
    q_fp8,
    k_fp8,
    v_fp8,
    k2q_row_ptr,
    k2q_q_indices,
    topK,
    blk_kv=blk_kv,
    causal=True,
    partial_dtype=torch.bfloat16,
    return_softmax_lse=True,
    cu_seqlens_q=cu_seqlens_q,
    cu_seqlens_k=cu_seqlens_k,
    max_seqlen_q=max_seqlen_q,
    max_seqlen_k=max_seqlen_k,
    schedule=schedule,
    qk_dtype=torch.float8_e4m3fn,
    pv_dtype=torch.bfloat16,
)
```

### CSR And Schedule Preprocessing

[`sparse_index_utils.py`](./sparse_index_utils.py) exposes the public CSR build
helper:

- `build_k2q_csr(...)`

For the SM100 runtime path, `build_k2q_csr` dispatches to the CUDA CSR builder
and can fuse schedule generation when `return_schedule=True`. Pass:

- `total_k=int(k.shape[0])`
- `max_seqlen_k=max(k_lens)`
- `max_seqlen_q=max(q_lens)`
- `total_rows=sum(ceil_div(k_len, blk_kv) for k_len in k_lens)`
- `qhead_per_kv=head_q // head_kv`

The reference helpers below are for tests, validation, and offline debugging;
do not use them in the latency-sensitive runtime path:

- `q2k_to_k2q(...)`
- `k2q_to_q2k(...)`
- `build_k2q_csr_torch_reference(...)`

## FP4 Indexer Kernel

The FP4 indexer is the SM100 block-score producer for sparse-index selection.
Its public entrypoint is
[`fp4_indexer_block_scores`](./fp4_indexer_interface.py). It returns per-Q,
per-head, per-KV-block max scores only; it does not run `torch.topk`, build
CSR metadata, build schedules, or copy data back to host.

Example:

```python
from fp4_indexer_interface import fp4_indexer_block_scores

scores = fp4_indexer_block_scores(
    q_fp4,
    k_fp4,
    q_scale,
    k_scale,
    cu_seqlens_q,
    cu_seqlens_k,
    cu_page_offsets,
    max_seqlen_q=max_seqlen_q,
    max_seqlen_k=max_seqlen_k,
    kv_indices=kv_indices,
    fp4_format="nvfp4",          # or "mxfp4"
    causal=True,
    qo_offset=None,
    scale_layout="preordered_mma",
)
```

### FP4 Input And Output Contract

- `q_fp4`: `[total_q, Hq, 64]` packed FP4 bytes. Logical head dimension is
  `D=128`, packed as two FP4 values per byte.
- `k_fp4`: `[total_pages, Hkv, 128, 64]` packed paged-K FP4 bytes.
- `cu_seqlens_q`, `cu_seqlens_k`: `[B + 1]`, CUDA `torch.int32`.
- `cu_page_offsets`: `[B + 1]`, CUDA `torch.int32` prefix sums over the
  per-batch page counts.
- `kv_indices`: `[sum_pages]`, CUDA `torch.int32`, contiguous physical page
  indices in the same order described by `cu_page_offsets`.
- `fp4_format`: `"mxfp4"` or `"nvfp4"`. MXFP4 uses `G=4` scale groups;
  NVFP4 uses `G=8`.
- `scores`: `[Hq, ceil(max_seqlen_k / 128), total_q]`, `torch.float32`.
  Invalid, out-of-range, or causally masked blocks are written as `-inf`.

`q_fp4` and `k_fp4` may use `torch.uint8`, `torch.int8`, or
`torch.float4_e2m1fn_x2` packed storage. `Hq` must be divisible by `Hkv`.
Packed FP4 tensors must be CUDA tensors, contiguous in the expected layout, and
128-byte aligned for the TMA paths.

### FP4 Scale Layouts

Scale dtype is format-specific: MXFP4 uses `torch.float8_e8m0fnu`, while NVFP4
uses `torch.float8_e4m3fn`.

`scale_layout="preordered_mma"` is the default production path. It expects Q
and K scale tensors already stored in the contiguous MMA storage layout
`(L, restM, restG, 32, 4, 4)` with element stride
`(512*restM*restG, 512*restG, 512, 16, 4, 1)`. For Q, `mn=total_q` and
`L=Hq`; for K, `mn=128` and `L=total_pages * Hkv`.

`scale_layout="public"` is intended for validation and integration bring-up.
It accepts public scale tensors:

- Q scale: `[total_q, Hq, G]`
- K scale: `[total_pages, Hkv, 128, G]`

This layout is easier to construct, but the interface must launch the
standalone scale reorder kernel before the score kernel.

### FP4 Score Computation

For each batch item, query token, query head, and 128-token KV block, the
kernel computes:

```text
scores[hq, k_block, q_global] =
    max(dot(dequant_fp4(q[q_global, hq]), dequant_fp4(k[k_token, hkv])))
```

where `hkv = hq // (Hq / Hkv)` and the max is taken over valid K tokens in that
KV block. With `causal=True`, bottom-right causal masking is used by default.
Passing `qo_offset` provides an explicit per-batch causal offset; it is only
valid for causal calls.

The steady-state `preordered_mma` prefill path launches the score kernel. The
`public` layout adds the scale reorder kernel. Short decode cases
(`max_seqlen_q <= 8` and `Hq / Hkv == 16`) use a Q-pack kernel before the
decode score kernel. Compact causal scheduling can also initialize scores with
`scores.fill_(-inf)`.

### FP4 Indexer Benchmark

[`test_fp4_indexer.py`](./test_fp4_indexer.py) provides the correctness tests
and benchmark CLI for the FP4 indexer.

List the built-in benchmark cases:

```bash
python test_fp4_indexer.py benchmark --list-cases
```

Run the default suite across both FP4 formats and both scale layouts:

```bash
python test_fp4_indexer.py benchmark --format both --scale-layout both
```

Run the production-style decode case:

```bash
python test_fp4_indexer.py benchmark \
  --case decode_uniform \
  --format nvfp4 \
  --scale-layout preordered_mma
```

Run one custom causal prefill shape:

```bash
python test_fp4_indexer.py benchmark \
  --sq 4096 --skv 4096 --causal \
  --format nvfp4 \
  --scale-layout preordered_mma
```

The CLI reports `Time ms` and `Eff TFLOPS`. The reference table below records
`Eff TFLOPS`; for causal cases, this counts only visible causal work. Capture
numbers on the target SM100 system after JIT warmup; they depend on GPU clocks,
driver, CUDA/PyTorch versions, and whether the measured path includes
public-scale reorder.

Reference production-path measurements, captured 2026-05-21. The two
columns are anonymized on the public README — `GPU1` is a higher-clock
SM100 part, `GPU2` is a lower-clock SM100 part. The exact product
mapping is intentionally not recorded in the open-source tree. Command:
`python test_fp4_indexer.py benchmark --format both --scale-layout preordered_mma --causal --warmup 5 --iters 20 --repeats 5`.

| Case | Format | Scale Layout | Shape | GPU1 Eff TFLOPS | GPU2 Eff TFLOPS |
|---|---|---|---|---:|---:|
| `prefill_q8k_k8k` | MXFP4 | `preordered_mma` | `B=1, q=8192, k=8192, Hq=64, Hkv=4, D=128, blk_kv=128, causal=True` | 767.094 | 649.924 |
| `prefill_q8k_k64k` | MXFP4 | `preordered_mma` | `B=1, q=8192, k=65536, Hq=64, Hkv=4, D=128, blk_kv=128, causal=True` | 1503.998 | 1218.399 |
| `decode_uniform` | MXFP4 | `preordered_mma` | `B=30, q=30x8, k=30x67584, Hq=64, Hkv=4, D=128, blk_kv=128, causal=True` | 1244.191 | 1181.831 |
| `decode_1x2x` | MXFP4 | `preordered_mma` | `B=30, q=30x8, k=135168 + 29x65253, Hq=64, Hkv=4, D=128, blk_kv=128, causal=True` | 996.365 | 946.715 |
| `decode_5x2x` | MXFP4 | `preordered_mma` | `B=30, q=30x8, k=5x135168 + 25x54067, Hq=64, Hkv=4, D=128, blk_kv=128, causal=True` | 1016.901 | 964.791 |
| `decode_1x3x` | MXFP4 | `preordered_mma` | `B=30, q=30x8, k=202752 + 29x62923, Hq=64, Hkv=4, D=128, blk_kv=128, causal=True` | 985.514 | 936.619 |
| `decode_1x4x` | MXFP4 | `preordered_mma` | `B=30, q=30x8, k=270336 + 29x60592, Hq=64, Hkv=4, D=128, blk_kv=128, causal=True` | 972.666 | 922.291 |
| `prefill_q8k_k8k` | NVFP4 | `preordered_mma` | `B=1, q=8192, k=8192, Hq=64, Hkv=4, D=128, blk_kv=128, causal=True` | 751.115 | 628.598 |
| `prefill_q8k_k64k` | NVFP4 | `preordered_mma` | `B=1, q=8192, k=65536, Hq=64, Hkv=4, D=128, blk_kv=128, causal=True` | 1447.070 | 1185.231 |
| `decode_uniform` | NVFP4 | `preordered_mma` | `B=30, q=30x8, k=30x67584, Hq=64, Hkv=4, D=128, blk_kv=128, causal=True` | 1207.832 | 1163.759 |
| `decode_1x2x` | NVFP4 | `preordered_mma` | `B=30, q=30x8, k=135168 + 29x65253, Hq=64, Hkv=4, D=128, blk_kv=128, causal=True` | 980.374 | 926.523 |
| `decode_5x2x` | NVFP4 | `preordered_mma` | `B=30, q=30x8, k=5x135168 + 25x54067, Hq=64, Hkv=4, D=128, blk_kv=128, causal=True` | 1001.852 | 944.185 |
| `decode_1x3x` | NVFP4 | `preordered_mma` | `B=30, q=30x8, k=202752 + 29x62923, Hq=64, Hkv=4, D=128, blk_kv=128, causal=True` | 967.486 | 908.008 |
| `decode_1x4x` | NVFP4 | `preordered_mma` | `B=30, q=30x8, k=270336 + 29x60592, Hq=64, Hkv=4, D=128, blk_kv=128, causal=True` | 959.060 | 905.458 |

## Testing

Run the full interface-level test file:

```bash
pytest -q test_sparse_atten.py
```

Or through `make`:

```bash
make tt
make vt
```

Useful focused runs:

```bash
pytest -q test_sparse_atten.py -k test_sparse_atten
pytest -q test_sparse_atten.py -k test_sparse_page_atten
```

## Benchmark

[`test_sparse_atten.py`](./test_sparse_atten.py) is the benchmark and profile
entrypoint for this repo. It uses the public interface only.

Default sparse attention benchmark:

```bash
python test_sparse_atten.py benchmark
```

Sparse page attention benchmark:

```bash
python test_sparse_atten.py benchmark --paged --causal --page-size 64 --seqused-trim 17
```

The output is reported in TFLOPS.

Customer sink-pattern benchmark:

```bash
python test_sparse_atten.py benchmark \
  --customer-case both \
  --backend cute \
  --q2k-pattern sink \
  --warmup 5 --iters 10
```

FP8 sink-pattern benchmark runs the bf16 baseline first, then fp8, and prints
`fp8_vs_bf16_fwd_speedup`:

```bash
python test_sparse_atten.py benchmark \
  --customer-case both \
  --backend cute \
  --q2k-pattern sink \
  --dtype fp8 \
  --warmup 5 --iters 10
```

You can also use:

```bash
make bb
make bb PAGED=1 CAUSAL=1 PAGE_SIZE=64 SEQUSED_TRIM=17
```

Quantized KV prefill benchmarks for the customer ring48k sink-pattern case:

```bash
# NVFP4 KV + BF16 Q, paged KV
python test_sparse_atten.py benchmark \
  --nvfp4-kv --paged \
  --customer-case ring48k \
  --q2k-pattern sink \
  --backend cute \
  --causal \
  --dtype bf16 \
  --partial-dtype bf16 \
  --warmup 10 --iters 100

# NVFP4 KV + FP8 Q, paged KV
python test_sparse_atten.py benchmark \
  --nvfp4-kv --paged \
  --customer-case ring48k \
  --q2k-pattern sink \
  --backend cute \
  --causal \
  --dtype fp8 \
  --partial-dtype bf16 \
  --warmup 10 --iters 100

# FP8 KV + BF16 Q, paged KV
python test_sparse_atten.py benchmark \
  --paged --fp8-kv \
  --customer-case ring48k \
  --q2k-pattern sink \
  --backend cute \
  --causal \
  --partial-dtype bf16 \
  --warmup 10 --iters 100

# FP8 Q/K/V storage, FP8 QK, BF16 PV, non-paged KV
python test_sparse_atten.py benchmark \
  --mixed-fp8-qkv-pv-bf16 \
  --customer-case ring48k \
  --q2k-pattern sink \
  --backend cute \
  --causal \
  --dtype bf16 \
  --partial-dtype bf16 \
  --warmup 10 --iters 100
```

NCU profiling:

```bash
make bm
```

Or directly:

```bash
ncu --profile-from-start no --set full -o profiles/ncu/ncu_all \
  python test_sparse_atten.py benchmark --profile
```

Nsight Systems e2e profile for `build_k2q_csr -> fwd`:

```bash
nsys profile --force-overwrite=true --sample=none --cpuctxsw=none \
  --trace=cuda,nvtx \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  -o nsys_reports/sparse_e2e_example_both_sink \
  python example.py --case both --warmup 1 --iters 1 --profile
```

This captures only the measured profiler range, after warmup/compile. The NVTX
ranges from `example.py` separate CSR build and FWD.

## Sparse Page Attention Notes

Sparse page attention is intended for inference-style flows.

Important constraints:

- sparse page attention is forward-only today
- `page_size` must divide `blk_kv` or be divisible by it
- `test_sparse_atten.py benchmark --paged` currently supports `--b 1`

## Paged FP8 Decode

The paged FP8 decode wrapper provides a low-latency forward path for the
decode step (small `seqlen_q`, large `seqlen_k`).  It is exposed through
[`SparseDecodePagedAttentionWrapper`](./interface.py) and uses a
`plan() → run()` API so the schedule (split-KV chunking, per-batch work
tiles) is built once and reused across many runs with matching shape —
only `seqused_k` is allowed to change shape-compatibly.

### Supported configuration

The decode path is intentionally narrower than the dense sparse path:

| Field | Required value |
|---|---|
| Architecture | SM100 |
| Q / K / V dtype | `torch.float8_e4m3fn` |
| Output dtype | `torch.bfloat16` |
| Head dim | `D=128` |
| `qhead_per_kv` | `16` (i.e. `num_qo_heads / num_kv_heads == 16`) |
| `seqlen_q` (per request) | small (decode), tested at `Sq ∈ {1, 8}` |
| `page_size` | `128` (must equal `blk_kv`) |
| Causal | `True` |
| Batch | `1 ≤ B ≤ 1024` |

`seqused_k` may vary across batch (variable-length decode is the design
target).  The schedule includes a load-balance heuristic that triggers
when the kv-length distribution is imbalanced (max/avg ≥ 1.5) — see the
inline comment in [`build_decode_schedule.cu`](./src/sm100/fwd_decode/build_decode_schedule/build_decode_schedule.cu)
for the exact formula.

### Usage

```python
import math, torch
from interface import SparseDecodePagedAttentionWrapper

device = "cuda"
B, Sq, head_kv, qhead_per_kv, dim = 32, 8, 4, 16, 128
head_q = head_kv * qhead_per_kv
page_size = 128

# Paged KV cache with variable per-batch valid length.
seqused_k = torch.randint(1, 16384 + 1, (B,), device=device, dtype=torch.int32)
max_seqlen_k = int(seqused_k.max().item())
max_pages_per_b = (max_seqlen_k + page_size - 1) // page_size
total_pages = B * max_pages_per_b
page_table = torch.arange(
    total_pages, device=device, dtype=torch.int32
).view(B, max_pages_per_b)

q = torch.randn(B * Sq, head_q, dim, device=device,
                dtype=torch.float16).to(torch.float8_e4m3fn)
k = torch.randn(total_pages, head_kv, page_size, dim,
                device=device, dtype=torch.float16).to(torch.float8_e4m3fn)
v = torch.randn(total_pages, head_kv, page_size, dim,
                device=device, dtype=torch.float16).to(torch.float8_e4m3fn)

wrapper = SparseDecodePagedAttentionWrapper(blk_kv=page_size, causal=True)
wrapper.plan(
    page_table=page_table,
    seqused_k=seqused_k,
    seqlen_q=Sq,
    max_seqlen_k=max_seqlen_k,
    num_qo_heads=head_q,
    num_kv_heads=head_kv,
    head_dim=dim,
)

# Optional: pre-allocate the output tensor to remove per-call alloc.
out = torch.empty_like(q, dtype=torch.bfloat16)
softmax_scale = 1.0 / math.sqrt(dim)

# Run is shape-stable — replan only when batch size or max page count
# changes.  Variable seqused_k values within the planned shape are OK.
for _ in range(num_decode_steps):
    wrapper.run(q, k, v, softmax_scale=softmax_scale, out=out)
```

### Compile-time logging

CUTE DSL kernels JIT-compile on first call; subsequent calls hit the
compile cache.  To distinguish a slow first compile (a few seconds)
from a kernel hang (>30s, treated as deadlock per `CLAUDE.md`):

```bash
MINIMAX_LOG_COMPILE=1 python your_script.py
```

This routes `cute.compile` timing to the `minimax` logger at DEBUG
level.  Equivalent in code:

```python
import logging
logging.getLogger("minimax").setLevel(logging.DEBUG)
```

## Repo Layout

High-signal files:

- [`interface.py`](./interface.py): public sparse attention interface
- [`fp4_indexer_interface.py`](./fp4_indexer_interface.py): public FP4 indexer block-score interface
- [`example.py`](./example.py): customer-facing e2e CSR schedule + attention example with NVTX
- [`sparse_index_utils.py`](./sparse_index_utils.py): public CSR build wrapper and reference helpers
- [`src/sm100/prepare_k2q_csr.py`](./src/sm100/prepare_k2q_csr.py): SM100 CUDA CSR builder dispatcher
- [`src/sm100/fp4_indexer.py`](./src/sm100/fp4_indexer.py): SM100 FP4 indexer kernel classes
- [`test_sparse_atten.py`](./test_sparse_atten.py): interface-level tests, benchmark CLI, and profile entrypoint
- [`test_fp4_indexer.py`](./test_fp4_indexer.py): FP4 indexer correctness tests and benchmark CLI
- [`Makefile`](./Makefile): setup, test, benchmark, and profiling shortcuts
- [`src/sm100/fwd`](./src/sm100/fwd): forward kernels (prefill)
- [`src/sm100/fwd_decode`](./src/sm100/fwd_decode): forward kernels (paged FP8 decode)

## Known Limitations

- The first run is often compile-dominated because CuTe DSL kernels are specialized and JIT-compiled.
- `D=128` is the only documented and tested head dimension in the current contract.
- The FP4 indexer currently returns block max scores only; topK selection and
  CSR construction remain caller-owned downstream steps.
- This repo is not packaged as a pip module yet; it is used directly from the source tree.
- Paged FP8 decode currently requires `qhead_per_kv=16`, `page_size=128`, and SM100. Other configurations are not supported by the schedule kernel.
- Paged FP8 decode `batch <= 1024`. The single-CTA schedule kernel stores per-batch state in shared memory; larger batches need a multi-CTA cooperative redesign (planned but not yet implemented).
- Paged FP8 decode requires `seqused_k[b] >= seqlen_q` for every batch (i.e. context must include the q-tokens being emitted — a batched-decode invariant), AND `seqused_k[b] % page_size ∈ {0, seqlen_q, 2·seqlen_q, ..., page_size − seqlen_q}` (the last partial page must hold a whole packed-GQA q-group, which is `seqlen_q` columns). Violations are caught at `plan()` with a clear `ValueError`. The same constraint exists in FA-style packgqa kernels in principle but never fires there because FA's typical use satisfies `seqlen_k ≥ seqlen_q` (decode emits 1 token; prefill is self-attention). Tracked as a kernel-level TODO (saturate `causal_col_limit ≥ 1` in mask.py).
- Combine kernel caps `max_splits ≤ 256` (LDGSTS path's sLSE smem + per-thread register pressure). The schedule kernel auto-caps `chunk_pages ≥ ceil(max_pages / 256)` so this is never hit in auto mode, but at very large kv (> 512K) the cap reduces parallelism slightly. Tracked as a follow-up: rewrite combine with multi-pass tree reduction to remove the cap.
