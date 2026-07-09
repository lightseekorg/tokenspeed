# gluon_bf16_moe — bf16 MoE on AMD CDNA4 (MI350X / gfx950)

Two-stage unquantized **bf16** (bf16 activation / bf16 weight) fused MoE in
Gluon: bf16 in, bf16 weights, fp32 MFMA accumulate, SwiGLU (g1u1) + silu,
routed-weight fold in stage 2.

Reference shape: DeepSeek-V3 TP=8 — `E=256, D=7168, I=256, topk=8`.

## Entry point

```python
from gluon_bf16_moe import gluon_bf16_moe
# hidden [T,D] bf16, w1 [E,2I,D] bf16 (gate rows [0:I], up [I:2I]),
# w2 [E,D,I] bf16, topk_ids [T,topk] int, topk_weights [T,topk] float
y = gluon_bf16_moe(hidden, w1, w2, topk_ids, topk_weights)   # -> [T,D] bf16
```

`gluon_bf16_moe` auto-selects a **decode** path for small M and a **prefill**
path otherwise (`decode=None` → on at `num_tokens <= 16`; override with
`decode=True/False`, `split_k`, `block_m`).

## Files

| File | Role |
|---|---|
| `moe.py` | End-to-end `gluon_bf16_moe(...)`; decode/prefill dispatch. |
| `moe_align_fused.py` | Decode block-align: one sync-free single-workgroup Gluon kernel (histogram → prefix → stable-rank compare-tile → scatter). |
| `moe_align_device.py` | Prefill block-align: parallel count + prefix + atomic scatter. |
| `stage1_kernel.py` | Prefill stage 1: gate+up GEMM + SwiGLU, a16w16 double-buffered pipeline + XCD-remap. |
| `stage1_splitk_kernel.py` | Decode stage 1: small-M tile (K=128, single LDS buffer) + split-K, then a fused reduce/SwiGLU. |
| `stage2_kernel.py` | Stage 2: down GEMM + routed-weight scale → `[T,topk,D]` scratch, then a reduce over topk. Auto-dispatches to the atomic path at M=1. |
| `stage2_decode_kernel.py` | Decode stage 2 (M=1): down GEMM + fp32 `buffer_atomic_add` straight into the output (no scratch, no reduce) — ~1.22x stage-2 win at M=1. |
| `_grid.py` | XCD-aware PID remap helper. |
| `tests/` | End-to-end correctness vs an fp32 torch reference. |

## Design notes

- **MFMA idiom** (gfx950 a16w16): `AMDMFMALayout(version=4, instr_shape=[16,16,32])`,
  `DotOperandLayout(k_width=8)`, `gl.amd.cdna3.mfma`.
- **Decode (M ≤ 16):** fused single-workgroup align + split-K stage 1 (occupancy
  at tiny M) with the decode tile; stage 2 partials+reduce, or the atomic
  accumulate path at M=1.
- **Prefill (M > 16):** device align + non-split XCD-remap double-buffered
  stage 1; stage 2 partials+reduce.

## Correctness

```
HIP_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/tc-moe \
python -m gluon_bf16_moe.tests.test_bf16_moe
```

Checks end-to-end vs an fp32 torch reference and split-K consistency; all bf16
GEMM diffs are within tolerance (cos = 1.0 vs the reference).

## Prerequisites

A Gluon-capable Triton build for gfx950 and an MI350X/MI355X GPU. Add the
triton `python/` dir and this package's parent (`kernels/cdna4/moe`) to
`PYTHONPATH`.
