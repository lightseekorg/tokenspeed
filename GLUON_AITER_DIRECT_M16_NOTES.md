# Gluon AITER-Style Direct M=16 Notes

Generated: 2026-06-16

## Fastest current configuration

For the experimental Gluon AITER-style direct path at `M=16`, the fastest measured configuration is the direct/local-stage BM16 path **without `.cg` cache modifiers**:

```bash
TS_WD_DISABLE=1
TS_PREFILL_BM16=1
TS_PREFILL_STAGE1_DIRECT=1
TS_PREFILL_STAGE1_LOCAL_STAGE=1
TS_PREFILL_STAGE1_TPW12=1
TS_PREFILL_STAGE1_UNMASK_FULL=1
TS_PREFILL_STAGE2_DIRECT=1
TS_PREFILL_STAGE2_LOCAL_STAGE=1
TS_PREFILL_STAGE2_TPW12=1
TS_PREFILL_STAGE2_UNMASK_FULL=1
# leave unset or set to 0
TS_PREFILL_DIRECT_CG=0
# leave unset/0; sorted-out contract was correct but slower
TS_PREFILL_STAGE2_SORTED_OUT=0
```

The important negative result: do **not** enable `.cg` on the current Gluon direct W loads. It lowers correctly, but regresses.

## Latest rocprof kernel timings

Run dir:

```text
/home/sanketp/work/tokenspeed-perf/.rocprofv3/direct_cg_modes_20260615_224315
```

Command shape was run inside Docker container `tokenspeed-perf`, with `HIP_VISIBLE_DEVICES=7`, `KONLY=gluon`, `MS=16`, `KITERS=30`.

| `TS_PREFILL_DIRECT_CG` | stage1 direct | stage2 direct | route | topk reduce | fp8 quant | Notes |
|---|---:|---:|---:|---:|---:|---|
| `0` / unset | 94.95 us | 52.44 us | 8.12 us | 5.28 us | 4.93 us | pre-bufferops fastest baseline |
| `w` | 121.50 us | 64.11 us | 8.47 us | 5.47 us | 5.49 us | `.cg` on W only; large regression |
| `scale` | 98.29 us | 53.54 us | 8.20 us | 5.27 us | 5.02 us | `.cg` on WScale only; small regression |
| `all` / `1` | 123.09 us | 64.98 us | 7.66 us | 5.26 us | 5.32 us | `.cg` on W + WScale; large regression |

After applying AITER-style direct stage1 Gather `buffer_load` and output `buffer_store`, the current fastest no-cg result is:

Run dir:

```text
/home/sanketp/work/tokenspeed-perf/.rocprofv3/stage1_bufferops_20260615_225806
```

| config | stage1 direct | stage2 direct | route | topk reduce | fp8 quant | Notes |
|---|---:|---:|---:|---:|---:|---|
| no `.cg` + stage1 Gather/store buffer ops | **94.02 us** | **51.92 us** | 8.03 us | 5.15 us | 5.01 us | fastest current direct config |

Fastest direct two-GEMM subtotal:

```text
stage1 + stage2 = 94.02 + 51.92 = 145.94 us
```

Approx listed full device work for the direct path:

```text
stage1 + stage2 + route + topk_reduce + fp8_quant
= 94.02 + 51.92 + 8.03 + 5.15 + 5.01
= 164.13 us
```

Quick harness event timing is noisier and includes more overhead. After the stage1 Gather/store buffer-op change, a representative `ITERS=20 EVENT=1` no-cg run reported:

```text
M=16 direct no-cg full forward: 184.71 us
correctness: OK, worst cosine = 1.0000
```

## `.cg` lowering status

Correct Gluon spelling is:

```python
gl.amd.cdna4.buffer_load(..., cache=".cg")
```

Incorrect spelling:

```python
gl.amd.cdna4.buffer_load(..., cache="cg")
```

fails with:

```text
ValueError: Cache modifier cg not supported
```

With `TS_PREFILL_DIRECT_CG=all`, TTGIR contains:

```text
cacheModifier = cg
```

and AMDGCN contains buffer loads with:

```asm
sc0 nt
```

So the modifier is propagated correctly. It is just slower for the current Gluon direct kernel shape.

## Why `.cg` regresses in current Gluon direct

The current Gluon-cg stage1 is not equivalent to AITER-cg despite matching the core GEMM skeleton.

Stage1 TTGIR core similarities:

| Item | Gluon cg | AITER W4A8 stage1 |
|---|---:|---:|
| `num_warps` | 4 | 4 |
| `num_stages` | 1 | 1 |
| shared memory | 16384 | 16384 |
| `ttg.local_alloc` | 4 | 4 |
| `ttg.local_load` | 4 | 4 |
| `tt.dot_scaled` | 2 | 2 |

Remaining differences observed in TTGIR/ASM:

| Difference | Gluon cg stage1 | AITER W4A8 stage1 |
|---|---|---|
| WScale raw load layout | `tensor<128x8xi8, #linear>` | `tensor<4x256xi8, #blocked1>` |
| Gather load | generic `tt.load` | `amdg.buffer_load` |
| Bias load | generic `tt.load` | `amdg.buffer_load ... cacheModifier = cg` |
| Output store | generic `tt.store` | `amdg.buffer_store` |
| `fastMath` on `tt.dot_scaled` | `false` | `true` |
| ASM `buffer_load` count | 22 | 15 |
| ASM `global_load` count | 9 | 6 |
| ASM `s_waitcnt` count | 51 | 43 |
| ASM `v_cndmask_b32_e64` count | 41 | 0 |
| ASM total `v_cndmask` count | 47 | 20 |
| VGPR | 88 | 84 |

Conclusion: `.cg` is not a standalone win. AITER benefits from `.cg` in combination with a tighter metadata/store contract and fewer predicated/scalar-generated loads. In current Gluon direct, `.cg` mainly turns more W loads into `sc0 nt` and slows the kernel.

## Next most likely improvements

Before revisiting `.cg`, make the Gluon direct stage1/stage2 memory contract closer to AITER:

1. Done: direct stage1 output store was converted from generic `gl.store` / `tt.store` to `gl.amd.cdna4.buffer_store` / `amdg.buffer_store`.
2. Done: direct stage1 Gather load was converted from generic `gl.load` to `gl.amd.cdna4.buffer_load`.
3. Convert bias load to `gl.amd.cdna4.buffer_load`; optionally test `.cg` on bias only.
4. Rework WScale load shape/layout toward AITER's raw `4x256` blocked load instead of current `128x8` linear load.
5. Keep tracking ASM counts, especially `v_cndmask_b32_e64`, `buffer_load`, `global_load`, and `s_waitcnt`.

## Relevant source and artifacts

Main source:

```text
/home/sanketp/work/tokenspeed-perf/tokenspeed-kernel-amd/python/tokenspeed_kernel_amd/ops/moe/fused_mxfp_gfx950.py
```

Current experimental kernels:

```text
_prefill_m16_stage1_direct_kernel
_prefill_m16_stage2_direct_kernel
_prefill_m16_topk_reduce_kernel
```

TTGIR/ASM cache for Gluon `.cg` comparison:

```text
/home/sanketp/work/tokenspeed-perf/.triton_cache/direct_cg_ttgir_20260615_224049
```

AITER W4A8 default `.cg` cache used for comparison:

```text
/home/sanketp/work/vllm_aiter_a8w4_20260615_123015/triton_cache_w4a8_default_fix_222210
```

AITER no-cg negative/contrast harness:

```text
/home/sanketp/work/vllm_aiter_a8w4_20260615_123015/aiter_w4a8_no_cg_bench.py
```
