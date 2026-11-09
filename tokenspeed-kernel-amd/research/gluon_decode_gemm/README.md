# Gluon GEMMs for Qwen3.8-27B decode projections (research, gfx950)

Work in progress. Two experimental dense BF16 Gluon GEMMs, `C = A @ W.T`, for
skinny verify-step projections (M = 64..512) of Qwen3.8-27B TP1 on MI355X. They
are not registered with `tokenspeed-kernel` and are not used by serving.

- `ops/gfx950/gemm/fp16/mm_3stage.py`: `gluon_mm_a16w16_3stage_gfx950`, a
  3-stage direct-to-LDS pipeline with optional split-K.
- `ops/gfx950/gemm/fp16/mm_streamk.py`: `gluon_mm_a16w16_streamk_gfx950`, a
  persistent Stream-K variant built on the same layouts.

Shapes (N, K): qkv (14336, 5120), o (5120, 6144), gdn_in (16480, 5120),
gate_up (34816, 5120), down (5120, 17408).

## Scripts

All scripts time with cold weights (rotating copies larger than the last-level
cache) under HIP-graph replay.

| script | purpose |
|---|---|
| `bench_gl3.py` | Config sweep or fixed configs (`--cfgs`) for `--mode gl3` and `--mode sk`. Baselines: `--mode torch` (default hipBLASLt) and `--mode tuned --tuned-file CSV` (replays a TunableOp results file). |
| `stress_gemm.py` | Repeat-launch correctness stress. It catches split-K and Stream-K counter races. |
| `bench_dp_tail.py` | Prototype of "full rounds of 256 tiles + split-K tail" built from two launches of the 3-stage kernel. |
| `gl3_best_cfgs.json`, `sk_best_cfgs.json` | Best config per (shape, M), in `--cfgs` format. |

A gl3 config is `(BM, BN, BK, WM, WN, NUM_STAGES, SPLIT_K, NUM_XCDS, WARP_PIPE)`.
A Stream-K config is `(BM, BN, BK, WM, WN, NUM_STAGES, GRID, NUM_XCDS)`.

```bash
python bench_gl3.py --mode gl3 --cfgs "$(cat gl3_best_cfgs.json)" --variants base,sched --out gl3.jsonl
python bench_gl3.py --mode torch --out torch.jsonl
python stress_gemm.py gl3 gl3_best_cfgs.json 200
```

## Results so far (MI355X, µs per call)

Best Gluon config per (shape, M) against torch default and TunableOp-tuned
hipBLASLt. Each value is the median of 3 passes.

| shape | M=64 torch / tuned / Gluon | M=128 | M=256 | M=512 |
|---|---|---|---|---|
| qkv | 32.0 / 32.5 / 33.1 | 45.2 / 41.4 / 40.8 | 70.0 / 57.2 / 56.9 | 88.0 / 77.2 / 77.8 |
| o | 21.4 / 20.0 / 21.6 | 26.1 / 24.4 / 26.6 | 34.5 / 32.3 / 38.3 | 48.4 / 46.1 / 57.1 |
| gdn_in | 43.5 / 39.6 / 45.5 | 54.6 / 49.1 / 54.8 | 66.9 / 66.2 / 73.9 | 93.5 / 91.6 / 120.7 |
| gate_up | 77.3 / 66.2 / 83.9 | 93.5 / 81.0 / 93.9 | 109.8 / 111.2 / 142.4 | 187.5 / 187.8 / 222.1 |
| down | 39.9 / 40.0 / 45.3 | 46.7 / 49.3 / 61.1 | 77.5 / 65.6 / 82.2 | 112.0 / 96.9 / 127.8 |

Summed over the whole verify-step GEMM time (M = 64..512; all calls of one
step: qkv x16, o x64, gdn_in x48, gate_up x64, down x64):

- Gluon vs tuned hipBLASLt: 13-28% slower.
- Gluon vs torch default: 6% faster to 18% slower.
- The MFMA scheduling hint (`schedule_hint="mfma-schedule"`) changes this by only 0-1.3%.

### Where the gap comes from

**Occupancy is not the cause.**
- gl3 uses 124-148 KB of LDS, so 1 workgroup per CU, which is 1-2 waves per SIMD.
- Tuned hipBLASLt runs at the same occupancy.
- gl3 already uses 16-byte direct-to-LDS loads with 3-stage prefetch, has no LDS bank conflicts, and reads only 1.01-1.13x the ideal HBM bytes at M=64.

**The cause is grid quantization against the 256 CUs.**
- hipBLASLt picks non-power-of-two macro tiles, such as 64x160, 64x80 and 64x48, so the grid is one round of 206-256 workgroups.
- Power-of-two Gluon tiles give:
  - gate_up M=64/128: 136 programs, so 53% of CUs are busy
  - gdn_in: 129 programs (50%)
  - o and down with split-K 2/4: 160 programs (62%)
  - gate_up M=256: 272 programs, a second round that runs only 16 programs
- qkv tiles into 224 programs, exactly like hipBLASLt, and it is the only shape at parity.

Sweeping N at M=64, K=5120 shows this directly:

| N | config | programs | gl3 TB/s | torch TB/s |
|---|---|---|---|---|
| 32768 | 64x128x64 | 256 | 5.37 | 5.26 |
| 34816 | 64x128x64 | 272 | 3.74 | 5.20 |
| 34816 | 64x256x64 | 136 | 4.33 | 5.20 |
| 65536 | 64x256x64 | 256 | 5.69 | 6.13 |

### Split-K = 3

K = 6144 (o) allows `SPLIT_K=3`: 80 tiles x 3 = 240 programs. The sweep now
includes it. Results:

| case | before | with SPLIT_K=3 | tuned |
|---|---|---|---|
| o M=64 | 23.7 µs | 19.0 µs | 19.6 µs |
| o M=128 | 26.1 µs | 23.5 µs | 23.6 µs |

So o M=64 now beats tuned and o M=128 matches it.

### Full rounds + tail (`bench_dp_tail.py`)

| case | main kernel only, 256 programs | main + split-K tail (2 launches) | tuned | gl3 best |
|---|---|---|---|---|
| gate_up M=64 | 63.2 µs (94% of N) | 74.7 µs | 63.2 µs | 83.5 µs |
| gate_up M=256 | 96.8 µs (94% of N) | - | 108 µs | 156.6 µs |
| gdn_in M=64 | 36.8 µs (99.4% of N) | 44.8 µs | 38.2 µs | 48.9 µs |

- Once every CU has one tile, the existing main loop runs at tuned-hipBLASLt speed.
- The separate tail launch costs 8-15 µs, so the tail has to live in the same kernel.

## Next step

Write one persistent kernel with grid = 256:
- First, full data-parallel rounds of one tile per program, using the current 3-stage loop.
- Then split the leftover tiles along K across all programs (a Stream-K tail), keeping the tile order M-fastest so W is read once.
- For down (K = 17408 = 2^10 x 17), use non-uniform K ranges plus a cheaper fixup.

Expected gain: 15-35% on gate_up, gdn_in, o and down, which should bring every
shape to within about 5% of tuned hipBLASLt.
