# Gluon GEMMs for Qwen3.8-27B decode projections (research, gfx950)

Work in progress. Three experimental dense BF16 Gluon GEMMs, `C = A @ W.T`, for
skinny verify-step projections (M = 64..512) of Qwen3.8-27B TP1 on MI355X. They
are not registered with `tokenspeed-kernel` and are not used by serving.

- `ops/gfx950/gemm/fp16/mm_3stage.py`: `gluon_mm_a16w16_3stage_gfx950`, a
  3-stage direct-to-LDS pipeline with optional split-K.
- `ops/gfx950/gemm/fp16/mm_streamk.py`: `gluon_mm_a16w16_streamk_gfx950`, a
  persistent Stream-K variant built on the same layouts.
- `ops/gfx950/gemm/fp16/mm_persistent.py`: `gluon_mm_a16w16_persistent_gfx950`,
  one program per CU running full rounds of tiles plus a split-K tail in a
  single launch.

Shapes (N, K): qkv (14336, 5120), o (5120, 6144), gdn_in (16480, 5120),
gate_up (34816, 5120), down (5120, 17408).

## Scripts

All scripts time with cold weights (rotating copies larger than the last-level
cache) under HIP-graph replay.

| script | purpose |
|---|---|
| `bench_gl3.py` | Config sweep or fixed configs (`--cfgs`) for `--mode gl3`, `--mode sk` and `--mode pers`. Baselines: `--mode torch` (default hipBLASLt) and `--mode tuned --tuned-file CSV` (replays a TunableOp results file). |
| `stress_gemm.py` | Repeat-launch correctness stress. It catches split-K and Stream-K counter races. |
| `bench_dp_tail.py` | Prototype of "full rounds of 256 tiles + split-K tail" built from two launches of the 3-stage kernel. |
| `gl3_best_cfgs.json`, `sk_best_cfgs.json`, `pers_best_cfgs.json` | Best config per (shape, M), in `--cfgs` format. |

A gl3 config is `(BM, BN, BK, WM, WN, NUM_STAGES, SPLIT_K, NUM_XCDS, WARP_PIPE)`.
A Stream-K or persistent config is `(BM, BN, BK, WM, WN, NUM_STAGES, GRID, NUM_XCDS)`.
The persistent kernel needs `GRID` <= the CU count, so sweep it with `--grid 256`.

```bash
python bench_gl3.py --mode gl3 --cfgs "$(cat gl3_best_cfgs.json)" --variants base,sched --out gl3.jsonl
python bench_gl3.py --mode pers --cfgs "$(cat pers_best_cfgs.json)" --out pers.jsonl
python bench_gl3.py --mode gl3 --tiles split --shapes gate_up --rows 128 --out split.jsonl
python bench_gl3.py --mode torch --out torch.jsonl
python stress_gemm.py gl3 gl3_best_cfgs.json 200
```

## Results so far (MI355X, µs per call)

Best Gluon config per (shape, M) against torch default and TunableOp-tuned
hipBLASLt, measured on an earlier ROCm stack. Each value is the median of 3
passes. The persistent kernel's results, on the current stack, are below.

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

## Persistent kernel (`mm_persistent.py`)

One launch with grid = 256 (one program per CU). With `T` output tiles:

1. Each program with a tail item first computes one K-slice ("piece") of one of
   the `T % 256` leftover tiles. It writes an FP32 partial and bumps the
   tile's counter. The host picks the number of pieces per tail tile so that
   `tail tiles x pieces` fills the grid.
2. Each program then runs `T // 256` full rounds, one complete tile per round,
   with the 3-stage loop.
3. The pieces of each tail tile wait on its counter. Each piece then reduces a
   share of the tile from the FP32 partials and stores it as BF16. The last
   piece resets the counter.

Running the tail first lets its partials land while the full rounds run.
Because the pieces spin on each other, every program must be resident, so the
launcher refuses `GRID` larger than the CU count.

Results on MI355X, µs per call, median of 3 passes. All four methods ran in
the same session. Tuned is a TunableOp file re-tuned on this stack. Tuning hung
on down M=512, so that row uses the hipBLASLt default. M=192 and 384 are
shown only for gate_up; the per-step sums below cover every M.

| shape | M | persistent | gl3 best | torch | tuned |
|---|---|---|---|---|---|
| qkv | 64 | 34.6 | 34.0 | 33.0 | 32.5 |
| qkv | 128 | 44.5 | 43.5 | 46.3 | 42.1 |
| qkv | 256 | 60.2 | 57.6 | 63.1 | 51.9 |
| qkv | 512 | 82.5 | 79.1 | 84.9 | 73.0 |
| o | 64 | 22.2 | 19.8 | 20.0 | 19.9 |
| o | 128 | 27.9 | 25.8 | 24.7 | 24.2 |
| o | 256 | 38.9 | 37.1 | 36.0 | 30.7 |
| o | 512 | 53.9 | 56.7 | 49.5 | 43.2 |
| gdn_in | 64 | 40.8 | 48.5 | 38.5 | 38.8 |
| gdn_in | 128 | 50.2 | 58.2 | 53.0 | 44.0 |
| gdn_in | 256 | 70.3 | 80.2 | 68.0 | 63.9 |
| gdn_in | 512 | 99.0 | 114.6 | 86.3 | 87.0 |
| gate_up | 64 | 72.8 | 90.9 | 67.0 | 65.3 |
| gate_up | 128 | 93.6 | 101.5 | 83.4 | 75.8 |
| gate_up | 192 | 113.7 | 138.3 | 108.2 | 90.3 |
| gate_up | 256 | 120.0 | 157.3 | 105.8 | 107.1 |
| gate_up | 384 | 173.7 | 164.9 | 128.7 | 128.1 |
| gate_up | 512 | 194.4 | 218.6 | 170.3 | 156.0 |
| down | 64 | 47.2 | 48.5 | 41.2 | 38.6 |
| down | 128 | 63.2 | 66.0 | 48.3 | 50.3 |
| down | 256 | 83.2 | 84.5 | 69.8 | 62.9 |
| down | 512 | 119.4 | 128.4 | 97.5 | 96.9 |

Summed over one verify step (weights as above):

| M | persistent vs gl3 | faster of the two vs tuned | faster of the two vs torch |
|---|---|---|---|
| 64 | 11% faster | 11% slower | 8% slower |
| 128 | 6% faster | 19% slower | 11% slower |
| 192 | 6% faster | 22% slower | 6% slower |
| 256 | 12% faster | 18% slower | 10% slower |
| 384 | 7% slower | 18% slower | 7% slower |
| 512 | 9% faster | 22% slower | 14% slower |

What this shows:

- The persistent kernel helps where the tile count leaves the grid short:
  gate_up (8-31% faster than gl3 except at M=384) and gdn_in (14-19% faster at
  M=64, 128, 256 and 512).
- On qkv, o and down the two are within about 10% of each other, either way.
  gl3 already fills the grid there (qkv tiles into 224 programs, o and down
  use split-K), so the persistent kernel has little to gain.
- It does not reach tuned hipBLASLt. On gate_up M=64:
  - The full rounds alone take 62.3 µs (5.38 TB/s), the same rate as gl3 and
    tuned on full grids.
  - The tail adds about 8 µs whatever the piece count, stage count or BK.
  - The FP32 reduction is close to free.
  - Running the tail before the full rounds, rather than after, cut the tail
    cost from about 14 to 8 µs.
  - Pipelining loads across item boundaries, so the next tile's prologue
    overlaps the current epilogue, gave no gain and made BK=64 slower, so it
    was dropped.

  The remaining gap is the short tail pieces: each runs only a few K steps,
  so its pipeline fill and drain are not hidden, and one CU cannot pull more
  bandwidth to finish its share faster. Grid fill alone does not close it.
  hipBLASLt avoids the tail by using non-power-of-two macro tiles that tile N
  into one even round.

## Non-power-of-two tiles (split tiles)

`BLOCK_M` and `BLOCK_N` of the 3-stage and persistent kernels may now be the
sum of two powers of two, such as 160 = 128 + 32 or 192 = 128 + 64. Each part
gets its own LDS ring, and the tile runs as up to four MFMA chains (row part x
column part), one accumulator each. This gives hipBLASLt-like tiles: gate_up
M=64 with 64x160 is 218 programs instead of 136 or 272. `--tiles split` sweeps
these tiles; it keeps grids that fill at least 70% of their last round.

Best of the power-of-two and split configs per (shape, M), µs per call, median
of 3 passes, all methods in one session. "old best" is the faster of gl3 and
the persistent kernel before this change. aiter is aiter's Triton a16w16 GEMM,
measured in an earlier session. The best configs are in `gl3_best_cfgs.json`
and `pers_best_cfgs.json`.

| shape | M | tuned | torch | aiter | old best | new best | new config |
|---|---|---|---|---|---|---|---|
| qkv | 64 | 32.6 | 32.9 | 32.1 | 34.0 | 33.9 | gl3 64x64 |
| qkv | 128 | 41.5 | 46.1 | 54.6 | 43.5 | 43.5 | gl3 |
| qkv | 192 | 50.1 | 48.8 | 76.4 | 51.8 | 52.0 | gl3 |
| qkv | 256 | 52.4 | 63.4 | 72.1 | 57.6 | 57.6 | gl3 |
| qkv | 384 | 68.9 | 77.3 | 102.4 | 72.4 | 66.8 | gl3 192x128 |
| qkv | 512 | 75.3 | 84.9 | 132.2 | 79.2 | 76.9 | gl3 192x192 |
| o | 64 | 20.0 | 20.2 | 18.9 | 19.8 | 19.6 | gl3 |
| o | 128 | 24.1 | 24.8 | 29.7 | 25.8 | 24.6 | gl3 |
| o | 192 | 28.1 | 30.9 | 41.5 | 30.2 | 28.8 | gl3 96x96, split-K 2 |
| o | 256 | 30.8 | 35.9 | 39.0 | 37.2 | 32.5 | gl3 64x80 |
| o | 384 | 36.6 | 52.3 | 41.4 | 38.7 | 38.7 | gl3 |
| o | 512 | 43.3 | 49.4 | 72.2 | 53.7 | 50.3 | gl3 192x128, split-K 2 |
| gdn_in | 64 | 38.5 | 38.4 | 55.2 | 40.8 | 39.5 | gl3 |
| gdn_in | 128 | 44.2 | 52.9 | 80.3 | 49.5 | 49.8 | persistent |
| gdn_in | 192 | 57.3 | 61.8 | 108.1 | 67.3 | 58.9 | gl3 96x160 |
| gdn_in | 256 | 63.9 | 68.0 | 110.9 | 70.5 | 63.4 | gl3 128x160 |
| gdn_in | 384 | 75.0 | 91.6 | 144.9 | 81.3 | 77.3 | gl3 192x160 |
| gdn_in | 512 | 86.0 | 86.4 | 178.7 | 97.2 | 97.4 | persistent 192x192 |
| gate_up | 64 | 65.0 | 67.1 | 88.7 | 72.7 | 71.4 | gl3 64x160 |
| gate_up | 128 | 75.0 | 81.7 | 132.0 | 93.6 | 84.1 | gl3 128x160 |
| gate_up | 192 | 90.4 | 108.2 | 187.1 | 116.8 | 96.2 | gl3 192x160 |
| gate_up | 256 | 108.1 | 105.9 | 176.7 | 121.8 | 122.0 | persistent |
| gate_up | 384 | 127.5 | 129.9 | 240.8 | 164.3 | 153.1 | persistent 192x192 |
| gate_up | 512 | 154.0 | 171.3 | 299.3 | 194.1 | 195.1 | persistent |
| down | 64 | 38.6 | 40.8 | 49.0 | 47.0 | 40.6 | gl3 64x80, split-K 4 |
| down | 128 | 50.1 | 48.3 | 78.7 | 63.3 | 56.5 | gl3 128x96, split-K 4 |
| down | 192 | 53.8 | 65.0 | 105.2 | 70.3 | 65.3 | gl3 |
| down | 256 | 62.5 | 70.0 | 104.5 | 83.7 | 72.2 | gl3 |
| down | 384 | 79.0 | 82.4 | 109.8 | 93.7 | 92.0 | gl3 |
| down | 512 | 97.2 | 97.1 | 193.5 | 119.2 | 112.9 | gl3 |

Summed over one verify step (weights as above), µs:

| M | tuned | torch | aiter | old best | new best | new vs tuned | new vs old |
|---|---|---|---|---|---|---|---|
| 64 | 10279 | 10569 | 13184 | 11428 | 10858 | +5.6% | -5.0% |
| 128 | 12325 | 13184 | 20112 | 14765 | 13659 | +10.8% | -7.5% |
| 192 | 14577 | 16811 | 27771 | 17967 | 15836 | +8.6% | -11.9% |
| 256 | 16791 | 17835 | 26964 | 19839 | 18468 | +10.0% | -6.9% |
| 384 | 20262 | 22570 | 33680 | 24053 | 22938 | +13.2% | -4.6% |
| 512 | 24182 | 25844 | 46854 | 29415 | 28832 | +19.2% | -2.0% |

- Split tiles close most of the gap where the power-of-two grid was short:
  gate_up M=128/192 (10-18% faster), gdn_in M=192/256 (10-13%), down M=64
  (14%) and qkv M=384, which now beats tuned.
- Power-of-two configs run within -0.6% to +0.4% of their earlier times, so
  the split support costs nothing when unused.
- The remaining gap is at M >= 256 on gate_up, gdn_in and down, where the
  best Gluon tile still needs a second partial round or split-K.

## Stream-K register spills

With large tiles (128x256, 256x128) the Stream-K kernel spilled to scratch,
up to 1156 bytes per lane without `pin_acc` and 52-232 bytes with it. Some
configs ran 5-9x slower than with `pin_acc` (gate_up M=256: 1395 µs).

Cause, from the ISA: the shared-tile reduction ran inside the main loop. At
the end of a partial segment, the program bumped the tile's counter and, if it
arrived last, loaded and summed the FP32 partials into a second
`BLOCK_M x BLOCK_N` tile. That tile was live at the same time as the
accumulator and the operand registers of the next tile's pipeline, so the
register allocator spilled values used inside the MFMA loop.

Fix: no reduction inside the loop.

- The first segment still arrives in the loop, right after storing its
  partial. The program before it usually finishes that tile last; in the
  rare case this program arrives last, it marks the counter with
  `-(pid + 1)` and reduces the tile after the loop.
- The last segment arrives after the loop, once the accumulator and operand
  registers are dead.

ISA of the `sk_best_cfgs.json` kernels, scratch bytes per lane and
VGPR/AGPR count:

| config (shape, M) | before, `pin_acc` | after, `pin_acc` | before | after |
|---|---|---|---|---|
| 128x256 w1x4 (qkv 384) | 68, 512/256 | 0, 437/189 | 1112 | 240 |
| 256x128 w2x2 (qkv 512) | 84, 512/256 | 0, 445/197 | 1156 | 276 |
| 256x128 w4x1 (gate_up, down 256) | 68, 512/256 | 0, 437/193 | 1128 | 292 |
| 256x128 w4x2 (gate_up 192/384/512) | 232, 256/4 | 0, 252/4 | 280 | 0 |
| 128x256 w2x2 (down 384/512) | 52, 512/256 | 0, 433/185 | 1096 | 236 |
| 128x128 w2x2 (gdn_in 256, gate_up 128) | 0 | 0 | 288-300 | 0 |

Every other config had no scratch before and has none after. Without
`pin_acc`, the remaining scratch of the 128x256 and 256x128 kernels holds
loop-invariant values: they are stored before the loop and reloaded only in
the segment-end paths, not in the MFMA loop. With `pin_acc` and a ragged N
(masked weight loads), 256x128 w4x2 still keeps 12 bytes of such spills.

Stream-K µs per call with the same configs, median of 3 passes:

| shape | M | before | after | before, `pin_acc` | after, `pin_acc` |
|---|---|---|---|---|---|
| qkv | 384 | 901.3 | 116.2 | 100.3 | 99.3 |
| qkv | 512 | 1173.7 | 129.0 | 119.6 | 116.5 |
| gate_up | 128 | 106.5 | 99.6 | 98.1 | 98.8 |
| gate_up | 192 | 136.4 | 126.4 | 360.0 | 169.3 |
| gate_up | 256 | 1395.4 | 161.3 | 144.6 | 143.4 |
| gate_up | 384 | 238.1 | 226.5 | 677.8 | 306.2 |
| gate_up | 512 | 250.8 | 239.1 | 708.3 | 311.2 |
| gdn_in | 256 | 82.2 | 77.6 | 75.6 | 77.0 |
| down | 256 | 742.1 | 116.7 | 102.1 | 101.2 |
| down | 384 | 1051.4 | 174.9 | 150.8 | 149.9 |
| down | 512 | 1356.2 | 207.8 | 183.7 | 181.9 |

The other 19 (shape, M) points change by -6% to +2%, with and without
`pin_acc`. Moving both arrivals after the loop also removed
the spills, but it made small configs 3-8% slower: a program then reduces its
first tile at the end of its range instead of leaving that to the program
before it. Stream-K remains slower than gl3 and the persistent kernel at
every point.

## aiter's BF16 GEMM solutions

aiter's `tuned_gemm` picks a BF16 a16w16 GEMM per shape from tuned CSVs.
Its solutions are torch (hipBLASLt), hipBLASLt by solution index, skinny
(small-M HIP kernels), asm, Triton, FlyDSL and opus. None is CK.

- Lookup key: (arch, CU count, padded M, N, K, dtypes, bias, ...). With no
  tuned row, a preshuffled weight goes to asm, a skinny shape to the skinny
  kernels, and anything else to torch.
- aiter's tuned CSV for Qwen3.8-27B (`qwen3_8_27b_bf16_tuned_gemm.csv`) has
  rows only for M = 16, 32, 48, 64 and 1024-8192. At M=64 it picks torch for
  qkv, gdn_in, gate_up and down and Triton for o. M = 128-512 falls back to
  torch.
- FlyDSL's a16w16 kernel is a pipelined MFMA GEMM with BM/BN tiles that need
  not be powers of two (BM 48/80/96, BN 80/96), 2-9 stages, split-K, and a
  tuner that prunes configs by tile-quantization efficiency and occupancy.
  Its decode variant streams weights with non-temporal loads and staggers K
  starts across workgroups; it is tuned only for two other models' shapes at
  M <= 128.

FlyDSL a16w16 measured with this README's timing (cold weights, HIP-graph
replay), best of up to 200 configs per point from its tuner space (`k_waves`
1, no `group_m`, at most 5 stages, 160-1024 programs), µs per call:

| shape | M=64 | 128 | 192 | 256 | 384 | 512 |
|---|---|---|---|---|---|---|
| qkv | 32.0 | 38.9 | 55.3 | 56.9 | 75.7 | 83.7 |
| o | 15.5 | 21.0 | 25.4 | 28.7 | 37.1 | 45.1 |
| gdn_in | 38.1 | 53.9 | 70.2 | 77.8 | 94.4 | 135.0 |
| gate_up | 77.0 | 100.3 | 131.3 | 154.7 | 198.0 | 212.5 |
| down | 36.8 | 48.8 | 67.6 | 78.2 | 86.8 | 121.1 |

- FlyDSL beats tuned hipBLASLt on o at M = 64-256 (o M=64: 15.5 vs 20.0 µs,
  split-K 6 on 64x128 tiles), on qkv and down at M = 64 and 128, and on
  gdn_in at M=64. Elsewhere it is slower, and at M >= 192 on gate_up and
  gdn_in it is 9-39% slower than the best Gluon kernel.
- Summed over a verify step it is 3% (M=64) to 33% (M=512) slower than
  tuned. The faster of Gluon and FlyDSL per shape is 0.2% faster than tuned
  at M=64 and 4-18% slower at M = 128-512.
- What Gluon can borrow: deeper split-K on o at small M (FlyDSL's o wins use
  split-K 3-6 with 128-wide N tiles; gl3's best o configs use split-K 1-3),
  and its tuner's pruning of configs by tile-quantization efficiency.

## Next step

- Close the M >= 256 gap on gate_up, gdn_in and down: wider split tiles with
  `BLOCK_K` 128, or split tiles in a split-K tail of the persistent kernel.
- Sweep gl3 split-K 4-6 on o at M <= 256, where FlyDSL is 12-21% faster.
- Try FlyDSL's decode tricks (non-temporal weight loads, staggered K starts)
  in gl3.
