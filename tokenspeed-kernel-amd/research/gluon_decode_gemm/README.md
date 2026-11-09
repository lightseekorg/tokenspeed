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

## Next step

Non-power-of-two `BLOCK_N` (such as 80, 160, 192) for the dense 3-stage kernel,
so the grid is one even round without a tail. This needs MFMA layouts and
loads that handle an N tile that is not a power of two.
