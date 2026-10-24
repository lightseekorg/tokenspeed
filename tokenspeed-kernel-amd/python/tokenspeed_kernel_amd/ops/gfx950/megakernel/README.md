# gfx950 decode megakernels

A decode megakernel runs one decoder layer's decode step as one persistent
launch. This directory holds the gfx950 parts that every model's megakernel
uses. Each model has its stages and layer kernels in its own subdirectory.

Read these documents with this one:

- [`docs/design/megakernel.md`](../../../../../../docs/design/megakernel.md):
  the invariants.
- [`tokenspeed_kernel/ops/megakernel/README.md`](../../../../../../tokenspeed-kernel/python/tokenspeed_kernel/ops/megakernel/README.md):
  the steps to add a model, the tests and the performance workflow.
- `tokenspeed-kernel/test/amd/ops/megakernel/test_megakernel_example_gfx950.py`:
  the smallest complete megakernel, of two stages. Copy its shape for a new
  model.

## Files

| File | Contents |
|---|---|
| `plan.py` | Host side: `Stage`, `Plan`, `PlanKey`, `placement`, `launch_tag`. A plan describes one launch. |
| `schedule.py` | Device side: the plan's sizes, and the tasks of each stage that this CTA runs. |
| `mailbox.py` | The tagged-pair protocol: `Mailbox`, `put`, the polls, `pre_poll`. |
| `mx.py` | The MXFP4-weight, MXFP4-activation GEMV, the MXFP4 quantizer and the puts of MXFP4 and BF16 values. |
| `bf16.py` | The BF16-weight, BF16-activation GEMV and the BF16 row loads. |
| `weights.py` | Host side: the weight and scale layouts that `mx.py` and `bf16.py` read. |
| `trace.py` | Timestamps for each wave and stage inside a launch. |

| Model directory | Model |
|---|---|
| [`qwen4_exp/`](qwen4_exp/README.md) | Qwen3.8-Flash-Next: Gated DeltaNet and QSA attention layers. |

## Launch contract

- The host builds a `plan.Plan` and passes `Plan.key` as the compile-time
  `PLAN`. A kernel compiles once for each plan, thus once for each row count
  and chain length.
- The launch has `Plan.grid` CTAs, and all of them must be resident at the
  same time. Thus each CU holds one CTA, the kernel has no spills, and no
  other kernel runs on the GPU during the launch.
- `scratch` is int64, at least `Plan.scratch_pairs` pairs. Every launch uses
  the same buffer, and nothing clears it.
- `step` is an int32 `[1]` device counter. The caller changes it before each
  step, inside the CUDA graph. The launch tag is
  `step * LAYER_SLOTS + layer + 1`, so `layer < LAYER_SLOTS` (64).
- `trace_buf` is `None`, or int64 `[grid, trace.EVENTS, num_warps]`. When it
  is `None`, the trace points compile to nothing.
- Values that change per step are runtime arguments: positions, sequence
  lengths, page tables, table widths and slots. The row count and the chain
  length are part of the plan.

## Plans

A `Plan` lists the stages in execution order and the mailbox regions.

- **Stages.** `Stage(name, tasks, base, reads, writes)`. Task `t` runs on CTA
  `(base + t) % grid`. A stage with more tasks than CTAs runs in rounds: task
  `t + grid` runs on the CTA of task `t`, after it. Each CTA runs its tasks
  of each stage in plan order.
- **The deadlock rule.** A stage reads only regions that an earlier stage
  writes. `Plan` checks this rule when you construct it. The rule needs every
  CTA to be resident.
- **Regions.** `(name, pairs)` gives the pairs of one row. Each row has its
  own copy of every region, `Plan.row_pairs` pairs after the previous row.
  Each region starts at a multiple of 32 pairs (`REGION_ALIGN_PAIRS`).
- **Rows and chain.** `rows` is the number of token rows in the launch.
  `chain` is the number of consecutive rows that are one request's tokens: 1
  for decode, `k + 1` for a speculative verify of `k` draft tokens. Row `r`
  is token `r % chain` of request `r // chain`.
- **Slices for tests.** `Plan.only(names)` keeps the named stages. The
  regions that they read before they write become `external`, and the test
  fills them.
- **The key.** `Plan.key` is the plan's canonical JSON. It is a `str`, so
  Triton can hash and serialize it as a compile-time argument. It also
  carries the `Plan` object for the compile-time helpers.
- **Placement.** A stage module's `stages(bases, ...)` takes the base of each
  stage from a table that the layer gives. `placement(bases, names)` checks
  that the table names exactly those stages, in that order.

## Device schedule

`schedule.py` reads the plan at compile time. Only the task indices depend on
the CTA's program id.

| Function | Result |
|---|---|
| `stage_task(PLAN, name)` | `(task, active)` of a stage with at most `grid` tasks. |
| `stage_rounds(PLAN, name)` | `(task, rounds)`: this CTA runs tasks `task + i * grid` for `i < rounds`. |
| `row_task(PLAN, task, PER_ROW)` | `(row, task, tok)` of a stage with `PER_ROW` tasks per row. `tok` is the row's token index in its request. |
| `rows`, `chain`, `grid`, `row_pairs`, `region`, `has_stage` | The plan's sizes and region offsets. |
| `stage_event(PLAN, name, point)` | The trace slot of a stage's trace point. |
| `head_sleep(PLAN, first, second, units)` | Delays the CTAs that have no task in the first two stages. |

## Mailboxes

A stage gives a 32-bit value to another CTA as an aligned 8-byte pair
`(tag << 32) | payload`. One 64-bit store writes the pair. The consumer loads
the pair again until the tag is the launch's tag. Thus the pair is its own
ready flag. The protocol needs no flag, no fence and no clear. A pair from an
earlier launch has a different tag.

Memory rules on gfx950 (coarse-grained VRAM):

- Every put and every poll load is agent scope (`sc1`). The vL1D serves a
  plain or `sc0` load, which can then spin on a stale line forever.
- The 64-bit store and load are single-copy atomic. Thus a pair never holds
  an old payload with a new tag.
- A poll traps (`s_trap 2`) after `MAX_RETRIES` retries (at least about
  8 s). A deadlock then stops the process with a GPU error, not a hang.
- A CTA never reads another CTA's plain global store from the same launch.
  For example, the current token's K and V come from mailboxes, not from the
  KV cache that another CTA writes in the same launch.
- Many waves that spin on `sc1` loads can starve the producers' stores, and
  the grid can then livelock. Before a poll that all waves of many CTAs make,
  `pre_poll` (or `pre_poll_spread`) lets wave 0 spin on one hint pair for each
  producer. The full poll then usually succeeds on its first load.

`Mailbox(scratch, tag, plan, trace_table)` is the view that a stage gets:

| Method | Use |
|---|---|
| `put(region, index, payload, mask)` | Publishes 32-bit payloads at pairs `index` of a region. |
| `poll(region, index)` | Waits for the pairs and returns the payloads. |
| `poll_many(regions, indices)`, `poll_pairs(offsets)` | Polls several index tensors, four in one round trip. `pairs(region, index)` gives the offsets for `poll_pairs`, so one tensor can mix regions. |
| `row(r)`, `row_offset(row)` | Address the copies of another row. |
| `pre_poll(region, hints)`, `pre_poll_spread(region, N)` | Let wave 0 spin on hint pairs before a wide poll. |
| `stamp(stage, point)` | Records a trace point. |

`mx.load_mxfp4_rows` polls the MXFP4 rows of a GEMV input in one round trip.

## The GEMV

A decode GEMV `y[n] = sum_k x[k] W[n, k]` reads much more weight than it
computes. Thus a stage spreads a weight over many CTAs, a few 16-row blocks
on each CTA. It splits K over waves, so that each CTA keeps enough loads in
flight.

- **The MFMA.** One `v_mfma_scale_f32_16x16x128_f8f6f4` multiplies a 16x128
  MXFP4 A tile by a 128x16 MXFP4 B tile of 16 weight rows. MFMA row `m` holds
  activation row `m % RP`. Thus a launch of several rows reads each weight
  once. Each row's result depends only on its own A row, so it is bit for bit
  the one-row result.
- **The K split.** The waves own the batch dimension of a rank-3 MFMA layout.
  Wave `s` runs k-iterations `s, s + S, s + 2 S, ...`, so the waves stream
  adjacent 1 KiB blocks at each moment. A fixed-order sum over the batch
  dimension adds the waves' results.
- **Issue, then compute.** `load_weight_tiles` issues every weight and scale
  load of a task and does not wait. `gemv` multiplies the activation rows by
  the loaded tiles and returns `[rows, N]`. `gemv_zero`, `gemv_row` and
  `gemv_result` serve a stage that reads other weights for each row (the
  routed experts).
- **Activations.** The A operand is in LDS as `[KP, 64]` packed E2M1 bytes
  and `[KP, 4]` UE8M0 scales, k-iteration major. `KP` is a power of two, at least
  the iteration count.
  Several rows end each row in padding (`PaddedSharedLayout`): the MFMA
  lanes of a multi-row GEMV read several rows at once, and rows whose
  starts share an LDS bank conflict (2.7 times slower at 4 rows).
- **Quantization.** `quantize_mxfp4` follows TokenSpeed's activation
  quantizer (Quark's "even" scale rule) on values rounded to BF16.
- **Puts.** `put_mxfp4_rows`, `put_mxfp4`, `put_bf16_rows` and `put_bf16`
  publish values. The row versions do not publish the padding rows past the
  launch's rows. `pack_bf16_pairs` and `bf16_half` pack and unpack BF16
  pairs.

Weight layouts (`weights.py` builds them on the host):

- **Tiles: N16 order.** In each 16-row block, each 64-byte K chunk (one
  k-iteration) is a 1 KiB block in MFMA lane order. Lane `g * 16 + n` reads
  row `n`, K bytes `16 g .. 16 g + 15`, as one `dwordx4`. These are the bytes
  of TokenSpeed's dense `preshuffle_mxfp4_weight` and of its N16 MoE banks.
  Thus a megakernel reads these weights in place.
- **`"k4"` scales.** `[N/16][ceil(KI/4)][64][4]` UE8M0. Dword `j` of lane
  `g * 16 + n` holds the scales of k-iterations `4 j .. 4 j + 3`. One dword
  load serves four MFMAs.
- **`"n32"` scales.** The 256-byte cells of TokenSpeed's N16 MoE banks, each
  cell for two k-iterations (`n32_scales`).
  With `"n32h"`, the two K-split waves of a block take contiguous halves of
  K, so one dword load serves two k-iterations (`load_mxfp4_rows` with
  `HALVES` puts the activation rows in that order).

## Trace

A kernel profiler sees one launch. The trace shows the time inside the
launch: when each wave of each CTA starts a stage (`START`), has its inputs
(`READY`), finishes the computation (`COMPUTED`) and ends the stage (`END`).

- Stamps are `s_memrealtime` values, a 100 MHz clock (`TICK_NS` = 10) that
  all CUs share. Thus CTAs on different XCDs align on one timeline.
- Stamps go to an LDS table (`ds_write_b64`, which `lgkmcnt` counts).
  `vmcnt` counts stores as well as loads. A global store per stamp would thus
  make later load waits also wait for the stamp.
- `trace.allocate` makes the table, and `trace.flush` copies it to
  `trace_buf` once, at the end of the kernel. A 0 means that the wave did not
  reach the point.

## Writing a stage

A stage function takes `(mailbox, task, bundles..., region names, STAGE)`.
Its body has this order:

1. Poll the inputs that are already published.
2. Issue the weight loads.
3. Poll the other inputs.
4. Compute.
5. Put the outputs.

Rules:

- **A poll waits for every load before it.** `vmcnt` completes in order, so
  a poll also waits for the weight loads that its CTA issued before the
  poll. Poll an input that is already published before the weight loads.
  Issue the weight loads before a poll that waits.
- **Tiled weights skip L1.** A launch reads each weight once. The tiled
  loads (`bf16.load_weight_tiles`, `mx.load_weight_tiles`) use `.cg`, which
  streams faster. Row-major loads that read part of a cache line per row
  keep L1.
- **Few pairs per thread.** A polled pair holds two VGPRs for its address
  and two for its value until the poll returns. Poll about 12 to 30 pairs
  per thread. Poll wide inputs of many rows in groups of rows.
- **Throttle wide polls.** Use `pre_poll` before a poll that all waves of
  many CTAs make.
- **Issue loads without waits.** The issue sequence is the stream: a wait in
  it holds back every later load. Three things add such waits:
  - A loaded value that the code uses at once. A `"k4"` scale dword stays
    whole until the MFMA takes its byte (`mx._scale_operand`).
  - A base for each lane. A base that a reduction computes (an expert from
    the routing) is not known to be wave-uniform. Every buffer load then
    becomes a loop over the lanes' bases. Mark the base with
    `_scheduling.wave_uniform_i32`.
  - A branch around loads. Two load sequences that join need copies at the
    join, and the copies wait for the loads. Select the base with a scalar
    `gl.where` and keep one sequence.
- **Issue early on shared CTAs.** If a CTA runs two stages, split the first
  stage into `<x>_issue` (the loads that need no mailbox input) and
  `<x>_finish` (the rest). A combined helper then issues the loads of both
  stages first. The early loads must not poll a region whose producer runs
  later on the same CTA.
- **Two kinds of stage.**
  - A stage that streams weights serves every row in each task, so it reads
    each weight once. Keep the rows as the leading tensor dimension (indices
    from `Mailbox.row_offset`). One poll, one reduction and one put then
    serve all rows.
  - A stage whose work is one request's state (a recurrence, attention, a
    conv) runs one task per row and task: `row, task, tok =
    schedule.row_task(PLAN, task, PER_ROW)`, then the one-row code on
    `mb.row(row)` and `bundle.row(row)`. In a verify, it carries the state
    from token `tok` to `tok + 1`. It reads the values of the launch's earlier
    tokens from their mailbox rows, not from the cache.
- **Operands go in bundles.** Each stage module has an `@aggregate` bundle
  for each operand group. The kernel takes one tuple for each bundle and
  builds the bundle at its top with `Bundle(*tuple)`.
- **Stamp the trace points** `START`, `READY` (after the polls), `COMPUTED`
  and `END`.

## Gluon notes

- `@aggregate` makes `__init__` from the annotations. A field that is `None`
  or a `str` becomes a constexpr. A tuple kernel argument can hold `None`
  members.
- JIT tuples cannot hold types. A JIT function's constexpr return value
  becomes a tensor when you assign or concatenate it. A
  `constexpr_function` does not receive the semantic object, and the JIT
  dependency scanner rejects plain Python helpers. Compute types and
  layouts in a `constexpr_function` from plain values, for example
  `offsets.type` (see `mailbox._poll_types`).
- `gl.inline_asm` takes tuples of arguments and result types. It expands a
  constraint group for each element of a tuple. Clobbers can go on the
  constraint of the last group of one element. Thus one `poll` serves any
  number of index tensors.
- A module-level int that JIT code uses must be a `gl.constexpr`. A name that
  is a constexpr in one place and a runtime value in another place of the
  same function does not compile.
- The backend schedule depends on the order of address arithmetic. A change
  of that order alone moved a layer by 0.2 to 0.4 us. Compare the timing of
  each change, also of a refactor.

## ISA checklist

Read the kernel's AMDGCN from the Triton cache. Look for these items:

- `sc1` on every poll load and every put.
- No spills: no `scratch_store` or `scratch_load`, and a private segment size
  of 0.
- No `s_load_dwordx16` whose values stay live across stages.
- The expected count of `v_mfma_scale_f32_16x16x128_f8f6f4`.
- The weight loads before the first poll of the stage.
- No `v_readfirstlane` loops around buffer loads (a base for each lane).

## Tests

Tests are in `tokenspeed-kernel/test/amd/ops/megakernel/`:

| Test | Covers |
|---|---|
| `test_plan.py` | The plan checks, the region offsets of each row, `Plan.only`, `Plan.key` and `placement`. |
| `test_mailbox_gfx950.py` | Stage rounds, and polls that read only the current launch's pairs. |
| `test_mx_gfx950.py` | The MXFP4 quantizer, the GEMV against a dequantized reference, rows against one-row launches, and the GEMV on TokenSpeed's MoE banks. |
| `test_megakernel_example_gfx950.py` | The two-stage example kernel. |

`megakernel_testing.py` holds the shared helpers: the MXFP references, and
`fill` and `payloads` to write and read the mailbox regions.
