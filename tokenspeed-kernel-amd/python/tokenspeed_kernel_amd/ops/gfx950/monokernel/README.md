# gfx950 monokernel building blocks

A decode monokernel runs a decoder layer as one persistent launch: one
workgroup stays resident on every CU, and workgroups hand work to each other
through memory instead of kernel boundaries. This directory holds gfx950
pieces such kernels share. They carry no model knowledge. The design
invariants are in `docs/design/monokernel.md`.

### gfx950 persistent launch helpers

`persistent/`: the launch entry, polls, joins and fail-stop deadlines
(`sync.py`), per-wave helpers (`wave.py`), the clock (`timing.py`), uncached
memory for cross-XCD words (`arena.py`), a host copy of the failure record
(`mirror.py`), and the LLVM intrinsics and inline asm they rest on
(`_intrinsics.py`, `intrinsics.ll`).

#### Contract

- The kernel is a plain launch of `GRID` (256) workgroups that are all
  resident, built with `persistent_compile_options()` (it links the
  helpers' LLVM intrinsics).
- Entry is two calls. `enter_issue(epochs, claims)` names the workgroup's
  role, `cta = (pid // 8) * 8 + xcd` from the hardware XCD id, with no
  memory access; the role may then issue its first weight or KV loads.
  `enter_check(entry, errors, arena, deadline_ticks, label, on_expiry)`
  returns this wave's `Launch` (`cta`, `xcd`, `slot`, 64-bit `epoch`,
  never-zero 32-bit `tag`, `copy = epoch & 1` for two-copy areas). Roles use
  `cta`, never the program id: dispatch starts on an XCD that depends on the
  kernels before.
- Every launch advances the workgroup's epoch line (one per program id) and
  claims its role's line (`ENTRY_WORDS` each, zeroed together once); a
  second workgroup on a role stops with `SITE_CLAIM`.
- Every wait is an equality on the launch tag and returns whether it was
  seen, per wave; the caller then runs `on_expiry(launch, site, detail)`,
  which must not return. A wait gives up at `deadline_ticks` (10 ns ticks)
  or when a peer stored its abort marker in this GPU's error words.
- Waits by hand-off class:
  - `poll_until(word, expected, launch, LOAD)`: one int32 word. `LOAD` is
    `"sc1"` (stored on this GPU), `".cv"` (stored by another GPU) or
    `"smem"` (scalar loads past the scalar cache: words stored on this XCD
    or in uncached memory). `poll_until_all` waits for one word per lane.
  - `poll_tags16(line, tag, launch, GLC, BARRIER)`: a 64 B line of 16 tags
    with scalar loads. `BARRIER` ends with a workgroup barrier (other waves
    read what the tags guard) or only with `wave_acquire` (this wave reads
    alone, as in per-wave loops).
  - `store_pairs` / `poll_pairs`: (value, tag) pairs, one 8-byte store each
    (`tag << 32 | value`), so no drain and no flag.
  - Payload plus flag: the payload stores, `release_drain()` in every
    storing wave, `gl.barrier()`, then the flag.
- `grid_join(joins, launch, on_expiry, SET, detail)` waits for every
  workgroup; prefer the data-driven waits above.
- `fail_stop(launch, site, detail)` records `(site, cta, detail, epoch,
  label)` in the error words and in the host page of a `FailureMirror` (if
  set: after a queue error device memory cannot be read), then traps.
  `record_failure` writes the record without the trap. Any subset of waves
  may call them; they have no workgroup barrier.
- `UncachedBuffer` allocates zeroed uncached device memory for words that
  scalar loads poll across XCDs.
- `wave.py` states per-wave facts Gluon cannot express yet: per-wave scalar
  values (`wave_scalar`), an opaque index that keeps LDS reads in a loop, a
  value ordered after a load (`after`, to publish once the wave's stores
  have completed), an issue-order point, and an LDS exchange among the
  waves of a workgroup without a barrier.

#### Algorithm

- One wave's vector memory operations complete in issue order, so a vector
  poll waits behind every load the wave issued before it. Scalar loads and
  atomics count separately (`lgkmcnt`): the `smem` polls never wait behind
  the role's weight loads.
- Entry: every wave reads the epoch word with a scalar `glc` load (`epoch`
  stays in scalar registers), the waves join at a barrier, and a relaxed
  atomic store writes epoch + 1 without waiting for the role's loads. The
  workgroup then exchanges `tag << 32 | pid` into its claim word; an old
  value with this launch's tag is a duplicate. The exchange's result
  returns after the role's loads issued before it, so a role issues before
  the entry only the weights it uses first.
- Polls run per wave: one owner lane and a one-instruction vote, or scalar
  loads. Every 64 retries a poll reads the clock (the first check starts the
  deadline, so a wait met sooner reads no clock) and the abort word, with a
  scalar load in scalar polls. After a vector poll, one workgroup barrier is
  the acquire point for the loads that follow.
- Tags repeat after 2**32 - 1 launches, so every tagged slot a later launch
  may poll is rewritten or zeroed in every launch.
- The join is a two-level top-bit toggle: the 32 workgroups of an XCD add to
  their XCD's line so that the 32nd arrival flips bit 31, and that arrival
  adds to the grid line the same way. Consecutive joins alternate two line
  sets, so no join needs a reset.
- Nothing invalidates or writes back L2: consumers on other XCDs read
  write-through payloads with `.cv` loads, or poll uncached memory.

### gfx950 MXFP4 MoE stream

`moe_stream.py`: the MoE phase of a persistent decode launch (EP1 slice, up
to 16 rows), as device functions `prefetch_moe` / `run_moe` and a standalone
launch for tests.

#### Contract

- Inputs: the routed ids `(rows, top_k)`, the MXFP4 activations and their
  swizzled scales, and `moe_stream_weights(experts, scale_lanes=...)`: the
  processed MoE layer's gdot128 weights and swizzled scales, read in place
  (what the EP1 direct decode reads). `scale_lanes` adds a lane-major copy
  of the gate/up scales, 32 MiB per layer of 256 local experts.
- Every down unit's FP32 `[16, 128]` result goes to a sink function with
  its per-token routes and columns; the caller's sink may push it to another
  GPU, and the standalone launch stores `out[route, :]` in BF16. The sink
  gets one 16-column strip per call, or with `DN_LINES` the unit's 4 strips
  at once. It must stay within the calling wave.
- Bitwise equal to the direct decode: the same per-column MFMA chain,
  SwiGLU, 32-value MXFP4 mids and down chain.
- Plan fields (constexpr): `RING`, `QUAD_MAP`, `DN_ORDER`, `DN_LINES`,
  `ADDR_SOFFSET`, `TAG_POLL`, `SCALE_LANES`, `TAIL_NUM` / `TAIL_DEN`.

#### Algorithm

- Every token is an A row (MFMA M = 16), so each touched expert's weights
  are read once. A gate/up strip is (expert, 16 columns, all K); a quad of 4
  waves runs the 4 strips of one 32-value mids group. A down unit is
  (expert, 64 output columns) per wave.
- Each wave keeps `RING` 1 KB weight chunks in flight and refills the slot
  it consumed `RING` steps ahead, into the next strip or unit when the
  current one ends, so the weight stream does not drain between items. Item
  metadata comes from LDS and readiness from scalar polls: a vector load the
  wave waits on would wait for every weight load before it.
- The quad exchanges its SwiGLU values through LDS (no barrier), each wave
  quantizes 4 rows of the group, stores them per route, and publishes its
  tag after a later load of its own has been waited for.
- `DN_LINES` changes which output column each MFMA column of a down unit
  computes, so that a lane's results across the unit's 4 strips are two
  whole 16-byte BF16 lines (8 adjacent columns each) instead of 4 runs of 4.
  The sink then writes 2 wide stores per lane. Strips 16 columns apart still
  share their scale dwords, and each column's sum is unchanged.
- The plan is closed-form from the ids in every workgroup: whole experts per
  XCD, the `E % 8` leftovers split by group. 16 tail workgroups per XCD
  (slots 0 and 16-30, which also run post-MoE work) take a `TAIL_NUM /
  TAIL_DEN` share of gate/up groups only; the other 16 take the rest, then
  the down units, earliest-ready expert first.

### gfx950 deferred residual norms

`deferred_norm.py` folds a residual boundary's post-norm and the next
pre-norm into their consumers.

#### Contract

- Producers publish four FP32 sums per row and column block, with
  `b' = b * post`: `sum b'^2`, `B = sum (alpha h)(beta b')`,
  `A = sum (alpha h)^2`, `C = sum (beta b')^2`.
- `deferred_norm_terms_gfx950` gives a producer the four terms per element,
  `[..., 4]` in that order, for it to sum over its own columns.
- `deferred_norm_scales_gfx950` takes them as `[rows, producers, 4]` with the
  producers across one wave's lanes and returns `inv1`, `inv2` per row.
- `deferred_norm_rows_gfx950` returns `h' = fma(beta, b post inv1, alpha h)`
  and `x = h' inv2 pre` elementwise; `post` / `pre` are weight plus offset
  rounded to BF16, as FP32 or BF16 operands.

#### Algorithm

`inv1 = rsqrt(sum b'^2 / W + eps_post)` and `inv2 = rsqrt((A + 2 inv1 B +
inv1^2 C) / W + eps_pre)`, the regular residual boundary's operations in its
order, so a consumer builds its own columns of `x` without waiting for the
normalized row.
