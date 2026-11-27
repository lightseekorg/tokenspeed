# KDA kernels

Each physical checkpoint holds only a dense recurrent base. Each resident
request slot holds every layer's FP32 correction U, normalized K and
multiplicative decay records plus one INT32 accepted count that all its layers
share: the number of records accepted since the dense base of the request's
pages. The logical endpoint is the base followed by
`S = S * decay + outer(U, K)` in record order. The base follows
`--mamba-ssm-dtype` (FP32 or BF16). Accumulation and history remain FP32.
Materialization casts the result to the dense base dtype.

Each slot also keeps, per layer, a ring of the request's raw pre-convolution
QKV rows (`KdaPreconvRing`): R = T + 3 rows for a verify window T, in the
convolution state's dtype. Positions count from the slot's last seed, not from
the request's sequence position: seeding fills rows 0..2 with the three inputs
before the next window, so a reseeded slot's first candidate goes to row 3
whatever the request's length, and each later input takes the next row mod R.
The three rows before a request's ring head are the convolution's history and
its next window goes from the head on; the reads and writes of one verify
never overlap. Every (slot, row, channel) of a ring is its own memory:
`KdaPreconvRing` rejects expanded or overlapping row and slot strides, while
strided views such as one layer of a `[slots, layers, R, channels]` ring stay
valid. Beside the count, the slot's INT32
ring state is 0 while the ring holds no rows (the page's convolution state is
current), else the ring head plus one. The two form the replay state, INT32
`[slots, 2]`, and every reset clears both. Between materializations a seeded
ring, not the page, holds the request's convolution window: decode rounds
write no page's convolution state, and a request's end, a retraction fold
and a prefill continuation write the ring's last three inputs into the page
before anything else reads it. At T = 4 on a Kimi-K3 TP8 rank
(69 layers, 4608 channels) the ring is 7 rows of 0.6064 MiB: 4.245 MiB a slot.

## CuTe prefill (`cutedsl_kda`)

The backend selects FlashInfer's ordinary or persistent CuTe KDA scan, reading
token-strided Q/K/V directly from convolution output. Persistent prefill admits
BF16 gates, head counts divisible by eight and gate bounds in `[-5, 0)`, and is
selected when `2 * requests * heads > SM count`; smaller workloads retain the
ordinary decomposed scan. FP32 initial and final states are
separate; the runtime retains its LCM gather/writeback and checkpoint policy.
Live device boundaries build the scan plan for both eager execution and
capacity-based CUDA graphs. Inputs beyond the live total follow the runtime's
zero-padding contract: native scans may load a full tile at the tail.
Persistent planning distributes 32-token chunks
over resident workers, with at most one exact FP32 handoff per request/head
chain. Its metadata launch also resets the handoff flags on every replay.
Scratch belongs to each invocation or its graph
pool; only compiled code is cached. The FlashInfer adapter owns the pinned
native launch ABI, avoiding the public wrapper's in-place state updates and
persistent GPU scratch.

## Verify and publication

`KdaRetainedRecords` supplies the layer's pre-convolution ring, the retained
history, packed request boundaries, and generation/position stamps. For each
batch row, `live_state_indices` selects the private physical page and
`record_slots` selects the record storage, the ring rows and the count
(`history.length`, the count column of the replay state). Record slot zero is
valid. Shared source pages hold dense checkpoints; the slot of a request
starting from one holds no accepted records. Each live destination may alias
only its own source.

Every verify producer (the CuTe fused step and window kernels, the Triton
split convolution producer, the AMD Gluon kernels) reads a request's three
convolution taps from ring rows `(head - 3 + i) mod R` and writes candidate t's
raw row to `(head + t) mod R`; `head` is the prepare's per-row snapshot
(`KdaPreconvRing.heads`), loaded beside the record slot, so the tap loads wait
on one round trip as the page-tap loads did. R is a compile-time value, so a
compare and an add or subtract replace the modulo. The row index stays below
R, so its INT32 math cannot wrap. A runtime row stride multiplies an INT64
row offset; the CuTe kernels compile the pitch of a ring whose rows are
contiguous (3 P channels) where that measured faster, and keep that offset,
below R * 3 P, in INT32. A request takes part as in the retained prepare: its
source and live pages lie in the pool and its record slot is resident.
Padding rows read a zero history and write no row, and touch no slot's rows
(slot 0 and empty rings included): the Triton producer and the one-token
kernel predicate their tap loads and row stores off, and the window kernel,
where predicated loads cost 10 to 35 us a round, points a padding row's tap
loads at the layer's convolution weights, which no launch writes during a
verify, and discards the values. A split verify runs
the producer (which takes the live pages for that predicate) on the ring and
then takes records without it (`preconv=None`). The producer runs one
program per request, channel block and token tile (one tile for windows up to
16 tokens at the default channel block), which loads its taps and inputs before it stores any ring or output
row. The one-token kernel stores its ring rows with an L2 `evict_last` hint to
ask L2 to retain them longer across surrounding state traffic. The ring must not share
storage with the verify's projection inputs or outputs; the Triton producer,
the fused CuTe entry points and the AMD launchers all reject such a ring
before any launch.

With a verify window T and a history capacity C, a count H overflows when it
lies outside [0, C - T]. Verify applies `eff` records to the live base: H, or
0 for an overflowing H, whose records the prepare (below) folded into the live
base. Each verify kernel derives `eff` itself, with one scalar select on its
compile-time capacity and window. It writes candidate U/K/decay into the
request slot after `eff` records and never changes the base or the count.
Rejected candidate suffixes need no clearing.

Commit is one launch per verify round. It publishes each request's count
and ring state and, only for a request that ended in the forward, folds its
records and writes the ring's window into its live page; a request that keeps
decoding has no page written (the verify wrote candidate t to ring row
`(head + t) mod R`, and the ring keeps the window). Its UINT64 `[layers, 13]`
descriptors contain the ring and conv pointers, the page count, the ring row
and three conv strides, the v-major recurrent-state pointer and its page, head,
value and key strides, then the ring's record-slot stride; they carry no raw
key/gate pointers. The replay state is not a layer's: the commit takes it as an
INT32 `[slots, 2]` tensor, and `record_slots` selects each request's. Every
program reads the round's `eff` and ring head from the prepare's snapshot
(`replay_snapshot`, INT32 `[requests, 2]`), never the replay state the
launch publishes. The launch's first `cdiv(batch, 256)` programs publish: each
writes the count of up to 256 carried rows once, to `eff` plus the
acceptance, and their ring states, to the row after the last accepted
candidate plus one, with no scratch and no cross-program reduction. A row is
carried when its source and live pages lie in the pool in a State group a
launched layer lists, its record slot is resident and its window is
non-empty; a row no layer carries (graph padding, a masked overlap row, a
row paged only in groups no launched layer lists) keeps its count. The other
`layers x min(heads, 16)` programs are end programs, one per layer and head
split, so the grid does not grow with the batch times the channel blocks.
Each loads the stop flags of up to 256 rows at once and visits only the rows
whose request ended (whose slot is set in `request_stopped`): a request that
keeps decoding costs its lane of those loads. For an ended row a layer
carries, the program checks every accepted stamp, then folds the records,
through this round's acceptance, into its heads of the live page and writes
its slice of the channels of the ring's last three accepted inputs, rows
`(head + take - 3 + i) mod R`, as the page's convolution taps; the publish
programs clear the row's count and ring state instead of advancing them. End
programs read only the snapshot, stamps, records and ring, and publish
programs write only the replay state, so no program of the launch reads what
another writes. Other commits read no record and write no dense state.
Stamp-ring capacity is independent of retained-history capacity.
Padding rows (a page at or below 0, or one past the checkpoint pool) leave all
pages, rings and replay states untouched.

## NVIDIA verification

The registry keeps the same split, one-token and multi-token producer choices:

| Kernel | Selection |
| --- | --- |
| `vllm_triton_kda_recoverssm_verify` | FP16/BF16 activations and generic head geometry |
| `cute_dsl_kda_recoverssm_verify` | SM100/103, BF16 activations, key dimensions aligned to 32 and at most 128, value dimensions aligned to 32, windows up to 16 |
| `cute_dsl_kda_fused_paged_verify_step` | SM100/103, BF16, 128-wide heads/gate rank, one-token windows |
| `cute_dsl_kda_fused_paged_verify_window` | The same fused producers for windows 2–16 and batches up to 16 |

`kda_fused_paged_verify_fuses_producers` resolves the fixed window and padded
batch geometry before the caller prepares projections. Split verification uses
the shared packed-boundary convolution producer and registry-selected core.
All choices store the same persistent FP32 normalized K/decay/U representation.
The CuTe fused kernels require unit-stride physical page indices, query
boundaries and correction channels. Record slot indices may be strided. Other strides use split producers and the stride-aware retained
verify core.

The one-token fused kernel projects the accepted history onto the current
query and key. For 1–16 accepted records, it scales the query and key by
the cumulative decay before multiplying them by the dense base. It adds
each retained U vector weighted by its key's two dot products. Its CTA has
eight warps, so two share an SM: four compute the convolution; the other
four compute the gate, from weights held as packed BF16 pairs, and then
apply the accepted decay factors to the retained keys. The checkpoint and
packed history have independent bulk-copy completion barriers. Strided
history uses its tensor strides. Once the gate and convolution are ready,
four warps project the checkpoint while the other four project the retained
records. Each history warp computes both products for up to four keys. It
combines the even and odd U contributions in FP32. Empty histories use only
the checkpoint products.
Longer histories replay in record order into a private FP32 state tile.
Count publication and end materialization use the common retained commit
ABI. A request that ended in the forward has its records folded and its ring
window written into its live page by the commit; if a layer that carries it
rejects its commit (stale stamps), that layer's first end program raises the
row's `request_ends` entry from `REQUEST_ENDED` to
`REQUEST_ENDED_INCOMPLETE` (a compare-and-swap) instead, so the control plane
never publishes a page that lacks accepted tokens or inputs.

## Retained preparation

Verify reads each request's live page as its dense base and never writes a page,
a count or a ring state. Before a verify forward, the caller runs
`resolve_kda_batched_retained_prepare` once for every layer, in one launch,
which reads the count H and the ring state and never writes them. A request
whose ring state is 0 gets ring rows 0..2 of every layer it takes part in
seeded from its source page's three convolution taps, the history the verify
would otherwise have read there; its verify starts at ring head 3. The launch
writes each row's snapshot: the count its verify applies (`eff`) and its ring
head. Records overflow and page moves leave the ring alone. In each layer, a request whose H
overflows (lies outside [0, C - T], with T the verify's window,
`draft_token_num`) gets the source state with records [0, H) applied in its
live page, in place where the pages did not move; an H past C applies none.
Otherwise a request whose live page differs from its source in the layer's
State group gets the source state copied there, with no record applied: its H
records stay valid on top of the copy, so the verify's `eff = H` holds
whether or not its pages moved. Every layer decides alike from the one shared
H, so no group's move affects another's layers. A request with a window
shorter than T overflows as soon as a full one would. Padding rows keep their
pages. The launch takes the commit's per-layer descriptors, which must share
the page count, the layer-to-group map and `[groups, requests]` source and
live page tables. The fold splits a
request's heads across programs only when few requests and layers would leave
SMs idle. Both vendors share this Triton kernel. Running the prepare twice in
one round would copy the unchanged source over the fold.

The core CuTe window solver keeps main's chunked equations, compensated gate
prefix sums, value tiles and clustered producer sharing. Bounded windows with
`window * abs(lower_bound) <= 48` form pairwise products through factored rows;
other windows sum intermediate gates directly so a large early gate cannot
erase later decay. Runtime counts, arena strides and stamp-ring capacity remain
runtime values in compiled launch interfaces. The window and split cores load
the live page's base tile with one bulk copy. On 128-row value tiles with room
for retained records, threads start loading the accepted records with the CTA
and skip the L2 prefetch. On 64-row value tiles with packed records, the fused
window kernel instead bulk-copies each record's key, decay and its tile's U
into shared memory when the grid holds at most one CTA per SM and the CTA's
footprint with the copy region fits the device's opt-in shared memory per
block. Tiles that use neither early loads nor the bulk copy -- including 32-row
tiles, unpacked records, 64-row tiles on larger grids where the copy region
could cost a CTA per SM, and 64-row tiles whose copy region exceeds the
shared-memory limit -- prefetch the accepted records into L2 instead. Like the
one-token kernel, both cores project the history instead of replaying it:
the base rows are scaled by the records' cumulative decay, each
record's key, weighted by the later decays, becomes a row that a separate pass
multiplies by the window's decayed keys and queries while the base copy lands,
and each value channel adds its records' U weighted by those rows' products.
Requests without records skip all of it.

## Output norm and prefill

`kda_fused_paged_verify` accepts explicit `output_norm: KdaOutputNorm | None`.
The CuTe fused producers apply gated RMSNorm in their epilogue using the BF16
core output and return `[tokens, heads * head_dim]`; other retained kernels
apply the same norm after attention. A missing norm returns ordinary attention
output. The norm's FP32 math and rounding order follow `rmsnorm_gated_sigmoid`.

Prefill keeps the capacity-aware scan, native state layout, prepared CuTe plans,
and cuDNN graph paths. Each layer's retained source is gathered and
reconstructed in FP32 from the rows' counts before its scan, by
`kda_materialize_retained_state`, which also writes a seeded ring's last
three inputs over the taps the gather copied into the live page, before the
prefill convolution reads them (an unseeded slot's page keeps the gathered
taps). Every layer reads the same counts and ring states, so
`kda_reset_retained_length` clears them once in a launch ordered after the
last KDA layer's gather; it skips padding rows (a live page at or below 0, or
at or past the pool's `page_count`) as the gather does. The next verify
reseeds the ring from the page the prefill wrote. Final/checkpoint dense
stores use the selected base dtype. Before a retraction stores its image or
anything reads the endpoint it published, `kda_materialize_retained_pages`
folds each victim's records into the live page the scheduler lists, layer by
layer, and writes a seeded ring's last three inputs as that page's
convolution state; `kda_reset_retained_length` then clears the slots' counts
and ring states once. Records overflow (the prepare's fold) and page moves
leave the ring as it is, so these, with a request's end in the commit, are
the only points that write a page's convolution state from the ring.

Retained numerical tests cover accepted-prefix rounds, boundary materialization,
the fold and ring window at a request's end, the single commit launch over
one and several stop-flag blocks and strided stop flags, retraction (into the
listed page, whatever the history's live pages) and prefill materialization
with and without a seeded ring, private/shared pages, stale stamps, graph metadata
refresh, strided arenas, FP32/BF16 base, norm fusion, ring seeding, ring
wrap, ring slots past 2**31 elements and scalar compile reuse. AMD uses the same retained
contract through its registered Gluon kernels and the shared prepare.

The `kda_fused_paged_verify` and `kda_retained_commit` benchmarks accept
`history_capacity` (default 16, at least `draft_token_num`).
`kda_fused_paged_verify` also accepts `history_length` (default 0, up to
`history_capacity - draft_token_num`), so the candidate window fits beside
that prefix. Each measurement uses the same accepted records.
