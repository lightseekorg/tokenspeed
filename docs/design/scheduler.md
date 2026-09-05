# Scheduler: admission, retraction and recovery

The C++ scheduler (`tokenspeed-scheduler/`) decides, once per round, what each
engine does next. This document covers one axis of that: **at what granularity
KV capacity is admitted**, what happens when admission fails, and how a
retracted request comes back — per engine role.

Companion documents: `event-loop.md` (the control/data plane split that
consumes these plans), `cache-concepts.md` (the vocabulary below —
prefix granularity, cache groups, LCM blocks).

## 1. Admission is per chunk

A prompt is prefilled in chunks bounded by `max_scheduled_tokens`
(`--chunked-prefill-size`). Capacity is admitted **for the chunk being
scheduled, never for the whole prompt**: `schedulePrefill` /
`schedulePrefillFirstChunk` build one `GroupDemand` per cache group sized by
this chunk's tokens, and the coordinator either grants the pages or the
request stays put.

Two adjustments ride on top of the raw chunk size:

**Alignment.** `AlignPrefillChunk` shortens a chunk so it ends on a prefix-page
boundary (or on a promotion boundary), because a page is the unit of prefix
caching — a chunk ending mid-page would leave a partial page that can never be
matched. A chunk that *completes* the prompt is exempt: there is no next chunk
to align for.

**Decode reserve.** The chunk that completes the prompt also reserves
`decode_input_tokens` (`completes_prefill ? reserve : 0`), so the request's
first decode step is guaranteed a slot. Intermediate chunks reserve nothing —
they are not about to decode.

### 1.1 Head-of-line: an incomplete prefill holds the queue

`holdsHeadOfLine` breaks the candidate loop after scheduling a chunk of a
prefill that is still incomplete. Nothing behind it is scheduled that round.

The reason is that per-chunk admission gives an in-progress prefill **no claim
on the capacity it still needs**. Let a newcomer take pages while a prompt is
half prefilled, and the half-prefilled one may never assemble its remaining
chunks — it holds pages, makes no progress, and eventually has to be retracted,
throwing away work already done.

The cost is one round of queue latency; the alternative risks a retraction
cycle. One full chunk already saturates the GPU, so interleaving a newcomer
into the same round buys no throughput to offset that risk.

**Holding is a property of the *incomplete*, decided at plan-build time.** A
chunk that reaches the end of its prompt moves the request to `PrefillDone`
inside the same plan build (`SchedulePrefillEvent` picks the successor state
by whether the window reaches `PrefillSize()`), so a prompt scheduled in full
never holds the line. A round can therefore carry **several prefills that
complete this round plus at most one truncated one — and the truncated one is
necessarily last**, because `holdsHeadOfLine` seals the phase the moment it
appears. With 6K of budget left and prompts of 2K / 1K / 10K waiting, the
round schedules the 2K and the 1K in full and the first 3K chunk of the 10K
prompt, then stops. The same rule makes the finishing round of a chunked
prefill cheap: its final chunk releases the line *within* the round, so the
prompts queued behind it start in that round, not the next.

**Decodes are not part of the line.** In mixed mode the decode batch is built
before the prefill phases and takes its token budget first (§3.3), so a round
is *all decodes + the completing prefills + at most one truncated prefill*.
Outside mixed mode, prefill and decode never share a round at all — decodes
get the round only when no prefill scheduled — so head-of-line only ever
orders prefills against each other, never a decode behind a prefill.

### 1.2 The one lookahead: the state-checkpoint tail

The single place capacity is reserved beyond the current chunk. For mamba /
state-checkpoint architectures, a prompt's **final** state checkpoint must land
on an aligned boundary; without a reservation the pages for that tail may not
be available when the prompt reaches it, and the checkpoint can never assemble.

So `FinalAlignedTailTokens` splits the last stretch into *body* + *aligned
tail*, and the body's admission also banks the tail's pages
(`state_checkpoint_tail_reserved`). The next round spends the reservation —
`schedulePrefill` reads-and-clears the flag on entry, asserts the tail
completes in one round, and skips re-shaping the sparse state demand.

Consequences worth knowing:

- **A banked tail is exempt from head-of-line.** Its remaining pages are
  already held, so nobody behind it can starve it, and the queue may move on.
- **In hybrid architectures the tail is banked in every cache group, not just
  the state group.** The mechanism is indirect: `admission_reserve` is
  broadcast to all groups, and `setSnapshotStatePrefillReserve` then narrows
  *only* the state group back to the tail. The history/KV group silently keeps
  `tail + decode`. Correct, but implicit — worth reading twice before changing
  either function.
- **Bound to state groups.** `shouldSplitFinalStateCheckpoint` requires
  `HasMambaStateGroup()`, so a pure-KV model never banks a tail and is
  head-of-line pinned until its final chunk. Extending the mechanism there is
  an open optimization (it would lower `MaxSingleRequestTokens`, so it needs
  its own capacity baseline).

- **Remote (D-role) admissions bank one growth block per state group.** A
  D-role admission has no local tail to split; the peer lands only the endpoint
  snapshot, so `schedulePrefillFirstChunk` (remote branch) reserves
  `block_granularity` extra tokens in every `LatestSnapshot` group, i.e. one
  more block for every prompt length. Without it each landed request needs a
  fresh **empty** parent per state group at its first boundary, and a full pool
  deadlocks silently because residents are retraction-exempt. Capacity planning
  counts 2 state blocks per admitted request, as for a local prefill.

### 1.3 What bounds a single request

`MaxSingleRequestTokens` is a **startup** bound computed by binary search over
`singleRequestLcmBlocksRequired`: the largest prompt whose worst-case working
set — aligned body + tail, decode reserve, overlap-depth protection, and for
sparse local recovery the two state parents — fits the pool. It is not a live
check against currently free capacity; a prompt within the bound can still fail
admission right now and simply waits.

## 2. Retraction: when admission fails

`maybeRetractForCapacity` fires when **no prefill made progress** this round
(`PlanBuild::NoPrefillProgress()`: nothing was admitted and no resident prefill
advanced a chunk) and an admission failed for capacity. Decode steps do not
count as progress — they release no capacity, so a round of pure decode leaves
a stalled prefill exactly as stuck as an empty one.

**Retract-and-grant, in one round.** The retraction serves a specific request —
the first candidate whose admission failed for capacity
(`AdmissionFeedback::capacity_blocker`). Victims are retracted and the blocked
admission is **retried in the same plan build**, looping (retract → retry →
retract) until it fits or the victims run out. The freed capacity therefore
reaches the request it was freed for within the round; there is never a free
page waiting for whoever asks first next round, which is what previously
required a cross-round capacity barrier. Two edges of the loop:

- **The victim may BE the blocker** (a resident request blocked on its own
  next page is the preferred victim). It comes back through the readmission
  phase; the grant is redirected to the first waiting prompt instead —
  granting the pages straight back to the victim's own readmission is the
  loop the grant exists to break.
- **A grant that cannot legally join its round's batch** (a D-role recovery
  chunk beside an already-built decode batch; a fused prefill beside decodes
  outside mixed mode) still retracts one victim, and the next round's phase
  order (§3) tries the blocker before any other claim.

**The victim's pages are released — and grantable — immediately**, even though
its L2 snapshot has not been copied yet. The runtime enqueues the D2H snapshot
copy on the **forward thread's stream** ahead of everything else the plan does
to those pages — the same stream carries the zeroing, fences the forwards, and
gates a granted remote prefill's RDMA trigger (see `DeviceHandle.execute` and
`event-loop.md`) — so the copy reads the old bytes whatever the scheduler does
with the pages. Store tickets consequently pin only their Host destinations;
the ack's one job is publishing the Host entry.

**Per-victim quiescence, not global.** A request whose own forward is still
out must not be retracted — its result would land on pages it no longer owns —
and one whose pages a PD transfer still pins must not be either. Both are
checked on the chosen victim; if it is not quiescent, retraction waits for it
rather than sacrificing a worse-ranked request. The one remaining global gate
is an in-flight load-back: it is writing pages its readmission owns, and the
victim policy cannot see that write. In-flight *stores* gate nothing anymore.

The forward-out check is a count on `fsm::ForwardState`, incremented when a
forward is scheduled and cleared when its result lands. It lives on the base
class rather than on the states that consume a *token*, because a forward is
out against the **pages**, and every forward state owns pages. An intermediate
prefill chunk produces no token but does write KV, so it reports back with an
empty `ExtendResult`: the arrival is the point, not the payload. Work this
engine does not perform — the peer's decode on a P node, the peer's prefill on
a D node — is not counted here; those are fenced by the PD transfer ack.

**Victim choice** (`chooseVictim`, shared by D and fused): an incomplete
prefill first — it has produced no output a client is reading, and its
computed chunks survive as a prefix for the retry — largest first, freeing the
most at once; then decode work by most newly releasable LCM blocks and fewest
tokens — the most capacity for the least lost work. Exempt in both tiers: a
request whose reserve already covers its whole generation
(`Request::ReserveCoversGeneration`) — retracting it frees exactly what its
readmission must take back, pure thrash.

The P role never retracts: `buildPrefillWorkerPlan` simply does not call
`maybeRetractForCapacity` (the only two call sites are the D and fused
grammars). See 3.1 for why.

## 3. Per role: explicit phases

Each role's plan builder is a sequence of **phases** over one stable
candidate order: **submission order** (`requests_` is a vector — the FIFO —
with a side index by id for lookups). It is identical on every rank because
the mirrored schedulers receive identical submission batches, so no sort is
needed for determinism, and within a phase older requests win — FIFO is the
fairness policy, not an accident of key order. There is no priority ladder:
what used to be a rank in a ladder is now the position of a phase in its
builder, readable top to bottom. A round schedules each request at most once
(`PlanBuild::Scheduled`), whatever states it moves through while the phases
run.

A pass's mutable state is split in two on a layer boundary. `PlanBuild` — the
output plan, the batch under construction, budgets, and the composition flags —
is held by the role grammars alone, and every operation enters the batch
through one gate, `pushOperation`, where budget and flag accounting live. The
per-request admission layer (`admit`, `schedulePrefill*`, `scheduleDecode`)
sees none of that: it receives only the output plan (to record fresh pages to
zero) and an `AdmissionFeedback` (`admission_failed`, `capacity_blocker`), so
it can report outcomes but never compose the batch.

### 3.1 P — prefill worker

**Phases:** completed prompts out on `plan.remote_decode` (their pages stay
pinned until the transfer finishes, so releasing them outranks feeding more
prompt work), then the shared local-prefill phases
(`scheduleLocalPrefillWork`): resident chunks, then new prompts.

**Retraction: none.** A P node's pressure valve is the transfer itself — pages
are pinned (`pd_transfer_pins_`) until the peer acknowledges, then released
wholesale. Retracting a prompt whose KV is mid-transfer would strand the decode
side.

**Recovery: n/a.** The readmission path is unreachable on this role.

### 3.2 D — decode worker

**Phases:**

1. Local recovery, alone in its batch — a resident recovery chunk if one is
   mid-prompt, else the one readmission this round may start (§4).
2. The decode batch (`scheduleDecodeBatch`) — every PrefillDone first decode
   and Decoding step; decodes consume no token budget on this role.
3. At most **one** remote admission — the whole prompt at once (the peer
   prefills it), riding `plan.remote_prefill` **beside** the decode batch: it
   consumes no token budget and no batch slot, so there is nothing to defer
   for. Capped at one per round because each reserves an entire prompt's
   pages; a queue's worth in one round would drain the pool before any KV
   arrives. Head-of-line (1.1) does not apply — there is no mid-way.
4. `maybeRetractForCapacity` (§2), whose grant also rides beside the batch
   (a remote admission) or joins it (a blocked decode).

The old "one of exactly three shapes per round" grammar is gone: a decode
batch and a remote admission coexist routinely, and only a local recovery
chunk still claims a round to itself (its load-back's layerwise streaming and
the recovery prefill are batch-global machinery).

**Retraction and recovery:** victims are chosen by the shared rule in §2 —
normally decode work (resident requests are decoding prompts the peer
prefilled), with a mid-prompt local recovery chunk as the one possible
prefill-tier victim. The victim's KV is written back to L2 (best-effort) and
it enters `fsm::Retracted`; recovery re-prefills locally, loading the
snapshot back (`LoadBackBatch`). A D-role victim recovers through this
ordered path even when there is no host cache to snapshot into (from
scratch), because the role has no other way back.

### 3.3 Fused — one engine, everything local

**Phases, mixed mode** (`enable_mixed_prefill_decode`): decodes first — a
client is streaming them, and a long prefill chunk must not starve them of
token budget — then the shared local-prefill phases (readmission first, since
it holds an L2 snapshot other admissions could evict; then resident chunks;
then new prompts) spend what remains, then `maybeRetractForCapacity`. The
decode batch leaves `state_prefill_reserve` (one state-checkpoint page of
budget) untouched when a mamba prefill is pending, since that prefill cannot
advance in sub-page chunks.

**Phases, non-mixed:** the prefill phases run first and alone; decodes get
the round only when no prefill scheduled. No state reserve is needed —
scheduling order is the capacity priority.

**Retraction:** the shared victim rule (§2): incomplete prefills first, then
decode work. Whether the victim's KV is stored depends on the host cache:
with one, the retraction becomes an L2 snapshot the readmission loads back;
without one the request re-prefills from scratch
(`has_recoverable_snapshot = false`) and competes for admission like a
newcomer rather than queueing behind other readmissions.

## 4. The recovery protocol

What remains of cross-request recovery bookkeeping is **one integer** on the
scheduler (`next_retraction_epoch_`); everything else is derived from the
`fsm::Retracted` states themselves. The former `RetractionRecovery` class —
barrier, recovering-pin, priority overrides — is gone; §2's same-round grant
and the rules below absorb each of its jobs.

**Readmission order** (`nextReadmission`) is derived, not stored. Each
retraction stamps a monotonic `retraction_epoch` and a `resumes_generation`
flag onto the `fsm::Retracted` state; among this round's candidates holding a
recoverable snapshot, victims with generated output first (they resume a
generation a client is already reading), then oldest epoch. The flag is
`Request::HasGeneratedOutput()` — token count above the submitted prompt
size — rather than "was the victim decoding": a victim taken mid-RECOVERY is
Prefilling again, but its generated tokens still exist (an earlier
retraction rebased them into its prefill window), and its standing survives.
A store-less fused retraction is not in this ordering at all — it has no L2
pages to load back, so it re-prefills through the ordinary admission path
(`admitsLikeNewPrompt`). There is no queue to keep in step with the FSM: a
request that finishes or aborts while retracted simply stops qualifying,
with no bookkeeping to prune.

**A readmission that does not fit, waits.** Its failed admission never
triggers retraction (it is never recorded as the capacity blocker): when the
readmission needs a victim, the two simply do not fit together, and swapping
them — a writeback, a load-back and a re-prefill per swap — is pure thrash.
The resident request keeps running and its completion frees the space. This
replaces the old `recovering_` head-of-line pin, and unlike escalation-bounded
ping-pong it makes the evict-each-other cycle structurally impossible. Nor
does a waiting readmission stall anyone else: decodes run regardless, and
only new-prompt admission is sealed behind it (a newcomer taking the pages it
waits for would starve it).

**Escalating headroom** is what keeps a request from being retracted forever.
Being retracted means the previous admission was still too optimistic, so
each retraction raises the decode headroom the next admission must secure:

```
Request::AdmissionHeadroom(safe_steps)
    = min(max_new_tokens, safe_steps * (1 + retraction_count))
```

with `safe_steps = 4096` — note the `1 +`: a *fresh* admission already
prepays one window (see `schedulePrefillFirstChunk`), so for prompts with
`max_new_tokens <= 4096` the reserve covers the whole generation up front and
retraction never touches them. Capped by the generation budget the request
could ever use, so after a couple of retractions it holds enough room to run
to completion — at which point `ReserveCoversGeneration` exempts it from the
victim policy and it **cannot be retracted again**. This is a per-request
adaptive backoff: it penalises only the request whose admission proved
over-optimistic, and never makes anyone else wait.

## 5. Invariants a change must preserve

- Admission never grants pages for tokens beyond the chunk being scheduled,
  except the state-checkpoint tail (1.2), which is banked for exactly one round
  and asserted against nesting, and the admission headroom (4).
- An incomplete local prefill is not overtaken (1.1) — unless its tail is
  already banked. Decodes are never hostage to it: they consume no fresh
  capacity within their reserve, so they keep running beside a stalled
  prefill.
- Retraction fires only when no prefill progressed and an admission failed
  (2). The chosen victim must be quiescent — no forward of its own in flight,
  no PD pin on its pages — and an in-flight load-back defers all retraction;
  in-flight stores defer nothing.
- Freed capacity is granted to the request it was freed for in the same plan
  build whenever the round's grammar admits the grant (2); the write-back →
  zero → load → forward order on the forward thread's stream is what makes
  the immediate release safe, and changing `DeviceHandle.execute`'s ordering
  breaks it.
- Only computed tokens are published as a prefix — `retractVictim` reads the
  window of an incomplete prefill rather than its whole token count.
- At most one readmission is in progress per role, by phase construction; a
  readmission that fails admission waits and never triggers retraction (4).
- A request whose reserve covers its remaining generation is never a victim
  (2); with the fresh-admission prepay this bounds retraction to requests
  whose `max_new_tokens` exceeds one safe-step window (or is undeclared).
