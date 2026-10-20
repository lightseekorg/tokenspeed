# Scheduler: admission, retraction and restore

The C++ scheduler (`tokenspeed-scheduler/`) decides, once per round, what each
engine does next. This document covers one axis of that: **at what granularity
KV capacity is admitted**, what happens when admission fails, and how a
retracted request comes back — per engine role.

Companion documents: `event-loop.md` (the control/data plane split that
consumes these plans), `cache-concepts.md` (the vocabulary below —
prefix granularity, cache groups, LCM blocks).

## 1. Admission is per chunk

The scheduler schedules a prompt's prefill in chunks bounded by
`max_scheduled_tokens` (`--chunked-prefill-size`). It admits capacity **for
the chunk being
scheduled, never for the whole prompt**: `schedulePrefill` /
`schedulePrefillFirstChunk` build one `GroupDemand` per cache group sized by
this chunk's tokens, and the coordinator either grants the pages or the
request stays put. Alongside the demands, one `RequestProgress` per request
carries what it computed since its previous admission — the prefix pages
completed and its computed-token count — which the coordinator publishes
and reclaims inside the same `Admit` (`AdvanceRequestProgress` in
`scheduler/operations/progress.h` is the one place that hashes those pages
and builds it; see [cache-concepts](cache-concepts.md#the-coordinator-layer-csrccachecoordinator)
for why publication rides with admission).

Decode-role remote admission covers the complete unmatched prompt in one
landing window. Local prefill chunk limits and junction cuts do not shorten
it: the peer transfers the full prompt's KV past the hit and its prompt-end
state (unless this node already holds that exact checkpoint, §3.2). The window
must reach `PrefillSize()` so the first decode consumes its reserved input
width. A truncated remote window would leave that reserve at zero and delay
later cache growth even though the device has committed the full prompt.

**L3 is fetched before admission, never by it.** An admission loads only
Host entries that exist (`PrefixProbe::host`), so nothing it loads can miss and
the Host-to-Device leg keeps its layer-wise overlap with the first chunk. The
objects the L3 shadow knows beyond the Host hit (`PrefixProbe::storage`, the
same matchers run over "Host-cached or registered") are handled by
`CacheCoordinator::PlanPrefetch` first: it decides before it touches the Host
pool — sizing the first `SchedulerConfig::l3_prefetch_min_pages` pages (the
explicit threshold; required with L3) against the free slots, the evictable
entries and the empty LCM parents, so a prefetch that cannot reach the
threshold evicts nothing and the request admits normally and computes those
pages — and only then acquires a Host block for every storage-tier row, prefix
page by prefix page, stopping at the first page the Host pool cannot take
whole (the pages before it stay). With a plan, `schedulePrefillFirstChunk` emits one
`PrefetchOperation` (`Cache.PrefetchOp`: rows in prefix-page order with their
`page_indices`, one op per request) and the request moves to
**`fsm::Prefetching`**: it pins only the Host blocks being filled — no Device
pages, no request-pool row — is skipped by admission without holding the head
of line (later `Submitted` prompts are admitted past it), is never a victim,
and can be aborted (the op's own pins keep the blocks until its ACK). The
runtime fetches in order, stops at the first failure or its timeout, and
acknowledges once with `cache::PrefetchDone{op_id, landed_pages}`
(replica-converged); the scheduler publishes the first `landed_pages` pages as
Host entries, frees the rest, forgets the unlanded keys from the storage
shadow, and returns the request to `Submitted` at its original queue position,
holding the published entries pinned until its admission acquires them. A
`landed_pages` outside `[0, num_pages]` is a runtime bug, not a scheduler
invariant: the ACK is logged and ignored and the op stays in flight. The
next admission is an ordinary Host hit. The cost: an L3 hit waits one fetch
before it is admitted, holding no Device page meanwhile. The D role probes the
Device alone (`ProbeDecodeDevicePrefix`) and never prefetches.

**What the probe may claim.** Before the first chunk, `matchPrefixAtAdmission`
probes the prefix cache for the prompt's leading pages and for a published
tail inside the page after them (`CacheCoordinator::ProbePromptPrefix`, see
[cache-concepts](cache-concepts.md#the-coordinator-layer-csrccachecoordinator)):
the hit boundary `T` is a token count, and `extend_prefix_lens` carries it
as is. The probe is bounded
in tokens, and the bound is the minimum of two rules: the configured replay
tail (`prefix_replay_tokens`, at least the final prompt token, which is always
recomputed to produce logits) and the request's own
`RequestSpec::max_cached_prefix_tokens` (default `INT32_MAX`, no bound); a
hit may end exactly at the bound. The
per-request bound exists for prompt (input) logprobs: a request that returns
them from position `s` needs logits for every position at or after `s`, and a
cached position has none, so the runtime admits it with the bound set to `s`
and the positions `>= s` are recomputed as ordinary prefill input whatever the
cache holds. The bound limits the probe itself, not a later trim, so excluded
hit pages are never claimed and the recomputed suffix lands on private pages.
Only a `Submitted` prompt is probed: a retracted request is never re-probed;
it comes back by restoring its image (§4), so no position whose logits were
produced is ever matched against another request's pages or recomputed. The
decode role of a disaggregated deployment never computes prompt rows (the
prefill node returns the logprobs), so the runtime leaves its bound at the
default.

Two adjustments ride on top of the raw chunk size. Both are pure token
arithmetic kept out of the planner: how a chunk is cut lives in
`scheduler/operations/prefill_chunk.h` (`PrefillChunkTokens` is the one
entry both prefill paths call), what each group demands for it in
`scheduler/operations/group_demands.h`.

**Junction cut.** A chunk is never shortened to a page boundary: a prompt
end publishes its partial page as a tail (§1.2), so a chunk ending mid-page
costs nothing. The one cut is the *junction* of a hybrid hit: when the KV
groups' hit `T` reaches past the longest state checkpoint `T_s`
(`AdmissionResult::junction_tokens`, kept in `CacheProgress::junction_tokens`
until a chunk passes it), the first chunk after `T_s` ends exactly at `T`, so
the natural checkpoint it leaves coincides with KV already cached and the next
prompt sharing that prefix resumes at `T` in every group. A chunk that ends at
the junction publishes like a prompt end (`PublicationPoint::kOwnerContinues`:
KV tail and state) and moves off the published blocks; every other chunk end
publishes only its completed full pages. The move-off needs fresh blocks the
completing chunk did not reserve, so it is best effort: when the pool cannot
fund the copies the admission replans without the tail and the owner keeps
its partial block private, publishing nothing at that point beyond its full
pages (`cache-concepts.md`, copy-in).

**Reserve.** What an admission holds beyond the chunk it computes is stated
once per round (`PrefillReserve`: decode width, whether the round completes
the prompt, prompt headroom, the sliding-window growth recycling cannot
fund) and turned into each group's page demand by
`ReservePrefillDemands` — the only writer of `GroupDemand::reserve_tokens`.
It picks the rule by the group's kind, never by call site:

- *Full-history* groups hold the decode slot — the chunk that
  completes the prompt reserves `decode_input_tokens`, so the first decode step
  is guaranteed a slot; intermediate chunks reserve nothing because they are
  not about to decode — raised on a decoding role's first chunk to the rest of the prompt
  plus the admission headroom (§4), so a partially prefetched request is never
  stranded.
- *Sliding-window* groups recycle slid-out pages, so they hold the decode
  slot, raised on a decoding role's first chunk, while another request holds
  pages, only to the growth recycling cannot fund, capped by the window's
  lookback (`window - 1` tokens) and the prompt headroom. That growth is the
  larger of two terms: a prefix hit (a shared page frees nothing when it
  slides out), and the rest of the prompt when a local chunk leaves some
  behind (the next chunk reads this one's lookback before any of it can slide
  out). The hit term is 0 in a replayable group, whose pages over the hit are
  private (§1.3), and once the prompt's own tokens after the hit reach
  `window - 1 + block_granularity`: every hit page then slides out before the
  first decode. A remote landing has no later local chunk, so it prepays only
  the hit. Any other page holder raises the reserve, not only one sharing the
  hit: the prepay deliberately covers hit pages that may become shared after
  the admission. With no other request holding pages the raise is 0 and the
  group holds only the decode slot: hit pages are then held by the prefix
  index alone and recycle like the request's own, and the single-request bound
  (§1.4) counts no prepay, so a prepay could leave a request the bound accepts
  inadmissible on its own. Growth left unfunded by pages taken after the
  admission is left to the victim policy's empty-round retraction (§2).
  Broadcasting the headroom to them once kept a 54K-token DeepSeek-V4 prompt
  waiting on a pool that had room for it.
- *State* groups reserve nothing: their table is two slots whatever the
  prompt length, and a decode step writes the live slot in place (§1.2).

The decode slot is reserved on **every** role, the P role included. A P node
never decodes locally, but with speculation configured the forward that
completes a prompt runs the drafter once and writes its candidate block into
the `decode_input_tokens` slots behind the prompt — the same window a decoding
role verifies into — before `plan.remote_decode` ships the candidates. Without
that reserve those rows have no page and fall to the dummy slot, and the block
attention that reads them back proposes garbage. What P does *not* reserve is
decode growth: no admission headroom, no overlap protection (§3.1). The
capacity model (§1.4) states the same split.

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

### 1.2 State groups: two slots, natural checkpoints

A State group (a recurrent layer's state, `AttnKind::kState`) keeps no
history: one block holds the whole state after some prefix, so the group has
no token span (`block_granularity = 0`) and its table is **two slots**
whatever the prompt length. Slot 0 is the input checkpoint a forward reads,
slot 1 the output it writes. A prefill row reads slot 0 and writes slot 1; a
decode row reads slot 0 when it holds a block and slot 1 otherwise, and
writes slot 1 — the row contract the runtime's `block_tables` carry. Every
admission states what the pair does through the group's `StateStep`
(`MakeGroupDemands` sets it for all State groups at once):

- `kRotate` — a prefill chunk, or the first decode of a fused request: the
  previous output becomes the input (slot 1 → slot 0) and a fresh block is
  acquired for slot 1. The input it displaces is released, which only frees
  it when nothing published it. On a prompt's first admission the pair is
  shaped instead: slot 0 is the checkpoint the probe matched or, with none,
  a **fresh zeroed block** — the zero state — and slot 1 is fresh. A request
  therefore holds its two blocks from its first admission on, and every
  later rotation swaps the released input for the fresh live slot; a pool
  sized at two blocks per live request (§1.4) back-pressures at admission
  instead of wedging at the first decode, when the rotation would otherwise
  need a block the pool no longer has.
- `kRelease` — every later decode step: slot 0 is let go and slot 1 is
  written in place. Nothing is acquired, so a decode consumes no state
  capacity (`ConsumeReservedTokens` skips State groups; the in-place fast
  path releases the input through `ReleaseStateInputs`).
- `kKeep` — a restore and the D role's first decode: the pair is left as it
  is (the landed or restored slot 1 is the state the decode continues from).
- `kLand` — the D role's remote landing: slot 0 is the claimed prompt-end
  checkpoint or null (nothing on this node reads a zero state), slot 1 a
  fresh block the peer's transfer fills.

**Natural checkpoints.** A state is published only at a point where its owner
stops writing it — the same `PublicationPoint` that publishes the KV tail
(§1): a fused prompt end (the first decode's admission publishes slot 1 under
the prompt's key before rotating it into slot 0), a chunk end that falls on a
junction (§1), and a finish or retraction on every role (`kFinal`: the P's
transferred prompt, the D's and the fused engine's last decoded state). An
ordinary chunk end or decode step publishes nothing for a State group
(`CompletedPages::publish_state` is false). The key is the key of the exact
token count `T`: the chain hash of the last page when `T` is a page multiple,
otherwise the tail hash `hash@T` the KV groups use — so one key names a
request's progress across every group kind (`cache-concepts.md`). A published
state block is never written again: rotation moves it to slot 0, where it is
read only, and the next step releases the request's reference while the
cache keeps the block.

**A finish settles when its last forward has landed.** `forward::Finish`
(and the P role's `pd::SucceededEvent`) moves a page-holding request to
`Finishing` and nothing else; after every batch of events,
`settleFinishedRequests` publishes the frontier of each `Finishing` request
with no forward in flight (`kFinal`: KV tail and state) and releases its
pages (`ReleaseEvent`). One path for every schedule: a request that finished
quiescent settles with its Finish event — the timing every caller always
saw — and one whose finish was decided on a landed token while the overlap
schedule had already dispatched the next decode settles when that decode's
result lands, at the frontier the tokens it reports leave. The live slot is
therefore never published while a forward writes it, and a paused loop, which
drains its forwards without building plans, settles its finishes all the
same. The runtime reports every landed forward row, tracked or not
(`generation_output_processor`), so the count always reaches zero. A
`Finishing` request is never scheduled and never a retraction victim; an
abort releases it without publishing.

**A finish publishes the state before the stop token.** The device ends a
request where the control plane does: a verify round keeps its accepted
tokens only through the first stop token and within the token limit, so the
state commit and the frontier never pass the token the finish is decided on,
and the decode the overlap schedule dispatched behind that round runs the
request as a padding row and reports no token. Whatever the schedule and
however many tokens a round accepts, the finish frontier `T` is therefore the
stop token's position (the last kept token's for a token limit): the state
and KV before it, a prefix of the next turn's prompt whether or not its chat
template keeps the stop token. A finish the device cannot decide (a stop
string, a grammar, a client abort) reports every token the round kept
instead, so the frontier still names the state the device holds
(`cache-concepts.md`, "Recurrent state").

**Hits.** The probe (§1, "What the probe may claim") resumes a hybrid model
at the longest `T` with a published state checkpoint in every State group and
KV coverage in every token-spanned group; the state lands in slot 0 as a
claimed cache block (a Host checkpoint is loaded into a fresh slot 0 like any
Host hit). When the KV groups reach further than the state (`kv_tokens >
num_common_tokens`), the difference is the junction: the first chunk ends
there and publishes the checkpoint the cache lacked, so the next prompt that
shares the prefix resumes at the KV boundary in every group. No alignment is
ever imposed on a chunk for the state's sake, and no checkpoint is produced
inside a forward: a chunk's state output is its own natural checkpoint.
`CacheProgress::junction_tokens` is the only progress a State group adds to
the request; the backend contract for the two slots is in
[Cache concepts](cache-concepts.md#recurrent-state-two-slots-natural-checkpoints).

**Capacity.** A State group holds at most two blocks per live request on
every role — the pair, never a page per token — so `ReservePrefillDemands`
reserves nothing for it and the capacity model (§1.4) charges `2` per request.
On the D role the landing writes slot 1 and slot 0 stays null unless this
node already holds the prompt-end checkpoint (§3.2).

**Retraction and restore.** A victim publishes its frontier like a finish, so
its slot 1 rides the L2 leg of the image when there is a Host cache and the
unpublished slot 0 goes to the snapshot pool (§2). The restore rebuilds the
pair in place (`StateStep::kKeep`), copying the published slot 1 into a fresh
block rather than claiming it — the resumed request writes it next — so a
resumed request needs no checkpoint it did not already own (§4).

### 1.3 Bounded replay

A sliding History group can be declared **replayable** (`CacheGroupConfig::
replayable`, see [Cache concepts](cache-concepts.md)): it leaves
prefix caching entirely — never matched, published or streamed — and the
model regenerates its rows from **re-fed prompt tokens**. DeepSeek V4.1's SWA
rows and compressor tails are the motivating case: caching them persistently
costs more than recomputing a bounded window, and the prefix hit should
depend on the global KV alone. What a hit re-feeds is the group's whole
retention window: retention keeps exactly what the queries after the hit
read, and every page the group retains must be regenerated, so there is no
second number to declare.

The cache facts live on the `CacheCoordinator`, next to the specs they
derive from: `ReplayWindowTokens()` (`W`, the largest `sliding_window_tokens`
over the replayable groups) and `ReplayTokens(P)` (`min(W, P)`, what a hit at
`P` must re-feed). The
scheduling rules live with the other chunk-cutting rules in
`scheduler/operations/prefill_chunk.h`; the forward planner never branches on
replay — the two decisions below reach the common path only through
`PrefillChunkTokens`, the one chunk-sizing helper both prefill paths call,
which also carries the junction cut (§1):

- **After a prefix hit at `P`**, the first chunk re-feeds `[P − min(W, P), P)`
  ahead of its new tokens, so the queries at and after `P` find the whole
  window regenerated. This is the only place tokens are re-fed.
- **No prompt's final chunk is shorter than `W`** (`ChunkKeepingFinalWindow`):
  a chunk that would leave `0 < remainder < W` is shortened so exactly `W`
  remain. A junction (§1) that falls inside that final window yields: no
  chunk end satisfies both rules, so the chunk passes the junction and the
  State groups publish their checkpoint at the prompt end instead of the
  request waiting forever. The model narrows its decoder to the prompt's last window, which
  must therefore arrive in one forward — as new tokens, never by re-feeding
  what an earlier chunk of the same request already computed.

The re-fed rows are ordinary forward input — they consume the round's token
budget like any other row and `input_length` counts them — but they are not
progress: `TokenContainer::Window{begin, size, replay}` keeps `begin`/`size`
as the tokens this chunk computes (`replay` is non-zero only on a hit's first
chunk), and `MakePrefillInfo` derives the model input
`[begin − replay, begin + size)`. The runtime sees the pair
`extend_prefix_lens[i] = begin − replay` and `extend_replay_lens[i] = replay` on the
`ForwardBatch`; positions `[extend_prefix_lens[i], extend_prefix_lens[i] +
extend_replay_lens[i])` regenerate the replayable groups only and must not be
written into any other group, whose rows already sit in the shared cached
pages the hit claimed.

Capacity: a replayable group claims no hit pages, so at first admission its
table is empty and `CacheCoordinator::Admit` itself materializes it as a
sparse private suffix from the replay window's first token — slots below stay
null holes, exactly as absolute-slot tables require — while closed groups keep
the dense demand beyond `P` the caller stated. The FSM derives the window's
`replay` from the same coordinator (`SchedulePrefillFirstChunkEvent`), so no
event or scheduler operation carries a replay parameter. Later chunks change
nothing.

Budget: `SchedulerConfig::Validate` requires `max_scheduled_tokens ≥ W +
max(W, P)` — a hit chunk re-feeds up to `W` and must still carry the prompt's
final window whole, and keeps at least a prefix page of new tokens when the
window is smaller than a page (a junction cut may shorten a chunk to any
positive count, so it imposes no minimum of its own) — the first chunk spends
the hit window before sizing its new tokens (and waits for a fresher budget
when none is left), and in fused mixed mode the decode batch leaves that same
amount for a pending local prefill (`MinPrefillChunkTokens`). Replay is derived at admission only: a retracted
request's replayable pages are imaged with every other group's, and the chunk
it resumes has `replay = 0` by construction, so nothing is re-derived at
restore (§4). Replayable groups combine freely with State groups: a junction
inside the final window yields to the window rule (above), and nothing else
about a State group constrains a chunk.

PD: a replayable group travels like any sliding-window group. The prefill
role replays on its own local hits exactly as the fused role does and, at
completion, transfers the group's retained tail (`full_suffix` selects the
pages intersecting the last `sliding_window_tokens − 1` positions). Every
page of that tail exists because a hit re-feeds the whole retention window:
the regenerated suffix starts at or before the tail. The decode role computes no prompt rows, so
`SchedulePrefillFirstChunkEvent` gives a remote prefill `replay = 0` and
`Admit` leaves a demand that already names the landing's sparse suffix
alone; the landed tail is what the first decode steps read, with no
regeneration anywhere.

### 1.4 What bounds a single request

`MaxSingleRequestTokens` is a **startup** bound computed by binary search over
`CapacityModel::SingleRequestGroupPages` (`csrc/scheduler/capacity_model.h`):
the largest prompt whose worst-case working set — dense or sliding history
with its decode reserve and overlap-depth protection, the blocks a tail
handover holds for one admission, two slots per State group — fits the pool.
It is not a live check against currently free capacity; a prompt within the
bound can still fail admission right now and waits. The bound counts
no sliding-window prepay (§1), and a request with no other request holding
pages prepays nothing, so the prepay never makes a request the bound accepts
inadmissible on its own. Beside other requests the prepay can ask for up to
one window lookback more than the bound counts; such a request waits until
their pages are released, at the latest until it is alone, unless later
arrivals keep taking the pages.

The `CapacityModel` is deliberately **config-only**: it reads every
`SchedulerConfig` field that is known before a pool exists and no
`total_pages`, validating that subset through
`SchedulerConfig::ValidateCapacityInputs()`. That is what lets the Python
recipes size a pool from the same model before the arena is allocated
(`recipes/scheduler_bridge.py` builds an unsized config and asks
`ConcurrentGroupPages(max_total_tokens, max_context_len)` for each group's
demand at `max_batch_size` live requests), and then lets the `Scheduler`
bound requests against the pool they sized. The per-request working set —
`decode_width + overlap_schedule_depth * decode_width` protected tokens, the
next verify window a speculative decode's draft steps write on the roles that
decode (`DraftWindowTokens`, which `scheduleDecode` reserves from), the
two slots of a State group (`kStateSlots`), the most blocks a sliding
window can need behind a boundary (one more than behind a page-aligned one,
`SlidingWindowBlocks`) — exists in that one file; neither side restates it. The two answers are tied by an
invariant the model's tests sweep: for one live request of `L` tokens,
`ConcurrentGroupPages(L, L)` is never below `SingleRequestGroupPages(L)` in
any group, so a pool sized for the configured concurrency admits every
request the bound accepts.

Per group, `ConcurrentGroupPages` charges: a State group two slots per live
request (the working set does not grow with history); a prefix-closed history
group `ceil(T / g)` dense pages plus,
per request, `ceil((g - 1 + protected) / g)` for the unaligned tail and the
protected tokens (the draft window included) that may spill past it; a sliding group, per request,
`ceil((min(W - 1, ctx) + decode_width + protected + g - 1) / g)` resident
pages, plus one in-flight prefill chunk behind its lookback (or, on the
decode role, the landing bound `min(dense, lookback + window)` per request).
With the prefix cache on, every token-spanned group adds per request the
blocks a **tail handover** holds for the length of one admission
(`tailHandoverBlocks`): a hitter copies a tail's full blocks out of shared
entries that stay pinned as sources until the copies land, and a fused or P
owner that keeps writing past its prompt end moves off every published tail
block — the partial one included — into fresh copies while the published
blocks are still in its table. A tail is at most `P − 1` tokens, so that is
`ceil((P − 1) / g)` blocks on a role that prefills locally and
`floor((P − 1) / g)` on the D role (its partial block is a fresh page, not a
copy); a sliding group caps it at its lookback. The charge is what lets a
pool sized for a request fund its copy-in; when a pool cannot (several
retraction-exempt requests reaching their first decode together, each
needing a copy), the owner publishes nothing at that point instead of
waiting (§1).

Speculative decode admission grows from the committed token frontier plus the verify
spans still in flight, the span being scheduled and the next span, which the
draft steps write; that last span stops at the request's token limit, so it
stays inside the admission limit, a request without a declared budget reserves
none, and the capacity model charges it per live request. Already reserved slots
cover that extent first. A prefill interruption must not charge the same
speculative slots again when decode resumes; otherwise rejected draft tokens
accumulate in the logical tables beyond the context-length bound.

A State group's working set is the pair, on every role and in every state
of the request: a chunk reads slot 0 and writes slot 1, a decode writes slot
1 in place, the D role lands into slot 1, and a restore rebuilds the pair
(§1.2). Tests cover small pools that must reject an oversized request instead
of accepting a request that can never produce a forward.

## 2. Retraction: when admission fails

`maybeRetractForCapacity` fires when **no prefill made progress** this round
(`PlanBuild::NoPrefillProgress()`: nothing was admitted and no resident prefill
advanced a chunk) and an admission failed for capacity. Decode steps do not
count as progress — they release no capacity, so a round of pure decode leaves
a stalled prefill exactly as stuck as an empty one. Imaging a victim needs a
snapshot pool (`SchedulerConfig::HasSnapshotPool()`, §4) and a blob slot; a
round that finds no victim it can image falls back to the last resort below
and **aborts** one instead, so a capacity block never waits on a host budget.

**A retracted request loses no work.** It is *suspended*: its block tables are
imaged to Host byte-for-byte, its Device pages are released, and the restore
(§4) copies the image back into fresh pages and resumes the request in the
state it left — mid-prefill at its next chunk, or at its next decode step. The
cost of a retraction is the image's bytes (proportional to the pages held)
plus the client-visible pause; nothing is recomputed.

**Retract-and-grant, in one round.** The retraction serves a specific request —
the first candidate whose admission failed for capacity
(`AdmissionFeedback::capacity_blocker`). Victims are retracted and the blocked
admission is **retried in the same plan build**, looping (retract → retry →
retract) until it fits or the victims run out. The freed capacity therefore
reaches the request it was freed for within the round; there is never a free
page waiting for whoever asks first next round, which is what previously
required a cross-round capacity barrier. The empty-round retraction (below) is
the one exception: it offers the freed pages to the round's decodes first. Two
edges of the loop:

- **The victim may BE the blocker** (a resident request blocked on its own
  next page is the preferred victim). It comes back through the restore
  phase, so the grant is redirected: to the next request whose admission
  failed for capacity this round (`AdmissionFeedback::capacity_blocked`, in
  phase order), else to the first waiting prompt — granting the pages straight
  back to the victim's own readmission is the loop the grant exists to break.
  With nobody to redirect to, the victim is **not** retracted (except in the
  empty-round retraction below, when a decode is left to take its pages): the
  retraction would cost its image's copies and its restore and serve no one,
  so it waits for a completion instead.
- **A grant that cannot legally join its round's batch** (a fused prefill
  beside decodes outside mixed mode) still retracts one victim, and the next
  round's phase order (§3) tries the blocker before any other claim.

**The image has two legs, both stream-ordered.** `retractVictim` first
publishes the victim's computed frontier into the Device index exactly as a
finish does (`AdvanceRequestProgress` at `PublicationPoint::kFinal` +
`CacheCompletedBlocks`: the full pages, the partial page as a tail, and each
State group's slot 1; §1.2), then `TierTransferManager::StartRetractionStores`
classifies every data slot of every table:

- *The L2 leg.* A slot whose Device block is a published prefix entry becomes
  an ordinary Host L2 entry, acquired — like every Host copy — in the Device
  block's bucket (`cache-concepts.md`, placement). Keys already Host-cached, or
  carried by a store still in flight, are pinned rather than copied again; the
  rest ride one `WriteBackOperation` issued with
  `StoreSourceGuard::kStreamOrdered`. The image holds a pinned `CacheBlockRef`
  on each Host entry until the restore lands, so the planner cannot evict it
  and `ClearCache` refuses, but the entries are published prefix cache like any
  other and later prompts may hit them. The pin follows the publication, not
  the ticket's block: when the store's ACK finds the key already canonical on
  Host (an L3 prefetch of the same page by another prompt landed first),
  `PrefixCacheIndex::Register` redirects the ticket to that entry, and
  `CompleteWriteBack` hands the entries as published to every `Retracted`
  request waiting for the op (`StoreLandedEvent`,
  `RetractionImage::FollowPublished`), which re-points its slot to the
  canonical block and lets its own unindexed block return to the pool. An
  image therefore never pins a Host block the index does not know, and the
  restore's L2 rows always come from published entries.
- *The pool leg.* Every other slot — a State group's unpublished slot 0,
  replayable groups, any block the index did not take, and with no Host cache
  the whole image — gets a block of the request-private
  **snapshot pool** (`snapshot_allocator`; never prefix-indexed, never
  evicted), again in the Device block's bucket. These rows, plus the
  **slot-state blob** of the request's runtime row (sampling state, speculative
  readiness — exported by the runtime into `snapshot_slot`), ride one
  `SnapshotStoreOperation`; the op may carry zero page rows, the blob always.
  A published slot whose Host L2 block cannot be acquired after evicting
  unpinned entries falls back to the pool.

An image **does not fit** when the pool cannot hold tail plus fallback, or
when no blob slot is free (`SnapshotSlotAllocator`, `max_retracted_requests`
slots). `retractVictim` then changes nothing (the publication it did is
written back as the victim's progress; unpinned Host entries evicted for the
attempt stay evicted) and reports the shortfall. **The last resort is an
abort, in one place.** `chooseVictim` ranks only candidates whose image fits
(`imageFits`: a blob slot is free and `CacheCoordinator::SnapshotPoolHolds`
the tail, the published pages being assumed to ride L2); when none does — and
without a pool none ever does — it names the newest retractable resident (the
least work lost; exempt requests excluded, they finish on their own reserve,
except in the empty-round retraction below)
and `Scheduler::onImageDoesNotFit` aborts it: `fsm::AbortEvent` frees its
pages and slot in that very round, the blocked grant proceeds exactly as after
a retraction, and the plan records it (`ExecutionPlan::aborts`, a
`SchedulerAbort{request_id, AbortReason::kImageDoesNotFit, detail}`) so the
runtime fails the request toward its client with the shortfall and the knob
to raise (`--retraction-snapshot-host-gb` / `--retraction-snapshot-ratio`,
`--retraction-snapshot-max-requests`).
The same site handles a victim that passed the probe but whose L2 leg fell
back to a pool that then could not take it. Waiting instead would deadlock
once every resident needs a page; charging worst-case host room at admission
would cost concurrency on every request to protect against a rare shortfall.
The debug knob (§2, below) never aborts: a forced retraction whose image does
not fit is refused and logged.

**The victim's pages are released — and grantable — immediately**, before
either copy has run. Both legs are stream-ordered: their tickets pin only the
Host destinations, and the runtime fences the **forward thread's stream** on
the D2H copies' completion ahead of everything else the plan does to those
pages — that stream carries the zeroing, fences the forwards, and gates a
granted remote prefill's RDMA trigger (see `DeviceHandle.execute` and
`event-loop.md`) — so the copies read the old bytes whatever the scheduler
does with the pages. These are the only stores that pay for their ordering on
the forward's critical path, and they have to: the pages are gone in the same
round.

**Every other store pins its Device sources until the ack.** Boundary
publications of a live request and the finish-time flush are issued with
`StoreSourceGuard::kPinnedUntilAck`: the ticket holds a `CacheBlockRef` on
each source, so the block stays cached and unevictable — the admission planner
cannot take it, `ClearDeviceCache` refuses, `NumNewlyReleasableLcmBlocks` does
not count it — until `CompleteWriteBack` publishes the Host entry and drops the
pin. Nothing else needs to know the copy is in flight, so the runtime copies
on its own stream and no forward waits. A cached block is never written again
by its owner (prefix reuse already depends on that), so the pin alone makes
the copy race-free. Both guards share one ticket (`StartPendingStores(guard)`
drains the publication queue; `StartRetractionStores` builds the image's L2
leg from the victim's tables), and the op carries `source_pinned` to the
runtime, which branches on the guard and never on the reason.

**Per-victim quiescence, not global.** A request whose own forward is still
out must not be retracted — its result would land on pages it no longer owns,
and the image must know the landed tokens — and one whose pages a PD transfer
still pins must not be either. Both are checked on the chosen victim; if it is
not quiescent, retraction waits for it rather than sacrificing a worse-ranked
request. Two global gates remain. An in-flight load-back: it is writing pages
its admission owns, and the victim policy cannot see that write. And an
in-flight *pinned* store: it holds Device capacity the ack returns by itself,
so retracting anyone for that capacity would be the thrash of §4 — the
blocked admission retries against the released pins next round instead.
Stream-ordered stores — both legs of an image — hold nothing and gate
nothing, and an in-flight restore gates nothing: its request is `Restoring`,
invisible to `chooseVictim`.

The forward-out check is a count in `fsm::ForwardResources`, incremented when
a forward is scheduled and cleared when its result lands. It lives in the
resource bundle rather than on the states that consume a *token*, because a
forward is out against the **pages**, and the bundle is what holds the pages.
Every page-holding state carries one bundle, and a transition moves it whole
to the successor state — so the count, like the pages, cannot be dropped on
the way from one state to the next.

The bundle follows one rule: **resources and publication progress land when
an admission succeeds; a state transition only moves them, never modifies
them.** The coordinator fills the block tables inside `Admit`; the scheduler
advances the cache progress (prefix-hash chain, pending junction) on a
copy, hands it to that same admission — which publishes the newly completed
pages — and writes it back to the request only after the admission succeeds.
A failed admission therefore leaves both untouched, and the retry re-derives
the same completed pages and asks for their publication again. Committing
progress before admission would record the pages as hashed while never
publishing them.
The scheduling events carry nothing but the shape of the next state (chunk
size, decode reserve). An intermediate prefill chunk produces no token but
does write KV, so it reports back with an empty `ExtendResult`: the arrival
is the point, not the payload. Work this
engine does not perform — the peer's decode on a P node, the peer's prefill on
a D node — is not counted here; those are fenced by the PD transfer ack.

**Victim choice** (`chooseVictim`, shared by D and fused). Neither tier loses
work, so the key is who frees the most for the least interruption: an
incomplete prefill first — no client is streaming it yet, and the mid-prompt
prefill is usually the request that blocked on its own next page, so
retracting it and granting its pages to a prompt that can finish is the
shortest path out of head-of-line — largest first, freeing the most at once;
then decode work by most newly releasable LCM blocks and fewest **generated**
tokens — the needed pages with the fewest victims disturb the fewest clients,
and among equal frees the client that has streamed least is interrupted.
Only candidates whose image fits are ranked (the abort fallback above takes
over when none does). Exempt in both tiers: a request whose reserve already
covers its whole generation (`Request::ReserveCoversGeneration`) — retracting
it frees exactly what its restore must take back, pure thrash — and it is not
aborted either, except in the empty-round retraction below. Excluded by state:
`Retracted`, `Restoring`, `RemotePrefilling`.

**Empty-round retraction.** The exemption trusts the admission reserve to cover
all growth, and a sliding-window group's does not always: it prepays only
against pages other requests hold at the admission (§1). A page that becomes
shared after the admission (a later prefix hit on it, or publish-time dedup of
identical prompt pages) frees nothing when it slides out, and a sequence still
shorter than the window, or a prompt whose next chunk still reads its
lookback, has nothing to recycle yet. When no resident can be retracted under
the exemption and the round built nothing — no forward, restore, remote
prefill or decode — no completion of it can free capacity, so the choice is
repeated with the exemption dropped. A round that runs anything keeps the
exemption: that work moves toward completions that free capacity without a
retraction. So does a round in which a finished request still waits for its
last forward (`Finishing`, §1.2): its pages come back when that forward
lands. The empty-round retraction changes who may be chosen and where
the freed pages go first; the quiescence checks, the global gates and the
abort of a victim whose image does not fit still apply. The round ran
nothing, so none of its decodes got a page, and they are offered the freed
pages before the blocker: a generated token survives any later retraction, so
each empty-round retraction that lets a decode run is progress no
retract/restore cycle can undo. Granted to a prompt instead, the pages can go
to one that blocks the same way on its next chunk and becomes the next
victim; granted to nobody (the victim was the blocker and nobody else is
blocked or waiting), the victim can be restored ahead of the decodes and take
the same pages back, round after round. A victim that is the blocker, with
nobody to redirect to, is therefore retracted here whenever the round has a
decode left to offer its pages to — there is no completion to wait for — and
otherwise waits as above. Only when no decode fits does the blocker get the
grant, by the same rules as for any victim, and the loop may take several
victims in one round until a decode or the blocked admission fits.

**Stall diagnostic.** `NextExecutionPlan` counts consecutive rounds whose plan
carries nothing — no forward, remote stream, cache op or abort — while
requests are live and nothing is out (no forward result, PD transfer or tier
transfer) that could change the next plan, and warns at 1024 such rounds and
every doubling after. It never changes a plan.

**Forced retraction** (`debug_force_retraction_interval`, a debug knob; `0`
off) exercises the path without pressure: every `|interval|` plans the oldest
`Decoding` (`> 0`) or `Prefilling` (`< 0`) request is *armed* — kept out of
that round's batch so its outstanding forward lands — and retracted at the
first plan where it is quiescent, bypassing the victim policy's exemption but
not the image-fit refusal. It runs at the start of the D and fused grammars,
before the restore phase, so the freed pages are granted by the ordinary
phases.

The P role never retracts: `buildPrefillWorkerPlan` does not call
`maybeRetractForCapacity` (the only two call sites are the D and fused
grammars). See §3.1 for why.

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
zero) and an `AdmissionFeedback` (`admission_failed`, `capacity_blocker`, the
`capacity_blocked` list the grant may be redirected to), so it can report
outcomes but never compose the batch.

### 3.1 P — prefill worker

**Phases:** completed prompts out on `plan.remote_decode` (their pages stay
pinned until the transfer finishes, so releasing them outranks feeding more
prompt work), then the shared local-prefill phases
(`scheduleLocalPrefillWork`): resident chunks, then new prompts.

**Reserve: the decode slot, nothing else.** The completing chunk reserves
`decode_input_tokens` like every role, because the drafter writes the first
candidate block there before the remote decode carries it to the peer. The
growth reserves stay off: no admission headroom (nothing is ever retracted),
no overlap protection (no local decode is ever in flight).

The transfer ACK (`pd::SucceededEvent`) publishes the prompt like a finish —
its full pages, its partial page as a tail and each State group's prompt-end
state (§1.2) — and releases request ownership; cached entries remain subject
to normal eviction.

**Retraction: none.** A P node's pressure valve is the transfer itself — pages
are pinned until the peer acknowledges, then released wholesale. Retracting a
prompt whose KV is mid-transfer would strand the decode side.

The PD pin is not recorded anywhere; it is a function of the FSM
(`Scheduler::pdTransferInFlight`). On this role every page-holding state is
pinned — the peer's decode reads the pages from the first scheduled chunk until
the PD ACK finishes or aborts the request. On the D role the pin is exactly
`RemotePrefilling`: the peer's prefill is writing the destination pages, and
`RemotePrefillDone` ends it by leaving that state. A fused engine never
transfers. Because the pin is the state, no event handler can forget to clear
it, and `Abort`/`Finish`/`RemotePrefillDone` release it by transitioning.

**Restore: n/a.** Nothing on this role is ever `Retracted`, so the restore
phase does not exist in its grammar; `Validate` refuses a snapshot pool on it.

### 3.2 D — decode worker

**Phases:**

0. Forced retraction, if the debug knob is set (§2).
1. The one restore this round may start (§4) — the first of the ranked
   `Retracted` requests whose landed image fits, resuming a streaming client,
   so it takes capacity ahead of fresh work. It rides **beside** the batch as
   a cache op: no token budget, no batch slot, no forward.
2. The decode batch (`scheduleDecodeBatch`) — every PrefillDone first decode
   and Decoding step; decodes consume no token budget on this role.
3. At most **one** remote admission — the whole prompt at once (the peer
   prefills it), riding `plan.remote_prefill` **beside** the decode batch: it
   consumes no token budget and no batch slot, so there is nothing to defer
   for. Capped at one per round because each reserves an entire prompt's
   pages; a queue's worth in one round would drain the pool before any KV
   arrives. Head-of-line (1.1) does not apply — there is no mid-way. A restore
   that failed for capacity in phase 1 **seals** this phase: a newcomer taking
   the pages it waits for would starve it.
   Remote admission therefore bypasses local prefill chunking and prefix
   promotion boundaries; the destination tables must cover the entire
   unmatched suffix advertised by the transfer manifest.
4. `maybeRetractForCapacity` (§2), whose grant also rides beside the batch
   (a remote admission) or joins it (a blocked decode).

A decode batch, a restore and a remote admission coexist routinely. Nothing on
this role is ever `Prefilling`: a prompt is the peer's work, and a retracted
request comes back by restore, never by a local prefill — so no round is ever
claimed by an extend forward, and the capacity model charges no prefill peak
(§1.4).

**What the D probe may claim.** The remote admission probes this node's own
Device cache (`CacheCoordinator::ProbeDecodeDevicePrefix`) so the peer
transfers only what is missing. For the token-spanned groups it is the
ordinary probe: `extend_prefix_lens` is the hit `T`, page-aligned or a
published tail, and the block containing an unaligned `T` is left a hole for
the peer to fill whole; the tail's full blocks below it, when a group's
blocks are smaller than a page, are the hitter's own copies like on every
role (`pages_to_copy`). For the State groups only one checkpoint counts: the one at
**exactly the prompt end** (`PrefixProbe::state_at_prompt_end`). When every
State group holds it, the admission claims it as slot 0 and reports
`state_prefix_cached` on the `PrefillOperation` (per extend row on the
`ForwardBatch`), and the peer omits the state from the transfer; the first
decode then reads slot 0 and writes slot 1 (`StateStep::kKeep`). A state
checkpoint at any shorter `T` is ignored and claims nothing — this node runs
no prefill that could continue from it, so the KV hit is trimmed
independently and the peer's prompt-end state lands in slot 1 as usual. On
finish (and on the transfer ACK) the D publishes its own frontier like a
fused engine: the KV tail and slot 1 (§1.2), which is how the exact
prompt-end checkpoint of a repeated prompt comes to exist on this node.

**Retraction and restore:** victims are chosen by the shared rule in §2 —
everything resident is decode work (prompts the peer prefilled, in
`PrefillDone` or `Decoding`); `RemotePrefilling` is pinned by the peer's
in-flight prefill and never a victim. The image is taken from this node's own
pages and nothing crosses the PD wire: the receiver and admission record were
already released at `RemotePrefillDone`, and the bootstrap token lives in the
request's token container, so a `PrefillDone` victim restored to `PrefillDone`
still gets its first decode exactly as before. On every role the **first
decode after a restore carries its input token explicitly**
(`decode_input_id = LastToken()`, the marker `fsm::RestoreMarker` that
`RestoreDoneEvent` sets and the first `ScheduleDecodeEvent` consumes): the
request sits in a new request-pool slot, and the capture its last forward left
belongs to the slot it was retracted from, so the device cannot fill the input
itself as it does for an ordinary overlapped decode. A restore runs no forward, so a
decode engine whose attention layout cannot run an extend (head TP,
`--attn-head-tp-size`; see `docs/serving/parallelism.md`) retracts and
restores like any other.

### 3.3 Fused — one engine, everything local

**Phases, mixed mode** (`enable_mixed_prefill_decode`): forced retraction
(§2) and the one restore this round may start (§4) first on every mode — the
restore resumes a streaming client and takes no budget — then decodes: a
client is streaming them, and a long prefill chunk must not starve them of
token budget; then the shared local-prefill phases (resident chunks, then new
prompts) spend what remains, then `maybeRetractForCapacity`. The decode batch
leaves `pending_prefill_reserve` untouched when a local prefill is pending:
the budget a hit chunk needs to re-feed its replay window and still advance
(`MinPrefillChunkTokens`, §1.3; zero without replayable groups). A restore
that failed for capacity seals new-prompt admission for the round
(`new_prompts_sealed`), as on the D role.

**Phases, non-mixed:** after the restore, the prefill phases run first and
alone; decodes get the round only when no prefill scheduled. No prefill
reserve is needed — scheduling order is the capacity priority.

**Retraction:** the shared victim rule (§2): incomplete prefills first, then
decode work. The Host cache decides the image's split, not whether there is
one: with L2 the published prefix — tail and state checkpoint included —
travels as pinned L2 entries and only the unpublished slots go to the snapshot
pool; without L2 the whole image goes to the pool.
Either way the request resumes where it stopped, and never competes for
admission as a newcomer.

## 4. Suspend and resume

What the scheduler keeps across requests for retraction is **one integer**
(`next_retraction_epoch_`) and **one slot allocator** (`snapshot_slots_`, a
`SnapshotSlotAllocator` handing out blob slots `1..max_retracted_requests`);
everything else lives on the two FSM states a suspended request moves through
and is dropped with them. There is no queue to keep in step with the FSM: a
request that finishes or aborts while suspended stops qualifying, and
its `CacheBlockRef`s release what it held.

**`fsm::Retracted` holds no Device pages.** It carries the request's token
container and cache progress, the `RetractionImage` — per group an
`ImageTable` (`num_blocks`, `reclaimed_prefix_blocks`, `available_tokens`,
and per data slot its `slot_index`, a pinned ref on the Host block that holds
the bytes — an L2 entry with its `CacheKey`, or a snapshot-pool block with
none) — the blob slot, the `ResumeShape` that names the state it resumes
(`ResumePrefilling{window, reserve}`, `ResumePrefillDone{window, reserve}` or
`ResumeDecoding{reserve}`), its `retraction_epoch` and `resumes_generation`
stamp, and `pending_store_ops`: every op the image waits for — both legs' own
ops and any earlier in-flight store carrying one of its keys. The image is
**landed** (`ImageLanded()`) when all of them have been acknowledged
(`cache::WriteBackDone`, `cache::SnapshotDone`); until then the request is
not a readmission candidate, because a restore started earlier would copy
bytes that have not arrived.

**Readmission order** (`rankedReadmissions`) is derived, not stored: among
this round's candidates whose image landed, victims with generated output
first (they resume a generation a client is already reading), then oldest
epoch. The flag is `Request::HasGeneratedOutput()` — token count above the
submitted prompt size — rather than "was the victim decoding": a victim taken
mid-prefill may still own generated tokens, and its standing survives.

**The restore phase** (`scheduleReadmission`, phase 1 of both grammars) tries
the ranked readmissions in turn until one restores — one restore per round —
scanning at most `kMaxRestoreAttemptsPerRound` (4) of them, because each
failed attempt is an admission-planner pass. So a 60K-token image that does
not fit the free Device pages does not hold a 2K image behind it hostage; it
does still **seal** new-prompt admission for the round (any landed image that
waited seals, whether or not a later one restored: a newcomer must not take
the pages it waits for). A readmission that found no request-pool slot stops
the scan without sealing — nothing later would get a slot, and neither would a
newcomer.

**The restore** (`scheduleRestore`) rebuilds the request on fresh Device
pages:

- `CacheCoordinator::Restore` gives every group a table of identical shape —
  same `num_blocks`, same null holes (a sliding group's reclaimed prefix, a
  State group's empty slot) — with one fresh block per imaged slot **in
  the imaged bucket**, then runs the ordinary `Admit` demand on the rebuilt
  tables inside the same planner pass: `DenseGrowth{0}` (`StateStep::kKeep`
  for a State group) plus the reserve `ReservePrefillDemands` derives from
  the resume shape exactly as for a first chunk — the decode slot when the
  request resumes decoding or its completed prompt, the rest of the prompt
  too when it resumes mid-prefill — and, on every resume shape, the escalated
  admission headroom (`Request::AdmissionHeadroom`, below): the restore is the
  admission that re-secures the room the retraction proved too optimistic,
  whichever state the victim was in. Cache-only blocks are evicted for it as
  for any admission, and a failed restore leaves nothing allocated.
- An L2 slot whose key still has a Device-cached canonical block is
  **claimed** instead of copied (the same bytes by construction — the victim's
  own block, cache-only since it was freed — and protected from the planner's
  eviction like a prefix hit). Every other slot becomes one row of a single
  `SnapshotRestoreOperation`, tagged with its source tier (`HostTier::kL2`
  rows carry their key, `kSnapshotPool` rows none). A copy fills its
  destination block whole, so — as for a Host hit's load-back — only the
  reserve pages appended beyond the imaged shape are listed in
  `plan.pages_to_zero`. The op also names the blob slot and the request's
  **new** request-pool index: the runtime imports the slot-state blob into
  that row.
- The request moves to **`fsm::Restoring`**: it holds the rebuilt tables (a
  `ForwardResources` bundle, so it occupies capacity like any resident), the
  image and the op id, is never scheduled, never a victim and never a
  readmission candidate. Both ends of every row, the blob slot and the new
  request-pool row stay pinned by the transfer manager until the ACK, so an
  abort while restoring cannot re-grant a page, slot or row the copy is still
  writing. A restore takes no token budget and no batch slot; it is a cache
  op beside the batch, like a remote admission.
- `cache::RestoreDone` republishes the L2-tier destinations into the Device
  prefix index (as a prefix load-back does), drops the image — the L2 pins go,
  the entries stay published and evictable; the pool blocks return — and
  `RestoreDoneEvent` moves the request to the state its shape names: a
  `Decoding` request decodes next round with no prefill, a `PrefillDone` one
  takes its first decode, a `Prefilling` one schedules its next chunk with
  `replay = 0`.

Why a `Restoring` state at all: the copy is asynchronous and the request must
hold its pages while it runs, yet nothing may be scheduled against pages whose
bytes are still arriving. `Retracted` holds no pages by definition, and the
schedulable states must stay free of a "but not yet" flag every phase would
have to test; a state the phases do not know is the one way to hold pages and
be invisible at once.

**Finish or abort while suspended** drops the state: from `Retracted`, the
pinned Host entries become ordinary evictable entries at once and the pool
blocks return when the copies still writing them land (the tickets hold the
refs); from `Restoring`, the rebuilt tables are freed and the restore's ACK
only drops its pins — it republishes nothing, because the token descriptors
the KV-event feed needs for a publication died with the request. The two
non-page resources an image op uses follow the same rule: the **blob slot**
is a `shared_ptr<SnapshotSlotIndex>` held by the state and by the store
(`InFlightSnapshotStore`) or restore (`InFlightSnapshotRestore`) ticket that
exports into or imports from it, so it returns only when the last of them
lets go; the **request-pool row** a restore imports into is owned by the
restore ticket until its ACK (`Restoring::resources.req_pool_index` is empty
meanwhile) and handed to the resumed state by `RestoreDoneEvent` — or dropped
with the ticket when the request is gone. Neither can be re-granted to another
request while a copy still writes it. A late ACK for a dropped image is
harmless.

**KV-event descriptors.** With `enable_kv_cache_events`, every Device
publication is a mutation of a boundary that must already have its token
descriptor (`registerKvEventPrefixPages`), and `DrainKvEvents` drops the
descriptor of a boundary with no cached child. A boundary is `BlockStored`
only while every group holds its child -- the State groups included, whose
checkpoint exists only at a natural point -- so a hybrid model emits
`BlockStored` for its prompt end, its finish and its junction cuts, never
for an interior chunk end's chain hash; a tail's descriptor carries the
rows below `T` and the residency check counts a tail's blocks from it. Retraction publishes twice
without an admission: `retractVictim` registers the newly hashed pages before
`CacheCompletedBlocks` (as the finish-time publication does), and the
`RestoreDone` handler registers the restored request's whole prefix chain
before `CompleteSnapshotRestore` republishes its L2-tier destinations — the
victim's pages left the Device when they were granted away, so their
descriptors are typically gone by the time the restore lands.

**A readmission that does not fit waits.** Its failed restore never
triggers retraction (it is never recorded as the capacity blocker): when the
restore needs a victim, the two do not fit together, and swapping
them — two images and two restores per swap — is pure thrash. The resident
request keeps running and its completion frees the space. Unlike
escalation-bounded ping-pong this makes the evict-each-other cycle
structurally impossible. Nor does a waiting readmission stall anyone else:
decodes run regardless, and only new-prompt admission is sealed behind it
(a newcomer taking the pages it waits for would starve it) — the remote
admission on the D role, the new-prompt tier on the fused role.

**Escalating headroom and the exemption — kept, as thrash damping.** Both
rules predate the snapshot model, when they also kept a recompute-style
retraction from losing work without bound. That reason is gone (nothing is
recomputed) and is not why they stay. They stay because a retraction is still
an expensive interruption — two Host copies, a restore, a client-visible
pause — and without them one long request can be imaged and restored on every
page boundary for as long as the pool is tight. Being retracted means the
previous admission was still too optimistic, so each retraction raises the
decode headroom the next admission must secure:

```
Request::AdmissionHeadroom(safe_steps)
    = min(RemainingNewTokens(), safe_steps * (1 + retraction_count))
```

with `safe_steps = 4096` — note the `1 +`: a *fresh* admission already
prepays one window (see `schedulePrefillFirstChunk`), so for prompts with
`max_new_tokens <= 4096` the reserve covers the whole generation up front and
retraction never touches them, except the empty-round retraction (§2).
Capped by the generation budget the request
could ever use, so after a couple of retractions it holds enough room to run
to completion — at which point `ReserveCoversGeneration` exempts it from the
victim policy and it **cannot be retracted again**, and is not aborted either
(§2): it needs no further page, so it finishes on its own reserve and frees
its pages then — except where a sliding-window group's growth outruns that
reserve (§1), which the empty-round retraction covers (§2). This is a
per-request adaptive backoff: it penalizes only the
request whose admission proved over-optimistic, and never makes anyone else
wait. The constant is a thrash bound only; no liveness argument rests on it.

The exemption compares the windows the admission secured against the budget
that was open **at that admission** (`Request::RemainingNewTokensAtAdmission`,
stamped by the first chunk's and the restore's scheduling events), never
against the current remaining budget. Decode spends the prepaid headroom
exactly as fast as it shrinks that budget, so judging the window against
today's remainder would count spent headroom as still held: a request that
outgrew a partial reserve would look covered the moment its remainder dipped
under the window — exactly when it needs a new page — and once every resident
request looked covered, retraction would have no victim short of the
empty-round retraction (§2), and nothing else could free that page.

**There is no retraction without an image.** The one case that used to
recompute — an L3 prefetch that missed after admission — no longer exists:
L3 objects are fetched into Host before admission (§1), so an admission's
load-back cannot miss and no forward is ever skipped or retracted for L3.

**Weight updates and flushes.** `Scheduler::RetractedSize()` counts the
requests suspended with an image (`Retracted` + `Restoring`), and every flush
(`ClearCache`, `ClearL1Cache`, and the `CanClearCache` probe the replica
MIN-reduces before clearing) refuses while it is non-zero: a suspended
request continues from its image once restored, so a flush under it — a
weight update — would resume old-weight KV under new weights. The pinned Host
entries of an L2 leg would refuse the Host clear by themselves, but a
snapshot-pool leg (always, with `--disable-kvstore`) is indexed nowhere and
visible through no pin, hence the explicit count. A flush is likewise refused
while any image copy is in flight (`HasAnyInFlight` includes the snapshot
ops). The pause drain counts suspended requests as waiting, so a correct
runtime never hits either refusal; they are defense in depth.
`SnapshotPoolFreeBlocks()` and `HostPoolPinnedBlocks()` are the leak checks:
with no request suspended both must read empty and zero.

**Configuration** is explicit on every role: `snapshot_allocator.total_pages`
(`1` = the null page alone = no image ever fits, so a capacity block aborts
the newest resident instead of imaging one; §2) and `max_retracted_requests`
(`0` with no pool, `> 0` with one; the runtime's slot-state arena has that
many rows plus the null row) are validated together, the P role refuses a
pool, and L3 storage is refused with a page-cyclic sharded group
(`cache-concepts.md`). `debug_force_retraction_interval` (§2) is the only
knob that chooses victims outside capacity pressure, and it is refused
without a pool. Every diagnostic names the binding field and the server arg
behind it (`num_snapshot_pages` from the pool the runtime resolved --
`--retraction-snapshot-host-gb`, `--retraction-snapshot-ratio`, or the
derived default of `docs/configuration/server.md`, "Retraction snapshot
pool" -- `max_retracted_requests` from `--retraction-snapshot-max-requests`,
`--debug-force-retraction-interval`), because the runtime surfaces the message
verbatim at startup. The scheduler takes page counts; how the runtime arrives
at them (the Host KVStore-style size-over-ratio rule, the tail-per-request
default beside L2 that this model's `SingleRequestGroupPages` and
`LcmBlocksNeededFor` count) is `python/tokenspeed/runtime/cache/l2/sizing.py`.

**Release note — the runtime lands with the pin.** This scheduler is not a
drop-in for the runtime on `main`: `SchedulerConfig.num_snapshot_pages` is
required (there is no default; `make_config` in
`python/tokenspeed/runtime/engine/scheduler_utils.py` must pass it, `1` on
engines that never retract), `ForwardEvent.Retract` is gone with nothing in
its place (`make_retract_event` in `scheduler_utils.py` and the forward-skip
in `engine/l3_cache_hooks.py` go), `Cache.LoadBackDoneEvent` loses its
`success` argument and `Cache.LoadBackOp` its `prefetch_from_storage` rows,
`waiting_prefix_hashes` is gone (the prefetch op is the probe), L3 engines
pass `SchedulerConfig.l3_prefetch_min_pages`, and the plan carries three new
cache op kinds (`Cache.PrefetchOp`, `Cache.SnapshotOp`, `Cache.RestoreOp`)
with three new ACKs (`Cache.PrefetchDoneEvent`, `Cache.SnapshotDoneEvent`,
`Cache.RestoreDoneEvent`) that `DeviceHandle` and the cache hooks must
execute and count, and `ExecutionPlan.aborts` lists the requests a capacity
retraction aborted (§2) for the runtime to fail toward their clients.
Following AGENTS.md's release
sequence, the `tokenspeed-scheduler` version bump is published first and the
runtime PR that pins it (`feat/retraction-snapshot-consume`: server args,
`make_config`, the event rename, the op dispatch and the slot-state exporters)
lands in the same change as the pin, never a release apart.

## 5. Invariants a change must preserve

- Admission never grants pages for tokens beyond the chunk being scheduled,
  except the decode reserve on the completing chunk (1) and the admission
  headroom (4) — which full-history groups hold whole and sliding-window
  groups at most one window lookback of, and only while another request holds
  pages: the growth recycling cannot fund on a first chunk (1). A request
  alone prepays nothing, so the prepay never makes a request the
  single-request bound (1.4) accepts inadmissible on its own. A State group
  holds two slots and nothing else (1.2).
  A replayable group's private suffix starts at the replay window, which is
  inside the forward's input, not beyond it (1.3). A restore grants exactly
  the imaged shape plus the same reserve a first chunk of its resume shape
  would (4).
- A replayable group is never matched, published or streamed (1.3); its
  re-fed rows are forward input that debits the token budget but never
  advances `num_computed_tokens`; only a hit's first chunk re-feeds, and no
  final chunk is shorter than the replay window (`ChunkKeepingFinalWindow`).
  Its request-private pages are imaged and restored like any other group's.
- A prefill demand's reserve is decided once per group, by kind, in
  `ReservePrefillDemands` (1); no later step rewrites `reserve_tokens`, and the
  helper asserts it found none set.
- An incomplete local prefill is not overtaken (1.1). Decodes are never hostage
  to it: they consume no fresh capacity within their reserve, so they keep
  running beside a stalled prefill.
- Retraction fires only when no prefill progressed and an admission failed
  (2). The chosen victim must be quiescent — no forward of its own in flight,
  no PD transfer against its pages (§3.1) — and an in-flight load-back or an
  in-flight pinned store defers all retraction; stream-ordered stores and
  in-flight restores defer nothing. A victim that would serve nobody is not
  retracted. A victim whose image does not fit is aborted through
  `onImageDoesNotFit` and nowhere else; the debug knob never aborts.
- Freed capacity is granted to the request it was freed for in the same plan
  build whenever the round's grammar admits the grant, except that the
  empty-round retraction offers it to the round's decodes first (2); the
  image copies →
  page copies (`ExecutionPlan::pages_to_copy`) → zero → load → forward order
  on the forward thread's stream is what makes the immediate release safe --
  a tail hit's copy source may be evicted and re-granted by a later admission
  of the same round, as a fresh page or as another copy's destination -- and
  changing `DeviceHandle.execute`'s ordering breaks it. Within the copies,
  the arena reads a block before any pair overwrites it (`CacheArena.
  copy_blocks` runs such pairs in a later wave), so every source is read as
  it was when the plan was built. A copy destination is never zeroed.
- A published block is never written again: a tail's blocks are copied into
  the hitter's table (or loaded from Host into private pages), an owner that
  keeps writing past a publication point moves to fresh copies of the blocks
  it published -- or, when the pool cannot fund the copies, publishes no
  tail at that point and keeps writing its private block -- and a restore
  copies the imaged tail blocks instead of claiming them (1,
  cache-concepts.md).
- A request whose admission prepaid its generation never needs a block it did
  not reserve: the decode slot is reserved on the completing chunk, a State
  group's rotation swaps a released block for a fresh one, and the owner's
  copy-in is best effort (1). Without this a pool full of retraction-exempt
  requests wedges at their first decodes.
  A sliding-window group's growth can outrun its reserve (1); the
  empty-round retraction keeps such a pool live (2).
- A store either pins its Device sources until the ack or is stream-ordered;
  never neither (2). Only the two legs of a retraction image are
  stream-ordered — their sources are granted away in the same round — and
  only such ops may be fenced ahead of the plan's page reuse by the runtime. A
  new store site chooses its guard explicitly (`StartPendingStores` has no
  default).
- Every tier transfer pairs blocks of equal residue (bucket): L2 store, L2
  prefix load, both image legs and the restore; the transfer manager asserts
  it when it builds a batch, so under page-cyclic sharding the rank that owns
  one end owns the other (`cache-concepts.md`).
- An admission loads Host entries only; L3 objects reach Host through a
  `PrefetchOperation` issued before admission (1), and a `Prefetching`
  request holds no Device page, no request-pool row and no head of line.
- Only computed tokens are published as a prefix, and exactly those.
  `Request::NumComputedTokens()` is the one frontier for prefix hashing and
  retention on admission and retraction, and the extent an image covers: the
  scheduled window end while prefilling (an incomplete prefill's whole token
  count would publish pages never computed), and every token but the last
  while decoding — feedback ends with the sampled token the next forward
  computes. It does not subtract the verify width: a decode result lands its
  accepted tokens, not a fixed number, so any margin is an estimate that lags
  the real endpoint and delays publication and reclaim behind it. A finish
  that overtakes a forward still in flight (an overlapped decode) publishes
  the KV rows below the frontier, which are final, but not the State
  groups' live slot that forward is writing (`PublicationPoint::kFinalKvOnly`).
- A `Retracted` request holds no Device pages; its image is held by that state
  alone (Host pins and pool refs) and dropped with it. It becomes a
  readmission candidate only once every store it waits for has landed, at
  most one restore starts per round (after a bounded scan of the ranked
  candidates), and a `Restoring` request is never scheduled, never a victim
  and never re-probed (4). A readmission that fails admission waits, seals
  new prompts for the round, and never triggers retraction (4).
- A request whose admission prepaid the generation budget open at that
  admission is never a victim (2) in a round that builds a forward, restore,
  remote prefill or remote decode, or waits for a finished request's last
  forward; with the
  fresh-admission prepay this bounds such retraction to requests whose
  `max_new_tokens` exceeds one safe-step window (or is undeclared). Spending
  a partial reserve never makes it qualify: the
  exemption is judged against the budget open at admission, not the current
  remainder (4). Only the empty-round retraction (2), in a round that builds
  nothing and waits for no finished request, retracts or aborts an exempt
  request.

### Tail-bucket admission ownership

An admission that continues a partially filled tail must charge the owning
bucket before allocating new parent blocks. Capacity checks use the same
ownership as allocation; otherwise a shared prefix can make a request appear
admissible in a bucket that cannot hold its tail. Coordinator regression tests
cover partial tails, owner continuation and shared-prefix admission.

If publication canonicalizes an owner's tail to an existing cached block,
the copy demand follows that canonical block's bucket. The admission plan
holds a reference to the canonical source through eviction and commit, so
evicting a companion group cannot invalidate the planned copy source.
