# Scheduler: admission, retraction and restore

The C++ scheduler (`tokenspeed-scheduler/`) decides, once per round, what each
engine does next. This document covers one axis of that: **at what granularity
KV capacity is admitted**, what happens when admission fails, and how a
retracted request comes back — per engine role.

Companion documents: `event-loop.md` (the control/data plane split that
consumes these plans), `cache-concepts.md` (the vocabulary below —
prefix granularity, cache groups, LCM blocks).

## 1. Admission is per chunk

An optional `--chunk-prefill-kv-size` also bounds the KV read by a local
prefill batch. The runtime converts its token capacity to
`max_kv_page_num` using `prefix_granularity`. Admission rounds each request's
cached-prefix-plus-current-chunk length up separately, then sums those page
counts; shared prefixes still count for every request. A candidate may use only
the remaining pages. When configured, the page-aligned token capacity must be
strictly greater than `max_req_input_len`; startup rejects an undersized budget
instead of reducing the prompt limit. Decode and remote PD landing do not spend
this local-prefill budget. Omitting the limit preserves the existing policy.

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

**L3 is fetched before admission, never by it.** A request waiting for QUERY
or GET is `Prefetching` and absent from forward candidates. `Submitted` keeps
its ordinary admission meaning, including mixed-mode token reservations and
retraction demand. The wait carries a `Bootstrapping` or `Submitted`
continuation so PD bootstrap can overlap it. A completed read's bounded Ready
hold moves with that continuation until admission acquires its Host sources.
`StorageTransferManager` owns query/GET/PUT tickets, Host references and the
bounded backup queue. The request holds no Device pages or request-pool slot
while preparing. Runtime workers report object and retirement facts; the C++
group matcher chooses the
reusable boundary and ordinary admission acquires the resulting Host entries.
A soft deadline permits recomputation without retiring native I/O. Failed GET
rows and detached GET jobs cannot publish data. An attached, retired GET may
publish its successful rows even when other rows failed; ordinary group
matching determines the reusable boundary. Buffers still owned by native I/O
remain held.
State groups use the same checkpoint matcher and fixed two-slot layout as
local hits. Retraction remains an exact image restore, never an L3 re-prefill.
The D role probes Device history and receives missing data from its peer.
See [asynchronous L3](async-l3-cache.md) for lifetime and admission limits.

A first admission with no free request-pool row skips prefix hashing and
matching. This is a slot constraint, not a KV capacity failure: it does not
trigger retraction. Asynchronous L3 preparation runs before forward scheduling
and can still fetch Host data while rows are full, because it owns no row.
Resident prefill chunks and restore admissions keep their existing paths.

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

The input's full-page chain hashes are prepared once per request. A failed
first admission reuses those immutable identities and probes the current
Device and Host contents again; it never reuses a previous residency result.
The pending match views the request-owned hashes. Mutable execution progress
copies the reused prefix only after admission succeeds.

Decode consumes an existing reserve in place when no page or pending State
checkpoint became complete and no sliding-window block needs reclaiming. This
does not copy the accumulated hash chain. Publication and capacity admission
still use a progress copy so a failed admission leaves the request unchanged.
Previously published checkpoint holders are released after the in-place step,
as they are after a successful admission.

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
A resident chunk that fails admission ends the prefill phases too: admitting
anything behind it would strand it.

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

For State groups with `checkpoint_granularity > 1`, only prefill-produced
states may become reusable checkpoints. Publication remains in the ordinary
scheduling path: a later admission publishes any retained in-forward
checkpoint before rotating or reclaiming the owner's blocks. The first fused
decode admission may deliver the preceding prefill's prompt-end and interior
checkpoints; it does not publish a decode-produced state. Later decode states
stay private at finish and retraction even when their positions are aligned.
A prefill finish may still publish its endpoint, and retraction still saves
the private live state for that request's own restore. The D role does not
publish its transferred prompt state in this restricted mode. Unrestricted
State groups keep the policy above. Forward result events carry no checkpoint
data and do not trigger publication.

**In-forward checkpoint (opt-in).** By default a prompt that diverges inside
the last page of an earlier long prompt, or resends it, resumes from an
earlier checkpoint: the prompt-end state cannot serve a hit that must replay
the final token. With `state_checkpoint_in_forward`
(`--enable-state-checkpoint-in-forward`, off by default, requires prefix
caching) the forward that spans the last prefix-page boundary before a long
prompt's replay tail (`PrefillStateCheckpointTokens`: a prompt longer than
`max_scheduled_tokens`; the tail is at least one token, `prefix_replay_tokens`
and the largest replayable group's window) also writes the state there. No
chunk is cut: the admission of that forward acquires one more block per State
group outside the two slots (`StateStep::checkpoint`,
`AdmissionResult::state_checkpoint_blocks`), the forward's rows carry it as a
third entry with the tokens before it (`extend_state_checkpoint_lens`), and
the runtime stores every State group's state at that position into it. The
request holds the block in `CacheProgress` until its next publication point --
the next chunk's or first decode's admission, a finish or a retraction --
which publishes it under the boundary's page hash, like a natural checkpoint.
A boundary a chunk ends on needs none (the chunk end's slot 1 is published
there, as at a junction). The scan emits states only whole pages from a
forward's start, and the budget a batch's decodes leave rarely ends on a page,
so a chunk ending less than a full chunk before the boundary ends on a page
when the budget reaches one, unless a junction at or past its end comes first
(the chunk after the junction starts there anyway). A chunk that would still
span the boundary from inside a page (a hit on a prompt-end tail) ends at the
boundary, and so does one whose admission cannot also hold the extra block (a
cache with no spare block): the next chunk's admission publishes its live slot
there, at the cost of one more round. Multi-turn traffic whose next turn
re-renders the end of the previous prompt then reuses the common prefix
immediately.

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
(the control plane's forward commit, `ts-control`'s `output`), so the count
always reaches zero. A `Finishing` request is never scheduled and never a
retraction victim; an abort releases it without publishing.

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
Host hit). The tiers combine under the Host-suffix rule
([Cache concepts](cache-concepts.md#the-coordinator-layer-csrccachecoordinator)):
the KV is the Device's prefix plus a Host suffix, and each State group's
checkpoint comes from whichever tier holds it, the Device first -- so Device
KV with a Host checkpoint resumes, loading the checkpoint alone. When the KV
groups reach further than the state (`kv_tokens >
num_common_tokens`), the difference is the junction: the first chunk ends
there and publishes the checkpoint the cache lacked, so the next prompt that
shares the prefix resumes at the KV boundary in every group. For unrestricted
State groups no alignment is imposed on a chunk, and apart from the opt-in
in-forward checkpoint above no checkpoint is produced inside a forward: a
chunk's state output is its own natural checkpoint.
The backend contract for the two slots is in
[Cache concepts](cache-concepts.md#recurrent-state-two-slots-natural-checkpoints).

**A superseded input leaves first.** A prompt that resumes at a tail
checkpoint on the Device -- where its previous turn ended
(`AdmissionResult::resumed_tail_hash`) -- dates it with its access epoch at
admission, and its finish dates the end state it publishes with the same epoch.
The two would rank level in the Device's eviction order, so block location would
decide, and as often as not the end state the next turn resumes at would leave
first. The finish that publishes the end state (on the P role, once the transfer
completes) therefore marks that input superseded (`MarkSupersededStateInput`) on
every role, under either Host write policy and without a Host tier: its
checkpoints and its tail's KV blocks are registered again at access epoch 0,
level with the blocks no cache entry holds. It stays cached and matchable -- the
same prompt sent again resumes there -- and is the first eviction candidate: an
admission or a restore that needs its room evicts it, counted as request
reclaim, and one whose later victims make the room alone keeps it. A request
that resumes there again unmarks it until its own finish (an earlier request
finishing meanwhile leaves it unmarked, even once an older request's
publication of the same tail has dated it back; a request's own restore, which
dates what it loads back with the request's epoch, is no later hit), and a
request that fails before its finish (a transfer that fails) marks nothing, nor
does a finish whose end state is the input itself (a D request that claimed its
prompt-end checkpoint and leaves before any decode). One at a page
boundary -- a junction or an in-forward checkpoint, which other prompts
resume at -- is never marked. On the fused role a turn also publishes its
prompt end, at its first decode and at the same epoch: that one and the end
state still rank level, since which of them the next turn resumes at depends
on whether the client sends the reply back as it was generated.

**Aligned state checkpoints.** A State group can declare
`checkpoint_granularity`, a multiple of the prefix-hash grain. Its published
and matched states must end on that boundary. Intermediate prefill forwards
end on whole segments; a remaining prompt fitting the budget runs once. The
existing in-forward checkpoint mechanism saves the entrance of its last
segment, `(prompt_tokens - 1) / grain * grain`, including when the whole
prompt fits one round. Thus 8192 tokens run in one forward while saving the
state at 4096. Prefix reuse stays strictly below the prompt length, so a full
8192 match resumes at most at 4096. Only that extra entrance is retained;
aligned prefill endpoints remain reusable through the same admission path.
A decode step reaching an aligned position creates no checkpoint. The grain constrains
each request's logical position, independently of its packed batch offset.
Unrestricted groups preserve their original scheduling and publication.

**Capacity.** A State group holds at most two blocks per live request on
every role — the pair, never a page per token — so `ReservePrefillDemands`
reserves nothing for it and the capacity model (§1.4) charges `2` per request.
An in-forward checkpoint's block is a third only between its forward and the
next publication point, and only when the pool holds it then (otherwise the
chunk is cut at the boundary); the bound charges nothing for it.
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

The runtime's usable request length is the minimum of the model context and
`MaxSingleRequestTokens`. Input admission leaves room for the first decode/verify
window. The generation budget is capped at `usable_length - prompt_length`;
that input reserve is not subtracted again from the remaining output budget.

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
arrivals keep taking the pages. Under Host write-back (§3.1) the Device
evicts by whole segments, and a segment holding a block in use cannot leave
— the request's own blocks included, such as the pages under the checkpoint
it resumed from. So once whole segments cannot fund the admission of a
request no other one holds pages beside, the blocks they passed over leave
one by one (`cache-concepts.md`, `AdmissionPlanner`): alone, there is no
release left to wait for. Only the admission's last variant takes them: an
optional part of it — a copy-in (§1) or an in-forward checkpoint's block
(§1.2) — is dropped first and never pays for such an eviction. Beside
another page holder no block leaves one by one: the request waits for that
holder's release (on the P role, until its PD transfer is acknowledged),
where write-through, which evicts block by block, could fund it at once.
The block-by-block pass lifts only the segment rule. Under either policy the
blocks the admission's own prefix hit keeps never leave, and the
tail-handover charge (below) is bucket-blind, so on a KV-sharded pool of a
few pages a request alone can still wait for good, when its hit holds the
only cache left and its copy of the tail needs a page in a bucket that is
full.

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
needing a copy, or a sharded group whose copy's bucket is full while other
buckets have holes — the charge is bucket-blind), the owner publishes
nothing at that point instead of waiting (§1), and a hitter's admission
waits.

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

When every resident still has an outstanding forward or PD transfer, capacity
retraction waits before ranking or planning images: no possible choice can
quiesce in that round. Otherwise the normal rank and Host-fit order applies,
including waiting for a better-ranked busy victim. Decode release counts are
computed once per candidate for each ranking pass, before sorting.

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
  phase order), else to the first waiting prompt, else to a resident prefill
  the head-of-line rule (§1.1) kept from trying this round — granting the
  pages straight back to the victim's own readmission is the loop the grant
  exists to break. The held-back prefill is who the victim's wait would
  starve: behind a prefill blocked on its next chunk no other prefill runs,
  so two prompts mid-prefill holding every page, with a waiting readmission
  sealing new prompts and no decode left, would wait for a completion
  nothing can bring. With nobody to redirect to, the victim is **not**
  retracted (except in the empty-round retraction below, when a decode is
  left to take its pages): the retraction would cost its image's copies and
  its restore and serve no one, so it waits for a completion instead.
- **A grant that cannot legally join its round's batch** (a fused prefill
  beside decodes outside mixed mode) still retracts one victim, and the
  blocker waits for the next round, whose restore phase runs first (§3.3):
  the victim's own readmission, once its image has landed, can take the freed
  pages back before the blocker tries. So while decodes keep running, the
  victim can be retracted and restored as often as every other round, paying
  its image and its restore each time, and a redirect (above) does not break
  that loop. It ends when the decodes finish: with no decode in its round,
  the grant joins that round.

**The image has two legs, both stream-ordered.** `retractVictim` first
publishes the victim's computed frontier into the Device index exactly as a
finish does (`AdvanceRequestProgress` at `PublicationPoint::kFinal` +
`CacheCompletedBlocks`: the full pages, the partial page as a tail, and each
State group's slot 1; §1.2), then `TierTransferManager::SplitRetractionImage`
classifies every data slot of every table and `StartRetractionStores` takes
the image as the split planned:

- *The L2 leg.* A slot whose Device block is a published prefix entry becomes
  an ordinary Host L2 entry, acquired — like every Host copy — in the Device
  block's bucket (`cache-concepts.md`, placement). Keys already Host-cached, or
  carried by a store still in flight that publishes them at its ACK (never an
  eviction write-back's segment, which may publish late or not at all), are
  pinned rather than copied again; the rest ride one `WriteBackOperation`
  issued with `StoreSourceGuard::kStreamOrdered`. The image holds a pinned
  `CacheBlockRef` on each Host entry until the restore lands, so the planner
  cannot evict it and `ClearCache` refuses, but the entries are published
  prefix cache like any other and later prompts may hit them. The pin follows
  the publication, not the ticket's block: when the store's ACK finds the key
  already canonical on Host (an L3 prefetch of the same page by another prompt
  landed first), `PrefixCacheIndex::Register` redirects the ticket to that
  entry, and `CompleteWriteBack` hands the entries as published to every
  `Retracted` request waiting for the op (`StoreLandedEvent`,
  `RetractionImage::FollowPublished`), which re-points its slot to the
  canonical block and lets its own unindexed block return to the pool. An image
  therefore never pins a Host block the index does not know, and the restore's
  L2 rows always come from published entries.
- *The pool leg.* Every other slot — a State group's unpublished slot 0,
  replayable groups, any block the index did not take, the published slots
  the Host does not take (below), and with no Host cache the whole image —
  gets a block of the request-private **snapshot pool** (`snapshot_allocator`;
  never prefix-indexed, never evicted), again in the Device block's bucket.
  These rows, plus the **slot-state blob** of the request's runtime row
  (sampling state, speculative readiness — exported by the runtime into
  `snapshot_slot`), ride one `SnapshotStoreOperation`; the op may carry zero
  page rows, the blob always.

The split plans the L2 leg before anything is acquired or evicted for it. The
slots to copy are grouped by boundary (a content hash, with every group's slot
at it), the shallowest first, and the Host takes the leading boundaries that
fit whole (`CacheCoordinator::NumHostBoundariesThatFit`, planned in
`AcquireHostBlocks`' order and with its victim ranking against a shadow of the
Host); the retraction (`AcquireHostBoundaries`) evicts exactly the planned
victims and places exactly those rows. The first boundary the Host cannot take
whole goes to the pool with every later one, even where the Host has room for
some of their rows. Under the write-back policy the plan takes Host room for
every boundary as an eviction write-back does, by whole segments first
(`AcquireHostSegments`' plan: whole victims, none above a page an eviction
write-back is still bringing, boundaries dropped from the deepest), then block
by block, as above, for the boundaries the segment step dropped, against the
Host less its victims. Where the two steps place fewer boundaries than block
by block alone, the block-by-block plan stands, so the Host takes at least as
many boundaries as the block-by-block plan would on the same Host, and
sometimes more, since its victims differ. Placing block by block may take the
Host page above an eviction write-back in flight. The write-back still
publishes at its ACK — the settlement reads the tail hint and the checkpoint
boundaries under its node, not the gap above — and the next dead-entry
collection takes it as dead entries under the gap, and may widen the gap: a
copy wasted, accepted because the image comes before the cache.

An image **does not fit** when no blob slot is free (`SnapshotSlotAllocator`,
`max_retracted_requests` slots) or when the pool cannot hold what the split
leaves it (`ImageShortfall`): `kSnapshotPool` when it could not even with
every published slot on the Host, `kHostL2` when it could but the Host takes
too few of them. `retractVictim` then reports the shortfall before anything is
acquired or evicted for the image (the publication it did is written back as
the victim's progress). **The last resort is an abort, in one place.**
`chooseVictim` ranks only candidates whose image fits (`imageShortfall`: a
blob slot is free and the split of the image the retraction would take once it
has published, projected without mutation, has no shortfall — the probe and
the retraction run the same plan of the L2 leg); when none does — and
without a pool none ever does — it names the newest retractable resident (the
least work lost; exempt requests excluded, they finish on their own reserve,
except in the empty-round retraction below) and `Scheduler::onImageDoesNotFit`
aborts it with that resident's shortfall: `fsm::AbortEvent` frees its pages
and slot in that very round, the blocked grant proceeds exactly as after a
retraction, and the plan records it (`ExecutionPlan::aborts`, a
`SchedulerAbort{request_id, AbortReason::kImageDoesNotFit, detail}`) so the
runtime fails the request toward its client with the shortfall and the knob to
raise (`--retraction-snapshot-host-gb` / `--retraction-snapshot-ratio`,
`--retraction-snapshot-max-requests`, or for `kHostL2` `--kvstore-size` (or
`--kvstore-ratio` when no size is given), with the Host's size, its pinned
entries and whether copies are in flight). The same site aborts a chosen
victim whose image its own publication left without room, which the probe does
not foresee: a tail that publication drops under the per-prior cap takes its
Host entries along; under the write-back policy the segment step ranks Host
victims by what the Device holds, which the publication adds to; and under KV
parallelism a slot the publication re-points to the Device entry already
holding its key rides the snapshot pool in that entry's bucket, not the one
the probe counted. The split made after the publication can then fall short.
Waiting instead would deadlock once every resident needs a page; charging
worst-case host room at admission would cost concurrency on every request to
protect against a rare shortfall.
The debug knob (below) is not capacity pressure: it probes its victim with
`imageShortfall` before `retractVictim` publishes anything; a victim that does
not fit is refused, logged and left running. A victim refused only after the
publication (any of the ways above) is aborted at the same site, never left
running: the publication registered blocks it would write again (its partial
page as a tail, and a State group's end state, which a decode step writes in
place), and a published block is never written again by its owner (the prefix
cache depends on it).

**The victim's pages are released — and grantable — immediately**, before
either copy has run. Both legs are stream-ordered: their tickets pin only the
Host destinations, and the runtime fences the **forward thread's stream** on
the D2H copies' completion ahead of everything else the plan does to those
pages — that stream carries the zeroing, fences the forwards, and gates a
granted remote prefill's RDMA trigger (see `DeviceHandle.execute` and
`event-loop.md`) — so the copies read the old bytes whatever the scheduler
does with the pages. The Host write-back policy's eviction stores (§3.1) are
the third stream-ordered kind, for the same reason: the pages an admission or
a restore evicts fund the admissions and restores of the same plan. These are
the only stores that pay for their ordering on the forward's critical path,
and they have to: the pages are gone in the same round.

**Every other store pins its Device sources until the ack.** Boundary
publications of a live request and the finish-time flush (write-through only)
are issued with
`StoreSourceGuard::kPinnedUntilAck`: the ticket holds a `CacheBlockRef` on
each source, so the block stays cached and unevictable — the admission planner
cannot take it, `ClearDeviceCache` refuses, `NumNewlyReleasableLcmBlocks` does
not count it — until `CompleteWriteBack` publishes the Host entry and drops the
pin. Nothing else needs to know the copy is in flight, so the runtime copies
on its own stream and no forward waits. A cached block is never written again
by its owner (prefix reuse already depends on that), so the pin alone makes
the copy race-free. Both guards share one ticket (`StartPendingStores(guard,
cause)` drains the publication queue; `StartRetractionStores` builds the
image's L2 leg from the victim's tables; `StartEvictionStores` drains what the
plan's admissions and restores evicted), and the op carries `source_pinned` to
the runtime, which branches on the guard and never on the cause. Whatever its
guard or cause, the op is a `WriteBackOperation`, batched in the plan's
`WriteBackBatch` (`WriteBackOp` in Python) and acknowledged by
`CompleteWriteBack`: "write back" in these names means any copy from the
Device to Host L2, under either Host write policy, not only under
`write_back` (§3.1). The `HostStoreCause`
(publication, finish, retraction or eviction) names the drain point; it only
selects the Host L2 counters a store's outcomes land in, which no scheduling
decision reads. They count deterministic events only (no clock), so the
mirrored schedulers on every rank hold the same values.

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
Stream-ordered stores — both legs of an image, and eviction write-backs —
hold nothing and gate nothing, and an in-flight restore gates nothing: its
request is `Restoring`, invisible to `chooseVictim`.

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
Only candidates whose image fits are ranked, the Host L2 leg planned as the
retraction places it (the abort fallback above takes over when none does).
Exempt in both tiers: a request whose reserve already covers its whole
generation (`Request::ReserveCoversGeneration`) — retracting it frees exactly
what its restore must take back, pure thrash — and it is not aborted either,
except in the empty-round retraction below. Excluded by state: `Retracted`,
`Restoring`, `RemotePrefilling`.

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
victim; granted to nobody (the victim was the blocker and no other request
wants its pages), the victim can be restored ahead of the decodes and take
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
not the image-fit refusal (probed before the publication; a refusal after it
aborts the victim, see "does not fit" above). It runs at the start of the D
and fused grammars, before the restore phase, so the freed pages are granted
by the ordinary phases.

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

**Host L2: write-back.** With `host_write_policy` `kWriteBack`
(`--kvstore-write-policy write_back`; this role or the fused one (§3.3), no
L3, and in the runtime one pipeline stage and no
query shard) nothing streams to the Host at publication. Instead what an
admission evicts for the cache leaves by whole segments — or, for a request
alone that whole segments cannot fund, block by block (§1.4) — and is
written back in the same plan, by the drain right after the build (one
stream-ordered op, §2);
how a segment is recorded, placed, published or dropped is in [Cache
concepts](cache-concepts.md#the-coordinator-layer-csrccachecoordinator). Every
attention-TP rank of the replica runs the same plan and copies the rows it
owns, each Host block in its Device block's bucket (§5, "Every tier transfer
pairs blocks of equal residue"), and each round's head hands the scheduler
only the ACKs every rank holds (`CacheOps::acknowledged`, `event-loop.md`), a
rank that owns none of an op's rows included. Nothing depends on an ACK
arriving before the next plan is built, so it can land several plans later: at
overlap depth 1, or across a rank's idle rounds under attention DP. So the two
tiers together hold distinct content up to about the sum of their capacities,
less pinned and in-flight blocks, where write-through keeps on the Host a copy
of what the Device holds.

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

**Host L2: write-back.** With `host_write_policy` `kWriteBack` this role
writes back as the P role does (§3.1): nothing streams to the Host at a
publication — the prompt end at the first decode, each page a decode
completes, the finish, a retraction — and what an admission or a restore
evicts for the cache leaves by whole segments (an admission of a request
alone also block by block, §1.4), copied to the Host in the plan that evicts
it. A mixed batch's admissions and decode steps share the one
drain after the build. A segment's ACK can also land after a plan that only
restores (§4). A retraction image's L2 leg takes the published slots not on
the Host already as §2 plans them (whole segments first), and the snapshot
pool the rest. The fused role adds three shapes to what the P role sees:

- A decoding request holds the KV pages under its prompt end, so when the
  prompt ends on a page boundary that checkpoint boundary's segment can leave
  the Device only after the request finishes.
- A turn that decoded leaves two checkpoint boundaries, its prompt end and
  its finish; the next turn continues from one of them, and the other, once
  evicted, is written back like any other cache.
- An eviction in a plan that only decodes is written back on that plan's
  ordered stream ahead of its forward, unthrottled: the evicted pages are
  not pinned, the forward reuses them, so the copy cannot wait for a later
  plan.

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
  for any admission, and a failed restore leaves nothing allocated. Under the
  write-back policy the cache victims a restore evicts are written back in
  the plan that evicts them, as an admission's are (§3.1), ahead of the
  restore's own copies: the plan's stores run before its loads.
- An L2 slot whose key still has a Device-cached canonical block is
  **claimed** instead of copied (the same bytes by construction — the victim's
  own block, cache-only since it was freed — and protected from the planner's
  eviction like a prefix hit). Every other slot becomes one row of a single
  `SnapshotRestoreOperation`, tagged with its source tier (`HostTier::kL2`
  rows carry their key, `kSnapshotPool` rows none). Under segment eviction a
  restore whose claims leave no plan plans again claiming nothing, so every
  slot is a row (`cache-concepts.md`, `AdmissionPlanner`). A copy fills its
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
checkpoint exists only at a natural point or the opt-in in-forward
checkpoint (§1.2) -- so a hybrid model emits `BlockStored` for its prompt
end, its finish, its junction cuts and the in-forward checkpoint's page
boundary, never for another interior chunk end's chain hash; a tail's
descriptor carries the rows below `T` and the residency check counts a
tail's blocks from it. Retraction publishes twice
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
many rows plus the null row) are validated together. The P role refuses a
snapshot pool. L3 addresses page-cyclic sharded groups through replica-stable
object keys ([cache concepts](cache-concepts.md)).
`debug_force_retraction_interval` (§2) is the only
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

**Runtime contract.** `num_snapshot_pages` is explicit (`1` without a pool).
The runtime executes and counts snapshot and restore ops alongside L2 copies.
Query, GET, PUT and cancellation are Host-only variants of `CacheOperation`.
Their `StorageDone`, `StorageTick` and `StorageDetach` events enter the same
scheduler head feedback point. `Prefetching` is a request waiting state; it
does not add a second admission or forward path. `ExecutionPlan.aborts` reports capacity
aborts for the runtime to finish toward clients. The runtime and scheduler
must use matching contracts; releases follow the sequence in AGENTS.md.

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
  `onImageDoesNotFit` and nowhere else; the debug knob aborts only a victim
  whose image is refused after its publication, and leaves one its probe
  refuses running with nothing published.
- Freed capacity is granted to the request it was freed for in the same plan
  build whenever the round's grammar admits the grant, except that the
  empty-round retraction offers it to the round's decodes first (2); the
  image copies →
  page copies (`ExecutionPlan::pages_to_copy`) → zero → load → forward order
  on the forward thread's stream is what makes the immediate release safe --
  a tail hit's copy source may be evicted and re-granted by a later admission
  of the same round, as a fresh page or as another copy's destination -- and
  changing `DeviceHandle.execute`'s ordering breaks it. Within the copies,
  the arena reads each source byte range before any pair overwrites it
  (`CacheArena.copy_blocks` orders waves across all groups by their field
  byte ranges). This includes a sliding-window block whose LCM parent is
  re-granted to another group with different packing. Disjoint child ranges
  remain independent, so parent reuse alone does not form a copy cycle.
  Every source is read as it was when the plan was built. A copy destination
  is never zeroed.
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
  never neither (2). Only the two legs of a retraction image and the Host
  write-back policy's eviction stores are stream-ordered — their sources are
  granted away in the same round — and only such ops may be fenced ahead of
  the plan's page reuse by the runtime. A new store site chooses its guard
  explicitly (`StartPendingStores` has no default). Under write-back no
  store pins its sources.
- What an admission or a restore evicts under write-back is drained in the
  plan that evicted it (`StartEvictionStores`, right after the build); no
  record outlives its plan (by construction: only the build records, and that
  drain takes every record; `ClearDeviceCache`, which runs between plans,
  asserts none is left), and the segments of one plan that depend on each
  other — a later admission's segment taking pages an earlier one's
  checkpoint still needs — are merged before the Host places them, so they
  arrive together or not at all (3.1).
- Every tier transfer pairs blocks of equal residue (bucket): L2 store, L2
  prefix load, both image legs and the restore; the transfer manager asserts
  it when it builds a batch, so under page-cyclic sharding the rank that owns
  one end owns the other (`cache-concepts.md`).
- An admission loads Host entries only; L3 objects reach Host through a
  storage transaction issued before admission (1); its waiting request
  holds no Device page, no request-pool row and no head of line.
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
