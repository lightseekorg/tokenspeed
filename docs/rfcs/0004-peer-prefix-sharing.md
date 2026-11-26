# RFC 0004: Sharing cached prefixes between engines

Status: Draft for discussion.

## Summary

Let any engine reuse a prefix that another engine, or another attention-DP
replica of the same engine, has cached on its Device or its Host. The reader
reads the pages directly into its own Device cache with one-sided RDMA reads
(or NVLink within a node) and resumes the prompt from there; the holder only
pins the pages in its scheduler. There is no central index owner and no
storage tier: every engine keeps sole ownership of its own Device and Host
caches, which run the write-back policy, and the router (Dynamo) supplies a
hint of where a prefix lives. Every inconsistency between the hint, the
holder and the reader ends as a shorter reuse, never as wrong data or a
stopped engine.

## Motivation

Agent workloads return to the same context turn after turn, but the next
turn often lands on another replica: another attention-DP rank of the same
node, or another node. Today that replica recomputes a prefix that sits in a
peer's cache. Measurements of a shared pool on two GB300 nodes (PR #3421)
show the size of the gap: with the next turn on another rank, turns 2-4 drop
from 3.1-4.5 s to about 1.8 s TTFT and two thirds of the prefill work
disappears.

PR #3421 gets there with one owner process that indexes, places, pins and
evicts every page of a pool the prefill ranks share. That owner is on the path
of every lookup, holds the whole index, receives every write and touch, and is
a single point of failure. Its tree also sees only the lineage writes carry,
so pages without a State checkpoint arrive as unlinked roots. This RFC keeps
each page under the engine that computed it, where the existing Device/Host
coordinator already maintains a complete tree, and moves only bytes between
engines.

### Prior art

- **EvolvingInfra** (MIT HAN lab) shares host KV across engines the same way:
  each engine serves its own host tier through a lease servicer, readers pull
  with one-sided Mooncake Transfer Engine reads, a Dynamo indexer fed by KV
  events gives the router per-worker hits, and the router attaches a pull hint
  to the request. Its source copy is demoted once a peer has read it in full.
  It shares the host tier only, routes through a HiCache storage backend, and
  limits agent-aware retention to models with a FULL-only cache tree.
- **SGLang HiCache L3** and **vLLM's offloading connector** treat remote KV as
  a flat key-value store: content-hash keys, per-pool hit rules
  (`TRAILING_PAGES` for SWA and Mamba state; a sliding-window suffix scan in
  vLLM), and no dependency structure across keys. Eviction is the store's own.

## Design

### Terms

- **Holder**: the engine replica (engine, attention-DP rank) whose Device or
  Host cache holds the pages a reader wants.
- **Reader**: the replica running the request.
- **Lease**: the holder's record of one reader's claim on a prefix, with a
  soft and a hard phase (below). It is scheduler state on the holder.
- **Servicer**: the thread on the replica's first member that receives peer
  messages and answers them.

Holder and reader must share the cache namespace: the same deployment
identity (configured model and revision, layout, attention TP, KV parallelism, backends,
quantization, compatibility epoch) and weight version. Under `numerics=auto`,
the live peer identity allows different MoE expert-parallel sizes. Other
numerical policies and persistent L3 keys retain the expert-parallel size.
Same namespace implies the same per-rank layout, so rank *i* of the reader reads only from rank *i* of the holder. Replicas in
different namespaces never share: the two compare identities when they first
connect and before every read (Faults, below).

### Invariants

1. **Each page has one owner.** A holder's pages are indexed, pinned and
   evicted only by the holder's scheduler. No other process decides about
   them. The holder's `BoundaryTree` stays complete because every page enters
   it through the holder's own publication or write-back.
2. **Hints are advisory.** The router's directory may be stale. A request
   never depends on it for correctness; a wrong hint costs one lookup.
3. **Bytes are kept only when the holder vouches for them.** The reader keeps
   a landing only after the holder answers that the read pages stayed
   untouched from the pin to the release. Anything else is recomputed.
4. **One path.** A landed prefix becomes ordinary Device entries before the
   forward that uses it. Admission, junction cuts, retraction, PD and
   write-back see a prefix the request already holds; no mode-specific
   forward or scheduling path exists.
5. **Mirrored ranks agree.** Every change a message makes on the holder, and
   every accepted landing on the reader, is applied by every member at a
   round's head, from facts the replica agreed on, in one order.
6. **No message waits for a round.** A holder answers between its rounds,
   while its scheduler is idle; only what an answer changes waits for the
   next head.

### Request flow

```
holder H (Device + Host, write-back)                 reader R
                                                     request arrives with a peer hint (H)
   (1) LOOKUP ◄──────────────────────────────────────  sent before admission, never awaited by a plan
       answered at once: boundary and pages
       (slot, tier, block id); soft pin at H's next head
                                                     (2) admission allocates Device pages for the
                                                         peer span; the request waits out of
                                                         forward candidates
   (3) PIN ◄─────────────────────────────────────────  from R's runtime once its plan is dispatched
       answered at once: are the pages still those;
       the hard pin is applied at H's next head
                                                     (4) READ, rank i from H's rank i, Device or Host
                                                         source -> R's Device pages, once R's plan's
                                                         device work is done
   (5) RELEASE ◄─────────────────────────────────────  after every rank's reads retired; asks to
       answered at once: were the pages untouched        demote after a complete read
       from the pin; release and demotion at H's next head
                                                     (6) a vouched landing: the request computes the
                                                         rest, and its landed pages publish as computed
                                                         ones do; anything else: admitted again locally
```

**Answers between rounds.** A holder's control thread blocks in each round
while it waits for the previous forward (`complete`), which for a long
prefill chunk is hundreds of milliseconds; answering at the next round would
make every lookup and pin wait that long. The scheduler is therefore shared
with the servicer under a lock that the control thread holds through a
round's scheduler work and releases while it waits on the device or on an
idle frontend. A loop waiting on a transfer -- a prefill replica on its PD
sends, a landing on its reads -- does neither and spins through
empty rounds, so the control thread also hands the lock to a waiting servicer
once per round. The servicer answers with reads only (`AnswerPeerLookup`,
`PeerPagesMatch`, `PeerLeaseHeld`, `PlanCalls`): an answer costs a network
round trip plus at most the rest of a round's scheduler work.

**Changes at the head.** What a message changes -- a lookup's soft pin, a
pin's block references, a release's demotion -- is not applied by the
servicer: applied on the first member alone, between a plan and the forward
results the round feeds back, it would interleave differently from the
other members' and their access epochs, hence their eviction orders, would
diverge. The servicer queues each change instead; the round's relay carries
the queue to every member, and each applies it at the next head before any
other event, in the order the answers were given (`advance_peer`).

The answers hold until then because nothing leaves a holder's cache outside
a plan: eviction, dead-entry collection and compaction run only while one is
built, and the next plan is built after the head that applies the pin. A
member whose data plane still runs the plan dispatched before the answer
touches no page an accepted pin named: that plan's evictions were already in
the scheduler state the answer read, and the pin rejects Device pages whose
H2D load has not been acknowledged by every replica member, and decoded
endpoints a retraction published whose deferred state commit has not been
acknowledged (below).

**Lookup and soft pin.** The holder probes its own Device and Host prefix
with the ordinary matchers, so a hybrid model's reply is already cut to a
boundary its SWA window and State checkpoints can resume from, and lists the
pages a reader's table would hold at each slot. A soft pin holds no block
reference: the lookup's change refreshes the entries, as a hit does, so they
leave last, and keeps the prompt up to the answered boundary. A soft pin
expires after a short deadline (hundreds of milliseconds). Soft pins never
cost the holder capacity it needs.

**Admission.** The reader admits the request against its own Device
capacity, treating the peer span as tokens that will arrive rather than be
computed. The local hit is cut back to a page boundary, the block boundary of
every group: the block holding a tail hit's end would otherwise be filled
both by the admission's copy or load and by the landing. The landing's
destinations leave the plan's pages to zero, since the landing fills them
whole.

**Pin.** The servicer checks that the lookup's prompt, up to the answered
boundary, still resolves to exactly the answered pages and that none of its
Device pages is an unacknowledged prefix-load destination. Nor may it be a
decoded endpoint that a retraction published in a State group whose decode
state the runtime commits only at that retraction
(`CacheGroupConfig::deferred_decode_state`, KDA): the plan carries a
`state_commit_fence`, and the pin waits for its `StateCommitDone`. Local
forwards can use these indexed pages through their layerwise H2D fences and
device order; remote reads have no such ordering. A pin refused during a load
or a commit falls back locally;
unrelated Device pages and published Host pages remain available. An accepted
answer queues the hard pin of exactly those pages; a pin that finds them
changed at the head holds nothing. The hard pin takes block references
(`CacheBlockRef`), so the pages can neither be evicted, relocated by
compaction nor moved by write-back. Coordinated holders release expired pins
and leave a tombstone. Independent holders use the liveness policy below.

**Read.** Readers pull: one-sided reads from the holder's registered Device
cache or Host pool into the reader's Device pages. The holder's data plane
takes no part. A reader starts its reads only once its own plan's device work
that precedes page reuse has run -- the write-back stores that still read
victims' old bytes, the copies, the zeroing -- and the plan's loads of its
local Host hit: a destination may be a block that plan freed, and the
device's reads of it are not ordered with the network's writes; and the
landing's end releases the request to a later plan's forward, which does not
wait for this plan's loads. PD's decode role starts its receive from the same
point. The landing waits on separate completion events for the default and
load streams after the current forward is launched. The default stream does
not wait for the entire load, so that forward keeps its layerwise load overlap.

**Release.** The servicer answers whether the pages stayed untouched from the
pin: either no plan ran on the holder since the pin's answer, or the hard pin
the head applied holds them now. It queues the release, and the demotion
when the reader asked for it and the answer was yes. A landing succeeds only
whole: the pin's answer confirmed the pages, every rank of the reader's
replica group read every page, and the release vouched for them. The request
then leaves `PeerLanding` as a request that computed the landed span itself:
it prefills on from the landing's end, and its next publication point
publishes the landed pages as it publishes computed ones. A State group's
live slot holds the holder's checkpoint at that end, which the next chunk
rotates in, as it would a forward's.

**Fallback.** Any shortfall (a stale hint, a soft pin that lapsed, changed
pages, a failed or late read, a disagreeing rank, an unvouched release, a
holder that stopped answering) gives the landing up whole: its pages go back
and the request is admitted again, as an ordinary prompt that asked its
holder already, from its local hit. A partial landing is not kept: a State
group's slots and a sliding window's retained blocks are shaped for the
admitted boundary. No failure stops the engine.

### Faults

**A holder goes down** at any step: a lookup without an answer by its
deadline admits locally; a pin, a read or a release without an answer by its
deadline gives the landing up. Its leases go with it.
Outstanding layout queries are coalesced only until their answer deadline;
a send failure clears the query immediately. Later landings can ask again
after a lost request or reply, and a weight update clears outstanding queries
along with the cached layouts.

**A reader goes down.** Soft leases expire on both drivers. Coordinated
holders expire hard pins under their existing timing contract. Independent
holders keep hard references until RELEASE or transport teardown. The
autonomous liveness policy below reports abandoned pins to the launcher.

**A holder restarts** under the same name: its answers and layout carry its
incarnation, and a reader drops a layout, and any answer, of another
incarnation before it reads; a lease of the dead holder is unknown to the new
one, whose answers then report changed pages or nothing held.

**A reader restarts:** its lease ids count from its incarnation
(`PeerConfig::incarnation`), so no lease of the new reader meets a tombstone
or a lease the holder kept for the old one.

**Different models or weights:** every answer carries the holder's namespace
identity; a reader reads only from a holder whose identity equals its own.
Equal token hashes under different weights would otherwise land wrong KV. The
identity includes the deployment facts described above and the weight version. A weight update
must flush and name its `weight_version`, as with L3; once every rank holds the new weights the
replica serves and reads under the new identity and forgets the layouts it
knew, so replicas mid-rollout never read each other.

**A late write on the reader:** a read that failed or timed out may still have
work queued at the NIC. The reader keeps the destinations held for a
quarantine longer than the transfer engine's own timeout before it reports the
landing given up, so no late write lands in a page another request was given.
A read that succeeded retired every write before it returned.

**A flush on the holder** (a weight update) is refused while a hard pin holds
pages, as while a PD transfer is in flight, and proceeds once the pins end. A soft pin holds nothing; a pin after the flush
finds the pages changed.

### Overlap

The design holds under the overlap schedule. On the holder, the servicer
answers while the control thread waits for forward N-1, after plan N is built
and dispatched: plan N's evictions are in the state it reads, the forward
running on the device writes only request-owned blocks, and plan N+1 is built
after the head that applies the pin. On the reader, a landing request runs no
forward, so none of its pages is written by one; it is no retraction victim,
and a lone request's compaction, which needs every other request to hold no
pages, never runs while it lands.

### Transfer

The transfer uses Mooncake Transfer Engine, not Mooncake Store. TokenSpeed's
C entry points (`transfer_engine_tokenspeed.h`, loaded by `ts-mooncake`) gain a
batched one-sided read beside the existing writes. Each engine registers its
Device cache arena and its Host pool once at startup.
Each reader rank runs at most four reads concurrently through the engine's
thread-safe entry points. Workers share a queue and report each job only
after its synchronous read returns; queued jobs retain their landing's pages
too. Shutdown drains the queue and joins every worker before the transfer
engine can unregister its memory.

- Across nodes: RDMA, GPUDirect where the NIC and GPU allow it.
- Within a node, between attention-DP replicas: NVLink (or the node's
  peer-to-peer path), selected by the transfer engine from the two endpoints.

Control messages (lookup, pin, release) travel between the replicas'
servicers over ZMQ, not through the data plane. A holder's name is its
logical endpoint on its node's shared control port, which the
router's hint carries; on first contact a reader fetches the holder's layout:
its incarnation and identity, and per rank its transfer engine session and
the field geometry of its Device arena and Host pool for every cache group.
The first member owns the bound listener continuously through startup. A
caller using an OS-selected port binds before constructing the replica's name
and scheduler, then moves that listener into the peer without rebinding.
Until the shim's NVLink path is wired, replicas on one node also read over
RDMA (the NIC's loopback); only the router's cost model tells the two apart.

### Copies and eviction

After a fetch the reader keeps its copy; it is an ordinary Device entry under
the reader's write-back policy.

The reader decides when the holder's copy becomes the spare one: once it has
read the whole confirmed span and published it, its release asks the holder to
demote the leased path. A release after a partial read or a fallback asks for
nothing, and the holder's copy keeps its rank. The router takes no part in this
decision. From the deepest leased node toward the root, a node is demoted only if
it has no child outside the leased path and no local access since the lookup;
the walk stops at the first node that fails. Demotion therefore marks a leaf
chain, and evicting it never opens a gap under another branch (under
write-back a gap makes everything below it dead, `cache-concepts.md`).

The chain is demoted as a superseded State checkpoint already is
(`CacheCoordinator::DemoteEntries`): each entry is dated back to the oldest
access epoch and marked demoted, so it leaves first on its tier; an entry
something holds is skipped; any later local hit, touch or publication dates it
forward and ends the demotion. A demoted Device entry is never written back to
the Host (`DemotedEntriesNotWrittenBack` counts them): a peer took a copy, and
writing it to the local Host would spend Host capacity and D2H bandwidth on a
duplicate. An engine with peer sharing enables these demotion marks.

### Router (Dynamo)

**Directory.** Engines already publish KV events. Each event gains the medium
it concerns (Device, or Host pinned as `CPU_PINNED`), which the Dynamo indexer
already models as `Placement { owner: (worker, dp rank), tier }`; an engine
with Host L2 publishes its Host blocks whether peer sharing is on or not.
Each medium's stream is prefix-closed for a router that continues a Device
match into the Host tier: a Device block is out once its parent is out on the
Device, a Host block once its parent is out on either medium. Under
write-back a Device eviction that writes back appears as a Device removal and,
once the copy lands, a Host store. The indexer needs no dependency structure;
the holder enforces it.

**Scoring.** Worker scoring today credits a candidate's own Device and Host
overlap and discounts prefill by shared-cache hits beyond its Device prefix.
Peer reuse adds one more credit: for candidate `w`, the best holder `h != w`
whose prefix extends `w`'s own by `Δ` blocks contributes

```
peer_credit(w, h) = Δ · block_size · weight − fetch_cost(Δ, w, h)
fetch_cost        = fixed(link) + per_token(link) · Δ · block_size + copy_price · Δ · block_size
link              = same node (NVLink)  |  different node (RDMA)
```

with separate parameters per link, so a sibling DP replica's prefix is worth
nearly its full length and a remote node's is discounted by the network.
`copy_price` makes the router prefer placing a request where its prefix
already is over copying it, which keeps duplicates down. Node membership comes
from worker registration.

**Hint.** The chosen placement travels in the request's routing hints: the
holder (worker id and DP rank) and the expected peer tokens. A worker without
the capability ignores the hint. A reader may also skip a hint whose expected
gain falls below a minimum fetch size.

### Engine roles and scope

The first version covers every role, with
attention TP and KV parallelism; replicated cache groups only, no KV-page
sharding; no pipeline stages and no query context parallelism. It needs the
Host tier (`--enable-kvstore`): the Host is half of the capacity a holder
serves, and its layout is the field list both sides read by. A prefill
replica that reuses a peer's prefix then transfers to its decode peer through
the ordinary PD path. A decode replica holds and never reads: it serves the
history it decoded, prompts and replies, so the next turn's prefill replica
reads a reply instead of computing it, and its own prompts come from its
prefill peer (RFC 0005, which also gives the decode role a Host tier under
write-back). Hybrid models (sliding window, State checkpoints) are in scope
through the holder's matchers.

Out of scope: a storage tier, a central owner, sharing across namespaces, and
moving PD itself onto the new read entry point.

### Configuration

`--peer-prefix-port` turns sharing on; each node binds one public port. Rank zero is
`host:port` and other attention-DP replicas are `host:port/rank`. The first
local replica leader forwards control frames to the addressed sibling over
loopback. The host comes from
`--peer-prefix-host` (the address the worker advertises to the router) or
else the detected local address. A holder hint must name the holder exactly
so: a reader matches the holder's answers by that name.
`--peer-prefix-answer-timeout-ms` bounds each wait for the holder (lookup,
layout, pin, release) and `--peer-prefix-min-fetch-tokens` the smallest read
worth a landing. Lease lifetimes, the lease limit and the read quarantine are
fixed (`cache/peer.py`). A process draws a random nonzero 32-bit incarnation
at startup, which its attention-TP members share. The engine host starts one
transfer engine per rank with the Device arena and the Host pool registered,
and the replica's members exchange their geometries over the rank mesh before
serving.

### Observability

Per replica: lookups sent and answered, soft-pin losses, hard-pin shortfalls,
fetched tokens by link, fallback tokens recomputed, read latency by link,
leases held by tier, demoted and dropped blocks and their returns.

## Phases

1. Transfer engine read entry point and registration of the Device arena and
   Host pool. Holder lease (soft and hard), reader fetch with explicit
   holder in the request, boundary fallback. Two-replica tests on one node and
   across two nodes; token-identical output with and without reuse.
2. Medium in KV events; Dynamo peer credit, link cost models and hints.
3. Holder demotion of the leaf chain a complete read released.
4. Larger scopes: KV-page sharding. The decode role as a holder came with
   RFC 0005.

## Testing

- Scheduler (C++): answers that change nothing, changes applied at the head,
  a pin of changed or still-loading pages, a release across a plan with and
  without the pin, lease phases and expiry, fallback to a local admission at
  every step, an abort while landing, a tail hit cut back to a page so no destination is a
  copy or load destination, demotion of a leaf chain and its stop at a branch
  or an access, on full-history and hybrid (sliding window and State) group
  sets.
- Engine host (Rust): control messages and deadlines over real sockets,
  recovery after a lost layout query, bounded concurrent reads and draining
  queued reads on shutdown.
- GPU: reads from Device and from Host sources, on one node and across nodes;
  reader and holder killed mid-transfer; output identical to recomputation.
- Router: scoring with each link model, hint round trip, stale directory.

## Open questions

1. Soft-pin and hard-pin deadlines.
2. Whether a landing should keep a shorter confirmed prefix for groups whose
   tables do not depend on the boundary (full history only).

### Observing peer reuse

With engine metrics enabled, the attention-DP owner records one
`tokenspeed:peer_consumed_tokens` sample per finished hinted request. Consumption
is the intersection of the confirmed remote interval with the prefix skipped by
the first committed prefill. Local prefix tokens and replayed tokens are excluded.
If the router supplied `peer_expected_tokens`, the same request also contributes
`expected - consumed` to `tokenspeed:peer_estimation_error_tokens`, labeled by
outcome. An older hint with no estimate still contributes consumption.

`tokenspeed:peer_stage_seconds_count` counts visits by stage and outcome.
Inspect lookup, admission, layout, pin, read_agreement, release and quarantine separately.
The lookup_to_ready and landing timers span multiple stages. hard_pin uses the
holder's clock, from PIN application until actual release or expiry.
`tokenspeed:peer_hard_pin_remaining_seconds` measures deadline minus that time.
`tokenspeed:peer_shard_seconds` measures device-fence wait, read queue delay and
transport service on each rank. Sum `tokenspeed:peer_outstanding_read_bytes`
over reader ranks for bytes queued or executing against a holder; count distinct
reader replicas with a positive sum to obtain concurrency at scrape time.
Use `tokenspeed:peer_read_bytes_total` for successful physical payload;
`tokenspeed:peer_read_source_bytes_total` splits it, and the Disk pages a holder
pushed with it ([RFC 0006](0006-disk-tier.md)), by the holder's `source_tier`:
`device`, `host` or `disk`.
Scheduler gauges track leases, hard-pinned blocks and requests looking or landing.
A reader process that dies before terminal observation cannot report a completed
sample; frontend failures and holder expiry remain independently observable.

### Protocol progress by replica topology

When peer prefix sharing is enabled, the runtime selects the protocol owner
from the number of members in the replica and the scheduler's configured
landing tier. A multi-member replica advances transfers in coordinated rounds.
Each shard owns part of the KV and reports its fence and READ outcome before
the replica releases the source or publishes the prefix. A single-member
replica uses a peer worker for Host or Device landings. Multiple independent
replicas can share a machine and still use the single-member path.

### Single-member Host landings

The scheduler issues a revocable Host reservation grant for the lookup and
consumes the terminal result in a normal round. The protocol worker tries the
cache lock to reserve unpublished destinations up to the granted boundary.
The lookup deadline applies to reply arrival. After a timely hit the worker retries
cache contention and unavailable Host credits, and the normal admission pass may
schedule the prompt locally until a destination is reserved. Committing local
admission or cancelling the request revokes the outstanding grant. Tier copies
and peer reads may overlap on separate pages. Each ticket retains its existing
block references through DMA retirement. Publishing a duplicate key adopts the
canonical Host reference. An aborted transfer releases its unpublished blocks
on retirement. Only the scheduler can
publish a completed landing for its owning request.

The existing servicer timestamps and forwards incoming messages to the existing
Host protocol worker. A pending lookup, PIN or reservation yields on cache
contention. RELEASE uses a capability bound to the hard lease's existing
source references. It drops their strong counts before acknowledging, including
while DP is waiting in a forward. Only the cache owner runs uniqueness observers,
demotes cache entries and returns unused storage to its pool. Duplicate releases
receive the same answer during the record lifetime. No extra thread is needed.

During shutdown the independent worker drops queued reservations and grants,
cancels landings still waiting to start a read, and drains submitted reads
and active holder pins. It continues serving RELEASE while draining.

### Single-member Device landings

Device (HBM) landings use adaptive admission and checkpoint-validated PIN.
The scheduler grants the reader
pages in a normal admission round. The peer worker starts the PIN exchange
only after the grant's Device fence. Its lookup deadline is measured at reply
arrival. The normal plan consumes the completed READ and RELEASE.

Adaptive admission probes the local prefix again when the request reaches
its place in the prefill queue. It compares the compute saved on the additional
prefix with the estimated exposed wait for a landing, including failed attempts.
The estimate starts with experimental Lite17B values and updates from retired
attempts on this reader, grouped by source tier and locality. Successful
samples price the interval from writable pages through RELEASE acknowledgement.
The first sample replaces the prior. After eight samples the update weight is
1/8. The scheduler separately measures dispatch to writable pages and RELEASE
to the first scheduled prefill round. It adds the median combined local wait
of the last five completed admissions with the same batch condition (empty,
partially occupied, or full), then subtracts the estimated time before a local
prefill slot once. Before five such samples it assumes zero local delay.
On an idle reader, a stable request-ID hash assigns one in sixteen eligible
borderline cases to try a READ when a fourfold cheaper landing would clear
the same 20 ms margin. Fresh local matches and capacity checks still apply.
Regular local
admission may commit first; a request still waiting for capacity can be
reconsidered next round. The existing block references hold committed landing
destinations through transfer retirement and cancellation.

The reader includes the original checkpoint in the PIN. If the lookup's
soft lease has expired, the holder resolves the same checkpoint and acquires
references in the same exchange. The reader checks the holder incarnation,
namespace, prefix length and full source-page mapping against the lookup
and its writable grant before issuing DMA. Eviction or a changed mapping
falls back to local prefill without a second lookup. While the holder has
unretired replay work, both ordinary and checkpoint PINs reject any mapping
containing Device pages. Host-only mappings remain eligible.

Coordinated replicas keep their existing namespace. Independent Host and Device
replicas use `async-host-v2:` and `async-device-v3:` respectively. During a
rolling upgrade, endpoints with different protocol versions fall back to local
prefill. A matching version is required on both sides for hard-pin liveness.

An autonomous reader retries a missing RELEASE acknowledgement once. A READ
still running after one third of the record lifetime renews its existing hard
pin and repeats at that interval until retirement. Short reads send no renewal.
A holder that receives neither renewal nor RELEASE for a full record lifetime
(60 seconds by default) reports an engine error identifying the lease. Recovery
requires replacing the holder process and its registered transport session.
The error leaves the native pin in place so a timer cannot recycle live DMA pages.

After 30 seconds without an admitted attempt on a degraded Device path, an
idle reader may probe a prefix that met the original cold-start gain threshold.
Per-request cost and feedback details are logged at debug level.

Reader decision details are available in `tokenspeed:peer_decisions_total`.
Local admission and no-attempt reasons count once at request termination and
describe the last observed choice. Incompatible lookup replies are classified
when received, using their existing progress protocol and namespace identity.

Each holder owner exports `tokenspeed:peer_holder_events_total` on its own
replica. The outcomes are lookup hit or miss, PIN accepted or rejected, and
RELEASE after a successful READ, cleanup, or rejection. A read release
confirms that the holder accepted the reader's success indication. Only the
reader observes how much KV its following forward uses. These metrics need
no changes to the peer messages.

At first prefill dispatch, the total prefix hit ratio includes the confirmed
Device peer interval that the forward skips beyond the reader's local hit.
`tokenspeed:prefix_peer_hit_tokens_total` counts these additional tokens.
A peer landing on Host already contributes through the normal Host cache hit.
A failed Device landing contributes only the local retry's hit and prompt.
Decode's local admission ratio measures its own cache independently.

On the independent reader, `lookup_reply_to_scheduler` times a positive reply
from arrival to collection by the control loop. Device landings with complete
timestamps also expose `grant_to_writable`, `writable_to_terminal`, and
`terminal_to_scheduler` through the stage histogram. The writable-to-terminal
interval includes the protocol work that follows the destination fence.
