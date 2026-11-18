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
identity (checkpoint, layout, attention TP, KV parallelism, backends,
quantization, compatibility epoch) and weight version, as the L3 key prefix
already defines it. Same namespace implies the same per-rank layout, so rank
*i* of the reader reads only from rank *i* of the holder. Replicas in
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
H2D load has not been acknowledged by every replica member.

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
Device pages is an unacknowledged prefix-load destination. Local forwards
can use these indexed pages through their layerwise H2D fences; remote reads
have no such ordering. A pin refused during a load falls back locally;
unrelated Device pages and published Host pages remain available. An accepted
answer queues the hard pin of exactly those pages; a pin that finds them
changed at the head holds nothing. The hard pin takes block references
(`CacheBlockRef`), so the pages can neither be evicted, relocated by
compaction nor moved by write-back. It has its own deadline, after which the holder's scheduler
releases it and leaves a tombstone.

**Read.** Readers pull: one-sided reads from the holder's registered Device
cache or Host pool into the reader's Device pages. The holder's data plane
takes no part. A reader starts its reads only once its own plan's device work
that precedes page reuse has run -- the write-back stores that still read
victims' old bytes, the copies, the zeroing -- the point PD's decode role
starts its receive from: a destination may be a block that plan freed, and
the device's reads of it are not ordered with the network's writes.

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

**A reader goes down:** the holder's soft and hard pins expire by their
deadlines; nothing leaks.

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
identity is the L3 deployment id (checkpoints, cache layout, numerics,
parallel layout) and the weight version. A weight update must flush and name
its `weight_version`, as with L3; once every rank holds the new weights the
replica serves and reads under the new identity and forgets the layouts it
knew, so replicas mid-rollout never read each other.

**A late write on the reader:** a read that failed or timed out may still have
work queued at the NIC. The reader keeps the destinations held for a
quarantine longer than the transfer engine's own timeout before it reports the
landing given up, so no late write lands in a page another request was given.
A read that succeeded retired every write before it returned.

**A flush on the holder** (a weight update) is refused while a hard pin holds
pages, as while a PD transfer is in flight, and proceeds once the pins end,
within one hard-pin lifetime. A soft pin holds nothing; a pin after the flush
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
servicer's endpoint (`host:port`, one per attention-DP replica), which the
router's hint carries; on first contact a reader fetches the holder's layout:
its incarnation and identity, and per rank its transfer engine session and
the field geometry of its Device arena and Host pool for every cache group.
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

The first version covers what write-back covers: fused and prefill roles, with
attention TP and KV parallelism; replicated cache groups only, no KV-page
sharding; no pipeline stages and no query context parallelism. It needs the
Host tier (`--enable-kvstore`): the Host is half of the capacity a holder
serves, and its layout is the field list both sides read by. A prefill
replica that reuses a peer's prefix then transfers to its decode peer through
the ordinary PD path. Hybrid models (sliding window, State checkpoints) are in
scope through the holder's matchers.

Out of scope: a storage tier, a central owner, the decode role, sharing across
namespaces, and moving PD itself onto the new read entry point.

### Configuration

`--peer-prefix-port` turns sharing on; attention-DP rank `r` binds
`port + r`, and a replica's name is `host:port`, with the host from
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
4. Larger scopes: KV-page sharding, other roles.

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
