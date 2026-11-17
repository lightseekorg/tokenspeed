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

Holder and reader must share the cache namespace: the same deployment
identity (checkpoint, layout, attention TP, KV parallelism, backends,
quantization, compatibility epoch) and weight version, as the L3 key prefix
already defines it. Same namespace implies the same per-rank layout, so rank
*i* of the reader reads only from rank *i* of the holder. Replicas in
different namespaces never share.

### Invariants

1. **Each page has one owner.** A holder's pages are indexed, pinned and
   evicted only by the holder's scheduler. No other process decides about
   them. The holder's `BoundaryTree` stays complete because every page enters
   it through the holder's own publication or write-back.
2. **Hints are advisory.** The router's directory may be stale. A request
   never depends on it for correctness; a wrong hint costs one lookup.
3. **Bytes are published only after the lease is confirmed.** The reader
   publishes fetched pages only after the holder confirms the hard pin held
   throughout the read. A page that cannot be confirmed is recomputed.
4. **One path.** A peer hit becomes ordinary Device entries before the
   forward that uses them. Admission, junction cuts, retraction, PD and
   write-back see a prefix the request already holds; no mode-specific
   forward or scheduling path exists.
5. **Mirrored ranks agree.** Every pin, release and demotion on the holder,
   and every accepted fetch length on the reader, is applied through the
   control plane in a plan, from facts all replicas of the replica group have
   agreed on.

### Request flow

```
holder H (Device + Host, write-back)                 reader R
                                                     request arrives with a peer hint (H, expected tokens)
   (1) LOOKUP ◄──────────────────────────────────────  sent before admission, never awaited by a plan
       H matches its Device and Host prefix,
       takes a soft pin, replies a manifest
       (boundary, per page: tier, per-rank address)
                                                     (2) admission allocates Device pages for the
                                                         peer span, as for a miss; the request waits
                                                         out of forward candidates
   (3) HARD PIN ◄────────────────────────────────────  from R's admission plan
       H turns the soft pin hard (what survived),
       replies the confirmed boundary
                                                     (4) READ, rank i from H's rank i, Device or Host
                                                         source -> R's Device pages
   (5) RELEASE ◄─────────────────────────────────────  with per-page completion and, after a
       H drops the pins, replies whether the              complete read, a request to demote
       hard pin held throughout; demotes the
       leased path when asked
                                                     (6) a complete landing: the request computes the
                                                         rest, and its landed pages publish as computed
                                                         ones do; a short one: admitted again locally
```

**Lookup and soft pin.** A lookup reaches the holder's control plane and is
applied in the holder's next plan, on every rank. The holder probes its own
Device and Host prefix with the ordinary matchers, so a hybrid model's reply
is already cut to a boundary its SWA window and State checkpoints can resume
from. A soft pin holds no block reference: the lookup refreshes the leased
entries, as a hit does, so they leave last, and the lease keeps the prompt up
to the answered boundary. If pressure evicts a soft-pinned page anyway, the
hard pin finds a shorter prefix. A soft pin expires after a short deadline
(hundreds of milliseconds). Soft pins therefore never cost the holder capacity
it needs.

**Admission and hard pin.** The reader admits the request against its own
Device capacity, treating the peer span as tokens that will arrive rather
than be computed, and sends the hard pin from that plan. The hard pin takes
block references (`CacheBlockRef`) under the lease, so the pages can neither
be evicted nor moved by write-back while the read runs. Under write-back a
page is on the Device or on the Host, never in flight between them once
pinned; the manifest names the tier it is read from. The hard pin has its own
deadline, after which the holder's scheduler releases it.

**Read.** Readers pull: one-sided reads from the holder's registered Device
cache or Host pool into the reader's Device pages. The holder's data plane
takes no part. Only published pages are leased, so their contents are final
and no device fence is needed on the holder.

**Release.** The holder releases the lease and replies whether the hard pin
held for its whole life. A landing succeeds only whole: the hard pin confirmed
the admitted boundary, every rank of the reader's replica group read every
page, and the pin held. The request then leaves `PeerLanding` as a request that
computed the landed span itself: it prefills on from the landing's end, and
its next publication point publishes the landed pages as it publishes computed
ones. A State group's live slot holds the holder's checkpoint at that end,
which the next chunk rotates in, as it would a forward's.

**Fallback.** Any shortfall (a stale hint, a soft-pin loss, a shorter hard
pin, an expired pin, a failed read, a disagreeing rank) gives the landing up
whole: its pages go back and the request is admitted again, as an ordinary
prompt that asked its holder already, from its local hit. A partial landing is
not kept: a State group's slots and a sliding window's retained blocks are
shaped for the admitted boundary, and a shorter one would need them reshaped.
No failure stops the engine.

**Late reads.** A reader that gives up still holds its destination pages until
its reads retire; the holder's hard pin keeps the source pages valid until the
release or the deadline, so a late read never observes reused memory.

### Transfer

The transfer uses Mooncake Transfer Engine, not Mooncake Store. TokenSpeed's
C entry points (`transfer_engine_tokenspeed.h`, loaded by `ts-mooncake`) gain a
batched one-sided read beside the existing writes. Each engine registers its
Device cache arena and its Host pool once at startup.

- Across nodes: RDMA, GPUDirect where the NIC and GPU allow it.
- Within a node, between attention-DP replicas: NVLink (or the node's
  peer-to-peer path), selected by the transfer engine from the two endpoints.

Control messages (lookup, hard pin, release) travel between the engines'
control planes over the existing ZMQ wire, not through the data plane.

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
it concerns (Device or Host pinned), which the Dynamo indexer already models as
`Placement { owner: (worker, dp rank), tier }`. Under write-back a Device
eviction that writes back appears as a Device removal and a Host store. The
indexer needs no dependency structure; the holder enforces it.

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
sharding; no pipeline stages and no query context parallelism. A prefill
replica that reuses a peer's prefix then transfers to its decode peer through
the ordinary PD path. Hybrid models (sliding window, State checkpoints) are in
scope through the holder's matchers.

Out of scope: a storage tier, a central owner, the decode role, sharing across
namespaces, and moving PD itself onto the new read entry point.

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

- Scheduler (C++): lease phases and expiry, soft-pin loss, hard-pin
  shortfall, fallback to a local admission at every step, an abort while
  landing, demotion of a leaf chain and its stop at a branch or an access, on
  full-history and hybrid (sliding window and State) group sets.
- Engine host (Rust): control messages and deadlines over real sockets.
- GPU: reads from Device and from Host sources, on one node and across nodes;
  reader and holder killed mid-transfer; output identical to recomputation.
- Router: scoring with each link model, hint round trip, stale directory.

## Open questions

1. Soft-pin and hard-pin deadlines.
2. Whether a landing should keep a shorter confirmed prefix for groups whose
   tables do not depend on the boundary (full history only).
