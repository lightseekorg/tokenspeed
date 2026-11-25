# RFC: Dependency-aware publication and GDS eviction for hybrid cache management

Status: Draft

## Summary

Make a recoverable hybrid prefix the unit of retention policy. A checkpoint
at token position T is useful only with its State components, attention
history, partial KV tail, and required sliding window at that same position.
Track these dependencies explicitly, and use them both when publishing a
recovery point and when deciding which cached allocations can leave.

Add `--prefix-cache-evict-policy {lru,gds}`, with `lru` as the default.
`lru` preserves the existing aged/retention ranking, segment selection,
cascades, block minimization, Host allocation and write-back settlement.
Default and explicit LRU must agree, and adding GDS must not change that
baseline behavior.
With Disk, both policies retain the separately implemented RFC 0006
clean-before-dirty Host placement. `gds` implements a FlashStates-inspired GreedyDual-Size
policy: rank recovery points by the recomputation they avoid per unit of
capacity their removal actually releases in the pressured tier, with watermarks that age
unused entries. Recompute that value when sharing or fallback dependencies
change.

The GDS implementation covers hybrid logical retention across Device and Host, existing
natural checkpoints, the existing in-forward production path, L2
write-through/write-back, write-back L2 with Disk, and the current peer prefix
paths. Both memory tiers may
order physical placement candidates differently, but neither may independently
destroy a surviving recovery route. Published Disk copies participate in
route discovery and valuation. Disk retains its existing LRU/retention policy;
unallocated Disk space or a queued cleaning write is not a preservation
guarantee. Separate logical
selection from physical reclamation and write-back construction, within the
existing scheduler, allocator, and execution path. The sections below specify
the target architecture and required refactoring.

## Motivation

A hybrid model cannot resume from arbitrary cached KV alone. Its recurrent
State summarizes the prefix at a particular position, while attention layers
still need history or a window. A State checkpoint without that history is
unusable. History with no compatible checkpoint may occupy capacity without
allowing the model to skip that part of prefill.

Agent and resampling workloads expose this mismatch. A request can return
after a long tool call, branch from a shared prompt, or sample repeatedly
from a shared context. Several recovery points may share most of their
KV. Removing a checkpoint may release only its State because other recovery
points still need its KV; removing the last checkpoint that needs a suffix
can free much more. Removing a shallower
checkpoint can also make a deeper one more valuable: the next fallback is
now farther away. Recency alone does not describe either effect.

Publication matters for the same reason. A useful recovery point must be
captured while all of its components represent the intended token position.
The final recurrent State cannot be relabeled as an earlier checkpoint, and
a final sliding window may already have overwritten rows needed there.
Publishing more checkpoints without a retention decision can trade saved
prefill for excessive State storage and export work. Evicting independent
blocks without tracking recovery can discard the value those exports created.

Host is an intended extension of the working cache: Device pressure should
be able to preserve useful recovery points there. Disk extends recoverability
further, but repeated SSD reads must not become a free substitute for memory
residency in the valuation. Keep a positive Disk read penalty to discourage
retaining points whose apparent savings depend on frequent, substantial Disk
reads, while allowing ordinary recovery through Host.

### Baseline behavior

TokenSpeed already provides important parts of this design:

- Natural publication supports non-block-aligned boundaries, including
  prompt-end and finished frontiers, subject to the model's numerical
  checkpoint contract and existing suppression rules.
- Existing in-forward checkpoints select a target, shape chunks, allocate
  extra output blocks, and run backend-specific production before publishing.
  The enable flag also implies endpoint filtering and retention hints; their
  separation is part of this proposal.
- Prefix matching checks the required groups. LCM allocation, canonical
  cache entries, State slot rotation, and transfer ownership are established
  mechanisms that this proposal retains.
- Device pressure selects candidates using LRU/retention. With eviction
  write-back enabled, dependency rules expand a candidate into a segment;
  allocation planning checks whether it satisfies demand, then minimizes
  unnecessary victims before committing.
- [RFC 0006][disk-rfc] and its implementation are already in the baseline.
  BoundaryTree judges gaps across Device, Host, and Disk. Disk prefetch fills
  missing Host components through the ordinary Host-suffix matcher.
  Cleaning queues follow write-back ACK order in the implementation; Host
  victim selection does not yet implement RFC 0006's clean-before-dirty
  invariant. GDS must close that gap rather than assume the invariant is
  already enforced.

That last mechanism is already more than independent block LRU. However,
its candidate value is still recency/retention, and Device segment selection
is coupled to the write-back setting. Segment expansion does not measure
saved recomputation per released capacity. A request ID cannot serve as the
ownership key for shared prefixes. The required change is to price recoverable
content and its actual reclaimable contribution, independently of whether L2
is enabled.

## Design

The design has four responsibilities: produce and publish exact checkpoints,
establish their recovery dependencies, value their retention, and commit
capacity changes under existing ownership rules. The flows below compare
the baseline with GDS; default LRU keeps its existing planner and settlement.

```text
Before
  coupled production / publication / retention settings
    -> produce and publish checkpoint components
       +-> write-through: ordinary Host admission
       +-> Device pressure: aged/retention and segment selection
             -> Device removal -> independent Host write-back admission
       +-> write-back ACK: queue Disk cleaning in arrival order
  Disk prefetch: credits -> Host allocation -> read -> Host hit

After (GDS)
  explicit production plan -> publish exact component identities
    -> shared dependency index identifies usable recovery points
       +-> optional Host copies: free or safely reclaimable capacity
       +-> capacity pressure: preview placement and rank logical losses
             -> verify survivors and reserve required Host capacity
             -> commit removals and transfers -> ACK publishes completed copies
  Host placement: safe duplicates -> Disk-clean copies -> dirty logical losses
  Cleaning: current Host-policy order -> dependency-ordered writes
  Disk prefetch: credits -> required Host reservation -> read -> re-probe
```

Physical placement may remove a copy while preserving recovery. Only the
joint retention plan may authorize a Device/Host decision that loses a
recovery point. Disk LRU and I/O invalidation remain independent sources of
route loss and must notify the dependency index before its next decision.

| Question | Key design |
| --- | --- |
| When is a checkpoint produced and published? | 1. Separate execution and publication from cache retention. |
| What makes a recovery point usable? | 2. Share one dependency model with prefix matching. |
| Which point should survive pressure? | 3. Compare marginal recovery benefit per released capacity. |
| How are those decisions applied across tiers? | 4. Plan placement jointly and commit under existing ownership rules. |

### 1. Separate checkpoint production from cache retention

The first boundary is between paying to compute a checkpoint and deciding
how long to cache it. An in-forward checkpoint shapes chunks, allocates
extra output capacity, and runs a State producer. Treating all of that as
"publication" makes an eviction-policy change accidentally control model
execution. Instead, the scheduler plans production, the progress path
publishes completed components, and the retention policy handles idle
capacity.

A **recovery point** is an exact content-addressed prefix at T, with a
compatible namespace and numerical State contract. **Production** computes
its State and auxiliary components; **publication** registers those results
under their correct identities. Neither promises a cache lifetime.
**Retention admission** decides whether already-produced content continues
occupying capacity; rejecting it does not reject its request or undo its
computation.

#### Checkpoint production and publication

The [baseline scheduler][checkpoint-schedule] already selects an existing
target, shapes the chunk, and allocates an extra output block for each State
group when the chunk spans it. If this optional capacity does not fit, the
supported route ends the chunk at the same target; a carried prompt that
must finish instead waits. This is allocation admission, before any result
exists for retention to compare.

Keep those supported targets and numerical routes. The
[runtime descriptor][checkpoint-runtime] carries offsets and output blocks;
GDN extracts intermediate scan output, while KDA currently scans the row's
prefix again. Conv/PLE components and the sliding window must also refer
to T. The [progress path][checkpoint-progress] publishes completed outputs
before reclaiming history that they need. Eager and capacity-graph execution
follow the same contract.

Make the boundaries explicit through three values:

| Value | What it establishes |
| --- | --- |
| `CheckpointMaterializationPlan` | The final chunk span, exact target T, query-relative offset, producer capability, output demand, and history/window preservation requirements. |
| `MaterializedCheckpoint` | Producer progress and owning output handles; planned, allocated, or zeroed memory is not proof that a checkpoint was computed. |
| `CheckpointPublication` | Exact identities and views of completed components to register; it neither invokes export computation nor grants retention priority. |

These names describe responsibilities, not a second execution API.
Revalidate the plan after budget, KV-limit, replay, junction, or capacity
adjustments. An interior route requires every consumer's capability and
strict offset constraints; an endpoint uses the existing natural output.
Retain `extend_state_checkpoint_len`, `InForwardStateCheckpoints`, and
`CompletedPages.state_checkpoint_pages` as runtime adapters initially,
including their page/token conversion checks.

The [existing configuration][checkpoint-config] also couples the enable flag
to endpoint filtering and superseded-checkpoint hints. Resolve
`CheckpointProductionConfig`, `CheckpointPublicationRules`, and
`CheckpointRetentionHints` once at startup, instead of interpreting that
flag independently in backend, finish, and eviction code. Preserve current
targets and natural filters, including role, resume-point, and coverage
conditions. Default LRU preserves its aging/demotion behavior; GDS does not
apply production-derived superseded hints on either tier or to stores in
flight. Explicit peer handoff, request-input reclamation, and lifecycle
invalidation retain their own causes and contracts.

Outputs and required windows remain owned until publication or cancellation
retires them. This lets a future KDA/GDN producer optimization change how
the same point is computed without changing what the cache retains. Once
components are published, the next question is whether they form an
executable recovery point.

### 2. Use one dependency model for matching and eviction

Published State is useful only with the attention data needed to resume
at the same T. Conversely, several checkpoints can share most of that
attention data. A request ID, a single block, or a State-complete flag cannot
describe this relationship. The policy therefore works with a **recovery
unit**: a non-owning description of the point and its required components,
using the same rules as prefix matching.

#### Recovery metadata

Place a `RecoveryIndex` at the cache coordinator, reusing `CheckpointTree`
lineage, canonical cache entries, and group geometry. Its identity is the
namespace, prefix content, exact T, and checkpoint semantics, independent
of request slots and physical locations.

| State | Meaning |
| --- | --- |
| State-complete | Every required State group has a checkpoint at T. |
| Memory-ready | Ready Device/Host copies form a legal executable route without a Disk read. |
| Device-usable | That route needs no Host load. |
| Disk-recoverable | Published Disk copies can fill the missing Host-side components of a legal route; a prefetch still needs credits and Host capacity. |
| Guaranteed pending recovery | A committed transfer holds the necessary destinations and dependencies; the promised route is not yet readable. |
| Broken | No legal resident, Disk-recoverable, or guaranteed pending route exists. |

The distinction matters for `Device KV[0,T) + Host State@T`: the point is
recoverable even though neither tier independently contains all its
components. It also prevents pending metadata from being mistaken for a
completed hit.

The route follows the [Host-suffix rule][host-suffix]. At `T=p+t`, with p
the full-page boundary, all attention groups share one split F. Required
history below F comes from Device; required history from F to p comes from
Host. A partial KV tail comes whole from one tier; each State group at T
can independently use Device or Host. SWA windows and sinks can lower F,
requiring additional Host history. A per-key union in
`Device / Host / Device` order does not establish a legal route.

For Disk discovery, use the existing
[`ReserveStoragePrefix -> ProbeHostSuffix` hook][storage-reservation]:
published Disk keys may stand in for missing **Host-side** keys. Retain the
physical source of each range and the exact keys that must be read. After
prefetch the route still has one Device prefix and one Host suffix, including
whole tails and exact State identities. This is not arbitrary three-tier
stitching, and ordinary admission must not expose a Disk key as a ready Host
hit. For example, a Disk-only ancestor plus Host suffix and State is
Disk-recoverable, not broken; the existing [Disk tests][disk-tests] exercise
this case.

Keep physical residency separate from readiness and promises. Shared range
summaries track Device, Host and Disk presence, reserved Host capacity, and
Disk read counts/bytes. A pending cleaning write is not Disk residency.
A prefetch ticket owns its reads and destinations but does not guarantee
the entire route: warm dependencies are temporarily pinned during Host
allocation, and ordinary admission re-probes after I/O.

Use this same resolver for matching and policy previews. The index stores
shared history, windows, sinks, and tails as ranges over the existing prefix
structure, with reverse dependencies for affected-point updates.
Publication, removal, relocation, a changed split, or pin release invalidates
the affected routes and policy inputs. Do not duplicate an entire history
list for every checkpoint or implement a second interpretation of SWA.

Disk publication, LRU eviction, read-failure invalidation, tail removal, and
clear use the same exact-key residency lifecycle. Disk removal can make a Host
copy dirty or break many descendant routes; recompute their fallback, benefit,
exclusive relief, and cleaning eligibility before choosing another victim.
It neither refreshes reuse stamps nor advances a memory-tier watermark.
BoundaryTree's dead test remains a structural check; alive is not by itself
proof of an executable Host-suffix route. Never collect a Host component as
an orphan merely because its only route needs a Disk read.

The index owns metadata, not cache blocks. `CacheBlockRef`, request/FSM
state, and transfer handles retain physical ownership; registrations and
residency updates have RAII lifetimes. The current implementation maintains
shared ranges and route summaries persistently. Candidate coverage and
selection are still built per pressure plan; a resident range index does
not imply a persistent priority queue.

### 3. Use one GDS comparison for retention admission and eviction

Recency alone cannot express the cost of losing a checkpoint. Its value
depends on the nearest surviving fallback; its capacity cost depends on
which other points share its allocations. Use those two marginal quantities
for both established points and newly produced points, so a cheap incoming
point can be rejected instead of displacing more valuable retained content.

#### GDS score

Use one logical recovery identity with a stamp and watermark per pressure
domain. For an eligible unit u and pressured tier t, the memory-only case is:

```text
p(u)       = its token position
f(u)       = nearest earlier surviving usable recovery point on its lineage
             (Device, legal mixed route, or guaranteed pending route), else 0
F_t(u)     = eligible copies exclusive to u in tier t
cancel_H(u)= reserved-in-the-plan Host demand cancelled by dropping u
s_D(u)     = sum(1 / DevicePacking(group)) over F_D(u)
s_H(u)     = sum(1 / HostPacking(group)) over F_H(u) and cancel_H(u)
g(u)       = C(f(u), p(u))
H_t(u)     = stamp_t(u) + g(u) / s_t(u)
Lambda_t   = watermark of committed normal logical choices in domain t
```

The size is capacity relief in the pressured tier, not the sum of bytes in
both tiers. Shared dependencies are excluded. On Host pressure, cancelling
an incoming migration counts because it removes the demand that caused the
pressure. This lets the same comparison choose between dropping the incoming
point and dropping an existing Host point. Exclude zero-relief candidates
before division: they cannot fund this domain's demand. Group/bucket packing
and paired demands remain subject to `ShadowOccupancy::Fits`; the numerical
size is not a fit proof.

Physical duplicate removal and migration do not advance either watermark.
Normal logical selection advances only the domain that made the choice;
on commit, set `Lambda_t = max(Lambda_t, selected ordinary H_t values)`,
using each score at its selection after earlier tentative changes.
Fresh rejection does not advance it. Do not compare two raw scores from
different domains as if their watermarks had identical units and history.
Without Host, the existing Device-only planner keeps its Device fallback rule.

Use the explicit prefill cost model from [FlashStates][vllm-gds]:

```text
C(a, b) = d * (b - a) + e * (b*b - a*a) / 200000
d > 0, e >= 0; both finite
```

Here 200000 is **2 times a reference length of 100000 tokens**, inherited
from FlashStates. The marginal cost at b is `d + e*b/100000`; both coefficients
have units of cost per token, and changing the normalizer requires rescaling
e. Full attention supplies the quadratic term; recurrent layers and SWA
beyond its fixed window contribute mainly to d. Calibrate for the deployment's
layer mix, windows, kernels, precision, hardware and parallelism, validate
over the intended context range, and configure mirrored ranks identically.
This is a prefill benefit proxy, not measured seconds or a full latency model
unless calibration establishes those units and costs.

Mixed-tier points contribute only their relief in the pressured tier. Host-only
points participate in Host pressure; Disk-only data can support recovery but
supplies no memory relief. A redundant Host copy is a placement candidate;
indispensable Host State/history participates in logical retention.

#### Account for Disk reads without making Disk a retention domain

With Disk, distinguish **copy removal** from **recovery loss**. Removing a
clean Host copy adds read-back cost while keeping the point; removing its
last State or indispensable dirty history can force recomputation. Do not
charge full prefill cost for the former or assign zero benefit to the latter
just because some history is on Disk.

For a legal route r, use `R(r) = A * read_blocks + B * aligned_bytes` over
unique missing Host keys, with explicit finite `A >= 0, B > 0` in C's cost
units. Agreed layout/owner geometry and fixed aggregation across ranks keep
the estimate deterministic; local measured latency must not change decisions.

Keep this positive penalty despite asynchronous prefetch. The requesting
prompt waits for all owners before admission; overlapping the read with
other work can hide latency but still consumes SSD bandwidth/IOPS, Host
landing capacity and credits. Pricing every read as free encourages more
Disk reliance until that overlap no longer holds. R prices recovery work and
contention risk, not the entire I/O duration on a request's critical path.
Validate it under both low load and sustained read pressure.

Memory-ready Device/Host routes have R=0: Host is an ordinary L2 recovery
tier, so Host use alone incurs no Disk penalty. Leaving H2D unpriced is an
explicit approximation even with layerwise overlap. Record actual Host-load
duration and bytes to assess it; overlap does not establish zero overhead.

For an actual logical-loss candidate u in the current shadow S:

```text
R_S(v)       = lowest read cost among v's legal routes in S
J_without(u) = min over surviving ancestors v (including cold v=0):
                 R_after(v) + C(p(v), p(u))
g_S(u)       = max(0, J_without(u) - R_S(u))
H_t(u)       = stamp_t(u) + g_S(u) / s_t(u)
```

Evaluate `after` with u's proposed removal closure. A descendant State does
not numerically depend on an ancestor State. If all R values are zero, the
minimum is the nearest surviving ancestor and this reduces to the original
formula. With Disk, a farther memory-ready point can be cheaper than a nearer
Disk fallback: with the 1K cost units below, R6 needing a read of cost 3
reaches R8 at cost `3+2=5`, while memory-ready R4 reaches it at cost 4.
Include the cold route so an expensive read is never credited
with more savings than recomputation. A structurally valid route can have
zero estimated benefit; that is different from being broken. Guaranteed
pending routes keep their protection and are never treated as readable hits.

For example, consider two eligible points in the Host comparison domain,
with equal stamps, equal reclaimable capacity and only cold recomputation
as their fallback. Both cost 100 units to recompute. A can resume from
Device/Host, so R(A)=0 and g(A)=100. B needs Disk reads costing 30 units,
so g(B)=70. B has the lower score. Setting R(B)=0 would erase that preference
for memory recovery. This is a relative valuation, not a ban on Disk-backed
points or an unconditional preference across different gains, sizes and stamps.

Safe clean copies precede dirty logical losses even when a dirty point has a
lower H_H, as RFC 0006 requires. Within the clean class, rank by added read
cost per actual Host relief with deterministic ties; no logical-loss charge
or Lambda_H update applies. Shared consumer counts and per-block read cost
provide a ranking estimate. Validate the selected removal through the resolver
and refresh after removal: alternative memory routes and the changing missing
read set affect the actual cost. The read penalty does not override this
clean-before-dirty rule.

#### Retention admission and priority updates

Initialize `stamp_t` to Lambda_t. Refresh on committed local resume, last
local-use release, and the existing [peer-use events][peer-rfc]. Read-only
probes, failed admission, copy arrival, prefetch dispatch/completion,
dependency repair and score recomputation do not manufacture reuse.

Sharing and fallback changes update `s`, `g`, eligibility, and H without
refreshing the stamp. Recompute affected priorities before choosing the next
minimum: checking only an old heap minimum misses a formerly expensive point
whose score has fallen. An incremental queue must publish those decreases,
reject stale versions, and bound stale-record growth; plan-local rebuilding
must produce the same current comparison. Ordinary equal scores use stable
publication order and content identity rather than pointer order or local
time. A broken unit has no recovery benefit, but live ownership still
protects its components.

A newly ready point is `fresh` until real use or pressure resolves its
retention. Compare it only after its required live ownership ends. Use free
capacity without eviction. Under pressure, reject a fresh point strictly
cheaper than every other eligible ordinary candidate without advancing
Lambda. Ties retain the fresh point; with no competitor, use ordinary
pressure selection. An actual resume ends unused freshness.

Thus GDS chooses which recovery capability to forgo. Optional-copy rejection,
duplicate removal and migration leave that capability intact. Design 4
turns these decisions into physical changes under each demand's contract.

#### A concrete example

**Shared history and suffix removal.**

Consider an idle Device-only cache, with no pins or other branches. Each KV
block covers 1,024 tokens and occupies one capacity unit. All State groups
at one checkpoint together occupy one unit. For this example,
`C(a,b)=(b-a)/1024`, and both stamps and Lambda are initially zero.

```text
tokens: 0          4096          8192
KV:     [B1 B2 B3 B4][B5 B6 B7 B8]
State:             S4            S8
recovery:          R4            R8
```

| Candidate | Fallback | What removing it releases | Gain g | Size s | H |
| --- | --- | --- | --- | --- | --- |
| R4 | 0 | S4 only; R8 still needs B1-B4 | 4 | 1 | 4.0 |
| R8 | R4 | S8 and B5-B8 | 4 | 5 | 0.8 |

Assume the real allocation can use the five units released by R8. GDS chooses
R8, retaining R4 and its shared history, and commits `Lambda=0.8`. The choice
reflects R4's small marginal footprint while the two points share KV. Prefix
length alone therefore does not determine the victim.

R4 now owns the only remaining recovery dependency on B1-B4. Its reclaimable
size grows from 1 to 5, so its score falls from 4.0 to 0.8 **without an
access**. If R4 is subsequently reused, its stamp becomes 0.8 and its score
becomes 1.6. Conversely, removing R4 first would move R8's fallback to zero
and make all eight KV blocks exclusive: `g=8`, `s=9`, `H=8/9`. Both gain and
size depend on what else remains cached.

For retention admission, suppose a new independent 1K recovery point is
produced while Lambda is 0.8. It needs one KV unit and one State unit, giving
`H=0.8+1/2=1.3`. At the next real pressure event, if the reused R4 at 1.6 is
the other eligible candidate, reject the fresh point's Device retention
first. Lambda stays 0.8. These are illustrative capacity and cost units,
not a performance claim for a particular model.

**Evicting and publishing an intermediate checkpoint.**

Take a separate snapshot with the same block sizes and cost model, now with
R4, R6, and R8 on one prefix path. All are idle and already retained. Earlier
evictions have raised Lambda to 2; R4 and R8 have since been reused, so their
stamps are 2, while R6's stamp remains 0.

```text
KV:       [B1 B2 B3 B4][B5 B6][B7 B8]
State:               S4     S6     S8
recovery:            R4     R6     R8
```

| Candidate | Fallback | What removing it releases | g | s | stamp | H |
| --- | --- | --- | --- | --- | --- | --- |
| R4 | 0 | S4 only | 4 | 1 | 2 | 6 |
| R6 | R4 | S6 only; R8 still needs B1-B6 | 2 | 1 | 0 | 2 |
| R8 | R6 | S8 and B7-B8 | 2 | 3 | 2 | 2 + 2/3 |

If one State unit is needed, GDS selects the intermediate R6. Only S6 leaves;
all eight KV blocks remain. R8 is still complete: its State does not depend
on S6's numerical contents. However, R8 now falls back to R4, and B5-B6 join
its exclusive suffix. Its gain grows from 2 to 4 and its reclaimable size
from 3 to 5, so `H(R8)=2+4/5=2.8`. R8's stamp stays 2; the eviction commits
`Lambda=max(2,2)=2`. R4's score is unchanged.

Later, a forward actually produces S6 again at a supported target. Once all
State components at that position are produced and their dependencies are
usable, publication recreates R6 using the canonical cached B1-B6, so the
retained KV footprint does not increase. Temporary forward allocations still
follow ordinary admission. After the producer releases its output holds, the
new R6 is a fresh candidate with initial stamp 2. Publication itself does not
refresh R8's stamp or decide R6's retention under future pressure.

| Effect on R8 | After evicting R6 | After publishing R6 again |
| --- | --- | --- |
| Fallback | R4 | R6 |
| Releasable contribution | S8 and B5-B8 | S8 and B7-B8 |
| Gain g | 4 | 2 |
| Size s | 5 | 3 |
| Stamp | 2 | 2 |
| H | 2.8 | 2 + 2/3 |

The new R6 has `g=2`, `s=1`, and `H=2+2/1=4`: its KV is shared with R8.
Adding this intermediate State makes R8 cheaper to evict again, even though
R8 was neither reused nor rewritten. Both deletion and publication must
therefore update dependent fallbacks and exclusive sizes before the next
GDS choice; inserting or removing only the changed checkpoint's heap entry
would leave the other priorities stale.

The examples show why publication and eviction both change other points'
priorities. The remaining issue is to realize a chosen loss or migration
without letting an independent tier decision invalidate another survivor.

### 4. Commit Device and Host changes in one transaction

Device pressure can often preserve a point on Host; Host pressure can destroy
a point whose other components remain on Device. Apply design 3's retention
decisions in one shared preview, then commit capacity and ownership changes.
The L2 modes differ in the demand they authorize and the lifetime of their
sources, not in how recovery dependencies are defined.

#### Plan placement and logical loss together

A joint plan overlays tentative residency on the index from design 2 and
records:

| Set | Purpose |
| --- | --- |
| F_D, F_H | Exact Device and Host copies to release. |
| W_H | Missing Host components needed by surviving recovery routes. |
| K_H | Ready Host dependencies to hold through required migration ACKs. |
| Q | Recovery identities explicitly selected for logical loss. |

Every originally memory-ready, Disk-recoverable, or guaranteed-pending point
outside Q must retain a legal final route under the plan's Disk snapshot.
Safe copy removal may turn a memory-ready point into a Disk-recoverable one;
it does not reserve capacity for its next read. Future Disk space and pending
cleaning cannot fund this transaction. No live owner or promised destination
may be revoked. Shared KV stays when another survivor needs it.

Device placement tries explicit demotions first, then deeper copies to avoid
unnecessary suffix copies. Host placement uses free space, dead/orphan or safe
Device-covered copies, then safe Disk-clean copies, before logical losses.
Clean means the same namespace/key is already published on Disk. Validate
the affected routes, whole-tail/State closure and holds before removing it.
Real group/bucket and paired-demand `Fits` checks govern both tiers; numerical
relief alone cannot prove fit. The clean-before-dirty classification also
applies to LRU, whose existing planner, dirty ordering and settlement remain
unchanged.

For Device demand, try preservation first: a proposed release creates W_H;
Host funds it with safe capacity, then compares H_H of eligible resident
losses and cancellation of incoming migrations (`cancel_H`). After each
choice, refresh routes, sharing and scores, and recheck both demands. If the
preservation trial fails, roll it back and select ordinary Device losses by
H_D. Host-only demand starts at the Host step. The two domains never compare
raw scores against each other.

For example, Device needs five units. A failed preservation trial leads to
an ordinary Device loss C with H_D=4 and relief 3. Preserving the next two-unit
release A needs two Host units. If Host loss B has H_H=12 and cancelling A
has H_H=14, both relieving two eligible units in the required bucket, choose
B and reserve A. Commit `Lambda_D=max(old,4)` and `Lambda_H=max(old,12)`;
A's migration adds no watermark update. The comparison never pits 4 against
12 or 14. Fresh rejection follows design 3; a failed whole plan commits none
of these changes.

#### Distinguish required capacity from optional copies

| Host demand | Allowed capacity decisions | Ownership and completion |
| --- | --- | --- |
| Write-back preservation | Fund W_H through the joint plan, including Host logical losses or cancellation of a proposed migration. | Reserve all destinations and hold K_H before Device release; eviction D2H remains unpinned and stream-ordered before address reuse. |
| Write-through replication | Use free or safely reclaimable capacity only; skip copies that do not fit. | Keep Device sources pinned until ACK; each completed copy adds residency. Rejection does not reject the Device point or advance a watermark. |
| Disk prefetch / peer Host landing | Fund required request demand through Host placement and eligible idle logical losses. It is neither fresh cache admission nor `cancel_H`. | The existing request/transfer ticket owns sources and landing capacity until completion or retirement. |

Write-through may accept individual components; one ACK need not establish
a complete Host route. It does not enable eviction write-back: later Device
pressure must use existing backups or choose logical loss, even if Host has
free space. Record skipped replicas as `host_copy_rejections`.

Disk requires write-back L2 and uses the existing Prefetching FSM, including
its bounded wait and Ready lifetime. Check operation, held-Host and
per-operation Disk credits **before** Host planning.
Protect exact Disk sources and warm dependencies, then preview and reserve
one complete boundary's Host landing. A failed trial changes nothing before
trying a shorter boundary or taking the existing prefetch fallback. After
all-rank read completion, publish Host components, establish the Ready hold,
and re-probe through ordinary admission. The ticket does not pin a complete
Device prefix, so it cannot promise the whole route throughout the wait.
Arrival alone is never reuse.

#### Reserve, release, then publish on completion

All GDS pressure plans follow the same commit boundary:

```text
preview removals, preservation and logical losses
  -> verify real capacity and every surviving route
  -> hold K_H, apply F_H and reserve all required W_H
  -> register destination-backed pending recovery, then release F_D
  -> issue transfers under their existing source-lifetime contract
  -> publish valid completed components after required ACKs
```

Reservations, pending registrations and preservation holds have RAII owners.
A required `PendingHostStore` consumes its reservations directly; the drain
cannot reject it through a second Host admission. Pending recovery is a
preservation promise, not a readable hit. W_H may include components whose
Device copies stay: completing the common Host suffix can require more than
F_D. K_H protects existing Host support until the migration completes.

Score order and physical release/transfer order are distinct. Commit whole
GDS decisions; legacy block-level Minimize cannot trim one without replanning
and validating survivors. LRU keeps its current Minimize and cascades.
Timeout, detach and cancellation preserve in-flight owners until retirement;
a D2H failure follows the runtime error path and never publishes corrupt data.
Explicit peer handoff retains its no-write-back disposition and access-epoch
checks, including queued and in-flight Host-to-Disk cleaning. Demoted results
must not publish at ACK unless a newer use has superseded the demotion.

#### Clean the Host content likely to leave next

The [baseline cleaner][disk-manager] takes ACK order as priority. Instead,
ACKs notify dirty membership; the current Host policy chooses the next useful
closure. GDS selects positive-benefit dirty points in increasing H_H, while
LRU keeps its legacy dirty order. Skip clean, safely Device-covered, broken,
zero-benefit, no-write-back and already-writing candidates. Cleaning itself
never advances a watermark.

Expand selected points into required dirty dependencies, including shared
history, deduplicate them, and write ancestors before descendants with exact
tail/group rules. **Policy order chooses the point; dependency order chooses
the writes.** Hold published Disk prerequisites during placement and I/O so
the incoming closure cannot evict its own support.

Keep RFC 0006's bounded free-plus-clean target, operation/Host-pin/write-byte
budgets and read priority. Long closures may progress over several plans;
only all-rank ACK publishes each completed row, and partial progress does not
announce a complete route. Host sources stay owned through retirement.
Device write-back never waits for cleaning, and pending writes do not create
reclaimable clean capacity.

Host publication, Disk residency, reuse and demotion update dirty eligibility
and priority. In particular, Disk eviction can make Host copies dirty without
a new write-back ACK or break a descendant route. Process those events before
another policy choice; an ACK FIFO alone is insufficient. Disk keeps its own
LRU/retention policy, with operation-scoped dependency holds and no third GDS
watermark.

#### Preserve capacity on planning failure

With the same residency, ownership and geometry, GDS must fund any demand
that the legacy eligible Device census can fund.

A failed preservation heuristic is not an admission failure. Discard its
Host victims and reservations, then retry Device logical loss without
migration. Ordinary choices use H_D and update Lambda_D only on commit.
If whole units cannot fund demand, exhaust the same physically eligible
Device census as the legacy allocator, respecting ownership, protected input
and real group/bucket `Fits`. Record every recovery route lost by this
exceptional block fallback; a held State does not pin otherwise unowned KV.
These removals have no finite single-unit exclusive score: count them
separately and advance neither watermark. Do not fresh-reject a request,
reissue write-back for a discarded point or revoke a live owner.

Only insufficient capacity after that complete check returns failure to the
existing scheduler recovery path. Failed plans change no residency, committed
retention statistics or watermarks; measure attempted CPU work separately.
Fixed-slot compaction may clear only safe duplicates or orphans. Validate
sufficient and insufficient capacity, shared history, held State, protected
inputs, buckets, partial tails and rollback under both L2 write policies.

## Implementation

### Required refactoring

The following refactors implement the four design responsibilities above.
They preserve the existing LRU planner, Host admission and write-back lifecycle.
Shared production, publication and observation changes must preserve LRU
behavior. The interfaces name responsibilities and may use existing adapters.

| Current coupling or assumption | Required change |
| --- | --- |
| `AdmissionPlanner` segment selection is selected through the eviction write-back flag. | Preserve this LRU behavior. Select GDS dependency planning only for explicit GDS, with its own preservation transaction under the existing scheduler entry. |
| The in-forward flag implies production, natural-point filtering, and superseded demotion. | Resolve explicit production configuration, publication rules, and retention hints once. Preserve current targets/filters and default LRU behavior; route hints by cause and tier, suppressing superseded hints on both GDS tiers while preserving explicit peer handoff. |
| Chunk shaping, extra-block demand, and production metadata are spread across scheduling and runtime. | Return an explicit materialization plan, revalidate it after chunk/capacity changes, and preserve the existing same-target chunk-end fallback. Keep current runtime adapters and producer routes. |
| Planned outputs and publication are described as one operation. | Represent producer progress and owning outputs separately from the publication descriptor; allocated or unwritten outputs must never become ready cache entries. |
| Lineage and per-entry metadata do not directly expose policy recovery dependencies. | Add non-owning recovery metadata and typed events, reuse existing lineage, and share dependency/Host-suffix queries with `PrefixProber`. |
| Cascade removal is partly implicit in `CacheEvictor`, and dead collection is enabled through write-back. | GDS returns explicit removal closures and actual released locations, with dependency cleanup independent of write-back. Keep LRU cascades unchanged. |
| Victim records double as Host write-back segments. | Separate target-tier removal, F_D, W_H, K_H, and write-back/discard disposition. Construct records in dependency order, independently of GDS ordering. |
| Host allocation follows Device removal in the eviction drain. | Under GDS, reserve required Host destinations in a joint overlay before Device removal and admit optional write-through copies without logical loss. Preserve LRU's drain-time Host allocation and settlement. |
| Peer demotion reports counts, while its correctness relies on access epochs and exact held pages. | Emit precise residency/use/demotion events and retain epochs alongside GDS stamps. Preserve the existing wire protocol and plan-only removal contract. |
| GDS range counters and routes ignore Disk, while BoundaryTree includes it. | Track all physical tiers separately from Host promises. Reuse the storage Host-hit hook; retain Disk sources/read requirements, and apply Disk removal events to dependent routes and scores. |
| Host placement has no Disk-clean class; GDS prices only recomputation. | Reclaim safe clean copies before dirty logical losses. Add explicit read-cost inputs and distinguish copy reload cost from loss of the unique State/history. Keep two memory watermarks. |
| Storage reservation queries and allocates in one operation. | Return an exact boundary/landing plan and apply it after credits and source protection. Request demand cannot be fresh-rejected or cancelled as a cache migration. |
| Cleaning consumes write-back ACKs as FIFO priority. | Share the current Host policy's candidate view, maintain dirty membership across Disk changes, and separate score order from dependency-ordered write construction. |
| Peer no-write-back is not enforced through cleaning retirement. | Carry generation/epoch-aware disposition through queued writes and tickets, with ACK-time validation and unchanged physical I/O ownership. |

`AdmissionPlanner` retains its existing LRU selection and minimization.
Explicit GDS uses `PlanGdsEviction` and, with Host, the joint tier planner.
Both integrate with the existing allocator, `CacheEvictor`, transfer manager
and scheduler lifecycle. GDS emits explicit removal, preservation and
reservation decisions; the policy does not acquire pages or start transfers.
LRU retains its current Host retention settings, pending-store transitions
and write-back accounting.

Checkpoint-tree aging is not pure recency. Replacing it with constant
benefit-per-relief and epoch ties would not preserve segment, cascade or
Minimize behavior. A common planner is deferred to a separate proposal:
it first needs complete metadata for every supported publication/import
entry and an explicit migration of write-back, cancellation and accounting
contracts. No legacy victim or lifecycle differences are accepted merely to
enable that experiment in this iteration.

Keep state ownership in the C++ cache/FSM layer and return explicit plan
results from read-only queries. Use RAII for registration, pending transitions,
reservations, and request-use records. Do not add backend-private request-slot
dictionaries or a second GPU allocator. Existing retraction snapshots remain
outside prefix retention, and admission, restore, PD, overlap, speculation,
and compaction continue through their established paths.

Bound dependency-update work and selection metadata growth. If selection
becomes incremental, preserve design 3's treatment of score decreases and
design 4's transaction boundary. The agreed peer clock continues to control
lease deadlines.

### Configuration and scope

```text
--prefix-cache-evict-policy lru          # default: recency/retention ranking
--prefix-cache-evict-policy gds
--prefix-cache-gds-prefill-cost D E      # explicit cost parameters
--prefix-cache-gds-disk-read-cost A B    # required for GDS with Disk
```

Use the [GDS calibration tools](https://github.com/lightseekorg/tokenspeed/blob/main/test/cache_gds_calibration/README.md) to fit
`D/E` from isolated prefill timings and `A/B` from native Disk reads; native
collection defaults to `cases_quick.json`, and configuration is exported only
after held-out validation passes.

Pass the resolved policy explicitly through server configuration, rank setup,
scheduler bindings, and coordinator construction. Initially GDS requires
prefix caching and cacheable State groups; unsupported combinations fail
validation. LRU-specific Device retention tuning must not silently become a
second ranking function inside GDS. Legacy Host retention settings apply to LRU; they do not authorize logical loss under GDS.

The release installs joint GDS retention on Device and Host. With Disk enabled,
published Disk copies contribute routes and read cost, while Disk retains its
own placement policy. GDS with Disk requires explicit read-cost calibration
in compatible units. The first implementation sums unique logical blocks using
their aligned Host block geometry; it does not predict queue delay or SSD parallelism.
Keep the positive read-cost model for GDS with Disk; asynchronous execution
does not make the parameters optional. No automatic overlap discount or
per-rank timing update is introduced. Host loads remain unpriced in this revision.
Preserve the existing Disk requirements (write-back L2, no L3); reject
unsupported combinations explicitly. No third GDS watermark, new Disk
replacement policy, or claim on future Disk capacity is introduced.
Peer ownership and transfer protocols are unchanged; local Host landing
capacity goes through the same dependency-aware retention interface.


## References

The TokenSpeed implementation baseline is `0e6b107c3a5afa96e38038aec831965e4a5111ce`.
The FlashStates implementation baseline is
`3cfd432cd83a54a5e538cad7b090e56c27a870f9`. Links below pin those versions;
the draft RFC supplies structure and protocol context, while current behavior
is grounded in the implementation.

- [FlashStates paper][paper];
- [RFC 0004: Sharing cached prefixes between engines][peer-rfc].
- [RFC 0006: Disk tier][disk-rfc]; [Disk read/cleaning lifecycle][disk-manager];
  [storage reservation and Host-hit hook][storage-reservation]; [Disk behavior tests][disk-tests].
- [Cache concepts and layering][cache-design]; [Host-suffix matching][host-suffix].
- [Checkpoint scheduling and allocation][checkpoint-schedule]; [runtime output contract][checkpoint-runtime]; [progress/publication][checkpoint-progress]; [configuration coupling][checkpoint-config].
- [Device admission planning][admission]; [eviction recording and segment merge][evictor].
- [Host segment placement][host-placement]; [write-back drain][write-back]; [Host settlement][settlement].
- [Runtime copy/load/landing ordering][device-order].
- [Peer leases][peer-exports]; [PIN/RELEASE answers][peer-servicer]; [holder demotion][peer-demotion].
- [Host landing ownership][peer-host]; [reader completion and quarantine][peer-landing].

[paper]: https://www.alphaxiv.org/pdf/2611.flashstate
[disk-rfc]: https://github.com/lightseekorg/tokenspeed/blob/0e6b107c3a5afa96e38038aec831965e4a5111ce/docs/rfcs/0006-disk-tier.md
[disk-manager]: https://github.com/lightseekorg/tokenspeed/blob/0e6b107c3a5afa96e38038aec831965e4a5111ce/tokenspeed-scheduler/csrc/cache/tier/disk_manager.cpp
[storage-reservation]: https://github.com/lightseekorg/tokenspeed/blob/0e6b107c3a5afa96e38038aec831965e4a5111ce/tokenspeed-scheduler/csrc/cache/coordinator/storage_queries.cpp
[disk-tests]: https://github.com/lightseekorg/tokenspeed/blob/0e6b107c3a5afa96e38038aec831965e4a5111ce/tokenspeed-scheduler/tests/cpp/test_disk_tier.cpp
[vllm-gds]: https://github.com/FutureMLS-Lab/vllm/blob/3cfd432cd83a54a5e538cad7b090e56c27a870f9/vllm/v1/core/hybrid_checkpoint/eviction.py#L84
[peer-rfc]: https://github.com/lightseekorg/tokenspeed/blob/0e6b107c3a5afa96e38038aec831965e4a5111ce/docs/rfcs/0004-peer-prefix-sharing.md
[cache-design]: https://github.com/lightseekorg/tokenspeed/blob/0e6b107c3a5afa96e38038aec831965e4a5111ce/docs/design/cache-concepts.md
[host-suffix]: https://github.com/lightseekorg/tokenspeed/blob/0e6b107c3a5afa96e38038aec831965e4a5111ce/tokenspeed-scheduler/csrc/cache/coordinator/prefix_prober.cpp#L650
[admission]: https://github.com/lightseekorg/tokenspeed/blob/0e6b107c3a5afa96e38038aec831965e4a5111ce/tokenspeed-scheduler/csrc/cache/coordinator/admission_planner.h#L212
[evictor]: https://github.com/lightseekorg/tokenspeed/blob/0e6b107c3a5afa96e38038aec831965e4a5111ce/tokenspeed-scheduler/csrc/cache/coordinator/cache_evictor.cpp#L169
[host-placement]: https://github.com/lightseekorg/tokenspeed/blob/0e6b107c3a5afa96e38038aec831965e4a5111ce/tokenspeed-scheduler/csrc/cache/coordinator/host_segments.cpp#L262
[write-back]: https://github.com/lightseekorg/tokenspeed/blob/0e6b107c3a5afa96e38038aec831965e4a5111ce/tokenspeed-scheduler/csrc/cache/tier/transfer_manager.cpp#L225
[settlement]: https://github.com/lightseekorg/tokenspeed/blob/0e6b107c3a5afa96e38038aec831965e4a5111ce/tokenspeed-scheduler/csrc/cache/coordinator/host_cache.cpp#L309
[device-order]: https://github.com/lightseekorg/tokenspeed/blob/0e6b107c3a5afa96e38038aec831965e4a5111ce/python/tokenspeed/runtime/execution/device.py#L568
[peer-exports]: https://github.com/lightseekorg/tokenspeed/blob/0e6b107c3a5afa96e38038aec831965e4a5111ce/tokenspeed-scheduler/csrc/cache/tier/peer_exports.cpp#L120
[peer-servicer]: https://github.com/lightseekorg/tokenspeed/blob/0e6b107c3a5afa96e38038aec831965e4a5111ce/tokenspeed-engine/crates/ts-control/src/peer/servicer.rs#L140
[peer-demotion]: https://github.com/lightseekorg/tokenspeed/blob/0e6b107c3a5afa96e38038aec831965e4a5111ce/tokenspeed-scheduler/csrc/cache/coordinator/cache_coordinator.cpp#L525
[peer-host]: https://github.com/lightseekorg/tokenspeed/blob/0e6b107c3a5afa96e38038aec831965e4a5111ce/tokenspeed-scheduler/csrc/cache/tier/peer_host_landings.cpp#L54
[peer-landing]: https://github.com/lightseekorg/tokenspeed/blob/0e6b107c3a5afa96e38038aec831965e4a5111ce/tokenspeed-engine/crates/ts-control/src/peer/landing.rs#L275
[checkpoint-schedule]: https://github.com/lightseekorg/tokenspeed/blob/0e6b107c3a5afa96e38038aec831965e4a5111ce/tokenspeed-scheduler/csrc/scheduler/operations/forward.cpp#L650
[checkpoint-runtime]: https://github.com/lightseekorg/tokenspeed/blob/0e6b107c3a5afa96e38038aec831965e4a5111ce/python/tokenspeed/runtime/layers/attention/backends/state/checkpoint.py#L70
[checkpoint-progress]: https://github.com/lightseekorg/tokenspeed/blob/0e6b107c3a5afa96e38038aec831965e4a5111ce/tokenspeed-scheduler/csrc/scheduler/operations/forward.cpp#L259
[checkpoint-config]: https://github.com/lightseekorg/tokenspeed/blob/0e6b107c3a5afa96e38038aec831965e4a5111ce/python/tokenspeed/runtime/engine/rank_setup.py#L342
