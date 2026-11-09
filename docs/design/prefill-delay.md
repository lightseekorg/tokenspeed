# Prefill delay: batching prefill work into rounds

A fused engine answers prefill and decode from one scheduling loop. Under
continuous traffic, a prompt that arrives while the engine decodes is admitted
on the very next round. That is the right default for a single replica, and
the wrong one for an attention-DP deployment:

* A round's forward costs the whole group `max` over its ranks: with
  attention DP every rank publishes its batch and joins the same collectives,
  so one rank extending a prompt lengthens the round for every rank.
* A rank whose batch carries prefill rows runs the extend path: eager, on the
  normal DeepEP legs (`use_deepep_low_latency` answers `all_decode_or_idle`
  for the group), and outside the decode CUDA graph. Every such round is a
  round the whole group does not spend decoding.
* A prefill that arrives alone is also a *small* prefill: it pays the fixed
  cost of an extend round (kernel launch, dispatch, graph switch) for one
  chunk instead of filling `max_scheduled_tokens`.

`--prefill-attempt-interval` and `--prefill-any-attempt-interval` add a gate:
the fused scheduler schedules no prefill work on most rounds and admits it in
**windows**, so several prompts (and several ranks' prompts) batch into the
same few rounds. The gate is the control-plane half of the mechanism; the
scheduler half is `Scheduler::PlanOptions::allow_prefill` and
`Scheduler::ProbePrefillGate`.

## 1. The two attempt points

The gate counts rounds since the last round that scheduled prefill work
(`rounds_since_prefill`):

| Point | When | Opens when |
| --- | --- | --- |
| ANY | `rounds != 0 && rounds % any_interval == 0` | any rank wants prefill |
| ALL | `rounds % attempt_interval == 0` | every rank that would decode also wants prefill |

ANY wins when both fall on the same round. Every other round plans decode
only. A round that scheduled prefill resets the counter to 0, and round 0 is
an ALL point, so a burst drains in consecutive prefill rounds; it is the
scattered arrival that waits for a window.

The two intervals are the whole policy:

* **ALL** is the throughput knob. Opening the gate costs the group a round on
  the extend path, so the ALL point opens it only when every rank that would
  otherwise decode has prefill work to put in that round: one rank's prefill
  never interrupts the others' decode alone.
* **ANY** is the latency bound. A rank whose prompt waits must not wait
  forever, so at least every `any_interval` rounds one rank's backlog is
  enough. It must be at least `attempt_interval`; otherwise the ALL points a
  burst relies on could never be reached twice in a row.

The gate applies to the fused role only. A prefill node (P role) is prefill
by construction, and a decode node's prompts are its peer's work; both ignore
`allow_prefill`, and `ServerArgs` refuses the intervals with a
disaggregated launch.

`ServerArgs.prefill_delay` returns `None` only when both intervals are unset.
An incomplete pair raises `ValueError` at the property as well as during
startup validation, including after programmatic configuration changes.

The gate also cannot be combined with `--prefill-decode-interval` (the local
policy in [scheduler §3.3](scheduler.md#33-fused--one-engine-everything-local)).
A local interval could close a rank's prefill after the group opens a window,
breaking alignment; the group gate could also hold a prompt whose local delay
budget is exhausted. `ServerArgs` refuses the combination. Leaving the gate
unset preserves the local interval's scheduling and per-request bound.

## 2. What a rank votes with

`Scheduler::ProbePrefillGate` answers two questions, read once per round:

* `wants_prefill`: a resident chunk is waiting, **or** a new prompt is
  waiting and the Device pool is at most `kPrefillWatermark` (0.9) full. A
  prompt still waiting on its storage read counts as new work: it holds no
  schedulable chunk or Device reservation yet.
* `has_decode_work`: the round would otherwise run a first decode or a
  decoding request.

A rank with no decode work protects nothing, so the ALL point does not treat
it as a blocker: `open = all(want || !decodes)`. A strict `all(want)` would
hold a prompt in an otherwise idle engine for a whole `any_interval`, and
would stall the first ranks of a batch whose peers already finished.

For new prompts the watermark also bounds what ANY can open: a rank above
0.9 with no resident chunk does not vote to open the group's gate. Its new
backlog waits for occupancy to fall to the watermark or below, so its wait
depends on the workload, not only `any_interval`. Resident chunks remain
eligible at high occupancy and can finish to release their reservations.

A pressured rank with decode work and no resident prefill publishes
`want=false, decodes=true`, blocking ALL for the whole group. Peers with
backlog may therefore wait for ANY even when their own pools have room. Under
skewed request routing, `any_interval` controls those peers' cadence too.

## 3. Agreement without a second collective

`probe_prefill_gate` is called in `EventLoop::dp_round`, and its two booleans
ride the per-round batch all-gather that already publishes `num_tokens`,
`batch_size` and `num_extends`. Each rank therefore sees every rank's probe
before its next plan, and the counter is updated from the same gathered rows
(`any(num_extends > 0)` resets it; an all-idle round resets it too, so a
prompt arriving at an idle engine sees an ALL point). Both inputs to the
decision are rank-identical, so every rank opens its gate on the same round —
which is what makes the prefills of different ranks land in the same rounds.
No extra collective is added, and no rank can diverge into a vote the others
do not join.

The cost of reading the previous round's probes is one round of staleness:
a prompt admitted this round is first considered by the next round's vote.
The gate's intervals are tens of rounds, so that is noise; and an open gate
with nothing to schedule plans exactly as a closed one.

The vote is the group's *permission*, not an order to every rank. A rank
plans prefill only when the vote opens **and** its own `wants_prefill` is
true, read fresh from the scheduler at plan time rather than from the
gathered copy. Above the watermark a rank with only new prompts publishes
`want=false`, even if its peers open an ANY window. A resident chunk makes
the local vote true regardless of pressure, but the scheduler still checks
the watermark before admitting new prompts in that plan.

Without attention DP there is no group to agree with: the same gate runs on
the rank's own probe, and only its batching effect remains.

## 4. What the gate does not hold

`allow_prefill=false` skips `scheduleLocalPrefillWork` on the fused role:
resident chunks as well as fresh prompts. A chunk that kept extending every
round would keep the group off its decode path for exactly as long, which is
what batching is meant to avoid; the next open round resumes it. Retraction
restores (`scheduleReadmission`) still run: they are cache operations that
resume a suspended request and take no prefill tokens. Decode work, aborts,
cache-op acknowledgments, storage ticks and weight updates are untouched.
With `enable_mixed_prefill_decode` the decode batch's
`pending_prefill_reserve` drops to 0 when the gate is closed, since no
prefill can ride beside it.

The same permission and watermark checks apply to capacity retraction:
an ineligible prefill neither triggers a retraction nor receives its grant.
Every prefill alternative in the blocker-redirect order is checked again,
so retrying admission cannot bypass the gate. Decode-driven retraction and
readmission restores keep their ordinary behavior; see
[scheduler §2](scheduler.md#2-retraction-when-admission-fails).

The gate only matters when the rank has a free batch slot: `max_batch_size`
is `--max-num-seqs` divided by the attention-DP size, and a full batch
already defers prefill silently (`PlanBuild::Full`). A deployment that wants
the gate to choose *when* to prefill, rather than have a full batch choose
for it, must leave slots free.
