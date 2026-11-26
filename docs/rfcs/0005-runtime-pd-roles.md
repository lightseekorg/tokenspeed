# RFC 0005: Switching prefill and decode roles at runtime

Status: Draft for discussion. Builds on RFC 0004 (peer prefix sharing) and
lifts its exclusion of the decode role.

## Summary

Let a disaggregated deployment change how many engines prefill and how many
decode while it serves, without restarting an engine and without losing its
caches. The transfer between prefill and decode stays as it is: P pushes the
KV to D. Three changes make a role switch cheap and useful:

1. **The decode role joins peer prefix sharing.** A D engine's Host becomes
   a prefix tier like any other role's (write-back, probed at remote
   admission), and a D engine serves peer reads of its Device and Host
   caches. A session's full history, its prompts and its replies, lives on
   the D that decoded it. With this change the next turn's P reads that
   history instead of recomputing it, whether or not roles ever switch.
2. **The role becomes a runtime setting.** An engine started with the roles
   it may take is sized for all of them. On a switch, every rank applies the
   new role and that role's scheduling preset at the same round. Its caches
   stay, and they are readable by peers in either role. In Dynamo, the worker
   re-registers under the other endpoint, and a planner decides the switches.
3. **A switch away from decode migrates its decodes instead of waiting for
   them.** The engine finishes each running decode, publishing the whole
   sequence, and takes the prefill role. The frontend resumes each request
   with a prefill leg on that same engine: its cache holds the sequence, so
   it computes one row and pushes the KV to a new decode engine over the
   ordinary PD path.

RFC 0004's scope section ("Out of scope: ... the decode role") is amended in
the same change.

## Motivation

Agent workloads swing between prefill-heavy phases (a long tool output, a
fresh context) and decode-heavy ones (long reasoning). A fixed P/D split is
the bottleneck in one phase and idle capacity in the other.

**Changing the split is expensive today.** An engine's role is fixed at
startup (`--disaggregation-mode`). A change of split means restarting
engines, which reloads weights, recaptures graphs and drops both caches:
minutes per switch, then a cold cache.

**The decode role wastes the history it holds.** In a multi-turn session,
the history at the end of a turn sits on two engines:

- the P that prefilled the turn holds its prompt;
- the D that decoded it holds the prompt and the reply.

The next turn's prompt is that history plus new input. Its P reuses its own
prefix cache, or with RFC 0004 a peer P's, but the reply exists only on the D,
which RFC 0004 cannot read from. So P recomputes the previous reply on every
turn, and a long reasoning reply is most of a turn's tokens. Once an engine
can switch roles, the exclusion also cuts off the caches of every engine
currently in the decode role.

### Prior art

- **EvolvingInfra** (MIT HAN lab) changes the P/D split of a pool of
  identical SGLang-derived engines at runtime.
  - **Roles.** The router keeps a prefill weight and a decode weight per
    worker, changed with a compare-and-set API.
  - **Transfer.** D pulls the KV from P with one-sided reads, chunk by
    chunk, while P computes.
  - **Role changes.** A worker moving from D to P stops taking decode legs
    and finishes its running ones up to a drain timeout. The rest are aborted
    and resumed elsewhere. The worker's cache stays.
  - **Planner.** It fits a step-time model per worker from scheduler traces
    and switches only when the predicted gain exceeds two standard errors.
- **Dynamo's planner** scales P and D replica counts, from profiles or from
  forward pass metrics, by starting and stopping engines.

### Alternative considered: pull instead of push

An engine with no role at all is possible on top of RFC 0004:
- every engine runs fused;
- a P leg samples the first token;
- a D leg reads the prompt from P and continues.

It was not chosen. Switching roles does not need it, and it costs what the
push path already provides:
- overlap of the transfer with P's prefill, without which the second token
  waits for the whole read;
- P and D layouts that may differ.

It would also turn every request into two legs joined by a pinned handoff.
This RFC uses continuations only to resume migrated decodes.

## Design

### Terms

- **Role**: an engine's scheduling role, `prefill`, `decode` or `null`
  (fused), as `--disaggregation-mode` sets it today.
- **Switchable roles**: the roles an engine may take at runtime, fixed at
  startup.
- **Switch**: moving an engine from one role to another.
- **Preset**: the runtime scheduling settings prepared at deployment for one
  role.

### Invariants

1. **The KV transfer between P and D is unchanged.** P pushes to D through
   ts-pd, layerwise or per chunk, with the same handshake, routes and
   manifests.
2. **A role is a parameter of the one scheduling path.** A switch changes
   the engine's role-dependent policies at a round boundary. It does not
   restart, reload or recapture anything, and it does not touch the caches.
3. **Every role holds; only roles that compute prompts read.** In every
   role, an engine serves peer reads of its Device and Host prefix caches.
   An engine reads from peers only in the roles that compute prompts (P and
   fused). D receives what it lacks from its P, as today.
4. **A switch never runs two roles at once.** The engine applies the new
   role only when no request of the old role remains. Every rank applies it
   at the same round.
5. **A switch loses no request and no output state.** Requests of the old
   role finish, or are migrated.
   - A migrated decode resumes with the tokens it already streamed carried
     as output, never re-sampled or re-streamed.
   - Its grammar, logprob and stop state continue where they stopped.

### The decode role in peer sharing

**D's Host becomes a prefix tier.** Today the D role refuses write-back,
because its Host holds only retraction images and its remote admission
probes the Device alone. This RFC removes both rules.
- **Write-back on D.** D runs write-back like any role: what an admission or
  a restore evicts is written back to Host in the same plan.
- **The Host joins remote admission's probe.** D's remote admission probes
  Device and Host. The Host hit is loaded into the request's pages by the
  admission plan, and the landing window from P starts past the whole hit,
  so P sends less.
  - **Whole pages only.** The Host continues the Device's history in whole
    pages, so the hit ends on a block boundary of every group and no block is
    both loaded and landed.
  - **State checkpoints.** A State checkpoint still counts only at exactly
    the prompt end and only on the Device.
- **Retraction images** stay as they are: their Host pins hold published
  entries, now beside ordinary ones.
- **Ordering.** A landing writes its blocks from outside the streams, and
  its end releases its request to a later plan's forward. So PD's decode
  receive and RFC 0004's peer reads wait for completion events covering:
  - the forwards in flight;
  - the write-back stores that read victims this plan regranted;
  - the copies and the zeroing;
  - this plan's loads of the request's Host hit, which the later forward's
    layerwise load waits do not cover.

  The default and load streams have separate events. The landing waits for
  both after launching the current forward, without making the default
  stream wait for the whole load and delaying that forward's layerwise work.

  Today the receive waits for the plan's zeroing only, and a plan may zero
  nothing.

**D is a holder.** Lookups, pins, reads and releases work on a D engine as on
any other: the servicer answers from the Device and Host trees, and pins
hold block references.
- **Demotion.** A release that asks for demotion moves D's Device copy to
  its Host, freeing Device pages for decodes. The session's next decode
  probes the Host and loads it back.
- **No new holder logic.** RFC 0004's holder logic is role-free once the
  validation stops refusing D.

**D never reads.** A decode request computes no prompt, so it carries no
peer hint and the engine ignores one. What D lacks after its own Device and
Host hit comes from its P.

**Validation.** `SchedulerConfig::Validate` drops two refusals:
- the D role from the peer check, which still requires prefix caching;
- the D role from the write-back check.

`validate_peer_prefix_options` drops `--disaggregation-mode decode`.

**Router.** The prefill router scores peer credit from the prefill workers'
KV events only. Decode workers' events, both media, join the directory it
reads holders from, so a P candidate gains credit for a history that a D
holds. A decode worker's peer endpoints are published in its runtime config,
as a prefill worker's are. Decode routing gains nothing new; it already
prefers the D whose cache overlaps most, and that D now also probes its
Host.

### Roles at runtime

**Startup.** `--disaggregation-roles` lists the roles an engine may take;
the only accepted list is `prefill,decode`. `--disaggregation-mode` names
the role it starts in, which must be listed. Without the flag, the role is
fixed as today. An engine with switchable roles is built for all of them:
- **Resident slots and decode permits** for the decode role, including its
  extra resident slots.
- **CUDA graphs** that any listed role runs, for decode batches and prefill
  chunks.
- **Both halves of the ts-pd peer** over one transfer engine and one
  registration of the cache arena: the prefill half's outbox and bootstrap
  endpoint, and the decode half's outbox.
- **The data plane's remote stream fences** for both halves.
- **Validation for every role.** The startup configuration is validated as
  each listed role, so no switch can produce a configuration that startup
  would refuse.

**What depends on the role.** The scheduler reads `SchedulerConfig::role`
in 48 places across 8 files. Each reads it as one of three kinds:
- **A per-round policy** (admission and landing windows, decode permits,
  chunk yield, checkpoint policies): it follows the current role.
- **A sizing at startup**: it is sized for the listed roles.
- **A cache policy** (what streams to Host): this RFC makes it role-free by
  giving D the Host tier.

Phase 2 classifies each read and tests it across a switch.

**The switch.** A control call `set_role(role, decodes)` on the engine:

1. **Refuses new requests of the old role.** A late one gets a retryable
   error.
2. **Ends the old role's requests.**
   - **From prefill:** prefills finish and their transfers complete, as P's
     requests always end. This takes about one prefill.
   - **From decode:** `decodes` says what happens to the running decodes,
     and has no default.
     - `migrate` finishes them for migration at the next round (below).
     - `wait` lets them run to their end.
3. **Applies the new role.** The change is queued at the next head, the
   round's relay carries it to every member, and each applies it before
   scheduling, as RFC 0004's changes are applied. The new role's preset
   takes effect with it.
4. **Switches the ts-pd peer** to the new role's half. The prefill half's
   bootstrap endpoint starts serving on a switch to prefill and stops on a
   switch away from it.
5. **Returns** once every rank runs the new role.

Until step 3 the engine takes no new request: about one prefill or one
round, or as long as the longest decode under `wait`. The Device and Host
caches, the peer servicer and its leases, and the KV event stream all run on
untouched.

**Migrating decodes.** A switch from decode to prefill with `migrate` moves
the running decodes to other decode engines, through the PD path that
already moves every request from P to D.

1. **Finish for migration.** At the next round, the engine finishes every
   decode, running or retracted, with the finish reason `migrated`.
   - **Publication.** This is an ordinary finish, not an abort, so it
     publishes the frontier as every finish does: the prompt and the tokens
     generated so far, up to the last token whose KV was computed, the
     partial last page included. A decode step had already published every
     full page.
   - **The stream.** It ends with the generated tokens and the finish reason.
   - **Retracted decodes.** A retracted decode's image drops, and its
     published pages stay.
2. **Switch.** With no decode left, the role changes at the following head.
3. **Resume.** For each `migrated` stream, the frontend sends a prefill leg
   to the same engine, now a prefill engine. The leg is a continuation: the
   original request, with the generated tokens carried as emitted output
   rather than appended to the prompt.
   - **On the old engine.** The prefix cache holds the whole sequence but
     the last generated token, so the leg computes one row and samples the
     next token. It then hands off to a decode engine that the decode router
     picks, with an ordinary push.
4. **Continue.** The new decode engine absorbs the emitted tokens and the
   pushed first token into its output state, and decodes on.

The client sees a pause, not a failure. It lasts one round, the frontend's
resend, a one-row prefill, and the push of the request's KV. A long context
costs mostly the push, which is the same transfer as its original handoff.

**Continuations.** A request may carry tokens already emitted to the client,
with their logprobs.
- **Output state.** The engine absorbs them into its output state (output
  ids, logprobs, reasoning tokens, the grammar matcher, the stop check,
  `max_tokens`) without streaming them, then computes their KV as input.
  The D role absorbs P's first token exactly so today
  (`remote_prefill_done` in ts-control), and a continuation shares that
  code.
- **Usage.** Usage counts emitted tokens as output, not prompt.
- **Why not append to the prompt.** Dynamo's migration appends generated
  tokens to the prompt. The new engine then starts the grammar matcher and
  penalties from the beginning, which breaks a structured output resumed
  mid-object. Planned migrations happen on every switch, so they must be
  exact.

**If the engine fails during the switch,** Dynamo's error-driven migration
resends the unfinished requests through any prefill engine. With decode
holders, that engine reads what the failed engine's peers still hold.

**What a switch keeps.**
- **To prefill:** the history of every session the engine decoded is in its
  caches, so those sessions' next turns hit locally, and other engines read
  them as peers.
- **To decode:** the prompts it prefilled stay readable, and a decode it
  receives probes them first.

### Scheduling presets

Each role has a preset prepared at deployment. A switch applies the new
role's preset; nothing adapts to the load.

**What a preset sets.** The per-round settings that differ between P and D:
- the round token budget and chunk size;
- the running batch bound;
- chunk yield;
- the prefill delay windows;
- short-prefill-first;
- mixed rounds.

A preset chooses values within what startup sized, and is validated as the
startup configuration is.

**How presets are given.** `--disaggregation-role-presets` is a JSON object
with one complete preset per listed role. There is no default preset, and no
setting falls back to the startup value.

### Router and planner (Dynamo)

**A worker's role is the endpoint it serves.** Dynamo separates prefill
workers (the prefill endpoint) from decode workers (the generate endpoint),
and its frontends follow discovery as instances come and go. A switch is
therefore driven by the worker:

1. The worker deregisters its current endpoint, so frontends stop sending it
   legs of the old role.
2. It calls `set_role` on its engine.
3. It registers the new endpoint, with that role's runtime config (a prefill
   worker's bootstrap endpoint, the peer endpoints).

Each router recovers the worker's cache state as it does for any new
instance, from the worker's local indexer. The worker exposes the switch on
its system port, as `/engine/set_role` with the role and, from decode, the
`decodes` choice.

**Resuming migrated decodes.** Dynamo's migration operator gains a planned
case beside its error case. A stream that ends with `migrated` is resumed
without spending a retry:
- the generated tokens travel as emitted output;
- `max_tokens` is reduced by their count;
- the prefill leg is pinned to the worker that migrated it
  (`prefill_worker_id`).

A resumed leg that reaches the worker before its prefill endpoint is
registered waits in the frontend. Error-driven migration keeps its current
behavior for engines without continuations.

**Prefill router.** The prefill router turns disaggregation on when it sees
prefill workers. A switch must never empty either pool, so the planner keeps
at least one worker in each role.

**Planner.** Dynamo's planner gains a role connector. At a fixed pool size,
its decision (how many workers prefill, how many decode) becomes switches
rather than replica changes: seconds instead of minutes, and no cold cache.
Three pieces are needed:
- **Forward pass metrics.** The TokenSpeed worker publishes them, so the
  load-based planner can fit prefill and decode step times. It publishes
  none today.
- **The connector.** It switches one worker per decision, the one whose move
  costs least: for decode to prefill, the least KV to migrate.
- **A switching rule with hysteresis.** Switch only when the predicted gain
  is significant against the fit's error, as EvolvingInfra's two standard
  errors over 60 s slices, and wait a settling time after each switch so the
  next decision measures the new split.

### Scope

- **Peer sharing as in RFC 0004**, plus the decode role: the Host tier on,
  no L3, replicated cache groups only, no pipeline stages, no query context
  parallelism.
- **Switchable roles: `prefill` and `decode` only.** `null` (fused) is not
  switchable: a fused engine takes requests with no bootstrap, which Dynamo
  routes as aggregated serving, a third pool beside the prefill and decode
  pools. `--disaggregation-roles` refuses it.
- **Layouts.** Engines of a deployment need not share a layout for PD,
  since ts-pd routes between unequal rank sets. Peer reads still need the
  holder's and the reader's layouts to match.

Out of scope: moving a decode directly between decode engines (transferring
its retraction image and restoring it elsewhere). Migration goes through a
one-row prefill leg on the migrating engine and the ordinary push.

### Configuration

- **Engine.**
  - `--disaggregation-roles prefill,decode`: the switchable roles, the only
    accepted list; absent keeps the role fixed.
  - `--disaggregation-mode`: the starting role.
  - `--disaggregation-role-presets`: one preset per listed role.
  - `--kvstore-write-policy write_back`: now accepted with the decode role.
  - `--peer-prefix-port`: now accepted with the decode role.
  - A control call `set_role(role, decodes)`, where `decodes` is `migrate`
    or `wait`.
  - Requests gain emitted tokens and their logprobs (ts-wire).
- **Dynamo worker.**
  - It publishes peer endpoints in either role.
  - `/engine/set_role` on the system port.
  - The engine's role flags, passed through.
- **Dynamo frontend.**
  - The prefill router reads peer holders from both pools.
- **Planner.**
  - The role connector, and its switching rule's settings.

### Observability

- **Engine.**
  - The current role, and the round of each switch.
  - Each switch's duration.
  - Decodes migrated, and each migrated request's pause between its last
    token before the switch and its first after.
  - Requests refused during a switch.
- **Peer sharing.**
  - RFC 0004's holder metrics, split by the holder's role.
  - Tokens of a P's prompt read from a D holder.
- **Remote admission on D.**
  - Device hit, Host hit, and tokens received from P.
- **Router.**
  - Workers per role over time.
  - Peer credit given to decode holders.

## Phases

1. **The decode role in peer sharing.**
   - Write-back on D.
   - D's remote admission probes Device and Host.
   - D as holder, with the validation changes.
   - Dynamo: decode workers' events and peer endpoints feed the prefill
     router's holders.

   This pays off in fixed-role deployments: multi-turn sessions stop
   recomputing replies.
2. **Roles at runtime in the engine.**
   - `--disaggregation-roles` and sizing for all listed roles.
   - Each of the 48 role reads classified and tested across a switch.
   - `set_role` with `migrate` and `wait`, application at the head, and
     presets.
   - The finish for migration.
   - Continuations: the ts-wire field, and absorption through the D role's
     first-token code.
   - ts-pd with both halves.
3. **Dynamo switches.**
   - `/engine/set_role`, with endpoint re-registration and index recovery.
   - Both pools kept non-empty.
   - Planned migration: `migrated` streams resumed as continuations on the
     migrating worker, without spending a retry.
4. **Planner.**
   - Forward pass metrics from TokenSpeed workers.
   - The role connector and the switching rule.

## Testing

- **Scheduler (C++).**
  - Write-back on the decode role.
  - Remote admission with a Host hit: the loads fill the hit, the landing
    window starts past it, and the remote stream waits for the plan's
    write-back stores.
  - A D holder's lookup, pin, read and demotion, on full-history and hybrid
    group sets.
  - A switch applied only when no request of the old role remains.
  - Every role-dependent policy before and after a switch in each direction.
  - A finish for migration publishes the frontier, the partial last page
    included, for running and retracted decodes.
  - A continuation's prefill leg on the migrating engine hits all but one
    row.
- **Engine host (Rust).**
  - `set_role` applied at the same head on every member.
  - Requests of the old role refused during the wait.
  - A continuation's output state equals the sampled one: output ids,
    logprobs, reasoning tokens, the grammar matcher, `max_tokens`. Emitted
    tokens are not streamed again.
  - ts-pd halves switched, with the bootstrap endpoint started and stopped.
- **Dynamo.**
  - A switch re-registers the endpoint and recovers the worker's index.
  - The prefill router credits a decode holder.
  - The planner keeps both pools non-empty.
  - A `migrated` stream is resumed as a continuation on the same worker, and
    an error still takes the error path.
- **GPU.**
  - In a fixed split, a second turn's P reads the first reply from D, and
    the output matches recomputation.
  - Switches in both directions under load, with no failed request.
  - Migrated decodes produce the same tokens as unmigrated ones, greedy and
    with a JSON grammar.
  - Outputs identical before and after a switch.
  - After a switch to prefill, a session decoded on that engine hits locally
    on its next turn.
