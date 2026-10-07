# Draft-tree speculation

This document records the invariants of draft-tree speculative decoding
(`--speculative-eagle-topk > 1`, EAGLE3 and EAGLE-style MTP drafters). A deviation from the rules here is a
bug unless this document is updated in the same change.

## The problem this solves

A chain drafter proposes one token per depth; a draft tree keeps the best
`K` candidates per step and verifies up to `N` nodes (`N <= 64`) in one target
forward, so a request accepts whichever root path the target agrees with. The
rest of the runtime — commit (`vc += accept_len`), the output processor, draft
step 0, KV and hidden-state bookkeeping — is written for a chain. Trees must
not grow a second copy of any of it.

## Invariants

### The tree is a parameter of the chain path

The sampling backend verifies the step's trees (`TreeVerifyBatch`: parents and
depths) with the `verify_tree` kernel, packing `predict` along the accepted
path. The TP-agreed path has one owner, the backend's packed verify output
(`SamplingBackend.accepted_path`); every consumer receives it explicitly:

* the attention backend moves the path's target KV (every layer) to the
  window's leading slots (`compact_verify_window`; the cache-group router owns
  a K/V address table per cache group and the `compact_window_rows` kernel copies words, so
  it is dtype-agnostic);
* `TreeSpec.compact_rows` moves the target hidden rows to the front of the
  window and the positions back from `vc + depth` to `vc + i`;
* stateful backends receive it through
  `commit_speculative_state_after_verify(accepted_path=...)` (`None` for a chain).

Everything downstream sees a chain.

A chain is the tree `parent[i] = i - 1`. With `topk == 1` no tree state exists
and the chain path runs unchanged. With `topk > 1` the output processor and the
draft step-0 forward are the chain's; the named tree-only pieces are candidate
selection after step 0 (`Eagle._seed_tree_lanes`) and, in a stateful backend's
commit, the steps that read `accepted_path` (the GDN source row and replay
payload packing).

### Position follows depth, slot follows node index

Node `i` of a request's verify window has RoPE position `vc + depth[i]` and KV
slot `write_locations[b * N + i]`. Only positions change for a tree:
`TreeSpec.depth_positions` shifts the window's `vc + i` to `vc + depth` before
the forward, and `TreeSpec.compact_rows` shifts them back after the forward
(row `i` then holds the path node at depth `i`).

The two shifts must cancel for any `depth_buf`, including graph warmup, which
replays the forward without the step prep: `depth_buf` and `mask_buf` start as
the chain the initial parents describe (`test_fresh_spec_is_the_chain`).

### One tree attention: trtllm-gen for the prefix, a small kernel for the tree

Every row sees the committed prefix, and the 64-bit ancestor mask applies only
to the last `W` keys (`W = N` for verify). The prefix is where the time goes at
long context, so it runs on trtllm-gen's own decode kernel: a causal
`q_len = R` decode over the `P` committed keys with its base-2 log-sum-exp,
which covers row `r`'s keys `[0, P - R + 1 + r)` (no head folding, so no limit
on `R x group`). `tree_window_attention` attends the rest of each row (the
prefix tail it missed and its masked window) and merges both in one Triton
kernel. Verify runs it with `R = W = N`; draft lanes with `R = K` over the
`(S - 1) * K`-key lane window (below). There is no size-dependent second path.

An FP8 KV cache (E4M3, unscaled like every FP8 KV cache here) changes no
structure: trtllm-gen takes the query cast to FP8 as it does for any decode,
and `tree_window_attention` takes the unquantized query and widens the
window's FP8 K/V to its dtype after loading them.

### Recurrent state follows the parent

Linear-attention (GDN and Mamba2) layers keep one conv window and one recurrent
state per verify node in the backend's verify scratch. Node `t` starts from the
state after its parent, not after node `t - 1`: `gdn_decode_mtp` and
`mamba2_verify_scan` take `parent_indices` and reload the parent's state at
branch points (a chain never reloads); `causal_conv1d_update` with `parent_indices` rebuilds each node's
window from its ancestors' inputs and the initial window. The commit copies the scratch row of the
last accepted node, `1 + path[accept_len - 1]`, which for a chain is the
familiar `accept_len`. The fused KDA verify kernel follows a chain and refuses
trees.

Draft trees use ReplaySSM like chains (on by default; staging a recurrent
state per node and per layer grows with the tree, Qwen3.8: 3 MiB x 48 layers
per node). Under ReplaySSM the verify
never writes the state pool. The state of every branch point (a node with a
child other than the next node) goes to one workspace shared by all layers
(`gdn_decode_mtp(intermediate_states_buffer=...)`, one layer's worth per node),
and a branch reloads its parent from there. Mamba2 has no such workspace: its
elementwise update lets a branch replay the parent's ancestors over the read
state, cheaper than storing and reloading a 4 MiB state. The commit packs the
accepted path's replay payload rows to the window's front (`compact_window_rows`)
and replays them like a chain (`gdn_replay_commit`, or `mamba2_replay_commit` for
Mamba2). With `--disable-replay-ssm`,
or where the replay kernel is unsupported, the per-node staged path above remains.

KV compaction covers the attention layers only (`history_group_by_layer`, read
by the router from its bound pool) and moves each physical region once per
cache group, at that group's own verify window.

### Draft lanes write the draft cache like any draft step

Drafting steps `1 .. S-1` run `K` lane rows per request. Step `s` lane `r`
writes draft-cache slot `frontier + (s - 1) * K + r` through the attention
prologue (`publish_draft_step_locations` with `K` tokens per request), like the
chain's draft steps (`(S - 1) * K <= N`, checked in `server_args`). Their K/V
are only valid while the round's tree is drafted; the target's verify
overwrites the window.

Known limitation, shared with the chain drafter: the decode reservation
guarantees pages only through the verify window (`vc + N`), and draft writes
reach past it (lanes up to `frontier + (S - 1) * K - 1`, chain steps up to
`frontier + S - 2`). A position past the allocated extent resolves to the
dummy slot, and attention reads it through page 0. This costs draft acceptance
only; target KV is rewritten by the next verify. The fix is a draft headroom
in the scheduler's decode reservation.

Lane attention is the same cascade with `R = K` over the draft paged cache:
every lane row sees the accepted frontier and, inside the window, only its
ancestors' slots (`lane_mask`). It reads `TreeDraftInputs` (the frontier and
window lengths `frontier + (S - 1) * K`, the lanes' ancestor masks, and
`active`, a Python flag set around each lane forward), which the drafter owns
and writes within the round: lengths once per round, masks by
`draft_tree_expand` for the next step. This is the one exception to the
refresh-only draft metadata contract of `unified_path.md`; it is graph-safe
because the buffers are bound once at fixed addresses and written by in-graph
ops before the lane forward reads them.

### The drafter scores, the tree selects

`Eagle._score_candidates` is the one place a drafter decides how candidates are
scored (today `logprob_topk` over the full draft vocabulary); `DraftTree` only
records `(scores, tokens)` and selects. A child's score is at most its parent's
(`draft_tree_expand` clamps child log-probabilities at 0). A new drafter changes
the scorer, not the tree machinery.

### Next round's tree rides with next round's tokens

The drafter's parents for a pool slot live in `RuntimeStates.future_parent_map`
next to its candidate tokens in `future_input_map`; rows reset to dummy tokens
(bootstrap, recovery) reset to the chain.

### Tree construction is deterministic

`DraftTree` keeps the best `N - 1` of the `K + (S - 1) K^2` scored candidates;
a child's cumulative log-probability never exceeds its parent's and ties go to
the lower candidate id (the parent's), so the kept set is a tree. Nodes are numbered depth first with each node's best child
first, so the most likely path is `0, 1, 2, ...`. NaN scores (padded requests)
rank last, so lanes and nodes are always fully written.

### Sampled verify keys noise by position

The triton sampling backend draws a verify row's target token by Gumbel-max
keyed by `(seed, position)`. For a tree the key is the node's position
`vc + depth`, never its row: each node row samples with its request's
parameters and its offset advanced by its depth, so the token accepted at every
position is the one plain decoding samples there. Acceptance is the greedy tree
walk over those draws (a child is accepted when it equals its parent's draw).
Backends that cannot do this keep `supports_tree_verify = False` and the
executor refuses tree drafting with them at startup.

## Scope

EAGLE3 and EAGLE-style MTP drafters (the `Eagle` drafter; the multi-depth `Mtp`
drafter refuses trees at startup); `greedy` and `triton` sampling backends; the `trtllm`
attention backend with bf16 or FP8 E4M3 KV in full-history KV cache groups, alone or inside the
hybrid linear-attention backend (GDN or Mamba2, ReplaySSM or staged); no structured output,
no mixed batches, no pipeline parallelism, no prefill/decode disaggregation, no
attention data parallelism, no sliding window or attention sinks in the target
or draft layers, no target or draft that reads request token history or
n-gram (Engram) input history, and separate target and draft attention backends.

Each attention backend node declares its own part through `tree_support()`
(verify and lanes, each supported or refused with a reason);
`resolve_tree_support` composes it over the target and draft backend trees
through `child_backends()` once at startup, before any bind, and reports every
blocker together. Composites never forward the question, so a new composite
cannot silently skip a child. Sliding window and attention sinks are the named
exception: they are per layer and per call, so the trtllm forward refuses them.

## Not scheduler or cache state

The only per-request fact that crosses steps is the next round's `parent[N]`,
an attribute of the candidate block `future_input_map` already carries on the
executor side. Lane K/V are round-local: they live in the request's draft
window until the next verify overwrites it (never prefix-matched,
transferred or freed with blocks), and the per-node GDN states live in
verify-time workspaces (the verify scratch, or the ReplaySSM node-state
workspace), so nothing here is a cache group for the C++ scheduler to own.

## Intended direction

* Tree refresh inside `refresh_decode_metadata`, and leaves branching on a
  tree mask in the verify metadata instead of the query shape.
* The parallel tree conv kernel as the only verify conv kernel if it benches at
  or above the serial chain kernel.

## Tests

* `tokenspeed-kernel/test/ops/test_tree_speculative.py` — the tree window
  kernel against fp32 references from a reference prefix partial (masks up to
  64 nodes, short prefixes, non-finite unseen slots), tree verify and KV-row
  compaction (`test_compact_window_rows_moves_every_buffer`, bf16 and fp8).
* `test/runtime/test_tree_attention_cascade.py` — verify and lanes through the
  trtllm leaf (trtllm-gen prefix plus window kernel) against fp32.
* `test/runtime/test_draft_tree.py` — tree construction against a per-request
  EAGLE-2 reference, depth-first order, strided roots, NaN scores.
* `test/runtime/test_tree_spec.py` — hidden-row and position compaction; the
  fresh-spec chain; per-group router compaction over aliased layer buffers.
* `test/runtime/test_tree_support_resolution.py` — backend capability
  resolution: supported trees, linear-attention layers, each named blocker.
* `test/runtime/test_cli_config_compat.py` — tree options and every startup refusal.
* `test/runtime/sampling/test_tree_sampling.py` — a chain-shaped tree verifies
  like the chain; every node samples the chain's draw at its depth.
* `tokenspeed-kernel/test/ops/test_attention_gdn.py`,
  `test/runtime/test_causal_conv1d_tree.py`, `test/runtime/test_qwen35_gdn_replay.py`
  — per-node GDN states and conv windows against a per-path reference; the
  ReplaySSM tree commit.
