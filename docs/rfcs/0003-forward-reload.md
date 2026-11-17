# RFC 0003: Reloading the forward pass in a running engine

Status: Proposed; development only.

## Summary

Add a development-only control operation, `reload_forward`, that brings
edited model, layer and kernel code into a running engine without a restart:
the engine drains, flushes its prefix caches, re-executes the edited Python
modules on every rank, warms their kernels and recaptures its CUDA graphs,
then resumes. Weights, the KV cache, communicators and tuned tactics stay. The
operation exists only with `--allow-reload-forward`; without the flag the
control loop refuses it on arrival, before it drains anything.

## Motivation

Kernel and model work iterates as: benchmark or profile, edit, measure again.
Each edit costs a server restart today, and on large models the restart, not
the measurement, dominates the loop:

| Model, GB200 | Restart (warm caches) | `reload_forward` |
| --- | --- | --- |
| Beam-1.0 NVFP4, TP4 | 286 s | 72 s |
| Qwen3-8B, TP1 | 183 s | 43 s |
| Qwen3-8B, attention DP 2 | not measured warm | 52 s |

A restart pays process start, weight loading, kernel warmup and graph
capture; a reload pays only the warmup and the capture. Restarting also loses
the profiler session: with a reload, one Nsight Systems session spans every
iteration, one report per profile window.

### Prior art

The open-source serving engines we surveyed in November 2026 cannot reload
model code in a running server; they keep graphs valid rather than rebuild
them:

- Recapture while serving exists only on narrow internal paths: elastic expert
  parallelism scale-up, or decode graphs after a weight update from disk, with
  prefill graphs left stale.
- Sleep and memory release keep or restore graphs at the same addresses.
- Faster restarts come from cached or preloaded weights, not from keeping the
  process.

## Goals and non-goals

Goals: reload edited Python forward code (models, layers, Triton, Gluon and
CuTe DSL kernels) on every rank in lockstep; graphs recaptured as a fresh boot
would capture them; a failed edit leaves the server serving the old code; no
cost or risk when the flag is off.

Non-goals: production use; reloading C++/CUDA extensions; changing parameter
or KV layouts, attention or MoE backends, or the parallel layout.

## Decisions taken

1. **A server flag, refused on arrival.** `--allow-reload-forward` reaches
   the Rust control loop through `ControlConfig`; without it `reload_forward`
   is answered at once and never drains. The data plane builds nothing for it
   either. It cannot be combined with `--enable-memory-saver`, whose tagged
   graph pool cannot be discarded and captured again, nor with prefill/decode
   disaggregation, whose roles would reload and resume independently.
2. **The existing drain, then the same-round gate.** The reload is a new
   `Drain::Reload`: admission stops, arrivals are held, running requests
   finish (`wait`) or are aborted (`abort`), exactly as for a pause. Once
   drained, and once Host write-backs settled, it joins the weight-operation
   queue, so every attention-DP rank runs it in the same round; it asks for a
   cache flush; its result converges like a weight update's. It then resumes
   on its own, unless a pause preceded it. It joins the weight-operation
   queue on arrival, keeping the frontend's order on every rank; the gate
   runs it only once every attention-DP rank holds the same reload (the
   caller's required `id`) and has drained, and a mismatched head waits
   instead of failing the loop. While it is pending, pauses and releases are
   refused; `continue_generation` cancels it, and so does its `timeout_s`,
   which every replica member applies in the same round, the ways out of a
   reload some DP engine never received.
3. **No new transport.** SMG already forwards any RL route name to the
   engine's control calls, so `POST /v1/rl/.../engine/reload_forward` works
   unchanged; `ts serve` adds `--enable-rl` to the gateway when the engine
   enables the reload.
4. **Patch in place, keep identity.** Re-importing a module creates new
   objects, but kernel plans, registry entries, `from` imports and live layers
   hold the old ones. Each re-executed module's old functions, classes,
   `functools` caches and Triton kernels are therefore updated in place from
   their new definitions (`tokenspeed_kernel.hot_reload`), and module globals
   point back at them. Only a definition pairs with a definition of the same
   name; an alias such as `backend = Fast` is rebound, never patched. A
   function whose closure changed shape is redirected to its new definition
   through a trampoline; deleted methods are removed and changed bases set.
   Modules that import or use a reloaded module are re-executed too, providers
   before their users, so copied constants and their caches of compiled
   kernels follow. The kernel and warmup registries and Triton's named-kernel
   table are re-keyed onto the old objects, and so are the elements of the
   tuples, lists, dicts and sets a module binds. Other module data is what
   re-execution makes it: caches start empty, and state set at run time
   returns to its import-time value unless the module keeps it with
   `globals().setdefault`; each reload names what it reset.
5. **One source, every rank.** The first rank reads the changed sources and
   broadcasts the bytes; every rank executes the same bytes and the ranks agree
   on each step's outcome before the next, so they never diverge on a file
   saved mid-reload.
6. **Warm before release, undo on failure.** The new code is warmed with the
   startup dummy prefill while the old graphs still exist. A failure to
   compile, import, patch, warm or capture undoes every in-place change from
   an undo log and, after a failed capture, recaptures the old code. The reply
   carries the error; only a failure to recapture the old code is fatal. This
   is a definition-level transaction, not process isolation: effects a failed
   module's top-level code had on other objects stay, and a failure on only
   some ranks inside a model collective hangs until the NCCL watchdog fires.
7. **Graphs captured as startup captures them.** The compile switch
   (`compile_monitor.is_serving`) reopens for the warmup and capture, so the
   kernels that differ between startup and serving are captured as at startup,
   and the compile monitor forgets the reloaded kernels' history. The heap
   frozen at startup is unfrozen so the old graphs' pools free, then frozen
   again. A reloaded engine is not a fresh one: constructed modules, processed
   weights, objects' own caches and the tuned tactics are kept, so a final
   change is benchmarked on a fresh start.

## Design

```
ts reload-forward / curl ──▶ SMG /v1/rl/.../reload_forward ──▶ ControlCall("reload_forward")
 control loop (Rust, every rank, relayed)
   Controls::handle ─ refuse (flag off, frozen, busy) ─▶ reply
                    └▶ Drain::Reload ─ drained, transfers and write-backs idle
                       └▶ weight-op gate (all DP ranks) ─ flush ─▶ DataPlane::reload_forward
 data plane (Python forward thread, every rank)
   ForwardReloader.reload
     plan (first rank): stale sources + dependents ─ broadcast
     hot_reload.reload_modules ─ agree ─ revert all if any rank failed
     reopen_startup; gc.unfreeze; forget served inputs (dummy inputs back to startup's)
     warm: executor.autotune(keep_loaded_tactics) ─ agree ─ undo on failure
     recapture: release_graphs; set_cache_pool(same pools); capture ─ agree ─ undo + recapture old on failure
     mark_serving; freeze_startup_heap
   converge (Rust) ─▶ reply; finish_reload resumes admission
```

What a reload re-executes: modules under `tokenspeed.runtime.models`,
`.layers`, `.moe`, `.sampling`, `tokenspeed_kernel.ops` and
`tokenspeed_kernel.thirdparty`. Edits elsewhere under the two packages are
reported as needing a restart. A request may name modules explicitly.

## Validation

All on GB200.

- Unit: `tokenspeed-kernel/test/test_hot_reload.py` (functions, classes,
  `super()` redirect, closure changes, caches, registrations, exact revert,
  Triton kernels with and without autotune), compile-monitor and registry
  tests; Rust state-machine tests in `control.rs` and a TP2 control-loop test
  in `tests/serving.rs`.
- GPU integration: `test/runtime/test_forward_reload_gpu.py`.
- Qwen3-8B, TP1 and attention DP 2: an edit to `models/qwen3.py` changes the
  greedy output; reverting it restores the exact original output; a syntax
  error is refused and a forward that raises is undone, both with the server
  still serving; GPU memory is flat over eight reloads.
- Beam-1.0 NVFP4, TP4, one server for all four rounds, 1024/1024 random
  prompts, median TPOT in ms:

  | Round | c=1 | c=16 | c=64 | Reload |
  | --- | --- | --- | --- | --- |
  | Fresh start | 3.51 | 6.63 | 11.50 | |
  | No-op reload | 3.53 | 6.60 | 11.20 | 72 s |
  | Gated-sigmoid Triton kernel slowed (1024-step loop) | 24.55 | 27.90 | 34.11 | 88 s |
  | Kernel restored | 3.53 | 6.61 | 11.25 | 73 s |

  The edit reached all four ranks inside the captured graphs, and the restored
  kernel returns to the fresh-start numbers. The reload re-executed the kernel
  module and its 59 dependents and patched 1994 objects in 0.4 s; the warmup
  took 3 s and the recapture 68 s. GPU memory stayed at 182487 MiB per GPU
  across reloads and no request failed.
- One Nsight Systems session across a reload wrote one report per profile
  window.
- Dogfooding on Beam-1.0 NVFP4, TP4, on its decode at 64 concurrent
  requests: 16 reloads over two servers, 80 to 100 s each, of kernel
  registrations, Triton kernels, a fused model projection and a one-shot log
  line that showed which projection path decode takes. Median TPOT at c=64 went
  from 11.61 to 10.22 ms without a restart. On a fresh server,
  no-op reloads of three module sets and byte-identical edits, then their
  revert, kept the single-request greedy outputs exactly.
- Speculative decoding, one server per row: a no-op reload, an edit
  negating the draft's logits, a syntax error, a draft forward that raises,
  the restore, and two forced reloads of the draft module. Acceptance length
  at concurrency 16:

  | Target + draft | Parallelism | Before | Logits negated | Restored | Reload |
  | --- | --- | --- | --- | --- | --- |
  | Llama-3.1-8B + EAGLE3 chain | TP1 | 2.03 | 1.00 | 2.01 | 51 s |
  | Llama-3.1-8B + EAGLE3 tree (top-k 4, 16 nodes) | TP1 | 3.16 | 1.00 | 3.18 | 50 s |
  | Llama-3.1-8B + EAGLE3 tree (top-k 4, 16 nodes) | TP2 | 2.86 | 1.00 | 2.89 | 80 s |
  | Qwen3.5-35B-A3B FP8 + MTP | TP1 | 3.42 | 1.00 | 3.42 | 114 s |
  | Qwen3.5-35B-A3B FP8 + MTP | attention DP2, EP2 | 3.40 | 1.00 | 3.42 | 19 s |
  | Qwen3.8-27B + DFlash2 | TP1 | 3.05 | 1.00 | 3.08 | 124 s |
  | MiniMax-M3 NVFP4 + DSpark | TP4 | 3.61 | 1.00 | 3.47 | 123 s |

  Restoring the draft restores the exact greedy outputs; failed edits are
  undone with the server serving; GPU memory is flat across reloads. These
  runs found that warmup and capture read state the last served batch left
  in their dummy inputs: an M-RoPE position override, the slots' cached
  lengths, and the last step's draft trees, whose compaction then shifted the
  capture's positions past the draft's rope table. A reload now returns them
  to their startup values first (`ModelExecutor.forget_served_inputs`).

## Review

Five rounds of adversarial review (Codex) and a day of dogfooding found the
issues below; the fixes are in this change:

| Finding | Disposition |
| --- | --- |
| A reload sent to only some DP engines blocks admission forever | `continue_generation` cancels a pending reload |
| A weight update queued during the drain orders differently across DP ranks | The reload is queued on arrival; the gate waits for readiness on every rank |
| (round 2) Refusing gated operations locally left DP operations unmatched | Replaced by queueing on arrival |
| (round 2) A cancel followed by a retry could match a different reload | Reloads carry an id the gate compares; mismatches wait |
| `backend = Fast` → `Slow` in one module patched `Fast` with `Slow` | Pairing by definition name; aliases are rebound |
| An exception while patching escaped without undo | The patch phase is inside the transaction |
| The warmup registry keyed by kernel objects kept old variants | Snapshot, restore and re-key it |
| A planner I/O error left peers waiting in the broadcast | Planning failures are broadcast as refusals |
| Import order put a consumer before its provider; copied constants stayed old | Source-import dependencies, providers first |
| Deleted methods and changed bases were kept silently | Removed and set, with undo; type changes reported |
| A revert left an earlier redirect pointing at the failed code | Redirect targets are in the undo log |
| PD roles would reload independently | Refused at startup |
| (round 2) A `<locals>` wrapper paired any closure | Ownership through the closed-over definition |
| (round 2) New `classmethod`/property overrides kept the temporary class in `super()` | Cells retargeted through descriptors |
| (round 2) Deleted class data, lambda overrides and module-level definitions stayed | Removed, using the bindings the last reload saw |
| (round 2) Deferred imports created false cycles | Only imports module initialization runs count |
| (round 2) Removed kernel and warmup registrations stayed; nested kernels lost identity | Dropped; inner kernels paired |
| (round 3) A reload one DP engine never gets blocks the weight operations behind it | `timeout_s`, expired in the same round on every replica member |
| (round 3) Two reloads without an id share a key | `id` is required |
| (round 3) Class bindings recorded from the whole dict caught attributes installed at run time; first-edit data deletions were missed; dynamically bound module data was deleted | Bindings come from the sources' syntax, snapshotted at startup; module data is reported, never deleted |
| (round 3) A consumer kept importing a definition its provider deleted | The reload fails, as a fresh import would |
| (round 3) `TYPE_CHECKING` imports created false cycles | Skipped with function bodies |
| (round 4) A batch that ended mid-prefill would leave `all_extends_mid_chunk` set, so recaptured EAGLE graphs skip their later draft steps (not reproduced: on GB200 an abort lands only after the prefill's last chunk) | Reset with the other served inputs |
| (round 4) A rollback's recapture started from the inputs a failed capture left | Every capture resets them first |
| (round 5) A tree lane forward that raised left the lane flag set, so the next draft forward took the lane cascade | The flag is scoped to the forward |
| (dogfood) A tuple of classes (`WEIGHT_LOADER_V2_METHODS`) held the new class objects, so live instances failed its `isinstance` checks | Container elements point back at the old objects |
| (dogfood) Registries other modules fill and flags set at startup silently returned to their import-time values | Each reload names the module data it reset |
| Shallow namespace snapshots do not undo top-level side effects; asymmetric collective failures hang; the workspace is frozen during warmup | Documented limits (see below) |

## Alternatives considered

- **Restart only the Python data plane, keeping weights in an owner process
  over CUDA IPC.** A real rollback boundary and no identity patching, but it
  still pays process start, communicator setup, warmup and capture, and needs
  an owner for every device allocation that should survive.
- **An explicit reloadable-forward interface** (modules declare what they
  cache and how to rebuild it). Narrower and reviewable, but every kernel
  module would have to adopt it before it helps; the generic patcher works on
  the code as it is and reports what it could not patch.
- **`fork()` of an initialized process.** Not usable with CUDA.

## Limits and follow-ups

- Recapture dominates a reload. Capturing fewer graphs for a development
  server helps today; faster capture helps startup as well.
- CUDA C++ kernels, `__init__` and layout changes, attention or MoE backend
  changes, and the vision encoder's graphs need a restart. CUDA C++ could
  follow cheaply: every `thirdparty/cuda` kernel is a prebuilt library loaded
  by `tvm_ffi.load_module` in a cached loader, so loading a content-addressed
  copy of a rebuilt library (the loader returns the loaded library for a path
  it has seen) and watching the library files would reload them the same way.
- A kernel compiled once and stored on a long-lived object, rather than in a
  module cache, keeps its old binary.
- `ts reload-forward` spends seconds importing `tokenspeed`, whose
  `__init__` imports torch and flash-attn to quiet their loggers; `curl` does
  not.
- Untested: pipeline parallelism and EPD.
- With more than one rank, Nsight Systems keeps each profile window's capture
  in the node's `/tmp` as a `.qdstrm` until the session ends;
  `QdstrmImporter -i` converts it on the spot.
- Not understood: on the first dogfood server, single-request greedy outputs
  changed across one reload of edited code and stayed changed after the edit
  was reverted; a fresh server did not reproduce it.
- The workspace stays frozen while the old graphs live, so an edit needing
  more workspace fails its warmup; the KV cache keeps its startup size.
- While a reload runs the control thread answers nothing else, a stop
  included.
- A natural extension: kernel overrides (`TOKENSPEED_KERNEL_OVERRIDE_*`) in
  the request, for A/B of registered kernels without an edit.
