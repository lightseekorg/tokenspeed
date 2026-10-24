# RFC 0002: Rust control plane

Status: Accepted; implemented through Phase 4 (`tokenspeed-engine/`), with
the deviations listed under Phase 4.

## Summary

Rewrite the engine control plane — everything above the data plane except the
C++ scheduler — in Rust: the scheduler event loop and its hooks, request
state and output processing, the data-parallel controller, and the frontend
client that `AsyncLLM` is today. Each rank becomes a Rust host process that
embeds CPython only for the data plane (the forward thread, `DeviceHandle`,
models, kernels). SMG and Dynamo talk to the rank processes directly over the
existing msgpack wire protocol, with no Python on the request path. The
Python `Engine` API stays, as a PyO3 wrapper over the Rust client.

The Rust control plane is developed beside the Python one until it reaches
parity on the CI matrix, then replaces it in one cut; the Python event loop,
`AsyncLLM` and the data-parallel controller are deleted in the same change.
There is no long-lived second path.

## Motivation

Every production frontend reaches the engine through Python today:

| Frontend | Path to the scheduler process |
| --- | --- |
| `ts serve` (default) | SMG gateway → gRPC → `smg_grpc_servicer.tokenspeed` (Python) → `AsyncLLM` → ZMQ |
| Dynamo | Dynamo frontend (Rust) → request plane → `dynamo.tokenspeed` worker (Python) → `AsyncLLM.generate_request` → ZMQ |
| `ts serve --headless` | SMG → msgpack ZMQ → scheduler (no `AsyncLLM`) |

Even the headless path ends in a Python event loop. The frontends already do
tokenization, detokenization and routing in Rust; the Python hops in between
add serialization, asyncio scheduling and per-chunk object construction, and
inside each rank process the control thread competes with `ForwardThread`
for the GIL (the online expert rebalance already moves its placement solver
into a child process for exactly this reason, see `docs/design/event-loop.md`).

The goals:

1. SMG, Dynamo and any other Rust frontend call TokenSpeed directly, through
   one wire protocol and one client library, with no Python process or
   thread between the HTTP layer and the C++ scheduler.
2. The rank process's GIL belongs to the data plane alone.
3. One control-plane implementation serves every frontend, including the
   Python `Engine` API.

Non-goals:

* Rewriting the C++ scheduler. It stays C++ and keeps its release process.
* Changing the data plane: models, attention backends, kernels, CUDA graphs,
  `ForwardThread` semantics, KV pools and the transfer engines' data faces.
* Changing scheduling semantics. Every rank keeps its own mirrored scheduler,
  aligned by rank-identical control-plane collectives, exactly as today.
* A performance claim. The motivation above is the frontend contract and
  GIL ownership, not an unmeasured speed-up.

## Decisions taken

1. **Topology: a Rust host per rank, frontends on the wire protocol.** Not a
   control-plane crate linked into the frontend process with plans shipped to
   workers: that would replace the mirrored-scheduler design with a
   centralized one, which is a different project.
2. **Migration: develop beside, cut over once at parity.** The Rust control
   plane sits behind a temporary launch switch during development, and the
   cut-over change deletes the Python control plane and the switch together.
3. **The Python `Engine` API stays**, as a PyO3 binding over the Rust client.
4. **Grammar engine: xgrammar's C++ core** (see Grammar).
5. **Mooncake: the Rust control plane owns the transfer engine through
   Mooncake's C API**; the L3 store client stays on the data plane (see
   Transfer and storage libraries).
6. **Metrics plugins keep their Python contract and run in a per-node relay
   process**, not in the rank process (see Metrics).
7. **Copy-event wait: native event handles waited on through
   `tokenspeed-kernel` C entries** (see the control/data plane boundary).
   Implemented differently: the control thread waits on the data plane's
   torch event (see the deviations under Phase 4).
8. **The NPU path is in scope** (see NPU).

## Target architecture

```
 SMG / Dynamo / Python Engine (PyO3)
        │  ts-client (Rust): request lifecycle, streaming, abort, DP routing,
        │  control ops — what AsyncLLM + DataParallelController do today
        ▼
 msgpack over ZMQ (today's schema, owned by ts-wire)
        ▼
 rank process (Rust host binary), one per rank
 ├─ control thread (Rust): event loop, hooks, request state, output
 │   processing, grammar matchers, PD/EPD/L3 control faces, metrics
 ├─ C++ scheduler: libtokenspeed_scheduler, linked directly
 ├─ CPU collectives (Rust): rank-identical agreements across TP/CP/PP/DP/EP
 └─ embedded CPython: the data plane
     └─ ForwardThread: pulls submissions from a Rust queue, runs
        DeviceHandle's named operations, posts results back
```

### Crates

A new top-level `tokenspeed-engine/` Cargo workspace, beside
`tokenspeed-scheduler/`:

| Crate | Owns | Replaces |
| --- | --- | --- |
| `ts-wire` | The IPC schema and codec: tagged, array-encoded msgpack, out-of-band tensor frames | `engine/io_struct.py` (`BaseReq` / `BaseBatchReq` structs, `MsgpackEncoder`) |
| `ts-client` | Request submission, output streams, abort, parallel sampling fan-out, control requests (profile, pause, weight ops, flush), DP rank selection | `AsyncLLM` minus tokenization, `DataParallelController` |
| `ts-control` | The event loop and everything it coordinates | `engine/event_loop.py` and its hooks, `request_handler.py`, `generation_output_processor.py`, `scheduler_utils.py`, `pause.py`, the hooks modules |
| `ts-collective` | CPU collectives over a TCP mesh among the ranks | the gloo groups the control plane uses through `torch.distributed` |
| `ts-host` | The rank binary: argument parsing, CPython embedding, startup and shutdown sequencing | `run_event_loop`, `_launch_subprocesses` |
| `ts-py` | PyO3 bindings: `Engine`, wire types for tests | the in-process `Engine` / `AsyncLLM` |

`ts-wire` becomes the single source of the schema. During parallel
development a conformance test round-trips every message type between the
Python msgspec structs and `ts-wire`; at cut-over the Python encode side is
deleted with `AsyncLLM`.

### The control/data plane boundary

`DeviceHandle` already lists, by name, everything the loop may ask of the
device side (27 operations: `execute`, the cache polls, the L3 operations,
the `run_*` methods, memory release/wake, NCCL suspend/resume, the expert
load operations, weight updates, multimodal work). That list becomes the
boundary, as an enum of submissions:

* **Pull, not call.** The Rust control thread never calls into Python on the
  per-round path. `ForwardThread` becomes the consumer of a Rust-owned FIFO:
  it blocks in Rust with the GIL released, receives a submission, runs the
  matching `DeviceHandle` operation, and posts the result to a Rust-owned
  reply slot. A control thread that called into Python would wait for the GIL
  behind the forward thread's launches (up to the interpreter's switch
  interval, 5 ms by default) — the stall this design exists to remove.
* **FIFO order is preserved.** Today `DeviceHandle` methods enqueue closures
  on `ForwardThread`; the Rust FIFO carries the same operations in the same
  order, so the same-round ordering contract in "Anatomy of a round" (stores,
  device copies, zeroing, load-backs, restores, remote streams, batch) holds
  unchanged.
* **Named operations only.** The enum is closed. A new device interaction is
  a new variant, reviewed against the same rules as a new `DeviceHandle`
  method today. There is no "run this callable" variant.
* **The plan crosses as the scheduler's own C++ objects.** The data plane
  reads `ExecutionPlan` / `ForwardBatch` through the nanobind types it uses
  today. The scheduler core is built as one shared library that both the
  nanobind module and the Rust host link, so a plan the Rust thread received
  from the scheduler is handed to Python by reference, without a copy or a
  second schema. The control plane's own access goes through a `cxx` facade
  over the same library.
* **Commit waits on the copy event without CUDA in the control plane.**
  (Implemented differently, with the control thread's completion queries:
  see the deviations under Phase 4.) The
  result's D2H copy lands in pinned buffers the data plane allocated at
  startup and registered with the host. The data plane records the copy
  event as today and posts its native handle (CUDA or HIP event, Ascend
  `aclrtEvent`) with the result. The control thread waits through two C
  entries exported by `tokenspeed-kernel`'s platform layer,
  `event_synchronize(handle)` and `event_query(handle)`, implemented once
  per vendor and resolving the vendor runtime at load time. `tokenspeed`
  links no vendor library, and the control plane issues no device work.
  The data plane keeps the Python event object referenced until the control
  plane releases the result, so a handle is never destroyed while it is
  waited on. This is the same wait as today's `copy_event.synchronize()`.

  Two alternatives were considered. Completion signalled by the data plane
  needs a thread that waits on each event; written in Python it is a
  GIL-holding thread on the commit path again, and written natively it is
  this same entry. A stream-ordered write of a sequence number into pinned
  host memory (`cuStreamWriteValue32` and its HIP counterpart), polled by
  the control thread, needs no runtime call at all, but it requires stream
  memory operations on every vendor and spends a core spinning. Revisit it
  only if profiles show the event wait itself matters.

`PlannedForward` today carries Python objects gathered from request state.
Each field gets a data-plane representation:

| Field | Representation across the boundary |
| --- | --- |
| `forward_op` | the scheduler's C++ value copy, by reference (above) |
| `sampling_params_list`, `stop_criteria`, `dp_metadata`, `input_logprob_plan`, `sampled_top_k`, `ngram_inputs`, `request_history_seeds` | plain values in a `#[repr(C)]` / numpy-viewable layout, frozen at submission |
| `grammar_inputs` | handles to Rust-owned matchers (see Grammar) |
| `multimodal_context` | handles to data-plane-owned feature records (see Multimodal) |

### Control-plane collectives

The loop's agreements — DP sync, the replica MIN/MAX reductions of cache and
L3 state, the weight-op gate, EPD admission broadcasts, EPLB agreements —
move to `ts-collective`: all-reduce (min/max/sum) over small integer arrays,
broadcast and all-gather of byte payloads, over per-group TCP connections
established at startup from the same rendezvous the ranks use today.
Payloads are small and latency matters more than bandwidth. Every
rank-identical-ordering rule in `docs/design/event-loop.md` carries over
unchanged; the groups (attention TP, CP, PP, attention DP, EP) are the same
groups.

### Request state and output processing

`RequestState`, finish checks (stop token ids, length, grammar termination),
logprob assembly, abort handling and the output batch sent to the frontend
move to Rust. Stop strings and detokenization stay where they are today: in
the frontend (SMG, Dynamo, or `ts-py` for the Python `Engine`).

### Grammar

Today the matchers are Python objects owned by the control plane and shared
with the forward thread (a registered exception to the capture contract).
The matchers move to Rust ownership on xgrammar's C++ core, the same engine
`grammar/xgrammar_backend.py` uses today, so compiled grammars and token
masks do not change. The xgrammar wheel ships its C++ headers and a static
library (`include/xgrammar/`, `lib/libxgrammar.a`); `ts-control` links them
through a `cxx` facade of its own rather than a community Rust binding.

* **Tokenizer info.** `TokenizerInfo.from_huggingface` needs the HF
  tokenizer, which only the Python side loads. The data plane computes the
  encoded vocabulary, vocabulary type and stop tokens at startup and passes
  them to the host as plain values; the host builds the C++
  `TokenizerInfo` from them.
* **Compilation** runs on a Rust worker pool with the compiled-grammar
  cache, off the round, as the Python grammar manager does today.
* **Reasoning structural tags** (`grammar/reasoning_structural_tag.py`) are
  ported with the grammar manager.
* **Capture.** The capturable grammar path's host function becomes a C
  callback into the Rust-owned matcher instead of a Python callable, which
  also removes the GIL from inside the captured graph. The bitmask apply
  kernel stays on the data plane.
* The eager-grammar overlap break stays in the single
  `_dispatch_depends_on_pending_commit` registry.

### Multimodal

Pixel preprocessing (HF processors) is Python and CPU-heavy; it must not run
in the rank process's interpreter. It runs in frontend-side preprocessing
workers or in the EPD encoder service, and payloads reach the rank process
through the existing shared-memory and RDMA transports. The control plane
holds feature handles only; the data plane owns the feature records and
their release, ordered behind the forward that captured them, as today.

### Subsystem placement

| Subsystem | Control plane (Rust) | Stays on the data plane (Python) |
| --- | --- | --- |
| Scheduling round, in-flight queue, advance call sites | all | — |
| L2 / snapshot cache ops | op adaptation, ACK polling, replica intersection | the `HostCacheExecutor` and its streams |
| L3 | the per-round storage agreement (clock, permissions, completions, pressure), cancellations, namespace / weight-version gates | the Host-only L3 executor and its storage backend clients (including the Mooncake store) |
| PD | bootstrap server, request registration, the Mooncake transfer engine: memory registration, transfer submission, status polling | KV packing into staging buffers (GPU copies), landing |
| EPD | the prefill receiver (encode-rank handshakes, receive slots, completion agreement); the encode rank's loop (request relay, batching, send-ring slots) and sender | receive-pool and send-ring memory, publish and reassembly, the vision tower and its embedding cache |
| Pause / memory saver | state machine, drain gates | release/wake regions |
| RL weight ops | the two FIFOs and the same-round gate | the weight load and NCCL group work |
| EPLB | controller, agreements, placement solver | load snapshot, apply placement |
| Metrics | metric definitions, Prometheus export, per-round samples, scheduler KV event publishing, events to the relay (see Metrics) | device-side counters read through named operations |
| Profiling | request handling, step counting | the profiler itself |
| GC hooks | — (no Python heap on the control plane) | full-collection cadence for the data plane's interpreter, rank-identical |
| Plugins | — | models, drafters, attention backends, cache recipes and pools (all data plane today) |

### Transfer and storage libraries

Mooncake publishes a C API and Rust crates for both of the components
TokenSpeed uses: the transfer engine (`transfer_engine_c.h`, the
`mooncake-transfer-engine/rust` crate) and the store (`store_c.h`,
`mooncake-store-rs`).

* **Transfer engine (PD).** A transfer engine instance is one set of RDMA
  resources and memory registrations, so a process must have one owner of
  it: two instances, one per language, would register the same memory
  twice. The Rust host owns it through the C API: initialization,
  registration of the addresses the data plane allocated (KV pools, pack
  scratch), the session id, transfer submission and status polling. The data
  plane keeps the GPU side of PD (the cache contract, the device fences a
  transfer waits on, landing) and hands the host addresses. The Python
  transfer worker threads go away with it. The page-batch write that
  TokenSpeed's Mooncake build adds to the Python binding
  (`batch_transfer_sync_write_pages`) has no C entry yet; it gets one in the
  same build, rather than being re-expressed as per-descriptor submissions.
* **Store (L3).** The store client stays where it is, behind
  `KvStoreStorage` on the data plane's Host-only L3 executor, with the
  in-memory test backend beside it. It keeps its own transfer engine inside the store
  client, as today, so it does not share an instance with PD. The control
  plane reaches it only through the named L3 operations. The C API makes a
  later move possible but nothing in this RFC needs it.

### Metrics

Three producers exist today: `EngineMetrics` in the event loop (attention-TP
rank 0), `KVTransferMetrics` in the PD connection, and `RequestMetrics` in
`AsyncLLM`. After the rewrite the first two live in `ts-control` and
`RequestMetrics` in `ts-client`. The metric definitions in
`metrics/collector.py` (names, labels, buckets) move to one Rust module, and
Prometheus export becomes native with the same names, so dashboards do not
change.

Output plugins (`docs/metrics-plugins.md`) keep their contract for plugin
authors: the `tokenspeed.plugins` entry point, `register_metric_reporter`,
`MetricEventSink`, `MetricNodeSession`, `--metrics-reporters` and
`--metrics-reporter-config`. What changes is where the sinks run:

* The node launcher, which already owns the `MetricNodeSession`s, starts one
  metrics relay process per node. The relay loads the plugins and calls
  `emit` and `close`.
* Rust producers send `MetricEvent` records (the same fields) to the relay
  over a local socket and never block. A full queue drops the event and
  counts the drop, matching the `emit` contract.
* The raw-output rules (unchanged gauge assignments suppressed, refreshed on
  the next update after one second) run on the producer side, so relay
  traffic stays bounded.
* A slow or failing plugin cannot stall a rank or take its GIL.

Two alternatives were rejected. A Rust plugin trait would make plugin
authors rebuild the host, since Rust has no stable ABI for loading plugins
dynamically, and it would break the contract that shipped with
pluggable reporters. A Python bridge thread inside the rank process would
put the GIL back next to `ForwardThread`, against goal 2.

### NPU

The Ascend path adopts the Rust host like the GPU paths. Little in the
control plane is vendor-specific:

* The data plane is embedded unchanged, `torch_npu` included.
* The copy-event wait goes through `torch_npu`'s event, as the CUDA and ROCm
  waits go through torch's (see the deviations under Phase 4).
* PD on Ascend selects Mooncake's Ascend transport through the same C API.
* The Ascend CI hosts are ARM, so the host binary is built and released for
  aarch64 Linux as well as x86-64.

Serving the supported NPU models on the Rust control plane is part of the
Phase 3 exit. The Ascend CI (`docs/npu-operator-ci.md`) covers operators
only, so this exit needs a serving run on Ascend hardware.

### Startup

1. `ts-host` parses arguments, initializes CPython and imports the data
   plane.
2. The data plane runs `build_device_side` and returns `DeviceSpecs` as plain
   values (cache geometry, cache groups, speculation widths, capability
   flags).
3. The host builds the C++ scheduler from those specs, connects the
   collectives and the frontend sockets, and sends the ready message.
4. The data plane arms the sync-debug mode, marks the JIT serving boundary
   and freezes its startup heap; then `ForwardThread` starts pulling.

### Frontends

* **SMG**: the headless msgpack mode is the target; the gRPC servicer and its
  `AsyncLLM` go away.
* **Dynamo**: a Rust TokenSpeed backend built on `ts-client` replaces the
  Python `dynamo.tokenspeed` worker. This needs no Rust control plane and can
  land first (Phase 1).
* **Python `Engine`**: `ts-py` wraps `ts-client`; tokenization and
  detokenization use the HF tokenizer in Python, as today.

## Invariants carried over

Every principle in `docs/design/event-loop.md` holds in the Rust loop, and
the document was rewritten against the Rust code at cut-over:

* Principle 1 becomes structural: the control plane is a different language
  with no CUDA bindings, and the submission enum is the closed list of
  device interactions.
* Principles 2–5, the hooks pattern and the checklist translate directly:
  one coordinator body, two explicit advance call sites, the single
  overlap-break registry, one KV event drain per round.

`docs/design/scheduler.md` and `docs/design/cache-concepts.md` are
unaffected; the scheduler does not change.

## Migration plan

There is no separate measurement phase. Parity is judged by the existing CI
matrix and the e2e regression matrix, not by recorded round traces, and the
facts under "To verify" are checked in the phase that first depends on them.

**Phase 1 — wire and client.** `ts-wire` with the conformance test,
`ts-client`, and the Dynamo Rust backend against the current Python loop in
msgpack mode. Exit: Dynamo serves through `ts-client` with no Python worker.
SMG already carries a client-side port of this protocol (its
`engine_zmq_client` crate, which models a prefix of each message); once
`ts-wire` exists, SMG depends on it instead of mirroring the schema.

Status: `ts-wire` and `ts-client` are in `tokenspeed-engine/`. A Dynamo
worker built on `ts-client` serves aggregated, attention-DP and P/D text
traffic, relays KV events, and matches the Python path token for token.
Before the exit, multimodal requests and the control operations (profiling,
pause, weight updates) still have to move onto the msgpack wire. Headless
mode gained what an in-process `AsyncLLM` used to provide: the prefill
launcher hosts the KV bootstrap server, requests carrying stop strings are
refused instead of reaching a tokenizer-less stop check, and the launcher
kills schedulers that ignore termination.

**Phase 2 — host and the aggregated path.** `ts-host`, the submission FIFO,
the `cxx` facade, `ts-collective`, request state and output processing,
behind the temporary launch switch. Scope: aggregated serving with TP, DP
attention, overlap depth 0/1, speculation, CUDA graphs, and the Host L2
tier and retraction snapshot pool the default configuration turns on; PP's
control-plane mechanics (depth pp_size, the last stage's results), whose
serving needs PD. Exit: the model CI jobs pass on the Rust control plane.

Phase 2 lands in steps, each checked token for token against the Python
loop:

1. `ts-scheduler`: the scheduler core and its nanobind module compiled into
   the host binary, with the module registered as a built-in before the
   interpreter starts. The data plane's `import tokenspeed_scheduler` then
   gets the host's own module, so the plan objects the host creates and the
   ones Python reads are the same C++ types from one build, which is what
   the shared-library wording above was for. A `cxx` facade gives the
   control thread the scheduler; a plan crosses to Python by moving the
   C++ object into a nanobind instance.
2. `ts-host`: the embedded interpreter, the data-plane bootstrap
   (`build_device_side` in Python, its `DeviceSpecs` returned as values)
   and the submission FIFO. First a Python pump thread consumes the FIFO
   and calls the existing `DeviceHandle` unchanged; folding it into
   `ForwardThread`, so no second Python thread exists, is a later step.
3. One rank serving: request state, finish checks and output batches in
   Rust, TP1/DP1/PP1, in-flight depth 0 and 1.
4. TP and attention DP over `ts-collective`; then speculation; then PP.

Status: steps 1–3 are in `tokenspeed-engine/` (`ts-scheduler`,
`ts-control`, `ts-host`) and `engine/rust_host.py`, behind
`TOKENSPEED_RUST_HOST` in the headless launcher. One rank serves token for
token like the Python loop at in-flight depth 0 and 1, with chunked prefill,
prefix-cache hits, seeded sampling and stop tokens. The pump thread has
since been folded into the forward thread: one Rust thread attached to the
interpreter runs every data-plane operation in queue order, and the Python
`ForwardThread` is gone; commit waits on the copy event from the control
thread (see the deviations under Phase 4). The Host L2 tier and the
retraction snapshot pool, both on by default, are served too: the Phase 2 exit runs the model
CI jobs with their defaults, so their cache ops moved here from Phase 3.
The control thread counts each plan's cache ops and, while any is open,
polls their acknowledgments at the head of the round, before it plans, as
the Python hooks do. L3 storage stays in Phase 3.

Step 4: `ts-collective` connects every pair of ranks once at startup, after
the data plane's rendezvous has exchanged their addresses. Each round, the
replica's first member relays the frontend's messages to the rest of the
replica (attention TP, across pipeline stages), and the replica intersects
its cache acknowledgments before advancing them; with attention DP every
rank shares its batch's size, and a rank without one runs the idle forward.
A pipeline's last stage shares each forward's results with the earlier
stages before they commit. A stop request rides the same collectives, so
the ranks stop after the same round. TP, attention DP, TP with DP and
speculative decoding serve token for token like the Python loop.

PP is a prefill-node feature in this codebase: `--pipeline-parallel-size >
1` requires `--disaggregation-mode prefill`, so serving with it comes with
PD in Phase 3. The PP mechanics above are covered by the GPU-free tests;
the only aggregated PP mode, the `TS_PP_DEBUG_ALLOW_NON_PREFILL` hatch,
gives meaningful first tokens only, and no PP-capable model runs on the
development node today.

Under speculative decoding with forced retractions and concurrent requests,
outputs depend on whether a snapshot or restore copy lands before the next
round polls, so neither loop is deterministic from run to run, and the
faster Rust rounds take different, equally valid schedules. With each poll
made to wait for the device first, both loops are deterministic and agree.

**Phase 3 — the remaining subsystems.** PD (and with it PP serving), EPD,
L3 storage, context parallelism, pause and memory saver, RL weight ops,
EPLB, grammar, multimodal, metrics plugins, profiling. Exit: the full CI
matrix and the e2e regression matrix pass on the Rust control plane.

Status: every subsystem runs on the Rust control plane. PD runs on the
`ts-pd` transfer peer, which owns the Mooncake transfer engine through its C
API (`ts-mooncake`) and plans every route, block manifest, layerwise chunk
and copy over the cache transfer contract the data plane builds at startup
(`pd/host_peer.py` keeps only the settings and the device fences); the
bootstrap server is `ts-pd`'s too, which the launcher hosts through
`tokenspeed_engine`. Layerwise streaming reads the producer's step counter
directly. The memory saver's release deregisters
the prefill peer's cache regions and registers them again on resume, and
gives the decode peer a fresh transfer session. The EPD prefill receiver is
`ts-pd`'s too (the encode-rank handshakes, the receive slots, the agreement
on landed embeddings); the data plane keeps the pools' memory, the publish
copies and the shard reassembly. An EPD encode rank runs `ts-host` in encode
mode: `ts-control`'s encode loop relays, batches and leases send-ring slots,
`ts-pd`'s sender writes the embeddings, and the data plane keeps the vision
tower, the embedding cache and the ring copies. L3
storage, context parallelism (which needs no control-plane logic of its
own), pause, RL weight operations, EPLB (controller, solver and move plan),
grammar-constrained decoding (`ts-grammar`, xgrammar's C++ library),
multimodal inputs, metrics (`ts-metrics`, writing Prometheus' multiprocess
files and feeding the plugin relay), KV events, profiling, prompt and top-k
logprobs and request token history are served, each checked against the
Python loop on GPU where the development node can run it.

**Phase 4 — cut over.** One change switches every launch path to `ts-host`
and deletes the Python event loop and hooks, `AsyncLLM`, the
data-parallel controller, the Python wire encoder, the gRPC servicer path
and the launch switch; rewrites `docs/design/event-loop.md`.

Status: every launch path starts `ts-host` ranks (`TOKENSPEED_HOST_BINARY`
overrides the binary the `tokenspeed-engine` wheel installs); the Python event
loop, its hooks, the request handler, the output processor, the data-parallel
controller, the engine core client and pickle IPC, the Python headless
transport, the Python PD transfer executors, the EPD prefill hooks and the
Python encode loop are deleted, and `docs/design/event-loop.md` describes the Rust loop. `ts-py`
(`tokenspeed_engine._native`) is the Python frontend's connection to the ranks.
Four deviations from the plan above:

* `AsyncLLM` stays, as the in-process frontend over `ts-py`: tokenization,
  detokenization, stop strings, `n > 1` fan-out and the `Engine` API stay in
  Python, and so does the Python encoding of requests onto the wire
  (`zmq_wire`), which `ts-py` sends as it is.
* `ts serve` keeps the gRPC servicer, now over that `AsyncLLM`. SMG's
  msgpack backend does not yet carry what the CI model jobs need: MRoPE
  multimodal inputs, EPD encode handshakes, shared-memory features and
  top-k/prompt logprobs. `ts serve --headless` serves SMG's msgpack mode as
  before. Moving `ts serve` onto it, and deleting the servicer, waits for
  SMG.
* The control thread calls into Python for the data plane's completion
  queries, and only for those: it waits on a launched forward's copy event
  through the torch event the data plane records (vendor-neutral, as the
  data plane already is; the wait releases the GIL), reads the landed
  results, and polls the cache acknowledgments, under a
  dispatch guard that rejects device work. Queued behind
  the forward thread instead, these would wait for every forward still being
  launched; the native event entries would need an implementation per
  vendor. The cost is the GIL on the commit path: a query can wait for the
  forward thread's Python, up to the interpreter's switch interval, as the
  Python event loop's commit did.
* The L3 tier runs in the host rather than on the data plane: the executor,
  its stores (the Mooncake store client through the store's C ABI, which
  TokenSpeed's Mooncake build ships, and the in-memory test store) and its
  metrics are `ts-l3`, so neither the per-round storage poll nor the
  workers take the GIL. The data plane resolves the tier's settings at
  startup (`cache/l3/factory.py`) and keeps the Host pool the executor
  reads and writes by address. The store client keeps its own transfer
  engine, apart from PD's.

While Phases 2–3 are open, a change to the Python control plane must land
with its Rust counterpart, or in the Rust plane's backlog with a test that
covers it. This is the main cost of the plan, and the reason Phase 3 should
not be long.

## Risks

* **Velocity.** The control plane has changed shape repeatedly this year
  (snapshot retraction, L3 pre-admission prefetch, cache-group unification).
  Parallel development doubles the cost of each such change until cut-over.
* **Three languages in one process.** Rust, C++ and Python share the rank
  process; crashes, signals, `faulthandler`, and shutdown need one owner
  (the Rust host).
* **Shared C++ library.** The nanobind module and the Rust host must load the
  same scheduler library instance; a mismatch breaks the by-reference plan
  hand-off.
* **Out-of-tree plugins** that touch control-plane state (none known today)
  lose their Python access to it.
* **Build and release.** A Cargo build joins the wheel build and CI; the
  rank binary must be shipped for every supported platform, x86-64 and
  aarch64 Linux.
* **One more process per node** for the metrics relay, when output plugins
  are selected.

## To verify

The decisions above rest on these facts, which are not yet checked. Each is
checked in the phase that first depends on it (1–2 in Phase 2, 3–4 in
Phase 3); 1 and 2 went with the native event wait (see the deviations under
Phase 4):

1. Each vendor's torch exposes the native event handle: `torch.cuda.Event`
   on CUDA and ROCm builds, and the `torch_npu` event for Ascend.
2. The vendor event calls behind the `tokenspeed-kernel` entries need no
   current device context on the calling thread, so the control thread never
   creates one.
3. The pinned xgrammar wheel still ships the C++ headers and static library
   (checked on 0.2.3; TokenSpeed pins 0.2.8). If not, `ts-control` builds
   xgrammar from source.
4. Mooncake's C API covers every transfer engine call TokenSpeed makes today
   (initialize, register/deregister, session id and RPC port, sync and async
   writes, status checks), with the page-batch write added.
