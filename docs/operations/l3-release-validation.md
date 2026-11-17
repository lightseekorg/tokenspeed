# L3 release validation and rollback runbook

This runbook covers the Mooncake Store L3 cache tier
(`--kvstore-storage-backend mooncake`, see
[server configuration](../configuration/server.md#host-l2-and-mooncake-store-l3)
and the [asynchronous L3 design](../design/async-l3-cache.md)): how to
validate a release candidate before enabling L3, and how to roll back
safely. L3 adds no second scheduling or execution path; validation and
rollback never change the path, only whether the storage backend is
attached.

## Gates and their status

| Gate | What it proves | How it runs | Status |
| --- | --- | --- | --- |
| G1 CPU/native protocol baseline | The executor's protocol against a live master: byte round trips, objects outliving their writer, KVP owner relocation, tenant isolation, recovery after master restart — repeated, fail-closed | Automated: the `Mooncake L3 Master Test` workflow runs `test/ci_system/l3_native_release_validation.py` | Required for every merge touching L3 |
| G2 GPU byte gate | Packed-byte D2H → PUT → GET → H2D round trips through the native store, byte-exact, writer process gone before the reader starts | Manual, on separately provisioned GPU hosts: `test/runtime/test_l3_flat_kv_e2e.py` with `MOONCAKE_GPU_TEST_CONFIG` | Required before L3 is enabled on any shared environment |
| G3 Native model serving gate | Model outputs with L3 match recomputation; prefix actually reused on every owning rank | Not yet established | **Pending — no evidence exists** |
| G4 Long mixed-workload soak | Hours of mixed read/write load with pressure, retraction and cancellation against the native store | Not yet established | **Pending — no evidence exists** |
| G5 Production rollout approval | G1–G4 evidence reviewed and accepted | Manual sign-off | **Pending until G1–G4 evidence exists** |

Do not treat G1 repetition as a soak, and do not treat any CPU result as
proof of GPU or RDMA correctness. Enabling L3 in production without G2–G4
evidence is not approved.

## G1 — CPU/native protocol baseline

The runner repeats the ts-l3 live master suite
(`cargo test --locked -p tokenspeed-l3 --test mooncake -- --test-threads=1
--nocapture`) in isolated subprocesses, one process group per iteration with
a hard per-iteration timeout. It strictly sets `MOONCAKE_REQUIRE_MASTER=1`,
checks the native store library and master binary before starting (the
tenant and restart tests always start their own master, so the binary is
required even when `MOONCAKE_MASTER` names an external one), and resolves
both paths to absolute before launching in the engine workspace. A suite run
counts as a pass only with a zero exit AND exactly the expected output: one
`running N tests` header, one `test result` summary with no failed, ignored,
measured or filtered-out tests, every one of the suite's five tests present
and ok (a deleted, renamed or unexpected test is a failure, never a smaller
pass), and no skip marker; `--nocapture` makes the marker visible in the raw
log. A launch error, an unreadable log or any unexpected runner error fails
the run, and the report never says `passed` before every requested iteration
completed.

Cleanup is identical on every exit path — normal leader exit, timeout,
interruption or a repeated cancellation signal: the iteration's own process
group is swept (a SIGTERM grace on the timeout and interrupt paths, then
SIGKILL), remaining members such as a leaked master are killed before the
leader is reaped, and SIGINT/SIGTERM are deferred until cleanup finishes. Only
the group the runner started is ever signalled; no other process is touched.
Evidence is durable: `report.json`, one `iteration-NNN.json` receipt and one
`iteration-NNN.log` raw-output file per iteration. The output directory must
be absent or empty; earlier evidence is never overwritten.

### Run locally

Prerequisites: a Rust toolchain, and the `tokenspeed-mooncake` wheel pinned
in `python/pyproject.toml` (it ships `libmooncake_store.so` and
`mooncake_master`). On Linux the wheel links CUDA 13 and RDMA user-space
libraries even for host-only TCP tests; the workflow's
`Install Mooncake master runtime` step lists the exact packages.

```bash
mooncake=$(python -c 'import tomllib; print(next(r for r in tomllib.load(open("python/pyproject.toml", "rb"))["project"]["dependencies"] if r.startswith("tokenspeed-mooncake")))')
python -m pip install "$mooncake"
rustup toolchain install stable --profile minimal
export TS_MOONCAKE_STORE_LIBRARY="$(python -c 'import mooncake, os; print(os.path.join(os.path.dirname(mooncake.__file__), "libmooncake_store.so"))')"

python test/ci_system/l3_native_release_validation.py \
  --iterations 3 \
  --iteration-timeout 900 \
  --terminate-timeout 20 \
  --wall-clock-budget 10800 \
  --engine-dir tokenspeed-engine \
  --output-dir /path/to/l3-validation-evidence   # must be absent or empty
```

`--wall-clock-budget` is the total wall clock the run may need in seconds;
`iterations × (iteration-timeout + 3 × terminate-timeout)` plus a 600-second
setup margin must fit it, including the bounded cleanup waits,
so a run cannot request repetitions its job cannot finish.

Exit codes: `0` passed, `1` failed or timed out, `130` interrupted, `2`
usage error (including bounds violations and a non-empty output directory).

### Run in CI

Pull requests run one iteration automatically. For a repetition run, use
**Actions → Mooncake L3 Master Test → Run workflow** with:

- `iterations`: suite repetitions, bounded to 1–20 (default 3).
- `iteration-timeout-seconds`: per-iteration timeout, bounded to 10–3600
  (default 900). `iterations × timeout` plus the setup margin must fit the
  job's wall-clock budget: 1800 s for the 30-minute pull-request job, 10800 s
  for the 180-minute dispatch job; the runner rejects what does not fit.

The workflow step `exec`s the runner, so job cancellation and the job
timeout signal the runner itself; it then writes an `interrupted` report
before exiting, and the `always()` upload still has evidence.

Receipts and logs upload as the
`l3-native-validation-<run_id>-<attempt>` artifact on success **or**
failure. `report.json` records the source identity, the exact command and
configuration, the resolved `tokenspeed-mooncake`/cargo/rustc versions (or
explicit `unavailable`), platform and elapsed time; it never collects the
process environment.

The source identity has three parts with exact semantics:

- `source.commit`: the commit that was actually checked out and tested,
  resolved from the checkout itself. On `pull_request` runs this is the
  merge commit the checkout produced, **not** the PR head; on push and
  dispatch runs it is the ref's commit.
- `source.dirty`: whether the working tree had modifications or untracked
  files. CI checkouts must show `false`; treat `true` (or `null`, when git
  could not determine it) as evidence that cannot be tied to a commit.
- `source.label`: an operator label recorded for cross-checking, never
  trusted as the tested revision. Pull-request runs label
  `pull_request-<number>-head-<sha>`; push and dispatch runs label
  `<event>-<sha>`.

## G2 — GPU byte gate (separately provisioned)

This gate is manual and needs its own hardware: a CUDA host for the test
processes, a running Mooncake master, and an **independent** storage owner
(provider) with `global_segment_size > 0` in a separate process or host.
The test clients mount nothing (`global_segment_size=0`), so writer and
reader processes can exit completely before the next one allocates GPU or
Host buffers.

```bash
export MOONCAKE_GPU_TEST_CONFIG='{
  "master_server_address": "store.example:50051",
  "local_hostname": "10.0.0.2",
  "metadata_server": "P2PHANDSHAKE",
  "protocol": "rdma",
  "global_segment_size": 0,
  "tenant_id": "l3-release-validation"
}'
python -m pytest test/runtime/test_l3_flat_kv_e2e.py -v
```

The native cases run both `direct` and `kernel` I/O settings, restore
objects into different block slots and check exact bytes, then delete them
and verify safely retired misses. Use `rdma` where the target deployment
uses RDMA, `tcp` otherwise. `test/runtime/test_l3_kvp_gpu.py` exercises
KVP owner relocation over RDMA with the same configuration and is
recommended additional evidence. Without `MOONCAKE_GPU_TEST_CONFIG` the
native cases skip; a skip is not evidence.

## Pending gates

- **G3 native model serving gate** — serve with
  `--kvstore-storage-backend mooncake` against the native store with fresh
  processes and empty local caches, verify successful GETs on every owning
  rank and the prefix actually reused, and compare outputs with a
  recomputation run (prompts must require information in the restored
  prefix). Not yet established; **pending**.
- **G4 long mixed-workload soak** — hours of mixed read/write load with
  memory pressure, retraction and client cancellation against the native
  store, watching the L3 pressure, health and quarantine gauges. Not yet
  established; **pending**. G1 repetition is repeated protocol validation
  and is not a substitute.
- **G5 production rollout approval** — sign-off requires the G1–G4
  evidence below; **pending** until it exists.

## Minimum evidence manifest

Attach the following to the release record before any rollout:

1. The CI run URL and its `l3-native-validation` artifact:
   `report.json` shows `overall_status: "passed"`, `source.dirty: false`,
   and a `source.commit` that matches the candidate: identical to the
   candidate SHA on push/dispatch runs, the candidate's merge commit on
   pull-request runs (whose `source.label` carries the PR head SHA). Every
   receipt passed, and the raw iteration logs are present. Dispatched
   repetition runs should show at least 3 passed iterations.
2. G2: the full pytest output for both I/O backends on the target GPU and
   interconnect, the exact `MOONCAKE_GPU_TEST_CONFIG` used (with any
   secrets redacted), and the master/provider versions.
3. The resolved versions: `report.json`'s `versions` section covers G1;
   record the same wheel version for the G2 hosts.
4. For G3/G4, once established: serving configurations, source revisions,
   reused-prefix and output-comparison results, and the soak's duration
   and gauge excerpts.

## Stop/Go criteria

**Stop** (do not merge, deploy or expand) on any of:

- A failed, timed-out or interrupted iteration, a skip marker, or a
  zero-test/ignored/measured/filtered/incomplete/misscoped suite in G1 —
  including a missing, renamed or unexpected test.
- A prerequisite failure (store library, master setup) — it fails the run
  closed; never bypass it.
- Any revision or version mismatch between the evidence and the candidate,
  a `source.dirty` that is not `false`, or an `unavailable` source commit.
- A byte mismatch, an unretired or unexpected object failure in G2, or a
  skipped native case counted as a pass.
- A quarantine or an unproven native retirement observed in any
  environment.

**Go** requires: merge — G1 green on the candidate (the PR's tested merge
commit, clean tree); enabling L3 on any shared environment — additionally G2
byte-exact on the target hardware; production rollout — G5 sign-off over the
complete manifest.

## Phased rollout

- **Phase 0 — L3 off (default).** Omit `--kvstore-storage-backend`: no L3
  executor, workers or SDK client are created, and serving runs the same
  single path on L1/L2.
- **Phase 1 — one replica, client-only.** Enable L3 on a single replica
  with `global_segment_size: 0` in
  `--kvstore-storage-backend-extra-config`, against an independently
  deployed master and providers. Watch the L3 pressure-gate, backend
  health and quarantine gauges.
- **Phase 2 — more replicas.** Expand only with identical weights, packed
  layout, parallel geometry and runtime compatibility epoch, so all
  replicas share one namespace.
- **Phase 3 — worker-mounted segments (optional).** Set a positive
  `global_segment_size` on inference workers only after provider behavior
  in the earlier phases is reviewed; account for the combined Host
  allocations.

Every phase requires G1 green on the deployed SHA and, before first
enablement on the target hardware, G2 byte-exact evidence.

## Degraded fallback vs unsafe-I/O quarantine

These are different states; do not conflate them.

**Cache-tier degraded fallback (no restart).** When queue count, queued
bytes or operation age reach a high watermark, the scheduler closes L3
read/write admission: new requests bypass L3 preparation and admit
ordinarily, nonowning backup candidates are dropped, and already submitted
I/O keeps its lifetime and deadlines. The gate reopens only when all
observations reach their low watermarks. Independently, the executor's
health monitor rejects work during a backend outage's cooldown and permits
one recovery probe. Serving continues on L1/L2 throughout. Operator
action: investigate provider/master health and queue sizing; there is no
data-safety concern and nothing to restart.

**Unsafe-I/O quarantine (restart required).** A native result that cannot
prove it stopped accessing its Host buffers quarantines every reference of
that job: the executor rejects all further L3 submissions and `close()`
reports that a process restart is required. Ordinary computation and valid
L2 hits continue within the remaining capacity. Operator action: schedule
a restart of the affected processes using the drain procedure below.

Never do any of the following: treat an empty queue or a soft-deadline
timeout as proof that native I/O retired; use `/flush_cache` or a cache
clear to free quarantined references (a full cache clear is refused while
storage operations, Ready holds or quarantine are outstanding); or
deregister storage while retirement is unknown.

## Drain and deregistration

To remove an inference worker safely:

1. Remove it from routing so it gets no new requests; let in-flight
   requests finish. Soft-deadline expiry and client cancellation do not
   release native I/O references.
2. Stop the process normally. At shutdown the L3 executor joins its
   workers with a deadline and closes the store only when native
   retirement is known.
3. If retirement stays unknown, the executor refuses store close and Host
   deregistration and keeps the store client for the process lifetime, as
   the data plane keeps the Host allocation. Let the process exit in that
   state — a dead client whose buffers were never freed is recoverable at
   the master/provider, while force-freeing buffers that native I/O may
   still write is not. Do not `kill -9` ahead of the executor's own
   report, and never use a timeout or flush as a force-free.
4. Master and providers have independent lifetimes: the master holds
   metadata, providers hold the cached bytes, and both can and should
   outlive inference workers. Deregister a provider only after every
   client has drained; reuse after inference restarts requires master and
   providers to retain the objects.

## Namespace compatibility and rollback to L2

Every object is keyed under a `tsl3-` namespace that covers the loaded
target/draft checkpoints, explicit weight versions, the packed Host
layout, the parallel geometry (including ordered group shard counts) and
the cache-producing runtime choices, isolated by a runtime compatibility
epoch. Identical configurations share a namespace; any difference starts a
separate, cold namespace — objects are never converted across epochs.

**Rollback.** Restart the workers without `--kvstore-storage-backend`.
No L3 executor or store client is created and the same single
serving/scheduling path runs L1/L2 only. A process restart is required —
not a reload — because the Host allocation and native I/O references are
held for the process lifetime. Follow the drain procedure above for the
outgoing processes.

**After rollback.** The shared namespace is untouched: objects remain on
the master and providers, whose lifetimes are independent of the inference
workers, and L2-only serving never reads them. Rolling forward again with
an identical configuration reuses them; a changed configuration starts
cold. Purge the remote namespace only once no deployment will roll forward
into it again. With L3 enabled, weight updates require an explicit
`weight_version` and a successful flush; a metadata-only version update is
rejected.

## What G1 repetition does not prove

G1 runs the executor protocol against a local master over TCP on CPU. Any
number of passed iterations still says nothing about RDMA behavior, GPU
byte movement, model outputs, availability under sustained load, or
production tail latency. Those belong to G2–G4 and remain pending until
their evidence exists.
