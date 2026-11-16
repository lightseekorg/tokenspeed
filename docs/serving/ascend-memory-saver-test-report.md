# Ascend Memory Saver: latest-main validation

Six NPU cases passed independent source-bound acceptance below.
Earlier main6d/tree317 and public4c results are
historical and are not transferred to this snapshot.

## Source and stack

- Main: `4f969038274c2851a7d9358bcdaa3e8181bc313d`.
- Common prerequisite: `c85e9a3656794f1bfb3f400f0a7d07a8fb0aa680`, owned by
  [PR #3397](https://github.com/lightseekorg/tokenspeed/pull/3397).
- Executed tree: `89e0f7101a0ba619f250d4fb271dcbd6a7167bf8`, 3,849 paths.
  The [feature snapshot](https://github.com/yanlonghu/tokenspeed/commit/c56ff4dd2f6b52add032a61ad58b7b6dc2fbf879)
  preserves this exact tested tree.
- Single-host Atlas A2 / 910B3, Linux aarch64, Python 3.12, Torch `2.9.0+cpu`,
  PTA `2.9.0-730` base `9d40ac3` plus six
  [official backports](ascend-memory-saver.md#before-you-start), build `ee1742`,
  and CANN/HCCL `9.0.0`. Stock PTA 2.13 remains historical compatibility only.

Engine `_host`/`_native`, module-host shim, scheduler 0.1.57 and C++ saver were
built normally from tree `65db98389e631a458f7192b3e1ad45ce3d10afba`.
Production/native blobs and modes are identical in the executed tree; subsequent
changes affect tests/docs only. Loaded modules and source/dependency identities
were checked. This does not imply every dependency was rebuilt.

## CPU and native results

| Group | Result and counting boundary |
| --- | --- |
| Native primary harnesses | 392 passed, 18 ignored; 13 nested subprocess summaries excluded. Serving 88 and scheduler 27 are included. |
| Startup/runtime | 271 passed, 8 skipped; no NPU initialization. |
| Weights/session/pools | 277 passed, 35 subtests passed, 4 hardware SDK parameters deselected. XML count 312 includes the 35 subtests; it is not 312 independent cases. |
| Fake-ACL backend | Four host-only state/fault tests passed; no device inference. |
| Actual host trace | One positive identity case and four identity-rejection cases passed without device initialization. |
| Native exit / no-init | Three pure-C phases observed GIL 1 and the original main PID/TID/thread-state. Expected bootstrap exit 1 and one expected failure notification; five explicit initialization attempts were refused. |

Groups, subtests and nested summaries are not combined into one pass total.
Controlled exit 1 is separate from the natural model-loaded exits below.

The startup subset of 13 is six real multiprocess Gloo cases plus seven capability
cases, counted once. Its eight skips are six device PP consumers, one second-GPU
case and one CUDA-device case; no device qualification follows from those skips.

## Six NPU cases

| Case | Observed result |
| --- | --- |
| TP2 protocol 3 eager KV / return hold | Both ranks physically released/remapped 943,718,400 KV bytes at the same VAs. Eight raw-prompt token IDs matched. A 300 ms Python-return hold prevented an early public reply. |
| TP2 natural-language KV | Two prompts × eight tokens matched before/after the same per-rank KV lifecycle. No forward, logit, sampling or token override was used. |
| TP1 cpu_backup partial | One of 310 canonical sources supplied a 2,048-byte norm parameter. Two prompts × eight tokens matched fresh effective B and differed from A; original weight VAs and graph remained. |
| TP1 full disk B discard | Two prompts × 12 tokens matched fresh B and differed from A; old cache count 192 became 0/0. Original parameter VAs and graph remained without recapture. Cache-flush and ambiguous/unsupported-format refusals preceded writes. |
| TP1 repository graph smoke | Three cycles discarded/remapped 254 MiB at the same VAs, poisoned pages before first replay, reused both original graphs and matched fresh outputs at batch sizes 1/3/4. Weights used CPU backup. |
| TP1 Memory Saver off | Eight tokens; actual model-loaded scheduler/launcher exited 0 naturally. |

All eleven scheduler processes across these cases exited 0. Leases were released
and devices returned idle. The return-hold probe establishes Python-return
ordering, not native wire-ACK timestamps, asymmetric native drain completion,
transfer-in-flight proof or performance. Natural-language recovery is checked
separately. Graph-pool discard does not by itself qualify weight discard.

## Retained failures and limits

The first graph attempt used an external backup-policy flag for a discard case.
The first partial attempt lacked an external helper import path. Both remain
FAIL records; only external CLI/environment setup changed before successful
retries. Frozen production and test sources were unchanged.

The ordinary non-NPU default-false native exit remains setup-blocked by missing
`tokenspeed_triton`; source/CPU isolation is not a runtime exit PASS. CUDA/ROCm
hardware was not run and no public or historical GPU result is inherited.
A separate fork-only NPU PP fixture probe stopped at the existing startup guard:
0 passed, 1 startup failure and 5 not run, before executor construction/forward.
It supplies no PP numerical or cross-process qualification and is outside the
core PR. Global-shape/cross-rank partial inventory remains deferred under #3456.
Production SDK/RDMA, cross-node PP/EP, native draft refresh, long-cycle and
performance qualification are outside this matrix.
