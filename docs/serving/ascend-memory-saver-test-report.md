# Ascend Memory Saver validation

## Latest-main integration: pending

Qualification for main `4f969038274c2851a7d9358bcdaa3e8181bc313d` is **PENDING**.
The final integrated tree, fresh native build and CPU/NPU matrix have not yet
been accepted in this report. The results below are retained history only.
Installation belongs to the [guide](ascend-memory-saver.md), invariants to the
[design](../design/ascend-acl-graph-memory-saver.md), and reproduction commands to
the [smoke-test README](../../test/runtime/ascend_memory_saver/README.md).

## Historical main6d source and build

- Tested tree: `317649c7642aef1c36f46bf23b98f22f384f38f6` (3,826 paths),
  preserved by [the feature snapshot](https://github.com/yanlonghu/tokenspeed/commit/6e02de4567f72bf5ad432450f4095224e6a51898).
- Main: `6d7d29b918d1b050602cd945151dc99c076bd67f`. Common dependency:
  public tree `4c351df13e1da62399885a4d1dd7f507cd7e8605`, commit
  [`54dd431a7c09e22b7ac1321ac2714acf43c23b29`](https://github.com/yanlonghu/tokenspeed/commit/54dd431a7c09e22b7ac1321ac2714acf43c23b29).
- Single-host Atlas A2 / 910B3, Linux aarch64, Python 3.12, Torch `2.9.0+cpu`,
  PTA `2.9.0-730` base `9d40ac3` plus the six
  [official backports](ascend-memory-saver.md#before-you-start), CANN/HCCL `9.0.0`.

Archive, Git blobs/modes and patch reconstruction matched. ABI3 host/client,
shim and matching scheduler were rebuilt; the saver qualified for exact-source,
ABI and artifact reuse. Installed wheel members matched loaded modules.

## Historical CPU and native gates

| Gate | Result and counting boundary |
| --- | --- |
| Native primary harnesses | 357 passed, 16 ignored; excludes 13 nested subprocess summaries and includes 88 serving tests. |
| Selected CPU union | 348 passed, 21 skipped: startup/runtime 270 + compatibility 78. Its 13 startup subset consists of 6 multiprocess Gloo and 7 capability cases; not added again. |
| Weights/session/discard | 220 passed, eight subtests, four hardware SDK parameters deselected; partial-update/F03 subsets included. |
| Separate probes | 5 initialization refusals, 8 adapter dispatch cases, 64 selector assertions, 4 bootstrap refusals and 2 startup behaviors passed; excluded from pytest totals. |
| Loaded identity negatives | Seven module/DSO/runtime path, hash and mapping cases passed. |
| Native exit | Expected exit 1, one expected failure signal; three pure-C phases held GIL 1 and the original main PID/thread/thread-state without NPU initialization. |

Local Gloo does not qualify cross-node HCCL. The PP fixture has zero own tests
and six device consumers skipped; it provides no PP hardware qualification.

## Historical model-loaded NPU cases

| Case | Observation |
| --- | --- |
| TP2 v3 eager KV | Both ranks unmapped/remapped 943,718,400 bytes at the same VA; eight-token outputs matched. A 312.024823 ms Python-return test gate held the public reply until after Python returned. |
| TP1 complete disk B discard | Two prompts × 12 tokens matched fresh B and differed from A; cache count 192 became 0/0. Original weight VAs and graph survived without recapture. Parameter pool 1,296 MiB; persistent backup 260 MiB. |
| TP1 Memory Saver off | Eight tokens; actual scheduler exited 0 naturally. |
| TP1 cpu_backup partial | One of 310 sources supplied a 2,048-byte norm parameter. Two prompts × eight tokens matched fresh effective B and differed from A; all 63 parameter VAs and original graph survived. |

All seven schedulers exited 0 and leased devices returned idle. The injected
gate establishes Python-return ordering, not native wire ACK, transfer-in-flight
timing or performance. Cancellation/retry coverage came from native serving
tests, not live NPU in-apply cancellation.

## Retained failures and limits

PTA-loaded omitted/false exit controls observed GIL 0/current-state 0 before
SIGABRT(-6); they remain failed negatives. The responsible third-party DSO was
not identified. The ordinary non-NPU exit baseline was setup-blocked by missing
`tokenspeed_triton`. Initial harness-export, spawned-directory and offline-model
setup failures remain retained. Six out-of-scope CUDA Iris / NPU `flash_ops`
setup failures remain unqualified after CPU deselection.

The test-only trace follow-up at tree
`c03e2d78238d1c5ceb398dea81cb7a3c1970b230` retained tree317 production/native
sources. CPU: eight passed, one built-in-extension skip; actual native identity:
one positive and four refusals without device initialization. Its A2 TP1 run
passed three repository-trace cycles with CPU-backed weights: 254 MiB graph
pool release/remap at the same VAs, poison before first replay, both original
graphs, fresh-output equality at batch sizes 1/3/4 and natural exit 0. This is
not full weight discard or a rerun of the four-case matrix.

CUDA/ROCm hardware was unavailable. Global-shape/cross-rank partial inventory
strengthening remains deferred under #3456. Production SDK/RDMA, cross-node
PP/EP, MTP/draft refresh, long-cycle and performance matrices were not rerun.
Earlier TP4/8 and broader results remain in the
[pinned historical report](https://github.com/lightseekorg/tokenspeed/blob/9bb491a36a6eb10b60713fe7d5f8ad460047f191/docs/serving/ascend-memory-saver-test-report.md),
not latest-main acceptance.
