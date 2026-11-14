# Ascend ACL Graph Memory Saver validation report

**PASS for the bounded latest-source functional scope:** root accepted all 20
required hardware groups and the required CPU qualification. Independent
architecture and raw-evidence reviews passed with no blocking source finding.
G01 final documentation, privacy and formatting review also passed. The
documentation-only refresh preserves all tested functional and test bytes.

[User guide](ascend-memory-saver.md) ·
[Design and CUDA comparison](../design/ascend-acl-graph-memory-saver.md) ·
[Acceptance commands](../../test/runtime/ascend_memory_saver/README.md)

## Validated source and stack

The frozen functional/test snapshot is tree
`6a4e33b25dea2e20b1c46b7aa50fec39c64abb6d`, with 3,797 source entries.
It integrates upstream main `def1ba90` with separate common protocol
[PR #3397](https://github.com/lightseekorg/tokenspeed/pull/3397) at
`2e0975c31d88661b2fe110e0fac860594ed62479`. That PR remains open, Ready for
Review and unmerged. [#3306](https://github.com/lightseekorg/tokenspeed/pull/3306)
and [#3310](https://github.com/lightseekorg/tokenspeed/pull/3310) are already
upstream. The source identifier is a tree, not a published commit; the final
documentation-only refresh does not change its functional or test bytes.
Four newly built native artifacts and dependency identities were checked
across six hosts.

| Dependency | Primary validated configuration |
| --- | --- |
| Device/OS | Atlas A2, Ascend 910B3, Linux aarch64, Python 3.12 |
| PyTorch | `2.9.0+cpu` |
| PTA | July 30 PTA 2.9.0-730 base `9d40ac3a2b3714df7e5d164bda28a42eff3282cd`, plus six official backports |
| Tested PTA / public equivalent | `ee1742b06d2c0e59cb0b357288ec29ee190cfd2d` / `74e6508b896f630e5d5cacd39a3877047e91c19a`; identical tree `546ac928ce6199c1e2f221d7f5279b4eef6713a8` |
| CANN/HCCL | 9.0.0 |
| Native saver/control | ABI 2; protocol 3 required for Ascend Saver |

The six official PTA changes are
[!39221](https://gitcode.com/Ascend/pytorch/merge_requests/39221),
[!45270](https://gitcode.com/Ascend/pytorch/merge_requests/45270),
[!30625](https://gitcode.com/Ascend/pytorch/merge_requests/30625),
[!32154](https://gitcode.com/Ascend/pytorch/merge_requests/32154),
[!32564](https://gitcode.com/Ascend/pytorch/merge_requests/32564), and
[!38431](https://gitcode.com/Ascend/pytorch/merge_requests/38431).
No additional PTA patches or system-library replacements are used.
[Build requirements](../../tokenspeed-kernel-npu/memory-saver/README.md#pta-options)

## Latest integration: CPU and hardware acceptance

| Required CPU check | Root-reviewed result |
| --- | --- |
| Runtime | 311 passed + 35 subtests; two genuine CUDA-only skips; four NPU SDK cases deferred to hardware |
| Protocol | 142 passed |
| Affected upstream | 349 passed; four genuine CUDA-only skips |
| Rust native | 407 passed, 16 ignored; nested subprocess summaries counted once |
| C++ | 1,515 passed; one expected ASan-only skip |
| Fake ACL / L3 library | 4 / 42 passed |

CPU checks make no live-Master or physical GPU claim. All four deferred NPU SDK
variants actually ran and passed in N19. CUDA/ROCm retain their existing
`torch_memory_saver==0.0.9.post1`, allocator/dependency paths and protocol 2
default. Physical NVIDIA/AMD feature and timing testing was unavailable;
CPU routing checks and code inspection are the available evidence.

All **20 latest-source hardware groups** are root accepted. Coverage comprises
13 singlecard cases, nine TP2/fault/HCCL supplemental jobs, the cross-host N10
transfer and 16 capped/full GSM8K jobs. These counts describe different views
of the declared plan and must not be added as independent acceptance totals.

| Groups | Accepted scope and limits |
| --- | --- |
| N00–N02 | Startup identity/live-backing checks, VMM and paused cleanup, pool ownership and backup/discard policy. |
| N03–N04 | Three-cycle plain graph discard and synthetic MTP graph backup, paired target/draft KV discard, fresh outputs and original graph/address checks. |
| N05 | 100 dummy-MTP graph/KV cycles; fresh output and verify/accept metrics match. **Weights stay resident** in this probe. |
| N06 | Actual tiny native EAGLE3, DFLASH and DSPARK, three cycles each; fresh outputs/metrics, original graph identities and virtual addresses. This qualifies graph recovery, not complete trained/native weight refresh. |
| N07–N08 | Complete disk A→B replacement, L2/L3 flush, same-version preflight retry and file-backed synthetic SDK target/draft refresh. |
| N09 | Actual loopback Mooncake TCP Store, CPU staging and synthetic SDK provider; complete target/draft refresh. This is not cross-host transport or production SDK qualification. |
| N10 | Actual CPU publisher → NPU reader cross-host TCP, B/version 2 then A/version 3. Each version transfers 311 target objects / 1,503,300,520 payload bytes with matching digests and owned connection evidence. B outputs/metrics match a separate fresh-B reference; A matches the initial freshly loaded A, with no separate fresh-A run. Original graph/VA survive. Planned master SIGTERM with exit 0 is distinct from forced worker cleanup. |
| N11 | Missing target, missing draft and asymmetric TP2 reader failure all refuse update/KV wake and remain sleeping. In the asymmetric case rank 0 reads 311 objects, rank 1 fails after 2 GETs and both commits fail. Restart proves **first token 0** and normal launcher exit, not full fresh-B equality. Namespace refusal is observed; the namespace field is test-assigned. |
| N12–N13 | TP2 disk/real TCP fresh-B equality and all-rank commit before one public success; drain cancellation without mapping changes, healthy preflight retry and expected post-unmap fail-stop. Faulted engine exit 1 is expected; outer verification exits 0. |
| N14 | Actual width-5 chain verification and clean width-6 capability refusal. |
| N15 | Eager initial collective sequence; graph/VMM initial replay plus three cycles. AllReduce sums are 3 then 5/7/9; rank-ordered AllGather values are checked in source, not full tensor dumps. VMM releases/remaps 2 MiB per rank; HCCL stays resident. |
| N16–N17 | Capped and full TP1/2/4/8 pause/control comparison coverage, detailed below. This acceptance does not imply TP4/8 numerical equivalence. |
| N19 | 517 passed + 35 subtests, two CUDA-only skips; all four real NPU SDK-stream variants passed. |
| N20 | Actual 1 GiB peer allocation/fill/readback while serving sleeps; driver-free HBM rises 3,350 MiB. Original graph/VA and same-checkpoint before/after outputs match; no independent fresh-model reference is claimed. |

Owned workers shut down normally without forced cleanup or leftovers; selected
cards are idle in final guards. The planned N10 master termination and N13
expected internal failure shutdown retain their explicit meanings. Stable
addresses and graph identities alone are not content-equality evidence;
output/metrics and weight witnesses supply separate checks.

## Latest full GSM8K results

Each job uses all 1,319 questions, a 512-token cap and concurrency 32, both
BEFORE and AFTER. Four TP pause/control pairs total eight jobs and **21,104
question executions**. All exit normally on unchanged boots with zero
recapture, no owned leftovers and passing all-eight-card final guards.
Pause arms retain six original graphs per rank.

| Configuration | Correct BEFORE /AFTER (of1319) | Wall elapsed ms BEFORE /AFTER | Mean wire TPOT ms BEFORE /AFTER | E2E output tokens/s BEFORE /AFTER |
| --- | --- | --- | --- | --- |
| TP1 pause | 838 / 842 | 245198.513 / 246853.068 | 22.506 / 22.673 | 1382.435 / 1370.629 |
| TP1 control | 838 / 842 | 241321.472 / 241950.900 | 22.148 / 22.230 | 1404.645 / 1398.399 |
| TP2 pause | 816 / 818 | 263944.746 / 263904.664 | 24.127 / 24.098 | 1290.437 / 1291.879 |
| TP2 control | 816 / 818 | 265944.418 / 271019.039 | 24.306 / 24.741 | 1280.734 / 1257.967 |
| TP4 pause | 844 / 836 | 262670.322 / 260738.698 | 23.808 / 23.743 | 1299.256 / 1301.989 |
| TP4 control | 822 / 830 | 265207.820 / 260951.470 | 24.291 / 23.855 | 1279.532 / 1301.016 |
| TP8 pause | 833 / 837 | 268274.037 / 274424.227 | 24.634 / 24.881 | 1263.290 / 1250.965 |
| TP8 control | 825 / 825 | 277112.091 / 276551.820 | 25.238 / 25.208 | 1229.109 / 1231.628 |

Wall elapsed time covers request submission through completion; control calls
are separate. Wire TPOT and E2E output throughput are observed workload
metrics, including instrumentation/host effects. These single paired jobs do
not establish a causal performance effect or a blanket no-regression result.

| TP | Changed token arrays within pause | Within control | Pause/control BEFORE differences | Pause/control AFTER differences |
| --- | ---: | ---: | ---: | ---: |
| 1 | 102 | 102 | 0 | 0 |
| 2 | 28 | 28 | 0 | 0 |
| 4 | 997 | 986 | 1,027 | 972 |
| 8 | 1,004 | 1,011 | 999 | 978 |

At TP1/2 the entire corresponding token arrays match across pause/control arms;
within-arm changes also occur without a pause. TP4/8 are not bitwise or
numerically equivalent. All five TP4 short split choices also occur in current
unpaused full BEFORE outputs, with 35 common-prefix pairs checked. Capped/full
workload and concurrency differ; no actual tensor/logit or arithmetic mechanism
proof was captured. These baseline counterexamples do not universally exclude
Saver effects. Numerical attribution (I00) and HCCL watchdog behavior (I01)
remain separately tracked.

### Capped comparison observations

Eight-question pairs use a 64-token cap and concurrency 1. All eight jobs
retain the original graph (one per rank in pause arms) with zero recapture.
Correct-of-eight at this cap is not full GSM8K quality.

| Configuration | Correct BEFORE /AFTER (of8) | Wall elapsed ms BEFORE /AFTER | Mean TPOT ms BEFORE /AFTER | E2E output tokens/s BEFORE /AFTER |
| --- | --- | --- | --- | --- |
| TP1 pause | 0 / 0 | 5184.797 / 5075.972 | 8.732 / 8.595 | 98.750 / 100.867 |
| TP1 control | 0 / 0 | 4955.612 / 4916.625 | 8.349 / 8.384 | 103.317 / 104.136 |
| TP2 pause | 0 / 0 | 5678.893 / 5320.258 | 9.471 / 9.054 | 90.158 / 96.236 |
| TP2 control | 0 / 0 | 5699.611 / 5623.465 | 9.487 / 9.428 | 89.831 / 91.047 |
| TP4 pause | 0 / 0 | 6218.901 / 5946.992 | 10.538 / 9.995 | 82.330 / 86.094 |
| TP4 control | 0 / 0 | 6015.462 / 5889.952 | 10.179 / 10.030 | 85.114 / 86.928 |
| TP8 pause | 0 / 0 | 6076.006 / 6175.038 | 10.251 / 10.432 | 84.266 / 82.914 |
| TP8 control | 0 / 0 | 6359.785 / 6231.398 | 10.723 / 10.562 | 80.506 / 82.165 |

### Latest first-use transition observations

Each full pause arm contains one first-use transition, not a steady timing
distribution. Wire spans run from frontend dispatch to reply. The ready endpoint
is the last successful physical-rank weight-commit **Python return**, and first
chunk means the first nonempty client chunk, not a captured native wire ACK.
First KV wake includes compilation and other overlapping work; it is not pure
compile time. Nested spans must not be added.

| TP | Suspend wire ms | Remap wire ms | Update+commit wire ms | KV wake wire ms | Last-rank commit return to first chunk ms | Suspend dispatch to first chunk ms |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 309.543 | 17.483 | 610.838 | 5152.372 | 5236.961 | 6180.523 |
| 2 | 178.672 | 18.173 | 508.920 | 5455.557 | 5553.252 | 6265.575 |
| 4 | 112.010 | 24.156 | 430.095 | 5555.600 | 5652.914 | 6226.065 |
| 8 | 83.102 | 20.495 | 319.723 | 5787.269 | 5895.737 | 6327.056 |

## Earlier N18 timing component

Separately accepted N18 measurements use old source tree
`9ce91417f8c6c9b77dbd125cad634e8e14895aa7` and native build
`7d91a917bd9de951d80d8777d3130081a74aed00`. They cover TP1/2/4/8 first use
plus three steady cycles and the three declared policy families. They are
**not latest-source 6a timings**. Strict rank/nested-span/synchronization checks,
physical effects, original graph/VA and natural cleanup were independently
reviewed; the final review rehashed 527 indexed evidence files.

| Old-source configuration | First ready → q1 chunk (ms) | Steady median (ms) |
| --- | ---: | ---: |
| TP1 plain weight backup | 5,405.212953 | 97.105553 |
| TP1 MTP retain | 5,414.328648 | 108.880512 |
| TP1 FILE SDK MTP refresh v4 | 5,554.919076 | 118.325001 |
| TP8 plain discard | 5,813.930098 | 117.211534 |

For reloads, ready is the last successful rank commit Python return; for backup
restore-only it is the restore-return boundary, with load/commit **N/A**. First
chunk is the intended q1 request's first nonempty client chunk. GC and pure
barrier time are **N/A** when unmeasured, not zero. Initial KV wake includes
compilation and other work; it is kept separate from three steady samples.

Refresh v4 uses **seven GSM prompts plus one raw-token weight witness**, with
q1 and the first seven prompts unchanged. Versions 2/3/4/5 and all assertions
are unchanged; all 32 after-output arrays equal independent fresh B. Initial A
q8 begins 12095, fresh B begins 0. The earlier all-math v3 attempt failed its
pretransition discriminator because its eight prompts did not expose the
swapped rows; it remains a failed attempt, not a completed transition sample.
The shared reference warms caches; this is not cold-performance evidence.

The frozen `32d7a642` → final common-protocol matched performance comparison
is **UNRUN / UNQUALIFIED**, a separate non-gating PR #3397 follow-up under
`root-n18-comparison-disposition.json`. Neither old N18 timings nor latest
GSM8K observations supply that missing comparison or prove no regression.

## Component and historical boundaries

Earlier `9ce91417` / DSpark-only `7a69332b` components remain preserved without
being counted as new 6a runs. The original DSpark auxiliary resource guard
records `resource.exit=1` despite an idle selected device; its later separate
32.743213887-second observation begins 786.009061979 seconds after failure
and does not reconstruct the missing original window. Current N06 acceptance
rests on its actual latest-source rerun.

Historical `7f89b744` / `32d7a642` qualification passed its declared 25 checks;
that checklist is distinct from current regression groups. Stock TorchNPU 2.13.0
is historical secondary compatibility: CANN 9.0.0 was tested, official pairing
is CANN 9.2.0, and there is no fresh current run on that route. HCCL rebuild /
graph-recapture prototypes are historical experiments, not delivered-path
comparisons. The delivered path retains HCCL and ACL Graph executables.

### FIA causal-length controls

Earlier FP16/BF16 controls distinguish full five-query causal verification at
KV length 105 from a narrowed second query requiring frontier 102 or an
absolute-position mask. Correct frontier/mask returns 0; the unmasked narrowed
query returns 28.578125 / 28.625. This is attention-metadata evidence, not a
full-model or Saver causality proof. Automatic captured drafting uses
Triton-computed lengths and preserves overlap; manually forcing narrowed FIA
remains restricted.

## Release scope and evidence references

The supported boundary is single-node aggregate TP1/2/4/8 with eligible greedy
chain models/kernels. Only managed physical backing is offloaded; reserved
virtual addresses, executables, persistent inputs and HCCL remain live.
Graph suspend discards shared target/draft device KV. Changed weights require
L2/L3 flush, restored mappings, complete load and all-rank commit before KV
wake. Driver-free readings measure reclaimed memory; reserved-address counters
can remain high. Trainer allocations must be freed before serving wake.

Real TCP evidence uses CPU staging and a synthetic SDK provider. Production
SDKs, RDMA, cross-node serving TP and complete trained/native draft refresh
remain outside qualification. Tests remain in their test directories with
English comments. Reproduce with the
[acceptance procedures](../../test/runtime/ascend_memory_saver/README.md).

The private evidence bundle retains raw logs, hashes and review records; it is
not a public source artifact. Key members are `root-hardware-ratification-6a.json`,
`latest-main-integration/root-cpu-review.json`,
`final-architecture-acceptance/{review.md,supplementreview.md}` and their JSON
checks, `release-evidence-review/final-old-timing-review.{md,json}`, and
`root-n18-comparison-disposition.json`. Source-lock and artifact hashes bind
these records to their actual phase; raw logs, binary/core files and private
addresses are not intended for public commits.

Private evidence integrity references (SHA256 prefixes):

| Bundle member | SHA256 prefix |
| --- | --- |
| `root-hardware-ratification-6a.json` | `fc638919de4114c7` |
| `latest-main-integration/root-cpu-review.json` | `2c46e4aecb84f1b9` |
| `final-architecture-acceptance/final-architecture-acceptance.json` | `8e0423bb52ccbae6` |
| `release-evidence-review/final-old-timing-review.json` | `1fd4ac62f9a57784` |
