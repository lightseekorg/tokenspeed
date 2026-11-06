# Communication capacity and Op bandwidth

`HardwareProfile` reads the existing JSON/YAML formats through the same reader.
Version 2 optionally accepts `communication_capacities`, a list of sourced,
route-specific denominators. Omission and an empty list preserve all existing
compute/HBM behavior. Neither the GPU name nor topology adjacency selects a
communication denominator. Input bytes, SHA256 and capture/offline equality
checks also cover these capacities.

`scope_metrics(entries, duration_ns, hardware, *, scope_kind=None,
communication_observation=None)` returns effective AlgBW and BusBW using the
existing `CollectiveTraffic` formulas. `scope_kind="op"` and exactly one cost
entry are required for `estimated_comm_utilization`. The observation is that
OpRecord's `scalar_args`. A layer/model can retain effective bandwidth in JSON,
but never receives communication utilization or a communication NVTX label.
The default `scope_kind=None` deliberately cannot claim an Op denominator.

`estimated_comm_utilization` requires a reviewed **directional payload model**.
The selected MNNVL one-shot model is described below. Its TX and RX ratios use
separate one-direction reference denominators and the same whole Op GPU span,
including fused compute and gaps. The reported ratio is the larger **modeled**
ratio (0.25 means25%); it is not a measured physical maximum or link counter.
Values above one remain visible with `estimated_comm_utilization_warning` and
are never clipped. Failed scopes, incomplete cost/time, single-rank collectives
and unknown or incompatible evidence remain N/A.

Pure and fused communication Op labels retain `commAlgBW`, `commBusBW` and
`Est.CommBWUtil`. AlgBW and BusBW always keep the NCCL-tests convention. A
capacity with `normalization: nccl_tests_bus_bytes` can supply
`estimated_comm_reference_efficiency = BusBW / capacity`, but this is a separate
reference-efficiency field. It does **not** supply `Est.CommBWUtil` without a
reviewed directional model. In particular, the two IPC reference entries do not
stand in for the selected MNNVL route.

## Known unmodeled communication

An explicit cost rule can establish that an Op communicates while leaving its
traffic unmodeled. It records a nonempty reason in `OpCost.assumptions` with the
exact prefix `comm_unavailable_reason: `, for example:

```python
"comm_unavailable_reason: distributed argmax NVLS traffic unmodeled; no AllReduce payload inferred"
```

The shared calculation deduplicates these reasons into `comm_unavailable_reason`;
native Op annotations retain `Comm=N/A: <reason>`. This marker creates no
`CollectiveTraffic`, payload bytes, group, bandwidth or capacity denominator and
does not change independent compute/HBM metrics or any explicitly modeled
traffic. A logical boundary-IO estimate can therefore coexist with unmodeled
communication. Absence of `communication` alone does not prove an Op communicates
and must not add a collective label or infer an AllReduce payload. Complete
communication metrics still require the explicit traffic and route contracts
below.

## Capacity entry

Each entry requires these fields:

| Field | Meaning |
| --- | --- |
| `id` | Unique identifier in this selected hardware profile |
| `bytes_per_second` | Positive finite decimal byte/s, per rank, one direction |
| `scope`, `direction`, `normalization` | `per_rank`, `one_direction`; normalization is `nccl_tests_bus_bytes` or `trtllm_mnnvl_oneshot_remote_payload` |
| `world_size` | Exact participating group size, at least two |
| `placement` | `single_node` or `multi_node` |
| `topology`, `transport`, `backend`, `algorithm`, `protocol` | Exact reviewed route tokens; no implicit wildcard or SKU fallback |
| `source_id`, `conversion_source_id`, `normalization_source_id` | IDs in the existing `sources` list |
| `published_value`, `published_unit`, `published_direction`, `published_scope` | Positive value; `byte/s` or `GB/s`; `one_direction` or `bidirectional_sum`; `per_rank` or `per_link` |
| `conversion_factor`, `conversion_reason` | 1 for one direction or 0.5 for a sourced symmetric bidirectional sum; reason explaining that conversion |
| `normalization_reason` | Explain why this directional bound matches the selected traffic model; equivalent BusBW reference efficiency is distinct from directional utilization |

`bytes_per_second` must equal `published_value * unit_scale *
conversion_factor`. A `per_link` source additionally requires `transport:
nvlink`, `active_link_count` and `link_version` (raw NVML enum), both positive integers,
`link_generation` (such as `"4.0"`), `link_version_source_id`, and
`required_link_rate_bytes_per_second`, the raw driver full-rate predicate; multiply
by `active_link_count` as well. All participating ranks must prove that exact
active port count/version and minimum per-port raw rate. The raw rate is only
a state-matching predicate and is never substituted for the published directional
capacity. The hardware profile records the required count;
the observation records the actual count. A theoretical reference is not an
achievable payload benchmark; encoding, protocol overhead and remote reductions
can change physical bytes. NVSwitch multicast/reduction paths require their own
reviewed normalization, not an inferred point-to-point route.

## Required OpRecord host metadata

The backend records the route actually selected for this Op, using
`current_op_metadata()` within its existing outer semantic wrapper. Keep nested
record suppression: don't add a second cost record for internal communication.
All values below are JSON scalars or tuples of scalars; offline JSON lists are
accepted without changing ordered rank identity.

| `scalar_args` key | Required value |
| --- | --- |
| `comm_group` | Exact `CollectiveTraffic.group` identity, not just group size; the legacy `nccl:` identity prefix does not prove a transport |
| `comm_world_size`, `comm_ranks`, `comm_rank` | Actual group size, ordered global ranks, and this process's global rank |
| `comm_rank_hosts`, `comm_rank_device_uuids` | Ordered host identities and GPU UUIDs, aligned one-to-one with `comm_ranks`; devices must be distinct |
| `comm_scope`, `comm_direction`, `comm_normalization` | `per_rank`, `one_direction`, and the selected reviewed model; these define the analysis, not measured traffic |
| `comm_placement` | `single_node` or `multi_node`, checked against rank hosts |
| `comm_topology`, `comm_transport`, `comm_backend`, `comm_algorithm`, `comm_protocol` | Actual route tokens, matching exactly one capacity entry |
| `comm_transport_source`, `comm_topology_source`, `comm_placement_source`, `comm_route_source` | Nonempty references to immutable evidence from this run (content hash plus source/record locator recommended); a configured preference is not proof of selected transport |
| `comm_rank_active_link_counts`, `comm_rank_link_versions`, `comm_rank_min_link_rate_bytes_per_second`, `comm_link_source` | Required for a `per_link` denominator: count, raw NVML version enum (6 means generation4.0) and minimum raw active-port rate (decimal byte/s) per ordered rank, plus immutable active-link evidence |

The recorder/caller must establish run/rank provenance and that evidence covers
this Op's group. The helper checks the group/size, distinct ordered devices,
placement, exact route tokens, and any selected hardware UUID. It does not read
the machine, infer a path, or verify arbitrary evidence-file contents. Runtime
backend instrumentation and offline graph binding remain responsible for
supplying genuine matching evidence; unknown route decisions must stay absent.
Dynamic choices belong to each replay's rebound OpRecord, never a copied warmup
choice. Plain accelerated AllReduce needs its own outer semantic wrapper when
it is the selected route.

The graph adapter must call `scope_metrics(..., scope_kind=kind,
communication_observation=record["scalar_args"] if kind == "op" else None)`
after rebinding that replay's metadata. Preserve all returned communication
fields in the output row. On any graph attribution/coverage/error failure,
clear `estimated_comm_utilization` alongside AlgBW/BusBW and set its missing
reason. Do not compute a Model/Layer communication percentage from child rates.

## H800 evidence and remaining selection

The NVIDIA-authored [H800 product brief 2631447](https://serverflow.ru/upload/iblock/88a/2fg6cfwqo83oxx9nynevdxkdllj1hbmg/%D0%A1%D0%BF%D0%B5%D1%86%D0%B8%D1%84%D0%B8%D0%BA%D0%B0%D1%86%D0%B8%D1%8F%20NVIDIA%20H800.pdf)
(February 2023, third-party mirror) lists NVLink 400 GB/s for H800 SXM and PCIe,
but does not label the direction. It alone does not justify 400 or 200 GB/s as a
rank-local one-direction denominator.

NVIDIA's [Hopper architecture description](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)
explicitly defines NVLink 4 as 25 GB/s in each direction per link. This provides
a possible composable reference: eight proven active NVLink 4 links at that
full rate would give `8 * 25e9 = 200e9` byte/s per endpoint per direction.
This conditional derivation does not borrow H100's 18-link count.
NVML version is an enum: NVIDIA's [version definitions](https://nvidia.github.io/cuda-python/cuda-bindings/13.4.4/module/generated/cuda.bindings.nvml.NvlinkVersion.html)
define raw `6` as NVLink4.0. Do not confuse raw6 with generation6 or raw4
with generation4. Preserve the raw enum in `comm_rank_link_versions`; the
capacity carries the raw enum plus its separately validated generation. The actual
H800 link version/count, full-rate state, endpoint mapping and selected route
must be established independently. NVIDIA's [nvidia-smi documentation](https://docs.nvidia.com/deploy/nvidia-smi/)
says `nvlink --status` reports active link bandwidth and `--dataRate` accounts
for FEC overhead; it does not establish that a status number is useful payload
bandwidth. Preserve raw observations and their units rather than multiplying an
unexplained raw rate. NV8 in `topo -m` alone proves neither this capacity nor the
transport used by a collective.

NVIDIA's [NCCL-tests bandwidth definition](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md)
provides the equivalent collective-byte convention. A backend using remote
reads, one/two-shot reduction or multicast must retain its exact route and the
reviewed meaning of that comparison; equivalent bytes need not be its actual
link bytes. The H800 example includes conditional200GB/s endpoint references for the
reviewed TRT-LLM IPC Lamport one/two-shot reference routes and the directional
MNNVL one-shot model. They require eight active
links, raw NVML enum6 (generation4.0), and a raw driver status rate matching
26.562GB/s on every rank. This raw rate only checks the selected link state;
the denominator is the separately published25GB/s directional rate times eight.
Known effective rates remain available with N/A utilization until actual route
and ordered rank placement evidence match those conditions. Hardware selection from a past capture is not rewritten by
editing a profile later.


## MNNVL one-shot directional payload model

For the reviewed fp16/bf16 one-shot kernel, let `S=T*H*2` be the full input
bytes and `P` the actual group size. Its16-byte vector stores partition the input
exactly: each rank writes its S-byte shard once to its CUDA symmetric-memory
multicast mapping. Each receiving rank's local replica contains one S-byte slot
per source rank; the kernel polls these local slots and reduces locally. The
modeled useful cross-rank payload is therefore:

| Direction | Payload per rank | Rate | Reference denominator |
| --- | --- | --- | --- |
| TX multicast injection | `S` | `S / whole_Op_GPU_span` | One rank's200GB/s TX reference when all hardware predicates match |
| RX from other ranks | `(P-1)*S` | `(P-1)*S / whole_Op_GPU_span` | One rank's200GB/s RX reference under the same predicates |

The model reports both directions' bytes, rates, capacities and ratios as
`estimated_comm_{tx,rx}_payload_bytes`,
`estimated_comm_{tx,rx}_payload_bytes_per_second`,
`estimated_comm_{tx,rx}_capacity_bytes_per_second`, and
`estimated_comm_{tx,rx}_utilization`. `communication_bottleneck_direction` is RX
for P>2 (a tie at P=2); `estimated_comm_utilization` is the larger modeled ratio.
TX and RX are not added before dividing by a bidirectional total. Standard
AllReduce BusBW remains `2*(P-1)/P*S/span`, independently of this model.

These are **useful remote-payload estimates**, not physical TX/RX byte counts.
The source proves the local replica exists but not whether its own contribution
physically loops back through the switch. That possible loopback, packet and
encoding overhead, retries and hardware control traffic are unmodeled. Lamport
sentinel values share the existing payload words; no separate application flag
payload is appended. Polling reads local replicas, buffer rotation/dirty-clear
writes are local, and reduction/epilogue compute is local; their time remains
in the full fused span. HBM effects are not NVLink RX bytes. The model does not
claim remote reduction hardware or infer switch bisection capacity.

The automatic binder requires the actual selected MNNVL object and one-shot
pattern0 (plain AR) or1 (AR+residual+RMSNorm), positive16-byte-aligned payload,
element size2, and a payload byte count agreeing with the cost's full array.
It checks the actual CUDA symmetric-memory allocator, nonzero multicast handle,
local/peer pointer geometry and ordered group rank identity. Allocations are
recorded once during workspace creation as immutable
`rank-N.mnnvl-allocation-SHA.json`, tied to that run's communication snapshot.
If a model allocates before the profiler is bound, host-only facts are frozen
at configured profile startup. No allocation/rendezvous, device read or probing
is added on the Op/replay path. `comm_allocation_source` and
`comm_payload_model_source` identify these facts and the reviewed source;
`comm_addressing=nvls_multicast` and `comm_allocator_backend=CUDA` retain the
actual route separately from `comm_workspace=mnnvl`/`comm_protocol=mnnvl`.
Missing APIs, allocator/mapping failures, a changed workspace, two-shot, other
patterns, multi-node placement, or missing link evidence keep the model N/A.

The source chain is the workspace creator's `symm_mem.empty/rendezvous` and
actual handle, PyTorch's [CUDA symmetric allocator](https://github.com/pytorch/pytorch/blob/v2.14.0/torch/csrc/distributed/c10d/symm_mem/CUDASymmetricMemory.cu)
(`cuMulticastCreate`, device/memory binding, then mapping), and the selected
launcher passing that mapping into the one-shot kernel's multicast vector store
and local poll/reduce. NVIDIA's [CUDA12.6 multicast support](https://docs.nvidia.com/cuda/archive/12.6.3/cuda-c-programming-guide/index.html#multicast-support)
defines this mapped NVLS route. The reviewed kernel uses a vector store to the
mapping; this documentation does not relabel it as a `multimem.red` instruction.
Kernel/creator/source hashes and actual allocator facts are retained for audit;
a function name alone is not transport evidence.

The `profiling` extra includes `PyYAML` and `nvidia-ml-py>=13.580.82` (`pynvml`).
Optional driver/NVML capabilities are probed once at configured startup. An
unsupported query or absent optional API records an unavailable reason and
leaves communication normalization N/A; it does not change execution backend.
The startup snapshot is an estimate's evidence, not continuous link monitoring.
A later link-state change requires a new capture to establish a new denominator.
Captured hardware bytes and hashes remain immutable: adding this model does not
recalculate old captures with a different hardware profile.
