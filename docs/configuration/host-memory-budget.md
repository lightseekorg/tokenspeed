# Deployment Host memory budget

Each rank checks its Host L2 and retraction snapshot pool against an equal
share of its node's available memory when it starts
([sizing](./server.md#host-l2-and-mooncake-store-l3)). That check sees one
process at one moment. It reserves nothing for other jobs, replicas or
prefill/decode worlds on the same node, and it does not count Mooncake client
segments, staging buffers or storage providers.

`tokenspeed cache-budget` checks a whole deployment offline. It sums the Host
memory an explicit JSON manifest declares, per node and per container, and
reports any limit those declarations exceed. It needs no GPU, model or
running server:

```bash
tokenspeed cache-budget deployment.json --format text   # breakdown
tokenspeed cache-budget deployment.json --format json   # machine-readable
```

The command exits with 0 when every node and container fits, 1 when any is
overcommitted and 2 when the manifest is unreadable or invalid. Invalid
manifests are not evaluated: the report lists every error with its JSON
path. The JSON report has sorted keys and lists, so identical manifests
produce identical output.

The result is a **declared budget estimate**. It reserves no memory, takes
no part in runtime admission, and does not measure or bound live RSS. A
manifest that fits can still run out of memory if its declarations are
wrong or the processes grow beyond them. The check is arithmetic over
declarations. It is not a sustained-pressure or tenant-isolation test.

[`host-memory-budget.example.json`](https://github.com/lightseekorg/tokenspeed/blob/main/docs/configuration/host-memory-budget.example.json)
is a complete manifest that fits. It describes two nodes with a
prefill/decode pair, an independent fused job with two replicas, a storage
provider and a store master.

## Manifest

The shape, abridged to one of a rank's six required allocations:

```json
{
  "schema_version": 1,
  "nodes": [{"id": "node-a", "memory_bytes": 2199023255552, "system_reserved_bytes": 68719476736}],
  "containers": [{"id": "prefill-0", "node": "node-a", "memory_limit_bytes": 966367641600}],
  "jobs": [{
    "id": "chat-prefill",
    "replica_count": 1,
    "processes_per_replica": {"inference_rank": 4},
    "replicas": [{"id": "0", "processes": [{
      "id": "ranks",
      "role": "inference_rank",
      "count": 4,
      "placement": {"container": "prefill-0"},
      "allocations": [
        {"id": "l2", "kind": "l2", "sharing": "private",
         "reservation": {"basis": "measured", "bytes": 150010000000}}
      ]
    }]}]
  }]
}
```

Every field is required. The only exception is the `note` string an
allocation may carry. Unknown fields, duplicate JSON keys, duplicate ids,
and references to unknown nodes, containers or allocations are errors.
Sizes and counts must be JSON integers. Floats, booleans, strings, negative
values, and zero limits or counts are refused.

- **Nodes.** `memory_bytes` is the node's Host memory. `system_reserved_bytes`
  is memory the manifest's processes cannot use: the OS, page tables,
  daemons and other tenants.
- **Containers.** `memory_limit_bytes` is the memory cgroup limit of the
  container, pod or Slurm step. This manifest represents one level of
  container limits. With nested cgroups, also check the combined allocations
  of all siblings against each ancestor's limit separately; choosing each
  process's tightest limit alone does not check those combined totals.
- **Jobs and replicas.** Each independently launched deployment is one job,
  and each replica lists its processes explicitly.
  `replica_count` must equal the number of replicas listed, and each
  replica's processes must add up to `processes_per_replica` for every
  role.
- **Processes.** One entry stands for `count` identical processes, such as
  the TP ranks of a replica. `placement` names either a container or a
  node, never both.

### Roles and allocations

Each role must declare each of its required kinds exactly once. If a process
has none of a kind, declare it with zero bytes rather than omitting it. Any
role may also add `other` allocations.

| Role | Required kinds |
| --- | --- |
| `inference_rank` | `l2`, `snapshot_pool`, `slot_state`, `l3_client_segment`, `l3_client_staging`, `process_headroom` |
| `storage_provider` | `provider_segment`, `process_headroom` |
| `auxiliary` (store master, gateway, ...) | `process_headroom` |

Where each value comes from:

- `l2` and `snapshot_pool` are the buffers each rank pins. The startup line
  `Host pinned memory budget: ... L2 reservation=... snapshot reservation=...`
  prints them in GB with two decimals; round those values up. The snapshot
  reservation already includes the slot-state arena, so declare
  `slot_state` as `included_in` it. The prefill role never retracts, so its
  snapshot pool and slot state are zero.
- `l3_client_segment` is the storage one rank mounts:
  `global_segment_size` divided by attention TP × PP. Clients configured with
  `global_segment_size: 0` mount none.
- `l3_client_staging` is the store client's local transfer buffer. It is
  currently 16 MiB per rank with L3 enabled and zero without it.
- `provider_segment` is the segment a storage provider mounts.
- `process_headroom` is the rest of the process's memory: Python, the
  device runtime, model staging and allocator caches. The rank startup check
  leaves 10 GiB per rank; measure peak RSS beyond the declared buffers to
  refine it.

### Reservations

A pinned buffer (`l2`, `snapshot_pool`, `slot_state`) can reserve more than
its tensor bytes, so the nominal size is never accepted. Declare each one in
one of these ways:

- `{"basis": "measured", "bytes": N}`: the bytes the allocator reserved.
- `{"basis": "pinned_rounding", "requested_bytes": N, "round_threshold_bytes": T}`:
  the runtime's own rounding contract. A request of at most `T` bytes rounds
  up to a power of two; a larger one reserves its requested size. The
  launcher's default `pinned_max_round_threshold_mb:1024` gives
  `T = 1073741824`. Use `null` for an allocator without a threshold, which
  rounds every request.
- `{"basis": "included_in", "allocation": "<id>"}`: these bytes are already
  counted in another private, measured or declared allocation of the same
  process.

Other kinds take `measured` or `declared` bytes, or `included_in`.

### Sharing and accounting

`sharing` is `private` or `shared`. A private allocation counts once for
each of its entry's `count` processes. A shared allocation counts once
however many processes declare it, for example a shared-memory segment
that every rank of a container maps. Its id is global, and every
declaration of it must agree on kind, reservation and memory domain. The
memory domain is the same container, or the same node outside any
container. An id that is shared in one place and private in another is an
error. `l2`, `snapshot_pool`, `slot_state`, `process_headroom`,
`l3_client_segment` and `l3_client_staging` belong to one process and cannot
be shared. Sharing the remote store's objects does not make the clients'
local segment or staging buffers shared allocations.

Each allocation is charged once to its container, if it has one, and once to
its node. A container's total is the allocations placed in it. A node's
total is its `system_reserved_bytes` plus every allocation placed on it,
inside a container or not. Container limits never add to the node's total.
When a node's container limits add up to more than its memory minus
`system_reserved_bytes`, the report adds a warning. The status still depends
only on declared allocations.
Totals equal to a limit fit.

## What the estimate leaves out

- Live usage, page cache, fragmentation, and growth after startup.
- Memory a process uses without declaring it, such as the store SDK's
  transfer-engine and RDMA registration overhead and ordinary allocator
  caches. Budget this as `process_headroom` or `other`.
- How cgroups charge shared memory. A shared allocation is charged to the
  single domain its declarations agree on.
- Nested cgroup limits: the report does not aggregate child containers
  against an ancestor container's limit.
- Runtime admission. The runtime's per-rank startup check and the L3
  operation and Host-reference credits are unchanged. They bound one rank's
  buffers and L3 ownership, not the deployment's RSS.
