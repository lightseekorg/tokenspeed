# Mooncake offload RPC client

This optional bridge talks to the deployed Mooncake master and SSD providers.
It does not replace Mooncake or add a storage service. The client exposes only
bounded metadata queries, SSD preparation, and source-buffer release.

Build explicitly against the deployed SDK's yalantinglibs headers, a C++20
compiler, nlohmann_json and OpenSSL:

```sh
cmake -S . -B build -DMOONCAKE_SOURCE_DIR=/path/to/deployed/mooncake
cmake --build build
```

`wire.h` is a wire-only declaration of the deployed protocol. It must not be
linked with a second definition of Mooncake's implementation classes. The
qualified RPC method names and reflected fields are protocol contracts. A
change in the provider SDK requires protocol compatibility validation, not a
silent fallback to another object representation. Library ABI version 4 is
checked by `ts_mooncake::offload::Client` before any call.

The master query returns complete SSD replicas only. It asks about explicitly
named keys, never enumerates or publishes the L3 inventory. Mooncake's normal
metadata lookup may grant its ordinary metadata lease; this does not allocate
an inference L1/L2 destination. Preparation uses the tenant-scoped key format
and returns source addresses, a transfer session, and a native buffer lease.
The client checks counts, sizes, address extents and remaining lifetime before
a read can begin. Deadlines cover both connection establishment and the RPC.

All calls and lease destruction belong on bounded I/O workers. An unused
prepared lease releases its buffers on drop. Beginning a read transfers it to
`ReadingLease`; dropping this without native retirement deliberately sends no
early release. The provider eventually reclaims that source using its TTL.
The caller must independently retain/quarantine destination Device pages until
writes have stopped. A retired read must obtain a successful release reply
before publishing its prefix; the reply is false if the source record is gone.
Timeout/cancellation is never retirement. Each prepared lease retains its original provider connection and exposes an
opaque local handle. Release never reconnects using a provider batch id: a
restarted provider may have reused that id for another reader. Cache-generation
checks and owner agreement remain the integration layer's responsibility. The bridge alone does not enable direct restores in inference.

The native integration test requires an explicit fixture with a task-owned
SSD object. It checks discovery, missing keys, release acknowledgment,
cancellation before a read, and uncertain-retirement behavior. It never
removes an object or mutates model weights.

## SSD object generations

The bridge ABI is 4. Metadata queries retain the exact SSD storage generation
returned by the Master, and preparation passes that generation through to the
Provider's physical object key. A zero generation preserves the legacy layout.
An old bridge is rejected at startup; a stale generation must fail the read,
never fall back to an unversioned key. This supports generation-aware Mooncake
SSD metadata without changing the server RPCs or on-disk format.

## Master discovery

For an `etcd://` master entry, also build the discovery companion against the
same Mooncake Store SDK and library used by inference:

```sh
cmake -S . -B build -DMOONCAKE_SOURCE_DIR=/path/to/deployed/mooncake \
  -DMOONCAKE_STORE_LIBRARY=/path/to/libmooncake_store.so
cmake --build build
```

Install both bridge libraries together. `TOKENSPEED_MOONCAKE_DISCOVERY_LIBRARY`
can override the companion path. Discovery uses Mooncake's ready-view and
`MC_STORE_CLUSTER_ID` semantics, including its certificate file environment
variables. It runs on the storage I/O thread before a metadata query, never in
the scheduler. Static Master addresses do not load the discovery companion.
The companion is separate from the wire-only RPC client so SDK implementation
classes cannot collide with RPC protocol declarations. Provider preparation
and release keep their original endpoint and connection across Master changes.
