# Experimental CUDA Lamport A2A

`cuda_lamport_a2a` is an opt-in TP4 BF16, intra-node NVLink experiment. It does
not change model dispatch or replace an existing backend. It exchanges channel
shards directly between `[M, K]` and `[4*M, K/4]`, including the inverse mapping,
without a separate pack or output-restoration kernel.

## Protocol

The design follows the direct-push and rotating-scratch ideas of the existing
TRT-LLM one-shot AllGather/ReduceScatter, but does not copy their sentinel
protocol. Each aligned 64-bit packet holds 32 payload bits and a 32-bit
generation. The receiver polls local packets until their generation matches,
then writes the unmodified payload to its output. Signed zero, infinities,
subnormals and NaN payloads are preserved. No separate cross-GPU barrier kernel
is needed; waiting is still present inside the packet-polling loop.

Three scratch generations prevent a fast sender from overwriting a slow
receiver. Finishing an exchange requires data from every peer, so a sender
cannot advance three exchanges ahead of a reader. A local CTA-entry counter
ensures every CTA reads the current generation before it advances. The grid
must not exceed the SM count. Multiple independent packet reads are pipelined
for larger messages; one read per thread avoids that overhead for small ones.

CUDA C++ is used for this initial protocol experiment to express the exact
aligned volatile 64-bit transactions explicitly. FlashInfer provides the
optional JIT build/FFI utilities, not the A2A algorithm. The implementation is
independent of TRT-LLM's bindings; a CuTe DSL port remains a possible follow-up.

## Contract and limits

- The Python entry point lives in `cuda_lamport.py`. Both packet and chunk
  exchange require **exactly four GPUs per process group on one host**,
  not necessarily four GPUs in the entire job. Peer indexing and scratch
  layouts are specialized for four peers; other group sizes are rejected.
- Prepare `CudaLamportA2AState(group, max_rows, channels, device, blocks)` on all
  four peers before capture. This creates symmetric scratch and compiles the
  kernel. All peers must agree on the physical shape and direction of each call.
- Inputs are contiguous BF16 matrices; `K` is a positive multiple of eight.
  Uneven/empty logical owners must be padded to the same positive physical `M`.
- Serialize this communicator and its consumers on one CUDA stream. The result
  borrows persistent **local output**, valid until the next call. Inputs must
  not alias output or communication scratch. Retain the state while graphs
  reference it; synchronize every rank before releasing it.
- Eager and CUDA Graph execution use the same kernel and GPU-owned generation.
  Recreate the communicator before `2^32` calls: generation overflow traps
  rather than risking acceptance of stale packets. There is no recovery from
  a missing peer; collective participation is mandatory.
- Packet storage is twice the payload size; three generations cost `6*S`
  scratch plus `S` output bytes per GPU, excluding metadata. Payload expansion
  also increases link traffic. This is a **low-latency**, not a large-message
  bandwidth optimization. Only four NVLink-connected GB300 GPUs have been
  measured; no cross-node, other-dtype or full-model claim is made.
- There is no implicit NCCL fallback. Existing production backends are unchanged.

## Optional large-message chunk exchange

Enable this explicitly on **every peer**, before capturing any graph:

```python
state.prepare_chunk_exchange(threshold_bytes=8 * 2**20 + 1)
```

The existing `cuda_lamport_a2a(state, inputs, inverse)` entry point then chooses
packet exchange below the threshold and chunk exchange at/above it. Both use
the same layout contract and return the same local output buffer. Peers must
agree on physical shapes, direction and threshold. Chunk exchange additionally
requires channels divisible by 32. The original packet-only behavior remains
available by not preparing chunk exchange.

The chunk kernel moves raw payload with 128-bit loads/stores rather than
doubling each 32-bit payload word with a tag. Each CTA owns a striped portion
of every peer's payload. Every writing thread executes a system fence, then a
CTA barrier. Four lanes publish per-peer readiness with system-release stores
and wait on system-acquire loads; a second CTA barrier precedes vectorized
local reads/output restoration. Three generations protect scratch reuse.
This does **not** omit required synchronization or normalize special values.

The selected large-message launch uses 1024 threads/CTA to increase independent
memory operations. This occupies more GPU resources than the 256-thread packet
kernel; interaction with concurrent GEMMs has not been benchmarked.

Packet and chunk exchange have **separate scratch and generation counters**.
Sharing raw chunk payload with packet scratch could make arbitrary data appear
to be a valid packet tag after a size transition. Preparation adds `3*S`
payload scratch and `3*4*blocks*8` flag bytes per GPU to the original workspace,
so combined scratch plus output is approximately `10*S`. No allocation or
host synchronization is performed by the forward call or graph replay.

The recommended threshold keeps exactly 8 MiB on the tuned packet kernel and
uses chunk exchange above it. An explicitly supplied threshold still takes
precedence; supplying 8 MiB selects chunk exchange at exactly 8 MiB.
Choose the threshold for your hardware and workload.
Correctness tests exercise both directions, repeated transitions between
packet/chunk sizes, delayed peers, arbitrary payload bits and graph replay.

### Medium-message packet tuning

For 4 through 8 MiB inclusive, packet exchange uses 1024 threads/CTA, writes
self-owned data directly to output, rotates peer publication order, and polls
three remote owners together. Other packet sizes retain the original launch.
Both variants use the same packet layout and full 32-bit generation IDs;
switching sizes needs neither additional scratch nor a new barrier. This
increases GPU occupancy; concurrent compute performance is not established.

The medium kernel additionally specializes each TP4 rank at compile time to
remove dynamic peer indexing and self-owner branches in its unrolled loops.
Packet format, generation checks, workspace size, and dispatch thresholds are
unchanged; four rank variants increase compiled code size. GPU tests cover all
ranks, both directions, and transitions across the packet/chunk boundary.

## Validation and measurement methodology

From the repository root, with optional CUDA/FlashInfer dependencies installed:

```bash
python -m pytest -q \
  test/runtime/distributed/test_comm_ops.py::TestCommOps::test_all_to_all_single
```

The existing runtime test spawns its own workers (do not launch pytest with
torchrun). It retains the two-/four-GPU runtime cases and adds one four-GPU
custom-kernel case, without changing runtime backend selection. Only the
custom case skips missing NVIDIA GPUs, full peer access, or optional FlashInfer;
the original runtime cases retain their existing requirements.

Correctness compares integer views against NCCL to check every bit. Coverage
includes both directions, minimal/tail widths, changing shapes and payloads,
an empty logical owner, delayed peers, consumers inside captured graphs, ring
reuse, and crossing the signed-int32 generation boundary.

Benchmark timings exclude startup/JIT. After 20 warmups, each sample times
10 replays of a graph containing 100 exchanges using CUDA events, takes the
maximum rank time, and reports the median of five samples. Output restoration
is included. The benchmark is communication-only: it does not establish a
projection or model speedup. AG/RS are latency references, not interchangeable
A2A algorithms; compare their message-size definitions explicitly.
