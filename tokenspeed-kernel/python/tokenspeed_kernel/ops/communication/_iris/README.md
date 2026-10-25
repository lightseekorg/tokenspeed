# Iris communication

Collectives share one process-wide Iris context and symmetric heap. Each
prepared operation retains its buffers; execution never searches for a suitable
state. Allocate in the same order on participating ranks before graph capture;
prepared capacities and addresses stay fixed.

The first preparation sizes the shared heap. Later states allocate from that
same fixed heap; Iris's allocator enforces capacity.

- **State:** the persistent buffers and references needed by an operation.
- **Epoch:** a GPU counter that distinguishes successive uses of synchronization
  flags, including graph replays. Epochs are 32-bit and wrap during long
  serving runs: compare them modulo 2**32 (`(seen - epoch) < 0` as int32), and
  derive slot indices from the unsigned value with a power-of-two slot count.
- **Scratch:** reusable buffers holding intermediate computation results.
- **Borrowed tensor:** a view, without a copy, into storage owned by the state.
  Another operation can overwrite its contents even while the view is alive.

Each implementation owns the synchronization and scratch it needs. Sharing is
explicit: K3 fusions reuse producer storage and pull scratch, with producers
finished and consumers ordered on one stream. Sharded activations are borrowed;
ordinary producer-direct reductions return results in independent storage.

`IrisBackend` retains generic operations by group and dtype. Models prepare
additional fusion state before cache sizing and bind it to their callers.
Generic operations can also initialize during warmup. Calls select among
prepared variants using the existing shape and topology policies. Different
staging geometries have separate synchronization buffers because their workgroup
IDs cover different element ranges. Triton RSAG remains independent.

Model preparation, dispatch, and fusions live in `kimi_k3/` and
`muse_spark/`. General collectives live in `all_reduce.py`,
`reduce_scatter.py`, `all_gather.py`, and `broadcast.py`. The all-gather module
also owns the width-parameterized RMSNorm/MXFP4 fusion, with a Lamport variant
that shares the BF16 reduction's mailbox. Exchanges with model-specific head
layouts, route packets, or normalization semantics stay in the model package,
along with dispatch and capacities.

Lamport reductions use data as the ready signal: peers push BF16 into
mailboxes holding an empty marker (negative zero, so payloads send it as zero),
and the receiver polls until no marker remains, restores the markers, and
advances a three-stage counter. Fusions sharing the one-shot mailbox keep its
per-512-element stage counters; the TP8 two-shot owns its mailboxes and
counters.

Muse Spark prepares its collectives for TP4/EP4 or TP8/EP8 on gfx950. Both
topologies keep the same global token cutoffs for Lamport, push, pull, and
owned-row paths. Owned rows use ceil(global rows / world size), with padded
rows masked; QKVG exchange derives each rank's head widths from the group size.

Configure the allocator and optional heap capacity on every rank before the
first preparation:

```python
from tokenspeed_kernel.ops.communication.iris import configure_iris

configure_iris(heap_size=None, allocator_type="torch")
```

`heap_size=None` sizes the heap automatically. `allocator_type=None` uses Iris's
default (`torch` in the pinned version). Names pass through to Iris;
`IRIS_ALLOCATOR` takes precedence. A live context cannot be reconfigured.
