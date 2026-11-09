# Iris communication

Collectives share one process-wide Iris context and symmetric heap. Each
prepared operation retains its buffers; execution never searches for a suitable
state. Allocate in the same order on participating ranks before graph capture;
prepared capacities and addresses stay fixed.

The first preparation sizes the shared heap. Later states allocate from that
same heap; the `torch` allocator enforces its capacity, while `vmem_chunked`
rounds allocations up to powers of two and grows the heap on demand.

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

A model can also install its prepared `ProducerAllReduce` as a group's ordinary
all-reduce. The backend then reduces caller-owned BF16 tensors in place through
`reduce_ordinary`: Lamport windows directly, the TP4 push window with a copy
back, and other payloads up to a measured per-group-size limit through the
producer buffer and the generic pull. Past the limit they fall back to RCCL.
Results never borrow state storage, but ordinary reductions write the producer
buffer and pull scratch, so no borrowed producer-buffer or pull-scratch view,
nor the ready flags guarding it, may be live across an ordinary reduction on
the group.

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
The TP8 two-shot's phases are device functions taking the stage, so a fused
kernel that produces the partials itself reuses its mailbox layout and
reduction. Muse's sliding attention tail does this: in one launch, attention
splits compute the rows, o_proj senders push straight into the
reduce-scatter slots, and other workgroups reduce, all-gather and finish the
attention boundary. The tail uses its own region, with each workgroup's epoch
as the stage.

`residual_rmsnorm.py` reduces BF16 rows fused with the residual add and
RMSNorm, for callers that prepare it per group and width (draft layers). Each
row has an owner that receives the partials, reduces, and pushes the reduced
row to every rank. It owns its mailboxes and needs no stage counters: an owner
restores its scatter markers before it publishes the reduced row, and no rank
starts its next launch before it has every reduced row, so row counts may
change between launches.

Muse Spark prepares its collectives for TP4/EP4 or TP8/EP8 on gfx950; its own
ordinary all-reduces use the same `reduce_ordinary` policy. Its pre-expert
gather can also route the rows: the Lamport gather, RMSNorm and MXFP4 programs
share one launch with independent per-row top-k programs. Both
topologies keep the same global token cutoffs for Lamport, push, pull, and
owned-row paths. Owned rows use ceil(global rows / world size), with padded
rows masked; QKVG exchange derives each rank's head widths from the group size.
With expert TP8 (EP1), batches whose routes outgrow the Lamport two-shot
all-reduce (44 rows) and stay below the owned-row cutoff reduce-scatter each
token's raw routes onto its owner through the owned-row route staging,
normalize them there, and gather the 8x smaller sums through the Lamport
one-shot mailbox. With Muse's output-column-split down copy, owned-row batches
instead gather the sorted MXFP4 intermediate slices, and ranks add their FP32
partial sums of squares in rank order for the routed RMSNorm. Per-program flags
pair the same tokens on every rank in each phase; each kernel exits after
peers finish reading its symmetric buffers.

Batches below the reduce-scatter range complete their routes column-owned in
one launch, with its own mailboxes and per-(token, owner) stages: rank q sums
the partials of columns [512q, 512q + 512) of every route and publishes them
with their partial sums of squares, then every rank normalizes, weights,
and sums all columns.

Configure the allocator and optional heap capacity on every rank before the
first preparation:

```python
from tokenspeed_kernel.ops.communication.iris import configure_iris

configure_iris(heap_size=None, allocator_type="torch")
```

`heap_size=None` sizes the heap automatically. `allocator_type=None` picks
`vmem_chunked` when the world spans hosts and Iris's `torch` default otherwise.
Names pass through to Iris; `IRIS_ALLOCATOR` takes precedence. A live context
cannot be reconfigured.

The heap spans the world. `discover_iris_world()`, called once during
distributed initialization, decides whether one heap can map every rank: a
world on one host can, and across hosts only gfx1250 with every GPU in one
fabric domain. Hosts are told apart by boot id, and the hostnames Iris uses
must group ranks the same way. Context creation refuses a world it cannot map.

The heap is built on the caller's current GPU, not on Iris's rank % visible
GPUs, so ranks may start past GPU 0.
