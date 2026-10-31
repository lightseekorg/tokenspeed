# Communication kernels

## Paged-row all-gather

`cute_dsl_paged_row_all_gather` (GB300 only) all-gathers rows of a paged cache
plane with the row pack fused into the transfer: each rank reads its own rows in
`fetch_slots` order and stores them straight into every rank's receive slot,
through the NVLink multicast address of one symmetric workspace (`multicast`
mode, two CTAs per window) or with one store per peer (`peer` mode, 24 CTAs).
The rows a reader addresses are byte-identical to an all-gather of the packed
rows. The KVP prefix-history reads (`PrefixHistoryGather` in the runtime) use
it in place of a copy plus an NCCL all-gather.

### Layout

One symmetric allocation per group: `[slot 0 | slot 1 | flags]`. A slot holds
`degree * cap_bytes`, so the capacity is a byte count. Epoch `e` writes slot
`e & 1`; rank `p`'s rows start at row `p * send_rows`, and rows
`[owned, send_rows)` of each rank are never written. The flags block holds
per-peer credit, arrival and header words and a CTA counter; the kernel header
in `_cute_dsl/paged_row_all_gather.py` describes the protocol.

### Setup

`create_paged_row_all_gather_state` is collective. Every vote, and the
symmetric-memory rendezvous, runs on a CPU (gloo) group over the gather ranks:
the CUDA symmetric-memory backend (the default) exchanges handles through the
group's store, so setup starts no NCCL work on the device communicators. The
caller's own verdict (`eligible`), the platform, the group size, the fabric
(from the map `gather_fabric_map` gathers, as the server's distributed
initializer does; without it the group keeps its all-gather), the kernel
module's import, allocation, mapping and compile are all voted on, so every
rank either gets a state or `None`. The state then runs one empty window
(epoch 1), which loads the module and checks every peer's flag words through
its unicast mapping, before it is returned. A failure inside the rendezvous or
that first window is fatal rather than voted; the first window waits at most
30 s, so a rank that fails before launching it makes the others trap within
that bound instead of after `timeout_ns`.

### Caller contract

- Every rank launches the same windows in the same order, with the same
  `send_rows` and header per window. The state numbers them: epoch 1 is its
  first window, and each launch returns its epoch.
- A launch makes `state.stream` wait for the current stream's work so far, so
  the cache rows and the reads of the slot's previous epoch (`e - 2`) are
  enqueued there first. Readers wait for the event the launch returns, which
  the launch of `e + 2` records again. At most one epoch per slot is in use:
  one reader at a time, at most two windows not yet released, and one
  consumer stream.
- Launches happen in eager mode only; a launch under graph capture raises.
- The plane is `[rows, *row]` of any dtype, contiguous, `row_bytes` per row,
  from a 16-byte aligned base; it and `fetch_slots` stay allocated and
  unwritten until the window's event completes.

### Failure

Every wait is bounded by `timeout_ns`. A peer that does not reach the window
in time, a credit or arrival ahead of the epoch (the ranks launched different
windows), or a header mismatch claims the pinned error record
(`state.read_error()`), prints, and traps: the group is unusable afterwards,
as after an NCCL timeout. A rank stuck outside the kernel is not detected by
the kernel itself; the peers' kernels time out instead.

### Concurrency

Do not overlap a window with kernels that need (nearly) every SM resident and
wait across ranks inside the kernel (MegaMoE, fused DP projections, DeepEP
low-latency): each side can occupy what the other waits on. The runtime keeps
the NCCL all-gather when one of them is configured.
Cross-rank kernels whose CTAs pair up by index across ranks -- NCCL, one-shot
or two-shot multimem and Lamport all-reduces, including PyTorch's multimem
all-reduce -- run alongside: they rely on CTAs being dispatched by index, as
NCCL's ring kernels already do, not on a CUDA guarantee.

### Memory order across synchronization domains

The state's stream runs the kernel in the CUDA remote memory-synchronization
domain, as NCCL runs its kernels, while the caller's kernels run in the
default domain. "Writes and fences are tagged with the ID, and a fence will
only order writes matching the fence's domain", so the compute kernels' fences
and kernel-end flushes stop waiting for this kernel's NVLink stores; in
exchange, "ordering or synchronization between distinct domains on the same
GPU requires system-scope fencing" (CUDA Programming Guide, memory
synchronization domains). Each hand-off:

- Rows to the consumer. A writer's payload stores (`multimem.st.relaxed.sys`
  or `st.global`) precede, through the CTA barrier and the `acq_rel.sys`
  counter atomic, the last CTA's `st.release.sys` of its arrival: base
  causality order is transitive, which makes the release cumulative over
  every payload store (PTX ISA memory consistency model). Each reader's
  kernel `ld.acquire.sys`'s every arrival before it ends. Stores, fences, flags
  and acquires all run inside these kernels, in the remote domain on every
  GPU, at system scope, so the rows are at system scope before the reader's
  kernel ends. The consumer is a later kernel in the default domain, ordered
  by an event; kernel end, event record and event wait synchronize and
  preserve proxies, which orders the multicast alias of the rows before the
  consumer's unicast reads. The kernels themselves never read the rows.
- The consumer of epoch `e - 2` to the stores of epoch `e` into the same slot.
  The reads are done when the consumer kernel ends; an event orders that
  before this kernel's `st.release.sys` of its credit, which every peer
  acquires (`sys`) before its first store into the slot.
- Cache rows and `fetch_slots` written by the caller's kernels to this
  kernel's loads: ordered by the ready event, in the other direction.
- The credit, arrival and header words are zeroed at setup (ordered by a
  stream synchronize and the allocation vote) and afterwards accessed only by
  these kernels, at system scope. The CTA counter's reset is gpu-scope, enough
  for a local counter whose next use follows the kernel's completion.

The event hand-offs between the two domains rest on the programming guide's
implicit fences "to implement synchronizes-with relationships at task
boundaries", on the PTX ISA's statement of the kernel and event
synchronization (which names no domain), and on NCCL's remote-domain kernels
working "with no code changes needed in other components", which rely on the
same hand-offs for their buffers. Neither document says at which scope a
task-boundary fence orders writes across domains: that link is not documented,
and the evidence for it is empirical (byte-exact gathers over long runs with
concurrent compute and collectives).
