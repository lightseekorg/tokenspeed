# Mamba2 selective state space scans

Four ops cover the Mamba2 SSD mixer. All take the checkpoint's `A_log` and
compute the decay rate `A = -exp(A_log)` inside the kernel, so no per-call
elementwise launch precedes them. `mamba2_chunk_scan` handles prefill,
`mamba2_state_update` one token per request, `mamba2_verify_scan` a speculative
verify window and `mamba2_replay_commit` the accepted part of that window. All
use the recurrence in the package docstring and keep states as
`[heads, head_dim, d_state]` with `d_state` last, the layout of the runtime's
recurrent-state pool, so no transpose is needed at the cache boundary.

## Prefill scan

The scan takes a packed varlen batch plus one initial state per sequence and
returns one final state per sequence. The final states have the initial states'
dtype, which is fp32 for Nemotron-H. A sequence without history starts from a
zero row. A chunked prefill resumes by passing the state the previous part
returned. The two-part result matches a single scan to within bf16 rounding.

The scan splits the token axis into logical chunks. A logical chunk never
crosses a sequence boundary or a multiple of `chunk_size`.
`build_mamba2_chunk_metadata` derives these chunks on the host from
`cu_seqlens`, which callers already hold on the host, and uploads them without
blocking. Every sequence must hold at least one token; the builder rejects an
empty one.

It is the chunked SSD algorithm in four Triton launches:

1. `_ssd_chunk_state_kernel`, per chunk and head: the step sizes, their
   in-chunk cumulative log decay, and the chunk's own state contribution
   `sum_t exp(decay_end - decay_t) * dt_t * x_t (x) B_t`. The decay-weighted
   `B` is split into two BF16 halves, so the contribution keeps near-FP32
   precision on BF16 tensor cores.
2. `_ssd_state_passing_kernel`, per sequence and head: carries the FP32 state
   across the sequence's chunks from its initial state, records each chunk's
   incoming state in BF16, and returns the final state.
3. `_ssd_chunk_cb_kernel`, per chunk and group: `C_t . B_s` for every token
   pair, shared by the heads of the group.
4. `_ssd_chunk_scan_kernel`, per 64-row block of a chunk and head: the causal
   in-chunk output from those pair products plus the incoming state's
   contribution, and `D * x`.

`chunk_size` must be a power of two of at least 16, and one launch holds at
most 65535 chunks, the CUDA cap on the grid axis the chunks occupy; decode and
verify likewise hold at most 65535 requests per launch. Final states match an
FP64 token-by-token recurrence to about `3e-6` relative error. On GB300 at
Nemotron-3 Super geometry the scan takes 465 us for 8192 tokens per layer.

## Operand contract

Every op validates its operands before launching, since the kernels index them
directly. `x`, `dt`, `B` and `C` must be contiguous in their last dimension and
`out` contiguous. States, both pool slots and initial states, must be dense
within a slot, the layout the replay commit also indexes, and pool slots must
not overlap. The kernels widen only the leading token, request or slot index to
64 bits, so the rest of one row of an operand must span fewer than 2^31
elements. The tensors an op writes, `out` and the state pool, must not share
storage with any other operand: scan programs read rows that other programs
write. Activations are BF16: the prefill scan feeds FP32 intermediates to
tensor cores in the activation dtype, where FP16 would overflow.

## State update

The update reads each request's state from `state_indices` and writes the new
state to `dst_state_indices`. Separate source and destination slots let
speculative verify keep the committed state intact. Rows whose index equals
`null_slot` are padding: their state is neither read nor written. Decode is a
one-token verify with a destination slot, so it runs the verify kernel.

## Speculative verify and replay

`mamba2_verify_scan` steps each request through its `T` verify tokens from one
read slot and optionally writes the state after every token to a `[batch, T]`
destination table. Without the table it leaves the pool untouched: the caller
keeps the verify inputs and `mamba2_replay_commit` later rebuilds only the
accepted state, which avoids staging `T` full states per layer (4 MiB each for
Nemotron-3 Super). The commit reads the recurrent replay payload
`[K | V | a | b]` shared with GDN, holding `B`, `x` and the raw `dt`; the SSD
recurrence has no `b` and ignores that slot. It replays every layer of the
round in one launch through a table of per-layer pool addresses.

Decode runs the verify kernel, and replay repeats its arithmetic step for
step. Both round the state through the pool dtype after every token, as
decode's reload does, and the destination write is a runtime branch, so
decode, verify and replay produce bit-identical states and verify outputs
match decode outputs bit for bit, for fp32 and bf16 state pools alike.

The verify kernel is bound by instruction issue, and at `d_state` 128 half of
each token's instructions went to the output's per-row sums across the warp.
For a 16-byte-aligned fp32 pool, where each lane loads 4 consecutive columns
of every row, lanes instead trade halves of their rows: 9 shuffles per 8 rows
rather than 40, pairing lanes as the warp reduction does, so the outputs keep
their bits. Other pools keep the warp reduction.

With `parent_indices` the window is a draft tree: token `t` continues from the
state after its parent token rather than token `t - 1`. At such a branch a
staged verify reloads the parent's destination row; without one the kernel
replays the parent's ancestors, which precede it in the window, over the
read state with the same per-token rounding. A few elementwise steps from
cached inputs cost less than staging a full state per token and layer (4 MiB on
Nemotron-3 Super) and reading it back. Each token's state equals a chain verify
of its root path bit for bit. Its output matches to rounding, bit for bit at
Nemotron's geometry with an fp32 state pool: the tree build contracts
multiply-adds differently, so outputs can differ in their last fp32 bits
(several bf16 ulps near zero) at the chain's accuracy against an fp64 reference.

## Tests

`test/nvidia/ops/attention/test_mamba2.py` compares the ops with a sequential
fp64 recurrence. It covers single-token and multi-chunk sequences, nonzero
initial states, resuming a split scan, padded update rows, and decode steps
that continue a prefill, checks verify and replay bit for bit against
consecutive decode updates, checks tree windows against chain verifies of each
token's root path, checks verify outputs against the recurrence for fp32,
misaligned and bf16 pools, and guards every kernel against recompiling when
the batch shape changes.
