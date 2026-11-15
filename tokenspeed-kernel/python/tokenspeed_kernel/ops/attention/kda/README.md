# KDA kernels

## RecoverSSM verify (`kda_recoverssm_verify`)

Speculative verify reads each request's checkpoint, runs the candidate window
through the KDA recurrence, and writes the outputs plus the recovery records
(FP32 corrections, raw key and gate, position stamps) that commit later
consumes. The checkpoint itself is never written. Two kernels are registered:

| Kernel | Platforms | Admits |
|---|---|---|
| `cute_dsl_kda_recoverssm_verify` | NVIDIA SM100 to SM103 | BF16 activations and key/gate records, FP32 state and corrections, key dim a multiple of 32 up to 128, value dim a multiple of 32, windows up to 16 |
| `vllm_triton_kda_recoverssm_verify` | NVIDIA | the reference; FP16 or BF16, any state dtype |

The split verify producer selects between them through the registry, so
shapes the CuTe DSL kernel does not admit fall back to Triton; layouts it
cannot address (token rows or state not 16-byte aligned, state tiles that are
not dense, strided indices or record channels, a strided one-element `A_log`)
and positive lower bounds, which would grow rather than decay, also run the
Triton kernel. Both share input validation and write
the same key/gate records and stamps.

The Triton kernel replays the recurrence token by token, one warp per value
slice. The CuTe DSL kernel solves the window in chunked form instead, one CTA
per (value tile, request, head):

1. Bulk copies bring the window's q, k, gate, and v rows and the CTA's
   `[rows, K]` checkpoint tile into shared memory once.
2. One thread per key forms the cumulative gate `G_t` and the decayed keys and
   queries `k_t e^G_t`, `q_t e^G_t`.
3. The checkpoint tile times those columns gives every token's view of the
   initial state, on CUDA cores in FP32.
4. One thread per value channel solves the window's lower-triangular system
   for the corrections and outputs.

The pairwise decayed key products of the triangular system come from the same
product as extra tile rows `k_j e^-G_j` and `q_j e^-G_j` when the gate is
bounded and `spec_query_len * |lower_bound| <= 48`, so `e^-G` stays a normal
FP32 number. Otherwise, they are summed pairwise with `e^(G_t - G_j)`, where
`G_t - G_j` adds up the gates in between rather than subtracting prefix sums,
so an early gate that dwarfs later ones cannot swallow them; the prefix sums
used for `e^G` are compensated. The exp, sigmoid, and normalization
approximations match the Triton kernel's. The products use unnormalized keys
and queries, which assumes activations far below the FP32 range (model
activations are), and `e^G` is formed before the checkpoint product, so a
checkpoint decayed past `e^-87` contributes exactly zero instead of a value
below `|state| * 1.2e-38`.

The value tile is 32, 64, or 128 rows, chosen from the number of
(request, head) pairs to balance occupancy against repeating the token-side
work in every tile; windows above 8 tokens use 32 rows, and the tile halves
until it divides the value dim.

## Fused one-token verify (`kda_fused_paged_verify`)

Decode without speculation verifies one-token windows.
`cute_dsl_kda_fused_paged_verify_step` (NVIDIA SM100 to SM103) runs that
verify together with its two producers, the gate projection `f_a @ f_b.T` and
the convolution, in one launch; `cute_dsl_kda_fused_paged_verify_window` does
the same for windows of 2 to 16 tokens (below). Results
match the split producers followed by `kda_recoverssm_verify`: the same
outputs, correction and key/gate records, stamps, and retained
pre-convolution rows, so commit is unchanged. The gate projection sums in its
own order, so a raw gate may differ from cuBLAS's by one BF16 step.

`kda_fused_paged_verify_fuses_producers` tells the caller whether the selected
kernel runs the producers itself for a window length; the kernels declare
`spec_query_len_max` 1 and 16, so callers that state no window length never
select them; the window kernel also declares `batch_size_max` 16, past which
the split producers' GEMM outruns its CUDA-core gate projection. They take
BF16 activations. Inputs it cannot address (rows or state not 16-byte
aligned, head widths or gate ranks other than 128, positive lower bounds) run
the split producers inside the same call.

One CTA of four warps covers one (request, head). It bulk-copies the head's
checkpoint into shared memory while thread t convolves channel t of q, k and
v. For the gate projection and the checkpoint product, warp w takes 32 keys or
32 checkpoint rows and lane l elements [4l, 4l + 4) of each, so every
warp-wide load reads one contiguous row, and a transposing butterfly leaves
each row's total in the lane of the matching thread. Weights load (or are
prefetched into L2) before the programmatic dependency wait, and so do the
checkpoint copy and the convolution history, which the forward never writes
(acceptance commits them after it); the projection's rows load after it.

### Multi-token windows

The window kernel keeps the RecoverSSM verify's grid, value tiles and window
math, and makes the C = 128 / VT tiles of a head one cluster that shares the
producers. CTA r convolves q and k channels [r VT, (r + 1) VT) and its own v
tile, projects the gate for keys [r VT, (r + 1) VT) (the one-token kernel's
lane mapping, two tokens per pass), and stores the BF16 rows, rounded as the
split producers store them, into the token rows of every CTA of the cluster
over distributed shared memory. The rows travel as asynchronous BF16-pair
stores that the receiving CTA's mbarrier counts, so a CTA waits only until
its own rows have landed; the window math then runs as in
`kda_recoverssm_verify`. Before the programmatic dependency wait the CTA
bulk-copies its checkpoint tile and f_b rows (into tile rows that phase 1
fills only after the gate projection) into shared memory and issues the
loads of the convolution taps and history and the gate and norm parameters,
first using any of them after the wait; every read of the projection rows
is issued right after it. A grid of one to two CTAs per SM claims enough
shared memory per CTA that two at most share an SM; left alone, the
scheduler packs it onto fewer SMs.

### Output norm

A KDA layer feeds its o-projection the gated RMSNorm of the attention output,
`rmsnorm(out) * w * sigmoid(gate)` per head. `try_kda_fused_paged_verify`
takes it as `output_norm` (`KdaOutputNorm`: the gate view, the weight, eps)
and then returns the normed `[tokens, heads * head_dim]` rows. The one-token
step applies it before its store: the thread of each channel holds the BF16
output, a warp sum and four shared-memory partials give the head's sum of
squares, and the rest is the norm kernel's math in its order. The window
kernel does the same per token, with each warp storing its partial to every
CTA of the cluster (counted on an mbarrier like the rows), so the four
partials add up in the same order. Only the order
of that sum differs from `rmsnorm_gated_sigmoid` on the stored output, so an
element may move by one BF16 step. Every other verify kernel, and the
backend's non-verify paths, run `rmsnorm_gated_sigmoid` after the attention
(`apply_kda_output_norm`).

## Batched commit (`kda_replay_commit`)

After sampling, one launch per state kind commits every layer's accepted
tokens: the recurrent states fold the accepted corrections and key/gate
records into the checkpoint, and the convolution histories take the accepted
window. Two kernels are registered for the recurrent half on NVIDIA:

| Kernel | Platforms | Admits |
|---|---|---|
| `cute_dsl_kda_batched_replay_commit` | NVIDIA SM100 to SM103 | BF16 records, FP32 state, 128-wide heads, windows up to 15 tokens (`spec_query_len_max`), a bounded gate |
| `triton_nvidia_kda_batched_replay_commit` | NVIDIA | FP16 or BF16, FP32 or BF16 state, any window |

The CuTe DSL kernel reproduces the Triton commit's CUDA-core path bit for bit:
`S <- fma(u_t, kn_t, S * d_t)` per token, with the normalized key `kn_t` and
decay `d_t` formed with the same instructions, the key's sum of squares in the
Triton kernel's one-warp lane order. Windows of 16 to 32 tokens, where the
Triton commit switches to a TF32x3 Tensor Core product, stay on Triton, and an
unbounded gate runs the Triton commit inside the same call. Both share input
validation and the convolution commit.

Where the Triton commit splits a head over 8 or 32 one-warp programs that each
rebuild the token operands, the CuTe DSL kernel runs 2 CTAs of 256 threads
per (layer, head, request), one per half of the value rows. Each thread issues
the loads of its 8 rows x 4 keys of the source state first, warp w forms
tokens w and w + 8 into shared memory, and after one barrier every thread
folds its values and stores them, so each state byte is read once and written
once (GB300, 69 layers x 12 heads, 8-token windows: 28 us per commit at 1
request and 179 us at 8, against 50 us and 228 us).
