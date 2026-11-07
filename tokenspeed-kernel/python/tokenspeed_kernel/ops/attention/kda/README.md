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
