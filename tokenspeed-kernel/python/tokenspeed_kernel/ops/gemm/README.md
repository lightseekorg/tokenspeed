# Dense GEMM dispatch

The H20 BF16 decode candidate routes a bounded set of single-row projections
through the existing `decode_gemv` entry. It uses the retained CuTe dot-product
and skinny GEMM drivers; it does not change their arithmetic or the generic
FlashInfer dispatch. The full H20 eligibility check admits a call-local candidate
at priority 14; existing matching registry kernels at the same or a higher
priority retain precedence. Eligibility is part of the private selection cache
key. The dispatcher is not globally registered: rejected calls keep the original
registry selection, including other SM90 products.
The table limits the candidate dispatch surface. GPU numerical validation and
end-to-end performance measurements remain pending.

| M | N | K | Candidate backend |
|---|---|---|---|
| 1 | 2624 | 6144 | LL BF16 dot product |
| 1 | 4096 | 2048 | Skinny GEMM |
| 1 | 160 | 6144 | LL BF16 dot product |
| 1 | 512 | 2048 | LL BF16 dot product |
| 1 | 512 | 6144 | LL BF16 dot product |
| 1 | 128 | 6144 | LL BF16 dot product |
| 1 | 1152 | 6144 | LL BF16 dot product |

Only devices named `NVIDIA H20` or `H20` with compute capability 9.0 are eligible.
The operands must be two-dimensional, contiguous BF16 tensors on the same CUDA
device, which must also be the current CUDA device, with compact row-major strides
and 32-byte-aligned pointers. An output destination must have the exact
shape, dtype and device, be aligned and contiguous for the candidate, and use
storage separate from both operands. Disjoint views of one allocation are also
excluded from this candidate. Other shapes, row counts, layouts and devices keep
the original generic selection and its output contract. Direct calls to the
private helper reject invalid output shape/dtype/device or shared operand
storage; the generic caller must continue to avoid aliasing operands.

The skinny shape uses an explicit configuration: one row, 64 threads, two outputs
per block, K unroll two and vector width eight. The LL driver retains its existing
single-row configuration. Each PDL policy has its own driver owners. A successful
eager startup call warms a `(device, M, N, K, PDL)` key; capture and serving refuse
cold keys instead of triggering a compile. Eager and captured execution use the
same kernel entry. The candidate leaves batch-invariant numerics and explicit
kernel overrides in the runtime's existing priority order.

CPU contracts cover selection, fallback and the no-compile gate. GPU validation
must separately establish finite outputs and numerical error against an FP32
reference, including caller-owned output buffers, then measure warm eager and
CUDA-graph execution with the same inputs. Different reductions are not assumed
bitwise equivalent. End-to-end accuracy and speed require separate validation.
