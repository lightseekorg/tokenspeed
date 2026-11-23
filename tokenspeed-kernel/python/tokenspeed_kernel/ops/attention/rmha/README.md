# Relative attention output precision

The prefill, cached-extend and decode facades require `output_dtype` explicitly.
`None` preserves legacy output precision. `torch.float32` requires the
accumulation result in FP32 before any BF16 or FP16 output store. Casting a
rounded output to FP32 does not satisfy this contract.

All three operations can return natural-log LSE alongside the output.
Decode requires an explicit `return_lse` flag. Empty decode contexts return
zero output and negative-infinity LSE. Padded extend rows follow the same
contract through the shared runtime backend.

The `fp32_output` selection trait keeps these requests on implementations
that support the requested output precision. Portable Triton supports both
legacy and FP32 outputs. On Blackwell, separate `*_fp32` FA4 registrations
serve FP32 outputs from the tokenspeed-mha kernels, which store the FP32
accumulator directly: prefill, cached extend and decode, each with or
without a sliding window; Blackwell attention does not run on Triton. A
decode launch's KV split count depends only on the batch shape, so CUDA
graphs keep it, and is a runtime argument of the one compiled kernel. FP32
output has no rounded fallback: unsupported geometry raises instead. Gluon
registrations support legacy outputs only. The planner applies the same trait
as execution.

`rel_mha_gate_sink` applies the gated-output and attention-sink epilogue to an
FP32 output in one kernel: `(out * sigmoid(gate)) * sigmoid(lse - sink)`,
rounded once to the requested dtype. On NVIDIA the sigmoid is the reference
kernels' `rcp.approx(ex2.approx(-x * log2(e)) + 1)`, elsewhere PyTorch's IEEE
one. `rel_mha_decode_with_kvcache` takes the gate and sink itself: a solution
that declares the `gate_sink` trait (the FA4 v2 FP32 decode, in its split
reduction) applies the same epilogue to the unrounded output before its one
store, and for any other solution the op applies `rel_mha_gate_sink` to the
FP32 output and LSE; the two give the same bits.

`rel_mha_head_norms` clips the Q, K and relative heads and RMS-normalizes the V
heads in one launch. A V head runs the `rmsnorm_precision` arithmetic (a shared
helper) and matches that op bit for bit. A clipped head rounds where the
`l2_clip` equation does; its FP32 norm sums in kernel order, so it may differ
from the CPU equation by one unit in the last place. Tests pin it bit for bit to
a standalone kernel of the same arithmetic (`test/l2_clip_helpers.py`).
Given the relative projection weight, a relative head's program also projects
its clipped head to relative logits with a 16-row tensor-core product, which
matches `torch.matmul` of the stored clipped heads bit for bit on B300. That
product spends a whole tensor-core tile on one token, so batches above 128
tokens (prefill) store the clipped heads and run that matmul instead.

The GQA prologue's relative head steps (`RelativeHeadNorm`) run the same
kernel with its K/V rows written once more into their cache slots
(`relative_head_prologue`), so a layer clips, normalizes, projects and writes
its KV in one launch; see `docs/design/attention-prologue.md`. Over a decode
batch the K and V programs also take the K/V short convolution
(`KVShortConv`): each convolves its head's channels as
`inkling_ring_sconv`'s decode kernel does, from the chunk's rows and the
request's ring, and writes its raw row to the ring and, at the request's last
row, the published window. A decode chunk's rows persist at ring rows its
taps never read, so the programs need no ordering.

Under PDL, `rel_mha_head_norms`, `rel_mha_gate_sink`, the decode split pass
and the FA4 decode's split reduction trigger their dependents before their own
wait: their successors read before waiting only constant operands or data
older than the projection at the head of the chain, so a weight-streaming
projection after them prefetches across the chain. The FA4 decode loads a
paged cache's earlier rounds before its wait; its split reduction loads the
gate and sink before its own, so the decode releases it only after waiting.
After its wait the reduction stages every split's maximum, sum and output row
in shared memory in one batch of asynchronous copies (one L2 round trip), then
merges them in split order.
