# KDA attention

Batched replay pointer descriptors carry an explicit `state_dtype`. NVIDIA
replay supports FP32 and BF16 storage with FP32 accumulation; Gluon replay
requires FP32 storage. Native NVIDIA prefill scans receive FP32 state, so
adapters widen BF16 checkpoints before calling their fixed ABI. State
publication is the rounding boundary, not each recurrent arithmetic step.

The `flashinfer` prefill solution (sm100/sm103, head dim 128, heads divisible
by eight, gate bound in [-5, 0)) runs FlashInfer's persistent CuTe DSL scan.
It takes the BF16 gate and beta logits as produced, the V-major FP32 state the
prefill gather yields, and advances that state in place. FlashInfer's public
`recurrent_kda` plans this kernel from the device boundaries with a per-call
`tolist()`, so the adapter plans from `cu_seqlens_cpu` and calls the kernel's
forward entry, reusing one plan for every layer that shares the boundaries.

On sm100/sm103 (BF16 activations, head dim 128) speculative verify and replay
are a raw-gate pair of persistent CuTe DSL kernels. The verify
(`_cute_dsl/verify.py`) runs the gate GEMM straight into the replay payload,
then one launch per layer convolves the raw projections from the committed
window, runs the recurrence over the draft positions and writes the output
plus the payload's raw QKV and beta rows; it stores no per-position state. The
batched replay (`_cute_dsl/replay.py`) reads that BF16 raw gate and applies the
decay gate in-kernel. Both prefetch the next work item's state tile and inputs
with bulk copies while computing the current one, and contract over K in
registers, so no barrier runs per token. The replay's `replay_raw_gate` trait
switches the runtime to this payload format, so the two are registered with
the same coverage. The convolution-window publication stays the shared Triton
launch; other NVIDIA targets run the Triton verify and replay.
