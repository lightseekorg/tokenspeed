# QSA sparse attention

Prefill uses FlashInfer FA2. Uniform decode on supported NVIDIA Blackwell devices uses the CuTe DSL kernel. Both solutions consume physical selected-slot rows and the full KV cache through the `qsa_sparse_attention` kernel boundary.

The FlashInfer runner keeps a plan for each actual query-row count, selected width, KV geometry and dtype. Prefill CUDA graphs retain eager attention breaks, so replay can encounter a new actual row count even when it reuses a captured token bucket. The first use of a row count must prepare a plan without a Python loop over its rows.

The runner plans from a zeroed packed mask and later fills the same shared mask buffer with the selected slots. Each row occupies `ceil(selected_width / 8)` bytes; mask offsets passed to FlashInfer are byte offsets. FlashInfer's packed-mask planning path can leave offsets in bits, so the runner binds its own byte-offset table after planning. Both the offsets and the shared slot and mask buffers remain valid when the plan is reused for another request. A directly captured standalone call pins its plan until its graph is released.

Correctness checks in `test/nvidia/ops/test_qsa_sparse_attention.py` cover selected-slot changes, widths aligned and unaligned to a byte, plan reuse across row counts, BF16 and FP8 caches, and CUDA graph replay. The planner test also rejects a return to FlashInfer's row-by-row bool-mask conversion.
