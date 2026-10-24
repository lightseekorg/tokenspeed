# Single-rank packed DSA attention

The non-KVP Ascend DSA backend now reuses the existing SFAD rank-major packed-Q
ABI with a rank dimension of one. It passes the latent and rotary query channels
together (width 576), rather than copying two contiguous tensors. The native
operator writes FP32 head-major attention. The adapter selects the live heads
before converting to the query dtype and preserves the requested token-major
or head-major backend output contract.

Small head counts still repeat to the native 32/64-head geometry. This does not
add a native 8-head kernel or change the native output dtype. No collectives are
introduced. KVP attention, indexer metadata, KV writes, scheduler and cache
allocation remain unchanged. An existing SFAD build with the rank-major packed
ABI is required; no native operator source or binary is changed by this PR.

Run these on an NPU with the registered SFAD implementation:

```bash
pytest -q tokenspeed-kernel-npu/test/test_dsa_packed.py test/runtime/test_ascend_dsa_verify.py
```

The native comparison uses identical Q/KV/indices, 8/32/64 heads and both backend
output layouts. Results must match the existing split-Q attention exactly.
These tests do not certify every native SFAD sequence length: the unmodified
split-Q path has a known eager/graph discrepancy at longer history lengths.
