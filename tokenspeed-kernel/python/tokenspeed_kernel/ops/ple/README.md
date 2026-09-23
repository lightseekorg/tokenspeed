# PLE kernels

The fused convolution/state kernel takes token and request counts as runtime
scalars. These counts bound token output and per-request state writes; they do
not determine static tile sizes. Eager batches with different counts reuse a
compiled kernel within Triton's scalar specialization buckets. CUDA graphs
capture the scalar values supplied at capture time.

The convolution regression tests in `test/nvidia/ops/test_ple_prototypes.py`
cover ragged and empty requests, verification windows, final states, CUDA graph
replay, and compiled-kernel reuse across token and request counts.
