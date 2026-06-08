---
name: optimize-amd-gpu-kernel
description: Optimizing kernel performance on AMD Instinct MI GPUs.
---

## General Methodology

* Understand the problem: check whether compute-, memory-, or latency-bound,
  and approach accordingly.
* Profile, focus on the top bottleneck; iterate in such loop.
* Use Gluon for explicit low-level controls.
* Pay attention to both `ttg` level and `llvm` level opportunities and balances.
* Fine to tune configuration parameters but don't solely focus on it and don't
  overdo it. Prefer to avoid complicated switch cases to overfit different
  problem sizes.

## Profiling Tools

* Use Proton for high-level TFLOp/s or TB/s calculation; check in code changes
  for `triton.jit`/`gluon.git` `repr` for reuse.
* Use `rocprofv3` in ROCm to understand low-level internals like counters.
* Proton also supports fine-grained profiling with `scope` APIs and
  instrumentation mode.

## Problem Approaches

### General optimizations

Applicable to various problems:

* Prefer to launch enough workgroups to fill the GPU.
* Ensure proper software pipelining to break dependencies in the same loop
  iteration.
* Prefer coaleased and vectorized async global memory load/store.
* If indexing range allows, prefer buffer load/store intrinsics in Gluon to
  avoid out-of-bound branches and overheads.
* Avoid shared memory bank conflict if possible. Use padding instead of
  swizzling.

### Compute bound problems

The key is to keep issuing MFMA instructions preferrably every cyle, and
avoid exposed memory instruction latencies. Generally two approaches:

* Use 4 waves per workgroup, and perform fine-grained per-instruction level
  interleaving in the same wave on one SIMD. Typically needs controlling
  LLVM knobs; can use `HIPOptions.llvm_fn_attrs`.
* Use 8 waves per workgroup, and perform course-grained multi-instruction
  level interleaving across 2 waves on the same SIMD, to make sure those
  two waves "ping-pong" among each other to overlap. Available via the
  `amd.warp_pipeline_stage` API.

Search and read AMD ISA docs and Triton codebase and examples to get
inspiration.

* If high VGPR pressure, consider slice along M/N in the hot loop and
  interleave to retire certain slices of loaded values earlier.

### Memory bound problems

The key is to saturate GPU memory bandwidth with enough inflight memory
instructions, and avoid exposed compute instruction cycles.

* Prefetch using async load with higher number of shared memory buffers.
* Use cache modifiers like `".cg"`, `".wt"`, etc. to control whether to
  cache at certain levels.

### Latency bound problems

* Fuse multiple small kernels into one kernel when possible.

### Small problem sizes

* Perform split-k style optimization and launch second reduction kernel to
  see if beneficial.

### Expensive epilogue

* Perform persistent kernrel style optimization to see if benefical.
