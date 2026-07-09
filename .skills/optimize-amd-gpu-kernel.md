---
name: optimize-amd-gpu-kernel
description: Optimizing kernel performance on AMD Instinct MI GPUs.
---

## General Methodology

* Start from representative model shapes.
* Understand whether the case is compute-, memory-, launch-, or latency-bound.
  Then decide whether to optimize compute/memory overlap, memory movement,
  kernel count, tune tiling and configurations, etc.
* Profile first, focus on the top bottleneck, change one thing, and rerun the
  same benchmark/profiling method.
* Use Gluon for explicit low-level control: buffer load/store, async copy to
  LDS, shared layouts, MFMA layout, wave count, and LLVM attributes.
* Pay attention to both `ttg` level and `llvm` level opportunities and balances.
* Tune configuration parameters, but do not overfit with many one-off switch
  cases.

## Profiling Tools

* Use Proton for high-level TFLOp/s or TB/s calculation; check in code changes
  for `triton.jit`/`gluon.jit` `repr` for reuse.
* Use `rocprofv3` in ROCm to understand low-level internals like counters.
* Proton also supports fine-grained profiling with `scope` APIs and
  instrumentation mode.

## Problem Approaches

### General optimizations

Applicable to various problems:

* Prefer to launch enough workgroups to fill the GPU.
* Ensure proper software pipelining to break dependencies in the same loop
  iteration.
* Prefer coalesced and vectorized async global memory load/store.
* If indexing range allows, prefer buffer load/store intrinsics in Gluon to
  avoid out-of-bound branches and overheads.
* Avoid shared memory bank conflict if possible. Use padding instead of
  swizzling.
* For async copy to LDS, arrange global load layouts so each thread issues wide,
  aligned loads where possible; 128-bit per-thread loads are a good target.

### Compute bound problems

The key is to keep issuing MFMA instructions preferably every cycle, and avoid
exposed memory instruction latencies. Generally two approaches:

* Use 4 waves per workgroup, and perform fine-grained per-instruction level
  interleaving in the same wave on one SIMD. Typically needs controlling LLVM
  knobs; can use `HIPOptions.llvm_fn_attrs`.
* Use 8 waves per workgroup, and perform coarse-grained multi-instruction level
  interleaving across 2 waves on the same SIMD, to make sure those two waves
  "ping-pong" among each other to overlap. Available via the
  `amd.warp_pipeline_stage` API.

Search and read AMD ISA docs and Triton codebase and examples to get
inspiration.

* If high VGPR pressure, consider slice along M/N in the hot loop and interleave
  to retire certain slices of loaded values earlier.

### Memory bound problems

The key is to saturate GPU memory bandwidth with enough inflight memory
instructions, and avoid exposed compute instruction cycles.

* Prefetch using async load with higher number of shared memory buffers.
* Use double or triple buffering only when it hides real latency. Extra buffers
  increase LDS/register pressure and may reduce occupancy or compiler quality.
* Use cache modifiers like `".cg"`, `".wt"`, etc. to control whether to cache at
  certain levels.

### Latency bound problems

* Fuse multiple small kernels into one kernel when possible.

### Small problem sizes

* Perform split-k style optimization and launch second reduction kernel to
  see if beneficial.
* Split-K can increase occupancy for high K, but the second reduction/finalize
  kernel costs several microseconds. Only route it when it is consistently
  faster than torch for the real shapes.

### Expensive epilogue

* Perform persistent kernel style optimization to see if beneficial.
* If an epilogue/reduction is unavoidable, consider whether it can be fused into
  the producer without hurting occupancy or memory behavior.
