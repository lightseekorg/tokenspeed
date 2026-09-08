# Qwen3.5 QKVZBA repacking

`fused_qkvzba_split_reshape_cat_contiguous` materializes the projection outputs
consumed by the GDN convolution, recurrent kernel and gated normalization.
Each input row contains `[Q | K | V | Z]` and `[B | A]` in final head order.
The leading dimension counts tokens, including flattened MTP tokens.

## Storage contract

Input tensors have unit inner strides. Row strides and base offsets may vary:
the merged projection splits QKVZ and BA from the same GEMM allocation, while
the separate projection returns two independent tensors. QKVZ and BA may have
different dtypes when supplied separately.

The outputs are contiguous and aligned to at least 16 bytes. Their element
ranges do not overlap one another, and they do not reuse the input storage.
The wrapper currently allocates four buffers. QKV and Z preserve the QKVZ
dtype; B and A preserve the BA dtype. The copy preserves every input bit, including signed zero, infinities
and NaN payloads, and does not write the projection inputs. Independent QKV
storage matters because the convolution update can modify it in place;
contiguous Z supports flattening heads for gated normalization without another
copy. Merely returning views would change these contracts.

Eager execution, graph capture and replay use the same wrapper and kernel.
There is no mode-specific layout or cached per-request state.

## Copy geometry

A CTA copies 2048 consecutive QKVZ elements using four warps, selecting either
the QKV or Z destination by feature offset. Tile zero also copies all B/A
heads with adjacent lanes; other tiles do not write gates. Bounds masks handle
partial tiles and non-power-of-two head counts. Actual row strides are
compile-time constants, allowing valid vectorization without assuming input
alignment that padded tensors do not provide. The launcher computes the
power-of-two gate width and passes it as a constexpr; runtime initialization
replaces `triton.next_power_of_2` with a host helper, so the kernel must not
call that helper itself.

The former implementation launched one single-warp CTA per token/QK head and
issued scalar gate copies for every V head. For 4 QK heads, 12 V heads and
128-dimensional heads, the new layout halves the CTA count and replaces six
single-lane gate loads/stores per old CTA with two lane-parallel copies per
token. No shared memory, synchronization or arithmetic on the values is needed.

## Validation and measurements

The `kernel-gen-optimize` staged evaluator ran on NVIDIA B200 (CC 10.0,
148 SMs), PyTorch 2.13.0+cu130 and Triton 3.7.1. Every candidate load applied
the project runtime initialization. The command backend captured
200 consecutive wrapper calls in a CUDA graph and measured replay with CUDA
events, dividing each sample by 200. The graph allocator reused released
output storage. Timing excludes Python allocation/dispatch and does not inject
cache flushes. Release uses the median of ten samples per workload. These are
kernel microbenchmarks, not end-to-end model speedups.

The contiguous-tile candidate passed L2, reducing the equally weighted mean
from 16.913 to 9.879 microseconds (41.6%). All ten comparison workloads improved.
The table shows the baseline medians used by that evaluation, the promoted
candidate's staged measurements and its fresh release measurements, in
microseconds. Head dimensions are 128; padded inputs use BF16 QKVZ with FP32
gates and deliberately unaligned addresses/strides.

| Tokens | QK/V heads | Input rows | Baseline | Candidate at L2 promotion | Release |
| ---: | ---: | --- | ---: | ---: | ---: |
| 1 | 4/12 | packed | 2.564 | 1.131 | 1.130 |
| 8192 | 4/12 | packed | 42.045 | 21.544 | 21.525 |
| 24 | 4/12 | packed | 2.814 | 1.255 | 1.255 |
| 64 | 4/12 | packed | 2.843 | 1.318 | 1.320 |
| 8192 | 4/8 | dense | 23.245 | 15.927 | 15.905 |
| 8 | 8/24 | packed | 2.731 | 1.172 | 1.168 |
| 8192 | 4/4 | dense | 18.721 | 7.355 | 7.354 |
| 8192 | 4/16 | dense | 32.285 | 26.250 | 26.258 |
| 65 | 2/6 | padded | 2.808 | 1.403 | 1.405 |
| 8192 | 4/12 | dense | 39.078 | 21.435 | 21.424 |

Release also passed three additional cases: 3 tokens with FP32 gates and
unaligned inputs; 257 FP16 tokens with QK/V head dimensions 64/128; and
32 FP32 tokens with QK/V head dimensions 128/64. All 13 workloads passed both
the tensor reference and an independent CPU bitwise reference with changed
inputs after graph capture. Hardware occupancy/stall counters were not
collected; generated PTX and measured latency support the design choice.

The repository regression suite covers merged projections, unaligned rows and
storage offsets, mixed dtypes, special values, empty input, tile tails,
non-overlapping contiguous outputs and changed-input graph replay. The GDN
verify launch regression also passes, preserving the elimination of the four
Torch elementwise kernels:

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m pytest -q \
  test/runtime/models/test_qwen3_5_fused_qkvzba.py \
  test/runtime/test_gdn_verify_launches.py
```
