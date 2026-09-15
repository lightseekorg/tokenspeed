# Native H3584 FlashInfer HT specialization

This directory vendors
`flashinfer/comm/mnnvl_cutedsl/kernel_ht/device_kernel.py` from FlashInfer
`v0.6.18` (`69ff11fc4954396d98326656dc85debd2223f637`) under its original
Apache-2.0 license. The upstream file SHA256 is
`076c6621d5456affa6c7255c868260a90904a3e4c624d18779d15f35a54c44a6`.
The specialized `device_kernel.py` SHA256 is
`a2bc36152b60b654a9a9c6eb5b715a3e4d884d0cedc0fb0193f30f440598496a`.
`primitives.py` is a new tail-predicated form of that revision's inline
multimem load/store operations; its SHA256 is
`05584df5b16e557f914c61ebb92fcd02884908a51222fc4fa8e0f7df0d71e2cb`.

The local device-kernel diff is intentionally narrow:

1. `reduction_vectors_per_thread` uses ceiling division.
2. The final, possibly out-of-range `bf16x8` reduction pack is guarded by
   predicated `multimem.ld_reduce` and `multimem.st` helpers.
3. The class is public and imports FlashInfer's unchanged shared primitives.

This permits Kimi K3's native H3584/TP8 geometry: 448 packs per row and 56
packs per TP-owned reduction shard. It removes the H4096 input/output copies
and RMSNorm rescaling required by padding.

## Geometry audit

| Upstream constraint | H3584/TP8 resolution |
| --- | --- |
| `hidden % 8 == 0` | 3584 / 8 = 448 BF16x8 packs |
| `active_ctas % tp == 0` | protocol rounds the persistent grid to TP8 |
| consumer count is a warp multiple | 448 consumers = 14 warps |
| `hidden % (consumer_threads * 8 * vectors_per_thread) == 0` | 448 x 8 x 1 = 3584 |
| packs divide TP | 448 / 8 = 56 packs per reduction shard |
| packs divide consumers | 448 / 448 = one clear vector per consumer |
| packs divide RMS copy threads | two token groups give 224 threads and two packs per thread |
| RMS threads form whole warps | 224 threads = seven warps per token group |
| shard-major RMS warps divide TP | disabled because seven does not divide TP8 |
| RMS pipeline storage fits staged rows | two groups x three RMS stages use six rows within seven producer stages |
| reduction packs divide reduction threads | generalized to ceiling division plus a predicated tail |

No tail handling is needed in the producer, consumer, mailbox clear, or
RMSNorm sections under this tuning. Only the MNNVL reduction has a partial
warp: with two reduction warps, lanes 56--63 issue neither the multicast load,
the optional residual load, nor the multicast store. The integer address may
be formed for those lanes, but it is never dereferenced.

Recommended initial all-reduce tuning on GB300 is `consumer_threads=448`,
`vectors_per_thread=1`, `reduction_warps=2`, `rms_token_groups=2`,
`rms_pipeline_stages=1`, `rms_shard_major=False`, and `stages=2`. Both 32- and
64-thread reduction variants should be measured: one warp performs two trips
(32 + 24 valid packs), while two warps perform one trip with eight inactive
lanes. `rms_token_groups=4` is not valid with 448 consumers because it would
produce 112 RMS threads (three and a half warps).

The exported top-k-16 finalize tuning uses the same 448-consumer geometry with
seven producer stages, two reduction warps, two RMS token groups, three RMS
pipeline stages, and PDL enabled. The supplied TP8 GB300 rank-MAX CUDA Graph
benchmarks are the qualification gates against standalone finalize + staged
multimem AllReduce + RMSNorm and against the complete production-ordered SiTU
MoE tail. Results from a different top-k or source revision do not qualify the
route. Its inherited call surface is:

```python
protocol.finalize_kernels[tuning](
    routed_output,
    expert_weights,
    permuted_indices,
    shared_output,
    residual_source,
    gamma,
    m,
    state=protocol.state,
    norm_output=norm_output,
    residual_output=residual_output,
)
```

The distributed communication test and complete-tail benchmark compile the
native kernel on TP8, validate the predicated reduction tail, and replay it
through changed-input CUDA Graphs before measuring it.

To verify provenance, check the hashes above and compare the recorded upstream
file directly:

```bash
sha256sum device_kernel.py primitives.py
git diff --no-index /path/to/flashinfer-0.6.18/kernel_ht/device_kernel.py \
  device_kernel.py
```
