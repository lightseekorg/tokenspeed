# HY4 Preview MXFP8 on Hopper

HY4 Preview uses ModelOpt MXFP8: E4M3 weights with one E8M0 scale for each
group of 32 input channels. The checkpoint format is detected automatically.
On Hopper, the baseline keeps those compressed weights and runs dense and
routed-expert projections with Marlin W8A16: 8-bit weights and BF16
activations. It does not quantize those activations to FP8.
The supported model format is ModelOpt MXFP8. A full BF16 checkpoint is
rejected because the current BF16 MoE path does not provide HY4's expert
parallelism, activation clamp and fixed-order reduction together.

Install the runtime and `tokenspeed-kernel` from compatible revisions. HY4
uses the shared DSA/FlashMLA implementation with learnable attention sinks
and BF16 sparse decode support; a FlashMLA package that only implements the
GLM interface is insufficient. The model adapter does not carry a separate
FlashMLA or CUTLASS source tree.

## Computation and checkpoint semantics

| Component | Baseline behavior |
| --- | --- |
| Quantized dense projections and routed experts | Original group-32 E4M3/E8M0 weights, Marlin W8A16, BF16 activations |
| Checkpoint-excluded projections, including the attention output gate | BF16 weights and activations |
| Absorbed MLA Q/K and V/output contractions | BF16 tensor weights materialized when loading |
| Fused indexer `wk` and `weights_proj` | `wk` is dequantized once and concatenated with the BF16 `weights_proj`; the fused projection runs in BF16 |
| Sparse main attention on Hopper | Shared FlashMLA BF16 arithmetic; the launch below uses BF16 main KV storage |
| Indexer key cache | FP8, independently of the main KV cache dtype |
| Router and iHC gate parameters | FP32; iHC reductions retain the model's BF16 boundaries |
| LM head | BF16 weight storage with FP32 logits |

The loader keeps weight and scale shards paired through tensor parallelism,
fused Q/KV loading, and fused gate/up loading. The Marlin adapter repacks
weights and scales once before graph capture. It rejects incomplete or
duplicate shards and unsupported scale encodings rather than silently
dequantizing ordinary dense projections to BF16. E8M0 byte `255` is not a
finite scale; byte `0` is also rejected because this Marlin conversion does
not support that encoding. Every tensor-parallel input partition must be a
multiple of 32.

The target and MTP loaders declare that attention-TP peers consume the same
filtered checkpoint stream. Ranks on each node divide background shard reads
within that group instead of each prefetching the full checkpoint. Every rank
still consumes its complete selected tensor stream and applies its usual
parameter sharding.

The adapter preserves HY4's numerical boundaries around these GEMMs:

- The attention gate rounds the sigmoid to BF16 before multiplication.
- Plain RoPE coefficients round through the model dtype. The indexer rotates
  the trailing RoPE slice in interleaved pairs; GLM's indexer layout differs.
- iHC checkpoint parameters retain FP32 precision, and each boundary uses
  the configured normalization epsilon.
- Latent KV normalization applies its learned weight once. BF16 caches use
  the dual Q/KV norm before the prologue; packed FP8 caches leave KV raw for
  the prologue's fused normalization and write.
- The expert output and its routing weight round separately to BF16, their
  product rounds to BF16, and the top-k sum accumulates in FP32.

## Launch template

The following is a configuration template for four nodes with eight H20 GPUs
each, not a measured capacity or performance result. Set the checkpoint path
and the coordinator address, and run it on every node with its own rank:

```bash
ts serve /path/to/hy4-mxfp8 \
  --served-model-name hy4 \
  --nnodes 4 --node-rank <0-to-3> \
  --dist-init-addr <node0-host>:25000 \
  --attn-tp-size 32 --dense-tp-size 32 \
  --enable-expert-parallel \
  --attention-backend dsa \
  --moe-backend marlin \
  --dtype bfloat16 \
  --kv-cache-dtype auto \
  --max-model-len 65536 \
  --max-num-seqs 64
```

Pipeline parallelism must remain at one. iHC requires complete token rows at
each sub-block boundary: use attention data parallelism or a layout where
both dense and MoE communication all-reduce the rows. A reduce-scatter
layout that breaks this requirement is rejected during construction.
Additional head parallelism and query context parallelism are not supported
by this adapter; ordinary attention TP/DP and KV context parallelism use the
shared DSA path.
Derived absorbed and repacked weights are prepared during loading; live
weight updates require a fresh model load.

MTP is supported by the checkpoint's own `model.mtp_layers.0` block and
shares the target embedding and LM head. Leave speculation off for the first
Marlin/group32 comparison. Measure MTP separately by adding
`--speculative-algorithm MTP --speculative-num-steps 3` on every node; its
acceptance rate and extra memory depend on the workload.

`ts serve` defaults HY4's reasoning and tool-call parsers to `hy_v4`, using
the model ID or local configuration. Explicit parser flags override these
defaults. Keep the tokenizer and chat template from the same checkpoint.
With `--grammar-backend xgrammar`, JSON schema constraints begin after the
tokenizer's matching think-end token; a `no_think` prompt applies them from
the first generated token.

## Validation scope

`test/runtime/test_hy_v4.py` exercises configuration, real parameter loading,
RoPE, sink binding and attention gating.
`test/runtime/test_modelopt_mxfp8.py` covers the quantized loader.
The Marlin dense/MoE and iHC kernel tests compare actual outputs with numerical
references and exercise supported boundary cases.

Passing these tests does not establish full-model quality, multi-node
correctness or throughput. Acceptance must use the same checkpoint,
tokenizer, prompts and decoding configuration for both backends, record the
selected kernels, and compare eager/graph outputs and caches, long-context
behavior, service failures, latency, throughput and peak GPU memory. Compare
group32 W8A8 as an explicit optional backend before changing the baseline.
