# DeepSeek V4.1 on Ascend

## Scope and numerical behavior

Two serving profiles share attention, metadata, PD transfer and native decode
graphs. Select the profile explicitly.

| Profile | Target experts | Dense | Target KV / Index | DSpark |
|---|---|---|---|---|
| `precision` | Checkpoint weights expanded to BF16 | BF16 | BF16 storage with V4.1 QDQ | Off |
| `performance` | Preconverted FlatQuant W4A4 | BF16 | Direct BF16, no target QDQ | On, BF16 draft experts |

Both retain one native decode graph per local batch-size bucket. Performance
profile draft KV retains its FP8 quantize/dequantize path. NPU cache options
must explicitly select `bf16-qdq` or `bf16`; resident packed cache is unsupported.
Temporary SWA/global codecs remain for QDQ and DSpark, not a third storage mode.

Dense FP8 checkpoint codes are decoded to BF16 on CPU after the shared weight
loader selects the local TP shard. E8M0 scales are folded into the weight once
after loading; inference uses BF16 GEMM. This avoids converting other ranks'
shards without changing the decoded weights or requiring an offline checkpoint.

The precision profile expands quantized checkpoint operands to BF16; its GEMM
accumulation differs from native low-bit inference. FlatQuant weights and
matrices are supplied offline. Native Index selection uses FP32 scores, while
candidate-pool generation preserves BF16 intermediate rounding.

## Dependencies and local builds

Use the Ascend environment and serving dependencies described in the
[NPU package README](../../tokenspeed-kernel-npu/README.md).
The additional `flash_ops` package must register:

- `custom.npu_sparse_flash_attention` for compact and direct paged attention;
- `custom.npu_lightning_indexer` for native BF16 index selection;
- `custom.flat_quant` for W4A4 activation quantization.

After sourcing CANN, build the repository's Ascend C kernels from the
repository root:

```bash
bash tokenspeed-kernel-npu/csrc/assert_indices/build.sh
bash tokenspeed-kernel-npu/csrc/token_shards/build.sh
for kernel in cache_gather candidate_gather index_metadata index_scan ngram paged_metadata route_weights w4a4 window_metadata; do
    bash "tokenspeed-kernel-npu/csrc/dsv41_${kernel}/build.sh"
done
```

The scripts target `dav-2201` and write libraries under
`tokenspeed-kernel-npu/python/tokenspeed_kernel_npu/lib/`. Dispatch depends on
supported input geometry. The V4.1 weight preparation hook checks required
native libraries before preparing weights; W4A4 additionally requires
`libdsv41_w4a4.so`. Missing libraries fail explicitly instead of silently changing
the implementation. `token_shards` remains optional shared infrastructure.
Do not rebuild them inside a running server.

## PD serving

The common launcher is
`test/ci_system/serve_deepseek_v41_flash_npu_dp_pd.sh ROLE LAYOUT PROFILE`.
Set environment variables explicitly on each host:

| Role | Required variables |
|---|---|
| All | `MODEL`, `SERVED_MODEL_NAME`, `HOST`, `PORT` |
| Prefill/decode | `DIST_INIT_ADDR`, `BOOTSTRAP_PORT`, `ASCEND_RT_VISIBLE_DEVICES` |
| Gateway | `PREFILL_URL`, `DECODE_URL`, `BOOTSTRAP_PORT`, `PROMETHEUS_PORT` |

`MODEL` is a local path. Only `performance` workers require `FLATQUANT_CONFIG`,
a local manifest path. The preconverted weight manifest is described
in the [MoE configuration](../../tokenspeed-kernel/python/tokenspeed_kernel/ops/moe/README.ascend.md).
List all 16 local devices explicitly for each worker. Gateway worker URLs use
`grpc://`. Run the following on the respective hosts after setting their variables:

```bash
bash test/ci_system/serve_deepseek_v41_flash_npu_dp_pd.sh prefill dp2 performance
bash test/ci_system/serve_deepseek_v41_flash_npu_dp_pd.sh decode dp2 performance
bash test/ci_system/serve_deepseek_v41_flash_npu_dp_pd.sh gateway dp2 performance
```

Replace `performance` with `precision` on all roles for the BF16 expert
comparison. Expanded BF16 experts require more memory; size the cache budget
and card count for the selected context length before serving.

Use `dp4` for TP4/DP4 decode instead. The `..._dp2_pd.sh` and `..._dp4_pd.sh`
wrappers accept a single role argument and select that layout with the
performance profile.

Prefill uses TP16/Dense TP8. Decode uses TP8/DP2 or TP4/DP4 with matching
Dense and LM-head TP, EP16, Engram TP16, and rank-ordered MoE combination.
DP2 captures every local decode batch size from 1 through 32, with a global
sequence limit of 64 to allow uneven DP admission; gateway concurrency remains
32. DP4 captures 1/2/4/8 with a global sequence limit of 32. Graph padding is
enabled: batches within the ladder use the smallest adequate captured graph.
DP replicas still use a common graph batch size when their live counts differ.
The launcher selects direct paged attention, AIV HCCL, `TASK_QUEUE_ENABLE=0`,
a DP-aware gateway and non-layerwise Mooncake transfer.

Routing retains the upstream policy: this recipe selects `round_robin`, which
in PD uses `bootstrap_room % dp_size`; it does not guarantee an evenly split
decode cohort. PD rank pins retain their upstream validation semantics.
P/D cache field layouts must match.

The 7,200-second receive deadline includes time queued for remote prefill;
`TOKENSPEED_DISAGGREGATION_WAITING_TIMEOUT` overrides it. It changes timeout
handling, not throughput. Decode prefix caching is disabled by this recipe.
