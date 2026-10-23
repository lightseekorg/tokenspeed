# GLM-5.3-Flash residual benchmarks

`residual.json` covers the complete public `mhc_pre` operation and `mhc_post`
at hidden width 4096 with four residual streams. Preprocessing includes its
internal RMS prenormalization, FP32 projection, coefficient computation with
20 Sinkhorn iterations, and combination into the BF16 layer input. It returns
the layer input, post coefficients, and combination matrix. There is no
separate projection case to double-count that work.

The checkpoint uses RMS epsilon `1e-5` and hyperconnection epsilon `1e-6`.
Residual activations are BF16; projection weights, scales, biases, and mixing
coefficients are FP32. The model applies output RMSNorm after `mhc_pre`, so
these cases explicitly use `fused_norm: false`. Output normalization and
attention/feed-forward computation are outside the measured operations.

## Row counts and provenance

| Rows | Purpose |
| --- | --- |
| 1, 2, 4, 8, 16 | Target-model single-token operation cases for concurrency 1–16 |
| 4, 8, 16, 32, 64 | Target verification with four tokens per request |
| 64, 65, 256, 257 | Both sides of current preprocessing dispatch boundaries |
| 848, 8192 | Remainder and full chunks for serial chunking of a 50,000-token prompt |

Overlapping rows occur once per operation. The row list and its order are part
of the expanded case identities. These are source-derived operation shapes,
not measured historical launch shapes. Multiple requests can change prefill
scheduling. The MTP draft layer disables mHC; none of these cases attributes
residual cost to that draft layer.

Sources:

- [Checkpoint configuration](https://huggingface.co/zai-org/GLM-5.3-Flash/blob/03eb5366286afd40d2221b1d9c63a6dd1ba4832e/config.json),
  under `text_config`.
- [Model call boundaries](https://github.com/lightseekorg/tokenspeed/blob/533394cc05ecfb17064161c34c5b07c995387b00/python/tokenspeed/runtime/models/glm53_flash.py)
  and [current registrations](https://github.com/lightseekorg/tokenspeed/blob/533394cc05ecfb17064161c34c5b07c995387b00/tokenspeed-kernel/python/tokenspeed_kernel/ops/residual/gluon.py).
- [Published workload reference](https://github.com/lightseekorg/amd-tracking/issues/16#issuecomment-6331985608):
  run completed October 21, 2026 and posted October 22. Its profile revision is
  `f668642005377b327789f54417ca0b1a9d727f7b`; profile duration shares establish
  relevance but do not establish launch shapes or wall-time shares.

At the source revision above, normal selection uses the specialized small-row
preprocessing registration through 64 rows, the portable implementation at
65–256 rows, and specialized prefill from 257 rows. Postprocessing uses the
shared gfx950 registration. The benchmarks preserve normal selection so later
dispatch changes are covered rather than pinned to these names.

## Validation and measurement

Each case checks three fresh, seeded, nonzero input sets against the real
PyTorch numerical reference before timing. All three preprocessing outputs and
the postprocessing output are checked. Persistent input tensors must remain
exactly unchanged. These operations require no input reset between invocations;
GPU tests exercise repeated graph replay and the unfused normalization boundary.

The hardware suite retains its five eager warmups, three replay warmups,
30 samples, and cold-cache graph timing. Input creation, reference validation,
compilation, capture, and cache clearing are outside the measured operation.
For a focused run from a prepared kernel environment:

```bash
python -m tokenspeed_kernel.benchmark.ci \
  --suite tokenspeed-kernel/benchmarks/amd/gfx950.json \
  --revision "$(git rev-parse HEAD)" \
  --case-filter '^glm53_flash\.residual\.' \
  --profiler none --output /tmp/glm53-residual.json
```

New GLM case IDs are additions in base/candidate comparisons, with no historical
speedup or slowdown. Existing DeepSeek V4.1 cases remain separate and unchanged.
