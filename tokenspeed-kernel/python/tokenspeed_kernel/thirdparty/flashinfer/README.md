# FlashInfer adapters

Adapters in this directory keep FlashInfer-specific calls and prepared weight
layouts behind `tokenspeed-kernel`. Public API validation, capability selection
and registration belong in the corresponding `ops/` family.

For SM120 V4.1, `sm12x_fused_moe.py` prepares original MXFP4 expert weights;
`sm12x_nvfp4_moe.py` preserves calibrated per-expert NVFP4 scales;
`sparse_mla_sm120.py` wraps sparse attention and `topk.py` wraps exact TopK.
The [MoE contracts](../../ops/moe/README.md) and [V4.1 attention
contracts](../../ops/attention/dsv41/README.md) describe their callers.
