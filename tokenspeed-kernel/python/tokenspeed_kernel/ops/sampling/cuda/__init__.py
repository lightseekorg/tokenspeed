from tokenspeed_kernel._vendor import export_vendor_symbols, noop_fn

__all__ = [
    "chain_speculative_sampling_target_only",
    "fused_topk_topp_prepare",
    "fused_topk_topp_renorm",
    "fused_topk_topp_workspace_size",
    "verify_chain_greedy",
]

globals().update(
    export_vendor_symbols(
        "nvidia",
        "tokenspeed_kernel.ops.sampling.cuda",
        __all__,
        fallback_by_name={"fused_topk_topp_prepare": noop_fn},
    )
)
