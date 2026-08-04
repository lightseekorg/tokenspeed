"""MiniMax M2 TRT-LLM communication kernels."""

from tokenspeed_kernel.registry import error_fn

minimax_allreduce_rms_qk = error_fn
trtllm_create_ipc_workspace_for_minimax = error_fn

try:
    from tokenspeed_kernel.ops.communication.trtllm.native import (
        minimax_allreduce_rms_qk,
        trtllm_create_ipc_workspace_for_minimax,
    )
except ImportError:
    pass
