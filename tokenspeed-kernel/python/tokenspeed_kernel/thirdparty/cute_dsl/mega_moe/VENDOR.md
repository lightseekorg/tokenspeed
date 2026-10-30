# TensorRT-LLM NVFP4 MegaMoE

Source: https://github.com/NVIDIA/TensorRT-LLM/tree/d1166c722ad3b81d7809c9ea98c05e0f1f860248/tensorrt_llm/_torch/cute_dsl_kernels/mega_moe_nvfp4

Revision: `d1166c722ad3b81d7809c9ea98c05e0f1f860248`.
Upstream files retain their Apache-2.0 notices; see LICENSE.
Local adaptation: package imports use the TokenSpeed kernel namespace; formatting follows the repository hooks.
`runner.py` is the TokenSpeed host adapter; it owns symmetric workspaces and launches.
`input_stage.py` is the TokenSpeed BF16-to-workspace adapter. Its input-quantization
scaling order follows FlashInfer's CUDA NVFP4 recipe, including the reciprocal
of the product of block and global decode scales; it stages routing in the same
launch. The expert epilogue keeps the TensorRT-LLM implementation above.

`dynamic_mainloop.py` separates NVVM operand-version detection from the CTA group enum rename for CuTeDSL 4.7.
