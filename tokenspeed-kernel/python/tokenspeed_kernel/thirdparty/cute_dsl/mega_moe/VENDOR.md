# TensorRT-LLM NVFP4 MegaMoE

Source: https://github.com/NVIDIA/TensorRT-LLM/tree/d1166c722ad3b81d7809c9ea98c05e0f1f860248/tensorrt_llm/_torch/cute_dsl_kernels/mega_moe_nvfp4

Revision: `d1166c722ad3b81d7809c9ea98c05e0f1f860248`.
Upstream files retain their Apache-2.0 notices; see LICENSE.
Local adaptation: package imports use the TokenSpeed kernel namespace; formatting follows the repository hooks.
`runner.py` is the TokenSpeed host adapter; it owns symmetric workspaces and launches.

`dynamic_mainloop.py` separates NVVM operand-version detection from the CTA group enum rename for CuTeDSL 4.7.
