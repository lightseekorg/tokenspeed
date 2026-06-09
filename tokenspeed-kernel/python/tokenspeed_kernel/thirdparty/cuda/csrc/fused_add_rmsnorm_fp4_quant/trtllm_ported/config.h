/*
 * Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */
#pragma once
#ifndef TRTLLM_CONFIG_H
#define TRTLLM_CONFIG_H

// Tokenspeed stub of tensorrt_llm/common/config.h.
//
// The flashinfer-bundled nv_internal headers (cudaUtils.h, quantization.cuh,
// cudaBf16Fallbacks.cuh, etc.) place their symbols directly in
// `tensorrt_llm::common` and `tensorrt_llm::kernels` (no inline ABI namespace).
// The TRT-LLM ws_layernorm sources we ported below use TRTLLM_NAMESPACE_BEGIN
// which originally wraps everything in `tensorrt_llm::_v1::...`. To stay
// consistent with the bundled flashinfer headers we degrade these macros to
// just the plain `tensorrt_llm` namespace so the two halves can interoperate.

#define TRTLLM_NAMESPACE_BEGIN                                                                                         \
    namespace tensorrt_llm                                                                                             \
    {

#define TRTLLM_NAMESPACE_END /* end namespace tensorrt_llm */ }

#define TRTLLM_ABI_NAMESPACE_BEGIN
#define TRTLLM_ABI_NAMESPACE_END

#endif // TRTLLM_CONFIG_H
