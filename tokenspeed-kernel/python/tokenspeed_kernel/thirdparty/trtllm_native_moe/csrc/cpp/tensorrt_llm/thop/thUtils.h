/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

// Minimal adaptation of TensorRT-LLM's thUtils.h for the source-vendored MoE
// prototype.  The full header depends on runtime::ITensor, while this runner
// only needs the Torch scalar aliases and nextPowerOfTwo().

#include "tensorrt_llm/common/config.h"

#include <ATen/cuda/CUDAContext.h>
#include <torch/custom_class.h>
#include <torch/script.h>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

constexpr auto FLOAT4_E2M1X2 = torch::ScalarType::Byte;
constexpr auto SF_DTYPE = torch::ScalarType::Byte;

inline int nextPowerOfTwo(int value)
{
    if (value <= 1)
    {
        return 1;
    }
    --value;
    value |= value >> 1;
    value |= value >> 2;
    value |= value >> 4;
    value |= value >> 8;
    value |= value >> 16;
    return value + 1;
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END
