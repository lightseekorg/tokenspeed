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

// Tokenspeed-side compatibility shims for the TRT-LLM ws_layernorm sources.
//
// flashinfer's bundled nv_internal copy of cudaUtils.h renames the helpers
// ConstInt<>/ConstBool<> to Int<>/Bool<>. Provide forwarding aliases so the
// imported TRT-LLM kernel files can keep their original spelling.

#include "tensorrt_llm/common/cudaUtils.h"

namespace tensorrt_llm
{
namespace common
{

template <int VALUE>
using ConstInt = Int<VALUE>;

template <bool VALUE>
using ConstBool = Bool<VALUE>;

} // namespace common
} // namespace tensorrt_llm
