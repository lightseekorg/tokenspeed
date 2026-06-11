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
