/*
 * Copyright (c) 2023 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "tvm_ffi_utils.h"

void rmsnorm_fused_parallel(TensorView input1, TensorView weight1, TensorView output1,
                            TensorView input2, TensorView weight2, TensorView output2,
                            double eps, bool enable_pdl);

TVM_FFI_DLL_EXPORT_TYPED_FUNC(rmsnorm_fused_parallel, rmsnorm_fused_parallel);
