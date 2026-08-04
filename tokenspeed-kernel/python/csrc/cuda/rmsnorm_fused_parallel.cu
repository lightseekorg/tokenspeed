/*
 * Copyright (c) 2024 by FlashInfer team.
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
#include <flashinfer/norm.cuh>

#include "tvm_ffi_utils.h"

using namespace flashinfer;

void rmsnorm_fused_parallel(TensorView input1, TensorView weight1, TensorView output1,
                            TensorView input2, TensorView weight2, TensorView output2,
                            double eps, bool enable_pdl) {
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(input1);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(input2);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(weight1);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(weight2);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(output1);
  CHECK_LAST_DIM_CONTIGUOUS_INPUT(output2);

  CHECK_DEVICE(weight1, input1);
  CHECK_DEVICE(output1, input1);
  CHECK_DEVICE(input2, input1);
  CHECK_DEVICE(weight2, input1);
  CHECK_DEVICE(output2, input1);

  CHECK_DIM(2, input1);   // input1: [batch_size, dim1]
  CHECK_DIM(2, input2);   // input2: [batch_size, dim2]
  CHECK_DIM(1, weight1);  // weight1: [dim1]
  CHECK_DIM(1, weight2);  // weight2: [dim2]
  CHECK_DIM(2, output1);  // output1: [batch_size, dim1]
  CHECK_DIM(2, output2);  // output2: [batch_size, dim2]

  unsigned int batch_size = input1.size(0);
  unsigned int dim1 = input1.size(1);
  unsigned int dim2 = input2.size(1);

  TVM_FFI_ICHECK_EQ(input2.size(0), batch_size);
  TVM_FFI_ICHECK_EQ(weight1.size(0), dim1);
  TVM_FFI_ICHECK_EQ(weight2.size(0), dim2);
  TVM_FFI_ICHECK_EQ(output1.size(0), batch_size);
  TVM_FFI_ICHECK_EQ(output1.size(1), dim1);
  TVM_FFI_ICHECK_EQ(output2.size(0), batch_size);
  TVM_FFI_ICHECK_EQ(output2.size(1), dim2);

  auto device = input1.device();
  cudaSetDevice(device.device_id);
  const cudaStream_t stream = get_stream(device);

  DISPATCH_DLPACK_DTYPE_TO_CTYPE_FP16(input1.dtype(), c_type, [&] {
    cudaError_t status = norm::RMSNormFusedParallel(
        static_cast<c_type*>(input1.data_ptr()), static_cast<c_type*>(weight1.data_ptr()),
        static_cast<c_type*>(output1.data_ptr()), dim1, static_cast<c_type*>(input2.data_ptr()),
        static_cast<c_type*>(weight2.data_ptr()), static_cast<c_type*>(output2.data_ptr()), dim2,
        batch_size, input1.stride(0), input2.stride(0), output1.stride(0), output2.stride(0),
        static_cast<float>(eps), enable_pdl, stream);
    TVM_FFI_ICHECK(status == cudaSuccess) <<
                "RMSNormFusedParallel failed with error code " << cudaGetErrorString(status);
    return true;
  });
}
