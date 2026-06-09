#include "tvm_ffi_utils.h"

void dsv3_router_gemm(TensorView output, TensorView mat_a, TensorView mat_b, bool enable_pdl);

TVM_FFI_DLL_EXPORT_TYPED_FUNC(dsv3_router_gemm, dsv3_router_gemm);
