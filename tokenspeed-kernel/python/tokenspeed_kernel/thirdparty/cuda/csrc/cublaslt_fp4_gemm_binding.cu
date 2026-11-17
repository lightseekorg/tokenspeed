#include "tvm_ffi_utils.h"

void cublaslt_fp4_gemm(TensorView out, TensorView a, TensorView b, TensorView a_sf, TensorView b_sf,
                       TensorView alpha, TensorView beta, TensorView workspace);

TVM_FFI_DLL_EXPORT_TYPED_FUNC(cublaslt_fp4_gemm, cublaslt_fp4_gemm);
