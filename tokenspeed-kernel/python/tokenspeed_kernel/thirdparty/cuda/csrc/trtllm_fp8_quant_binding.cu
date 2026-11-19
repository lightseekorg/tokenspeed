#include "tvm_ffi_utils.h"

void fp8_quantize_1x128(TensorView output, TensorView scales, TensorView input, bool use_ue8m0);

void quantize_e4m3_per_token(TensorView output, TensorView scales, TensorView input);

TVM_FFI_DLL_EXPORT_TYPED_FUNC(fp8_quantize_1x128, fp8_quantize_1x128);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(quantize_e4m3_per_token, quantize_e4m3_per_token);
