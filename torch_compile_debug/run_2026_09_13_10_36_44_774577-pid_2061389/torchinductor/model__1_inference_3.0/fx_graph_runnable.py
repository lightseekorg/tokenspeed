s3 = 128256


import os
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '1'
os.environ['TORCHINDUCTOR_CACHE_DIR'] = '/tmp/torchinductor_root'
os.environ['TORCHINDUCTOR_COMPILE_THREADS'] = '1'

import torch
from torch import tensor, device
import torch.fx as fx
from torch._dynamo.testing import rand_strided
from math import inf
import torch._inductor.inductor_prims



import torch._dynamo.config
import torch._inductor.config
import torch._functorch.config
import torch.fx.experimental._config
torch._dynamo.config.assume_static_by_default = False
torch._inductor.config.allow_buffer_reuse = False
torch._inductor.config.fallback_random = True
torch._inductor.config.compile_threads = 1
torch._inductor.config.comprehensive_padding = False
torch._inductor.config.triton.unique_kernel_names = True
torch._inductor.config.trace.enabled = False
torch._inductor.config.trace.save_real_tensors = False
torch._functorch.config.functionalize_rng_ops = False
torch._functorch.config.fake_tensor_allow_unsafe_data_ptr_access = True
torch._functorch.config.unlift_effect_tokens = True
torch._functorch.config.selective_decompose = False



isolate_fails_code_str = None





if "__compile_source__" in globals():
    import inspect as __after_aot_inspect
    import linecache as __after_aot_linecache
    __after_aot_filename = __after_aot_inspect.currentframe().f_code.co_filename
    __after_aot_linecache.cache[__after_aot_filename] = (
        len(__compile_source__),
        None,
        __compile_source__.splitlines(True),
        __after_aot_filename,
    )
# torch version: 2.10.0+cpu
# torch cuda version: None
# torch git version: 449b1768410104d3ed79d3bcfe4ba1d65c7f22c0


# torch.cuda.is_available()==False, no GPU info collected

from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    
    
    def forward(self, arg0_1, arg1_1, arg2_1):
        convert_element_type = torch.ops.prims.convert_element_type.default(arg1_1, torch.float32);  arg1_1 = None
        _log_softmax = torch.ops.aten._log_softmax.default(convert_element_type, -1, False);  convert_element_type = None
        unsqueeze = torch.ops.aten.unsqueeze.default(arg2_1, -1);  arg2_1 = None
        gather = torch.ops.aten.gather.default(_log_softmax, -1, unsqueeze);  _log_softmax = unsqueeze = None
        squeeze = torch.ops.aten.squeeze.dim(gather, -1);  gather = None
        return (squeeze,)
        
def load_args(reader):
    reader.symint(128256)  # arg0_1
    buf0 = reader.storage(None, 2*s3, device=device(type='npu', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf0, (1, s3), dtype=torch.bfloat16, is_leaf=True)  # arg1_1
    buf1 = reader.storage(None, 4*s7, device=device(type='npu', index=0), dtype_hint=torch.int32)
    reader.tensor(buf1, (1,), dtype=torch.int32, is_leaf=True)  # arg2_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='symbolic', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='symbolic', check_str=None)
        # mod(*args)