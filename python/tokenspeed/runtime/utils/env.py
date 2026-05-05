# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import warnings
from contextlib import contextmanager
from typing import Any

from tokenspeed.runtime.utils.pdl import pdl_enabled
from tokenspeed.runtime.utils.server_args import ServerArgs

global_server_args_dict: dict = {
    "attention_backend": ServerArgs.attention_backend,
    "sampling_backend": ServerArgs.sampling_backend,
    "attention_use_fp4_indexer_cache": ServerArgs.attention_use_fp4_indexer_cache,
    "disable_deepseek_v4_fast_mhc": ServerArgs.disable_deepseek_v4_fast_mhc,
    "deepseek_v4_mega_moe_max_num_tokens": ServerArgs.deepseek_v4_mega_moe_max_num_tokens,
    "deepseek_v4_indexer_prefill_max_logits_mb": ServerArgs.deepseek_v4_indexer_prefill_max_logits_mb,
    "triton_attention_reduce_in_fp32": ServerArgs.triton_attention_reduce_in_fp32,
    "kv_cache_dtype": ServerArgs.kv_cache_dtype,
    "enable_nan_detection": ServerArgs.enable_nan_detection,
    "enable_p2p_check": ServerArgs.enable_p2p_check,
    "mapping": ServerArgs.mapping,
    "force_deterministic_rsag": ServerArgs.force_deterministic_rsag,
    "low_latency_max_num_tokens_per_gpu": ServerArgs.low_latency_max_num_tokens_per_gpu,
    "device": ServerArgs.device,
    "draft_model_path_use_base": ServerArgs.draft_model_path_use_base,
    "disable_pdl": ServerArgs.disable_pdl,
    "enable_prefix_caching": ServerArgs.enable_prefix_caching,
    "mla_disable_ragged": ServerArgs.mla_disable_ragged,
    "chunked_prefill_size": ServerArgs.chunked_prefill_size,
    "mla_chunk_multiplier": ServerArgs.mla_chunk_multiplier,
    "ep_num_redundant_experts": ServerArgs.ep_num_redundant_experts,
    "ep_dispatch_algorithm": ServerArgs.ep_dispatch_algorithm,
    "enable_eplb": ServerArgs.enable_eplb,
    "mm_mode": ServerArgs.mm_mode,
    "comm_fusion_max_num_tokens": ServerArgs.comm_fusion_max_num_tokens,
    "enable_allreduce_fusion": ServerArgs.enable_allreduce_fusion,
    "max_prefill_tokens": ServerArgs.max_prefill_tokens,
    "max_model_len": ServerArgs.max_model_len,
    "max_num_seqs": ServerArgs.max_num_seqs,
    "moe_backend": ServerArgs.moe_backend,
    "flashinfer_mxfp4_moe_precision": ServerArgs.flashinfer_mxfp4_moe_precision,
    "enforce_eager": ServerArgs.enforce_eager,
    "max_cudagraph_capture_size": ServerArgs.max_cudagraph_capture_size,
    "cudagraph_capture_sizes": ServerArgs.cudagraph_capture_sizes,
    "disable_prefill_graph": ServerArgs.disable_prefill_graph,
    "prefill_graph_max_tokens": ServerArgs.prefill_graph_max_tokens,
    "mamba_track_interval": ServerArgs.mamba_track_interval,
    "all2all_backend": ServerArgs.all2all_backend,
}


def global_server_args_dict_update(server_args: ServerArgs):
    global_server_args_dict.update(
        {
            "attention_backend": server_args.attention_backend,
            "sampling_backend": server_args.sampling_backend,
            "attention_use_fp4_indexer_cache": server_args.attention_use_fp4_indexer_cache,
            "disable_deepseek_v4_fast_mhc": server_args.disable_deepseek_v4_fast_mhc,
            "deepseek_v4_mega_moe_max_num_tokens": server_args.deepseek_v4_mega_moe_max_num_tokens,
            "deepseek_v4_indexer_prefill_max_logits_mb": server_args.deepseek_v4_indexer_prefill_max_logits_mb,
            "triton_attention_reduce_in_fp32": server_args.triton_attention_reduce_in_fp32,
            "kv_cache_dtype": server_args.kv_cache_dtype,
            "enable_nan_detection": server_args.enable_nan_detection,
            "enable_p2p_check": server_args.enable_p2p_check,
            "mapping": server_args.mapping,
            "force_deterministic_rsag": server_args.force_deterministic_rsag,
            "low_latency_max_num_tokens_per_gpu": server_args.low_latency_max_num_tokens_per_gpu,
            "device": server_args.device,
            "draft_model_path_use_base": server_args.draft_model_path_use_base,
            "speculative_algorithm": server_args.speculative_algorithm,
            "disable_pdl": server_args.disable_pdl,
            "enable_prefix_caching": server_args.enable_prefix_caching,
            "mla_disable_ragged": server_args.mla_disable_ragged,
            "chunked_prefill_size": server_args.chunked_prefill_size,
            "mla_chunk_multiplier": server_args.mla_chunk_multiplier,
            "ep_num_redundant_experts": server_args.ep_num_redundant_experts,
            "ep_dispatch_algorithm": server_args.ep_dispatch_algorithm,
            "enable_eplb": server_args.enable_eplb,
            "mm_mode": server_args.mm_mode,
            "comm_fusion_max_num_tokens": server_args.comm_fusion_max_num_tokens,
            "enable_allreduce_fusion": server_args.enable_allreduce_fusion,
            "max_prefill_tokens": server_args.max_prefill_tokens,
            "max_model_len": server_args.max_model_len,
            "max_num_seqs": server_args.max_num_seqs,
            "moe_backend": server_args.moe_backend,
            "flashinfer_mxfp4_moe_precision": server_args.flashinfer_mxfp4_moe_precision,
            "enforce_eager": server_args.enforce_eager,
            "max_cudagraph_capture_size": server_args.max_cudagraph_capture_size,
            "cudagraph_capture_sizes": server_args.cudagraph_capture_sizes,
            "disable_prefill_graph": server_args.disable_prefill_graph,
            "prefill_graph_max_tokens": server_args.prefill_graph_max_tokens,
            "all2all_backend": server_args.all2all_backend,
        }
    )
    pdl_enabled.cache_clear()


class EnvField:
    def __init__(self, default: Any):
        self.default = default
        #  we use None to indicate whether the value is set or not
        # If the value is manually set to None, we need mark it as _set_to_none.
        # Always use clear() to reset the value, which leads to the default fallback.
        self._set_to_none = False

    def __set_name__(self, owner, name):
        self.name = name

    def parse(self, value: str) -> Any:
        raise NotImplementedError()

    def get(self) -> Any:
        value = os.getenv(self.name)
        if self._set_to_none:
            assert value is None
            return None

        if value is None:
            return self.default

        try:
            return self.parse(value)
        except ValueError as e:
            warnings.warn(
                f'Invalid value for {self.name}: {e}, using default "{self.default}"'
            )
            return self.default

    def is_set(self):
        #  If None is manually set, it is considered as set.
        return self.name in os.environ or self._set_to_none

    def get_set_value_or(self, or_value: Any):
        #  Ugly usage, but only way to get custom default value.
        return self.get() if self.is_set() else or_value

    def set(self, value: Any):
        if value is None:
            self._set_to_none = True
            os.environ.pop(self.name, None)
        else:
            self._set_to_none = False
            os.environ[self.name] = str(value)

    @contextmanager
    def override(self, value: Any):
        backup_present = self.name in os.environ
        backup_value = os.environ.get(self.name)
        backup_set_to_none = self._set_to_none
        self.set(value)
        yield
        if backup_present:
            os.environ[self.name] = backup_value
        else:
            os.environ.pop(self.name, None)
        self._set_to_none = backup_set_to_none

    def clear(self):
        os.environ.pop(self.name, None)
        self._set_to_none = False

    @property
    def value(self):
        return self.get()

    def __bool__(self):
        raise RuntimeError(
            "Please use `envs.YOUR_FLAG.get()` instead of `envs.YOUR_FLAG`"
        )

    def __len__(self):
        raise RuntimeError(
            "Please use `envs.YOUR_FLAG.get()` instead of `envs.YOUR_FLAG`"
        )


class EnvStr(EnvField):
    def parse(self, value: str) -> str:
        return value


class EnvBool(EnvField):
    def parse(self, value: str) -> bool:
        value = value.lower()
        if value in ["true", "1", "yes", "y"]:
            return True
        if value in ["false", "0", "no", "n"]:
            return False
        raise ValueError(f'"{value}" is not a valid boolean value')


class EnvInt(EnvField):
    def parse(self, value: str) -> int:
        try:
            return int(value)
        except ValueError:
            raise ValueError(f'"{value}" is not a valid integer value')


class EnvFloat(EnvField):
    def parse(self, value: str) -> float:
        try:
            return float(value)
        except ValueError:
            raise ValueError(f'"{value}" is not a valid float value')


class Envs:
    # fmt: off

    # Model download
    TOKENSPEED_USE_MODELSCOPE = EnvBool(False)

    # Test and debug
    TOKENSPEED_CUDA_COREDUMP = EnvBool(False)
    TOKENSPEED_CUDA_COREDUMP_DIR = EnvStr("/tmp/tokenspeed_cuda_coredumps")
    TOKENSPEED_PROFILE_WITH_STACK = EnvBool(True)
    TOKENSPEED_TEST_REQUEST_TIME_STATS = EnvBool(False)
    TOKENSPEED_PROFILER_DIR = EnvStr("/tmp")
    TOKENSPEED_CI_SMALL_KV_SIZE = EnvInt(-1)
    TOKENSPEED_NVTX = EnvBool(False)

    # Scheduler
    TOKENSPEED_BLOCK_NONZERO_RANK_CHILDREN = EnvBool(True)

    # Mooncake
    TOKENSPEED_KVSTORE_MOONCAKE_CONFIG_PATH = EnvStr(None)
    MOONCAKE_MASTER = EnvStr(None)
    MOONCAKE_LOCAL_HOSTNAME = EnvStr("localhost")
    MOONCAKE_TE_META_DATA_SERVER = EnvStr("P2PHANDSHAKE")
    MOONCAKE_GLOBAL_SEGMENT_SIZE = EnvStr("4gb")
    MOONCAKE_PROTOCOL = EnvStr("tcp")
    MOONCAKE_DEVICE = EnvStr("")
    MOONCAKE_MASTER_METRICS_PORT = EnvInt(9003)
    MOONCAKE_CHECK_SERVER = EnvBool(False)
    TOKENSPEED_DISAGGREGATION_FAILED_SESSION_TTL = EnvInt(30)
    TOKENSPEED_DISAGGREGATION_HEARTBEAT_INTERVAL = EnvFloat(5.0)
    TOKENSPEED_DISAGGREGATION_HEARTBEAT_MAX_FAILURE = EnvInt(2)
    TOKENSPEED_DISAGGREGATION_QUEUE_SIZE = EnvInt(4)
    TOKENSPEED_DISAGGREGATION_THREAD_POOL_SIZE = EnvInt(-1)
    TOKENSPEED_DISAGGREGATION_BOOTSTRAP_TIMEOUT = EnvInt(120)
    TOKENSPEED_DISAGGREGATION_WAITING_TIMEOUT = EnvInt(300)
    TOKENSPEED_PD_LAYERWISE_DEBUG = EnvBool(False)

    # Quantization
    TOKENSPEED_NVFP4_GEMM_SWIGLU_NVFP4_QUANT = EnvBool(True)
    TOKENSPEED_MINIMAX_AR_USE_TRITON = EnvBool(False)

    # EPLB
    TOKENSPEED_EXPERT_DISTRIBUTION_RECORDER_DIR = EnvStr("/tmp")

    # Runtime behavior
    TOKENSPEED_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN = EnvBool(False)
    TOKENSPEED_DETOKENIZER_MAX_STATES = EnvInt(1 << 16)
    TOKENSPEED_FORCE_FAKE_FULL_NVLINK = EnvBool(False)
    TOKENSPEED_HEALTH_CHECK_TIMEOUT = EnvInt(20)
    TOKENSPEED_HOST_IP = EnvStr("")
    TOKENSPEED_LOGGING_CONFIG_PATH = EnvStr(None)
    TOKENSPEED_MAMBA_SSM_DTYPE = EnvStr("float32")
    TOKENSPEED_MODEL_REDIRECT_PATH = EnvStr(None)
    TOKENSPEED_MOE_PADDING = EnvBool(False)
    TOKENSPEED_MOE_CONFIG_DIR = EnvStr(None)
    TOKENSPEED_ENABLE_TORCH_INFERENCE_MODE = EnvBool(True)
    TOKENSPEED_NUMA_AWARE_WORKER_AFFINITY = EnvBool(True)

    # fmt: on


envs = Envs()
