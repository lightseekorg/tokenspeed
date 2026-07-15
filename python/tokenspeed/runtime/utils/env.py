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

from tokenspeed.runtime.utils.pdl import pdl_enabled
from tokenspeed.runtime.utils.server_args import ServerArgs

global_server_args_dict: dict = {
    "attention_backend": ServerArgs.attention_backend,
    "sampling_backend": ServerArgs.sampling_backend,
    "attention_use_fp4_indexer_cache": ServerArgs.attention_use_fp4_indexer_cache,
    "deepseek_v4_mega_moe_max_num_tokens": ServerArgs.deepseek_v4_mega_moe_max_num_tokens,
    "deepseek_v4_indexer_prefill_max_logits_mb": ServerArgs.deepseek_v4_indexer_prefill_max_logits_mb,
    "deepseek_v4_prefill_chunk_size": ServerArgs.deepseek_v4_prefill_chunk_size,
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
    "mm_attention_backend": ServerArgs.mm_attention_backend,
    "comm_fusion_max_num_tokens": ServerArgs.comm_fusion_max_num_tokens,
    "enable_allreduce_fusion": ServerArgs.enable_allreduce_fusion,
    "max_prefill_tokens": ServerArgs.max_prefill_tokens,
    "max_model_len": ServerArgs.max_model_len,
    "max_num_seqs": ServerArgs.max_num_seqs,
    "mamba_ssm_dtype": ServerArgs.mamba_ssm_dtype,
    "moe_backend": ServerArgs.moe_backend,
    "enforce_eager": ServerArgs.enforce_eager,
    "max_cudagraph_capture_size": ServerArgs.max_cudagraph_capture_size,
    "cudagraph_capture_sizes": ServerArgs.cudagraph_capture_sizes,
    "disable_prefill_graph": ServerArgs.disable_prefill_graph,
    "prefill_graph_max_tokens": ServerArgs.prefill_graph_max_tokens,
    "mamba_track_interval": ServerArgs.mamba_track_interval,
    "all2all_backend": ServerArgs.all2all_backend,
}


def global_server_args_dict_update(server_args: ServerArgs):

    # Export the PDL kill-switch: tokenspeed_kernel cannot import runtime modules.
    if server_args.disable_pdl:
        os.environ["TOKENSPEED_DISABLE_PDL"] = "1"
    global_server_args_dict.update(
        {
            "attention_backend": server_args.attention_backend,
            "sampling_backend": server_args.sampling_backend,
            "attention_use_fp4_indexer_cache": server_args.attention_use_fp4_indexer_cache,
            "deepseek_v4_mega_moe_max_num_tokens": server_args.deepseek_v4_mega_moe_max_num_tokens,
            "deepseek_v4_indexer_prefill_max_logits_mb": server_args.deepseek_v4_indexer_prefill_max_logits_mb,
            "deepseek_v4_prefill_chunk_size": server_args.deepseek_v4_prefill_chunk_size,
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
            "speculative_num_draft_tokens": server_args.speculative_num_draft_tokens,
            "disable_pdl": server_args.disable_pdl,
            "enable_prefix_caching": server_args.enable_prefix_caching,
            "mla_disable_ragged": server_args.mla_disable_ragged,
            "chunked_prefill_size": server_args.chunked_prefill_size,
            "mla_chunk_multiplier": server_args.mla_chunk_multiplier,
            "ep_num_redundant_experts": server_args.ep_num_redundant_experts,
            "ep_dispatch_algorithm": server_args.ep_dispatch_algorithm,
            "enable_eplb": server_args.enable_eplb,
            "mm_attention_backend": server_args.mm_attention_backend,
            "comm_fusion_max_num_tokens": server_args.comm_fusion_max_num_tokens,
            "enable_allreduce_fusion": server_args.enable_allreduce_fusion,
            "max_prefill_tokens": server_args.max_prefill_tokens,
            "max_model_len": server_args.max_model_len,
            "max_num_seqs": server_args.max_num_seqs,
            "mamba_ssm_dtype": server_args.mamba_ssm_dtype,
            "moe_backend": server_args.moe_backend,
            "enforce_eager": server_args.enforce_eager,
            "max_cudagraph_capture_size": server_args.max_cudagraph_capture_size,
            "cudagraph_capture_sizes": server_args.cudagraph_capture_sizes,
            "disable_prefill_graph": server_args.disable_prefill_graph,
            "prefill_graph_max_tokens": server_args.prefill_graph_max_tokens,
            "all2all_backend": server_args.all2all_backend,
        }
    )
    pdl_enabled.cache_clear()
