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

"""The arguments of the server."""

import argparse
import dataclasses
import json
import os
import random
from typing import Literal

from tokenspeed_kernel.platform import current_platform

from tokenspeed.runtime.distributed.mapping import Mapping, _resolve_parallelism_sizes
from tokenspeed.runtime.layers.attention.linear.chunk_delta_h import (
    CHUNK_SIZE as FLA_CHUNK_SIZE,
)
from tokenspeed.runtime.utils import (
    get_amdgpu_memory_capacity,
    get_colorful_logger,
    get_nvgpu_memory_capacity,
    is_valid_ipv6_address,
    maybe_model_redirect,
    nullable_str,
)
from tokenspeed.runtime.utils.network import is_port_available

logger = get_colorful_logger(__name__)

ENABLE_CP = os.environ.get("ENABLE_CP", "false").lower() in ("true", "1")


def str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value!r}")


@dataclasses.dataclass
class ServerArgs:
    # Model and tokenizer
    model: str
    tokenizer: str | None = None
    tokenizer_mode: str = "auto"
    skip_tokenizer_init: bool = False
    load_format: str = "auto"
    trust_remote_code: bool = True
    dtype: str = "auto"
    kv_cache_dtype: str = "auto"
    kv_cache_quant_method: str = "none"
    quantization: str | None = None
    quantization_param_path: nullable_str = None
    max_model_len: int | None = None
    device: str = "cuda"
    served_model_name: str | None = None
    revision: str | None = None

    # Port for the HTTP server
    host: str = "127.0.0.1"
    port: int = 8000

    # Memory and scheduling
    gpu_memory_utilization: float | None = None
    max_num_seqs: int | None = None
    max_total_tokens: int | None = None
    chunked_prefill_size: int | None = None
    max_prefill_tokens: int = 8192
    block_size: int = 64
    # special kv cache
    mamba_ssm_dtype: str = "float32"
    mamba_track_interval: int = 256
    max_mamba_cache_size: int | None = None
    mamba_full_memory_ratio: float = 0.9

    # Other runtime options
    stream_interval: int = 1
    stream_output: bool = False
    # Inline detokenization is the only supported path and is intentionally
    # not configurable from the CLI.
    enable_inline_detokenizer: bool = True
    seed: int | None = None
    distributed_timeout_seconds: int | None = None
    download_dir: str | None = None
    # Used for customizing extensible models
    ext_yaml: str | None = None
    base_gpu_id: int = 0
    gpu_id_step: int = 1

    # Logging
    log_level: str = "info"
    log_level_http: str | None = None
    enable_log_requests: bool = False
    log_requests_level: int = 0
    enable_metrics: bool = False
    decode_log_interval: int = 40
    metrics_reporters: list[str] | None = None
    app_key: str | None = None

    # API related
    api_key: str | None = None
    enable_cache_report: bool = False
    kv_events_config: str | None = None

    # Data parallelism
    data_parallel_size: int | None = None
    load_balance_method: str = "shortest_queue"
    load_watch_interval: float = 0.02

    # Expert parallelism
    ep_size: int = 1
    init_expert_location: str = "trivial"
    ep_num_redundant_experts: int = 0
    ep_dispatch_algorithm: (
        Literal[
            "static",
            "dynamic",
            "fake",
            "static_with_zero_expert",
            "dynamic_with_zero_expert",
        ]
        | None
    ) = None
    eplb_algorithm: str = "auto"
    expert_distribution_recorder_mode: (
        Literal["stat", "stat_approx", "per_pass", "per_token"] | None
    ) = None
    expert_distribution_recorder_buffer_size: int | None = None
    enable_expert_distribution_metrics: bool = False
    enable_eplb: bool = False

    # MoE backend
    moe_backend: str = "auto"
    draft_moe_backend: str | None = None
    all2all_backend: str = "none"
    deepep_mode: Literal["auto", "normal", "low_latency"] = "auto"
    flashinfer_mxfp4_moe_precision: str = "default"
    disable_flashinfer_cutlass_moe_fp4_allgather: bool = False

    # KVStore
    enable_kvstore: bool = False
    kvstore_ratio: float = 2.0
    kvstore_size: int = 0
    kvstore_io_backend: str = "kernel"
    kvstore_mem_layout: str = "layer_first"
    kvstore_storage_backend: str | None = None
    kvstore_storage_backend_extra_config: str | None = None
    enable_mla_l1_5_cache: bool = False

    # Multi-node distributed serving
    dist_init_addr: str | None = None
    nnodes: int = 1
    node_rank: int = 0

    # Hugging Face model config overrides in JSON
    hf_overrides: str = "{}"
    preferred_sampling_params: str | None = None

    # Kernel backend
    attention_backend: str | None = None
    drafter_attention_backend: str | None = None
    sampling_backend: str | None = None
    attention_use_fp4_indexer_cache: bool | None = None
    use_trtllm_ragged_deepseek_prefill: bool | None = None

    # DeepSeek V4
    disable_deepseek_v4_fast_mhc: bool = False
    deepseek_v4_mega_moe_max_num_tokens: int = 0
    deepseek_v4_indexer_prefill_max_logits_mb: int = 512
    deepseek_v4_prefill_chunk_size: int = 4

    # Grammar backend
    grammar_backend: str = "none"
    grammar_compile_timeout_secs: float = 30.0
    grammar_compile_max_retries: int = 2
    disable_any_whitespace: bool = False
    # Force the synchronous eager grammar fallback even on CUDA. Useful
    # for parity-testing against the captured-grammar path (output should
    # match; throughput will be lower since the sync stalls every step).
    disable_capturable_grammar: bool = False

    # Speculative decoding
    draft_model_path_use_base: bool | None = False
    speculative_config: str | None = None
    speculative_algorithm: str | None = None
    speculative_draft_model_path: str | None = None
    speculative_draft_model_quantization: str | None = "unquant"
    speculative_num_steps: int = 3
    speculative_eagle_topk: int = 1
    speculative_num_draft_tokens: int | None = None
    eagle3_layers_to_capture: str | None = None
    # Logprob support flags — all OFF by default. Enabling extends the
    # captured CUDA-graph footprint; requests asking for logprobs on a
    # server started without the matching flag will receive empty logprobs.
    enable_output_logprobs: bool = False

    # LoRA adapter serving
    enable_lora: bool = False
    # Maximum number of non-pinned LoRA adapters resident in GPU memory at
    # once.  Adapters beyond this cap are LRU-evicted to the CPU pool.
    max_loras: int = 4
    # Maximum LoRA rank supported (caps adapter loading; larger = more GPU memory).
    max_lora_rank: int = 64
    # Maximum number of LoRA adapters cached in CPU pinned memory.  When
    # an adapter is evicted from this pool it falls back to its disk path
    # (assumed durable) and is reloaded on next use.  ``None`` ⇒ default
    # to ``4 * max_loras``.
    max_loras_cpu: int | None = None
    # Scheduler-side LoRA scheduling policy.  ``"lru"`` (default) just
    # relies on the manager's LRU; ``"admission"`` (future) gates batches
    # that don't fit in GPU; ``"pack"`` (future) sorts the queue to reuse
    # resident adapters.
    lora_scheduling_policy: str = "lru"

    # Runtime options
    disable_pdl: bool = False
    enable_prefix_caching: bool = True
    disable_kvstore: bool = False
    enforce_eager: bool = False
    disable_cuda_graph_padding: bool = False
    enable_cudagraph_gc: bool = False
    enable_nccl_nvls: bool = False
    enable_symm_mem: bool = False
    disable_custom_all_reduce: bool = False
    disable_overlap_schedule: bool = False
    disable_tf32: bool = False
    force_deterministic_rsag: bool = False
    disable_sampling_tp_sync: bool = False
    low_latency_max_num_tokens_per_gpu: int = 256
    max_cudagraph_capture_size: int | None = None
    disable_prefill_graph: bool | None = False
    prefill_graph_max_tokens: int | None = 128
    cudagraph_capture_sizes: list[int] | None = None
    enable_nan_detection: bool = False
    enable_nvtx: bool = False
    enable_p2p_check: bool = False
    triton_attention_reduce_in_fp32: bool = False
    delete_ckpt_after_loading: bool = False
    weight_loader_prefetch_checkpoints: bool = False
    weight_loader_prefetch_num_threads: int = 4
    enable_memory_saver: bool = False
    enable_custom_logit_processor: bool = False
    mla_disable_ragged: bool = False
    warmups: str | None = None

    # parallel strategy
    nprocs_per_node: int | None = None
    world_size: int | None = None
    attn_tp_size: int | None = None
    dense_tp_size: int | None = None
    moe_tp_size: int | None = None
    mapping: Mapping | None = None

    mla_chunk_multiplier: int = 4
    mm_mode: str = "none"

    # For PD disaggregation: can be "null" (not disaggregated), "prefill" (prefill-only), or "decode" (decode-only)
    disaggregation_mode: str = "null"
    disaggregation_bootstrap_port: int = 8998
    disaggregation_transfer_backend: str = "mooncake"
    disaggregation_ib_device: str | None = None
    disaggregation_layerwise_interval: int = 1
    pdlb_url: str | None = None

    skip_server_warmup: bool = False

    # For communication + norm fusion
    comm_fusion_max_num_tokens: int = 2048
    enable_allreduce_fusion: bool = False

    enable_expert_parallel: bool = False

    @property
    def mamba_cache_chunk_size(self) -> int:
        return max(FLA_CHUNK_SIZE, self.block_size)

    def __post_init__(self):
        self.resolve_basic_defaults()
        self.resolve_parallelism()
        self.resolve_memory_and_scheduling()
        self.resolve_kernel_backends()
        self.resolve_cache()
        self.resolve_speculative_decoding()
        self.resolve_communication()
        self.resolve_disaggregation()
        self.validate()

    def resolve_basic_defaults(self):
        self.model = maybe_model_redirect(self.model)

        if self.kv_cache_dtype == "fp8":
            self.kv_cache_dtype = "fp8_e4m3"

        self.resolve_config_aliases()

        # Set missing default values
        if self.tokenizer is None:
            self.tokenizer = self.model

        if self.served_model_name is None:
            self.served_model_name = self.model

        if self.seed is None:
            self.seed = random.randint(0, 1 << 30)

    def resolve_config_aliases(self):
        if self.use_trtllm_ragged_deepseek_prefill is not None:
            self.mla_disable_ragged = not self.use_trtllm_ragged_deepseek_prefill

        if self.speculative_config is not None:
            try:
                config = json.loads(self.speculative_config)
            except json.JSONDecodeError as exc:
                raise ValueError("--speculative-config must be valid JSON") from exc

            if not isinstance(config, dict):
                raise ValueError("--speculative-config must be a JSON object")

            method = config.get("method")
            if method is not None and self.speculative_algorithm is None:
                self.speculative_algorithm = str(method).upper()

            draft_model = config.get("model")
            if draft_model is not None and self.speculative_draft_model_path is None:
                self.speculative_draft_model_path = str(draft_model)

            num_speculative_tokens = config.get("num_speculative_tokens")
            if num_speculative_tokens is not None:
                self.speculative_num_steps = int(num_speculative_tokens)

        if self.speculative_num_draft_tokens is None:
            self.speculative_num_draft_tokens = self.speculative_num_steps + 1

    def resolve_memory_and_scheduling(self):
        if current_platform().is_amd:
            gpu_mem = get_amdgpu_memory_capacity()
        elif current_platform().is_nvidia:
            gpu_mem = get_nvgpu_memory_capacity()
        else:
            # GPU memory is not known yet or no GPU is available.
            gpu_mem = None

        # Set GPU memory utilization, which depends on the tensor parallelism size.
        if self.gpu_memory_utilization is None:
            if self.mapping.world_size >= 16:
                self.gpu_memory_utilization = 0.79
            elif self.mapping.world_size >= 8:
                self.gpu_memory_utilization = 0.81
            elif self.mapping.world_size >= 4:
                self.gpu_memory_utilization = 0.85
            elif self.mapping.world_size >= 2:
                self.gpu_memory_utilization = 0.87
            else:
                self.gpu_memory_utilization = 0.88

        # Set the chunked prefill token budget.
        if self.chunked_prefill_size is None:
            self.chunked_prefill_size = 8192

        # Set CUDA graph max capture size.
        if self.max_cudagraph_capture_size is None:
            # Based on detailed statistics, when serving TP1/TP2 models on lower-end GPUs with HBM<25G, you can either disable CUDA graph or set max_cudagraph_capture_size to a very small value to reduce graph memory overhead, with almost no impact on performance. TP4/TP8 serving still needs CUDA graph for high performance, and 80 is enough for lower-end GPUs.
            if gpu_mem is not None and gpu_mem < 25_000:
                if self.mapping.world_size < 4:
                    self.max_cudagraph_capture_size = 8
                else:
                    self.max_cudagraph_capture_size = 80
            elif self.speculative_algorithm:
                self.max_cudagraph_capture_size = 80
            else:
                self.max_cudagraph_capture_size = 160

        # Set max number of sequences.
        if self.max_num_seqs is None:
            if self.speculative_algorithm:
                self.max_num_seqs = 80
            else:
                self.max_num_seqs = 160

    def resolve_kernel_backends(self):
        # Choose kernel backends
        # attention_backend default is NOT set here — deferred to
        # AttnInitializer.modify_args where both hardware and model arch are known.

        if self.sampling_backend is None:
            # ``flashinfer`` is the only built-in backend that respects per-request
            # ``temperature`` / ``top_p`` / ``top_k``. ``greedy`` is argmax-only
            # (see ``GreedySamplingBackend.sample``: *"sampling_info is ignored
            # for single-step (always argmax)"*) — fast for hand-tuned greedy
            # decoding but silently wrong for any serving deployment where
            # requests carry sampling params, since the model collapses into
            # repetition-mode loops within a few hundred steps. Default to the
            # sampling-respecting backend on NVIDIA where flashinfer is
            # available, fall back to greedy elsewhere; users can still opt
            # into greedy explicitly via ``--sampling-backend greedy``.
            if current_platform().is_nvidia:
                self.sampling_backend = "flashinfer"
            else:
                self.sampling_backend = "greedy"

    def resolve_parallelism(self):
        world_size = self.world_size
        nprocs_per_node = self.nprocs_per_node
        nnodes = 1 if self.nnodes is None else self.nnodes

        attn_tp_size = self.attn_tp_size
        attn_dp_size = self.data_parallel_size

        # ``ENABLE_CP`` interprets attention TP size as CP size.
        attn_cp_size = 1
        if ENABLE_CP:
            attn_cp_size, attn_tp_size = attn_tp_size, 1

        if world_size is None:
            world_size = 1
            if attn_tp_size is not None:
                world_size *= attn_tp_size
            if attn_cp_size is not None:
                world_size *= attn_cp_size
            if attn_dp_size is not None:
                world_size *= attn_dp_size
            logger.info(
                "Inferred world_size (%s) from attn_tp_size (%s) x attn_cp_size (%s) x attn_dp_size (%s)",
                world_size,
                attn_tp_size,
                attn_cp_size,
                attn_dp_size,
            )
        else:
            logger.info("Specified world_size (%s)", world_size)

        attn_tp_size, attn_cp_size, attn_dp_size = _resolve_parallelism_sizes(
            world_size, attn_tp_size, attn_cp_size, attn_dp_size
        )

        # Dense layers still default to full TP participation when no
        # dedicated dense_tp_size is provided.
        dense_tp_size = self.dense_tp_size
        if self.dense_tp_size is None:
            # dense always do tp now.
            dense_tp_size = world_size
        dense_dp_size = None

        # --enable-expert-parallel auto-sets ep_size = world_size
        if self.enable_expert_parallel and self.ep_size == 1:
            self.ep_size = world_size
            logger.info("--enable-expert-parallel: auto-setting ep_size=%s", world_size)

        # MoE parallel sizes default to consuming the full world size unless
        # the user overrides them explicitly.
        moe_ep_size = 1 if self.ep_size is None else self.ep_size
        moe_tp_size = (
            world_size // moe_ep_size if self.moe_tp_size is None else self.moe_tp_size
        )
        moe_dp_size = None

        self.mapping = Mapping(
            world_size=world_size,
            attn_tp_size=attn_tp_size,
            attn_cp_size=attn_cp_size,
            attn_dp_size=attn_dp_size,
            dense_tp_size=dense_tp_size,
            dense_dp_size=dense_dp_size,
            moe_tp_size=moe_tp_size,
            moe_ep_size=moe_ep_size,
            moe_dp_size=moe_dp_size,
            nprocs_per_node=nprocs_per_node,
            nnodes=nnodes,
            base_gpu_id=self.base_gpu_id,
            gpu_id_step=self.gpu_id_step,
        )

        # Impl constraints:
        if self.mapping.moe.has_tp and self.mapping.moe.has_ep:
            raise ValueError("MoE TP and EP cannot be both > 1")

        logger.info("Parallelism configuration:\n%s", self.mapping)

    def resolve_cache(self):
        # Handle KVStore settings.
        self._handle_kvstore()
        self.validate_cache_options()

    def resolve_speculative_decoding(self):
        # Keep drafter backend consistent with the main model unless explicitly set.
        if (
            self.speculative_algorithm is not None
            and self.drafter_attention_backend is None
        ):
            self.drafter_attention_backend = self.attention_backend

        if (
            self.speculative_algorithm == "MTP"
            and self.speculative_draft_model_path is None
        ):
            self.draft_model_path_use_base = True

        if self.draft_model_path_use_base:
            self.speculative_draft_model_path = self.model

        if self.speculative_draft_model_path == self.model:
            self.draft_model_path_use_base = True

        if self.speculative_draft_model_quantization == "unquant":
            self.speculative_draft_model_quantization = None

        if self.eagle3_layers_to_capture is not None:
            self.eagle3_layers_to_capture = [
                int(x) for x in self.eagle3_layers_to_capture.split(",")
            ]

    def resolve_communication(self):
        # Auto-enable allreduce fusion on supported single-node TP configurations.
        platform = current_platform()
        if (
            not self.enable_allreduce_fusion
            and (current_platform().is_hopper_plus or platform.is_amd)
            and self.mapping.nnodes == 1
            and self.mapping.has_attn_tp
            and not self.mapping.has_attn_dp
        ):
            self.enable_allreduce_fusion = True
            logger.info("Auto-enabled allreduce fusion")

        if self.mapping.attn.tp_size != self.mapping.dense.tp_size:
            self.comm_fusion_max_num_tokens = -1
            self.enable_allreduce_fusion = False
            logger.info(
                "allreduce is forbidden due to different attn_tp_size: %s and dense_tp_size: %s!",
                self.mapping.attn.tp_size,
                self.mapping.dense.tp_size,
            )

    def resolve_disaggregation(self):
        if self.enable_lora:
            # LoRA delta path is baked into the captured graph: the per-token
            # slot index buffer (LoraManager.weight_indices_buf) is bound at
            # capture and updated in place at replay, with slot 0 reserved as
            # a zero-delta no-adapter fallback.
            #
            # PDL stays disabled: the TVM-JIT RMSNorm kernel (rmsnorm_cute) is
            # compiled on first call with a fixed dtype and cannot handle the
            # bfloat16↔float32 casting that the LoRA bmm path performs.
            self.disable_pdl = True
            # Default the CPU pool to 4× the GPU pool so adapter swap-out
            # to disk is rare in steady state.
            if self.max_loras_cpu is None:
                self.max_loras_cpu = 4 * self.max_loras
            if self.max_loras_cpu < self.max_loras:
                raise ValueError(
                    f"max_loras_cpu ({self.max_loras_cpu}) must be ≥ "
                    f"max_loras ({self.max_loras}) — every GPU-resident "
                    "adapter must also fit in the CPU pool."
                )

        # PD disaggregation
        if self.disaggregation_mode == "prefill":
            self.enforce_eager = True
            logger.warning("CUDA graph is disabled for prefill server")
        elif self.disaggregation_mode == "decode":
            # Prefix caching stays configurable for decode servers.
            logger.info(
                "enable_prefix_caching=%r for decode server",
                self.enable_prefix_caching,
            )

        # Prefill graph disable logic is handled by AttnInitializer.modify_args
        # after the attention backend is resolved.

        if (
            self.disaggregation_mode == "prefill"
            and self.load_balance_method != "round_robin"
        ):
            assert (
                not self.mapping.has_attn_dp
            ), f"Not Supported when {self.disaggregation_mode=} {self.load_balance_method=} {self.mapping.attn.dp_size=}"

    def _handle_kvstore(self):
        if self.disaggregation_mode == "decode":
            self.enable_kvstore = False
            logger.info("Decode instance has set enable_kvstore to False!")
        elif not self.disable_kvstore:
            self.enable_kvstore = True

        if self.kvstore_storage_backend == "mooncake":
            if self.kvstore_mem_layout == "layer_first":
                self.kvstore_mem_layout = "page_first"
                logger.warning(
                    "Mooncake storage backend does not support layer_first layout, switching to %s layout",
                    self.kvstore_mem_layout,
                )

            if self.kvstore_io_backend == "direct":
                self.kvstore_io_backend = "kernel"
                logger.warning(
                    "Mooncake storage backend uses page_first layout, which requires kernel io backend"
                )

    def validate_cache_options(self):
        if self.enable_kvstore and not self.enable_prefix_caching:
            raise ValueError(
                "KVStore and disabled prefix caching are mutually exclusive "
                "and cannot be used at the same time. Please use only one of them."
            )

    def validate(self):
        if (
            self.max_num_seqs is not None
            and self.max_num_seqs < self.mapping.attn.dp_size
        ):
            raise ValueError(
                f"max_num_seqs must be >= attn_dp_size: {self.max_num_seqs=} < {self.mapping.attn.dp_size=}"
            )

        if self.mapping.has_attn_cp and self.max_num_seqs > 1:
            raise ValueError("CP attention is enabled but max_num_seqs > 1")

        if self.mapping.has_attn_dp:
            if self.chunked_prefill_size > self.max_prefill_tokens:
                raise ValueError(
                    f"chunked_prefill_size must be <= max_prefill_tokens: {self.chunked_prefill_size=} > {self.max_prefill_tokens=}"
                )

        if self.deepseek_v4_prefill_chunk_size <= 0:
            raise ValueError("deepseek_v4_prefill_chunk_size must be positive")

        if self.enable_eplb and (self.expert_distribution_recorder_mode is None):
            self.expert_distribution_recorder_mode = "stat"
            logger.info(
                "EPLB is enabled. The expert_distribution_recorder_mode is automatically set."
            )

        if (self.enable_eplb or (self.init_expert_location is not None)) and (
            self.ep_dispatch_algorithm is None
        ):
            self.ep_dispatch_algorithm = "static"
            logger.info(
                "EPLB is enabled or init_expert_location is provided. ep_dispatch_algorithm is configured."
            )

        from tokenspeed.runtime.utils.env import envs

        envs.TOKENSPEED_MAMBA_SSM_DTYPE.set(self.mamba_ssm_dtype)
        if not self.disable_pdl:
            os.environ.setdefault("TORCHINDUCTOR_ENABLE_PDL", "1")
            # Enable PDL for fused attention kernels.
            os.environ.setdefault("TRTLLM_ENABLE_PDL", "1")
        os.environ.setdefault("TLLM_LOG_LEVEL", "INFO")

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser):
        parser.allow_abbrev = False

        # Model and port args
        parser.add_argument(
            "model_path",
            nargs="?",
            metavar="model",
            default=None,
            help="The model name or path (positional argument). "
            "Equivalent to --model.",
        )
        parser.add_argument(
            "--model",
            "--model-path",
            metavar="MODEL",
            type=str,
            default=None,
            help="The path of the model weights. This can be a local folder or a Hugging Face repo ID.",
        )
        parser.add_argument(
            "--tokenizer",
            metavar="TOKENIZER",
            type=str,
            default=ServerArgs.tokenizer,
            help="The path of the tokenizer.",
        )
        parser.add_argument(
            "--host", type=str, default=ServerArgs.host, help="The host of the server."
        )
        parser.add_argument(
            "--port", type=int, default=ServerArgs.port, help="The port of the server."
        )
        parser.add_argument(
            "--tokenizer-mode",
            type=str,
            default=ServerArgs.tokenizer_mode,
            choices=["auto", "slow", "deepseek_v4"],
            help="Tokenizer mode. 'auto' will use the fast "
            "tokenizer and model-specific tokenizer hooks if available, "
            "'slow' will always use the slow tokenizer.",
        )
        parser.add_argument(
            "--skip-tokenizer-init",
            action=argparse.BooleanOptionalAction,
            default=ServerArgs.skip_tokenizer_init,
            help="If set, skip init tokenizer and pass input_ids in generate request",
        )
        parser.add_argument("--ext-yaml", type=str, default=None)
        parser.add_argument(
            "--load-format",
            type=str,
            default=ServerArgs.load_format,
            choices=[
                "auto",
                "pt",
                "safetensors",
                "npcache",
                "dummy",
                "extensible",
            ],
            help="The format of the model weights to load. "
            '"auto" will try to load the weights in the safetensors format '
            "and fall back to the pytorch bin format if safetensors format "
            "is not available. "
            '"pt" will load the weights in the pytorch bin format. '
            '"safetensors" will load the weights in the safetensors format. '
            '"npcache" will load the weights in pytorch format and store '
            "a numpy cache to speed up the loading. "
            '"dummy" will initialize the weights with random values.',
        )
        parser.add_argument(
            "--trust-remote-code",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="Whether or not to allow for custom models defined on the Hub in their own modeling files.",
        )
        parser.add_argument(
            "--dtype",
            type=str,
            default=ServerArgs.dtype,
            choices=["auto", "half", "float16", "bfloat16", "float", "float32"],
            help="Data type for model weights and activations.\n\n"
            '* "auto" will use FP16 precision for FP32 and FP16 models, and '
            "BF16 precision for BF16 models.\n"
            '* "half" for FP16. Recommended for AWQ quantization.\n'
            '* "float16" is the same as "half".\n'
            '* "bfloat16" for a balance between precision and range.\n'
            '* "float" is shorthand for FP32 precision.\n'
            '* "float32" for FP32 precision.',
        )
        parser.add_argument(
            "--kv-cache-dtype",
            type=str,
            default=ServerArgs.kv_cache_dtype,
            choices=["auto", "fp8", "fp8_e4m3"],
            help='Data type for kv cache storage. "auto" will use model data type. "fp8" is an alias for "fp8_e4m3".',
        )
        parser.add_argument(
            "--kv-cache-quant-method",
            type=str,
            default=ServerArgs.kv_cache_quant_method,
            choices=["none", "per_token_head"],
            help="kv cache quant method",
        )
        parser.add_argument(
            "--quantization",
            type=str,
            default=ServerArgs.quantization,
            choices=[
                "fp8",
                "nvfp4",
                "w8a8_fp8",
                "compressed-tensors",
            ],
            help="The quantization method.",
        )
        parser.add_argument(
            "--quantization-param-path",
            type=nullable_str,
            default=None,
            help="Path to the JSON file containing the KV cache "
            "scaling factors. This should generally be supplied, when "
            "KV cache dtype is FP8. Otherwise, KV cache scaling factors "
            "default to 1.0, which may cause accuracy issues. ",
        )
        parser.add_argument(
            "--max-model-len",
            metavar="MAX_MODEL_LEN",
            type=int,
            default=ServerArgs.max_model_len,
            help="The model's maximum context length. Defaults to None (will use the value from the model's config.json instead).",
        )
        parser.add_argument(
            "--device",
            type=str,
            default="cuda",
            choices=["cuda"],
            help="The device type.",
        )
        parser.add_argument(
            "--served-model-name",
            type=str,
            default=ServerArgs.served_model_name,
            help="Override the model name returned by the v1/models endpoint in OpenAI API server.",
        )
        parser.add_argument(
            "--revision",
            type=str,
            default=None,
            help="The specific model version to use. It can be a branch "
            "name, a tag name, or a commit id. If unspecified, will use "
            "the default version.",
        )
        # Memory and scheduling
        parser.add_argument(
            "--gpu-memory-utilization",
            metavar="GPU_MEMORY_UTILIZATION",
            type=float,
            default=ServerArgs.gpu_memory_utilization,
            help="The fraction of GPU memory to use for model weights and KV cache. Use a smaller value if you see out-of-memory errors.",
        )
        parser.add_argument(
            "--max-num-seqs",
            metavar="MAX_NUM_SEQS",
            type=int,
            default=ServerArgs.max_num_seqs,
            help="Maximum number of sequences to process concurrently.",
        )
        parser.add_argument(
            "--max-total-tokens",
            type=int,
            default=ServerArgs.max_total_tokens,
            help="The maximum number of tokens in the memory pool. If not specified, it will be automatically calculated based on the memory usage fraction. "
            "This overrides the automatically calculated token pool size.",
        )
        parser.add_argument(
            "--chunked-prefill-size",
            metavar="CHUNKED_PREFILL_SIZE",
            type=int,
            default=ServerArgs.chunked_prefill_size,
            help="Maximum number of tokens the scheduler may issue in a single iteration. Setting this to -1 disables chunked prefill.",
        )
        parser.add_argument(
            "--block-size",
            metavar="BLOCK_SIZE",
            type=int,
            default=ServerArgs.block_size,
        )

        # KVStore
        parser.add_argument(
            "--disable-kvstore",
            action="store_true",
            help="Disable KVStore",
        )
        parser.add_argument(
            "--kvstore-ratio",
            type=float,
            default=ServerArgs.kvstore_ratio,
            help="The ratio of the size of the KVStore host memory pool to the size of the device pool.",
        )
        parser.add_argument(
            "--kvstore-size",
            type=int,
            default=ServerArgs.kvstore_size,
            help="The size of the KVStore host memory pool in gigabytes, which will override kvstore_ratio if set.",
        )
        parser.add_argument(
            "--kvstore-io-backend",
            type=str,
            choices=["direct", "kernel"],
            default=ServerArgs.kvstore_io_backend,
            help="The IO backend for KVStore transfer between CPU and GPU.",
        )
        parser.add_argument(
            "--kvstore-mem-layout",
            type=str,
            choices=[
                "layer_first",
                "page_first",
                "page_head",
            ],
            default=ServerArgs.kvstore_mem_layout,
            help="The layout of the KVStore host memory pool.",
        )
        parser.add_argument(
            "--kvstore-storage-backend",
            type=str,
            choices=["mooncake"],
            default=ServerArgs.kvstore_storage_backend,
            help="The storage backend for KVStore. "
            "Built-in backends: mooncake. "
            "For dynamic backend, use --kvstore-storage-backend-extra-config to specify: "
            "backend_name (custom name), module_path (Python module path), class_name (backend class name).",
        )
        parser.add_argument(
            "--kvstore-storage-backend-extra-config",
            type=str,
            default=ServerArgs.kvstore_storage_backend_extra_config,
            help="A dictionary in JSON string format containing extra configuration for the storage backend.",
        )
        parser.add_argument(
            "--enable-mla-l1-5-cache",
            action="store_true",
            help="Enable MLA L1.5 cache in disaggregation paths.",
        )
        # Mamba Cache
        parser.add_argument(
            "--mamba-ssm-dtype",
            type=str,
            default=ServerArgs.mamba_ssm_dtype,
            choices=["float32", "bfloat16"],
            help="It is used to tune mamba ssm dtype",
        )
        parser.add_argument(
            "--mamba-track-interval",
            type=int,
            default=ServerArgs.mamba_track_interval,
            help="The interval to track the mamba state during decode.",
        )
        parser.add_argument(
            "--max-mamba-cache-size",
            type=int,
            default=ServerArgs.max_mamba_cache_size,
            help="The maximum number of Mamba cache chunks. If unset, the pool size is profiled from available memory.",
        )
        parser.add_argument(
            "--mamba-full-memory-ratio",
            type=float,
            default=ServerArgs.mamba_full_memory_ratio,
            help="Memory ratio used to split cache budget between Mamba state chunks and full-attention KV cache.",
        )

        parser.add_argument(
            "--max-prefill-tokens",
            metavar="MAX_PREFILL_TOKENS",
            type=int,
            default=ServerArgs.max_prefill_tokens,
            help=(
                "Maximum prefill-token budget used when chunked prefill is "
                "disabled. Per-iteration scheduling is controlled by "
                "--chunked-prefill-size."
            ),
        )
        # Other runtime options
        parser.add_argument(
            "--stream-interval",
            type=int,
            default=ServerArgs.stream_interval,
            help="The interval (or buffer size) for streaming in terms of the token length. A smaller value makes streaming smoother, while a larger value makes the throughput higher",
        )
        parser.add_argument(
            "--stream-output",
            action="store_true",
            help="Whether to output as a sequence of disjoint segments.",
        )
        parser.add_argument(
            "--seed",
            metavar="SEED",
            type=int,
            default=ServerArgs.seed,
            help="The random seed.",
        )
        parser.add_argument(
            "--distributed-timeout-seconds",
            metavar="DISTRIBUTED_TIMEOUT_SECONDS",
            type=int,
            default=ServerArgs.distributed_timeout_seconds,
            help="Set timeout for torch.distributed initialization.",
        )
        parser.add_argument(
            "--download-dir",
            type=str,
            default=ServerArgs.download_dir,
            help="Model download directory for huggingface.",
        )
        parser.add_argument(
            "--base-gpu-id",
            type=int,
            default=ServerArgs.base_gpu_id,
            help="The base GPU ID to start allocating GPUs from. Useful when running multiple instances on the same machine.",
        )
        parser.add_argument(
            "--gpu-id-step",
            type=int,
            default=ServerArgs.gpu_id_step,
            help="The delta between consecutive GPU IDs that are used. For example, setting it to 2 will use GPU 0,2,4,...",
        )

        # Logging
        parser.add_argument(
            "--log-level",
            type=str,
            default=ServerArgs.log_level,
            help="The logging level of all loggers.",
        )
        parser.add_argument(
            "--log-level-http",
            type=str,
            default=ServerArgs.log_level_http,
            help="The logging level of HTTP server. If not set, reuse --log-level by default.",
        )
        parser.add_argument(
            "--enable-log-requests",
            action=argparse.BooleanOptionalAction,
            default=ServerArgs.enable_log_requests,
            help="Log metadata, inputs, outputs of all requests. The verbosity is decided by --log-requests-level",
        )
        parser.add_argument(
            "--log-requests-level",
            type=int,
            default=0,
            help="0: Log metadata. 1. Log metadata and partial input/output. 2. Log every input/output.",
            choices=[0, 1, 2],
        )
        parser.add_argument(
            "--enable-metrics",
            action="store_true",
            help="Enable log metrics.",
        )
        parser.add_argument(
            "--metrics-reporters",
            action="append",
            choices=["prometheus"],
            default=["prometheus"],
            help="Select metrics reporter(can be specified multiple times)",
        )

        parser.add_argument(
            "--app-key",
            type=str,
            default=ServerArgs.app_key,
            help="Set app key of the server",
        )

        parser.add_argument(
            "--decode-log-interval",
            type=int,
            default=ServerArgs.decode_log_interval,
            help="The log interval of decode batch.",
        )
        # API related
        parser.add_argument(
            "--api-key",
            type=str,
            default=ServerArgs.api_key,
            help="Set API key of the server. It is also used in the OpenAI API compatible server.",
        )
        parser.add_argument(
            "--enable-cache-report",
            action="store_true",
            help="Return number of cached tokens in usage.prompt_tokens_details for each openai request.",
        )
        parser.add_argument(
            "--kv-events-config",
            type=str,
            default=ServerArgs.kv_events_config,
            help=(
                "JSON KV cache event publisher config. Set "
                "'enable_kv_cache_events': true and publisher 'zmq' to "
                "publish device prefix-cache mutations."
            ),
        )

        # Data parallelism
        parser.add_argument(
            "--data-parallel-size",
            metavar="DATA_PARALLEL_SIZE",
            type=int,
            default=ServerArgs.data_parallel_size,
            help="The data parallelism size. If not set, inferred from world_size and attn_tp_size.",
        )
        parser.add_argument(
            "--load-balance-method",
            type=str,
            default=ServerArgs.load_balance_method,
            help="The load balancing strategy for data parallelism.",
            choices=[
                "round_robin",
                "shortest_queue",
                "minimum_cache_usage",
            ],
        )
        parser.add_argument(
            "--load-watch-interval",
            type=float,
            default=ServerArgs.load_watch_interval,
            help="The interval of load watching in seconds.",
        )

        # Expert parallelism
        parser.add_argument(
            "--expert-parallel-size",
            "--ep-size",
            type=int,
            default=ServerArgs.ep_size,
            help="The expert parallelism size.",
        )
        parser.add_argument(
            "--init-expert-location",
            type=str,
            default=ServerArgs.init_expert_location,
            help="Initial location of EP experts.",
        )
        parser.add_argument(
            "--ep-num-redundant-experts",
            type=int,
            default=ServerArgs.ep_num_redundant_experts,
            help="Allocate this number of redundant experts in expert parallel.",
        )
        parser.add_argument(
            "--ep-dispatch-algorithm",
            type=str,
            default=ServerArgs.ep_dispatch_algorithm,
            help="The algorithm to choose ranks for redundant experts in expert parallel.",
        )
        parser.add_argument(
            "--eplb-algorithm",
            type=str,
            default=ServerArgs.eplb_algorithm,
            help="Chosen EPLB algorithm",
        )
        parser.add_argument(
            "--expert-distribution-recorder-mode",
            type=str,
            default=ServerArgs.expert_distribution_recorder_mode,
            help="Mode of expert distribution recorder.",
        )
        parser.add_argument(
            "--expert-distribution-recorder-buffer-size",
            type=int,
            default=ServerArgs.expert_distribution_recorder_buffer_size,
            help="Circular buffer size of expert distribution recorder. Set to -1 to denote infinite buffer.",
        )
        parser.add_argument(
            "--enable-expert-distribution-metrics",
            action="store_true",
            help="Enable logging metrics for expert balancedness",
        )
        parser.add_argument(
            "--enable-eplb",
            action="store_true",
            help="Enable EPLB algorithm",
        )
        parser.add_argument(
            "--moe-backend",
            type=str,
            default=ServerArgs.moe_backend,
            help="MoE runner backend: auto, triton, triton_kernel, flashinfer_mxfp4, marlin, etc.",
        )
        parser.add_argument(
            "--draft-moe-backend",
            type=str,
            default=ServerArgs.draft_moe_backend,
            help="MoE runner backend for the draft model in speculative decoding. "
            "If not set, defaults to --moe-backend.",
        )
        parser.add_argument(
            "--flashinfer-mxfp4-moe-precision",
            type=str,
            choices=["default", "bf16"],
            default=ServerArgs.flashinfer_mxfp4_moe_precision,
            help="Computation precision of flashinfer mxfp4 moe.",
        )
        parser.add_argument(
            "--all2all-backend",
            metavar="ALL2ALL_BACKEND",
            type=str,
            default=ServerArgs.all2all_backend,
            help="MoE all-to-all backend: none, deepep, etc.",
        )
        parser.add_argument(
            "--deepep-mode",
            type=str,
            choices=["normal", "low_latency", "auto"],
            default=ServerArgs.deepep_mode,
            help="Select the mode when enable DeepEP MoE, could be `normal`, `low_latency` or `auto`. Default is `auto`, which means `low_latency` for decode batch and `normal` for prefill batch.",
        )
        parser.add_argument(
            "--disable-flashinfer-cutlass-moe-fp4-allgather",
            action="store_true",
            help="Disable flashinfer cutlass MoE FP4 allgather.",
        )

        # Multi-node distributed serving
        parser.add_argument(
            "--dist-init-addr",
            type=str,
            help="The host address for initializing distributed backend (e.g., `192.168.0.2:25000`).",
        )
        parser.add_argument(
            "--nnodes", type=int, default=ServerArgs.nnodes, help="The number of nodes."
        )
        parser.add_argument(
            "--node-rank", type=int, default=ServerArgs.node_rank, help="The node rank."
        )

        # Model override args
        parser.add_argument(
            "--hf-overrides",
            metavar="HF_OVERRIDES",
            type=str,
            help="A dictionary in JSON string format used to override default model configurations.",
            default=ServerArgs.hf_overrides,
        )
        parser.add_argument(
            "--preferred-sampling-params",
            type=str,
            help="json-formatted sampling settings that will be returned in /get_model_info",
        )

        # Kernel backend
        attention_backend_choices = [
            "mha",
            "fa3",
            "fa4",
            "triton",
            "flashinfer",
            "trtllm",
            "trtllm_mla",
            "flashmla",
            "tokenspeed_mla",
            "hybrid_linear_attn",
        ]
        parser.add_argument(
            "--attention-backend",
            type=str,
            choices=attention_backend_choices,
            default=ServerArgs.attention_backend,
            help="Choose the kernels for attention layers.",
        )
        parser.add_argument(
            "--drafter-attention-backend",
            type=str,
            choices=attention_backend_choices,
            help="Attention backend for drafter model in speculative decoding. "
            "If not specified, uses the same backend as the main model (attention_backend).",
        )
        parser.add_argument(
            "--sampling-backend",
            type=str,
            choices=["greedy", "flashinfer", "flashinfer_full"],
            default=ServerArgs.sampling_backend,
            help="Sampling backend. "
            "When unspecified, defaults to 'flashinfer' on NVIDIA and 'greedy' elsewhere. "
            "'greedy': argmax + verify_chain_greedy, zero sampling-param plumbing. "
            "'flashinfer': temperature/top_k/top_p via one fused kernel (top_k_top_p_sampling_from_logits); "
            "min_p and penalties silently ignored. "
            "'flashinfer_full': adds min_p plus frequency/presence/repetition penalties and logit_bias "
            "via the softmax+renorm+min_p kernel sequence. "
            "Allocates a counts[max_req_pool_size, vocab_size] int32 buffer (substantial memory).",
        )
        parser.add_argument(
            "--attention-use-fp4-indexer-cache",
            "--attention-config.use-fp4-indexer-cache",
            "--attention_config.use_fp4_indexer_cache",
            type=str_to_bool,
            nargs="?",
            const=True,
            default=ServerArgs.attention_use_fp4_indexer_cache,
            help="Use the MXFP4 sparse attention indexer cache layout.",
        )
        parser.add_argument(
            "--attention-config.use-trtllm-ragged-deepseek-prefill",
            "--attention-config.use_trtllm_ragged_deepseek_prefill",
            "--attention_config.use_trtllm_ragged_deepseek_prefill",
            dest="use_trtllm_ragged_deepseek_prefill",
            type=str_to_bool,
            nargs="?",
            const=True,
            default=ServerArgs.use_trtllm_ragged_deepseek_prefill,
            help="Use ragged prefill for DeepSeek MLA attention.",
        )
        parser.add_argument(
            "--disable-deepseek-v4-fast-mhc",
            action="store_true",
            default=ServerArgs.disable_deepseek_v4_fast_mhc,
            help=(
                "Disable the DeepSeek V4 fast multi-head capture (mHC) layer. "
                "Falls back to the PyTorch reference when set or when nvcc is "
                "not available."
            ),
        )
        parser.add_argument(
            "--deepseek-v4-mega-moe-max-num-tokens",
            type=int,
            default=ServerArgs.deepseek_v4_mega_moe_max_num_tokens,
            help=(
                "DeepSeek V4 MegaMoE staging-buffer cap on tokens per forward "
                "(0 = derive from chunked-prefill / cuda-graph budgets)."
            ),
        )
        parser.add_argument(
            "--deepseek-v4-indexer-prefill-max-logits-mb",
            type=int,
            default=ServerArgs.deepseek_v4_indexer_prefill_max_logits_mb,
            help=(
                "DeepSeek V4 sparse indexer prefill workspace cap (MiB) for the "
                "softplus_sqrt logits buffer."
            ),
        )
        parser.add_argument(
            "--deepseek-v4-prefill-chunk-size",
            type=int,
            default=ServerArgs.deepseek_v4_prefill_chunk_size,
            help=(
                "Maximum number of requests per DeepSeek V4 FlashMLA prefill " "chunk."
            ),
        )
        parser.add_argument(
            "--grammar-backend",
            type=str,
            choices=["xgrammar", "none"],
            default=ServerArgs.grammar_backend,
            help="Grammar backend. 'none' disables grammar-guided decoding entirely ",
        )
        parser.add_argument(
            "--grammar-compile-timeout-secs",
            type=float,
            default=ServerArgs.grammar_compile_timeout_secs,
            help="Per-compile wallclock budget before the request is aborted.",
        )
        parser.add_argument(
            "--grammar-compile-max-retries",
            type=int,
            default=ServerArgs.grammar_compile_max_retries,
            help="Compile timeouts allowed before a grammar key is permanently rejected.",
        )
        parser.add_argument(
            "--disable-any-whitespace",
            action="store_true",
            default=ServerArgs.disable_any_whitespace,
            help="Compile xgrammar JSON grammars in tight mode (no arbitrary "
            "whitespace between tokens). Mitigates models that wedge into "
            "endless whitespace until length cutoff. xgrammar only.",
        )
        parser.add_argument(
            "--disable-capturable-grammar",
            action="store_true",
            default=ServerArgs.disable_capturable_grammar,
            help="Force the synchronous eager grammar fallback even on CUDA. "
            "For parity-testing the captured-grammar path: output should "
            "match; throughput will be lower (sync stall every step).",
        )
        parser.add_argument(
            "--mla-disable-ragged",
            action="store_true",
            help="Disable the ragged prefill wrapper on MLA kernel backends during EXTEND.",
        )

        # Speculative decoding
        parser.add_argument(
            "--draft-model-path-use-base",
            action="store_true",
            help="The path of the draft model weights use the path of the base model",
        )
        parser.add_argument(
            "--speculative-config",
            "--speculative_config",
            type=str,
            default=ServerArgs.speculative_config,
            help="JSON speculative decoding configuration. Supported keys are method, model, and num_speculative_tokens.",
        )
        parser.add_argument(
            "--speculative-algorithm",
            type=str,
            choices=["EAGLE3", "MTP"],
            help="Speculative algorithm.",
        )
        parser.add_argument(
            "--speculative-draft-model-path",
            type=str,
            help="The path of the draft model weights. This can be a local folder or a Hugging Face repo ID.",
        )
        parser.add_argument(
            "--speculative-draft-model-quantization",
            type=str,
            default=ServerArgs.speculative_draft_model_quantization,
            help="Quantization method for the draft model. Defaults to 'unquant'.",
        )
        parser.add_argument(
            "--speculative-num-steps",
            type=int,
            help="The number of steps sampled from draft model in Speculative Decoding.",
            default=ServerArgs.speculative_num_steps,
        )
        parser.add_argument(
            "--speculative-eagle-topk",
            type=int,
            help="The number of tokens sampled from the draft model in each speculative step.",
            choices=[1, 2, 4, 8],
            default=ServerArgs.speculative_eagle_topk,
        )
        parser.add_argument(
            "--speculative-num-draft-tokens",
            type=int,
            help="The number of tokens sampled from the draft model in Speculative Decoding.",
            default=ServerArgs.speculative_num_draft_tokens,
        )
        parser.add_argument(
            "--enable-output-logprobs",
            action="store_true",
            default=ServerArgs.enable_output_logprobs,
            help="Enable per-token sampled-token logprobs. OFF by default; enabling extends the captured CUDA-graph footprint. Requests asking for logprobs on a server without this flag receive empty logprobs.",
        )
        parser.add_argument(
            "--eagle3-layers-to-capture",
            type=str,
            help="The layers of Eagle3 to capture.",
            default=ServerArgs.eagle3_layers_to_capture,
        )

        # Runtime options
        parser.add_argument(
            "--disable-pdl",
            action="store_true",
            help="Disable PDL launch.",
        )
        # LoRA adapter serving
        parser.add_argument(
            "--enable-lora",
            action="store_true",
            default=ServerArgs.enable_lora,
            help="Enable LoRA adapter serving.",
        )
        parser.add_argument(
            "--max-loras",
            type=int,
            default=ServerArgs.max_loras,
            help="Maximum number of non-pinned LoRA adapters in GPU memory at once.",
        )
        parser.add_argument(
            "--max-lora-rank",
            type=int,
            default=ServerArgs.max_lora_rank,
            help="Maximum LoRA rank supported across all loaded adapters.",
        )
        parser.add_argument(
            "--max-loras-cpu",
            type=int,
            default=ServerArgs.max_loras_cpu,
            help=(
                "Maximum number of LoRA adapters cached in CPU pinned "
                "memory.  Defaults to 4 × --max-loras.  Adapters evicted "
                "from this pool are reloaded from disk on next use."
            ),
        )
        parser.add_argument(
            "--lora-scheduling-policy",
            type=str,
            default=ServerArgs.lora_scheduling_policy,
            choices=["lru", "pack"],
            help=(
                "Scheduler-side LoRA scheduling policy. ``lru`` (default) "
                "submits requests in arrival order and relies on the "
                "manager's LRU pool. ``pack`` sorts the admission queue "
                "by lora_id so adapter-shared requests cluster, reducing "
                "eviction churn when working_set > max_loras_cpu and "
                "traffic is bursty."
            ),
        )

        prefix_cache_group = parser.add_mutually_exclusive_group()
        prefix_cache_group.add_argument(
            "--enable-prefix-caching",
            action="store_true",
            default=ServerArgs.enable_prefix_caching,
            help="Enable prefix caching.",
        )
        prefix_cache_group.add_argument(
            "--no-enable-prefix-caching",
            dest="enable_prefix_caching",
            action="store_false",
            help="Disable prefix caching.",
        )
        parser.add_argument(
            "--enforce-eager",
            action="store_true",
            help="Disable CUDA graph.",
        )
        parser.add_argument(
            "--disable-cuda-graph-padding",
            action="store_true",
            help="Disable cuda graph when padding is needed. Still uses cuda graph when padding is not needed.",
        )
        parser.add_argument(
            "--enable-cudagraph-gc",
            action="store_true",
            help="Enable garbage collection during CUDA graph capture. If disabled (default), GC is frozen during capture to speed up the process.",
        )
        parser.add_argument(
            "--enable-nccl-nvls",
            action="store_true",
            help="Enable NCCL NVLS for prefill heavy requests when available.",
        )
        parser.add_argument(
            "--enable-symm-mem",
            action="store_true",
            help="Enable NCCL symmetric memory for fast collectives.",
        )
        parser.add_argument(
            "--disable-custom-all-reduce",
            action="store_true",
            help="Disable the custom all-reduce kernel and fall back to NCCL.",
        )
        parser.add_argument(
            "--disable-overlap-schedule",
            action="store_true",
            help="Disable the overlap scheduler, which overlaps the CPU scheduler with GPU model worker.",
        )
        parser.add_argument(
            "--disable-tf32",
            action="store_true",
            help="Disable forcing TF32 on for cuBLAS/cuDNN. By default the server sets "
            "NVIDIA_TF32_OVERRIDE=1 and TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1.",
        )
        parser.add_argument(
            "--max-cudagraph-capture-size",
            metavar="MAX_CUDAGRAPH_CAPTURE_SIZE",
            type=int,
            default=ServerArgs.max_cudagraph_capture_size,
            help="Set the maximum batch size for CUDA graph capture.",
        )
        parser.add_argument(
            "--cudagraph-capture-sizes",
            metavar="CUDAGRAPH_CAPTURE_SIZE",
            type=int,
            nargs="+",
            help="Set the list of batch sizes for CUDA graph capture.",
        )
        parser.add_argument(
            "--disable-prefill-graph",
            action="store_true",
            help="Disable cuda graph for prefill.",
        )
        parser.add_argument(
            "--prefill-graph-max-tokens",
            type=int,
            default=ServerArgs.prefill_graph_max_tokens,
            help="Max query tokens to capture when enable prefill graph",
        )
        parser.add_argument(
            "--enable-nan-detection",
            action="store_true",
            help="Enable the NaN detection for debugging purposes.",
        )
        parser.add_argument(
            "--enable-nvtx",
            action="store_true",
            help="Emit NVTX ranges around input_prep / target_forward / "
            "sampling / drafter stages for nsys profiling. Off by default "
            "(true no-op — no NVTX calls are made). Also enabled by "
            "TOKENSPEED_NVTX=1.",
        )
        parser.add_argument(
            "--enable-p2p-check",
            action="store_true",
            help="Enable the full GPU P2P access check, otherwise trust the driver's P2P report.",
        )
        parser.add_argument(
            "--triton-attention-reduce-in-fp32",
            action="store_true",
            help="Cast the intermediate attention results to fp32 to avoid possible crashes related to fp16."
            "This only affects Triton attention kernels.",
        )
        parser.add_argument(
            "--delete-ckpt-after-loading",
            action="store_true",
            help="Delete the model checkpoint after loading the model.",
        )
        parser.add_argument(
            "--weight-loader-prefetch-checkpoints",
            action="store_true",
            help=(
                "Prefetch safetensors checkpoint shards into OS page cache before "
                "loading. Local ranks split the shard list to reduce repeated reads "
                "from shared filesystems."
            ),
        )
        parser.add_argument(
            "--weight-loader-prefetch-num-threads",
            type=int,
            default=ServerArgs.weight_loader_prefetch_num_threads,
            help="Number of background threads per rank for checkpoint prefetching.",
        )
        parser.add_argument(
            "--enable-memory-saver",
            action="store_true",
            help="Allow saving memory using release_memory_occupation and resume_memory_occupation",
        )
        parser.add_argument(
            "--enable-custom-logit-processor",
            action="store_true",
            help="Enable users to pass custom logit processors to the server (disabled by default for security)",
        )
        # Server warmups
        parser.add_argument(
            "--skip-server-warmup",
            action="store_true",
            help="If set, skip warmup.",
        )
        parser.add_argument(
            "--warmups",
            type=str,
            required=False,
            help="Specify custom warmup functions (csv) to run before server starts eg. --warmups=warmup_name1,warmup_name2 "
            "will run the functions `warmup_name1` and `warmup_name2` specified in warmup.py before the server starts listening for requests",
        )

        parser.add_argument(
            "--tensor-parallel-size",
            "--tp",
            type=int,
            default=None,
            help="Sets tensor parallelism size uniformly (equivalent to --attn-tp-size). "
            "Cannot be used together with --attn-tp-size.",
        )
        parser.add_argument(
            "--enable-expert-parallel",
            action="store_true",
            help="Enable expert parallelism by automatically setting ep_size to world_size.",
        )

        # Specify different parallel strategies, different combinations correspond to different communication groups and weight partitioning, as well as different communication methods
        parser.add_argument(
            "--attn-tp-size",
            type=int,
            default=ServerArgs.attn_tp_size,
            help="Specify tp size for attn part",
        )
        parser.add_argument(
            "--dense-tp-size",
            type=int,
            default=ServerArgs.dense_tp_size,
            help="Specify tp size for dense part, default equals nprocs-per-node, if non dp_attn && combine_dense mode, this parameter will be overridden by attn_tp_size",
        )
        parser.add_argument(
            "--moe-tp-size",
            type=int,
            default=ServerArgs.moe_tp_size,
            help="Specify tp size for MoE part, default equals nprocs-per-node, if non dp_attn && combine_dense mode, this parameter will be overridden by attn_tp_size",
        )
        parser.add_argument(
            "--nprocs-per-node",
            type=int,
            default=ServerArgs.nprocs_per_node,
            help="Number of processes to start per node",
        )
        parser.add_argument(
            "--world-size",
            type=int,
            default=ServerArgs.world_size,
            help="Total number of processes across all nodes.",
        )
        parser.add_argument(
            "--force-deterministic-rsag",
            action="store_true",
            help="Enable force deterministic rsag.",
        )
        parser.add_argument(
            "--disable-sampling-tp-sync",
            action="store_true",
            help="Skip broadcasting sampler outputs across the attention TP "
            "group. Only safe when the sampling kernels are deterministic.",
        )
        parser.add_argument(
            "--low-latency-max-num-tokens-per-gpu",
            type=int,
            default=ServerArgs.low_latency_max_num_tokens_per_gpu,
            help="Low latency max num tokens per gpu",
        )

        parser.add_argument(
            "--mla-chunk-multiplier",
            type=int,
            default=ServerArgs.mla_chunk_multiplier,
            help=(
                "Per-iter MLA chunked-prefill chunk capacity multiplier; "
                "the actual capacity is chunked_prefill_size * mla_chunk_multiplier."
            ),
        )

        # Disaggregation
        parser.add_argument(
            "--disaggregation-mode",
            type=str,
            default="null",
            choices=["null", "prefill", "decode"],
            help='Only used for PD disaggregation. "prefill" for prefill-only server, and "decode" for decode-only server. If not specified, it is not PD disaggregated',
        )
        parser.add_argument(
            "--comm-fusion-max-num-tokens",
            type=int,
            default=ServerArgs.comm_fusion_max_num_tokens,
            help="Max num tokens for communication fusion workspace",
        )
        parser.add_argument(
            "--enable-allreduce-fusion",
            action="store_true",
            help="Enable allreduce fusion for improved decode performance. Auto-enabled on supported single-node TP configurations.",
        )
        parser.add_argument(
            "--disaggregation-bootstrap-port",
            type=int,
            default=ServerArgs.disaggregation_bootstrap_port,
            help="Bootstrap server port on the prefill server. Default is 8998.",
        )
        parser.add_argument(
            "--disaggregation-transfer-backend",
            type=str,
            default=ServerArgs.disaggregation_transfer_backend,
            choices=["mooncake", "mooncake_async"],
            help="The backend for disaggregation transfer. Default is mooncake.",
        )
        parser.add_argument(
            "--disaggregation-ib-device",
            type=str,
            default=ServerArgs.disaggregation_ib_device,
            help="The InfiniBand devices for disaggregation transfer, accepts single device (e.g., --disaggregation-ib-device mlx5_0) "
            "or multiple comma-separated devices (e.g., --disaggregation-ib-device mlx5_0,mlx5_1). "
            "Default is None, which triggers automatic device detection when mooncake backend is enabled.",
        )
        parser.add_argument(
            "--disaggregation-layerwise-interval",
            type=int,
            default=ServerArgs.disaggregation_layerwise_interval,
            help="The interval of layerwise transfer for disaggregation. Default is 1.",
        )
        parser.add_argument(
            "--pdlb-url",
            type=str,
            default=None,
            help="The URL of the PD disaggregation load balancer. If set, the prefill/decode server will register with the load balancer.",
        )

        # Multi-modal inference mode
        parser.add_argument("--mm-mode", type=str, default=ServerArgs.mm_mode)

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace):
        args.ep_size = args.expert_parallel_size

        # Resolve model (positional model arg vs --model)
        positional_model = getattr(args, "model_path", None)
        if positional_model is not None and args.model is not None:
            raise ValueError(
                "Cannot specify model both as a positional argument and --model. "
                "Use one or the other."
            )
        if positional_model is not None:
            args.model = positional_model
        if args.model is None:
            raise ValueError(
                "Model is required. Provide it as a positional argument "
                "(e.g., `tokenspeed serve <model>`) or via --model/--model-path."
            )

        # --tensor-parallel-size → --attn-tp-size
        tensor_parallel_size = getattr(args, "tensor_parallel_size", None)
        if tensor_parallel_size is not None:
            if args.attn_tp_size is not None:
                raise ValueError(
                    "Cannot specify both --tensor-parallel-size and --attn-tp-size. "
                    "--tensor-parallel-size is an alias for --attn-tp-size."
                )
            args.attn_tp_size = tensor_parallel_size

        # Only pass fields that argparse actually produced. Falling back to
        # ``None`` for missing attrs would silently clobber dataclass defaults
        # for non-CLI-exposed fields (e.g. ``enable_inline_detokenizer``).
        attrs = [attr.name for attr in dataclasses.fields(cls)]
        return cls(
            **{attr: getattr(args, attr) for attr in attrs if hasattr(args, attr)}
        )

    def url(self):
        if is_valid_ipv6_address(self.host):
            return f"http://[{self.host}]:{self.port}"
        return f"http://{self.host}:{self.port}"


def prepare_server_args(argv: list[str]) -> ServerArgs:
    """
    Prepare the server arguments from the command line arguments.

    Args:
        args: The command line arguments. Typically, it should be `sys.argv[1:]`.

    Returns:
        The server arguments.
    """
    parser = argparse.ArgumentParser(allow_abbrev=False)
    ServerArgs.add_cli_args(parser)
    raw_args = parser.parse_args(argv)
    server_args = ServerArgs.from_cli_args(raw_args)
    return server_args


ZMQ_TCP_PORT_DELTA = 233


@dataclasses.dataclass
class PortArgs:
    # The ipc filename for AsyncLLM to receive BatchTokenIDOut directly
    # from the scheduler (zmq).
    tokenizer_ipc_name: str
    # The ipc filename for scheduler (rank 0) to receive inputs from tokenizer (zmq)
    scheduler_input_ipc_name: str

    # The port for nccl initialization (torch.dist)
    nccl_port: int

    # The ipc filename for rpc call between Engine and Scheduler
    rpc_ipc_name: str

    # The ipc filename for Scheduler to send metrics
    metrics_ipc_name: str

    # The ipc filename for Tokenizer and worker tokenizer
    tokenizer_worker_ipc_name: str | None

    @staticmethod
    def init_new(server_args: ServerArgs, dp_rank: int | None = None) -> "PortArgs":
        port = server_args.port + random.randint(100, 1000)
        while True:
            if is_port_available(port):
                break
            if port < 60000:
                port += 42
            else:
                port -= 43

        # DP attention. Use TCP + port to handle both single-node and multi-node.
        if server_args.mapping.nnodes == 1 and server_args.dist_init_addr is None:
            # Only use default port fallback when dp_size == 1
            # For dp_size > 1, we need explicit dist_init_addr to avoid port conflicts
            if server_args.mapping.has_attn_dp:
                raise ValueError(
                    f"When dp_size > 1 (dp_size={server_args.mapping.attn.dp_size}), you must provide --dist-init-addr. "
                    f"Example: --dist-init-addr 127.0.0.1:4000"
                )
            dist_init_addr = ("127.0.0.1", server_args.port + ZMQ_TCP_PORT_DELTA)
        else:
            dist_init_addr = server_args.dist_init_addr.split(":")
        assert (
            len(dist_init_addr) == 2
        ), "please provide --dist-init-addr as host:port of head node"

        dist_init_host, dist_init_port = dist_init_addr
        dist_init_port = int(dist_init_port)

        # Scan forward until we find a port cluster where all derived ports are free.
        # This handles the case where a previous engine instance left ports in
        # TIME_WAIT or its child processes haven't fully terminated yet.
        # Note: the port at offset +1 (formerly detokenizer_port) is intentionally
        # skipped so the rest of the port layout stays stable for any external
        # tooling that indexed off the historical port cluster.
        while True:
            port_base = dist_init_port + 1
            rpc_port = port_base + 2
            metrics_ipc_port = port_base + 3
            if dp_rank is None:
                # TokenizerManager to DataParallelController
                scheduler_input_port = port_base + 4
            else:
                scheduler_input_port = port_base + 2 + 1 + dp_rank
            rpc_ipc_port = scheduler_input_port + 1
            if all(
                is_port_available(p)
                for p in [
                    dist_init_port,
                    port_base,
                    rpc_port,
                    metrics_ipc_port,
                    scheduler_input_port,
                    rpc_ipc_port,
                ]
            ):
                break
            dist_init_port += 10

        return PortArgs(
            tokenizer_ipc_name=f"tcp://{dist_init_host}:{port_base}",
            scheduler_input_ipc_name=f"tcp://{dist_init_host}:{scheduler_input_port}",
            nccl_port=port,
            rpc_ipc_name=f"tcp://{dist_init_host}:{rpc_port}",
            metrics_ipc_name=f"tcp://{dist_init_host}:{metrics_ipc_port}",
            tokenizer_worker_ipc_name=None,
        )
