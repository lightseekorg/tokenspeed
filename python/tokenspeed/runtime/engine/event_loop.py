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

import faulthandler
import signal
import time
from collections import OrderedDict
from dataclasses import dataclass

import psutil
import setproctitle
import torch
import torch.distributed as dist
import zmq
from tokenspeed_scheduler import PD, Cache, ExecutionEvent, ForwardEvent, Scheduler

from tokenspeed.runtime.cache.executor.flat_memory_executor import (
    FlatMemoryExecutor,
)
from tokenspeed.runtime.cache.executor.memory_executor import (
    MemoryExecutor,
    MemoryExecutorConfig,
)
from tokenspeed.runtime.cache.transfer.types import CacheKind
from tokenspeed.runtime.cache_capabilities import (
    validate_cache_runtime_capabilities,
)
from tokenspeed.runtime.configs.model_config import ModelConfig
from tokenspeed.runtime.configs.paged_cache_spec import (
    scheduler_ext_flat_kvcache,
    validate_flat_scheduler_config,
)
from tokenspeed.runtime.distributed.process_group_manager import (
    process_group_manager as pg_manager,
)
from tokenspeed.runtime.engine.generation_output_processor import OutputProcesser
from tokenspeed.runtime.engine.memory_occupation import MemoryOccupationController
from tokenspeed.runtime.engine.pause import PauseController
from tokenspeed.runtime.engine.request_handler import RequestHandler
from tokenspeed.runtime.engine.scheduler_utils import (
    advance_forward,
    cache_event_from_payload,
    cache_event_key,
    cache_event_to_payload,
    cache_sync_debug_enabled,
    make_config,
    pool_to_flat_block_pools,
    pool_to_paged_cache_groups,
    pool_to_prefix_cache_adjunct_spec,
    pop_common_cache_event_payloads,
    scheduler_admission_path,
    scheduler_backend_identity,
    should_use_overlap_schedule,
)
from tokenspeed.runtime.execution.distributed_initializer import (
    DistributedConfig,
    DistributedInitializer,
)
from tokenspeed.runtime.execution.factory import (
    ModelExecutorConfig,
    create_model_executor,
    create_model_runner,
)
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.execution.types import ModelExecutionResult
from tokenspeed.runtime.flat_kv_metrics import (
    collect_flat_pool_metrics,
    flat_pool_equivalent_pages,
    summarize_flat_pool_aggregate,
)
from tokenspeed.runtime.grammar.capturable_grammar import GrammarStepInputs
from tokenspeed.runtime.layers.attention.registry import create_attn_components
from tokenspeed.runtime.metrics.collector import EngineMetrics
from tokenspeed.runtime.pd.decode_executor import DisaggDecodeExecutor
from tokenspeed.runtime.pd.factory import (
    create_kv_transfer,
    get_kv_args,
)
from tokenspeed.runtime.pd.kv_events import (
    EventPublisherFactory,
    KVEventBatch,
    NullEventPublisher,
    drain_scheduler_kv_events,
    scheduler_kv_events_to_wire_events,
)
from tokenspeed.runtime.pd.mooncake.entities import KVManagerArgs
from tokenspeed.runtime.pd.prefill_executor import DisaggPrefillExecutor
from tokenspeed.runtime.sampling.sampling_params import SamplingParams
from tokenspeed.runtime.utils import (
    configure_logger,
    get_colorful_logger,
    get_zmq_socket,
)
from tokenspeed.runtime.utils.exceptions import get_exception_traceback
from tokenspeed.runtime.utils.nvtx import nvtx_range
from tokenspeed.runtime.utils.process import register_usr_signal
from tokenspeed.runtime.utils.server_args import PortArgs, ServerArgs
from tokenspeed.runtime.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter

logger = get_colorful_logger(__name__)


def calc_l3_query_hashes(scheduler, tokens: list[int]) -> list[str]:
    return scheduler.calc_rolling_hash(tokens, apply_match=True)


# Sleep between iterations while frozen (PAUSED_ALL) so the keep-mode pause does
# not busy-spin a CPU core waiting for /resume.
_PAUSED_IDLE_SLEEP_S = 0.001

# Per-pool rows allocate C++/Python containers and update labeled gauges. Keep
# them off the decode loop while refreshing often enough for observability.
_FLAT_POOL_DETAIL_INTERVAL_S = 1.0


def _forward_op_executes_model_forward(forward_op, *, is_disagg_decode: bool) -> bool:
    """Return whether ``forward_op`` will enter the model forward path.

    On decode-side PD, EXTEND ops only start remote KV receive; the model
    forward runs after the remote prefill completes and the scheduler advances
    the request into decode. Treating those EXTEND ops as model work makes
    idle DP ranks enter dummy collectives that the active rank will not match.
    """
    if forward_op is None:
        return False
    if sum(forward_op.input_lengths) <= 0:
        return False
    if is_disagg_decode and forward_op.num_extends() > 0:
        return False
    return True


class _NullSender:
    """No-op ZMQ sender for non-rank-0 workers."""

    @staticmethod
    def send_pyobj(x):
        return None


@dataclass(frozen=True)
class DpForwardMetadata:
    global_num_tokens: list[int]
    global_batch_size: list[int]
    global_forward_mode: list[int]
    all_decode_or_idle: bool
    all_extend: bool
    need_idle_forward: bool


class EventLoop:
    def __init__(
        self,
        server_args: ServerArgs,
        port_args: PortArgs,
        gpu_id: int,
        attn_tp_rank: int,
        dp_rank: int,
        global_rank: int,
    ) -> None:
        # Do not pass server_args further down the stack after this point.

        self.server_args = server_args
        self.port_args = port_args
        self.gpu_id = gpu_id
        self.global_rank = global_rank
        flat_kvcache_ext = scheduler_ext_flat_kvcache()

        self.model_config = self._load_model_config(server_args.model)
        if server_args.speculative_draft_model_path is not None:
            draft_model_config = self._load_model_config(
                server_args.speculative_draft_model_path,
                is_draft_worker=True,
            )
        else:
            draft_model_config = None

        min_per_gpu_mem = self._init_distributed()

        target, draft = create_model_runner(
            server_args, self.model_config, draft_model_config, gpu_id, global_rank
        )
        self.use_overlap_schedule = should_use_overlap_schedule(
            disable_overlap_schedule=server_args.disable_overlap_schedule,
            disaggregation_mode=server_args.disaggregation_mode,
        )
        self.overlap_schedule_depth = int(self.use_overlap_schedule)
        decode_input_tokens = (
            server_args.speculative_num_draft_tokens
            if server_args.speculative_algorithm is not None
            else 1
        )

        (
            attn_backend,
            token_to_kv_pool,
            draft_attn_backend,
            draft_token_to_kv_pool,
            self.max_total_num_tokens,
            mamba_pool_total_chunks,
            mamba_pool,
        ) = create_attn_components(
            server_args,
            self.model_config,
            gpu_id,
            global_rank,
            min_per_gpu_mem,
            server_args.enable_memory_saver,
            draft_model_config,
            decode_input_tokens=decode_input_tokens,
            overlap_schedule_depth=self.overlap_schedule_depth,
        )
        validate_cache_runtime_capabilities(
            target_pool=token_to_kv_pool,
            draft_pool=draft_token_to_kv_pool,
            disaggregation_mode=server_args.disaggregation_mode,
            enable_kvstore=server_args.enable_kvstore,
        )
        paged_cache_groups = pool_to_paged_cache_groups(token_to_kv_pool)
        validate_flat_scheduler_config(
            flat_kvcache_ext=flat_kvcache_ext,
            paged_cache_groups=paged_cache_groups,
            attn_backend=attn_backend,
            kv_pool=token_to_kv_pool,
            speculative_algorithm=server_args.speculative_algorithm,
        )

        num_total_pages = self.max_total_num_tokens // server_args.block_size
        hf_config = getattr(self.model_config, "hf_config", None)
        text_config = getattr(hf_config, "text_config", None) if hf_config else None
        has_mamba = getattr(self.model_config, "mambaish_config", None) is not None or (
            text_config is not None and hasattr(text_config, "mamba2_cache_params")
        )

        mapping = server_args.mapping
        # The C++ scheduler's req_pool_idx range is rank-local and 1-based:
        # real rows are 1..max_batch_size, row 0 is reserved, and CUDA graph
        # padding needs one non-real sink row after the scheduler-owned range.
        per_rank_max_batch = server_args.max_num_seqs // max(mapping.attn.dp_size, 1)
        req_pool_padding_index = per_rank_max_batch + 1

        model_executor_config = ModelExecutorConfig.from_server_args(
            server_args=server_args,
            model_config=self.model_config,
            max_req_pool_size=req_pool_padding_index,
            gpu_id=gpu_id,
            global_rank=global_rank,
            num_total_pages=num_total_pages,
            overlap_schedule_depth=self.overlap_schedule_depth,
        )
        self.model_executor = create_model_executor(
            server_args=server_args,
            config=model_executor_config,
            model_runner=target,
            draft_model_runner=draft,
            attn_backend=attn_backend,
            token_to_kv_pool=token_to_kv_pool,
            draft_attn_backend=draft_attn_backend,
            draft_token_to_kv_pool=draft_token_to_kv_pool,
            mamba_pool=mamba_pool,
        )

        # Reserve one token slot because request validation uses a strict
        # ``< max_req_len`` check against the model context length.
        self.max_req_input_len = self.model_config.context_len - 1
        self.attn_tp_size = server_args.attn_tp_size or mapping.attn.tp_size
        self.world_size = server_args.world_size or mapping.world_size
        self.attn_tp_rank = attn_tp_rank
        self.attn_tp_cpu_group = pg_manager.get_process_group(
            "gloo", server_args.mapping.attn.tp_group
        )
        self._pending_cache_event_payloads: OrderedDict[tuple[str, int], dict] = (
            OrderedDict()
        )
        # All ranks submit identical cache plans (the C++ scheduler is mirrored),
        # so a local in-flight counter mirrors across ranks: if it's 0 here, no
        # rank has anything pending. Lets us skip the TP collective in
        # _commit_cache_results entirely when nothing is in flight.
        self._num_inflight_cache_ops = 0
        self.dp_rank = dp_rank
        self.dp_size = mapping.attn.dp_size
        self.has_dp = mapping.has_attn_dp
        if self.has_dp:
            self.world_cpu_group = pg_manager.get_process_group(
                "gloo", mapping.world_group
            )
            self._dp_local_info = torch.zeros(1, 3, dtype=torch.int32)
            self._dp_global_info = torch.zeros(mapping.world_size, 3, dtype=torch.int32)
        if not server_args.enable_kvstore:
            logger.warning(
                "KVStore L2 cache will not be used during normal execution, but it will still be used when retraction happens."
            )

        mamba_l2_host_slots = 0
        if has_mamba and server_args.enable_mamba_l2:
            if server_args.mamba_l2_host_slots > 0:
                mamba_l2_host_slots = server_args.mamba_l2_host_slots
            elif server_args.mamba_l2_host_gb > 0 and mamba_pool is not None:
                slot_bytes = int(
                    mamba_pool.conv_state.shape[0]
                    * (
                        mamba_pool.conv_state[0, 0].nbytes
                        + mamba_pool.ssm_state[0, 0].nbytes
                    )
                )
                mamba_l2_host_slots = int(
                    server_args.mamba_l2_host_gb * (1024**3) // max(slot_bytes, 1)
                )
            else:
                mamba_l2_host_slots = max(
                    int(mamba_pool_total_chunks * server_args.mamba_l2_ratio), 1
                )

        mem_cfg = MemoryExecutorConfig(
            layer_num=self.model_config.num_hidden_layers,
            page_size=server_args.block_size,
            host_ratio=server_args.kvstore_ratio,
            host_size_gb=server_args.kvstore_size,
            host_parallel_count=max(
                int(getattr(server_args.mapping, "nprocs_per_node", 1) or 1), 1
            ),
            io_backend=server_args.kvstore_io_backend,
            host_layout=server_args.kvstore_mem_layout,
            storage_backend=server_args.kvstore_storage_backend,
            storage_backend_extra_config=server_args.kvstore_storage_backend_extra_config,
            model_name=server_args.model,
            enable_mamba_l2=server_args.enable_mamba_l2,
            mamba_l2_host_slots=mamba_l2_host_slots,
            mamba_l2_layout=server_args.mamba_l2_layout,
            mamba_l2_io_backend=server_args.mamba_l2_io_backend,
        )
        if flat_kvcache_ext and server_args.enable_kvstore:
            if server_args.kvstore_storage_backend is not None:
                raise NotImplementedError(
                    "flat scheduler build (TOKENSPEED_FLAT_KVCACHE) has no L3 "
                    "storage tier yet; unset --kvstore-storage-backend."
                )
            self.memory_executor = FlatMemoryExecutor(
                device_pool=token_to_kv_pool,
                host_ratio=server_args.kvstore_ratio,
                host_size_gb=server_args.kvstore_size,
            )
            num_host_pages = self.memory_executor.num_host_pages
        elif not token_to_kv_pool.supports_hierarchical_kv_cache:
            if server_args.enable_kvstore:
                raise NotImplementedError(
                    "This KV cache pool does not support hierarchical cache "
                    "(kvstore); pass --disable-kvstore."
                )
            self.memory_executor = None
            num_host_pages = 0
        else:
            self.memory_executor = MemoryExecutor(
                device_pool=token_to_kv_pool,
                config=mem_cfg,
                is_dp_attention_enabled=self.has_dp,
                tp_group=self.attn_tp_cpu_group,
                draft_device_pool=draft_token_to_kv_pool,
                mamba_pool=mamba_pool,
            )
            num_host_pages = self.memory_executor.host_pool.page_num

        # Flat host tier acks loadbacks (LoadBackDoneEvent), so they join the
        # inflight accounting in _submit_cache_ops; radix loadbacks never ack.
        self._loadback_acks_expected = getattr(
            self.memory_executor, "emits_loadback_acks", False
        )

        self._kv_events_enabled = (
            EventPublisherFactory.is_enabled(server_args.kv_events_config)
            and attn_tp_rank == 0
        )

        if has_mamba and server_args.max_mamba_cache_size is None:
            logger.info(
                f"Mamba radix cache enabled without explicit max_mamba_cache_size. "
                f"Auto-derived mamba_pool_total_chunks={mamba_pool_total_chunks} "
                f"(ratio={server_args.mamba_full_memory_ratio})."
            )

        # Adjunct enabled only when pool opts in AND prefix-caching switch is on.
        flat_block_pools = pool_to_flat_block_pools(token_to_kv_pool)
        self._paged_cache_groups = paged_cache_groups
        prefix_cache_adjunct = None
        required_groups = token_to_kv_pool.prefix_cache_required_group_ids
        if required_groups is not None and server_args.enable_prefix_caching:
            prefix_cache_adjunct = pool_to_prefix_cache_adjunct_spec(required_groups)
        scheduler_backend, _ = scheduler_backend_identity(flat_kvcache_ext)
        scheduler_cfg = make_config(
            num_device_pages=(
                0
                if flat_block_pools
                else self.max_total_num_tokens // server_args.block_size
            ),
            max_scheduled_tokens=server_args.chunked_prefill_size,
            max_batch_size=per_rank_max_batch,
            page_size=server_args.block_size,
            scheduler_backend=scheduler_backend,
            num_host_pages=num_host_pages,
            disable_l2_cache=not server_args.enable_kvstore,
            enable_l3_storage=server_args.kvstore_storage_backend is not None,
            prefetch_threshold=4,  # Keep this hard-coded until it becomes configurable.
            role=server_args.disaggregation_mode,
            enable_kv_cache_events=self._kv_events_enabled,
            decode_input_tokens=decode_input_tokens,
            overlap_schedule_depth=self.overlap_schedule_depth,
            disable_prefix_cache=not server_args.enable_prefix_caching,
            enable_mamba=has_mamba,
            mamba_cache_chunk_size=server_args.mamba_cache_chunk_size,
            mamba_pool_total_chunks=mamba_pool_total_chunks,
            enable_mamba_l2=server_args.enable_mamba_l2,
            mamba_l2_host_slots=mamba_l2_host_slots,
            paged_cache_groups=paged_cache_groups,
            flat_block_pools=flat_block_pools,
            enable_mixed_prefill_decode=server_args.enable_mixed_batch,
            prefix_cache_adjunct=prefix_cache_adjunct,
        )
        admission_path = scheduler_admission_path(
            flat_kvcache_ext=flat_kvcache_ext,
            config=scheduler_cfg,
        )
        logger.info(
            "Scheduler config: kv_backend=%s admission_path=%s "
            "block_size=%s num_device_pages=%s "
            "max_scheduled_tokens=%s decode_input_tokens=%s "
            "overlap_schedule_depth=%s disable_l2_cache=%s "
            "max_batch_size=%s (global max_num_seqs=%s, dp_size=%s) "
            "mamba_pool_total_chunks=%s enable_mamba=%s "
            "disable_prefix_cache=%s flat_block_pools=%s "
            "paged_cache_groups=%s",
            scheduler_backend,
            admission_path,
            scheduler_cfg.block_size,
            scheduler_cfg.num_device_pages,
            scheduler_cfg.max_scheduled_tokens,
            scheduler_cfg.decode_input_tokens,
            scheduler_cfg.overlap_schedule_depth,
            scheduler_cfg.disable_l2_cache,
            scheduler_cfg.max_batch_size,
            server_args.max_num_seqs,
            self.dp_size,
            mamba_pool_total_chunks,
            has_mamba,
            scheduler_cfg.disable_prefix_cache,
            [pool.pool_id for pool in flat_block_pools],
            [group.group_id for group in paged_cache_groups],
        )
        self.scheduler = Scheduler(scheduler_cfg)
        token_to_kv_pool.bind_paged_cache_scheduler(self.scheduler)
        if attn_tp_rank == 0:
            self.kv_event_publisher = EventPublisherFactory.create(
                server_args.kv_events_config,
                attn_dp_rank=dp_rank,
            )
        else:
            self.kv_event_publisher = NullEventPublisher(attn_dp_rank=dp_rank)

        self._init_interprocess_comm()

        # Pause/resume control state. Shared with the request handler, which
        # drives the control-request side; the event loop reads the gate.
        self._pause = PauseController(self.send_to_tokenizer)

        # GPU-memory data plane (release/resume_memory_occupation). Reuses the
        # pause controller's drain machinery; frees memory via the memory-saver
        # adapter once the scheduler drains. See memory_occupation.py.
        # A shared flat arena must reset scheduler metadata and advance the same
        # generation that repairs its device mappings on wake. Pools without an
        # arena keep the legacy prefix-cache capability gate.
        self._flat_kv_release_generation: int | None = None
        flat_cache_arena = self._flat_cache_arena()
        if flat_cache_arena is not None:
            kv_cache_release_allowed = callable(
                getattr(self.scheduler, "reset_flat_kv_cache", None)
            )
        else:
            kv_cache_release_allowed = (
                not self.server_args.enable_prefix_caching
                or callable(getattr(self.scheduler, "reset_prefix_cache", None))
            )
        self._memory = MemoryOccupationController(
            send_func=self.send_to_tokenizer,
            pause_controller=self._pause,
            adapter=TorchMemorySaverAdapter.create(
                enable=self.server_args.enable_memory_saver
            ),
            enabled=self.server_args.enable_memory_saver,
            reset_caches_fn=self._reset_caches_for_release,
            kv_repair_fn=self._kv_repair_after_wake,
            kv_cache_release_allowed=kv_cache_release_allowed,
        )

        self.metrics = EngineMetrics(
            labels={
                "model_name": server_args.served_model_name,
                "app_key": server_args.app_key or "",
                "dp_rank": str(dp_rank),
            },
            enabled=(
                server_args.enable_metrics
                and attn_tp_rank == 0
                and "prometheus" in (server_args.metrics_reporters or [])
            ),
        )
        self._next_flat_pool_detail_at = 0.0

        self.request_handler = RequestHandler(
            server_args=self.server_args,
            hf_eos_token_id=self.model_config.hf_eos_token_id,
            max_req_len=self.model_config.context_len - 1,
            vocab_size=self.model_config.vocab_size,
            recv_func=self.recv_from_tokenizer,
            send_func=self.send_to_tokenizer,
            get_load_fn=self._get_load,
            get_internal_state_fn=self._get_internal_state,
            architectures=self.model_config.hf_config.architectures,
            pause_controller=self._pause,
            memory_controller=self._memory,
            model_runner=target,
        )

        self.output_processor = OutputProcesser(
            send_to_tokenizer=self.send_to_tokenizer,
            attn_tp_rank=attn_tp_rank,
            spec_algorithm=self.server_args.speculative_algorithm,
            spec_num_tokens=(
                self.server_args.speculative_num_draft_tokens
                if self.server_args.speculative_algorithm is not None
                else None
            ),
            stream_interval=self.server_args.stream_interval,
            enable_log_request_stats=self.server_args.enable_log_request_stats,
            metrics=self.metrics,
        )
        self.prefetch_threshold = scheduler_cfg.prefetch_threshold

        if server_args.disaggregation_mode != "null":
            kv_args = get_kv_args(
                global_rank,
                global_rank,
                server_args.disaggregation_ib_device,
                token_to_kv_pool,
                draft_token_to_kv_pool,
                mamba_pool,
            )
            pd_manager_args = KVManagerArgs(
                bootstrap_port=server_args.disaggregation_bootstrap_port,
                dist_init_addr=server_args.dist_init_addr,
                world_size=server_args.world_size or mapping.world_size,
                dp_size=server_args.data_parallel_size or mapping.attn.dp_size,
                attn_tp_rank=attn_tp_rank,
                attn_dp_rank=dp_rank,
                is_mla_backend=False,
                draft_is_mla_backend=False,
                enable_metrics=False,
                enable_mla_l1_5_cache=server_args.enable_mla_l1_5_cache,
                served_model_name=server_args.served_model_name,
                app_key=server_args.app_key,
                metrics_reporters=server_args.metrics_reporters,
                enable_dp_attention=self.has_dp,
            )
            self.kv_transfer = create_kv_transfer(
                mode=server_args.disaggregation_mode,
                backend=server_args.disaggregation_transfer_backend,
                args=pd_manager_args,
                kv_args=kv_args,
                gloo_group=self.attn_tp_cpu_group,
                page_size=token_to_kv_pool.page_size,
            )
            self._setup_pd_layerwise_transfer(
                server_args.disaggregation_layerwise_interval
            )
            # EPD: a multimodal prefill node is also the encode->prefill embedding
            # SINK (independent of kv_transfer, its P->D KV source) -- it receives
            # each image's embedding from encode workers over Mooncake so the
            # prefill skips the vision tower. The admission controller owns the
            # receive jobs, the rank-synced admission drain, and the optional NCCL
            # row-shard reassembly; None for decode/encode/text-only nodes.
            from tokenspeed.runtime.epd.prefill_admission import (
                make_epd_prefill_admission,
            )

            self.epd_admission = make_epd_prefill_admission(
                server_args,
                global_rank,
                model_config=self.model_config,
                model_executor=self.model_executor,
                mapping=mapping,
                attn_tp_rank=self.attn_tp_rank,
                attn_tp_size=self.attn_tp_size,
                attn_tp_cpu_group=self.attn_tp_cpu_group,
                pg_manager=pg_manager,
            )
            # Staged EPD request payloads (request_id -> (spec, state, bootstrap)),
            # held here while the controller (rid-keyed, like kv_transfer) runs the
            # async receive; popped in _drain_ready_epd_embeddings on admit/abort.
            self._epd_staged: dict = {}
        else:
            self.kv_transfer = None
            self.epd_admission = None
            self._epd_staged: dict = {}

    def _setup_pd_layerwise_transfer(self, interval: int) -> None:
        if not isinstance(self.kv_transfer, DisaggPrefillExecutor):
            return
        if interval <= 0:
            return

        from tokenspeed.runtime.pd.utils import StepCounter

        step_counter = StepCounter(self.model_executor.device, self.gpu_id)
        self.model_executor.attn_backend.register_step_counter(step_counter)
        if self.model_executor.draft_attn_backend is not None:
            self.model_executor.draft_attn_backend.register_step_counter(step_counter)
        self.kv_transfer.register_layerwise_step_counter(step_counter, interval)

    def _is_epd_request(self, state) -> bool:
        """True iff this request's images are encode-routed (smg injected per-image
        encode handshakes) -- it must wait for its embeddings (staged via the EPD
        admission controller, polled in _drain_ready_epd_embeddings) before being
        scheduled. Caller guards on self.epd_admission (only a multimodal prefill
        node has one); everything else admits immediately.
        """
        mm = getattr(state, "multimodal_inputs", None)
        return mm is not None and any(
            getattr(it, "encode_handshake", None) for it in mm.mm_items
        )

    def _assert_epd_embeddings_received(self, multimodal_context) -> None:
        """EPD invariant: every handshaked item is filled with its embedding by the
        async EPD admission drain (EpdPrefillAdmission.drain) BEFORE admission, so by
        it is already encoded. This is a defensive check, not a receive: a handshaked
        item that reached the forward un-received leaked past async admission (the
        only EPD admission path) -- fail loud instead of running the tower or
        publishing shard-only rows. No-op for non-EPD / text-only requests.
        """
        if (
            self.epd_admission is None
            or multimodal_context is None
            or not multimodal_context.has_extend_inputs()
        ):
            return
        for mm in multimodal_context.mm_inputs:
            if mm is None:
                continue
            missing = [
                i
                for i, item in enumerate(mm.mm_items)
                if getattr(item, "encode_handshake", None) is not None
                and item.encoded is None
            ]
            if missing:
                raise RuntimeError(
                    f"EPD: handshaked items {missing} reached the prefill forward "
                    "un-received; they must be admitted via the EPD admission drain"
                )

    def _drain_ready_epd_embeddings(self) -> None:
        """Admit EPD requests whose async embedding receives completed this cycle.

        The EpdPrefillAdmission controller DECIDES (poll + rank-lockstep MIN
        all-reduce + reassemble) and returns (admitted, failed); here we ACT on
        those decisions with the EventLoop's collaborators -- register/abort the
        P->D sender, submit admitted requests, finish failed ones. No-op (and no
        collective) on non-EPD nodes.
        """
        if self.epd_admission is None:
            return
        # Pause gate: withhold EPD admission while paused, mirroring the non-EPD
        # admit_blocked gate -- else the drain below would submit and RUN reassembled
        # specs during the pause. Staged receives wait in _pending until resume.
        # Rank-safe: admit_blocked is rank-identical, so all ranks skip together.
        if self._pause.admit_blocked:
            return
        admitted_ids, failed_ids = self.epd_admission.drain()
        for rid in failed_ids:
            spec, state, bootstrap = self._epd_staged.pop(rid)
            # Signal the dual-dispatched decode that this request failed so its KV
            # receiver fails (FailedEvent -> _process_kv_transfer_events abort)
            # instead of waiting forever for KV the prefill will never send. The
            # prefill never registered a P->D sender (deferred to admission), so the
            # decode has no other reliable way to learn (heartbeat only trips on a
            # dead prefill /health). Best-effort: only reaches decodes that already
            # pre-allocated.
            if (
                isinstance(self.kv_transfer, DisaggPrefillExecutor)
                and bootstrap is not None
            ):
                try:
                    self.kv_transfer.abort(rid, bootstrap)
                except Exception as exc:  # never let it wedge the loop
                    logger.warning(
                        "EPD abort->decode signal failed for rid=%s: %s",
                        rid,
                        exc,
                    )
            state.set_finish_with_abort("EPD embedding receive failed or timed out")
            self.output_processor.publish_finished_at_admission(rid, state)
        admitted_specs = []
        for rid in admitted_ids:
            spec, state, bootstrap = self._epd_staged.pop(rid)
            # Aborted mid-receive (no abort path, so drain still returns it admitted):
            # don't register the P->D sender or submit -- that runs a wasted forward
            # and leaks the sender. Stream its finish instead.
            if state.finished:
                self.output_processor.publish_finished_at_admission(rid, state)
                continue
            # Register the P->D sender now (deferred from admission) -- the request
            # is about to enter the scheduler.
            if self.kv_transfer is not None:
                self.kv_transfer.register(rid, bootstrap)
            admitted_specs.append(spec)
        if admitted_specs:
            self.scheduler.submit_requests(admitted_specs)
        elif self.epd_admission.has_pending():
            # Nothing advanced this cycle but requests are still receiving; yield the
            # GIL so the Python daemon transfer/recv threads run (rank-consistent:
            # admitted/leftover are rank-identical here).
            time.sleep(0.0005)

    def _commit_cache_results(self) -> None:
        if self.memory_executor is None:
            return
        cache_results = self.memory_executor.poll_results()
        self._num_inflight_cache_ops -= len(cache_results)
        for event in cache_results:
            payload = cache_event_to_payload(event)
            self._pending_cache_event_payloads[cache_event_key(payload)] = payload

        # The gather below is a collective, but cache-op completion is async and
        # not lock-step across ranks, so local state (_num_inflight_cache_ops /
        # _pending_cache_event_payloads) diverges transiently. A rank-local skip
        # would let some ranks gather while others return, deadlocking the group.
        # Agree on the skip via a cheap single-int all_reduce.
        # NOTE: For non-DFLASH algorithms, cache ops are deterministic across
        # ranks, so the local short-circuit is safe and avoids collective overhead.
        local_has_work = bool(
            self._num_inflight_cache_ops != 0 or self._pending_cache_event_payloads
        )
        if self.server_args.speculative_algorithm == "DFLASH":
            if not self._cache_group_has_work(local_has_work):
                return
        else:
            if not local_has_work:
                return

        ready_payloads = self._pop_ready_cache_event_payloads()
        if not ready_payloads:
            return
        logger.debug(
            "[cache_poll] got %s synchronized results, advancing scheduler",
            len(ready_payloads),
        )
        ec = ExecutionEvent()
        for payload in ready_payloads:
            e = cache_event_from_payload(payload)
            logger.debug(
                "[cache_poll] event: op_id=%s success=%s type=%s request_id=%s",
                e.op_id,
                e.success,
                type(e).__name__,
                getattr(e, "request_id", "N/A"),
            )
            ec.add_event(e)
        self.scheduler.advance(ec)
        logger.debug("[cache_poll] scheduler.advance() done")
        self._publish_scheduler_kv_events()

    def _publish_scheduler_kv_events(self) -> None:
        raw_events = drain_scheduler_kv_events(
            self.scheduler,
            enabled=self._kv_events_enabled,
        )
        if not raw_events:
            return

        events = scheduler_kv_events_to_wire_events(raw_events)
        if not events:
            return

        self.kv_event_publisher.publish(
            KVEventBatch(ts=time.time(), events=events, attn_dp_rank=self.dp_rank)
        )

    def _cache_group_has_work(self, local_has_work: bool) -> bool:
        """Whether ANY attn-tp rank has cache work this step (unanimous via a
        single-int MAX all_reduce, far cheaper than the payload gather it
        guards). Deciding from rank-local state alone deadlocks the group; see
        _commit_cache_results.

        Args:
            local_has_work: This rank's view of whether any cache op is in
                flight or any polled payload awaits commit.

        Returns:
            ``True`` if any rank has work (all must gather); ``False`` only when
            every rank is idle.
        """
        if self.attn_tp_size == 1:
            return local_has_work
        flag = torch.tensor([1 if local_has_work else 0], dtype=torch.int32)
        dist.all_reduce(flag, op=dist.ReduceOp.MAX, group=self.attn_tp_cpu_group)
        return bool(flag.item())

    def _pop_ready_cache_event_payloads(self) -> list[dict]:
        local_payloads = list(self._pending_cache_event_payloads.values())
        if self.attn_tp_size == 1:
            ready_payloads = local_payloads
        else:
            gathered_payloads = [None] * self.attn_tp_size
            dist.all_gather_object(
                gathered_payloads,
                local_payloads,
                group=self.attn_tp_cpu_group,
            )
            ready_payloads = pop_common_cache_event_payloads(gathered_payloads)
            if self.attn_tp_rank == 0 and cache_sync_debug_enabled():
                pending_ops = [
                    [(payload["kind"], payload["op_id"]) for payload in rank_payloads]
                    for rank_payloads in gathered_payloads
                ]
                if len({tuple(rank_ops) for rank_ops in pending_ops}) > 1:
                    logger.info(
                        "[cache_sync] rank=%s pending_ops=%s ready_ops=%s",
                        self.global_rank,
                        pending_ops,
                        [
                            (payload["kind"], payload["op_id"])
                            for payload in ready_payloads
                        ],
                    )

        for payload in ready_payloads:
            self._pending_cache_event_payloads.pop(cache_event_key(payload), None)
        return ready_payloads

    def _dispatch_forward(
        self,
        forward_op,
        sampling_params_list,
        *,
        stats,
        dp_metadata=None,
        grammar_inputs=None,
    ):
        """Execute one forward step; return (results, on_first_token).

        results is None when the step produces no model output (Path 2/3).
        Both event_loop and event_loop_overlap call this method; they differ
        only in *when* they call post_process on the returned results.

        Path 1 — no PD:              run forward, return (results, None)
        Path 2 — decode, extend:     trigger RDMA receive, return (None, None)
        Path 3 — prefill, decode:    send KV to decode side, return (None, None)
        Path 4 — prefill, extend:    run prefill forward, return (results, on_first_token)
        """
        # Scheduler stats also carry flat-pool observability payloads. Keep the
        # ModelExecutor boundary limited to the three legacy logging scalars.
        num_active_pages = stats["num_active_pages"]
        num_cached_pages = stats["num_cached_pages"]
        num_queue_reqs = stats["num_queue_reqs"]
        dp_global_num_tokens = (
            dp_metadata.global_num_tokens if dp_metadata is not None else None
        )
        dp_global_bs = (
            dp_metadata.global_batch_size if dp_metadata is not None else None
        )
        dp_all_decode_or_idle = (
            dp_metadata.all_decode_or_idle if dp_metadata is not None else False
        )
        dp_all_extend = dp_metadata.all_extend if dp_metadata is not None else False
        multimodal_context = self._get_multimodal_context_for_forward(forward_op)

        self.model_executor.update_block_table(forward_op)

        if self.kv_transfer is None:
            # Path 1: normal (no disaggregation)
            self.model_executor.reset_valid_cache_length(forward_op)
            return (
                self.model_executor.execute_forward_op_with_log(
                    forward_op,
                    sampling_params_list,
                    num_active_pages=num_active_pages,
                    num_cached_pages=num_cached_pages,
                    num_queue_reqs=num_queue_reqs,
                    dp_global_num_tokens=dp_global_num_tokens,
                    dp_global_bs=dp_global_bs,
                    dp_all_decode_or_idle=dp_all_decode_or_idle,
                    dp_all_extend=dp_all_extend,
                    grammar_inputs=grammar_inputs,
                    multimodal_context=multimodal_context,
                ),
                None,
            )

        elif isinstance(self.kv_transfer, DisaggDecodeExecutor):
            # Decode node
            if forward_op.num_extends() > 0:
                # Path 2: new requests waiting for remote KV — trigger RDMA receive
                self.kv_transfer.reset_valid_cache_length(
                    forward_op,
                    self.model_executor.runtime_states,
                    self.model_executor.execution_stream,
                    self.model_executor.device,
                )
                self.kv_transfer.execute(forward_op)
                self.model_executor.reset_remote_prefill_mamba_inputs(forward_op)
                return None, None
            else:
                # Path 3b: decode batch — normal forward
                self.model_executor.reset_valid_cache_length(forward_op)
                return (
                    self.model_executor.execute_forward_op_with_log(
                        forward_op,
                        sampling_params_list,
                        num_active_pages=num_active_pages,
                        num_cached_pages=num_cached_pages,
                        num_queue_reqs=num_queue_reqs,
                        dp_global_num_tokens=dp_global_num_tokens,
                        dp_global_bs=dp_global_bs,
                        dp_all_decode_or_idle=dp_all_decode_or_idle,
                        dp_all_extend=dp_all_extend,
                        multimodal_context=multimodal_context,
                    ),
                    None,
                )

        else:
            # Prefill node (only reached from event_loop, never event_loop_overlap)
            if not isinstance(self.kv_transfer, DisaggPrefillExecutor):
                raise TypeError("kv_transfer must be a DisaggPrefillExecutor.")
            if forward_op.num_extends() == 0:
                # Path 3: all prefill done — send KV to decode side
                self.kv_transfer.execute(forward_op)
                return None, None
            else:
                # Path 4: extend batch — run prefill forward
                self.model_executor.reset_valid_cache_length(forward_op)
                self.kv_transfer.prepare_prefill(forward_op)
                # EPD invariant: handshaked items are filled by the async
                # EPD admission drain before admission; assert none reached
                # the forward un-received (no-op for non-EPD / text-only requests).
                self._assert_epd_embeddings_received(multimodal_context)
                return (
                    self.model_executor.execute_forward_op_with_log(
                        forward_op,
                        sampling_params_list,
                        num_active_pages=num_active_pages,
                        num_cached_pages=num_cached_pages,
                        num_queue_reqs=num_queue_reqs,
                        dp_global_num_tokens=dp_global_num_tokens,
                        dp_global_bs=dp_global_bs,
                        dp_all_decode_or_idle=dp_all_decode_or_idle,
                        dp_all_extend=dp_all_extend,
                        grammar_inputs=grammar_inputs,
                        multimodal_context=multimodal_context,
                        capture_next_input_ids=True,
                    ),
                    self.kv_transfer.store_prefill_token,
                )

    def _get_multimodal_context_for_forward(self, forward_op):
        if not self.model_config.is_multimodal_active:
            return None

        num_extends = forward_op.num_extends()
        mm_inputs = []
        has_mm = False
        for index, rid in enumerate(forward_op.request_ids):
            state = self.output_processor.rid_to_state.get(rid)
            if state is not None and index < num_extends:
                state.maybe_extend_multimodal_mrope_positions()
            item = getattr(state, "multimodal_inputs", None) if state else None
            mm_inputs.append(item)
            has_mm = has_mm or item is not None
        if not has_mm:
            return None

        from tokenspeed.runtime.multimodal.inputs import MultimodalForwardContext

        return MultimodalForwardContext(
            mm_inputs=mm_inputs,
            extend_prefix_lens=list(forward_op.extend_prefix_lens),
            extend_seq_lens=list(forward_op.input_lengths[:num_extends]),
        )

    def _build_mamba_layerwise_cow(
        self, execution_plan, forward_op
    ) -> dict[int, list[int]]:
        if forward_op is None:
            return {}
        loaded_mamba_slots: set[int] = set()
        for cache_op in execution_plan.cache:
            if not isinstance(cache_op, Cache.LoadBackOp):
                continue
            dst_by_kind = getattr(cache_op, "dst_pages_by_kind", None)
            if dst_by_kind is None:
                dst_groups = getattr(cache_op, "dst_pages", [])
            else:
                dst_groups = dst_by_kind.get(CacheKind.MAMBA.value, [])
            for dst_pages in dst_groups:
                loaded_mamba_slots.update(int(page) for page in dst_pages)
        if not loaded_mamba_slots:
            return {}

        cow_src_indices = getattr(forward_op, "mamba_cow_src_indices", None)
        working_indices = getattr(forward_op, "mamba_pool_indices", None)
        if cow_src_indices is None or working_indices is None:
            return {}

        cow_by_src: dict[int, list[int]] = {}
        for cow_src, working in zip(list(cow_src_indices), list(working_indices)):
            cow_src = int(cow_src)
            working = int(working)
            if cow_src < 0 or working < 0 or cow_src not in loaded_mamba_slots:
                continue
            cow_dsts = cow_by_src.setdefault(cow_src, [])
            if working not in cow_dsts:
                cow_dsts.append(working)
        return cow_by_src

    def _submit_cache_ops(self, execution_plan, forward_op) -> None:
        if self.memory_executor is None:
            return
        mamba_layerwise_cow = self._build_mamba_layerwise_cow(
            execution_plan, forward_op
        )
        if mamba_layerwise_cow:
            self.model_executor.set_layerwise_mamba_cow_done(mamba_layerwise_cow)
            self.memory_executor.set_mamba_layerwise_cow(mamba_layerwise_cow)
        self.memory_executor.submit_plan(execution_plan)
        for op in execution_plan.cache:
            if isinstance(op, Cache.WriteBackOp):
                self._num_inflight_cache_ops += len(op.op_ids)
            elif isinstance(op, Cache.LoadBackOp):
                # Radix loadbacks are fire-and-forget (no ack, nothing in
                # flight); the flat host tier acks one LoadBackDone per op_id.
                if self._loadback_acks_expected:
                    self._num_inflight_cache_ops += len(op.op_ids)
            elif isinstance(op, (Cache.PrefetchOp, Cache.BackUpOp)):
                self._num_inflight_cache_ops += 1
            else:
                raise ValueError(f"unsupported cache op kind: {type(op).__name__}")
        self._setup_layerwise_loadback(execution_plan)

    def _setup_layerwise_loadback(self, execution_plan) -> None:
        host_exec = getattr(self.memory_executor, "host_exec", None)
        available_pools = (
            getattr(host_exec, "pools", {}) if host_exec is not None else {}
        )
        consumer_indices_by_kind: dict[CacheKind, list[int]] = {
            kind: [] for kind in available_pools
        }
        for cache_op in execution_plan.cache:
            if isinstance(cache_op, Cache.LoadBackOp):
                for op_id in cache_op.op_ids:
                    for kind in consumer_indices_by_kind:
                        producer_idx = self.memory_executor.get_producer_index(
                            kind, op_id
                        )
                        if (
                            producer_idx is not None
                            and producer_idx not in consumer_indices_by_kind[kind]
                        ):
                            consumer_indices_by_kind[kind].append(producer_idx)
        for kind, consumer_indices in consumer_indices_by_kind.items():
            self.memory_executor.set_consumer(
                kind, consumer_indices if consumer_indices else -1
            )

    def _flush_mamba_retract_states(self, forward_op) -> None:
        """Copy draft->working mamba states when retract occurred (no forward scheduled)."""
        if forward_op is not None:
            return
        if self.model_executor.drafter is None:
            return
        if self.model_executor.runtime_states.mamba_pool is None:
            return
        self.model_executor.flush_mamba_draft_to_working_on_retract()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_model_config(
        self, model_path: str, is_draft_worker: bool = False
    ) -> ModelConfig:
        server_args = self.server_args
        quantization = server_args.quantization
        if is_draft_worker:
            quantization = server_args.speculative_draft_model_quantization
        return ModelConfig(
            model_path,
            trust_remote_code=server_args.trust_remote_code,
            revision=server_args.revision,
            context_length=server_args.max_model_len,
            model_override_args=server_args.hf_overrides,
            dtype=server_args.dtype,
            quantization=quantization,
            server_args=server_args,
            is_draft_worker=is_draft_worker,
        )

    def _init_distributed(self) -> float:
        max_num_input_tokens = (
            self.server_args.chunked_prefill_size
            if self.server_args.chunked_prefill_size > 0
            else self.server_args.max_prefill_tokens + self.server_args.max_model_len
        )
        distributed_config = DistributedConfig.from_server_args(
            server_args=self.server_args,
            port_args=self.port_args,
            gpu_id=self.gpu_id,
            global_rank=self.global_rank,
            hidden_size=self.model_config.hidden_size,
            max_num_tokens=max_num_input_tokens,
        )
        return DistributedInitializer.initialize(distributed_config)

    def _init_interprocess_comm(self):
        context = zmq.Context(2)
        if self.attn_tp_rank == 0:
            self.recv_from_tokenizer = get_zmq_socket(
                context, zmq.PULL, self.port_args.scheduler_input_ipc_name, False
            )
            self.send_to_tokenizer = get_zmq_socket(
                context, zmq.PUSH, self.port_args.tokenizer_ipc_name, False
            )
        else:
            self.recv_from_tokenizer = None
            self.send_to_tokenizer = _NullSender()

    # ------------------------------------------------------------------
    # Shared step helpers
    # ------------------------------------------------------------------

    def _reap_or_keep_buffered_spec(self, spec) -> bool:
        """Resolve a buffered spec on resume; return True if it should be admitted.

        A buffered spec was already registered in ``rid_to_state`` before it was
        withheld, so if it was aborted while paused it never reached the
        scheduler and the forward path can never reap it. Handle that here:

        - state missing  -> already published and reaped; drop silently.
        - state finished -> aborted in place. Stream a terminating finish for
          pause-initiated aborts (the passive client is still waiting) and drop
          the registered state so the rid does not leak; client-initiated aborts
          already tore down their own state, so just reap.
        - otherwise      -> still live; admit it.
        """
        state = self.output_processor.rid_to_state.get(spec.request_id)
        if state is None:
            return False
        if state.finished:
            self.output_processor.reap_finished_orphan(spec.request_id, state)
            return False
        return True

    def _process_new_requests(self):
        recv_reqs = self.request_handler.recv_reqs()
        # Snapshot the pause state before dispatch: process_requests may flip it
        # mid-batch. If it was not blocked before but is after, a pause control
        # message was processed in this very batch — which is what makes the
        # FIFO edge below detectable (see TODO(pause-fifo)).
        pause_blocked_before = self._pause.admit_blocked
        new_req_specs, new_req_states, bootstrap_infos, abort_rids = (
            self.request_handler.process_requests(recv_reqs)
        )
        # Sweep TTL-expired abort markers every iteration. Without this
        # the map only gets cleaned inside ``mark_abort``, so a burst of
        # stale-cancel traffic followed by silence leaves the last batch
        # of entries sitting past their TTL (and potentially re-aborting
        # reused rids). Amortized O(1): expired entries are always at
        # the front of the insertion-ordered dict.
        self.output_processor.sweep_pending_aborts()
        # Abort both registered and grammar-queued requests. Without the
        # grammar_manager.mark_abort call, a request aborted mid-compile
        # would finish compiling and get admitted before being noticed.
        grammar_manager = self.request_handler.grammar_manager
        for rid in abort_rids:
            self.output_processor.mark_abort(rid)
            grammar_manager.mark_abort(rid)

        # A pause(mode="abort") cancels every in-flight request through the same
        # marker path as a client abort; they finish on their next scheduled
        # step, then the drain check resolves the pause reply.
        if self._pause.consume_abort_all():
            for rid in list(self.output_processor.rid_to_state.keys()):
                # notify_client=True: pause aborts a passive client's request,
                # so it must receive a terminating finish (unlike a client abort).
                self.output_processor.mark_abort(rid, notify_client=True)
                grammar_manager.mark_abort(rid)

        # abort/wait also cancel requests still compiling in the grammar queue:
        # they are not yet in rid_to_state or the scheduler, so the sweep above
        # and the drain check both miss them. A finished state makes the next
        # get_ready_grammar_requests pass publish them instead of admitting, so
        # they never run under post-resume weights or strand the drain.
        if self._pause.consume_cancel_grammar():
            for _, state, _ in grammar_manager.grammar_queue:
                state.set_finish_with_abort("Aborted by pause", notify_client=True)

        # On resume, flush specs buffered while paused even when no new request
        # arrives this iteration. This must run before the ``if not ready:
        # return`` guard below, which would otherwise strand buffered specs
        # until the next inbound request. Specs aborted while paused are reaped
        # in place (terminating finish + state cleanup) rather than admitted, so
        # they don't burn a scheduler slot or leak their rid — see
        # ``_reap_or_keep_buffered_spec``.
        if not self._pause.admit_blocked and self._pause.buffered_specs:
            specs = [
                spec
                for spec in self._pause.take_buffered_specs()
                if self._reap_or_keep_buffered_spec(spec)
            ]
            if specs:
                self.scheduler.submit_requests(specs)

        # Partition new requests by grammar readiness. Compile-bound requests
        # are queued in GrammarManager and admitted in a later iteration when
        # their futures resolve (see _drain_ready_grammar_requests below).
        ready = []
        for spec, state, bootstrap in zip(
            new_req_specs, new_req_states, bootstrap_infos
        ):
            # Requests pre-marked finished (e.g. invalid session ID aborted
            # in RequestHandler) skip grammar compilation entirely — we'd
            # just be wasting a compile slot on a response we're about to
            # abort anyway, and the terminal response would be delayed by
            # the compile/timeout window.
            if state.finished:
                ready.append((spec, state, bootstrap))
                continue
            if grammar_manager.process_req_with_grammar(state):
                ready.append((spec, state, bootstrap))
            else:
                grammar_manager.add_to_queue(spec, state, bootstrap)

        # Drain any previously-queued requests whose grammar just finished
        # compiling. With attn_tp > 1 this also drives the per-iter all_gather
        # that keeps grammar admission in sync across ranks.
        ready.extend(grammar_manager.get_ready_grammar_requests())

        if not ready:
            return

        admitted_specs = []
        for spec, state, bootstrap in ready:
            # Grammar-aborted (invalid grammar, timed-out compile, or missing
            # backend) requests must not enter the scheduler — they have no
            # valid grammar to mask logits with, and we don't want to spend a
            # prefill slot on a request that's already finished. Publish the
            # finish_reason directly so the client still gets a response.
            if state.finished:
                self.output_processor.publish_finished_at_admission(
                    spec.request_id, state
                )
                continue

            if isinstance(self.kv_transfer, DisaggDecodeExecutor):
                state.computed_length = state.input_length
            self.output_processor.register(spec.request_id, state)
            is_epd = self.epd_admission is not None and self._is_epd_request(state)
            # EPD: DEFER the P->D sender registration to admission (in
            # EpdPrefillAdmission.drain, just before submit_requests). Registering
            # it now -- while the request is staged and NOT yet in the C++ scheduler
            # -- would let DisaggPrefillExecutor.generate_events poll the sender and
            # emit a BootstrappedEvent that the scheduler's requests_.at(rid) THROWS
            # on (no such request yet). Non-EPD requests register now (submitted this
            # same call).
            if self.kv_transfer is not None and not is_epd:
                self.kv_transfer.register(spec.request_id, bootstrap)

            if self.memory_executor is not None:
                hashes = calc_l3_query_hashes(self.scheduler, spec.tokens)
                if hashes and len(hashes) > self.prefetch_threshold:
                    hit_pages = self.memory_executor.query_l3_pages(hashes)
                    logger.debug(
                        "[cache_op] L3 query: rid=%s hash_pages=%s hit_pages=%s threshold=%s",
                        spec.request_id,
                        len(hashes),
                        hit_pages,
                        self.prefetch_threshold,
                    )
                    spec.rolling_hashes = hashes
                    spec.storage_hit_pages = hit_pages
            # EPD prefill: hold a request whose images are encode-routed OUT of the
            # scheduler until its per-image embeddings have been received (started
            # here, polled in EpdPrefillAdmission.drain, which registers the P->D
            # sender + submits once ready). It is output_processor-registered above;
            # the sender registration + submission are both deferred. Non-EPD
            # requests admit immediately as before. Rank-identical because `ready` is
            # rank-synced (recv_reqs broadcast + grammar gather).
            if is_epd:
                self.epd_admission.stage(
                    spec.request_id, state.multimodal_inputs.mm_items
                )
                self._epd_staged[spec.request_id] = (spec, state, bootstrap)
            else:
                admitted_specs.append(spec)

        # Pause gate: while paused, withhold new requests from the scheduler
        # (running requests keep stepping); buffered specs are flushed on resume
        # above, ahead of any newly-admitted ones, preserving FIFO order.
        #
        # TODO(pause-fifo): recv_reqs() drains the socket non-blocking, so a
        # generate request that arrived *before* a pause control message can be
        # coalesced into the same batch and reach here after the pause flipped
        # admit_blocked. Such a pre-pause request is buffered as post-pause work
        # instead of running (wait) / being aborted (abort). Correct handling
        # needs the batch processed as an ordered stream that respects the
        # control request's FIFO position. Tracked as a follow-up; until then we
        # warn when the coalescing condition is observed so it is not silent.
        if self._pause.admit_blocked:
            if admitted_specs and not pause_blocked_before:
                logger.warning(
                    "Pause engaged in the same recv batch as %d generate "
                    "request(s) (rids=%s); their FIFO order relative to the "
                    "pause is not preserved, so a pre-pause request may be "
                    "buffered as post-pause work and run only after resume. "
                    "See TODO(pause-fifo).",
                    len(admitted_specs),
                    [spec.request_id for spec in admitted_specs],
                )
            self._pause.buffer_specs(admitted_specs)
            return

        if admitted_specs:
            self.scheduler.submit_requests(admitted_specs)

    @nvtx_range("loop:commit", color="rapids")
    def _commit_forward_results(
        self,
        forward_op,
        results: ModelExecutionResult,
        on_first_token=None,
    ):
        self.request_handler.forward_ct += 1
        forward_mode = ForwardMode.from_num_extends(
            forward_op.num_extends(),
            len(forward_op.request_ids),
        )
        self.request_handler._profile_batch_predicate(forward_mode)

        # post_process_forward_op calls sync() — after this, CPU tensors are ready
        is_prefill_instance = isinstance(self.kv_transfer, DisaggPrefillExecutor)
        request_changes = self.output_processor.post_process_forward_op(
            forward_op,
            results,
            is_prefill_instance=is_prefill_instance,
            on_first_token=on_first_token,
        )

        # Accumulate decode stats from synced results (no GPU sync)
        if forward_op.num_extends() <= 0:
            bs = len(forward_op.request_ids)
            self.model_executor.accumulate_decode_stats(results, bs)

        return request_changes

    def _get_forward_op(self, execution_plan):
        """Return the next forward op from the given plan, or None if there is nothing to run."""
        forward_ops = execution_plan.forward
        if len(forward_ops) == 0 or len(forward_ops[0].request_ids) == 0:
            return None
        return forward_ops[0]

    def _handle_flat_oom_terminals(self, execution_plan) -> None:
        """Surface flat-KV OOM terminals to their clients as abort finishes.

        The C++ flat scheduler terminalizes a request that can never fit the
        flat pool (AbortEvent inside the scheduler; the reaper reclaims its
        resources) and reports its id on ``plan.flat_oom_request_ids``
        (always empty on radix builds). The scheduler side is already fully
        torn down — do NOT send a ForwardEvent.Abort back — but the client is
        still waiting on the response stream, so finish the request with an
        abort here (mirrors the PD FailedEvent handling above, minus the
        scheduler abort).
        """
        oom_rids = getattr(execution_plan, "flat_oom_request_ids", None)
        if not oom_rids:
            return
        for rid in oom_rids:
            state = self.output_processor.rid_to_state.get(rid)
            if state is None:
                # rid already gone (e.g. a client abort raced ahead).
                logger.debug(
                    "flat OOM terminal for rid=%s: state missing; skipping", rid
                )
                continue
            if state.finished:
                # Already carries a finish (an abort raced ahead of the
                # terminal). C++ reports this rid exactly once and no future
                # forward op will reap it, so resolve it here (same orphan
                # rule as _reap_or_keep_buffered_spec).
                self.output_processor.reap_finished_orphan(rid, state)
                continue
            state.set_finish_with_abort(
                "flat KV cache cannot fit this request: prompt exceeds pool "
                "capacity (OOM)"
            )
            self.output_processor.publish_finished_at_admission(rid, state)

    def _process_kv_transfer_events(self, kv_transfer_events: list) -> list:
        processed = []
        for event in kv_transfer_events:
            processed.append(event)
            if isinstance(event, PD.SucceededEvent) and isinstance(
                self.kv_transfer, DisaggPrefillExecutor
            ):
                req_id = event.request_id
                processed.extend(self.output_processor.finish_prefill_request(req_id))
            elif isinstance(event, PD.RemotePrefillDoneEvent):
                req_id = event.request_id
                bootstrap_token = event.bootstrap_token
                self.output_processor.on_remote_prefill_done(req_id, bootstrap_token)
                if isinstance(self.kv_transfer, DisaggDecodeExecutor):
                    candidate_info = self.kv_transfer.pop_remote_spec_candidate_ids(
                        req_id
                    )
                    if candidate_info is not None:
                        req_pool_idx, candidate_ids = candidate_info
                        self.model_executor.write_remote_spec_candidate_ids(
                            req_pool_idx, candidate_ids
                        )
            elif isinstance(event, PD.FailedEvent):
                # A PD/EPD transfer failed: the decode KV receiver timed out (e.g. the
                # prefill aborted on embedding timeout so the KV never arrives), or a
                # transfer errored. The C++ scheduler's FailedEvent handler is a no-op,
                # so without this the request is never finished and the CLIENT HANGS
                # FOREVER (the decode is its response stream). Finish it with an abort
                # (streams the error to the client) and abort it in the scheduler so its
                # slot/KV is freed. AbortEvent is valid from the decode-waiting state
                # (forward_events.cpp: AbortEvent(Prefilling&&) -> Finished).
                req_id = event.request_id
                state = self.output_processor.rid_to_state.get(req_id)
                if state is not None:
                    state.set_finish_with_abort(
                        "PD/EPD remote transfer failed or timed out"
                    )
                    self.output_processor.publish_finished_at_admission(req_id, state)
                    abort = ForwardEvent.Abort()
                    abort.request_id = req_id
                    processed.append(abort)

        return processed

    def _get_internal_state(self) -> dict:
        """Return read-only scheduler/cache identity for diagnostics and gates.

        The existing ``GetInternalState`` control request already fans out to
        every attention rank.  Keeping the snapshot behind that cold control
        seam avoids adding work to scheduling/model-forward paths while making
        compile-time flat/radix selection and per-pool reset observable.
        """

        state = {
            "scheduler_flat_kvcache": scheduler_ext_flat_kvcache(),
            "overlap_schedule_depth": int(self.overlap_schedule_depth),
        }
        graph_evidence_fn = getattr(
            getattr(self, "model_executor", None),
            "cuda_graph_runtime_evidence",
            None,
        )
        if callable(graph_evidence_fn):
            state["cuda_graph_evidence"] = graph_evidence_fn()
        if not state["scheduler_flat_kvcache"]:
            return state

        snapshot_fn = getattr(self.scheduler, "flat_pool_snapshots", None)
        if callable(snapshot_fn):
            state["flat_pool_snapshots"] = [
                {
                    "pool_id": str(snapshot.pool_id),
                    **{
                        name: int(getattr(snapshot, name))
                        for name in (
                            "total_blocks",
                            "usable_blocks",
                            "free_blocks",
                            "active_blocks",
                            "cached_evictable_blocks",
                            "pinned_cached_blocks",
                            "reserved_blocks",
                            "bytes_per_block",
                        )
                    },
                }
                for snapshot in snapshot_fn()
            ]

        generation_fn = getattr(self.scheduler, "flat_kv_generation", None)
        quiescent_fn = getattr(self.scheduler, "flat_kv_cache_quiescent", None)
        if callable(generation_fn):
            state["flat_kv_generation"] = int(generation_fn())
        if callable(quiescent_fn):
            state["flat_kv_quiescent"] = bool(quiescent_fn())

        arena = self._flat_cache_arena()
        if arena is not None:
            state["flat_arena_generation"] = int(arena.arena_generation)

        # Shared arenas own the already-validated plan. Pools without an arena
        # may expose a single flat plan for cold-path diagnostics.
        plan = getattr(arena, "plan", None) if arena is not None else None
        if plan is None:
            plan = next(
                (
                    candidate
                    for pool in self._kv_pools()
                    if (candidate := getattr(pool, "flat_memory_plan", None))
                    is not None
                ),
                None,
            )
        if plan is not None:
            state["flat_plan_fingerprint"] = str(plan.plan_fingerprint)
            state["flat_kv_plan"] = {
                "pools": [
                    {
                        "pool_id": str(pool.pool_id),
                        "total_blocks": int(pool.total_blocks),
                        "bytes_per_block": int(pool.bytes_per_block),
                        "storage_schema_hash": str(pool.storage_schema_hash),
                    }
                    for pool in plan.pools
                ],
                "groups": [
                    {
                        "group_id": str(spec.group_id),
                        "pool_id": str(spec.pool_id),
                        "rows_per_page": int(spec.rows_per_page),
                        "entry_stride_tokens": int(spec.entry_stride_tokens),
                        "owner_mask": int(spec.owner_mask),
                    }
                    for spec in plan.scheduler_group_specs
                ],
            }
        return state

    def _get_load(self):
        """Return load metrics for the DP load balancer."""
        from tokenspeed.runtime.engine.io_struct import GetLoadReqOutput

        num_total_pages = self.max_total_num_tokens // self.server_args.block_size
        flat_aggregate = self._flat_pool_aggregate()
        if flat_aggregate is None:
            available = self.scheduler.available_kv_pages()
            num_used_pages = num_total_pages - available
        else:
            # DP admission pressure is governed by the tightest component
            # pool, not a sum of unrelated local page-ID domains.
            _, num_used_pages = flat_pool_equivalent_pages(
                flat_aggregate,
                equivalent_total_pages=num_total_pages,
            )
        num_waiting = self.scheduler.waiting_size()
        # num_reqs: running + waiting (used by SHORTEST_QUEUE balancing)
        num_running = len(self.output_processor.rid_to_state)
        return GetLoadReqOutput(
            dp_rank=self.dp_rank,
            num_reqs=num_running + num_waiting,
            num_waiting_reqs=num_waiting,
            num_pages=num_used_pages,
        )

    def _flat_pool_aggregate(self):
        aggregate_fn = getattr(self.scheduler, "flat_pool_aggregate", None)
        if not callable(aggregate_fn):
            return None
        return aggregate_fn()

    def _collect_flat_pool_metrics_if_due(
        self,
        equivalent_total_pages: int,
    ):
        if not self.metrics.enabled:
            return ()
        now = time.monotonic()
        if now < getattr(self, "_next_flat_pool_detail_at", 0.0):
            return ()
        self._next_flat_pool_detail_at = now + _FLAT_POOL_DETAIL_INTERVAL_S
        snapshot_fn = getattr(self.scheduler, "flat_pool_snapshots", None)
        if not callable(snapshot_fn):
            return ()
        snapshots = snapshot_fn()
        if not snapshots:
            return ()
        _, pool_metrics = collect_flat_pool_metrics(
            snapshots,
            equivalent_total_pages=equivalent_total_pages,
        )
        return pool_metrics

    def _dp_sync_and_check(self, forward_op) -> DpForwardMetadata:
        """Synchronize DP ranks with CPU-only metadata.

        All ranks call this before GPU forward work. The gathered metadata is
        used for eager token-aware collectives and for choosing a common padded
        CUDA graph shape during decode.
        """
        import torch.distributed as dist

        executes_model_forward = _forward_op_executes_model_forward(
            forward_op,
            is_disagg_decode=isinstance(self.kv_transfer, DisaggDecodeExecutor),
        )
        num_tokens = sum(forward_op.input_lengths) if executes_model_forward else 0
        batch_size = len(forward_op.request_ids) if executes_model_forward else 0
        if not executes_model_forward:
            forward_mode = ForwardMode.IDLE
        else:
            forward_mode = ForwardMode.from_num_extends(
                forward_op.num_extends(),
                batch_size,
            )

        self._dp_local_info[0, 0] = num_tokens
        self._dp_local_info[0, 1] = batch_size
        self._dp_local_info[0, 2] = int(forward_mode)
        dist.all_gather_into_tensor(
            self._dp_global_info,
            self._dp_local_info,
            group=self.world_cpu_group,
        )
        global_num_tokens = self._dp_global_info[:, 0].tolist()
        global_batch_size = self._dp_global_info[:, 1].tolist()
        global_forward_mode = self._dp_global_info[:, 2].tolist()
        any_rank_has_work = max(global_num_tokens) > 0
        need_idle_forward = num_tokens == 0 and any_rank_has_work
        all_decode_or_idle = all(
            mode
            in (
                int(ForwardMode.DECODE),
                int(ForwardMode.IDLE),
            )
            for mode in global_forward_mode
        )
        # Replicated prefill-graph gate (see PrefillGraph._select_bucket).
        all_extend = all(
            mode == int(ForwardMode.EXTEND) for mode in global_forward_mode
        )
        return DpForwardMetadata(
            global_num_tokens=global_num_tokens,
            global_batch_size=global_batch_size,
            global_forward_mode=global_forward_mode,
            all_decode_or_idle=all_decode_or_idle,
            all_extend=all_extend,
            need_idle_forward=need_idle_forward,
        )

    def _get_scheduler_stats(self):
        """Query scheduler for page usage and queue depth."""
        num_total_pages = self.max_total_num_tokens // self.server_args.block_size
        flat_aggregate = self._flat_pool_aggregate()
        if flat_aggregate is None:
            available = self.scheduler.available_kv_pages()
            active = self.scheduler.active_kv_pages()
            cached = num_total_pages - available
            flat_summary = None
            flat_pool_metrics = ()
        else:
            if self.metrics.enabled:
                flat_summary = summarize_flat_pool_aggregate(
                    flat_aggregate,
                    equivalent_total_pages=num_total_pages,
                )
                active = int(flat_summary["active_equivalent_pages"])
                cached = int(flat_summary["pressure_equivalent_pages"])
            else:
                active, cached = flat_pool_equivalent_pages(
                    flat_aggregate,
                    equivalent_total_pages=num_total_pages,
                )
                flat_summary = None
            flat_pool_metrics = self._collect_flat_pool_metrics_if_due(num_total_pages)
        return {
            "num_active_pages": active,
            "num_cached_pages": cached,
            "num_queue_reqs": self.scheduler.waiting_size(),
            "flat_summary": flat_summary,
            "flat_pool_metrics": flat_pool_metrics,
        }

    def _record_scheduler_iteration_metrics(
        self, stats: dict, num_iteration_tokens: int
    ) -> None:
        self.metrics.record_scheduler_iteration(
            running=len(self.output_processor.rid_to_state),
            waiting=stats["num_queue_reqs"],
            num_active_pages=stats["num_active_pages"],
            num_total_pages=self.max_total_num_tokens // self.server_args.block_size,
            num_iteration_tokens=num_iteration_tokens,
            flat_summary=stats.get("flat_summary"),
            flat_pool_metrics=stats.get("flat_pool_metrics", ()),
        )

    # ------------------------------------------------------------------
    # Pause / resume helpers
    # ------------------------------------------------------------------

    def _reset_caches_for_release(self) -> None:
        """Invalidate cache metadata before KV is discarded on release.

        Pools without a shared arena use their legacy prefix-cache reset when
        available. A shared flat arena requires the scheduler to be quiescent,
        invalidates every cached block, and advances its pool-set generation.
        That generation is retained until the matching wake repairs page 0.
        """
        arena = self._flat_cache_arena()
        if arena is None:
            reset = getattr(self.scheduler, "reset_prefix_cache", None)
            if callable(reset):
                reset()
            return
        if self._flat_kv_release_generation is not None:
            raise RuntimeError("flat KV cache already has an unrepaired release")
        reset_flat = getattr(self.scheduler, "reset_flat_kv_cache", None)
        if not callable(reset_flat):
            raise RuntimeError(
                "flat KV release requires scheduler reset_flat_kv_cache()"
            )
        generation = reset_flat()
        if (
            isinstance(generation, bool)
            or not isinstance(generation, int)
            or generation <= 0
        ):
            raise RuntimeError(
                "scheduler reset_flat_kv_cache() returned an invalid generation: "
                f"{generation!r}"
            )
        self._flat_kv_release_generation = generation

    def _kv_pools(self) -> list:
        """All KV pools whose pages are tagged ``kv_cache`` — the target pool and
        the draft pool in speculative-decoding runs. Release/repair must walk the
        SAME set, so both derive it here rather than enumerating pools by hand."""
        pools = []
        for attr in ("token_to_kv_pool", "draft_token_to_kv_pool"):
            pool = getattr(self.model_executor, attr, None)
            if pool is not None:
                pools.append(pool)
        return pools

    def _flat_cache_arena(self) -> object | None:
        """Return the shared flat device-cache arena, if configured.

        Target and draft pools are owner views over one union plan. Multiple
        distinct arenas would make scheduler generation/reset ownership
        ambiguous, so reject that topology before any release mutation.
        """
        arena = None
        for pool in self._kv_pools():
            candidate = getattr(pool, "device_cache_arena", None)
            if candidate is None:
                continue
            if arena is not None and candidate is not arena:
                raise RuntimeError(
                    "target/draft flat KV pools must share one device-cache arena"
                )
            arena = candidate
        return arena

    def _kv_repair_after_wake(self) -> None:
        """Zero re-mapped KV buffers (garbage after re-map) for every KV pool,
        including the draft pool in spec-decode runs — its allocations are tagged
        ``kv_cache`` too, so a wake that skipped it would feed the draft model
        stale KV. FP8 KV scales ride with the weights region, so no scale reset
        is needed here."""
        arena = self._flat_cache_arena()
        if arena is not None:
            generation = self._flat_kv_release_generation
            if generation is None:
                raise RuntimeError(
                    "flat KV wake has no matching scheduler reset generation"
                )
            repaired_generation = arena.repair_after_wake(
                expected_generation=generation
            )
            if repaired_generation != generation:
                raise RuntimeError(
                    "flat KV arena returned an unexpected generation: "
                    f"expected={generation}, actual={repaired_generation!r}"
                )
            self._flat_kv_release_generation = None
            return

        for pool in self._kv_pools():
            if hasattr(pool, "clear_kv_buffers"):
                pool.clear_kv_buffers()

    def _paused_idle_step(self, prev_forward_op=None, prev_results=None) -> None:
        """Run one iteration under ``PAUSED_ALL`` (keep mode): no new forward
        work, but keep DP ranks in lockstep, service the drain check, and yield
        the CPU so the freeze does not busy-spin a core."""
        if prev_results is not None:
            request_changes = self._commit_forward_results(
                prev_forward_op, prev_results
            )
            advance_forward(self.scheduler, request_changes)
            self._publish_scheduler_kv_events()

        if self.has_dp:
            dp_metadata = self._dp_sync_and_check(None)
            # While memory is released the weights region is unmapped; an idle
            # forward runs the model and would read freed memory. All DP ranks
            # release together, so skipping the idle forward stays consistent
            # across ranks (the small DP sync above still runs to keep lockstep).
            if dp_metadata.need_idle_forward and not self._pause.released:
                self.model_executor.execute_idle_forward(
                    dp_metadata.global_num_tokens,
                    dp_metadata.global_batch_size,
                    dp_metadata.all_decode_or_idle,
                )

        self._pause.maybe_finish_drain(self.scheduler)
        time.sleep(_PAUSED_IDLE_SLEEP_S)

    # ------------------------------------------------------------------
    # Event loops
    # ------------------------------------------------------------------

    def event_loop(self):
        """Non-overlapping scheduler loop."""
        while True:
            self._process_new_requests()
            # EPD prefill: admit requests whose async embedding receives completed
            # this cycle (rank-synced). Fixed position right after
            # _process_new_requests so the drain's TP collective ordering is
            # rank-identical every cycle.
            self._drain_ready_epd_embeddings()
            self._commit_cache_results()
            if self._pause.forward_blocked:
                self._paused_idle_step()
                continue
            execution_plan = self.scheduler.next_execution_plan()
            self._publish_scheduler_kv_events()
            self._handle_flat_oom_terminals(execution_plan)

            forward_op = self._get_forward_op(execution_plan)
            self._submit_cache_ops(execution_plan, forward_op)
            self._flush_mamba_retract_states(forward_op)

            stats = self._get_scheduler_stats()
            num_iter_tokens = (
                sum(forward_op.input_lengths) if forward_op is not None else 0
            )

            # DP sync: all ranks must participate even when idle.
            dp_metadata = None
            if self.has_dp:
                dp_metadata = self._dp_sync_and_check(forward_op)
                if dp_metadata.need_idle_forward:
                    self.model_executor.execute_idle_forward(
                        dp_metadata.global_num_tokens,
                        dp_metadata.global_batch_size,
                        dp_metadata.all_decode_or_idle,
                    )
                    self._record_scheduler_iteration_metrics(stats, num_iter_tokens)
                    continue

            request_changes = []

            if forward_op is not None:
                sampling_params_list = self._gather_sampling_params(forward_op)
                grammar_inputs = self._gather_grammar_state(forward_op)
                self._mark_stats_scheduled(forward_op)
                results, on_first_token = self._dispatch_forward(
                    forward_op,
                    sampling_params_list,
                    dp_metadata=dp_metadata,
                    stats=stats,
                    grammar_inputs=grammar_inputs,
                )
                if results is not None:
                    request_changes.extend(
                        self._commit_forward_results(
                            forward_op, results, on_first_token
                        )
                    )

            if self.kv_transfer is not None:
                kv_transfer_events = self.kv_transfer.generate_events()
                request_changes.extend(
                    self._process_kv_transfer_events(kv_transfer_events)
                )

            if request_changes:
                advance_forward(self.scheduler, request_changes)
                self._publish_scheduler_kv_events()

            # Resolve a deferred abort/wait pause reply once in-flight work drains.
            self._pause.maybe_finish_drain(self.scheduler)

            self._record_scheduler_iteration_metrics(stats, num_iter_tokens)

    def _mark_stats_scheduled(self, forward_op) -> None:
        # Stamp the pre-forward "scheduled" time on each request's stats tracker
        # so the queue/prefill split is anchored before the forward (idempotent:
        # only the first forward a request appears in sets it). --enable-log-request-stats.
        if not self.server_args.enable_log_request_stats or forward_op is None:
            return
        now = time.time()
        rid_to_state = self.output_processor.rid_to_state
        for rid in forward_op.request_ids:
            st = rid_to_state.get(rid)
            if st is not None:
                st.stats.mark_scheduled(now)

    def _gather_sampling_params(self, forward_op) -> list[SamplingParams]:
        """Look up per-request SamplingParams from the output processor. The
        sampling backend does its own flip detection + RNG state management
        internally, so we only need the scalar params here."""
        return [
            self.output_processor.rid_to_state[rid].sampling_params
            for rid in forward_op.request_ids
        ]

    def _gather_grammar_state(self, forward_op) -> GrammarStepInputs | None:
        """Build ``GrammarStepInputs`` for the current batch, or ``None``.

        Returns ``None`` when no request in this batch has a grammar — the
        model_executor short-circuits then. Otherwise carries the grammars
        list + per-EXTEND-slot ``advance_mask`` (False on intermediate
        chunked-prefill chunks, since the sampled token is discarded by
        post_process and must not advance the matcher).
        """
        rid_to_state = self.output_processor.rid_to_state
        grammars = [rid_to_state[rid].grammar for rid in forward_op.request_ids]
        if not any(grammars):
            return None

        advance_mask = None
        num_extends = forward_op.num_extends()
        if num_extends > 0:
            bs = len(forward_op.request_ids)
            extend_prefix_lens = forward_op.extend_prefix_lens
            extend_input_lengths = forward_op.input_lengths[:num_extends]
            advance_mask = [True] * bs
            for i in range(num_extends):
                rid = forward_op.request_ids[i]
                # This chunk completes prefill iff it processes the final
                # token of the prompt; intermediate chunks don't.
                advance_mask[i] = (
                    extend_prefix_lens[i] + extend_input_lengths[i]
                    >= rid_to_state[rid].input_length
                )

        return GrammarStepInputs(grammars=grammars, advance_mask=advance_mask)

    def event_loop_overlap(self):
        """
        Overlapping scheduler loop: post-process the previous step's results
        while the current step's forward pass is in flight.
        """
        # EPD invariant: the async embedding drain (EpdPrefillAdmission.drain)
        # that admits EPD requests runs ONLY in event_loop(), never here. A
        # prefill node that receives encode embeddings must therefore run the
        # non-overlap loop -- should_use_overlap_schedule enforces this by forcing
        # prefill -> non-overlap. Assert it rather than trusting that external
        # coupling: if a prefill ever reached this loop, every EPD request would
        # stage into the admission controller and hang forever with no drain.
        assert self.epd_admission is None, (
            "EPD prefill must run the non-overlap event_loop(); the embedding "
            "drain is not wired into event_loop_overlap()"
        )
        prev_results: ModelExecutionResult = None
        prev_forward_op = None

        while True:
            # Order this iter's default-stream writes (KVAllocator,
            # update_block_table, prefix_cache writes to req_to_page)
            # after the prev iter's forward on execution_stream that
            # reads the same tensor. Non-blocking on host.
            torch.cuda.default_stream().wait_stream(
                self.model_executor.execution_stream
            )
            self._process_new_requests()
            self._commit_cache_results()
            if self._pause.forward_blocked:
                # Freeze: commit any in-flight (overlapped) step — a forward
                # already on the GPU can't be un-launched — then idle.
                self._paused_idle_step(prev_forward_op, prev_results)
                prev_results = None
                prev_forward_op = None
                continue
            execution_plan = self.scheduler.next_execution_plan()
            self._publish_scheduler_kv_events()
            self._handle_flat_oom_terminals(execution_plan)

            forward_op = self._get_forward_op(execution_plan)
            self._submit_cache_ops(execution_plan, forward_op)
            self._flush_mamba_retract_states(forward_op)

            stats = self._get_scheduler_stats()
            num_iter_tokens = (
                sum(forward_op.input_lengths) if forward_op is not None else 0
            )

            grammar_inputs = None
            if forward_op is not None:
                # Gather both sampling params and grammar state BEFORE the
                # prev_results commit below — that commit can finish requests
                # and pop them from output_processor.rid_to_state, which would
                # KeyError when we look up rids that are still in the current
                # forward_op.
                sampling_params_list = self._gather_sampling_params(forward_op)
                grammar_inputs = self._gather_grammar_state(forward_op)

            # DP sync: all ranks must participate even when idle.
            dp_metadata = None
            if self.has_dp:
                dp_metadata = self._dp_sync_and_check(forward_op)
                if dp_metadata.need_idle_forward:
                    if prev_results is not None:
                        request_changes = self._commit_forward_results(
                            prev_forward_op, prev_results
                        )
                        advance_forward(self.scheduler, request_changes)
                        self._publish_scheduler_kv_events()
                        prev_results = None
                        prev_forward_op = None
                    self.model_executor.execute_idle_forward(
                        dp_metadata.global_num_tokens,
                        dp_metadata.global_batch_size,
                        dp_metadata.all_decode_or_idle,
                    )
                    self._record_scheduler_iteration_metrics(stats, num_iter_tokens)
                    continue

            # ---- dispatch current forward first (async GPU launch) ----
            # Issue curr's forward before committing prev so the GPU runs curr
            # while the CPU syncs/post-processes prev. Committing prev first
            # would block the CPU on prev's copy_event and leave the GPU idle
            # until dispatch — visible as a gap between forwards in the trace.
            #
            # Eager grammar exception: setup_grammar_step reads each matcher's
            # current state to fill the bitmask. Under the overlap pattern the
            # matcher hasn't been advanced yet by prev's accept_token (commit
            # below), so the fill would use a one-step-stale state and let the
            # model sample a token the matcher then rejects. Capturable
            # grammar dodges this with an in-graph hostfunc that advances
            # before fill; eager has no equivalent, so we commit prev first
            # whenever this batch carries grammars. Costs the dispatch/commit
            # overlap for grammar batches but is correct.
            request_changes = []
            curr_has_grammar = grammar_inputs is not None
            eager_grammar_needs_advance = (
                curr_has_grammar
                and prev_results is not None
                and self.model_executor.eager_grammar_buffers is not None
            )
            if eager_grammar_needs_advance:
                request_changes.extend(
                    self._commit_forward_results(prev_forward_op, prev_results)
                )
                prev_results = None
                prev_forward_op = None

            curr_results = None
            if forward_op is not None:
                self._mark_stats_scheduled(forward_op)
                curr_results, _ = self._dispatch_forward(
                    forward_op,
                    sampling_params_list,
                    dp_metadata=dp_metadata,
                    stats=stats,
                    grammar_inputs=grammar_inputs,
                )

            # ---- post-process previous step (overlapped with current forward) ----
            if prev_results is not None:
                request_changes.extend(
                    self._commit_forward_results(prev_forward_op, prev_results)
                )

            # ---- collect KV transfer events ----
            if self.kv_transfer is not None:
                kv_transfer_events = self.kv_transfer.generate_events()
                request_changes.extend(
                    self._process_kv_transfer_events(kv_transfer_events)
                )

            if request_changes:
                advance_forward(self.scheduler, request_changes)
                self._publish_scheduler_kv_events()

            # Resolve a deferred abort/wait pause reply once in-flight work drains.
            self._pause.maybe_finish_drain(self.scheduler)

            self._record_scheduler_iteration_metrics(stats, num_iter_tokens)

            prev_results = curr_results
            prev_forward_op = forward_op


def run_event_loop(
    server_args: ServerArgs,
    port_args: PortArgs,
    pipe_writer,
):
    mapping = server_args.mapping
    gpu_id = mapping.rank % mapping.nprocs_per_node + server_args.base_gpu_id
    attn_tp_rank = mapping.attn.tp_rank
    dp_rank = mapping.attn.dp_rank
    global_rank = mapping.rank

    setproctitle.setproctitle(f"tokenspeed::scheduler_{dp_rank}")
    faulthandler.enable()
    parent_process = psutil.Process().parent()
    register_usr_signal()

    prefix = f" ATTN TP RANK {attn_tp_rank}"
    configure_logger(server_args, prefix=prefix)

    try:
        if server_args.disaggregation_mode == "encode":
            # The encode role is LM-free; run the lightweight vision-tower loop
            # instead of building the full EventLoop (KV/LM scheduler).
            from tokenspeed.runtime.epd.encode_loop import (
                run_encode_loop,
            )

            run_encode_loop(server_args, port_args, pipe_writer, gpu_id, global_rank)
            return

        flat_kvcache_ext = scheduler_ext_flat_kvcache()
        scheduler_backend, flat_build_option = scheduler_backend_identity(
            flat_kvcache_ext
        )
        logger.info(
            "KV cache scheduler backend=%s (TOKENSPEED_FLAT_KVCACHE=%s)",
            scheduler_backend,
            flat_build_option,
        )
        event_loop = EventLoop(
            server_args,
            port_args,
            gpu_id,
            attn_tp_rank,
            dp_rank,
            global_rank,
        )
        pipe_writer.send(
            {
                "status": "ready",
                "max_total_num_tokens": event_loop.max_total_num_tokens,
                "max_req_input_len": event_loop.max_req_input_len,
                "max_num_seqs": server_args.max_num_seqs,
                "chunked_prefill_size": server_args.chunked_prefill_size,
                "max_model_len": event_loop.model_config.context_len,
            }
        )

        if event_loop.has_dp:
            # All DP schedulers must finish initialization before any rank enters
            # the loop and starts the first DP metadata collective.
            dist.barrier(group=event_loop.world_cpu_group)

        if event_loop.use_overlap_schedule:
            event_loop.event_loop_overlap()
        else:
            event_loop.event_loop()

    except Exception:
        traceback = get_exception_traceback()
        logger.error("Scheduler hit an exception: %s", traceback)
        parent_process.send_signal(signal.SIGUSR1)
