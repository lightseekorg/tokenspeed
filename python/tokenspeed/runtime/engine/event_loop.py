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
import threading
import time
from collections import deque
from dataclasses import dataclass

import psutil
import setproctitle
import torch
import torch.distributed as dist
import zmq
from tokenspeed_scheduler import Scheduler

from tokenspeed.runtime.cache.l2.executor import L2CacheExecutor
from tokenspeed.runtime.configs.model_config import ModelConfig
from tokenspeed.runtime.distributed.process_group_manager import (
    process_group_manager as pg_manager,
)
from tokenspeed.runtime.engine.cache_hooks import L2CacheHooks
from tokenspeed.runtime.engine.generation_output_processor import OutputProcesser
from tokenspeed.runtime.engine.io_struct import IpcReceiver, IpcSender, NullSender
from tokenspeed.runtime.engine.load_snapshot import (
    DirectLoadSnapshotSink,
    LoadSnapshotPublisher,
    NullLoadSnapshotPublisher,
)
from tokenspeed.runtime.engine.memory_occupation import MemoryOccupationController
from tokenspeed.runtime.engine.pause import PauseController, PauseHooks
from tokenspeed.runtime.engine.request_handler import RequestHandler
from tokenspeed.runtime.engine.scheduler_utils import (
    advance_scheduler,
    aligned_max_scheduled_tokens,
    log_gpu_memory_summary,
    make_config,
    pool_to_cache_groups,
    resolve_dspark_prefix_replay_tokens,
    scheduler_cache_geometry_from_pool,
    should_use_overlap_schedule,
)
from tokenspeed.runtime.epd.prefill_hooks import EpdPrefillHooks
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
from tokenspeed.runtime.grammar.capturable_grammar import GrammarStepInputs
from tokenspeed.runtime.layers.attention.registry import create_attn_components
from tokenspeed.runtime.metrics.collector import EngineMetrics
from tokenspeed.runtime.multimodal.inputs import multimodal_context_for_forward
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
from tokenspeed.runtime.pd.topology import PDParallelTopology
from tokenspeed.runtime.pd.transfer_hooks import PdTransferHooks
from tokenspeed.runtime.sampling.sampling_params import SamplingParams
from tokenspeed.runtime.utils import (
    configure_logger,
    get_colorful_logger,
    get_zmq_socket,
)
from tokenspeed.runtime.utils.env import envs
from tokenspeed.runtime.utils.exceptions import get_exception_traceback
from tokenspeed.runtime.utils.nvtx import nvtx_range
from tokenspeed.runtime.utils.process import register_usr_signal
from tokenspeed.runtime.utils.server_args import PortArgs, ServerArgs
from tokenspeed.runtime.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter

logger = get_colorful_logger(__name__)


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
        shutdown_event: threading.Event | None = None,
    ) -> None:
        # Do not pass server_args further down the stack after this point.

        self.server_args = server_args
        pd_topology = None
        if server_args.disaggregation_mode != "null":
            pd_topology = PDParallelTopology.from_mapping(server_args.mapping)
            pd_topology.require_cache_pd_supported()
        self.port_args = port_args
        self.gpu_id = gpu_id
        self.global_rank = global_rank
        self.shutdown_event = shutdown_event or threading.Event()

        self.model_config = self._load_model_config(server_args.model)
        if server_args.speculative_draft_model_path is not None:
            draft_model_config = self._load_model_config(
                server_args.speculative_draft_model_path,
                is_draft_worker=True,
            )
        else:
            draft_model_config = None

        prefix_replay_tokens = resolve_dspark_prefix_replay_tokens(
            speculative_algorithm=server_args.speculative_algorithm,
            enable_prefix_caching=server_args.enable_prefix_caching,
            enable_kvstore=server_args.enable_kvstore,
            disaggregation_mode=server_args.disaggregation_mode,
            draft_model_path_use_base=server_args.draft_model_path_use_base,
            draft_model_config=draft_model_config,
        )

        min_per_gpu_mem = self._init_distributed()

        target, draft = create_model_runner(
            server_args, self.model_config, draft_model_config, gpu_id, global_rank
        )
        self.multimodal_encoder_dtype = target.multimodal_encoder_dtype
        if server_args.disaggregation_mode in ("null", "prefill"):
            # Keep this after all target/draft weights are loaded and before
            # create_attn_components profiles memory for the KV-cache budget.
            target.prepare_multimodal_runtime()
        self.use_overlap_schedule = should_use_overlap_schedule(
            disable_overlap_schedule=server_args.disable_overlap_schedule,
            disaggregation_mode=server_args.disaggregation_mode,
        )
        self.overlap_schedule_depth = int(self.use_overlap_schedule)
        # In-flight depth of the unified event loop: how many dispatched
        # forwards may await commit at once. 0 = commit in the same iteration
        # (classic non-overlap); 1 = the overlap schedule (CPU post-processes
        # step N-1 while the GPU runs step N); pp_size = the prefill chunk
        # pipeline. Distinct from overlap_schedule_depth: that one sizes
        # decode KV reservations in the C++ scheduler and recipes, this one
        # only queues commits.
        if server_args.mapping.has_pp:
            self.in_flight_depth = server_args.mapping.pp_size
        else:
            self.in_flight_depth = int(self.use_overlap_schedule)
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
            self.cache_storage,
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

        self._scheduler_cache_geometry = scheduler_cache_geometry_from_pool(
            token_to_kv_pool
        )
        geometry = self._scheduler_cache_geometry
        # The contract is the one source of admitted capacity.
        self.max_total_num_tokens = geometry.token_capacity
        num_total_pages = geometry.num_device_pages
        cache_groups = pool_to_cache_groups(token_to_kv_pool)
        # Resolve the scheduler limit before ModelExecutorConfig sizes input
        # buffers. Lowering the limit is safe; a configured chunk smaller than
        # one state checkpoint block is rejected by aligned_max_scheduled_tokens instead of
        # silently increasing a frozen buffer limit.
        max_scheduled_tokens = server_args.chunked_prefill_size
        if server_args.enable_prefix_caching:
            max_scheduled_tokens = aligned_max_scheduled_tokens(
                server_args.chunked_prefill_size,
                cache_groups,
            )
            if max_scheduled_tokens != server_args.chunked_prefill_size:
                logger.warning(
                    "chunked_prefill_size=%s is not a multiple of the "
                    "state-snapshot checkpoint grain; using %s so recurrent-state "
                    "pages can register for prefix-cache reuse.",
                    server_args.chunked_prefill_size,
                    max_scheduled_tokens,
                )
                server_args.chunked_prefill_size = max_scheduled_tokens
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
            prefix_granularity=geometry.prefix_granularity,
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
        )

        # Per-rank GPU memory breakdown (weights by group, KV/graph/non-torch).
        # rank0 only; best-effort, never fails startup.
        if attn_tp_rank == 0:
            log_gpu_memory_summary(
                target.model,
                gpu_id,
                global_rank,
                logger,
                draft_model=draft.model if draft is not None else None,
                kv_pool=token_to_kv_pool,
                draft_kv_pool=draft_token_to_kv_pool,
            )

        self.attn_tp_size = server_args.attn_tp_size or mapping.attn.tp_size
        self.world_size = server_args.world_size or mapping.world_size
        self.attn_tp_rank = attn_tp_rank
        self.attn_tp_cpu_group = pg_manager.get_process_group(
            "gloo", server_args.mapping.attn.tp_group
        )
        self.dp_rank = dp_rank
        self.dp_size = mapping.attn.dp_size
        self.has_dp = mapping.has_attn_dp
        if self.has_dp:
            self.world_cpu_group = pg_manager.get_process_group(
                "gloo", mapping.world_group
            )
            self._dp_local_info = torch.zeros(1, 3, dtype=torch.int32)
            self._dp_global_info = torch.zeros(mapping.world_size, 3, dtype=torch.int32)
        if server_args.enable_kvstore:
            if server_args.kvstore_storage_backend is not None:
                raise NotImplementedError(
                    "the cache-group scheduler has no L3 storage tier; unset "
                    "--kvstore-storage-backend"
                )
            l2_cache_executor = L2CacheExecutor(
                device_pool=token_to_kv_pool,
                draft_pool=draft_token_to_kv_pool,
                host_ratio=server_args.kvstore_ratio,
                host_size_gb=server_args.kvstore_size,
                io_backend=server_args.kvstore_io_backend,
            )
            num_host_pages = l2_cache_executor.num_host_pages
        else:
            l2_cache_executor = None
            num_host_pages = 0
        # L2 cache-op submission + rank-synced completion tracking (see
        # cache_hooks.py); a no-op shell when kvstore is disabled.
        self._cache_hooks = L2CacheHooks(
            l2_cache_executor,
            speculative_algorithm=server_args.speculative_algorithm,
            attn_tp_rank=attn_tp_rank,
            attn_tp_size=self.attn_tp_size,
            attn_tp_cpu_group=self.attn_tp_cpu_group,
            global_rank=global_rank,
        )

        self._kv_events_enabled = (
            EventPublisherFactory.is_enabled(server_args.kv_events_config)
            and attn_tp_rank == 0
        )

        self._pd_cache_enabled = server_args.disaggregation_mode in (
            "prefill",
            "decode",
        )
        if self._pd_cache_enabled:
            if not token_to_kv_pool.arena.supports_disaggregation:
                raise RuntimeError(
                    "PD disaggregation requires a unified cache contract"
                )
            unsupported = []
            if server_args.enable_mixed_batch:
                unsupported.append("mixed prefill/decode batches")
            if (
                server_args.speculative_algorithm is not None
                and server_args.disaggregation_layerwise_interval > 0
                and not getattr(
                    self.model_executor.drafter,
                    "supports_pd_layerwise_finalization",
                    False,
                )
            ):
                unsupported.append(
                    f"{server_args.speculative_algorithm} layerwise transfer"
                )
            if server_args.enable_memory_saver:
                unsupported.append("memory saver/release")
            # Prefill is forced onto the non-overlap loop by
            # should_use_overlap_schedule(). Decode uses the ordinary overlap
            # loop and the scheduler's one-step protected cache reservation.
            if (
                self.use_overlap_schedule
                and server_args.disaggregation_mode != "decode"
            ):
                unsupported.append("overlap scheduling outside the Decode role")
            backend = server_args.disaggregation_transfer_backend
            if getattr(backend, "value", backend) != "mooncake":
                unsupported.append("non-Mooncake transfer backend")
            if unsupported:
                raise NotImplementedError(
                    "Paged-cache PD currently does not support: "
                    + ", ".join(unsupported)
                )
        # Backend/pool compatibility is validated inside ModelExecutor
        # (validate_scheduler_config), before CUDA-graph capture.
        self._cache_groups = cache_groups
        scheduler_cfg = make_config(
            num_device_pages=geometry.num_device_pages,
            max_scheduled_tokens=max_scheduled_tokens,
            max_batch_size=per_rank_max_batch,
            prefix_granularity=geometry.prefix_granularity,
            num_host_pages=num_host_pages,
            disable_l2_cache=not server_args.enable_kvstore,
            enable_l3_storage=server_args.kvstore_storage_backend is not None,
            role=server_args.disaggregation_mode,
            enable_kv_cache_events=self._kv_events_enabled,
            decode_input_tokens=decode_input_tokens,
            overlap_schedule_depth=self.overlap_schedule_depth,
            disable_prefix_cache=not server_args.enable_prefix_caching,
            prefix_replay_tokens=prefix_replay_tokens,
            cache_groups=cache_groups,
            enable_mixed_prefill_decode=server_args.enable_mixed_batch,
        )
        scheduler_cfg.enable_pd_cache = self._pd_cache_enabled
        logger.info(
            "Scheduler config: prefix_granularity=%s num_device_pages=%s "
            "max_scheduled_tokens=%s decode_input_tokens=%s "
            "overlap_schedule_depth=%s disable_l2_cache=%s "
            "max_batch_size=%s (global max_num_seqs=%s, dp_size=%s) "
            "disable_prefix_cache=%s prefix_replay_tokens=%s "
            "cache_groups=%s",
            scheduler_cfg.prefix_granularity,
            scheduler_cfg.num_device_pages,
            scheduler_cfg.max_scheduled_tokens,
            scheduler_cfg.decode_input_tokens,
            scheduler_cfg.overlap_schedule_depth,
            scheduler_cfg.disable_l2_cache,
            scheduler_cfg.max_batch_size,
            server_args.max_num_seqs,
            self.dp_size,
            scheduler_cfg.disable_prefix_cache,
            scheduler_cfg.prefix_replay_tokens,
            [group.group_id for group in cache_groups],
        )
        self.scheduler = Scheduler(scheduler_cfg)
        self.max_single_request_tokens = self.scheduler.max_single_request_tokens()
        self.max_model_len = min(
            self.model_config.context_len, self.max_single_request_tokens
        )
        input_reserve = (
            1
            if server_args.disaggregation_mode == "prefill"
            else max(decode_input_tokens, 1)
        )
        self.max_req_input_len = self.max_model_len - input_reserve
        if self.max_req_input_len < 1:
            raise RuntimeError(
                "Paged cache cannot admit one request with the configured "
                f"decode reserve: max_single_request_tokens="
                f"{self.max_single_request_tokens}, reserve={input_reserve}"
            )
        logger.info(
            "Single-request token limit: cache=%s model=%s effective=%s max_input=%s",
            self.max_single_request_tokens,
            self.model_config.context_len,
            self.max_model_len,
            self.max_req_input_len,
        )
        token_to_kv_pool.bind_cache_scheduler(self.scheduler)
        if attn_tp_rank == 0:
            self.kv_event_publisher = EventPublisherFactory.create(
                server_args.kv_events_config,
                attn_dp_rank=dp_rank,
            )
        else:
            self.kv_event_publisher = NullEventPublisher(attn_dp_rank=dp_rank)

        self._init_interprocess_comm()
        self._init_load_snapshot_publisher()

        # Pause/resume control state. Shared with the request handler, which
        # drives the control-request side; the event loop reads the gate.
        # PauseHooks is the loop-side integration (see pause.py) — the normal
        # scheduling paths below only carry single-line hooks into it.
        self._pause = PauseController(self.send_to_tokenizer)
        self._pause_hooks = PauseHooks(self, self._pause)

        # GPU-memory data plane (release/resume_memory_occupation). Reuses the
        # pause controller's drain machinery; frees memory via the memory-saver
        # adapter once the scheduler drains. See memory_occupation.py.
        # Releasing KV is only safe if any prefix cache it backs can be cleared:
        # either prefix caching is off, or the scheduler exposes a clear. Decide
        # once here (static config) and let the controller reject unsafe releases.
        kv_cache_release_allowed = (
            not self.server_args.enable_prefix_caching
            or callable(getattr(self.scheduler, "clear_l1_cache", None))
        )
        self._memory = MemoryOccupationController(
            send_func=self.send_to_tokenizer,
            pause_controller=self._pause,
            adapter=TorchMemorySaverAdapter.create(
                enable=self.server_args.enable_memory_saver
            ),
            enabled=self.server_args.enable_memory_saver,
            reset_caches_fn=self._pause_hooks.reset_caches_for_release,
            kv_repair_fn=self._pause_hooks.kv_repair_after_wake,
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

        self.request_handler = RequestHandler(
            server_args=self.server_args,
            hf_eos_token_id=self.model_config.hf_eos_token_id,
            max_req_len=self.max_model_len - 1,
            vocab_size=self.model_config.vocab_size,
            recv_func=self.recv_from_tokenizer,
            send_func=self.send_to_tokenizer,
            clear_cache_fn=self.scheduler.clear_cache,
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
            physical_context_len=(
                self.model_config.context_len + self.server_args.spec_context_pad
            ),
            metrics=self.metrics,
        )
        if server_args.disaggregation_mode != "null":
            assert pd_topology is not None
            pp_layer_window = None
            if server_args.mapping.has_pp:
                from tokenspeed.runtime.distributed.pp_stage import (
                    pp_layer_window as resolve_pp_layer_window,
                )

                pp_layer_window = resolve_pp_layer_window(
                    self.model_config.num_attention_layers, server_args.mapping
                )
            kv_args = get_kv_args(
                global_rank,
                global_rank,
                server_args.disaggregation_ib_device,
                token_to_kv_pool,
                model_config=self.model_config,
                draft_model_config=draft_model_config,
                pp_layer_window=pp_layer_window,
            )
            pd_manager_args = KVManagerArgs(
                bootstrap_port=server_args.disaggregation_bootstrap_port,
                dist_init_addr=server_args.dist_init_addr,
                topology=pd_topology,
                enable_metrics=False,
                served_model_name=server_args.served_model_name,
                app_key=server_args.app_key,
                metrics_reporters=server_args.metrics_reporters,
                enable_dp_attention=self.has_dp,
            )
            # PP: transfer-status consensus must span every stage — all
            # ranks run the same deterministic scheduler and must agree on
            # Bootstrapped/Succeeded events, and the KV for one request is
            # produced by pp*tp ranks together.
            kv_sync_group = (
                pg_manager.get_process_group("gloo", server_args.mapping.world_group)
                if server_args.mapping.has_pp
                else self.attn_tp_cpu_group
            )
            self.kv_transfer = create_kv_transfer(
                mode=server_args.disaggregation_mode,
                backend=server_args.disaggregation_transfer_backend,
                args=pd_manager_args,
                kv_args=kv_args,
                gloo_group=kv_sync_group,
            )
            if isinstance(self.kv_transfer, DisaggPrefillExecutor):
                # P-side layerwise KV streaming: wire the step counter between
                # the attn backends and the KV sender (a no-op for interval<=0).
                self.kv_transfer.setup_layerwise_transfer(
                    self.model_executor,
                    self.gpu_id,
                    server_args.disaggregation_layerwise_interval,
                )
            # EPD: a multimodal prefill node is also the encode->prefill embedding
            # SINK (independent of kv_transfer, its P->D KV source) -- it receives
            # each image's embedding from encode workers over Mooncake so the
            # prefill skips the vision tower. The admission controller owns the
            # receive jobs, the rank-synced admission drain, and the optional NCCL
            # row-shard reassembly; None for decode/encode/text-only nodes.
            # EpdPrefillHooks is the loop-side integration (see prefill_hooks.py).
            from tokenspeed.runtime.epd.prefill_admission import (
                make_epd_prefill_admission,
            )

            self._epd_hooks = EpdPrefillHooks(
                self,
                make_epd_prefill_admission(
                    server_args,
                    global_rank,
                    model_config=self.model_config,
                    model_executor=self.model_executor,
                    mapping=mapping,
                    attn_tp_rank=self.attn_tp_rank,
                    attn_tp_size=self.attn_tp_size,
                    attn_tp_cpu_group=self.attn_tp_cpu_group,
                    pg_manager=pg_manager,
                ),
            )
        else:
            self.kv_transfer = None
            self._epd_hooks = EpdPrefillHooks(self, None)
        # PD transfer-event integration (see pd/transfer_hooks.py); a no-op
        # when PD is disabled.
        self._pd_hooks = PdTransferHooks(self)

    def _publish_scheduler_kv_events(self) -> None:
        """Drain the KV events the C++ scheduler accumulated and publish them.

        Drain semantics: events queue up inside the scheduler across any
        number of mutations (advance / next_execution_plan), so one call at
        the event-loop tail — its only call site — publishes everything the
        round produced, in order, as a single batch.
        """
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

    def _dispatch_forward(
        self,
        forward_op,
        sampling_params_list,
        dp_metadata=None,
        stats=None,
        grammar_inputs=None,
        cache_zero_event=None,
    ):
        """Execute one forward step; return (results, on_first_token).

        results is None when the step produces no model output (Path 2/3);
        otherwise the loop queues (results, on_first_token) in ``in_flight``
        and commits them later (post_process at the queue head).

        Path 1 — no PD:              run forward, return (results, None)
        Path 2 — decode, extend:     trigger RDMA receive, return (None, None)
        Path 3 — prefill, decode:    send KV to decode side, return (None, None)
        Path 4 — prefill, extend:    run prefill forward, return (results, on_first_token)
        """
        if stats is None:
            stats = {}
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
        multimodal_context = (
            multimodal_context_for_forward(
                forward_op, self.output_processor.rid_to_state
            )
            if self.model_config.is_multimodal_active
            else None
        )

        if self.kv_transfer is None:
            # Path 1: normal (no disaggregation)
            self.model_executor.reset_valid_cache_length(forward_op)
            return (
                self.model_executor.execute_forward_op_with_log(
                    forward_op,
                    sampling_params_list,
                    dp_global_num_tokens=dp_global_num_tokens,
                    dp_global_bs=dp_global_bs,
                    dp_all_decode_or_idle=dp_all_decode_or_idle,
                    dp_all_extend=dp_all_extend,
                    grammar_inputs=grammar_inputs,
                    multimodal_context=multimodal_context,
                    **stats,
                ),
                None,
            )

        elif isinstance(self.kv_transfer, DisaggDecodeExecutor):
            # Decode node
            if forward_op.num_extends() > 0 and not forward_op.is_local_prefill():
                # Path 2: new requests waiting for remote KV — trigger RDMA receive
                self.model_executor.prepare_remote_cache_slots(
                    list(forward_op.request_pool_indices[: forward_op.num_extends()])
                )
                self.kv_transfer.reset_valid_cache_length(
                    forward_op,
                    self.model_executor.runtime_states,
                    self.model_executor.execution_stream,
                    self.model_executor.device,
                )
                if self._pd_cache_enabled and cache_zero_event is not None:
                    # Page zeroing runs asynchronously on a CUDA stream,
                    # while Mooncake/GPUDirect writes are not ordered by that
                    # stream. Do not publish the destination manifest until the
                    # newly assigned pages are fully sanitized.
                    cache_zero_event.synchronize()
                self.kv_transfer.execute(forward_op)
                return None, None
            else:
                # Decode and local recovery-prefill batches execute normally.
                self.model_executor.reset_valid_cache_length(forward_op)
                return (
                    self.model_executor.execute_forward_op_with_log(
                        forward_op,
                        sampling_params_list,
                        dp_global_num_tokens=dp_global_num_tokens,
                        dp_global_bs=dp_global_bs,
                        dp_all_decode_or_idle=dp_all_decode_or_idle,
                        dp_all_extend=dp_all_extend,
                        multimodal_context=multimodal_context,
                        **stats,
                    ),
                    None,
                )

        else:
            # Prefill node (the overlap schedule is disabled for the P role;
            # under PP the in-flight depth is the chunk-pipeline depth instead)
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
                self._epd_hooks.assert_embeddings_received(multimodal_context)
                return (
                    self.model_executor.execute_forward_op_with_log(
                        forward_op,
                        sampling_params_list,
                        dp_global_num_tokens=dp_global_num_tokens,
                        dp_global_bs=dp_global_bs,
                        dp_all_decode_or_idle=dp_all_decode_or_idle,
                        dp_all_extend=dp_all_extend,
                        grammar_inputs=grammar_inputs,
                        multimodal_context=multimodal_context,
                        capture_next_input_ids=True,
                        **stats,
                    ),
                    self.kv_transfer.store_prefill_token,
                )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_model_config(
        self, model_path: str, is_draft_worker: bool = False
    ) -> ModelConfig:
        server_args = self.server_args
        quantization = server_args.quantization
        dtype = server_args.dtype
        if is_draft_worker:
            quantization = server_args.speculative_draft_model_quantization
            if dtype == "auto":
                # A draft is fed the target's hidden states and borrows its
                # embedding and LM head, so the two dtypes have to agree.
                dtype = self.model_config.dtype
        return ModelConfig(
            model_path,
            trust_remote_code=server_args.trust_remote_code,
            revision=server_args.revision,
            context_length=server_args.max_model_len,
            model_override_args=server_args.hf_overrides,
            dtype=dtype,
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
        # Chunk-pipeline: request I/O is owned by GLOBAL rank 0 only —
        # every stage's tp_rank-0 would otherwise try to open the one
        # frontend socket pair. recv_reqs broadcasts over the world group.
        owns_request_io = (
            self.server_args.mapping.rank == 0
            if self.server_args.mapping.has_pp
            else self.attn_tp_rank == 0
        )
        if owns_request_io:
            if self.server_args.zmq_msgpack:
                # SMG drives the scheduler directly: it binds the sockets and
                # this engine connects in over the msgpack wire; the handshake
                # (engine identity, ready response) lives in zmq_msgpack.
                from tokenspeed.runtime.engine import zmq_msgpack

                self.recv_from_tokenizer, self.send_to_tokenizer = (
                    zmq_msgpack.connect_msgpack_engine_for_loop(context, self)
                )
            else:
                self.recv_from_tokenizer = IpcReceiver(
                    get_zmq_socket(
                        context,
                        zmq.PULL,
                        self.port_args.scheduler_input_ipc_name,
                        False,
                    )
                )
                self.send_to_tokenizer = IpcSender(
                    get_zmq_socket(
                        context, zmq.PUSH, self.port_args.tokenizer_ipc_name, False
                    )
                )
        else:
            self.recv_from_tokenizer = None
            self.send_to_tokenizer = NullSender()

    def _init_load_snapshot_publisher(self) -> None:
        self._load_snapshot_observed_while_paused = False
        if self.attn_tp_rank != 0:
            self.load_snapshot_publisher = NullLoadSnapshotPublisher()
        elif self.server_args.zmq_msgpack:
            self.load_snapshot_publisher = DirectLoadSnapshotSink(
                self.send_to_tokenizer.set_load_snapshot
            )
        else:
            self.load_snapshot_publisher = LoadSnapshotPublisher(
                self.port_args.metrics_ipc_name,
                self.dp_rank,
                self.server_args.load_watch_interval,
            )

    # ------------------------------------------------------------------
    # Shared step helpers
    # ------------------------------------------------------------------

    def _request_abort_or_mark(
        self, request_id: str, _reason: str, *, notify_client: bool = False
    ) -> None:
        """Mark an abort; scheduler release follows the normal lifecycle event."""
        if notify_client:
            self.output_processor.mark_abort(request_id, notify_client=True)
        else:
            self.output_processor.mark_abort(request_id)

    def _process_new_requests(self):
        recv_reqs = self.request_handler.recv_reqs()
        if recv_reqs:
            # A paused loop still accepts control messages and registers new
            # requests into its Python-side buffers. Re-sample once after that
            # batch instead of heartbeating the pre-mutation tuple forever.
            self._mark_load_snapshot_dirty()
        # Pause-state snapshot for withhold_admissions below: it must be
        # taken before process_requests, which may flip the state mid-batch.
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
            self._request_abort_or_mark(rid, "client cancelled request")
            grammar_manager.mark_abort(rid)

        self._pause_hooks.apply_transitions(grammar_manager)

        # Partition new requests by grammar readiness. Compile-bound requests
        # are queued in GrammarManager and admitted in a later iteration when
        # their futures resolve (get_ready_grammar_requests below).
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

            if self._pd_cache_enabled and bootstrap is None:
                raise ValueError(
                    "Paged cache PD request is missing bootstrap information"
                )
            if isinstance(self.kv_transfer, DisaggDecodeExecutor):
                state.computed_length = state.input_length
            self.output_processor.register(spec.request_id, state)
            # EPD prefill: an encode-routed request is staged OUT of the
            # scheduler until its embeddings arrive; its P->D sender
            # registration and submission are both deferred to the EPD
            # admission drain (see EpdPrefillHooks.try_stage for why).
            if self._epd_hooks.try_stage(spec, state, bootstrap):
                continue
            if self.kv_transfer is not None:
                self.kv_transfer.register(spec.request_id, bootstrap)
            admitted_specs.append(spec)

        if self._pause_hooks.withhold_admissions(admitted_specs, pause_blocked_before):
            return

        if admitted_specs:
            self.scheduler.submit_requests(admitted_specs)

    @nvtx_range("loop:commit", color="rapids")
    def _pp_broadcast_output_tokens(self, forward_op, results) -> None:
        """Align sampled tokens across pipeline stages before commit.

        Only the last stage samples; the other stages produced placeholder
        outputs. Every rank's C++ scheduler expects the REAL first token in
        the final chunk's ExtendResult, and every stage's KV sender needs the
        bootstrap token, so the last stage broadcasts (output_tokens,
        output_lengths) over the PP gloo group and the others adopt them.
        Runs on the commit path (queue head), off the dispatch hot path.
        """
        mapping = self.server_args.mapping
        if not mapping.has_pp:
            return
        group = pg_manager.get_process_group("gloo", mapping.pp_group)
        src_global_rank = mapping.pp_group[-1]
        results.sync()
        payload = [None]
        if mapping.is_last_pp_rank:
            payload = [
                (
                    results.output_tokens.cpu(),
                    results.output_lengths.cpu(),
                    (
                        results.next_input_ids
                        if results.next_input_ids is None
                        else results.next_input_ids.cpu()
                    ),
                )
            ]
        dist.broadcast_object_list(payload, src=src_global_rank, group=group)
        if not mapping.is_last_pp_rank:
            tokens, lengths, next_ids = payload[0]
            results.output_tokens = tokens
            results.output_lengths = lengths
            results.next_input_ids = next_ids

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
        self._pp_broadcast_output_tokens(forward_op, results)

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

    def _dp_sync_and_check(self, forward_op) -> DpForwardMetadata:
        """Synchronize DP ranks with CPU-only metadata.

        All ranks call this before GPU forward work. The gathered metadata is
        used for eager token-aware collectives and for choosing a common padded
        CUDA graph shape during decode.
        """
        # Whether forward_op will enter the model forward path. On decode-side
        # PD, EXTEND ops only start remote KV receive; the model forward runs
        # after the remote prefill completes and the scheduler advances the
        # request into decode. Treating those EXTEND ops as model work makes
        # idle DP ranks enter dummy collectives that the active rank will not
        # match.
        is_disagg_decode = isinstance(self.kv_transfer, DisaggDecodeExecutor)
        executes_model_forward = (
            forward_op is not None
            and sum(forward_op.input_lengths) > 0
            and not (
                is_disagg_decode
                and forward_op.num_extends() > 0
                and not forward_op.is_local_prefill()
            )
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
        available = self.scheduler.available_kv_pages()
        active = self.scheduler.active_kv_pages()
        return {
            "num_active_pages": active,
            "num_cached_pages": (
                self._scheduler_cache_geometry.num_usable_pages - available
            ),
            "num_queue_reqs": self.scheduler.waiting_size(),
        }

    def _observe_load_snapshot(self, stats: dict) -> None:
        """Project the latest pre-forward planning sample onto each load sink."""
        num_running = len(self.output_processor.rid_to_state)
        num_waiting = stats["num_queue_reqs"]
        num_active_pages = stats["num_active_pages"]
        max_total_pages = self._scheduler_cache_geometry.num_usable_pages
        self.load_snapshot_publisher.observe(
            (
                num_running,
                num_waiting,
                num_active_pages,
                stats["num_cached_pages"],
                max_total_pages,
            )
        )

    def _mark_load_snapshot_dirty(self) -> None:
        """Request one fresh scheduler sample on the next paused iteration."""
        self._load_snapshot_observed_while_paused = False

    def _maybe_observe_paused_load(self, had_changes: bool) -> None:
        """Frozen rounds: sample the load once per pause — and again after any
        round that changed scheduler state (committed changes, or a dirty mark
        from new requests / cache-op completions) — instead of on every idle
        spin. Called at the loop tail, after the advance, so the sample
        reflects the applied changes.
        """
        if self.attn_tp_rank != 0:
            return
        if had_changes or not self._load_snapshot_observed_while_paused:
            self._observe_load_snapshot(self._get_scheduler_stats())
            self._load_snapshot_observed_while_paused = True

    def _record_scheduler_iteration_metrics(
        self, stats: dict, num_iteration_tokens: int
    ) -> None:
        self.metrics.record_scheduler_iteration(
            running=len(self.output_processor.rid_to_state),
            waiting=stats["num_queue_reqs"],
            num_active_pages=stats["num_active_pages"],
            num_total_pages=self._scheduler_cache_geometry.num_usable_pages,
            num_iteration_tokens=num_iteration_tokens,
        )

    # ------------------------------------------------------------------
    # Event loops
    # ------------------------------------------------------------------

    def _shutdown_complete(self) -> bool:
        return self.shutdown_event.is_set()

    def _drain_in_flight(self, in_flight) -> list:
        """Commit every queued forward, oldest first; return their changes."""
        request_changes = []
        while in_flight:
            fo, res, oft = in_flight.popleft()
            request_changes.extend(self._commit_forward_results(fo, res, oft))
        return request_changes

    def _dispatch_depends_on_pending_commit(self, forward_op, grammar_inputs) -> bool:
        """Whether the upcoming dispatch reads state that only a pending
        commit produces, so the in-flight queue must drain first.

        The single registry of overlap-breaking dependencies — add new rules
        here, not in ``event_loop``:

        - P-side PD handoff batch: ``kv_transfer.execute`` needs the final
          chunk's bootstrap token, which only lands at commit.
        - Eager grammar: ``setup_grammar_step`` reads each matcher's current
          state to fill the bitmask, and the matcher only advances at the
          pending step's commit (``accept_token``). Capturable grammar dodges
          this with an in-graph hostfunc; eager has no equivalent, so trade
          the overlap away for grammar batches.
        """
        if forward_op.num_extends() == 0 and isinstance(
            self.kv_transfer, DisaggPrefillExecutor
        ):
            return True
        return (
            grammar_inputs is not None
            and self.model_executor.eager_grammar_buffers is not None
        )

    def event_loop(self):
        """The one scheduler loop, parameterized by in-flight depth.

        ``in_flight_depth`` is how many dispatched forwards may await commit:

        - 0: commit in the same iteration (classic non-overlap behavior).
        - 1: dispatch the current forward before committing the previous one,
          so the CPU post-processes step N-1 while the GPU runs step N (the
          overlap schedule).
        - pp_size: the prefill chunk pipeline — consecutive chunks occupy
          different pipeline stages; the head's copy_event sync is the
          backpressure.

        Correctness never depends on the depth: any dispatch whose inputs
        depend on a pending commit's side effects drains the queue first
        (``_dispatch_depends_on_pending_commit`` is the single registry of
        those rules), and rounds that run no real forward (pause/freeze,
        DP idle) drain it fully.

        Scheduler feedback is only ever an explicit ``advance_scheduler`` call
        in this loop body — helpers return events, never advance. Two calls:
        cache-op completions at the head of the round (so this round's plan
        sees them) and forward results at the tail (they only exist after
        dispatch); everything else funnels into ``request_changes``.
        """
        in_flight: deque = deque()
        depth = self.in_flight_depth
        while not self._shutdown_complete():
            if depth > 0:
                # Order this iter's default-stream writes (prefix_cache
                # page-table writes) after the pending forwards on
                # execution_stream that read the same tensors. Non-blocking
                # on host; a no-op when nothing is in flight.
                torch.cuda.default_stream().wait_stream(
                    self.model_executor.execution_stream
                )
            num_tracked_reqs = len(self.output_processor.rid_to_state)
            self._process_new_requests()
            if len(self.output_processor.rid_to_state) != num_tracked_reqs:
                self._mark_load_snapshot_dirty()
            # EPD prefill: admit requests whose async embedding receives completed
            # this cycle (rank-synced). Fixed position right after
            # _process_new_requests so the drain's TP collective ordering is
            # rank-identical every cycle. A no-op without an EPD admission
            # controller (every non-EPD deployment).
            self._epd_hooks.drain_ready_embeddings()
            cache_events = self._cache_hooks.poll_ready_events()
            if cache_events:
                # Advanced at the HEAD of the round (not funneled into the
                # tail advance) so completed cache ops are visible to this
                # round's next_execution_plan — deferring them would delay
                # cache-gated admissions by a full round.
                advance_scheduler(self.scheduler, cache_events)
                self._mark_load_snapshot_dirty()

            # Every path in this round appends its committed results here;
            # they feed back into the scheduler through the single
            # advance_scheduler call at the tail.
            request_changes = []
            forward_op = None
            # An idle round (freeze or DP idle) runs no dispatch and — as
            # before the paths were unified — no kv-transfer event poll.
            idle_round = False

            if self._pause.forward_blocked:
                # Freeze: dispatched forwards can't be un-launched; commit them
                # all before idling.
                request_changes.extend(self._drain_in_flight(in_flight))
                self._pause_hooks.paused_idle_step()
                idle_round = True
            else:
                self._load_snapshot_observed_while_paused = False
                execution_plan = self.scheduler.next_execution_plan()
                cache_zero_event = self.model_executor.zero_cache_pages(
                    execution_plan.pages_to_zero
                )
                self._cache_hooks.submit(execution_plan)

                forward_op = self._get_forward_op(execution_plan)
                stats = self._get_scheduler_stats()
                self._observe_load_snapshot(stats)
                num_iter_tokens = (
                    sum(forward_op.input_lengths) if forward_op is not None else 0
                )
                # Record once per iteration, from the same pre-dispatch
                # snapshot as ``stats`` (the running gauge counts requests
                # admitted but not yet committed-finished this round —
                # consistent with waiting/pages).
                self._record_scheduler_iteration_metrics(stats, num_iter_tokens)

                # DP sync: all ranks must participate even when idle. Checked
                # right after forward_op is derived so an idle round commits
                # pending steps and skips the per-batch work below (the
                # gathers are local and read-only, so ordering them after the
                # collective is rank-safe).
                dp_metadata = None
                if self.has_dp:
                    dp_metadata = self._dp_sync_and_check(forward_op)
                    if dp_metadata.need_idle_forward:
                        request_changes.extend(self._drain_in_flight(in_flight))
                        self.model_executor.execute_idle_forward(
                            dp_metadata.global_num_tokens,
                            dp_metadata.global_batch_size,
                            dp_metadata.all_decode_or_idle,
                        )
                        idle_round = True

            if not idle_round:
                grammar_inputs = None
                sampling_params_list = None
                if forward_op is not None:
                    # Gather sampling params and grammar state BEFORE any
                    # pending commit below — a commit can finish requests and
                    # pop them from output_processor.rid_to_state, which would
                    # KeyError on rids still present in the current forward_op.
                    sampling_params_list = self._gather_sampling_params(forward_op)
                    grammar_inputs = self._gather_grammar_state(forward_op)

                if (
                    in_flight
                    and forward_op is not None
                    and self._dispatch_depends_on_pending_commit(
                        forward_op, grammar_inputs
                    )
                ):
                    request_changes.extend(self._drain_in_flight(in_flight))

                if forward_op is not None:
                    self._mark_stats_scheduled(forward_op)
                    results, on_first_token = self._dispatch_forward(
                        forward_op,
                        sampling_params_list,
                        dp_metadata=dp_metadata,
                        stats=stats,
                        grammar_inputs=grammar_inputs,
                        cache_zero_event=cache_zero_event,
                    )
                    if results is not None:
                        in_flight.append((forward_op, results, on_first_token))

                # Commit from the head once the queue exceeds the depth
                # (immediately at depth 0; one step behind at depth 1; a full
                # pipeline behind under PP). A round with no new work drains
                # fully so results never wait on future traffic.
                effective_depth = depth if forward_op is not None else 0
                while len(in_flight) > effective_depth:
                    fo, res, oft = in_flight.popleft()
                    request_changes.extend(self._commit_forward_results(fo, res, oft))

                request_changes.extend(self._pd_hooks.poll_transfer_events())

            # The forward-result feedback point: everything this round
            # committed reaches the scheduler here, before the next round
            # plans. (Cache-op completions advance at the head instead — see
            # _cache_hooks.poll_ready_events — but through the same advance_scheduler,
            # the only caller of scheduler.advance.)
            if request_changes:
                advance_scheduler(self.scheduler, request_changes)

            # Publish KV events once per round, after the last scheduler
            # mutation: the drain empties everything the round accumulated
            # (head cache advance, plan, tail advance) in order, as one batch.
            self._publish_scheduler_kv_events()

            if self._pause.forward_blocked:
                self._maybe_observe_paused_load(had_changes=bool(request_changes))

            # Resolve a deferred abort/wait pause reply once in-flight work drains.
            self._pause.maybe_finish_drain(self.scheduler)

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

    def close(self) -> None:
        self.load_snapshot_publisher.close()
        # Best-effort: tell an attached SMG frontend this engine is going away
        # (msgpack mode only; the pickle sender has no such helper) so the
        # worker is marked dead instead of staying healthy-idle.
        send_engine_dead = getattr(self.send_to_tokenizer, "send_engine_dead", None)
        if callable(send_engine_dead):
            send_engine_dead()
        close_transfer = getattr(self.kv_transfer, "close", None)
        if callable(close_transfer):
            close_transfer()


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
    # Re-assert the NVSHMEM IB traffic class in every inference process:
    # NVSHMEM reads it from the process environment at bootstrap, and worker
    # processes may be spawned without inheriting the launcher's setting.
    if envs.NVSHMEM_IB_TRAFFIC_CLASS.is_set():
        envs.NVSHMEM_IB_TRAFFIC_CLASS.set(envs.NVSHMEM_IB_TRAFFIC_CLASS.get())
        logger.info("NVSHMEM_IB_TRAFFIC_CLASS=%d", envs.NVSHMEM_IB_TRAFFIC_CLASS.get())
    faulthandler.enable()
    parent_process = psutil.Process().parent()
    register_usr_signal()

    prefix = f" ATTN TP RANK {attn_tp_rank}"
    configure_logger(server_args, prefix=prefix)

    event_loop = None
    shutdown_event = threading.Event()
    previous_sigterm_handler = None
    try:
        if server_args.disaggregation_mode == "encode":
            # The encode role is LM-free; run the lightweight vision-tower loop
            # instead of building the full EventLoop (KV/LM scheduler).
            from tokenspeed.runtime.epd.encode_loop import (
                run_encode_loop,
            )

            run_encode_loop(server_args, port_args, pipe_writer, gpu_id, global_rank)
            return

        # Convert SIGTERM into a loop-owned stop request so the current
        # scheduler iteration and ordinary runtime cleanup can finish.
        if threading.current_thread() is threading.main_thread():
            previous_sigterm_handler = signal.getsignal(signal.SIGTERM)
            signal.signal(
                signal.SIGTERM,
                lambda _signum, _frame: shutdown_event.set(),
            )

        if torch.cuda.is_available():
            # Warm up CUPTI before EventLoop init captures any CUDA graph
            # (decode/prefill/encoder). A profiler that first attaches AFTER
            # capture invalidates the captured graphs — every later replay
            # dies with cudaErrorLaunchFailure — which would forbid runtime
            # /start_profile on graph-mode servers. One empty profiler
            # session loads CUPTI ahead of every capture, making runtime
            # attach/detach safe.
            from torch.profiler._utils import _init_for_cuda_graphs

            _init_for_cuda_graphs()

        event_loop = EventLoop(
            server_args,
            port_args,
            gpu_id,
            attn_tp_rank,
            dp_rank,
            global_rank,
            shutdown_event,
        )
        pipe_writer.send(
            {
                "status": "ready",
                "max_total_num_tokens": event_loop.max_total_num_tokens,
                "max_req_input_len": event_loop.max_req_input_len,
                "max_single_request_tokens": event_loop.max_single_request_tokens,
                "max_num_seqs": server_args.max_num_seqs,
                "chunked_prefill_size": server_args.chunked_prefill_size,
                "max_model_len": event_loop.max_model_len,
                "multimodal_encoder_dtype": event_loop.multimodal_encoder_dtype,
                "cache_storage": getattr(event_loop, "cache_storage", None),
            }
        )

        if event_loop.has_dp:
            # All DP schedulers must finish initialization before any rank enters
            # the loop and starts the first DP metadata collective.
            dist.barrier(group=event_loop.world_cpu_group)

        event_loop.event_loop()

    except Exception:  # noqa: BLE001 - process boundary; report and signal parent
        traceback = get_exception_traceback()
        logger.error("Scheduler hit an exception: %s", traceback)
        parent_process.send_signal(signal.SIGUSR1)
    finally:
        if event_loop is not None:
            try:
                event_loop.close()
            except Exception:  # noqa: BLE001 - best-effort teardown; signal parent
                logger.error(
                    "Scheduler transport shutdown failed: %s",
                    get_exception_traceback(),
                )
                parent_process.send_signal(signal.SIGUSR1)
        if (
            previous_sigterm_handler is not None
            and threading.current_thread() is threading.main_thread()
        ):
            signal.signal(signal.SIGTERM, previous_sigterm_handler)
