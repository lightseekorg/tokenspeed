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
from collections import OrderedDict
from dataclasses import dataclass

import psutil
import setproctitle
import torch
import torch.distributed as dist
import zmq
from tokenspeed_scheduler import PD, Cache, ExecutionEvent, Scheduler

from tokenspeed.runtime.cache.executor.memory_executor import (
    MemoryExecutor,
    MemoryExecutorConfig,
)
from tokenspeed.runtime.configs.model_config import ModelConfig
from tokenspeed.runtime.distributed.process_group_manager import (
    process_group_manager as pg_manager,
)
from tokenspeed.runtime.engine.generation_output_processor import OutputProcesser
from tokenspeed.runtime.engine.request_handler import RequestHandler
from tokenspeed.runtime.engine.scheduler_utils import (
    advance_forward,
    cache_event_from_payload,
    cache_event_key,
    cache_event_to_payload,
    cache_sync_debug_enabled,
    make_config,
    pop_common_cache_event_payloads,
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
from tokenspeed.runtime.grammar.capturable_grammar import GrammarStepInputs
from tokenspeed.runtime.layers.attention.registry import create_attn_components
from tokenspeed.runtime.pd.decode_executor import DisaggDecodeExecutor
from tokenspeed.runtime.pd.factory import (
    create_pd_kv_transfer,
    get_kv_args,
)
from tokenspeed.runtime.pd.mooncake.entities import ManagerArgs
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

logger = get_colorful_logger(__name__)


def calc_l3_query_hashes(scheduler, tokens: list[int]) -> list[str]:
    return scheduler.calc_rolling_hash(tokens, apply_match=True)


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
        )

        num_total_pages = self.max_total_num_tokens // server_args.block_size
        model_executor_config = ModelExecutorConfig.from_server_args(
            server_args=server_args,
            model_config=self.model_config,
            max_req_pool_size=server_args.max_num_seqs,
            gpu_id=gpu_id,
            global_rank=global_rank,
            num_total_pages=num_total_pages,
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
        mapping = server_args.mapping
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

        mem_cfg = MemoryExecutorConfig(
            layer_num=self.model_config.num_hidden_layers,
            page_size=server_args.block_size,
            host_ratio=server_args.kvstore_ratio,
            host_size_gb=server_args.kvstore_size,
            io_backend=server_args.kvstore_io_backend,
            host_layout=server_args.kvstore_mem_layout,
            storage_backend=server_args.kvstore_storage_backend,
            storage_backend_extra_config=server_args.kvstore_storage_backend_extra_config,
            model_name=server_args.model,
        )
        is_deepseek_v4_pool = (
            type(token_to_kv_pool).__name__ == "DeepseekV4TokenToKVPool"
        )
        if is_deepseek_v4_pool:
            if server_args.enable_kvstore:
                raise NotImplementedError(
                    "DeepSeek V4 baseline does not support hierarchical cache "
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
            )
            num_host_pages = self.memory_executor.host_pool.page_num

        # For DP attention, max_batch_size must be per-rank to avoid
        # req_pool_allocator overflow.  The C++ scheduler allocates
        # req_pool_slots based on this value, so it must match the
        # per-DP-rank budget (same division used in cuda_graph_wrapper).
        per_rank_max_batch = server_args.max_num_seqs // max(self.dp_size, 1)
        scheduler_cfg = make_config(
            num_device_pages=self.max_total_num_tokens // server_args.block_size,
            max_scheduled_tokens=server_args.chunked_prefill_size,
            max_batch_size=per_rank_max_batch,
            page_size=server_args.block_size,
            num_host_pages=num_host_pages,
            disable_l2_cache=not server_args.enable_kvstore,
            enable_l3_storage=server_args.kvstore_storage_backend is not None,
            prefetch_threshold=4,  # Keep this hard-coded until it becomes configurable.
            role=server_args.disaggregation_mode,
            decode_input_tokens=(
                server_args.speculative_num_draft_tokens
                if server_args.speculative_algorithm is not None
                else 1
            ),
            num_mamba_slots=mamba_pool_total_chunks,
        )
        logger.info(
            "Scheduler config: page_size=%s num_device_pages=%s "
            "max_scheduled_tokens=%s decode_input_tokens=%s disable_l2_cache=%s "
            "max_batch_size=%s (global max_num_seqs=%s, dp_size=%s) "
            "num_mamba_slots=%s",
            scheduler_cfg.page_size,
            scheduler_cfg.num_device_pages,
            scheduler_cfg.max_scheduled_tokens,
            scheduler_cfg.decode_input_tokens,
            scheduler_cfg.disable_l2_cache,
            scheduler_cfg.max_batch_size,
            server_args.max_num_seqs,
            self.dp_size,
            mamba_pool_total_chunks,
        )
        self.scheduler = Scheduler(scheduler_cfg)

        self._init_interprocess_comm()

        self.request_handler = RequestHandler(
            server_args=self.server_args,
            hf_eos_token_id=self.model_config.hf_eos_token_id,
            max_req_len=self.model_config.context_len - 1,
            vocab_size=self.model_config.vocab_size,
            recv_func=self.recv_from_tokenizer,
            send_func=self.send_to_tokenizer,
            get_load_fn=self._get_load,
            architectures=self.model_config.hf_config.architectures,
        )

        self.output_processor = OutputProcesser(
            send_to_tokenizer=self.send_to_tokenizer,
            global_rank=global_rank,
            spec_algorithm=self.server_args.speculative_algorithm,
            spec_num_tokens=(
                self.server_args.speculative_num_draft_tokens
                if self.server_args.speculative_algorithm is not None
                else None
            ),
            stream_interval=self.server_args.stream_interval,
        )
        self.prefetch_threshold = scheduler_cfg.prefetch_threshold

        if server_args.disaggregation_mode != "null":
            kv_args = get_kv_args(
                global_rank,
                global_rank,
                server_args.disaggregation_ib_device,
                token_to_kv_pool,
                draft_token_to_kv_pool,
            )
            pd_manager_args = ManagerArgs(
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
            self.pd_kv_transfer = create_pd_kv_transfer(
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
        else:
            self.pd_kv_transfer = None

    def _setup_pd_layerwise_transfer(self, interval: int) -> None:
        if not isinstance(self.pd_kv_transfer, DisaggPrefillExecutor):
            return
        if interval <= 0:
            return

        from tokenspeed.runtime.pd.utils import StepCounter

        step_counter = StepCounter(self.model_executor.device, self.gpu_id)
        self.model_executor.attn_backend.register_step_counter(step_counter)
        if self.model_executor.draft_attn_backend is not None:
            self.model_executor.draft_attn_backend.register_step_counter(step_counter)
        self.pd_kv_transfer.register_layerwise_step_counter(step_counter, interval)

    def _commit_cache_results(self) -> None:
        if self.memory_executor is None:
            return
        cache_results = self.memory_executor.poll_results()
        self._num_inflight_cache_ops -= len(cache_results)
        for event in cache_results:
            payload = cache_event_to_payload(event)
            self._pending_cache_event_payloads[cache_event_key(payload)] = payload

        # Local short-circuit: counter mirrors across ranks (same plan every-
        # where), so if nothing is in flight here AND nothing is buffered
        # awaiting commit, no rank has anything to gather.
        if self._num_inflight_cache_ops == 0 and not self._pending_cache_event_payloads:
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
        execution_plan,
        dp_metadata=None,
        stats=None,
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

        self.model_executor.update_block_table(forward_op)

        if self.pd_kv_transfer is None:
            # Path 1: normal (no disaggregation)
            self.model_executor.reset_valid_cache_length(forward_op)
            return (
                self.model_executor.execute_forward_op_with_log(
                    forward_op,
                    sampling_params_list,
                    dp_global_num_tokens=dp_global_num_tokens,
                    dp_global_bs=dp_global_bs,
                    dp_all_decode_or_idle=dp_all_decode_or_idle,
                    grammar_inputs=grammar_inputs,
                    **stats,
                ),
                None,
            )

        elif isinstance(self.pd_kv_transfer, DisaggDecodeExecutor):
            # Decode node
            if forward_op.num_extends() > 0:
                # Path 2: new requests waiting for remote KV — trigger RDMA receive
                self.pd_kv_transfer.reset_valid_cache_length(
                    forward_op,
                    self.model_executor.runtime_states,
                    self.model_executor.execution_stream,
                    self.model_executor.device,
                )
                self.pd_kv_transfer.execute(forward_op)
                return None, None
            else:
                # Path 3b: decode batch — normal forward
                self.model_executor.reset_valid_cache_length(forward_op)
                return (
                    self.model_executor.execute_forward_op_with_log(
                        forward_op,
                        sampling_params_list,
                        dp_global_num_tokens=dp_global_num_tokens,
                        dp_global_bs=dp_global_bs,
                        dp_all_decode_or_idle=dp_all_decode_or_idle,
                        **stats,
                    ),
                    None,
                )

        else:
            # Prefill node (only reached from event_loop, never event_loop_overlap)
            assert isinstance(self.pd_kv_transfer, DisaggPrefillExecutor)
            if forward_op.num_extends() == 0:
                # Path 3: all prefill done — send KV to decode side
                self.pd_kv_transfer.execute(forward_op)
                return None, None
            else:
                # Path 4: extend batch — run prefill forward
                self.model_executor.reset_valid_cache_length(forward_op)
                self.pd_kv_transfer.prepare_prefill(forward_op)
                return (
                    self.model_executor.execute_forward_op_with_log(
                        forward_op,
                        sampling_params_list,
                        dp_global_num_tokens=dp_global_num_tokens,
                        dp_global_bs=dp_global_bs,
                        dp_all_decode_or_idle=dp_all_decode_or_idle,
                        grammar_inputs=grammar_inputs,
                        **stats,
                    ),
                    self.pd_kv_transfer.store_prefill_token,
                )

    def _submit_cache_ops(self, execution_plan) -> None:
        if self.memory_executor is None:
            return
        self.memory_executor.submit_plan(execution_plan)
        for op in execution_plan.cache:
            if isinstance(op, (Cache.WriteBackOp, Cache.LoadBackOp)):
                self._num_inflight_cache_ops += len(op.op_ids)
            elif isinstance(op, (Cache.PrefetchOp, Cache.BackUpOp)):
                self._num_inflight_cache_ops += 1
            else:
                raise ValueError(f"unsupported cache op kind: {type(op).__name__}")
        self._setup_layerwise_loadback(execution_plan)

    def _setup_layerwise_loadback(self, execution_plan) -> None:
        consumer_indices = []
        draft_consumer_indices = []
        for cache_op in execution_plan.cache:
            if isinstance(cache_op, Cache.LoadBackOp):
                for op_id in cache_op.op_ids:
                    producer_idx = self.memory_executor.get_producer_index(op_id)
                    if (
                        producer_idx is not None
                        and producer_idx not in consumer_indices
                    ):
                        consumer_indices.append(producer_idx)
                    draft_idx = self.memory_executor.get_draft_producer_index(op_id)
                    if (
                        draft_idx is not None
                        and draft_idx not in draft_consumer_indices
                    ):
                        draft_consumer_indices.append(draft_idx)
        self.memory_executor.set_consumer(consumer_indices if consumer_indices else -1)
        self.memory_executor.set_draft_consumer(
            draft_consumer_indices if draft_consumer_indices else -1
        )
        # Fence WriteBack against this iter's ``set_kv_buffer``: the
        # scheduler can re-allocate a freed-but-not-yet-written-back slot
        # to a new prefill / decode within the same iter. ``set_kv_buffer``
        # runs before any ``wait_until`` in attention, so nothing else
        # orders writeback's reads against the new writes. Cheap when
        # write_stream is idle.
        host_exec = getattr(self.memory_executor, "host_exec", None)
        if host_exec is not None:
            self.model_executor.execution_stream.wait_stream(host_exec.write_stream)

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

    def _process_new_requests(self):
        recv_reqs = self.request_handler.recv_reqs()
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

            if isinstance(self.pd_kv_transfer, DisaggDecodeExecutor):
                state.computed_length = state.input_length
            self.output_processor.register(spec.request_id, state)
            if self.pd_kv_transfer is not None:
                self.pd_kv_transfer.register(spec.request_id, bootstrap)

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
            admitted_specs.append(spec)

        if admitted_specs:
            self.scheduler.submit_requests(admitted_specs)

    @nvtx_range("loop:commit", color="rapids")
    def _commit_forward_results(
        self, forward_op, results: ModelExecutionResult, on_first_token=None
    ):
        self.request_handler.forward_ct += 1
        forward_mode = (
            ForwardMode.EXTEND if forward_op.num_extends() > 0 else ForwardMode.DECODE
        )
        self.request_handler._profile_batch_predicate(forward_mode)

        # post_process_forward_op calls sync() — after this, CPU tensors are ready
        is_prefill_instance = isinstance(self.pd_kv_transfer, DisaggPrefillExecutor)
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

    def _process_pd_events(self, pd_events: list) -> list:
        processed = []
        for event in pd_events:
            processed.append(event)
            if isinstance(event, PD.SucceededEvent) and isinstance(
                self.pd_kv_transfer, DisaggPrefillExecutor
            ):
                req_id = event.request_id
                processed.extend(self.output_processor.finish_prefill_request(req_id))
            elif isinstance(event, PD.RemotePrefillDoneEvent):
                req_id = event.request_id
                bootstrap_token = event.bootstrap_token

                self.output_processor.on_remote_prefill_done(req_id, bootstrap_token)

        return processed

    def _get_load(self):
        """Return load metrics for the DP load balancer."""
        from tokenspeed.runtime.engine.io_struct import GetLoadReqOutput

        available = self.scheduler.available_kv_pages()
        num_total_pages = self.max_total_num_tokens // self.server_args.block_size
        num_used_pages = num_total_pages - available
        num_waiting = self.scheduler.waiting_size()
        # num_reqs: running + waiting (used by SHORTEST_QUEUE balancing)
        num_running = len(self.output_processor.rid_to_state)
        return GetLoadReqOutput(
            dp_rank=self.dp_rank,
            num_reqs=num_running + num_waiting,
            num_waiting_reqs=num_waiting,
            num_pages=num_used_pages,
        )

    def _dp_sync_and_check(self, forward_op) -> DpForwardMetadata:
        """Synchronize DP ranks with CPU-only metadata.

        All ranks call this before GPU forward work. The gathered metadata is
        used for eager token-aware collectives and for choosing a common padded
        CUDA graph shape during decode.
        """
        import torch.distributed as dist

        num_tokens = sum(forward_op.input_lengths) if forward_op is not None else 0
        batch_size = len(forward_op.request_ids) if forward_op is not None else 0
        if forward_op is None:
            forward_mode = ForwardMode.IDLE
        elif forward_op.num_extends() > 0:
            forward_mode = ForwardMode.EXTEND
        elif self.server_args.speculative_algorithm is not None:
            forward_mode = ForwardMode.TARGET_VERIFY
        else:
            forward_mode = ForwardMode.DECODE

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
                int(ForwardMode.TARGET_VERIFY),
            )
            for mode in global_forward_mode
        )
        return DpForwardMetadata(
            global_num_tokens=global_num_tokens,
            global_batch_size=global_batch_size,
            global_forward_mode=global_forward_mode,
            all_decode_or_idle=all_decode_or_idle,
            need_idle_forward=need_idle_forward,
        )

    def _get_scheduler_stats(self):
        """Query scheduler for page usage and queue depth."""
        available = self.scheduler.available_kv_pages()
        active = self.scheduler.active_kv_pages()
        num_total_pages = self.max_total_num_tokens // self.server_args.block_size
        return {
            "num_active_pages": active,
            "num_cached_pages": num_total_pages - available,
            "num_queue_reqs": self.scheduler.waiting_size(),
        }

    # ------------------------------------------------------------------
    # Event loops
    # ------------------------------------------------------------------

    def event_loop(self):
        """Non-overlapping scheduler loop."""
        while True:
            self._process_new_requests()
            self._commit_cache_results()
            execution_plan = self.scheduler.next_execution_plan()
            self._submit_cache_ops(execution_plan)

            forward_op = self._get_forward_op(execution_plan)

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
                    continue

            request_changes = []

            if forward_op is not None:
                stats = self._get_scheduler_stats()
                sampling_params_list = self._gather_sampling_params(forward_op)
                grammar_inputs = self._gather_grammar_state(forward_op)
                results, on_first_token = self._dispatch_forward(
                    forward_op,
                    sampling_params_list,
                    execution_plan,
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

            if self.pd_kv_transfer is not None:
                pd_events = self.pd_kv_transfer.generate_events()
                request_changes.extend(self._process_pd_events(pd_events))

            if request_changes:
                advance_forward(self.scheduler, request_changes)

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
            execution_plan = self.scheduler.next_execution_plan()

            self._submit_cache_ops(execution_plan)

            forward_op = self._get_forward_op(execution_plan)
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
                        prev_results = None
                        prev_forward_op = None
                    self.model_executor.execute_idle_forward(
                        dp_metadata.global_num_tokens,
                        dp_metadata.global_batch_size,
                        dp_metadata.all_decode_or_idle,
                    )
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
                stats = self._get_scheduler_stats()
                curr_results, _ = self._dispatch_forward(
                    forward_op,
                    sampling_params_list,
                    execution_plan,
                    dp_metadata=dp_metadata,
                    stats=stats,
                    grammar_inputs=grammar_inputs,
                )

            # ---- post-process previous step (overlapped with current forward) ----
            if prev_results is not None:
                request_changes.extend(
                    self._commit_forward_results(prev_forward_op, prev_results)
                )

            # ---- collect PD events ----
            if self.pd_kv_transfer is not None:
                pd_events = self.pd_kv_transfer.generate_events()
                request_changes.extend(self._process_pd_events(pd_events))

            if request_changes:
                advance_forward(self.scheduler, request_changes)

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

        # Prefill nodes have no steady-state decode stream to overlap over;
        # overlap scheduling adds complexity with negligible benefit, so always
        # fall back to the non-overlapping loop for prefill instances.
        is_prefill_instance = server_args.disaggregation_mode == "prefill"
        if not server_args.disable_overlap_schedule and not is_prefill_instance:
            event_loop.event_loop_overlap()
        else:
            event_loop.event_loop()

    except Exception:
        traceback = get_exception_traceback()
        logger.error("Scheduler hit an exception: %s", traceback)
        parent_process.send_signal(signal.SIGUSR1)
