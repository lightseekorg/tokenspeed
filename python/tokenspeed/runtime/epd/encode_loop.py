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

"""EPD encode-worker loop: a lightweight, LM-free scheduler subprocess.

The encode role runs the vision tower and ships image embeddings to prefill
workers over Mooncake; it owns no KV cache and never runs the language model, so
it does NOT use the full :class:`EventLoop`/C++ scheduler (that machinery -- paged
KV, chunked prefill, retract, token budget -- is an impedance mismatch for a ViT).
Instead ``run_event_loop`` branches here when ``disaggregation_mode == "encode"``.

The assembly is: load the model, stand up the Mooncake encode manager + bootstrap
server, the DisaggEncodeExecutor, and the EncodeWorker, reusing the SAME request
IPC the LM scheduler uses: a ZMQ PULL
on ``port_args.scheduler_input_ipc_name``. The smg grpc_servicer's TokenSpeedEncoder
handler sends an :class:`EncodeRequest` (pickled) over that channel; the loop
drains them into ``EncodeWorker.submit`` and runs ``step`` to encode + transfer.

TP: the vision tower is TP-sharded, so all encode ranks run each batch in lockstep
-- rank 0 owns the gateway ZMQ and broadcasts every batch to the TP group. The
embedding transport is a 1->N broadcast: prefill_tp must be a multiple of
encode_tp. Data-parallel encode workers inside one server are not supported;
horizontal scale comes from multiple independent encode servers selected by the
gateway.
"""

from __future__ import annotations

import time

import zmq

from tokenspeed.runtime.cache.embedding_cache import (
    EmbeddingCache,
    TieredEmbeddingCache,
)
from tokenspeed.runtime.epd.encode_scheduler import EncodeScheduler
from tokenspeed.runtime.epd.encode_worker import EncodeWorker
from tokenspeed.runtime.utils import get_colorful_logger, get_zmq_socket
from tokenspeed.runtime.utils.env import envs

logger = get_colorful_logger(__name__)


# Vision-embedding cache capacity. L1 lives in GPU VRAM; the optional L2 lives in
# host DRAM and catches L1 evictions so duplicate images skip the tower even past
# the VRAM working set. Both are whole-MiB env overrides. L2 defaults to 0
# (disabled): the host tier is opt-in. NOTE both knobs are PER ENCODE PROCESS (per
# TP rank): at encode TP>1, every co-located rank allocates its own L1+L2, so
# budget host DRAM as tp_size * EMBED_CACHE_DRAM_MB.
def _embedding_cache_bytes(env_field) -> int:
    """Whole-MiB env field -> bytes.

    EnvField handles parsing and defaults; negative capacities are rejected here
    with an env-named error.
    """
    mb = env_field.get()
    if mb < 0:
        raise ValueError(f"{env_field.name} must be >= 0 MiB, got {mb}")
    return mb * 1024 * 1024


def _make_embedding_cache(l1_bytes: int, l2_bytes: int, device: str):
    """Select the encode embedding cache: a plain single-tier VRAM
    :class:`EmbeddingCache` by default, or a two-tier :class:`TieredEmbeddingCache`
    (VRAM L1 + host-DRAM L2) when the L2 capacity is enabled (``l2_bytes > 0``)."""
    if l2_bytes > 0:
        logger.info(
            "EPD encode embedding cache: L1(VRAM)=%d MiB, L2(host DRAM)=%d MiB",
            l1_bytes >> 20,
            l2_bytes >> 20,
        )
        return TieredEmbeddingCache(l1_bytes, l2_bytes, device=device)
    logger.info(
        "EPD encode embedding cache: L1(VRAM)=%d MiB (host-DRAM L2 disabled)",
        l1_bytes >> 20,
    )
    return EmbeddingCache(l1_bytes)


def _build_manager_args(server_args, mapping):
    """EmbeddingManagerArgs for the encode Mooncake endpoint.

    EPD embedding transfer currently supports TP only. Horizontal scale is via
    separate encode servers selected by the gateway; attention DP inside one
    encode server is rejected so it cannot be accidentally advertised as TP.
    """
    from tokenspeed.runtime.epd.entities import EmbeddingManagerArgs

    if mapping.attn.dp_size != 1:
        raise ValueError(
            "disaggregation_mode=encode currently supports encode "
            f"data_parallel_size == 1, got {mapping.attn.dp_size}"
        )
    bootstrap_host = None
    if server_args.dist_init_addr:
        bootstrap_host = server_args.dist_init_addr.split(":", 1)[0]
    return EmbeddingManagerArgs(
        bootstrap_port=server_args.disaggregation_bootstrap_port,
        tp_size=mapping.attn.tp_size,
        bootstrap_host=bootstrap_host,
    )


def _maybe_install_encoder_cudagraph(model, server_args) -> bool:
    """Install the vision-encoder CUDA-graph wrapper as ``model.image_encoder``,
    mirroring the aggregated path's hook in ``execution/model_executor.py``.

    The encode loop never builds a ModelExecutor, so this is where the encode
    worker opts into capture/replay of the tower instead of running it eager. Same
    gate as the aggregated install: the model exposes the builder, multimodal is
    active, the env flag is on, and the attention backend is graph-capturable. The
    wrapper IS the model's ``image_encoder`` seam (lazy capture on first encode);
    the executor's IMAGE path dispatches through ``model.image_encoder`` and falls
    back to eager ``get_image_feature`` (the default) when this returns False.
    Returns whether the wrapper was installed.
    """
    if not (
        hasattr(model, "make_encoder_cudagraph_wrapper")
        and getattr(model, "is_multimodal_active", True)
        and envs.TOKENSPEED_MM_ENABLE_ENCODER_CUDA_GRAPH.get()
        and server_args.mm_attention_backend != "flashinfer_cudnn"
    ):
        return False
    model.image_encoder = model.make_encoder_cudagraph_wrapper(model.mapping)
    logger.info("EPD encode worker: vision-encoder CUDA graph installed")
    return True


def _build_encode_worker(server_args, port_args, gpu_id, global_rank):
    """Assemble the encode worker: model + Mooncake manager + bootstrap server +
    executor + scheduler + cache, driven from the real ServerArgs."""
    from tokenspeed.runtime.configs.model_config import ModelConfig
    from tokenspeed.runtime.epd.encode_executor import (
        DisaggEncodeExecutor,
    )
    from tokenspeed.runtime.epd.entities import EmbeddingArgs
    from tokenspeed.runtime.epd.mooncake.conn import MooncakeEmbeddingBootstrapServer
    from tokenspeed.runtime.epd.mooncake.encode import (
        MooncakeEmbeddingManagerEncode,
    )
    from tokenspeed.runtime.execution.distributed_initializer import (
        DistributedConfig,
        DistributedInitializer,
    )
    from tokenspeed.runtime.execution.factory import create_model_runner

    mapping = server_args.mapping
    attn_tp_rank = mapping.attn.tp_rank
    device = f"cuda:{gpu_id}"

    model_config = ModelConfig(
        server_args.model,
        trust_remote_code=server_args.trust_remote_code,
        revision=server_args.revision,
        context_length=server_args.max_model_len,
        model_override_args=server_args.hf_overrides,
        dtype=server_args.dtype,
        quantization=server_args.quantization,
        server_args=server_args,
    )
    DistributedInitializer.initialize(
        DistributedConfig.from_server_args(
            server_args=server_args,
            port_args=port_args,
            gpu_id=gpu_id,
            global_rank=global_rank,
            hidden_size=model_config.hidden_size,
            max_num_tokens=server_args.chunked_prefill_size or 8192,
        )
    )
    # The encode worker only needs the vision tower. The model is built
    # vision-only (LM construction + LM weight load skipped) via the
    # encoder_only gate derived from disaggregation_mode=="encode". The tower
    # is used via DisaggEncodeExecutor. No KV/mamba pool is allocated: the encode
    # loop never builds a ModelExecutor.
    model = create_model_runner(server_args, model_config, None, gpu_id, global_rank)[
        0
    ].model
    # Opt into vision-encoder CUDA-graph capture (mirrors the aggregated path,
    # which the encode worker bypasses by never building a ModelExecutor).
    _maybe_install_encoder_cudagraph(model, server_args)

    manager_args = _build_manager_args(server_args, mapping)
    embedding_args = EmbeddingArgs(
        engine_rank=global_rank,
        gpu_id=gpu_id,
        ib_device=server_args.disaggregation_ib_device,
        embedding_data_ptr=0,
        embedding_data_len=0,
    )
    # The encode worker is the Mooncake data source: it hosts its own bootstrap
    # server (prefill workers discover it via the handshake the gateway injects).
    # At TP>1 only rank 0 binds the port; every rank still registers its own
    # rank_ip/rank_port to the rank-0 bootstrap host, so each prefill rank can
    # look up the encode rank it pairs with (contiguous blocks of prefill ranks
    # share one encode rank; encode_tp=1 -> all prefill ranks pair encode rank 0).
    if attn_tp_rank == 0:
        MooncakeEmbeddingBootstrapServer(server_args.disaggregation_bootstrap_port)
    manager = MooncakeEmbeddingManagerEncode(manager_args, embedding_args)
    executor = DisaggEncodeExecutor(manager, model, device=device)

    l1_bytes = _embedding_cache_bytes(envs.TOKENSPEED_EPD_ENCODE_EMBED_CACHE_MB)
    l2_bytes = _embedding_cache_bytes(envs.TOKENSPEED_EPD_ENCODE_EMBED_CACHE_DRAM_MB)
    cache = _make_embedding_cache(l1_bytes, l2_bytes, device)
    scheduler = EncodeScheduler(
        max_tokens_per_batch=server_args.chunked_prefill_size or 8192,
        max_items_per_batch=server_args.max_num_seqs,
    )
    return EncodeWorker(executor, scheduler, cache), model_config


def run_encode_loop(server_args, port_args, pipe_writer, gpu_id, global_rank):
    """Run the encode-worker loop until the parent process exits.

    Drains :class:`EncodeRequest`s off the shared scheduler-input ZMQ channel into
    EncodeWorker.submit, then runs EncodeWorker.step to encode + ship each pending
    item over Mooncake. Synchronous; no KV, no LM forward.
    """
    worker, model_config = _build_encode_worker(
        server_args, port_args, gpu_id, global_rank
    )

    # TP coordination: the vision tower is TP-sharded (collective ops), so every
    # encode rank must run each batch in lockstep. Only rank 0 owns the gateway
    # ZMQ; it broadcasts each drained batch to the TP group so all ranks submit
    # the same requests and step together. (TP=1 -> rank 0 only, no broadcast.)
    attn_tp_rank = server_args.mapping.attn.tp_rank
    attn_tp_size = server_args.attn_tp_size or server_args.mapping.attn.tp_size
    tp_cpu_group = None
    broadcast_pyobj = None
    attn_tp_src_rank = server_args.mapping.attn.tp_group[0]
    if attn_tp_size > 1:
        from tokenspeed.runtime.distributed.process_group_manager import (
            process_group_manager as pg_manager,
        )
        from tokenspeed.runtime.utils.common import broadcast_pyobj as _bcast

        broadcast_pyobj = _bcast
        tp_cpu_group = pg_manager.get_process_group(
            "gloo", server_args.mapping.attn.tp_group
        )

    context = zmq.Context(2)
    recv_from_gateway = None
    if attn_tp_rank == 0:
        recv_from_gateway = get_zmq_socket(
            context, zmq.PULL, port_args.scheduler_input_ipc_name, False
        )

    # Unblock the launcher. The encode role has no KV pool, so the token/seq
    # fields are nominal (kept for envelope compatibility with the LM ready msg).
    pipe_writer.send(
        {
            "status": "ready",
            "max_total_num_tokens": 0,
            "max_req_input_len": model_config.context_len,
            "max_num_seqs": server_args.max_num_seqs,
            "chunked_prefill_size": server_args.chunked_prefill_size,
            "max_model_len": model_config.context_len,
        }
    )

    while True:
        # Rank 0 drains the gateway ZMQ without blocking; other ranks get the
        # same batch by broadcast below.
        new_reqs = []
        if attn_tp_rank == 0:
            while True:
                try:
                    request = recv_from_gateway.recv_pyobj(flags=zmq.NOBLOCK)
                except zmq.Again:
                    break
                new_reqs.append(request)

        if attn_tp_size > 1:
            # Unconditional every iteration: this is the TP rendezvous that keeps
            # all ranks stepping the (collective) vision tower in lockstep.
            new_reqs = broadcast_pyobj(
                new_reqs, attn_tp_rank, tp_cpu_group, src=attn_tp_src_rank
            )

        for request in new_reqs:
            worker.submit(request)
        drained = len(new_reqs) > 0

        # Encode + ship one scheduler batch. step() returns 0 when idle. All ranks
        # hold the same pending set, so they make the same step decision together.
        did = worker.step()

        # Yield the GIL when there is no fresh ZMQ work, OR when sends are deferred
        # on a full ring. The daemon transfer-workers that free ring slots are
        # GIL-bound, so spinning here would starve them and wedge the ring; a brief
        # yield lets them free slots so the next drain_deferred() can ship.
        if (
            not drained and did == 0 and not worker.has_pending()
        ) or worker.has_deferred():
            time.sleep(0.0005)
