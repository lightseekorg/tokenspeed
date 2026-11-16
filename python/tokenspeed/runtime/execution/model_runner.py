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

from __future__ import annotations

import glob
import inspect
import os
from contextlib import ExitStack
from itertools import chain
from typing import TYPE_CHECKING

import torch

from tokenspeed.runtime.configs.numerics import require_verified_numerics
from tokenspeed.runtime.distributed.process_group_manager import (
    process_group_manager as pg_manager,
)
from tokenspeed.runtime.execution.model_update import (
    ModelUpdateAdapter,
    model_update_adapter_for,
)
from tokenspeed.runtime.execution.multimodal_runtime import MultimodalRuntime
from tokenspeed.runtime.execution.weight_loader import WeightLoader
from tokenspeed.runtime.execution.weight_update_group import (
    _assert_not_split,
    _no_default_group_split,
)
from tokenspeed.runtime.layers.linear import (
    prepare_dp_linear_communication,
    release_dp_linear_communication,
)
from tokenspeed.runtime.layers.moe.utils import initialize_moe_config
from tokenspeed.runtime.model_loader.weight_utils import (
    non_unit_kv_scale_message,
    record_non_unit_kv_scales,
)
from tokenspeed.runtime.models.base.weight_update import (
    complete_weight_update_names,
    weight_update_session,
)
from tokenspeed.runtime.moe.expert_location import (
    build_expert_placement,
    get_global_expert_location_metadata,
    set_global_expert_location_metadata,
)
from tokenspeed.runtime.moe.expert_location_updater import (
    ExpertLocationUpdater,
    build_expert_location_updater,
)
from tokenspeed.runtime.multimodal.embedder import warmup_multimodal_encoders
from tokenspeed.runtime.utils import get_colorful_logger
from tokenspeed.runtime.utils.env import global_server_args_dict_update
from tokenspeed.runtime.utils.hf_transformers_utils import resolve_architecture
from tokenspeed.runtime.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter

if TYPE_CHECKING:
    from tokenspeed.runtime.configs.model_config import ModelConfig
    from tokenspeed.runtime.execution.context import ForwardContext
    from tokenspeed.runtime.layers.logits_processor import LogitsProcessorOutput
    from tokenspeed.runtime.model_loader.loader import (
        BaseModelLoader,
        DefaultModelLoader,
    )
    from tokenspeed.runtime.multimodal.inputs import MultimodalForwardContext
    from tokenspeed.runtime.utils.server_args import ServerArgs

logger = get_colorful_logger(__name__)


def infer_multimodal_encoder_dtype(model: torch.nn.Module) -> str | None:
    """Return the dtype of the loaded multimodal encoder when discoverable."""
    for path in (
        ("visual",),
        ("vision_tower",),
        ("audio_tower",),
        ("model", "visual"),
        ("model", "vision_tower"),
        ("model", "audio_tower"),
    ):
        component = model
        for name in path:
            component = getattr(component, name, None)
            if component is None:
                break
        else:
            dtype = getattr(component, "dtype", None)
            if isinstance(dtype, torch.dtype):
                return str(dtype).removeprefix("torch.")
            if isinstance(component, torch.nn.Module):
                for tensors in (component.parameters(), component.buffers()):
                    for tensor in tensors:
                        if tensor.is_floating_point():
                            return str(tensor.dtype).removeprefix("torch.")
    return None


def validate_output_vocabulary(model: torch.nn.Module, vocab_size: int) -> None:
    """Check the loaded target's logical logits width before allocating samplers."""
    # Models without logits (encoder workers and intermediate PP stages) have
    # no output vocabulary. Inspect the model interface, not padded/sharded weights.
    processor = getattr(model, "logits_processor", None)
    if processor is None:
        return
    actual = processor.config.vocab_size
    if actual != vocab_size:
        raise ValueError(
            f"{type(model).__name__} produces a logical output vocabulary of "
            f"{actual}, but the runtime vocab_size is {vocab_size}. "
            "The model must construct its output head and logits processor "
            "with the checkpoint's output_vocab_size."
        )


class ModelRunner:
    def __init__(
        self,
        # Configuration
        model_config: ModelConfig,
        server_args: ServerArgs,
        gpu_id: int,
        global_rank: int,
        *,
        checkpoint_load_group: tuple[int, ...] | None,
        is_draft_worker: bool = False,
    ):
        """Initialize ModelRunner with injected dependencies.

        Args:
            model_config: The model to build and load.
            server_args: Parsed server arguments.
            gpu_id: Local device index.
            global_rank: This worker's global rank.
            checkpoint_load_group: Global ranks that load this model together,
                for a distributed loader's collectives; None means every rank.
                A pipeline stage's draft names its stage, since the other
                stages may not build it (``create_model_runner``).
            is_draft_worker: Whether this is the speculative draft model.
        """
        # Store configuration
        self.model_config = model_config
        self.server_args = server_args
        self.device = server_args.device
        self.gpu_id = gpu_id
        self.global_rank = global_rank
        self.mapping = server_args.mapping
        self.is_generation = model_config.is_generation
        self.is_multimodal = model_config.is_multimodal
        self.is_draft_worker = is_draft_worker
        self.checkpoint_load_group = checkpoint_load_group
        # Tensor shapes of the checkpoint the last disk reload read; until one
        # succeeds, update_weights_from_disk reads them from model_path.
        self._disk_reload_shapes: dict[str, tuple[int, ...]] | None = None
        self._weight_update_pg: torch.distributed.ProcessGroup | None = None
        self._weight_update_device: torch.device | None = None
        # Model Updater SDK client for the Mooncake weight update; None
        # without --model-update-config and on the draft runner. Holds only
        # the arguments until the first update imports the SDK (on the
        # forward thread).
        self.model_update: ModelUpdateAdapter | None = model_update_adapter_for(
            server_args, global_rank=global_rank, is_draft_worker=is_draft_worker
        )
        # Set by load_model from the model's forward signature.
        self._model_forward_accepts_spec_step_idx: bool = False
        self.mambaish_config = getattr(model_config, "mambaish_config", None)
        self.is_hybrid_gdn = getattr(model_config, "is_hybrid_gdn", False)
        # Target and draft alike: the envelope covers every model that serves.
        require_verified_numerics(
            server_args.numerics,
            model_profile=model_config.model_profile,
            architecture=resolve_architecture(model_config.hf_config),
            quantization=model_config.quantization,
            vocab_size=model_config.vocab_size,
        )

        draft_moe_override = (
            self.is_draft_worker
            and server_args.draft_moe_backend is not None
            and server_args.draft_moe_backend != server_args.moe_backend
        )
        if draft_moe_override:
            saved_moe_backend = server_args.moe_backend
            server_args.moe_backend = server_args.draft_moe_backend

        # Auto-detect FP8 KV cache from checkpoint quant config (e.g. NVFP4 models
        # with kv_cache_quant_algo: "FP8" in hf_quant_config.json).
        if server_args.kv_cache_dtype == "auto":
            quant_cfg = model_config._parse_quant_hf_config()
            if quant_cfg is not None:
                kv_algo = quant_cfg.get("kv_cache_quant_algo")
                if isinstance(kv_algo, str) and kv_algo.upper() == "FP8":
                    server_args.kv_cache_dtype = "fp8_e4m3"
                    logger.info(
                        "Auto-detected kv_cache_dtype=fp8_e4m3 from checkpoint "
                        f"quant config (kv_cache_quant_algo={kv_algo!s})",
                    )

        global_server_args_dict_update(server_args)
        initialize_moe_config(server_args)

        self.memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=server_args.enable_memory_saver, device=self.device
        )
        # The target's expert placement (redundant replicas, load counters) is
        # process-global so its model picks it up while it is built and loaded.
        # A draft routes its own experts trivially: the target's placement is
        # hidden while the draft is built, then restored for serving.
        # Online rebalancing (--enable-eplb) moves the target's expert weights
        # between slots: its staging buffer and P2P channels are reserved here,
        # after the load and before the cache profile sizes the KV arena.
        self.expert_location_updater: ExpertLocationUpdater | None = None
        if self.is_draft_worker:
            target_placement = get_global_expert_location_metadata()
            set_global_expert_location_metadata(None)
            try:
                self.load_model()
            finally:
                set_global_expert_location_metadata(target_placement)
        else:
            set_global_expert_location_metadata(
                build_expert_placement(server_args, model_config)
            )
            self.load_model()
            if server_args.enable_eplb:
                self.expert_location_updater = build_expert_location_updater(
                    self.model, model_config, server_args
                )
        if draft_moe_override:
            server_args.moe_backend = saved_moe_backend
            global_server_args_dict_update(server_args)
            initialize_moe_config(server_args)

    def load_model(self):
        self.model = WeightLoader.load_model(
            model_config=self.model_config,
            server_args=self.server_args,
            device=self.device,
            gpu_id=self.gpu_id,
            memory_saver_adapter=self.memory_saver_adapter,
            checkpoint_load_group=self.checkpoint_load_group,
            is_draft_worker=self.is_draft_worker,
        )
        if not self.is_draft_worker:
            validate_output_vocabulary(self.model, self.model_config.vocab_size)
        self._model_forward_accepts_spec_step_idx = self._forward_accepts_kwarg(
            self.model, "spec_step_idx"
        )

    @property
    def forward_accepts_spec_step_idx(self) -> bool:
        """Whether the model's ``forward`` declares ``spec_step_idx``.

        :meth:`forward` passes ``spec_step_idx`` through only when this holds
        (``**kwargs`` alone does not count); a drafter that selects depth by
        step must check it at construction rather than discover at serve time
        that every step ran depth 0.
        """
        return self._model_forward_accepts_spec_step_idx

    @property
    def multimodal_encoder_dtype(self) -> str | None:
        return infer_multimodal_encoder_dtype(self.model)

    def prepare_multimodal_runtime(self) -> None:
        """Prepare loaded multimodal encoders for serving.

        This is an explicit post-load phase because it can execute substantial
        GPU work. Language workers must call it before KV-cache memory
        profiling so retained encoder graph pools and lazy buffers are included
        in the cache budget. Encoder-only EPD workers call the same phase even
        though they do not allocate a KV cache.
        """
        self.encoder_graph_wrappers = MultimodalRuntime.install_encoder_graphs(
            self.model, self.server_args
        )
        if self.encoder_graph_wrappers:
            logger.info(
                "Multimodal encoder CUDA graphs installed for "
                f"{sorted(self.encoder_graph_wrappers)!s}",
            )

        warmup_device = torch.device(self.device)
        if warmup_device.type == "cuda" and warmup_device.index is None:
            warmup_device = torch.device("cuda", self.gpu_id)
        warmup_multimodal_encoders(
            self.model,
            device=warmup_device,
        )

    def prepare_communication_runtime(self, max_num_tokens: int) -> bool:
        """Allocate model communication buffers before cache planning.

        The expert load counters' live-row mask is reserved here too: it is
        sized by the largest MoE input a forward can carry (the MoE TP-EP
        group's all-gather of ``max_num_tokens`` rows per rank), must exist
        before the first forward (captured graphs hold its address) and is
        accounted for by the cache profile like the communication buffers.
        """
        placement = get_global_expert_location_metadata()
        if (
            not self.is_draft_worker
            and placement is not None
            and placement.load_rows is not None
        ):
            placement.reserve_load_rows(
                self.server_args.mapping.moe.tp_ep_size * max_num_tokens
            )
        prepared = prepare_dp_linear_communication(
            self.model, max_num_tokens, self.model_config.dtype, backend=None
        )
        prepare = getattr(self.model, "prepare_communication_runtime", None)
        if prepare is None:
            return prepared
        return bool(prepare(max_num_tokens)) or prepared

    def release_dp_linear_communication(self) -> None:
        """Collectively release this model's DP projection communication only.

        Drain forwards and destroy referencing graphs first; keep process groups
        alive until release finishes on every peer. Cache re-capture retains
        these resources, so graph release alone must not call this method.
        """
        release_dp_linear_communication(self.model)

    @staticmethod
    def _forward_accepts_kwarg(model, name: str) -> bool:
        try:
            parameters = inspect.signature(model.forward).parameters
        except (TypeError, ValueError):
            return False

        return name in parameters

    def forward(
        self,
        ctx: ForwardContext,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        captured_hidden_states: torch.Tensor | None = None,
        input_embeds: torch.Tensor | None = None,
        multimodal_context: MultimodalForwardContext | None = None,
        spec_step_idx: int | None = None,
        pp_inbound=None,
        **kwargs,
    ) -> LogitsProcessorOutput:
        if pp_inbound is not None:
            kwargs["pp_inbound"] = pp_inbound
        if not self.is_generation:
            kwargs["get_embedding"] = True
        if captured_hidden_states is not None:
            kwargs["captured_hidden_states"] = captured_hidden_states
        if input_embeds is not None:
            kwargs["input_embeds"] = input_embeds
        if multimodal_context is not None:
            kwargs["multimodal_context"] = multimodal_context
        if spec_step_idx is not None and self.forward_accepts_spec_step_idx:
            kwargs["spec_step_idx"] = spec_step_idx

        return self.model.forward(
            ctx,
            input_ids,
            positions,
            **kwargs,
        )

    # ------------------------------------------------------------------ #
    # RL online weight sync: receive NCCL-broadcast weights from a trainer.
    #
    # Mirrors the slime/SGLang sender contract: the trainer occupies ranks
    # ``0..rank_offset-1`` and each inference worker joins at
    # ``rank_offset + global_rank``. The trainer broadcasts each named weight
    # from rank 0 (in ``names`` order); this worker receives in the same order
    # and applies via the model's ``load_weights`` (the same name->param mapping,
    # including fused/stacked params, used by the initial load).
    # ------------------------------------------------------------------ #

    def init_weights_update_group(self, obj) -> tuple[bool, str]:
        """Join the trainer's ``torch.distributed`` NCCL weight-update group.

        The trainer (slime/sglang dialect) creates the peer group with
        ``init_process_group(init_method="tcp://addr:port", rank=0, world_size)``
        and pushes weights via ``dist.broadcast(..., src=0)``. We must rendezvous
        through the *same* torch TCP-store + NCCL-unique-id handshake — a
        ``StatelessProcessGroup``/``PyNcclCommunicator`` keys its store
        differently and never forms a joint communicator with a torch group, so
        the broadcast would deadlock. Build a standalone, non-default group (via
        the same private helper torch's own ``init_process_group`` uses) so it
        never collides with the engine's own world. Because that world group is
        created with a bound device, we also have to guard the new group
        against torch silently splitting it off the engine's own communicator
        instead of rendezvousing with the trainer.
        """
        from packaging.version import parse as _parse_version
        from torch.distributed.distributed_c10d import (
            Backend,
            PrefixStore,
            _new_process_group_helper,
            _world,
            default_pg_timeout,
            rendezvous,
        )

        try:
            rank = int(obj.rank_offset) + self.global_rank
            world_size = int(obj.world_size)
            group_name = str(getattr(obj, "group_name", "weight_update_group"))
            backend = Backend(str(getattr(obj, "backend", "nccl")))
            device = torch.device(f"cuda:{self.gpu_id}")
            torch.cuda.set_device(device)

            timeout = default_pg_timeout
            init_method = f"tcp://{obj.master_address}:{int(obj.master_port)}"
            store, rank, world_size = next(
                rendezvous(init_method, rank, world_size, timeout=timeout)
            )
            store.set_timeout(timeout)
            store = PrefixStore(group_name, store)
            opt = (
                "backend_options"
                if _parse_version(torch.__version__) >= _parse_version("2.6")
                else "pg_options"
            )
            with _no_default_group_split():
                pg, _ = _new_process_group_helper(
                    world_size,
                    rank,
                    [],
                    backend,
                    store,
                    group_name=group_name,
                    **{opt: None},
                    timeout=timeout,
                )
            _world.pg_group_ranks[pg] = {i: i for i in range(world_size)}
            try:
                _assert_not_split(pg, device)
            except RuntimeError:
                torch.distributed.destroy_process_group(pg)
                raise

            self._weight_update_pg = pg
            self._weight_update_device = device
            logger.info(
                f"weight-update group joined: rank={rank:d} world_size={world_size:d} "
                f"device={device!s} group={group_name!s}",
            )
            return True, "weight update group initialized"
        except Exception as e:  # noqa: BLE001 - surface to the control plane
            logger.exception("init_weights_update_group failed")
            return False, str(e)

    def update_weights_from_distributed(self, obj) -> tuple[bool, str]:
        """Receive trainer-broadcast weights over the NCCL group and load them."""
        import torch.distributed as dist

        if self.mapping.moe_weight_sharing_size is not None:
            return (
                False,
                "Peer expert weights are immutable; restart to change the checkpoint",
            )
        pg = self._weight_update_pg
        if pg is None:
            return False, "weight update group not initialized"
        try:
            names = list(obj.names)
            dtype_names = list(obj.dtype_names)
            shapes = [tuple(s) for s in obj.shapes]
            device = self._weight_update_device

            def _recv():
                # NCCL broadcasts are ordered collectives: receive each weight in
                # the trainer's send order (rank 0 is the trainer) and hand it to
                # the model's loader (the same name->param mapping, including
                # fused/stacked params, used by the initial load).
                for name, dtype_name, shape in zip(names, dtype_names, shapes):
                    buf = torch.empty(
                        shape, dtype=getattr(torch, dtype_name), device=device
                    )
                    dist.broadcast(buf, src=0, group=pg)
                    yield name, buf

            # The update loads to completion so the model stays consistent,
            # then fails on a scale. A BaseCausalLM session screens the stream
            # itself and raises at its end; this wrap covers the models that
            # take no session (multimodal wrappers).
            rejected: list[str] = []
            with weight_update_session(
                [self.model], require_complete=False, cpu_group=None
            ):
                self.model.load_weights(record_non_unit_kv_scales(_recv(), rejected))
            torch.cuda.synchronize(device)
            if rejected:
                return False, (
                    f"applied {len(names)} weights, but the update is rejected and "
                    f"its weight version not advanced: {non_unit_kv_scale_message(rejected)}; "
                    "resend the weights without KV-cache scales"
                )
            return True, f"updated {len(names)} weights"
        except Exception as e:  # noqa: BLE001 - surface to the control plane
            logger.exception("update_weights_from_distributed failed")
            return False, str(e)

    def check_ascend_mooncake_update(
        self, models: list[torch.nn.Module]
    ) -> tuple[bool, str]:
        """Check the local model contract without opening the SDK or writing weights."""
        from tokenspeed.runtime.model_loader.loader import post_load_transformed_modules

        if self.model_update is None:
            return False, "Mooncake weight updates require --model-update-config"
        for model in models:
            try:
                complete_weight_update_names(model)
                if not model.supports_weight_discard:
                    raise ValueError(
                        f"{type(model).__name__} has no audited Ascend in-place weight update"
                    )
                transformed = post_load_transformed_modules(model)
                if transformed:
                    raise ValueError(
                        f"{type(model).__name__} transformed checkpoint weights at startup: "
                        + ", ".join(transformed)
                    )
            except (TypeError, ValueError, NotImplementedError) as exc:
                return False, str(exc)
        return True, "ready for weight update"

    def update_weights_from_mooncake(
        self,
        version: int,
        models: list[torch.nn.Module],
        *,
        require_complete: bool,
        snapshot: dict | None = None,
    ) -> tuple[bool, str]:
        """Read one committed weight-store version into ``models`` in place.

        Runs on the forward thread. The SDK streams partial ``load_weights``
        calls, so the models are bracketed in a weight-update session. Wait for
        the runner's accelerator to complete the update before acknowledging it
        and clear its allocator cache on both success and failure.

        Args:
            version: The committed version to load.
            models: Target first, then the draft when the server's
                ``--model-update-draft-weights`` is ``refresh``.
            require_complete: Validate the entire canonical source inventory;
                required whenever old model weights were discarded. The Ascend
                memory saver checks the complete stream for its in-place layout.
            snapshot: Optional SDK publication directory for this version.

        Returns:
            ``(ok, message)`` for the control plane.
        """
        if self.mapping.moe_weight_sharing_size is not None:
            return (
                False,
                "Peer expert weights are immutable; restart to change the checkpoint",
            )
        if self.model_update is None:
            return False, (
                "update_weights_from_mooncake requires the server to start with "
                "--model-update-config (target runner only)"
            )
        if self.device == "npu" and self.server_args.enable_memory_saver:
            return self._update_ascend_weights_from_mooncake(models, version, snapshot)
        device_module = torch.get_device_module(self.device)
        try:
            with weight_update_session(
                models,
                require_complete=require_complete,
                cpu_group=pg_manager.get_process_group(
                    "gloo", self.mapping.world_group
                ),
            ):
                result = self.model_update.update(models, version, snapshot=snapshot)
            device_module.synchronize(self.gpu_id)
            return True, f"applied model version {version}: {result}"
        except Exception as e:  # noqa: BLE001 - surface to the control plane
            logger.exception("update_weights_from_mooncake failed")
            return False, str(e)
        finally:
            device_module.empty_cache()

    def _update_ascend_weights_from_mooncake(
        self, models: list[torch.nn.Module], version: int, snapshot: dict | None
    ) -> tuple[bool, str]:
        """Stream a complete SDK version into resident Ascend parameter storage.

        Models are ordered with the target first so it owns writes to shared
        parameters. The SDK supplies ready CPU or local NPU tensors. Wait for
        copies inside the session so every rank reports device errors before
        post-load hooks, and wait for queued device work before freeing staging
        cache. A failed read, an explicit SDK False, or a device error makes the
        local update fail. The caller owns replica-wide version commit.
        """
        assert self.model_update is not None
        device = torch.device(self.device, self.gpu_id)
        device_module = torch.get_device_module(device)
        try:
            with ExitStack() as cleanup:
                cleanup.callback(device_module.empty_cache)
                # Post-load work can queue kernels even on an error path.
                cleanup.callback(device_module.synchronize, device)
                with weight_update_session(
                    models,
                    require_complete=True,
                    cpu_group=pg_manager.get_process_group(
                        "gloo", self.mapping.world_group
                    ),
                    source_device=device,
                ):
                    try:
                        result = self.model_update.update(
                            models, version, snapshot=snapshot
                        )
                        if result is False:
                            raise RuntimeError(
                                f"Model Updater SDK reported failure for version {version}"
                            )
                        message = str(result)
                    finally:
                        # Include asynchronous copy errors in rank agreement,
                        # before any model can enter post-load collectives.
                        device_module.synchronize(device)
            return True, f"applied model version {version}: {message}"
        except Exception as e:  # noqa: BLE001 - surface to the control plane
            logger.exception("update_weights_from_mooncake failed")
            return False, str(e)

    def update_weights_from_disk(
        self, obj, *, require_complete: bool = False
    ) -> tuple[bool, str]:
        """Reload this worker's parameters in place from a checkpoint directory.

        The checkpoint is streamed the way the startup loader streams it
        (``DefaultModelLoader.get_checkpoint_weights``) into the model's own
        ``load_weights``, so fused and stacked parameters map, the checkpoint
        directory is bound and the load group is chosen as on the initial
        load. The reported count comes from the tensors the loader actually
        yielded, not from the request: a checkpoint that yields nothing is a
        failure, not a success that updated nothing.

        Refusals write nothing: a model whose weights the startup loader
        transformed after loading (``post_load_transformed_modules``) cannot
        take checkpoint tensors at all, and a safetensors checkpoint with an
        unreadable shard or a tensor shaped differently from the checkpoint
        the model holds would fail partway through the load. A failure after
        tensors have reached the model says the weights may be partial.
        """
        if require_complete and self.device == "npu":
            return self._update_complete_ascend_disk(obj)

        from tokenspeed.runtime.model_loader.loader import (
            DefaultModelLoader,
            post_load_transformed_modules,
        )

        if self.mapping.moe_weight_sharing_size is not None:
            return (
                False,
                "Peer expert weights are immutable; restart to change the checkpoint",
            )
        model_path = str(obj.model_path)
        transformed = post_load_transformed_modules(self.model)
        if transformed:
            more = len(transformed) - 3
            return False, (
                "update_weights_from_disk cannot reload this model in place: the "
                "startup loader transformed its weights after loading ("
                + ", ".join(transformed[:3])
                + (f", and {more} more" if more > 0 else "")
                + "), and a checkpoint holds them untransformed. Nothing was "
                f"written; restart the engine with {model_path!r} instead"
            )
        load_format = obj.load_format or self.server_args.load_format
        consumed = 0
        try:
            loader = self._disk_reload_loader(load_format)
            if not isinstance(loader, DefaultModelLoader):
                return False, (
                    f"load_format {load_format!r} selects "
                    f"{type(loader).__name__}, which does not read checkpoint "
                    "files; an in-place reload needs a file-backed format"
                )

            shapes, problem = self._check_disk_reload(loader, model_path)
            if problem is not None:
                return False, f"{problem}; nothing was written"

            redundant_names = frozenset()
            if require_complete:
                # Validate canonical sources before any writes when headers are
                # available. The session also checks the actual tensor stream.
                redundant_sources = getattr(
                    self.model, "weight_discard_redundant_checkpoint_names", None
                )
                if redundant_sources is not None:
                    redundant_names = redundant_sources()
                if shapes is not None:
                    required_names = self.model.required_weight_update_names()
                    sources = shapes.keys() - redundant_names
                    missing = required_names - sources
                    unexpected = sources - required_names
                    if missing or unexpected:
                        return False, (
                            "Incomplete checkpoint: "
                            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}; "
                            "nothing was written"
                        )
            weights = loader.get_checkpoint_weights(
                model_path, None, self.model, self.model_config.mapping
            )

            def _counted():
                nonlocal consumed
                for name, tensor in weights:
                    # A tied export can duplicate storage under a second name.
                    if name in redundant_names:
                        continue
                    consumed += 1
                    yield name, tensor

            # Same session and KV-cache scale screen as a distributed load.
            rejected: list[str] = []
            with weight_update_session(
                [self.model], require_complete=require_complete, cpu_group=None
            ):
                self.model.load_weights(
                    record_non_unit_kv_scales(
                        _counted(),
                        rejected,
                    )
                )
            if consumed == 0:
                return False, (
                    f"no checkpoint tensors found at {model_path!r}; "
                    "nothing was updated"
                )
            if rejected:
                return False, (
                    f"{non_unit_kv_scale_message(rejected)}; a weight update keeps "
                    f"the KV-cache scales loaded at startup ({consumed} checkpoint "
                    "tensors had reached the model)"
                )
            if self.device == "npu":
                torch.npu.synchronize(self.gpu_id)
            elif self.device != "cpu":
                torch.cuda.synchronize(torch.device(f"cuda:{self.gpu_id}"))
        except Exception as e:  # noqa: BLE001 - surface to the control plane
            logger.exception("update_weights_from_disk failed")
            if consumed == 0:
                return False, str(e)
            return False, (
                f"{e} (after {consumed} checkpoint tensors had reached the "
                "model, so its weights may now mix the old and new "
                "checkpoints: reload a complete checkpoint or restart the "
                "engine)"
            )

        self.model_config.model_path = model_path
        if shapes is not None:
            self._disk_reload_shapes = shapes
        logger.info(f"weights reloaded from {model_path!s} ({consumed:d} tensors)")
        return True, (
            f"Succeeded to update model weights from {model_path} "
            f"({consumed} checkpoint tensors read)"
        )

    def _disk_reload_loader(self, load_format: str) -> BaseModelLoader:
        """Construct the checkpoint loader with the runner's startup settings.

        Reuse the download, prefetch and rank-group settings used to build the
        model. Ascend saver admission restricts the returned reader before
        it opens a checkpoint so file checks can finish without a device
        collective or a stale cached copy.
        """
        from tokenspeed.runtime.configs.load_config import LoadConfig
        from tokenspeed.runtime.model_loader.loader import get_model_loader

        return get_model_loader(
            LoadConfig(
                load_format=load_format,
                download_dir=self.server_args.download_dir,
                ext_yaml=self.server_args.ext_yaml,
                weight_loader_prefetch_checkpoints=self.server_args.weight_loader_prefetch_checkpoints,
                weight_loader_prefetch_num_threads=self.server_args.weight_loader_prefetch_num_threads,
                checkpoint_load_group=self.checkpoint_load_group,
            )
        )

    def check_ascend_disk_update(self, obj) -> tuple[bool, str]:
        """Check saver disk formats before cache invalidation or a transaction."""
        try:
            self._ascend_disk_reload_loader(obj)
        except Exception as exc:
            return False, f"{exc}; nothing was written"
        return True, "ready for Ascend disk weight update"

    def _ascend_disk_reload_loader(self, obj) -> DefaultModelLoader:
        """Use a direct reader and refuse an ambiguous AUTO checkpoint folder."""
        from tokenspeed.runtime.configs.load_config import LoadFormat
        from tokenspeed.runtime.model_loader.loader import (
            CheckpointFileSelection,
            DefaultModelLoader,
        )
        from tokenspeed.runtime.model_loader.weight_utils import (
            filter_duplicate_safetensors_files,
            filter_files_not_needed_for_inference,
        )

        load_format = obj.load_format or self.server_args.load_format
        direct_formats = (
            LoadFormat.AUTO,
            LoadFormat.SAFETENSORS,
            LoadFormat.PT,
            LoadFormat.MISTRAL,
        )
        # Validate before constructing a custom loader. NPCACHE can retain an
        # older checkpoint at the same path; InstantTensor is a CUDA reader.
        if (
            not isinstance(load_format, str)
            or load_format.lower() not in direct_formats
        ):
            raise ValueError(
                "Ascend Memory Saver disk reload requires a direct CPU checkpoint "
                f"reader; load_format {load_format!r} is unsupported. Use "
                "'safetensors', 'pt', 'mistral', or 'auto' with one checkpoint "
                "file family"
            )
        loader = self._disk_reload_loader(load_format)
        if type(loader) is not DefaultModelLoader:
            raise ValueError(
                "Ascend Memory Saver disk reload requires DefaultModelLoader"
            )
        fall_back_to_pt = getattr(self.model, "fall_back_to_pt_during_load", False)
        weight_files = (
            self.model.checkpoint_weight_files
            if isinstance(self.model, CheckpointFileSelection)
            else None
        )
        folder, _, _ = loader._prepare_weights(
            str(obj.model_path), None, fall_back_to_pt, weight_files
        )
        if loader.load_config.load_format == LoadFormat.AUTO and weight_files is None:
            # Follow startup's shard selection and exclusions, but inspect all
            # eligible families instead of silently preferring older safetensors.
            families = []
            safetensors = glob.glob(os.path.join(folder, "*.safetensors"))
            if safetensors:
                safetensors = filter_duplicate_safetensors_files(
                    safetensors, folder, "model.safetensors.index.json"
                )
            if safetensors:
                families.append("safetensors")
            extensions = ("bin", "pt") if fall_back_to_pt else ("bin",)
            for extension in extensions:
                files = filter_files_not_needed_for_inference(
                    glob.glob(os.path.join(folder, f"*.{extension}"))
                )
                if files:
                    families.append(extension)
            if len(families) > 1:
                raise ValueError(
                    f"Ascend Memory Saver load_format 'auto' is ambiguous at "
                    f"{str(obj.model_path)!r}: found checkpoint file families "
                    f"{', '.join(families)}. Select an explicit load_format or "
                    "publish one checkpoint family in an immutable version directory"
                )
        return loader

    def _update_complete_ascend_disk(self, obj) -> tuple[bool, str]:
        """Restore discarded Ascend weights after their storage is remapped.

        The caller has released KV and begun a weight transaction. Every TP rank
        reads the candidate inventory directly on CPU. The first agreement
        prevents any write when a rank refuses that inventory. The complete
        session validates the actual stream and gathers read and copy failures
        again before post-load hooks. Return local success only after device
        work completes. The host commits the replica result. A failed load leaves
        the begun transaction invalid even when the inventory refused all writes.
        """
        from tokenspeed.runtime.model_loader.loader import (
            post_load_transformed_modules,
        )

        model_path = str(obj.model_path)
        consumed = 0
        try:
            # Ascend memory saver uses one single-stage TP replica. Match the
            # Rust weight request's participants, without involving other ranks.
            tp_group = self.mapping.attn.tp_group
            cpu_group = (
                pg_manager.get_process_group("gloo", tp_group)
                if len(tp_group) > 1
                else None
            )
            preparation_error = None
            try:
                transformed = post_load_transformed_modules(self.model)
                if transformed:
                    raise ValueError(
                        f"Cannot reload transformed Ascend weights: {transformed}"
                    )
                loader = self._ascend_disk_reload_loader(obj)
                shapes = self._check_complete_ascend_disk_reload(loader, model_path)
            except Exception as exc:
                preparation_error = exc
            if cpu_group is not None:
                # A header or local read error must stop peers before any
                # rank starts writing parameters.
                failures = [None] * torch.distributed.get_world_size(cpu_group)
                torch.distributed.all_gather_object(
                    failures,
                    None if preparation_error is None else str(preparation_error),
                    group=cpu_group,
                )
                if preparation_error is None:
                    for rank, failure in enumerate(failures):
                        if failure is not None:
                            raise ValueError(
                                f"Disk reload refused by group rank {rank}: {failure}"
                            )
            if preparation_error is not None:
                raise preparation_error

            def _counted():
                nonlocal consumed
                for name, tensor in loader.get_checkpoint_weights(
                    model_path, None, self.model, self.model_config.mapping
                ):
                    consumed += 1
                    yield name, tensor

            # Check the actual stream again in case a checkpoint changed
            # after preflight, and converge before entering post-load hooks.
            with weight_update_session(
                [self.model],
                require_complete=True,
                cpu_group=cpu_group,
                source_device=torch.device("cpu"),
            ):
                try:
                    self.model.load_weights(_counted())
                    if not consumed:
                        raise ValueError(
                            f"no checkpoint tensors found at {model_path!r}; nothing was updated"
                        )
                finally:
                    torch.npu.synchronize(self.gpu_id)
            torch.npu.synchronize(self.gpu_id)
        except Exception as exc:  # noqa: BLE001 - surface to the control plane
            logger.exception("complete Ascend disk weight update failed")
            if consumed:
                return (
                    False,
                    f"{exc} (after {consumed} checkpoint tensors reached the model)",
                )
            return False, str(exc)

        self.model_config.model_path = model_path
        self._disk_reload_shapes = shapes
        logger.info(f"weights reloaded from {model_path!s} ({consumed:d} tensors)")
        return (
            True,
            f"Succeeded to update model weights from {model_path} ({consumed} checkpoint tensors read)",
        )

    def _check_complete_ascend_disk_reload(
        self, loader: DefaultModelLoader, model_path: str
    ) -> dict[str, tuple[int, ...]]:
        """Validate canonical source names and shapes before the first TP slice.

        Use only the candidate files and the live model's checkpoint contract.
        Safetensors provides a header per shard. Binary checkpoints require a
        bounded scan with the direct reader and a second read during loading.
        Declared aliases are optional and have no independent destination.
        Duplicate sources still fail. This check writes no model parameters.
        """
        from tokenspeed.runtime.model_loader.loader import CheckpointFileSelection
        from tokenspeed.runtime.model_loader.weight_utils import (
            safetensors_tensor_shapes,
        )

        required = complete_weight_update_names(self.model)
        expected = self.model.weight_update_source_shapes()
        redundant = self.model.weight_discard_redundant_checkpoint_names()
        if required & redundant:
            raise ValueError("Required checkpoint sources cannot also be aliases")
        if missing_shapes := required - expected.keys():
            raise ValueError(
                f"Missing checkpoint source shapes for {sorted(missing_shapes)}; nothing was written"
            )
        fall_back_to_pt = getattr(self.model, "fall_back_to_pt_during_load", False)
        weight_files = (
            self.model.checkpoint_weight_files
            if isinstance(self.model, CheckpointFileSelection)
            else None
        )
        _, files, use_safetensors = loader._prepare_weights(
            model_path, None, fall_back_to_pt, weight_files
        )
        if use_safetensors:
            entries = chain.from_iterable(
                safetensors_tensor_shapes([file]).items() for file in files
            )
        else:
            # Binary checkpoints have no independent tensor header. Read the
            # inventory one shard at a time so missing or oversized sources
            # are detected before any TP loader can slice or write a tensor.
            entries = (
                (name, tuple(tensor.shape))
                for name, tensor in loader.get_checkpoint_weights(
                    model_path, None, self.model, self.model_config.mapping
                )
            )
        shapes = {}
        for name, shape in entries:
            if name in shapes:
                raise ValueError(
                    f"Duplicate checkpoint weight {name!r}; nothing was written"
                )
            shapes[name] = shape
        sources = shapes.keys() - redundant
        if sources != required:
            raise ValueError(
                "Incomplete checkpoint: "
                f"missing={sorted(required - sources)}, unexpected={sorted(sources - required)}; "
                "nothing was written"
            )
        for name in sorted(required):
            if shapes[name] != expected[name]:
                raise ValueError(
                    f"Checkpoint weight {name!r} has shape {shapes[name]}; "
                    f"expected checkpoint shape {expected[name]} before TP slicing; "
                    "nothing was written"
                )
        return shapes

    def _check_disk_reload(
        self, loader, model_path: str
    ) -> tuple[dict[str, tuple[int, ...]] | None, str | None]:
        """Check a safetensors checkpoint's headers before any tensor is loaded.

        Both checks read headers only. A shard that will not parse or is
        shorter than its header declares, or a tensor whose shape differs from
        the same name in the checkpoint this model holds, would fail the load
        partway, after the tensors before it were written. Names the two
        checkpoints do not share are left to ``load_weights``, which alone
        knows how it maps them.

        Returns:
            ``(shapes, problem)``: the checkpoint's tensor shapes (None when
            it is not safetensors) and why it must be refused (None if not).
        """
        from tokenspeed.runtime.model_loader.loader import CheckpointFileSelection
        from tokenspeed.runtime.model_loader.weight_utils import (
            safetensors_tensor_shapes,
        )

        fall_back_to_pt = getattr(self.model, "fall_back_to_pt_during_load", False)
        weight_files = (
            self.model.checkpoint_weight_files
            if isinstance(self.model, CheckpointFileSelection)
            else None
        )
        _, files, use_safetensors = loader._prepare_weights(
            model_path, None, fall_back_to_pt, weight_files
        )
        if not use_safetensors:
            return None, None
        try:
            shapes = safetensors_tensor_shapes(files)
        except ValueError as e:
            return None, str(e)

        held = self._disk_reload_shapes
        if held is None:
            try:
                _, held_files, held_safetensors = loader._prepare_weights(
                    self.model_config.model_path,
                    getattr(self.model_config, "revision", None),
                    fall_back_to_pt,
                    weight_files,
                )
                held = safetensors_tensor_shapes(held_files) if held_safetensors else {}
            except Exception as e:  # noqa: BLE001 - only the shape check is lost
                logger.warning(
                    "update_weights_from_disk: cannot read the shapes of "
                    f"{self.model_config.model_path!s} ({e!s}); "
                    "checking the new checkpoint's shards only"
                )
                held = {}
        # The default loader fills a one-element parameter from any
        # one-element tensor, so those match whatever their shapes.
        mismatched = [
            (name, shape, held[name])
            for name, shape in shapes.items()
            if name in held
            and shape != held[name]
            and not (torch.Size(shape).numel() == 1 == torch.Size(held[name]).numel())
        ]
        if mismatched:
            name, shape, expected = mismatched[0]
            more = len(mismatched) - 1
            return shapes, (
                f"{model_path}: tensor {name!r} has shape {list(shape)}, but "
                f"{list(expected)} in the checkpoint the model holds"
                + (f" ({more} more mismatched)" if more else "")
            )
        return shapes, None

    def destroy_weights_update_group(self, obj) -> tuple[bool, str]:
        """Tear down the trainer weight-update NCCL group joined in ``init``.

        When a training run ends the trainer drops its end of the group, so the
        worker must release its side too -- free the NCCL communicator and the
        torch ``_world`` bookkeeping ``init_weights_update_group`` registered --
        instead of leaking it until engine shutdown. A fresh run then re-inits a
        clean group. Idempotent: tearing down when no group is live is a success
        so a trainer that always calls destroy (e.g. slime) never errors.
        """
        pg = self._weight_update_pg
        if pg is None:
            return True, "weight update group not initialized"

        import torch.distributed as dist
        from torch.distributed.distributed_c10d import _world

        try:
            dist.destroy_process_group(pg)
            # init() registered this group's rank map via the low-level helper;
            # drop it explicitly in case destroy_process_group left it behind.
            _world.pg_group_ranks.pop(pg, None)
        except Exception as e:  # noqa: BLE001 - surface to the control plane
            logger.exception("destroy_weights_update_group failed")
            return False, str(e)
        self._weight_update_pg = None
        self._weight_update_device = None
        logger.info("weight-update group destroyed")
        return True, "weight update group destroyed"
