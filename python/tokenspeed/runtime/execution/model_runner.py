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

import inspect
from typing import TYPE_CHECKING

import torch

from tokenspeed.runtime.execution.weight_loader import WeightLoader
from tokenspeed.runtime.layers.moe.utils import initialize_moe_config
from tokenspeed.runtime.utils import get_colorful_logger
from tokenspeed.runtime.utils.env import global_server_args_dict_update
from tokenspeed.runtime.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter

if TYPE_CHECKING:
    from tokenspeed.runtime.configs.model_config import ModelConfig
    from tokenspeed.runtime.execution.context import ForwardContext
    from tokenspeed.runtime.layers.logits_processor import LogitsProcessorOutput
    from tokenspeed.runtime.multimodal.inputs import MultimodalForwardContext
    from tokenspeed.runtime.utils.server_args import ServerArgs

logger = get_colorful_logger(__name__)


class ModelRunner:
    def __init__(
        self,
        # Configuration
        model_config: ModelConfig,
        server_args: ServerArgs,
        gpu_id: int,
        global_rank: int,
        is_draft_worker: bool = False,
    ):
        """Initialize ModelRunner with injected dependencies."""
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
        self.mambaish_config = getattr(model_config, "mambaish_config", None)
        self.is_hybrid_gdn = getattr(model_config, "is_hybrid_gdn", False)
        self.sliding_window_size = getattr(
            model_config.hf_config, "sliding_window", None
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
                        "quant config (kv_cache_quant_algo=%s)",
                        kv_algo,
                    )

        global_server_args_dict_update(server_args)
        initialize_moe_config(server_args)

        self.memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=server_args.enable_memory_saver
        )
        self.load_model()
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
        )
        self._model_forward_accepts_spec_step_idx = self._forward_accepts_kwarg(
            self.model, "spec_step_idx"
        )

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
        out_cache_loc: torch.Tensor,
        req_pool_indices: torch.Tensor | None = None,
        seq_lens: torch.Tensor | None = None,
        extend_prefix_lens: torch.Tensor | None = None,
        captured_hidden_states: torch.Tensor | None = None,
        input_embeds: torch.Tensor | None = None,
        multimodal_context: MultimodalForwardContext | None = None,
        spec_step_idx: int | None = None,
    ) -> LogitsProcessorOutput:
        kwargs = {}
        if req_pool_indices is not None:
            kwargs["req_pool_indices"] = req_pool_indices
        if seq_lens is not None:
            kwargs["seq_lens"] = seq_lens
        if extend_prefix_lens is not None:
            kwargs["extend_prefix_lens"] = extend_prefix_lens
        if not self.is_generation:
            kwargs["get_embedding"] = True
        if captured_hidden_states is not None:
            kwargs["captured_hidden_states"] = captured_hidden_states
        if input_embeds is not None:
            kwargs["input_embeds"] = input_embeds
        if multimodal_context is not None:
            kwargs["multimodal_context"] = multimodal_context
        if spec_step_idx is not None and getattr(
            self, "_model_forward_accepts_spec_step_idx", False
        ):
            kwargs["spec_step_idx"] = spec_step_idx

        return self.model.forward(
            ctx,
            input_ids,
            positions,
            out_cache_loc,
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
        never collides with the engine's own world.
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

            self._weight_update_pg = pg
            self._weight_update_device = device
            logger.info(
                "weight-update group joined: rank=%d world_size=%d device=%s group=%s",
                rank,
                world_size,
                device,
                group_name,
            )
            return True, "weight update group initialized"
        except Exception as e:  # noqa: BLE001 - surface to the control plane
            logger.exception("init_weights_update_group failed")
            return False, str(e)

    def update_weights_from_distributed(self, obj) -> tuple[bool, str]:
        """Receive trainer-broadcast weights over the NCCL group and load them."""
        import torch.distributed as dist

        pg = getattr(self, "_weight_update_pg", None)
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

            self.model.load_weights(_recv())
            torch.cuda.synchronize(device)
            return True, f"updated {len(names)} weights"
        except Exception as e:  # noqa: BLE001 - surface to the control plane
            logger.exception("update_weights_from_distributed failed")
            return False, str(e)

    def destroy_weights_update_group(self, obj) -> tuple[bool, str]:
        """Tear down the trainer weight-update NCCL group joined in ``init``.

        When a training run ends the trainer drops its end of the group, so the
        worker must release its side too -- free the NCCL communicator and the
        torch ``_world`` bookkeeping ``init_weights_update_group`` registered --
        instead of leaking it until engine shutdown. A fresh run then re-inits a
        clean group. Idempotent: tearing down when no group is live is a success
        so a trainer that always calls destroy (e.g. slime) never errors.
        """
        pg = getattr(self, "_weight_update_pg", None)
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
