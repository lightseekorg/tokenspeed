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
        self.page_size = server_args.block_size
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
