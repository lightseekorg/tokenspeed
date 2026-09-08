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

"""Factory helpers for model runners and model executors."""

from __future__ import annotations

from typing import TYPE_CHECKING

import tokenspeed.runtime.layers.attention.backends  # noqa: F401  # trigger register_backend() calls
from tokenspeed.runtime.configs.model_config import ModelConfig
from tokenspeed.runtime.execution.drafter import get_drafter_impl
from tokenspeed.runtime.execution.model_executor import (
    ModelExecutor,
    ModelExecutorConfig,
)
from tokenspeed.runtime.execution.model_runner import ModelRunner
from tokenspeed.runtime.sampling.registry import create_sampling_backend
from tokenspeed.runtime.utils.nvtx import set_nvtx_enabled
from tokenspeed.runtime.utils.server_args import ServerArgs

if TYPE_CHECKING:
    from tokenspeed.runtime.layers.attention.backends.base import AttentionBackend
    from tokenspeed.runtime.layers.attention.kv_cache.base import CachePool


def _eagle_aux_layer_ids(hf_config) -> list[int] | None:
    """Return EAGLE3 capture ids from a draft config, including K3 text config.

    K3 wraps the language configuration in ``text_config``.  Draft exports may
    place ``eagle_config`` either on that text config or on the top-level
    wrapper, so inspect both without falling back to the target's defaults.
    """
    candidates = [hf_config]
    text_config = (
        hf_config.get("text_config")
        if isinstance(hf_config, dict)
        else getattr(hf_config, "text_config", None)
    )
    if text_config is not None:
        candidates.append(text_config)

    for config in candidates:
        if isinstance(config, dict):
            eagle_config = config.get("eagle_config")
            direct_ids = config.get("eagle_aux_hidden_state_layer_ids")
        else:
            eagle_config = getattr(config, "eagle_config", None)
            direct_ids = getattr(config, "eagle_aux_hidden_state_layer_ids", None)
        if isinstance(eagle_config, dict):
            ids = eagle_config.get("eagle_aux_hidden_state_layer_ids")
        elif eagle_config is not None:
            ids = getattr(eagle_config, "eagle_aux_hidden_state_layer_ids", None)
        else:
            ids = direct_ids
        if ids:
            return list(ids)
    return None


def _wire_draft_to_target_model(
    server_args: ServerArgs,
    model_runner: ModelRunner,
    draft_model_runner: ModelRunner,
) -> None:
    """Model-to-model wiring that must happen right after both models load.

    Runs before create_attn_components profiles free memory for the KV-cache
    budget, so weights the draft shares with the target (embed/LM head) are
    released before profiling instead of being double-counted.
    """
    DrafterImpl = get_drafter_impl(
        server_args.speculative_algorithm, draft_model_runner.model
    )
    if DrafterImpl.shares_target_embed_head:
        embed, head = model_runner.model.get_embed_and_head()
        draft_model = draft_model_runner.model
        module_setter = getattr(type(draft_model), "set_embed_and_head_module", None)
        if module_setter is not None:
            lm_head = getattr(model_runner.model, "lm_head", None)
            if lm_head is None:
                raise ValueError(
                    "Draft model requires the target's complete lm_head module."
                )
            module_setter(draft_model, embed, lm_head)
        else:
            draft_model.set_embed_and_head(embed, head)
    if server_args.speculative_algorithm == "EAGLE3" and hasattr(
        model_runner.model, "set_eagle3_layers_to_capture"
    ):
        # capture the layers the draft was trained on, not the default
        aux_layer_ids = server_args.eagle3_layers_to_capture or _eagle_aux_layer_ids(
            draft_model_runner.model_config.hf_config
        )
        model_runner.model.set_eagle3_layers_to_capture(aux_layer_ids)


def create_model_runner(
    server_args: ServerArgs,
    model_config: ModelConfig,
    draft_model_config: ModelConfig | None,
    gpu_id: int,
    global_rank: int,
):
    """Create the main model runner and optional draft model runner."""
    model_runner = ModelRunner(
        model_config=model_config,
        gpu_id=gpu_id,
        server_args=server_args,
        global_rank=global_rank,
    )

    draft_model_runner = None
    if draft_model_config is not None:
        draft_model_runner = ModelRunner(
            model_config=draft_model_config,
            gpu_id=gpu_id,
            server_args=server_args,
            global_rank=global_rank,
            is_draft_worker=True,
        )
        if server_args.speculative_algorithm is not None:
            _wire_draft_to_target_model(server_args, model_runner, draft_model_runner)

    return model_runner, draft_model_runner


def create_model_executor(
    server_args: ServerArgs,
    config: ModelExecutorConfig,
    model_runner: ModelRunner,
    attn_backend: AttentionBackend,
    token_to_kv_pool: CachePool,
    draft_model_runner: ModelRunner | None = None,
    draft_attn_backend: AttentionBackend | None = None,
    draft_token_to_kv_pool: CachePool | None = None,
) -> ModelExecutor:
    """Create the model executor with its sampler configuration."""
    if server_args.enable_nvtx:
        set_nvtx_enabled(True)

    max_bs = config.max_num_seqs // max(config.data_parallel_size, 1)

    max_draft_tokens_per_req = (
        config.spec_num_tokens if config.spec_algo is not None else 1
    )

    sampling_backend = create_sampling_backend(
        server_args,
        max_bs=max_bs,
        max_draft_tokens_per_req=max_draft_tokens_per_req,
        device=config.device,
        max_req_pool_size=config.max_req_pool_size,
        vocab_size=config.vocab_size,
        # Same TP group as LogitsProcessor.
        tp_group=model_runner.mapping.attn.tp_group,
    )

    return ModelExecutor(
        config=config,
        model_runner=model_runner,
        attn_backend=attn_backend,
        token_to_kv_pool=token_to_kv_pool,
        sampling_backend=sampling_backend,
        draft_model_runner=draft_model_runner,
        draft_attn_backend=draft_attn_backend,
        draft_token_to_kv_pool=draft_token_to_kv_pool,
    )
