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
from tokenspeed.runtime.layers.attention.configs.base import is_block_drafter
from tokenspeed.runtime.models.target_capture import TargetCaptureConfigurator
from tokenspeed.runtime.sampling.registry import create_sampling_backend
from tokenspeed.runtime.utils.nvtx import set_nvtx_enabled
from tokenspeed.runtime.utils.server_args import ServerArgs
from tokenspeed.runtime.utils.startup_timing import startup_phase

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


def _share_target_embed_and_head(model_runner: ModelRunner, draft_model) -> None:
    """Alias the target's embedding/LM head into an EAGLE-style draft.

    Binds what the target reports. Off the pipeline that is both sides, and
    the draft's copies are dropped before the KV-cache budget is profiled. On
    a pipeline only the last stage runs the drafter, so stages before it bind
    nothing; the last stage holds the target head but not the embedding (that
    lives on the first stage), so it shares the head alone and the draft must
    keep the embedding its checkpoint ships -- checked after the call, since a
    draft that aliases ``None`` into its embedding fails only at the first
    forward.
    """
    mapping = model_runner.mapping
    if mapping.has_pp and not mapping.is_last_pp_rank:
        return
    _check_shared_head_layout(model_runner.model, draft_model)
    embed, head = model_runner.model.get_embed_and_head()
    if embed is None and not mapping.has_pp:
        raise ValueError("Draft model requires the target's embedding weight.")
    module_setter = getattr(type(draft_model), "set_embed_and_head_module", None)
    if module_setter is not None:
        lm_head = getattr(model_runner.model, "lm_head", None)
        if lm_head is None:
            raise ValueError(
                "Draft model requires the target's complete lm_head module."
            )
        module_setter(draft_model, embed, lm_head)
    else:
        if head is None:
            raise ValueError("Draft model requires the target's lm_head weight.")
        draft_model.set_embed_and_head(embed, head)
    if embed is None:
        _require_draft_embedding(draft_model)


def _require_draft_embedding(draft_model) -> None:
    """Fail loudly when a draft bound ``embed=None`` and kept no embedding.

    The draft reports the weights it drafts with through
    ``get_embed_and_head``; a pipeline-capable draft keeps its checkpoint
    ``embed_tokens`` when the target shares none.
    """
    draft_cls = type(draft_model)
    reporter = getattr(draft_cls, "get_embed_and_head", None)
    draft_embed = reporter(draft_model)[0] if reporter is not None else None
    if draft_embed is None:
        raise ValueError(
            f"{draft_cls.__name__} has no embedding after the target shared its "
            "head alone: on a pipeline's last stage the target embedding lives "
            "on the first stage, so the draft must keep the embed_tokens its "
            "checkpoint ships when embed is None and report it from "
            "get_embed_and_head."
        )


def _check_shared_head_layout(target_model, draft_model) -> None:
    """A draft sharing the target's LM head must shard it the same way.

    The target's head follows ``mapping.lm_head`` (vocab-sharded over
    attention-DP ranks under ``--lm-head-tp-size``); a draft still building
    its head on the attention TP group would take the shard as a whole-vocab
    weight and sample garbage silently. Compare the logits processors, which
    carry the layout for both.
    """
    target = target_model.logits_processor
    draft = draft_model.logits_processor
    if (
        target.tp_group == draft.tp_group
        and target.tp_size == draft.tp_size
        and target.dp_lm_head_tp == draft.dp_lm_head_tp
    ):
        return
    raise ValueError(
        f"{type(draft_model).__name__} builds its LM head over "
        f"{draft.tp_size} rank(s) {draft.tp_group} but the target's head is "
        f"sharded over {target.tp_size} rank(s) {target.tp_group} "
        "(--lm-head-tp-size); this drafter cannot share a head in that layout"
    )


def configure_draft_target(
    server_args: ServerArgs,
    model_runner: ModelRunner,
    draft_model_runner: ModelRunner,
) -> None:
    """Model-to-model wiring that must happen right after both models load.

    Runs before create_attn_components profiles free memory for the KV-cache
    budget, so weights the draft shares with the target (embed/LM head) are
    released before profiling instead of being double-counted.
    """
    draft_model = draft_model_runner.model
    DrafterImpl = get_drafter_impl(server_args.speculative_algorithm, draft_model)
    if (
        draft_model_runner.model_config.requires_request_token_history
        and not DrafterImpl.supports_request_token_history
    ):
        raise NotImplementedError(
            f"draft model requires request-token history, but drafter "
            f"{DrafterImpl.__name__} does not thread it through its forwards"
        )
    if server_args.speculative_algorithm in ("DFLASH", "DSPARK"):
        if not isinstance(draft_model, TargetCaptureConfigurator):
            raise TypeError(
                f"{type(draft_model).__name__} must implement TargetCaptureConfigurator "
                f"for {server_args.speculative_algorithm}."
            )
        draft_model.configure_target(
            model_runner.model, model_runner.model_config.hf_text_config
        )
    if DrafterImpl.shares_target_embed_head:
        _share_target_embed_and_head(model_runner, draft_model)
    if server_args.speculative_algorithm == "EAGLE3" and hasattr(
        model_runner.model, "set_eagle3_layers_to_capture"
    ):
        # capture the layers the draft was trained on, not the default
        aux_layer_ids = server_args.eagle3_layers_to_capture or _eagle_aux_layer_ids(
            draft_model_runner.model_config.hf_config
        )
        model_runner.model.set_eagle3_layers_to_capture(aux_layer_ids)


def pipeline_stage_builds_draft(spec_algo: str | None, mapping) -> bool:
    """Whether this rank builds the draft model at all.

    Off the pipeline, and on its last stage, the draft is built: only the last
    stage samples, so it alone drafts. A stage before the last still builds a
    block drafter's model (DSPARK), which produces its draft context on every
    stage from the target taps that stage owns. An EAGLE-style draft (MTP)
    reads only the last stage's final hidden states, so the other stages skip
    its construction, weights and memory: nothing is wired there and
    ``ModelExecutor`` takes ``draft_model_runner=None``. The skip leaves no
    collective the other stages would have to join: the draft's weight loading
    synchronizes within its stage (``checkpoint_load_group``), the cache
    budget's world all-reduce runs with or without a draft, and the layer
    communicators a draft reuses are the target's, scoped to the stage.
    """
    if not mapping.has_pp or mapping.is_last_pp_rank:
        return True
    return is_block_drafter(spec_algo, is_draft=True)


def create_model_runner(
    server_args: ServerArgs,
    model_config: ModelConfig,
    draft_model_config: ModelConfig | None,
    gpu_id: int,
    global_rank: int,
):
    """Create the main model runner and optional draft model runner.

    The draft runner is None without a draft config and on the pipeline
    stages that neither draft nor produce draft context
    (``pipeline_stage_builds_draft``).
    """
    mapping = server_args.mapping
    with startup_phase("weights.target", rank=global_rank):
        model_runner = ModelRunner(
            model_config=model_config,
            gpu_id=gpu_id,
            server_args=server_args,
            global_rank=global_rank,
            # Every stage reads the whole checkpoint and keeps its layers, so
            # a distributed loader may shard the reads over the world.
            checkpoint_load_group=None,
        )

    draft_model_runner = None
    if draft_model_config is not None and pipeline_stage_builds_draft(
        server_args.speculative_algorithm, mapping
    ):
        with startup_phase("weights.draft", rank=global_rank):
            draft_model_runner = ModelRunner(
                model_config=draft_model_config,
                gpu_id=gpu_id,
                server_args=server_args,
                global_rank=global_rank,
                # Other stages may build no draft: a distributed loader must
                # synchronize within this stage, never the world.
                checkpoint_load_group=(
                    mapping.attn.world_group if mapping.has_pp else None
                ),
                is_draft_worker=True,
            )
        if server_args.speculative_algorithm is not None:
            configure_draft_target(server_args, model_runner, draft_model_runner)

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
