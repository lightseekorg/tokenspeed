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

from abc import abstractmethod
from contextlib import AbstractContextManager
from typing import TYPE_CHECKING

import torch
from tokenspeed_kernel.ops.sampling import argmax as sampling_argmax

from tokenspeed.runtime.execution.drafter.speculative_sampling import (
    DraftProposalSampler,
)
from tokenspeed.runtime.execution.model_runner import ModelRunner

if TYPE_CHECKING:
    from tokenspeed.runtime.execution.context import ForwardContext
    from tokenspeed.runtime.execution.input_buffer import InputBuffers
    from tokenspeed.runtime.execution.runtime_states import RuntimeStates
    from tokenspeed.runtime.execution.tree_spec import TreeSpec
    from tokenspeed.runtime.layers.attention.backends.base import AttentionBackend
    from tokenspeed.runtime.layers.attention.kv_cache.base import CachePool
    from tokenspeed.runtime.layers.logits_processor import LogitsProcessorOutput
    from tokenspeed.runtime.sampling.backends.base import SamplingBackend


class BaseDrafter:
    # Whether the draft model reuses the target's embedding and LM head
    # weights (set via set_embed_and_head right after both models load, in
    # create_model_runner, so the draft's own copies are dropped before the
    # KV-cache budget is profiled).
    shares_target_embed_head = False

    # True only when ``run`` returns after every cache-writing draft pass has
    # been enqueued on the caller's CUDA stream. CachePD uses that guarantee to
    # publish one final readiness event for the complete speculative chain.
    supports_pd_layerwise_finalization = False

    # Whether this drafter threads a request-token-history view into every
    # draft forward. A draft model whose config requires the history must be
    # served by a drafter that declares this, or fail at startup instead of
    # forwarding without it.
    supports_request_token_history = False

    # Whether this drafter proposes one token per step from its own logits
    # and can therefore sample it from the recorded distribution q
    # (--enable-speculative-sampling). Block drafters propose a whole block
    # greedily and leave this False.
    supports_speculative_sampling = False

    def __init__(
        self,
        spec_num_tokens: int,
        spec_num_steps: int | None = None,
        draft_model_runner: ModelRunner | None = None,
        runtime_states: RuntimeStates | None = None,
        input_buffers: InputBuffers | None = None,
        attn_backend: AttentionBackend | None = None,
        token_to_kv_pool: CachePool | None = None,
        vocab_size: int | None = None,
    ):
        self.spec_num_tokens = spec_num_tokens
        self.spec_num_steps = spec_num_steps
        self.draft_model_runner = draft_model_runner
        self.runtime_states = runtime_states
        self.input_buffers = input_buffers
        # This round's per-group scheduler tables, published by the runner's
        # metadata prep; block drafters re-run the unified refresh with them
        # inside their step loop (write locations themselves come from the
        # draft router: publish_draft_step_locations serves the step window,
        # draft_write_locations_uniform resolves side-write scratch).
        self.round_block_tables = None
        self.attn_backend = attn_backend
        self.token_to_kv_pool = token_to_kv_pool
        self.vocab_size = vocab_size
        # Bound by ``bind_sampling_backend`` under --enable-speculative-sampling;
        # None keeps the argmax proposal.
        self.draft_sampler: DraftProposalSampler | None = None

    def set_cache_pool(self, token_to_kv_pool: CachePool | None) -> None:
        """Take a replacement pool; a drafter that caches views rebuilds them."""
        self.token_to_kv_pool = token_to_kv_pool

    def bind_sampling_backend(self, sampling_backend: SamplingBackend) -> None:
        """Take the verifier's per-request sampling pools for sampled drafting.

        Called once by ``ModelExecutor`` after construction. A no-op unless
        the executor allocated ``RuntimeStates.draft_probs``
        (--enable-speculative-sampling); then the drafter must be a chain
        drafter that proposes one token per step, and its proposals are
        drawn from ``q`` and recorded there (``sample_draft_step``). Sampling
        needs the whole distribution, so the draft model's logits processor
        is switched off its fused (TP-sharded) argmax here, before the first
        draft forward.

        Args:
            sampling_backend: The executor's verifier, owner of the
                pool-indexed temperature / top-k / seed scalars.

        Raises:
            ValueError: The flag is on but this drafter proposes greedily.
        """
        if self.runtime_states is None or self.runtime_states.draft_probs is None:
            return
        if not self.supports_speculative_sampling:
            raise ValueError(
                "--enable-speculative-sampling needs a chain drafter that samples "
                f"one token per step from its own distribution; "
                f"{type(self).__name__} proposes a whole block greedily"
            )
        self.draft_sampler = DraftProposalSampler(
            pools=sampling_backend.speculative_sampling_pools(),
            runtime_states=self.runtime_states,
            input_buffers=self.input_buffers,
            spec_num_tokens=self.spec_num_tokens,
            vocab_map=self.draft_vocab_map(),
            device=self.runtime_states.draft_probs.device,
        )
        self.draft_model_runner.model.logits_processor.require_full_vocab_logits()

    def draft_vocab_map(self) -> torch.Tensor | None:
        """``[V_draft]`` int64 full-vocab id per draft logit column, or None
        when the draft head spans the target vocabulary. Eagle3 hot-token
        heads override this; ``sample_draft_step`` returns ids in the draft
        vocab either way and the caller maps them."""
        return None

    def sample_draft_step(
        self,
        logits_output: LogitsProcessorOutput,
        *,
        step: int,
    ) -> torch.Tensor:
        """One draft step's proposed token ids, ``[bs]`` in the draft vocab.

        Without draft-prob sampling this is the logits processor's fused
        argmax when it ran, else the canonical argmax of the logits. With
        it, the token is sampled from ``q`` at the request's temperature
        (greedy rows keep the argmax) and ``q`` is recorded for verify. The
        row count is the logits' (the padded graph batch under replay).

        Args:
            logits_output: The step's draft forward output, one row per
                request.
            step: Draft step index (verify candidate column ``step + 1``).
        """
        sampler = self.draft_sampler
        if sampler is None:
            if (
                self.runtime_states is not None
                and self.runtime_states.draft_probs is not None
            ):
                raise RuntimeError(
                    f"{type(self).__name__} records no draft distributions: "
                    "bind_sampling_backend was not called"
                )
            if logits_output.next_token_ids is not None:
                return logits_output.next_token_ids
            return sampling_argmax(logits_output.next_token_logits)
        return sampler.propose(logits_output.next_token_logits, step=step)

    def wire_target(self, target_model: torch.nn.Module) -> None:
        """Wire this drafter to the loaded target model.

        Called once by ``ModelExecutor`` right after the drafter is
        constructed. Subclasses bind execution resources such as target weights
        and output heads here. Capture configuration belongs to model setup
        before drafter construction; this method must not change it. The
        default drafter needs nothing from the target.

        Args:
            target_model: The target ``torch.nn.Module`` the drafter
                speculates for.
        """

    def prepare_target_forward(self, ctx: ForwardContext) -> None:
        """Hook before the round's target forward.

        Called by ``ModelExecutor`` with the context the target is about to
        run under. A drafter that wants work done while the target runs
        attaches it to ``ctx`` here (DFLASH attaches its incremental capture
        sink, ``ctx.target_capture_sink``); the context is per forward, so
        nothing attached outlives the round. The default needs nothing.

        Args:
            ctx: The target forward's context.
        """

    def on_target_weights_updated(self) -> None:
        """Refresh draft-side state derived from shared target weights.

        Called after an in-place target weight update completes and before the
        device thread accepts another forward. Most drafters do not cache
        derived target weights and therefore need no action.
        """

    def idle_forward_global_num_tokens(
        self, global_num_tokens: list[int], global_bs: list[int]
    ) -> list[list[int]]:
        """The IDLE draft forwards an idle attention-DP rank runs per round,
        as the per-rank token counts each one reports: entry ``i`` sizes the
        forward with ``spec_step_idx=i``, and the list length is the number
        of draft forwards the active ranks run.

        The idle rank runs no rows of its own; it mirrors the collective
        sizing of the ranks that do, so the list follows the drafter's step
        loop. The default is the Eagle chain: step 0 runs the target's rows
        (its verify window per decode request), steps 1+ one row per request
        (``global_bs``). Drafters with another loop override this — block
        drafters run one forward, multi-depth MTP the target's rows at every
        depth.

        Args:
            global_num_tokens: The round's per-rank target token counts.
            global_bs: The round's per-rank decode request counts.

        Returns:
            One ``global_num_tokens`` per IDLE draft forward, in step order.
        """
        steps = int(self.spec_num_steps or 0)
        if steps == 0:
            return []
        return [global_num_tokens] + [global_bs] * (steps - 1)

    @property
    def captures_prefill_graph(self) -> bool:
        """Whether ``capture_prefill_graph`` records one, for the projection."""
        return False

    def release_prefill_graph(self) -> None:
        """Drop a captured draft prefill graph. Drafters without one need no action."""

    def capture_prefill_graph(
        self, stream: torch.cuda.Stream, observer: AbstractContextManager[None]
    ) -> None:
        """Capture draft prefill work after target capture, when prefill graphs
        are enabled. Drafters without a separate prefill graph need no action.
        """

    def bind_tree(self, tree_spec: TreeSpec) -> None:
        """Draft trees (--speculative-eagle-topk > 1); only EAGLE-style drafters expand lanes."""
        raise NotImplementedError(
            f"{type(self).__name__} cannot draft trees (--speculative-eagle-topk > 1); "
            "tree drafting needs an EAGLE-style drafter (EAGLE3, or MTP served by Eagle)"
        )

    @abstractmethod
    def run(
        self,
        base_ctx: ForwardContext,
        logits_output: LogitsProcessorOutput,
        output_tokens: torch.Tensor,
        accept_lengths: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def draft(self, *args, **kwargs) -> torch.Tensor | None:
        raise NotImplementedError
