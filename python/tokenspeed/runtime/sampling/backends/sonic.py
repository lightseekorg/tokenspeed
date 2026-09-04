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

"""Sonic Sampler backend — fused single-step sampling + chain verification.

One fused Triton pipeline (`fused_singular`, two kernels, for sample;
`fused_multistep`, three, for spec-decode verify) handles a mixed batch (greedy + temperature/top_k/top_p/min_p
+ penalties + grammar) selected per row by an `Indicator` bitfield — no separate
greedy path or per-mode graph.

Buffers are either **slot-indexed** per-request state (flags, scalars, noise,
grammar, counts, bias) in one `SamplingBuffers` sized to ``max_req_pool_size + 1``
and gathered at ``slot_mapping[row]`` (= ``req_pool_indices``), written on the
device once in ``_reset_slot``; or **batch-indexed**
I/O sized to max batch (scratch/values/indices owned by ``TwoStageTiling``).
Noise is drawn inside sonic's kernels (the copies under
``tokenspeed_kernel.thirdparty.sonic``) from Philox keyed by the slot's seed
and its cache length (``sampling_info.valid_cache_lengths``), at the
candidates the kernels actually read: no noise plane, no per-step refresh,
draws that depend only on the request, and every TP rank drawing the same.

The runtime's unified sampling rule routes every decode row through
``verify()`` — non-speculative serving is its ``N == 1`` case (a one-column
candidate window with nothing to accept). ``fused_multistep`` needs a lookahead
of at least one, so that case resolves through ``fused_singular``: the same
kernel ``sample()`` uses, reading the same slot state and noise, so the two
paths are bitwise identical (pinned by ``test_sonic_backend.py``).

Not yet supported: DP-sampling, top-k output logprobs, EAGLE3 ``d2t`` cross-vocab.

Requires sonic-sampler 1.0.0 for its buffers, indicators, dispatch tables and
top-k sub-kernels; the fused kernels come from ``tokenspeed_kernel``. The
package is optional: the registry only registers this backend when it is
importable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from tokenspeed_kernel.ops.sampling.sonic import (
    MAX_K,
    SamplingBuffers,
    ScopedIndicators,
    ThreeStageWarpConfig,
    available,
    fused_multistep,
    fused_singular,
    make_tiling,
)
from tokenspeed_kernel.platform import pdl_enabled
from typing_extensions import override

from tokenspeed.runtime.sampling.backends.base import (
    SamplingBackend,
    SamplingBackendConfig,
)
from tokenspeed.runtime.sampling.registry import register_backend
from tokenspeed.runtime.sampling.sampling_params import _SAMPLING_EPS
from tokenspeed.runtime.sampling.utils import gather_token_logprobs_torch
from tokenspeed.runtime.utils.nvtx import nvtx_range

if TYPE_CHECKING:

    from tokenspeed_kernel.ops.sampling.sonic import (
        Selection,
        TopKStrategy,
        TwoStageWarpConfig,
        Verification,
    )

    from tokenspeed.runtime.layers.logits_processor import LogitsProcessorOutput
    from tokenspeed.runtime.sampling.sampling_batch_info import SamplingBatchInfo
    from tokenspeed.runtime.sampling.sampling_params import SamplingParams


def _sanitize(sp: SamplingParams) -> tuple[float, int, float, float]:
    """Sanitize a request's ``(temperature, top_k, top_p, min_p)`` for sonic.

    ``allocate()`` derives the ``Indicator`` bits from the same values it writes
    into the buffers (``Indicator.from_params`` tests them exactly), so
    sanitizing here keeps every bit agreeing with what the kernel reads:
    epsilon-neutral temperature stays ``1.0`` (bit off — skips a wasted
    ``logits / t`` vocab pass), and ``top_k`` is capped at MAX_K so the ``-1``
    "disabled" sentinel lands on the bounded top-MAX_K path (bit off). The
    greedy early return also pins temperature to 1.0, so the slot's indicator
    is exactly ``GREEDY`` and the kernel skips the temperature pass. ``min_p =
    1.0`` is remapped to 0.999 because ``from_params`` tests the open interval
    (1.0 would drop the MIN_P bit); the bf16 buffer rounds it back to 1.0,
    whose pivot = max keeps only max-probability tokens, as requested."""

    if sp.top_k == 1:  # TokenSpeed's greedy encoding

        return 1.0, 1, 1.0, 0.0

    temperature = 1.0

    if abs(sp.temperature - 1.0) > _SAMPLING_EPS:

        temperature = float(sp.temperature)

    min_p = float(sp.min_p)

    if min_p >= 1.0:

        min_p = 0.999

    return temperature, min(int(sp.top_k), MAX_K), float(sp.top_p), min_p


class SonicSamplingBackend(SamplingBackend):
    """Fused sampling + chain verification via sonic-sampler. Mixed greedy /
    temperature / top_k / top_p / min_p in one pipeline; ``top_k`` bounded by
    ``MAX_K = 128``. Per-request state is slot-indexed by ``req_pool_indices``,
    written once at admission.

    top_k semantics: finite ``top_k >= 128`` never reaches the backend —
    ``SamplingParams.verify()`` rejects it at request time (same limit as the
    flashinfer fused kernel). ``top_k = -1`` (full vocab) arrives as the
    ``_TOP_K_DISABLED`` sentinel and is realized as **bounded top-128
    truncation**: the kernel's bit-packed reduction keeps only the top-128
    candidates per row, so mass beyond them is dropped (negligible for the
    peaked distributions a model emits at temperature <= 1, but significant on
    flat ones or at high temperature). flashinfer's ``-1``
    samples the full vocab; this is the one distribution difference the
    backend chooses. Inherited from sonic's 16-bit packed ranking: candidates
    are compared at bf16 resolution (temperature, penalties and logit bias are
    applied in bf16, and the scaled logits are re-rounded), so where the fp32
    backends separate near-tied tokens sonic's greedy pick is only guaranteed
    to lie within one bf16 ulp of the maximum (pinned by
    ``test_sonic_backend.py``); temperature/top_p/min_p are stored in bf16 too. Under
    TP, decode counts for penalties accumulate in-kernel from each rank's own
    draw, before the rank-0 broadcast that decides the emitted token.

    Requires bf16 logits (sonic's kernels compile for that dtype only) and
    ``vocab_size >= 512`` (its top-k reduction's minimum)."""

    _HAS_POOL_STATE = True
    _SUPPORTS_DP_VERIFY = False

    @override
    def __init__(self, config: SamplingBackendConfig) -> None:

        super().__init__(config)

        if config.vocab_size < 512:

            raise ValueError(
                f"SonicSamplingBackend requires config.vocab_size >= 512, got "
                f"{config.vocab_size} (sonic's top-k reduction minimum)"
            )

        if config.device is None:

            raise ValueError("SonicSamplingBackend requires config.device")

        self.vocab_size = int(config.vocab_size)
        self.max_bs = int(config.max_bs)
        self.device = config.device

        # n_max = γ+1 under spec-decode, else 1; drives the verify timestep dim.
        self.n_max = max(1, int(config.max_draft_tokens_per_req))
        self.gamma = self.n_max - 1
        self.spec = self.n_max > 1

        # Slot-indexed state, +1 for the padding sentinel; timesteps=n_max serves sample and verify.
        self.pool_rows = config.max_req_pool_size + 1

        major, minor = torch.cuda.get_device_capability(self.device)
        self._dispatch: dict[
            tuple[int, int | None],
            tuple[TwoStageWarpConfig | ThreeStageWarpConfig, TopKStrategy, int],
        ] = {}

        with torch.device(self.device):

            self.tiling, self.values, self.indices = make_tiling(
                arch=major * 10 + minor,
                vocab_size=self.vocab_size,
                batch_size=self.max_bs,
                lookahead=self.gamma,
                unpacked_buffers=self.spec,
            )

            # Batch-indexed working space, persistent (captured kernel never allocs).
            self.out_tok = torch.empty((self.max_bs, 1), dtype=torch.int32)
            self.ones_buf = torch.ones((self.max_bs,), dtype=torch.int32)

            if self.spec:

                # verify() drafted-tokens I/O; see ``_fused_multistep``.
                self.v_drafted = torch.empty(self.max_bs, self.n_max, dtype=torch.int32)
                self.accept_buf = torch.empty((self.max_bs,), dtype=torch.int32)

            # Per-slot Philox seeds; never-admitted slots (0, padding sentinel) keep the backend's.
            self.seeds = torch.full(
                (self.pool_rows,), config.random_seed, dtype=torch.int64
            )

        self.buffers = SamplingBuffers.default(
            size=self.pool_rows,
            timesteps=self.n_max,
            vocab_size=self.vocab_size,
            device=self.device,
        )

        # xgrammar int32 and sonic uint32 share bit semantics; CUDA has no uint32 index_put.
        self.grammar = self.buffers.grammar.view(torch.int32)

        # ``from_params`` derives the GRAMMAR bit from a host bitmask's non-None-ness only.
        self.grammar_placeholder = torch.zeros((0,), dtype=torch.int32, device="cpu")

        # The slot-indexed buffers every launch gathers at ``slot_mapping[row]``.
        buffers = self.buffers
        counts = buffers.repetition.counts
        pen = buffers.repetition
        self._slot_state = dict(
            indicators=buffers.flags.target,
            temperature=buffers.temperature,
            top_k=buffers.top_k,
            top_p=buffers.top_p,
            min_p=buffers.min_p,
            grammar=buffers.grammar,
            context_counts=counts.context,
            decode_counts=counts.decode,
            repetition_penalties=pen.multiplicative,
            frequency_penalties=pen.frequency,
            presence_penalties=pen.presence,
            logit_bias=buffers.bias,
        )

        self._warmup()

    # ------------------------------------------------------------------ #
    # JIT warm-up
    # ------------------------------------------------------------------ #
    def _warmup(self) -> None:
        """Launch every batch size once on dummy logits gathering slot 0.

        Graph capture only compiles the decode row counts it captures; the
        eager paths (prefill rows of a mixed round, uncaptured batch sizes)
        would otherwise hit new kernel variants at runtime, and each Triton
        compile stalls the whole server for seconds. Triton caches by tuned
        config and integer specialization of ``batch_size``, so most launches
        here are cache hits."""

        with torch.device(self.device):

            logits = torch.zeros(
                (self.max_bs * self.n_max, self.vocab_size), dtype=torch.bfloat16
            )
            slot = torch.zeros((self.max_bs,), dtype=torch.int64)
            offsets = torch.zeros((self.pool_rows,), dtype=torch.int32)
            cand = torch.zeros((self.max_bs, self.n_max), dtype=torch.int32)

        for bs in range(1, self.max_bs + 1):

            self._fused_singular(logits[:bs], slot[:bs], offsets)

            if self.spec:

                self._fused_multistep(
                    logits[: bs * self.n_max], slot[:bs], offsets, cand[:bs]
                )

        # The warm-up gathered slot 0 with counting on; leave it clean.
        self.reset_capture_state()

    # ------------------------------------------------------------------ #
    # Per-slot admission state (flip detection, via the base class)
    # ------------------------------------------------------------------ #
    @override
    def _reset_slot(self, pool_idx: int, sp: SamplingParams) -> None:
        """Write a newly-assigned slot's per-request state once (called by the
        base ``prepare_step`` on a slot's owning-rid flip), straight into
        sonic's device buffers like the sibling backends. sonic's own
        ``allocate`` stages everything through host pinned relays with
        unfenced non-blocking copies (a re-admitted slot could overwrite a
        still-queued copy's source), histograms an always-empty prompt through
        a vocab-wide pinned row and draws a noise plane nothing reads.

        ``ScopedIndicators.from_params`` derives the ``Indicator`` bits from
        the same sanitized values written here, so every bit agrees with what
        the kernel reads; the GRAMMAR bit comes from ``sp.has_grammar`` (the
        masks arrive per step through ``_scatter_grammar``). ``greedy_drafts``:
        drafts are point draws (chain EAGLE, topk=1), so stochastic rows verify
        under sonic's masked-rejection regime."""

        temperature, top_k, top_p, min_p = _sanitize(sp)

        bias: dict[int, float] | None = None

        if sp.logit_bias:

            bias = {int(t): float(v) for t, v in sp.logit_bias.items()}

        scoped = ScopedIndicators.from_params(
            size=1,
            timesteps=self.n_max,
            multiplicative=[float(sp.repetition_penalty)],
            frequency=[float(sp.frequency_penalty)],
            presence=[float(sp.presence_penalty)],
            biases=[bias],
            temperature=[temperature],
            top_k=[top_k],
            top_p=[top_p],
            min_p=[min_p],
            top_logprobs=[0],
            bitmasks=[self.grammar_placeholder if sp.has_grammar else None],
            greedy_drafts=True,
        )

        buffers = self.buffers
        pen = buffers.repetition

        buffers.flags.indicators.update(scoped, [pool_idx])
        buffers.flags.packed[pool_idx, 0].fill_(int(scoped.target[0]))
        buffers.flags.packed[pool_idx, 1].fill_(int(scoped.draft[0]))

        buffers.temperature[pool_idx].fill_(temperature)
        buffers.top_k[pool_idx].fill_(top_k)
        buffers.top_p[pool_idx].fill_(top_p)
        buffers.min_p[pool_idx].fill_(min_p)

        pen.multiplicative[pool_idx].fill_(float(sp.repetition_penalty))
        pen.frequency[pool_idx].fill_(float(sp.frequency_penalty))
        pen.presence[pool_idx].fill_(float(sp.presence_penalty))
        pen.counts.decode[pool_idx].zero_()

        row = buffers.bias[pool_idx]
        row.zero_()

        if bias:

            row[list(bias)] = torch.tensor(
                list(bias.values()), dtype=torch.bfloat16, device=self.device
            )

        self.grammar[pool_idx].fill_(-1)
        self.seeds[pool_idx].fill_(int(sp.seed))

    @override
    def reset_capture_state(self) -> None:
        """Warm-up routes all rows to slot 0 and accumulates its decode counts;
        zero them so the captured graph reads a clean baseline."""

        self.buffers.repetition.counts.decode[0].zero_()

    def _scatter_grammar(
        self,
        vocab_mask: torch.Tensor,
        slot_mapping: torch.Tensor,
        bs: int,
        n: int,
    ) -> None:
        """Scatter xgrammar's row-indexed ``[bs*n, W]`` bitmask (row ``i*n+j`` =
        request ``i``, position ``j``) into the slot-indexed ``grammar`` buffer
        ``[pool_rows, n_max, W]``. xgrammar int32 and sonic uint32 share bit
        semantics (SET = allowed; token ``t`` -> word ``t//32``, bit ``t%32``), so
        it's a reinterpret copy through an int32 view (CUDA has no uint32
        index_put). Captured/graph-safe (same pattern as the draft-probs scatter).

        Mixed prefill+decode spec rounds slice the row-indexed mask per request,
        so both halves arrive misaligned (the runtime's mask slicing is the fix).
        Only the verify half is detectable from shapes (the prefill half has
        ``bs`` rows either way), and it raises in the same step, before the
        mis-scattered rows are read. That check also fires on mixed rounds with
        no grammar request when a grammar backend is configured (all-ones
        mask). The plane is slot-resident: a GRAMMAR-bit slot must receive a
        mask every step, which ``bind_grammar_mask_buf`` guarantees whenever a
        grammar backend is on; admission resets the slot's rows to all-ones."""

        w = self.grammar.shape[-1]

        if vocab_mask.shape[-1] != w:

            raise RuntimeError(
                f"grammar bitmask width {vocab_mask.shape[-1]} != sonic grammar "
                f"buffer width {w} (vocab mismatch: backend vocab vs config "
                f"vocab_size={self.vocab_size})"
            )

        # Row-count guard for the mixed-round misalignment described above.
        if vocab_mask.shape[0] != bs * n:

            raise RuntimeError(
                f"grammar bitmask rows {vocab_mask.shape[0]} != bs*n = {bs * n}: "
                "vocab_mask is misaligned with this sub-batch (mixed "
                "prefill+decode spec batches slice the row-indexed mask by "
                "request). Grammar + spec decode requires an aligned mask per "
                "sub-batch."
            )

        self.grammar[slot_mapping, :n] = vocab_mask.view(bs, n, w)

    # ------------------------------------------------------------------ #
    # Tuned dispatch
    # ------------------------------------------------------------------ #
    def _tuning(
        self, rows: int, gamma: int | None
    ) -> tuple[
        TwoStageWarpConfig | ThreeStageWarpConfig | None, TopKStrategy | None, int
    ]:
        """The tuned ``(warp_config, strategy, block_n)`` for this row count via
        ``TwoStageTiling.tuning`` (sample: ``bs``; verify: ``bs * (γ+1)``, like
        the high-level interfaces), widened to a three-stage config for verify.
        Off the packaged table the tiling returns sonic's defaults. A pure
        function of ``(rows, gamma)``; ``_warmup`` fills the table."""

        key = (rows, gamma)

        if key not in self._dispatch:

            config, strategy, block_n = self.tiling.tuning(rows)

            if gamma is not None:

                config = ThreeStageWarpConfig.from_config(config=config, gamma=gamma)

            self._dispatch[key] = (config, strategy, block_n)

        return self._dispatch[key]

    def _check_logits(self, logits: torch.Tensor) -> None:

        if logits.dtype != torch.bfloat16 or logits.shape[-1] != self.vocab_size:

            raise ValueError(
                f"SonicSamplingBackend requires bf16 logits of width {self.vocab_size} "
                f"(sonic's kernels compile for bf16 only), got {logits.dtype} "
                f"x {logits.shape[-1]}"
            )

    # ------------------------------------------------------------------ #
    # Sampling (inside the captured graph)
    # ------------------------------------------------------------------ #
    def _fused_singular(
        self, logits: torch.Tensor, slot_mapping: torch.Tensor, offsets: torch.Tensor
    ) -> Selection:

        self._check_logits(logits)

        bs = logits.shape[0]
        warp_config, strategy, block_n = self._tuning(bs, None)

        return fused_singular(
            logits=logits,
            **self._slot_state,
            scratchpad=self.tiling.scratchpad[:bs],
            output_tokens=self.out_tok[:bs],
            slot_mapping=slot_mapping,
            enable_pdl=pdl_enabled(),
            is_prefill=False,
            update_counts=True,  # accumulate decode counts for penalties
            return_logprobs=False,
            block_n=block_n,
            topk_strategy=strategy,
            warp_config=warp_config,
            noise_seeds=self.seeds,
            noise_offsets=offsets,
            noise_steps=self.n_max,
        )

    def _sample_rows(
        self,
        logits_output: LogitsProcessorOutput,
        sampling_info: SamplingBatchInfo,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One fused single-step draw per row: the body shared by ``sample()``
        and the ``N == 1`` (non-speculative decode) case of ``verify()``."""

        logits = logits_output.next_token_logits

        bs = logits.shape[0]

        # Padded rows carry slot 0 and write it (grammar, counts); the runtime never allocates it.
        slot_mapping = sampling_info.req_pool_indices[:bs]

        # Native grammar: scatter the bitmask; the kernel masks in-fused.
        if sampling_info.vocab_mask is not None:

            self._scatter_grammar(sampling_info.vocab_mask, slot_mapping, bs, 1)

        sel = self._fused_singular(
            logits, slot_mapping, sampling_info.valid_cache_lengths
        )
        sampled = sel.tokens.view(-1)

        # TP-rank sync (rank 0 wins): gathered logits are not bit-identical across ranks.
        self.maybe_broadcast(sampled)

        self._write_logprob_outputs(logits_output, logits, sampling_info, sampled)

        return sampled, self.ones_buf[:bs]

    def _write_logprob_outputs(
        self,
        logits_output: LogitsProcessorOutput,
        logits: torch.Tensor,
        sampling_info: SamplingBatchInfo,
        tokens: torch.Tensor,
    ) -> None:
        """Selected-token logprobs over the raw (grammar-masked) distribution,
        like flashinfer. Native grammar doesn't touch ``logits``, so the mask is
        applied to a copy here; only paid when logprobs are enabled (off by
        default)."""

        if not self.config.enable_output_logprobs:

            return

        if sampling_info.vocab_mask is not None:

            logits = logits.clone()
            sampling_info.apply_vocab_mask(
                logits=logits, vocab_mask=sampling_info.vocab_mask
            )

        logits_output.next_token_logprobs = gather_token_logprobs_torch(logits, tokens)

    @override
    @nvtx_range("sampling:sample", color="yellow")
    def sample(
        self,
        logits_output: LogitsProcessorOutput,
        sampling_info: SamplingBatchInfo,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        return self._sample_rows(logits_output, sampling_info)

    def _fused_multistep(
        self,
        logits: torch.Tensor,
        slot_mapping: torch.Tensor,
        offsets: torch.Tensor,
        candidates: torch.Tensor,
    ) -> Verification:
        """One ``fused_multistep`` launch over ``[bs * n, V]`` logits.

        ``drafted[:, :γ]`` holds the γ proposals and is passed as
        ``output_tokens`` too: the kernel stores only the corrected token at the
        rejection offset (the rest of the row must already be the drafts), so
        this keeps sonic's in-place semantics with a stable captured pointer
        (the default would clone-allocate). The trailing bonus column is
        output-only: the kernel masks it out of the draft load and
        short-circuits the residual correction there. No draft probabilities:
        sonic's masked-rejection regime natively verifies point-draw drafts
        (accept iff u <= p(draft), residual = p with the drafted token zeroed)."""

        self._check_logits(logits)

        bs, n = candidates.shape
        gamma = n - 1

        drafted = self.v_drafted[:bs]
        drafted[:, :gamma].copy_(candidates[:, 1:])
        drafted[:, gamma:].zero_()

        warp_config, strategy, block_n = self._tuning(bs * n, gamma)

        return fused_multistep(
            logits=logits,
            **self._slot_state,
            drafted_tokens=drafted,
            lookahead=gamma,
            block_n=block_n,
            scratchpad=self.tiling.scratchpad[: bs * n],
            values=self.values[: bs * n],
            indices=self.indices[: bs * n],
            output_tokens=drafted,
            slot_mapping=slot_mapping,
            enable_pdl=pdl_enabled(),
            update_counts=True,  # accumulate decode counts for penalties
            return_logprobs=False,
            topk_strategy=strategy,
            warp_config=warp_config,
            noise_seeds=self.seeds,
            noise_offsets=offsets,
            noise_steps=self.n_max,
        )

    @override
    @nvtx_range("sampling:verify", color="yellow")
    def verify(
        self,
        logits_output: LogitsProcessorOutput,
        sampling_info: SamplingBatchInfo,
        candidates: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Chain speculative verification via ``fused_multistep``.

        ``candidates`` is ``[bs, n]`` (n = γ+1): column 0 is the last verified
        token, 1..γ the drafts. sonic compares its per-position selection against
        ``drafted[s]`` at the same index, so we pass ``drafted = candidates[:, 1:]``
        and read back ``Verification.tokens`` (the accepted prefix + next token)
        with ``accept_length = offsets + 1``. Greedy and stochastic rows resolve in
        one pipeline via per-row indicators; the drafts are point draws, handled
        natively by sonic's masked-rejection regime (``draft_probabilities=None``:
        accept iff u <= p(draft), residual = p with the drafted token zeroed).

        ``n == 1`` is non-speculative decode (the runtime's unified sampling
        rule): nothing to accept, one token to draw, ``accept_length == 1`` —
        exactly ``sample()``, so it takes the ``fused_singular`` path. Widths
        other than 1 and ``n_max`` are refused (``v_drafted`` is ``[max_bs,
        n_max]``)."""

        bs, n = candidates.shape

        if n == 1:

            return self._sample_rows(logits_output, sampling_info)

        if n != self.n_max:

            raise RuntimeError(
                f"verify candidates width {n} != configured n_max {self.n_max}"
            )

        logits = logits_output.next_token_logits  # flat [bs * n, V]

        slot_mapping = sampling_info.req_pool_indices[:bs]

        # Native per-draft-position grammar: scatter then mask in-fused.
        if sampling_info.vocab_mask is not None:

            self._scatter_grammar(sampling_info.vocab_mask, slot_mapping, bs, n)

        ver = self._fused_multistep(
            logits, slot_mapping, sampling_info.valid_cache_lengths, candidates
        )

        predict = ver.tokens.view(-1)
        accept_lengths = torch.add(ver.offsets.view(-1), 1, out=self.accept_buf[:bs])

        # TP-rank sync — see sample().
        self.maybe_broadcast(predict, accept_lengths)

        self._write_logprob_outputs(logits_output, logits, sampling_info, predict)

        return predict, accept_lengths


if available:

    register_backend("sonic", SonicSamplingBackend)
