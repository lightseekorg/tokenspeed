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

One fused Triton pipeline (`sample`, two kernels; `verify`, three, for
spec-decode) handles a mixed batch (greedy + temperature/top_k/top_p/min_p
+ penalties + grammar) selected per row by an `Indicator` bitfield — no separate
greedy path or per-mode graph.

Buffers are either **slot-indexed** per-request state (indicator bits, scalars,
grammar, counts, bias) in one ``SlotState`` sized to ``max_req_pool_size + 1``
and gathered at ``slot_mapping[row]`` (= ``req_pool_indices``), written on the
device once in ``_reset_slot``; or **batch-indexed**
I/O sized to max batch (scratchpad keys, verify's candidate values/indices).
Noise is drawn inside sonic's kernels from Philox keyed by the slot's seed
and the absolute position of the token drawn (``_prepare_sampling_offsets``),
at the candidates the kernels actually read: no noise plane, no per-step
refresh, draws that depend only on the request, and every TP rank drawing the
same. The slot rows are part of the retraction image (``slot_state_rows``).

The runtime's unified sampling rule routes every decode row through
``verify()`` — non-speculative serving is its ``N == 1`` case (a one-column
candidate window with nothing to accept). ``verify`` needs a lookahead
of at least one, so that case resolves through ``sample``: the same
kernel ``sample()`` uses, reading the same slot state and noise, so the two
paths are bitwise identical (pinned by ``test_sonic_backend.py``).

Under TP with a vocabulary-parallel head the sampled rows can stay
vocab-sharded (``vocab_shard_top_k``): each rank selects its shard's best
``MAX_K`` candidates with the slot state applied at global token ids, the ranks
gather only those (``gather_topk_pairs``), and every rank draws from their merge
what it would draw from the gathered rows (bf16 bit for bit; fp32 among the
exact fp32 top-``MAX_K``). Every rank merges the same gathered bytes with the
same slot state and noise, so sharded draws agree across ranks without the TP
sync broadcast.

Synthetic acceptance (``--synthetic-acceptance-length``, benchmarking only)
runs the verify for its cost and then forces each request's length; the
token at the cutoff comes from the candidates the verify selected
(``_synthetic_outputs``).

Not yet supported: DP-sampling, draft trees, top-k output logprobs, EAGLE3
``d2t`` cross-vocab.

The kernels and indicator bits are sonic-sampler's, vendored in
``tokenspeed_kernel.thirdparty.sonic_sampler``; ``tokenspeed_kernel.ops.sampling.sonic``
launches them and picks their config per launch from a table (``pick``);
nothing tunes at startup.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import torch
from tokenspeed_kernel.ops.sampling import sonic, vocab_topk_merge_supports
from tokenspeed_kernel.ops.sampling.sonic import (
    MAX_K,
    MAX_OFFSET,
    Indicator,
    SlotState,
    SonicConfig,
    pick,
    scratchpad_columns,
)
from tokenspeed_kernel.platform import pdl_enabled
from typing_extensions import override

from tokenspeed.runtime.layers.sampled_vocab_shard import gather_topk_pairs
from tokenspeed.runtime.sampling.backends.base import (
    SamplingBackend,
    SamplingBackendConfig,
    SpeculativeSamplingPools,
)
from tokenspeed.runtime.sampling.registry import register_backend
from tokenspeed.runtime.sampling.sampling_params import _SAMPLING_EPS, SamplingParams
from tokenspeed.runtime.sampling.utils import gather_token_logprobs_torch
from tokenspeed.runtime.utils.nvtx import nvtx_range

if TYPE_CHECKING:

    from tokenspeed.runtime.layers.logits_processor import LogitsProcessorOutput
    from tokenspeed.runtime.layers.sampled_vocab_shard import SampledVocabShard
    from tokenspeed.runtime.sampling.sampling_batch_info import SamplingBatchInfo
    from tokenspeed.runtime.sampling.tree_verify import DraftTreeQ, TreeVerifyBatch


# Logits sonic ranks: bf16, or fp32 (soft-capped heads) sampled exactly at the candidates bf16 keys pick.
_LOGITS_DTYPES = (torch.bfloat16, torch.float32)

# Never-admitted slots' fp32 temperature, top-p, min-p, repetition, frequency and presence.
_EXACT_DEFAULTS = (1.0, 1.0, 0.0, 1.0, 0.0, 0.0)


def _sanitize(sp: SamplingParams) -> tuple[float, int, float, float]:
    """Sanitize a request's ``(temperature, top_k, top_p, min_p)`` for sonic.

    ``_reset_slot`` derives the ``Indicator`` bits from the same values it writes
    into the slot state (``Indicator.from_params`` tests them exactly), so
    sanitizing here keeps every bit agreeing with what the kernel reads:
    epsilon-neutral temperature stays ``1.0`` (bit off — skips a wasted
    ``logits / t`` vocab pass), and ``top_k`` is capped at MAX_K so the ``-1``
    "disabled" sentinel lands on the bounded top-MAX_K path (bit off). The
    greedy early return also pins temperature to 1.0, so the slot's indicator
    is exactly ``GREEDY`` and the kernel skips the temperature pass. ``min_p =
    1.0`` is remapped to 0.999 because ``from_params`` tests the open interval
    (1.0 would drop the MIN_P bit); the bf16 buffer rounds it back to 1.0
    and ``_reset_slot`` stores 1.0 in the fp32 row, whose pivot = max keeps
    only max-probability tokens, as requested."""

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
    backend chooses. Inherited from sonic's 16-bit packed ranking: bf16 logits'
    candidates are compared at bf16 resolution (temperature, penalties and
    logit bias are applied in bf16, and the scaled logits are re-rounded), so
    where the fp32 backends separate near-tied tokens sonic's greedy pick is
    only guaranteed to lie within one bf16 ulp of the maximum (pinned by
    ``test_sonic_backend.py``); temperature/top_p/min_p are stored in bf16 too.
    fp32 logits (soft-capped heads) are sampled exactly: their bf16 keys only
    pick the top-128 candidates, whose fp32 values are re-read and given fp32
    temperature, top_p, min_p and penalties (logit bias stays bf16), and
    greedy is the first exact fp32 maximum. Under TP, decode counts for
    penalties accumulate in-kernel from each rank's own draw. Sharded draws
    are the emitted tokens (every rank draws the same, no broadcast); draws
    from gathered whole rows (output logprobs, cumulative probabilities, NaN
    detection) are followed by the rank-0 broadcast that decides the emitted
    token.

    Requires bf16 or fp32 logits and ``vocab_size >= 512`` (its top-k
    reduction's minimum)."""

    _HAS_POOL_STATE = True
    _SUPPORTS_DP_VERIFY = False
    # Forced lengths over sonic's verify; see ``_synthetic_outputs``.
    _SUPPORTS_SYNTHETIC_ACCEPTANCE = True

    # Prompt token counts the runtime never fills: every slot reads zeros.
    constant_slot_state = ("state.context_counts", "_exact_state.context_counts")

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

        # The kernels' row offsets are int32; the largest launch must fit (``sonic.sample`` checks each).
        if self.max_bs * self.n_max * self.vocab_size > MAX_OFFSET:

            raise ValueError(
                f"SonicSamplingBackend needs max_bs * draft tokens * vocab_size <= "
                f"{MAX_OFFSET}, got {self.max_bs} * {self.n_max} * {self.vocab_size}"
            )

        rows = self.max_bs * self.n_max

        with torch.device(self.device):

            # Batch-indexed working space, persistent (captured kernel never allocs).
            self.scratchpad = torch.empty(
                (rows, scratchpad_columns(self.vocab_size)), dtype=torch.uint32
            )
            self.out_tok = torch.empty((self.max_bs, 1), dtype=torch.int32)
            self.lengths_buf = torch.ones((self.max_bs,), dtype=torch.int32)

            if self.spec:

                # verify() drafted-tokens I/O and candidate scratch; see ``_fused_multistep``.
                self.v_drafted = torch.empty(self.max_bs, self.n_max, dtype=torch.int32)
                self.v_accepted = torch.empty((self.max_bs, 1), dtype=torch.int32)
                self.accept_buf = torch.empty((self.max_bs,), dtype=torch.int32)
                self.values = torch.empty((rows, MAX_K), dtype=torch.float32)
                self.indices = torch.empty((rows, MAX_K), dtype=torch.int32)
                # Synthetic acceptance's target token per request.
                self.synthetic_tokens = torch.empty((self.max_bs,), dtype=torch.int32)

            # Vocabulary-sharded rows: this rank's candidates and every rank's.
            self._shard_ranks = self._vocab_shard_ranks(config)
            # Candidates carry draft probabilities recorded over each rank's shard: a third block, and their merged copy.
            self._candidate_drafts: torch.Tensor | None = None
            if self._shard_ranks is not None:
                width = 2 * MAX_K
                if self.spec and config.enable_speculative_sampling:
                    width = 3 * MAX_K
                    self._candidate_drafts = torch.empty(
                        (rows, MAX_K), dtype=torch.float32
                    )
                self._shard_pairs = torch.empty((rows * width,), dtype=torch.float32)
                self._shard_landing = torch.empty(
                    (self._shard_ranks * rows * width,), dtype=torch.float32
                )

            # fp32 temperature, top-p, min-p and penalties, read with fp32 logits so none rounds to bf16.
            self.exact_scalars = (
                torch.tensor(_EXACT_DEFAULTS, dtype=torch.float32)[:, None]
                .repeat(1, self.pool_rows)
                .contiguous()
            )

        # The slot-indexed state every launch gathers at ``slot_mapping[row]``; never-admitted slots keep the backend's seed.
        self.state = SlotState.allocate(
            self.pool_rows, self.n_max, self.vocab_size, config.random_seed, self.device
        )
        self._exact_state = self.state.with_scalars(self.exact_scalars)

        # xgrammar's int32 bitmask layout, scattered per step by ``_scatter_grammar``.
        self.grammar = self.state.grammar

        # fp32 logits compile their own kernels: warmed the first time they arrive.
        self._fp32_warm = False

        # The kernels run on CUDA only; a state built elsewhere (tests) has nothing to compile.
        if torch.device(self.device).type == "cuda":

            self._warmup(torch.bfloat16)

    # ------------------------------------------------------------------ #
    # Vocabulary-sharded rows
    # ------------------------------------------------------------------ #
    @staticmethod
    def _vocab_shard_ranks(config: SamplingBackendConfig) -> int | None:
        """Ranks a vocabulary-parallel head shards the sampled rows over, or None to read whole rows.

        Output logprobs, cumulative probabilities and NaN detection read whole rows.
        """
        tp_size = 1 if config.tp_group is None else len(config.tp_group)
        if (
            tp_size <= 1
            or config.enable_output_logprobs
            or config.enable_output_cumulative_probs
            or config.enable_nan_detection
        ):
            return None
        return tp_size

    @override
    def vocab_shard_top_k(self) -> int | None:
        return None if self._shard_ranks is None else MAX_K

    @override
    def supports_vocab_shard(
        self, dtype: torch.dtype, columns: int, tp_size: int
    ) -> bool:
        # sonic's reduction tiles at least 512 columns.
        return (
            self._shard_ranks == tp_size
            and dtype in _LOGITS_DTYPES
            and columns >= 512
            # The draw merges a power-of-two count of ranks' candidates.
            and tp_size & (tp_size - 1) == 0
            and columns * tp_size == self.vocab_size
            and vocab_topk_merge_supports(tp_size, MAX_K)
        )

    @override
    def verifies_vocab_shard_draft_probs(self) -> bool:
        # Each rank's candidates carry q over its own columns through the gather.
        return self._shard_ranks is not None

    @override
    def speculative_sampling_pools(self) -> SpeculativeSamplingPools:
        """The slot state's own rows, which the drafter samples its proposals with.

        The fp32 rows hold each request's temperature and top-p, and the slot
        state its top-k (1 for greedy requests) and Philox seed, all written
        on admission and part of the slot image.
        """
        return SpeculativeSamplingPools(
            temperature=self._exact_state.temperature,
            top_k=self.state.top_k,
            top_p=self._exact_state.top_p,
            seed=self.state.seeds,
        )

    def _gather_pairs(
        self, staged: torch.Tensor, tp_size: int, tp_group: tuple[int, ...]
    ) -> torch.Tensor:
        """Every rank's ``[rows, w]`` candidate pairs as ``[tp_size, rows, w]``, landing in this backend's buffer."""
        rows, width = staged.shape
        landing = self._shard_landing[: tp_size * rows * width]
        return gather_topk_pairs(
            staged, landing.view(tp_size * rows, width), tp_size, tp_group
        )

    def _vocab_shard(self, shard: SampledVocabShard) -> sonic.VocabShard:
        """The launch's view of ``shard``: its slice of the vocabulary and the gather of every rank's candidates."""
        return sonic.VocabShard(
            vocab_start=shard.vocab_start,
            logit_scale=shard.logit_scale,
            softcap=shard.softcap,
            gather=functools.partial(
                self._gather_pairs, tp_size=shard.tp_size, tp_group=shard.tp_group
            ),
            pairs=self._shard_pairs,
        )

    @override
    def warmup_vocab_shard_sampling(
        self,
        shard: SampledVocabShard,
        candidates: torch.Tensor,
        *,
        draft_q: DraftTreeQ | None,
        with_vocab_mask: bool,
        draft_probs: torch.Tensor | None,
    ) -> None:
        """Launch every batch size once over dummy shard logits, as ``_warmup`` does over whole rows.

        Every rank runs the same launches, so their gathers pair up. Grammar
        masks change no sonic kernel (they are slot state); sonic reads the
        runtime's ``draft_probs`` (whole rows or this rank's columns), not the
        sparse ``draft_q``.
        """
        del candidates, draft_q, with_vocab_mask
        columns = self.vocab_size if draft_probs is None else draft_probs.shape[2]
        with torch.device(self.device):
            logits = torch.zeros(
                (self.max_bs * self.n_max, shard.logits.shape[1]),
                dtype=shard.logits.dtype,
            )
            slot = torch.zeros((self.max_bs,), dtype=torch.int64)
            shifted = torch.zeros((self.max_bs + 1,), dtype=torch.int64)[1:]
            offsets = torch.zeros((self.pool_rows,), dtype=torch.int64)
            cand = torch.zeros((self.max_bs, self.n_max), dtype=torch.int32)
        for slots in (slot, shifted):
            for bs in range(1, self.max_bs + 1):
                self._fused_singular(logits[:bs], slots[:bs], offsets, shard=shard)
                if self.spec:
                    for table in self._warmup_draft_tables(columns):
                        verified = self._fused_multistep(
                            logits[: bs * self.n_max],
                            slots[:bs],
                            offsets,
                            cand[:bs],
                            shard=shard,
                            draft_probs=table,
                        )
                        if self._synthetic_lengths is not None:
                            self._synthetic_outputs(
                                cand[:bs], slots[:bs], offsets, *verified, row_offset=0
                            )
        self.reset_capture_state()

    # ------------------------------------------------------------------ #
    # JIT warm-up
    # ------------------------------------------------------------------ #
    def _warmup_draft_tables(
        self, columns: int | None = None
    ) -> tuple[torch.Tensor | None, ...]:
        """The draft-probability tables serving verifies with: none, and a one-slot table of ``columns`` (default the vocabulary) under ``enable_speculative_sampling``, its strides the runtime's."""

        if not self.config.enable_speculative_sampling:

            return (None,)

        table = torch.zeros(
            (1, self.n_max, columns or self.vocab_size),
            dtype=torch.float32,
            device=self.device,
        )

        return (None, table)

    def _warmup(self, dtype: torch.dtype) -> None:
        """Launch every batch size once on dummy logits gathering slot 0.

        Graph capture only compiles the decode row counts it captures; the
        eager paths (prefill rows of a mixed round, uncaptured batch sizes)
        would otherwise hit new kernel variants at runtime, and each Triton
        compile stalls the whole server for seconds. Each launch takes the
        config serving picks for its rows, and Triton caches by config and
        integer specialization of ``batch_size``, so most launches here are
        cache hits."""

        with torch.device(self.device):

            logits = torch.zeros(
                (self.max_bs * self.n_max, self.vocab_size), dtype=dtype
            )
            slot = torch.zeros((self.max_bs,), dtype=torch.int64)
            # Mixed rounds pass the decode rows' slots at an offset, a pointer Triton specializes apart from 16-byte aligned ones.
            shifted = torch.zeros((self.max_bs + 1,), dtype=torch.int64)[1:]
            # The dtype of the base class's offsets pool, which serving passes.
            offsets = torch.zeros((self.pool_rows,), dtype=torch.int64)
            cand = torch.zeros((self.max_bs, self.n_max), dtype=torch.int32)

        for slots in (slot, shifted):

            for bs in range(1, self.max_bs + 1):

                self._fused_singular(logits[:bs], slots[:bs], offsets)

                if self.spec:

                    for draft_probs in self._warmup_draft_tables():

                        verified = self._fused_multistep(
                            logits[: bs * self.n_max],
                            slots[:bs],
                            offsets,
                            cand[:bs],
                            draft_probs=draft_probs,
                        )

                        if self._synthetic_lengths is not None:

                            self._synthetic_outputs(
                                cand[:bs], slots[:bs], offsets, *verified, row_offset=0
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
        sonic's device buffers like the sibling backends. sonic's own host-side
        slot management (unfenced copies through pinned relays, a per-slot
        noise plane) is not vendored.

        ``Indicator.from_params`` derives the bits from the same sanitized
        values written here, so every bit agrees with what the kernel reads;
        the GRAMMAR bit comes from ``sp.has_grammar`` (the masks arrive per
        step through ``_scatter_grammar``). ``greedy_draft``: drafts are point
        draws (chain EAGLE, topk=1), so stochastic rows verify under sonic's
        masked-rejection regime, unless the drafter samples from and records
        its q (``enable_speculative_sampling``)."""

        temperature, top_k, top_p, min_p = _sanitize(sp)

        bias: dict[int, float] | None = None

        if sp.logit_bias:

            bias = {int(t): float(v) for t, v in sp.logit_bias.items()}

        bits = Indicator.from_params(
            grammar=sp.has_grammar,
            multiplicative=float(sp.repetition_penalty),
            frequency=float(sp.frequency_penalty),
            presence=float(sp.presence_penalty),
            logit_bias=bias is not None,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            min_p=min_p,
            greedy_draft=not self.config.enable_speculative_sampling,
        )

        state = self.state
        # The fp32 row keeps a requested min_p of 1.0 exact; 0.999 only sets the MIN_P bit.
        exact_min_p = 1.0 if min_p and float(sp.min_p) >= 1.0 else min_p

        state.indicators[pool_idx].fill_(int(bits))
        state.top_k[pool_idx].fill_(top_k)
        state.decode_counts[pool_idx].zero_()

        for rows, row_min_p in ((state, min_p), (self._exact_state, exact_min_p)):

            values = (
                temperature,
                top_p,
                row_min_p,
                float(sp.repetition_penalty),
                float(sp.frequency_penalty),
                float(sp.presence_penalty),
            )
            for row, value in zip(
                (
                    rows.temperature,
                    rows.top_p,
                    rows.min_p,
                    rows.repetition,
                    rows.frequency,
                    rows.presence,
                ),
                values,
                strict=True,
            ):

                row[pool_idx].fill_(value)

        row = state.logit_bias[pool_idx]
        row.zero_()

        if bias:

            row[list(bias)] = torch.tensor(
                list(bias.values()), dtype=torch.bfloat16, device=self.device
            )

        self.grammar[pool_idx].fill_(-1)
        state.seeds[pool_idx].fill_(int(sp.seed))

    @override
    def slot_state_rows(self, slot: int) -> list[torch.Tensor]:
        """The slot's rows ``_reset_slot`` writes, which a retraction images and restores."""

        state = self.state
        rows = slice(slot, slot + 1)

        return [
            state.indicators[rows],
            state.temperature[rows],
            state.top_k[rows],
            state.top_p[rows],
            state.min_p[rows],
            state.repetition[rows],
            state.frequency[rows],
            state.presence[rows],
            state.decode_counts[rows],
            state.logit_bias[rows],
            self.grammar[rows],
            state.seeds[rows],
            *(row[rows] for row in self.exact_scalars),
        ]

    @override
    def reset_capture_state(self) -> None:
        """Warm-up routes all rows to slot 0 and accumulates its decode counts;
        zero them so the captured graph reads a clean baseline."""

        self.state.decode_counts[0].zero_()

    def _scatter_grammar(
        self,
        vocab_mask: torch.Tensor,
        slot_mapping: torch.Tensor,
        bs: int,
        n: int,
    ) -> None:
        """Scatter xgrammar's row-indexed ``[bs*n, W]`` bitmask (row ``i*n+j`` =
        request ``i``, position ``j``) into the slot-indexed ``grammar`` buffer
        ``[pool_rows, n_max, W]``, both int32 with the same bit semantics (SET =
        allowed; token ``t`` -> word ``t//32``, bit ``t%32``), so it's a plain
        copy. Captured/graph-safe (same pattern as the draft-probs scatter).

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
    def _check_logits(
        self, logits: torch.Tensor, shard: SampledVocabShard | None
    ) -> SlotState:
        """The slot state these logits launch with: fp32 logits read the fp32 scalars."""

        width = self.vocab_size if shard is None else self.vocab_size // shard.tp_size

        if logits.dtype not in _LOGITS_DTYPES or logits.shape[-1] != width:

            raise ValueError(
                f"SonicSamplingBackend requires bf16 or fp32 logits of width "
                f"{width}, got {logits.dtype} x {logits.shape[-1]}"
            )

        if logits.dtype == torch.bfloat16:

            return self.state

        # Sharded launches warm in ``warmup_vocab_shard_sampling``; whole rows warm on first use.
        if (
            shard is None
            and not self._fp32_warm
            and not torch.cuda.is_current_stream_capturing()
        ):

            self._fp32_warm = True
            self._warmup(torch.float32)

        return self._exact_state

    # ------------------------------------------------------------------ #
    # Sampling (inside the captured graph)
    # ------------------------------------------------------------------ #
    def _fused_singular(
        self,
        logits: torch.Tensor,
        slot_mapping: torch.Tensor,
        offsets: torch.Tensor,
        config: SonicConfig | None = None,
        shard: SampledVocabShard | None = None,
    ) -> torch.Tensor:
        """One draw per row into ``out_tok``, counted for penalties; returns the ``[bs, 1]`` tokens.

        With ``shard``, ``logits`` are this rank's raw head logits of its shard.
        """

        state = self._check_logits(logits, shard)

        bs, width = logits.shape

        return sonic.sample(
            logits,
            state,
            slot_mapping,
            offsets,
            self.out_tok[:bs],
            self.scratchpad[:bs],
            config or pick(width, bs),
            enable_pdl=pdl_enabled(),
            shard=None if shard is None else self._vocab_shard(shard),
        )

    def _sample_rows(
        self,
        logits_output: LogitsProcessorOutput,
        sampling_info: SamplingBatchInfo,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One fused single-step draw per row: the body shared by ``sample()``
        and the ``N == 1`` (non-speculative decode) case of ``verify()``."""

        shard = logits_output.vocab_shard
        logits = logits_output.next_token_logits if shard is None else shard.logits

        bs = logits.shape[0]

        # Padded rows carry slot 0 and write it (grammar, counts); the runtime never allocates it.
        slot_mapping = sampling_info.req_pool_indices[:bs]

        # Native grammar: scatter the bitmask; the kernel masks in-fused.
        if sampling_info.vocab_mask is not None:

            self._scatter_grammar(sampling_info.vocab_mask, slot_mapping, bs, 1)

        sampled = self._fused_singular(
            logits,
            slot_mapping,
            self._prepare_sampling_offsets(sampling_info),
            shard=shard,
        ).view(-1)

        # TP-rank sync (rank 0 wins) of draws from gathered rows; sharded draws merge the same gathered candidates on every rank.
        if shard is None:
            self.maybe_broadcast(sampled)

        self._write_logprob_outputs(logits_output, logits, sampling_info, sampled)

        # The runtime's stop conditions cut the returned lengths in place, so every call refills them.
        lengths = self.lengths_buf[:bs]
        lengths.fill_(1)

        return sampled, lengths

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
        config: SonicConfig | None = None,
        shard: SampledVocabShard | None = None,
        draft_probs: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One ``verify`` launch over ``[bs * n, V]`` logits; returns ``(drafted, accepted)``.

        ``drafted[:, :γ]`` holds the γ proposals and is the output too: the
        kernel stores only the corrected token at the
        rejection offset (the rest of the row must already be the drafts), so
        this keeps sonic's in-place semantics with a stable captured pointer
        (the default would clone-allocate). The trailing bonus column is
        output-only: the kernel masks it out of the draft load and
        short-circuits the residual correction there. Without ``draft_probs``
        (point-draw drafts) sonic's masked-rejection regime accepts iff
        u <= p(draft), residual = p with the drafted token zeroed; with the
        drafter's recorded ``draft_probs`` it accepts iff u * q(draft) <=
        p(draft), residual = max(p - q, 0)."""

        state = self._check_logits(logits, shard)

        bs, n = candidates.shape
        gamma = n - 1

        drafted = self.v_drafted[:bs]
        drafted[:, :gamma].copy_(candidates[:, 1:])
        drafted[:, gamma:].zero_()
        accepted = self.v_accepted[:bs]

        sonic.verify(
            logits,
            state,
            slot_mapping,
            offsets,
            drafted,
            accepted,
            self.scratchpad[: bs * n],
            self.values[: bs * n],
            self.indices[: bs * n],
            config or pick(logits.shape[1], bs * n),
            enable_pdl=pdl_enabled(),
            shard=None if shard is None else self._vocab_shard(shard),
            draft_probs=draft_probs,
            reject_threshold=self.config.spec_reject_draft_prob_threshold,
            candidate_drafts=self._candidate_drafts,
        )

        return drafted, accepted

    def _synthetic_outputs(
        self,
        candidates: torch.Tensor,
        slot_mapping: torch.Tensor,
        offsets: torch.Tensor,
        drafted: torch.Tensor,
        accepted: torch.Tensor,
        *,
        row_offset: int,
    ) -> None:
        """Replace a verify's outcome with the synthetic acceptance lengths.

        Each request keeps its first ``L - 1`` drafts and then the target's
        token at position ``L - 1`` (``sonic.synthetic_targets``, drawn from
        the candidates the verify just left), as the greedy and flashinfer
        backends override theirs. The decode counts that penalties read keep
        the verify's own tokens: synthetic output is for benchmarking only.

        Args:
            candidates: ``[bs, n]`` the verified window (last token, drafts).
            slot_mapping: ``[bs]`` slot of each request.
            offsets: The verify's per-slot Philox offsets.
            drafted: ``[bs, n]`` the verify's output tokens; rewritten.
            accepted: ``[bs, 1]`` the verify's accepted drafts; rewritten.
            row_offset: The requests' first row in the step's full batch.
        """
        bs, n = candidates.shape
        lengths = self.synthetic_lengths(candidates, row_offset)
        tokens = sonic.synthetic_targets(
            self.values[: bs * n],
            self.indices[: bs * n],
            self.state,
            slot_mapping,
            offsets,
            lengths,
            self.synthetic_tokens[:bs],
            enable_pdl=pdl_enabled(),
        )
        drafted[:, : n - 1].copy_(candidates[:, 1:])
        drafted.scatter_(1, (lengths - 1).long()[:, None], tokens[:, None])
        accepted.copy_((lengths - 1)[:, None])

    @override
    @nvtx_range("sampling:verify", color="yellow")
    def verify(
        self,
        logits_output: LogitsProcessorOutput,
        sampling_info: SamplingBatchInfo,
        candidates: torch.Tensor,
        *,
        tree: TreeVerifyBatch | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Chain speculative verification via sonic's ``verify``.

        ``candidates`` is ``[bs, n]`` (n = γ+1): column 0 is the last verified
        token, 1..γ the drafts. sonic compares its per-position selection against
        ``drafted[s]`` at the same index, so we pass ``drafted = candidates[:, 1:]``
        and read back ``drafted`` (the accepted prefix + next token) with
        ``accept_length = accepted + 1``. Greedy and stochastic rows resolve in
        one pipeline via per-row indicators; the drafts are point draws, handled
        natively by sonic's masked-rejection regime (``draft_probabilities=None``:
        accept iff u <= p(draft), residual = p with the drafted token zeroed).

        ``n == 1`` is non-speculative decode (the runtime's unified sampling
        rule): nothing to accept, one token to draw, ``accept_length == 1`` —
        exactly ``sample()``, so it takes the ``sample`` path. Widths
        other than 1 and ``n_max`` are refused (``v_drafted`` is ``[max_bs,
        n_max]``)."""

        if tree is not None:

            raise RuntimeError("SonicSamplingBackend verifies draft chains only")

        bs, n = candidates.shape

        if n == 1:

            return self._sample_rows(logits_output, sampling_info)

        if n != self.n_max:

            raise RuntimeError(
                f"verify candidates width {n} != configured n_max {self.n_max}"
            )

        # Flat [bs * n, V], or this rank's [bs * n, V_s] shard.
        shard = logits_output.vocab_shard
        logits = logits_output.next_token_logits if shard is None else shard.logits

        slot_mapping = sampling_info.req_pool_indices[:bs]

        # Native per-draft-position grammar: scatter then mask in-fused.
        if sampling_info.vocab_mask is not None:

            self._scatter_grammar(sampling_info.vocab_mask, slot_mapping, bs, n)

        offsets = self._prepare_sampling_offsets(sampling_info)
        predict, accepted = self._fused_multistep(
            logits,
            slot_mapping,
            offsets,
            candidates,
            shard=shard,
            draft_probs=sampling_info.draft_probs,
        )

        # Benchmark acceptance replaces the outcome after the verify ran for its cost.
        if self._synthetic_lengths is not None:

            self._synthetic_outputs(
                candidates,
                slot_mapping,
                offsets,
                predict,
                accepted,
                row_offset=sampling_info.batch_row_offset,
            )

        predict = predict.view(-1)
        accept_lengths = torch.add(accepted.view(-1), 1, out=self.accept_buf[:bs])

        # TP-rank sync of gathered rows only — see sample().
        if shard is None:
            self.maybe_broadcast(predict, accept_lengths)

        self._write_logprob_outputs(logits_output, logits, sampling_info, predict)

        return predict, accept_lengths


register_backend("sonic", SonicSamplingBackend)
