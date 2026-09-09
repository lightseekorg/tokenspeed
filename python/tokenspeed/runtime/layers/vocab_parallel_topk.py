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

"""Top-k over a vocabulary-parallel LM head without gathering whole rows."""

from __future__ import annotations

import torch

from tokenspeed.runtime.distributed.comm_ops import all_gather_into_tensor
from tokenspeed.runtime.layers.logits_processor import (
    fused_softcap_generic,
    should_apply_lm_head_quant_method,
)
from tokenspeed.runtime.utils import get_colorful_logger

logger = get_colorful_logger(__name__)

#: A token id above this is not exact in the fp32 the packed gather carries it in.
_MAX_EXACT_ID = 2**24


class VocabParallelTopK:
    """Per-row top-k over a vocab-parallel head, gathering only the survivors.

    Only ``top_k`` candidates per row survive, so all-gathering every rank's
    whole logits row moves orders of magnitude more bytes than the answer
    needs. Taking each shard's local top-k first is exact only when a shard's
    local index maps to a global token id by a constant offset, which
    :attr:`enabled` establishes once at construction.

    Args:
        lm_head: The vocabulary-parallel head to project through.
        tp_size: Ranks the vocabulary is sharded over.
        tp_rank: This rank's index within ``tp_group``.
        tp_group: The process group the packed all-gather runs on.
        vocab_size: The model's whole vocabulary size.
        top_k: Candidates to keep per row.
        max_rows: Widest row count a call will ever carry, which sizes the
            resident staging and landing buffers.
        logit_scale: Multiplier applied to the shard's logits, or None.
        softcapping: Final logit softcapping, or None. Monotone in the logit,
            so it changes the scores but never which candidates survive.
        skip_all_gather: The owning processor keeps its logits rank-local.
        dp_sampling_enabled: The owning processor samples per DP rank.
    """

    def __init__(
        self,
        lm_head,
        tp_size: int,
        tp_rank: int,
        tp_group,
        vocab_size: int,
        top_k: int,
        max_rows: int,
        logit_scale: float | None,
        softcapping: float | None,
        skip_all_gather: bool,
        dp_sampling_enabled: bool,
    ) -> None:
        self.lm_head = lm_head
        self.tp_size = int(tp_size)
        self.tp_group = tp_group
        self.vocab_size = int(vocab_size)
        self.top_k = int(top_k)
        self.max_rows = int(max_rows)
        self.logit_scale = logit_scale
        self.softcapping = softcapping
        self._enabled = False
        self._radix_topk = None
        self._shard_seq_lens: torch.Tensor | None = None
        self._gather_buffers: tuple[torch.Tensor, torch.Tensor] | None = None
        self._plan(int(tp_rank), bool(skip_all_gather), bool(dp_sampling_enabled))

    @property
    def enabled(self) -> bool:
        """Whether a shard-local top-k is exact for this head's geometry."""
        return self._enabled

    def _plan(self, tp_rank: int, skip_all_gather: bool, dp_sampling: bool) -> None:
        head = self.lm_head
        shard = getattr(head, "shard_indices", None)
        if (
            self.tp_size <= 1
            or shard is None
            or not hasattr(head, "weight")
            or skip_all_gather
            or dp_sampling
        ):
            return
        num_org = int(shard.num_org_elements)
        if self.vocab_size > _MAX_EXACT_ID:
            logger.info(
                "Vocab-parallel top-k disabled: a token id in a %d-token "
                "vocabulary is not exact in the fp32 the packed all-gather "
                "carries it in.",
                self.vocab_size,
            )
            return
        if (
            int(shard.num_added_elements) != 0
            or int(shard.num_org_elements_padded) != num_org
            or num_org * self.tp_size != self.vocab_size
            or int(shard.org_vocab_start_index) != num_org * tp_rank
        ):
            logger.info(
                "Vocab-parallel top-k disabled: this %d-way vocabulary shard "
                "is padded or carries added tokens, so a shard-local index is "
                "not a global token id.",
                self.tp_size,
            )
            return
        self._enabled = True
        self._radix_topk = self._probe_radix_topk(num_org)

    def _probe_radix_topk(self, num_cols: int):
        """The vendored single-pass radix top-k, if it serves this shard.

        The vendored TensorRT-LLM kernel takes the ``[rows, vocab/tp]``
        top-16 in one pass. Probed here, at the widest row count, so its
        scratch arena is allocated before the caller is ever captured.

        Args:
            num_cols: Columns in this rank's vocabulary shard.

        Returns:
            The runner class, or None to keep ``torch.topk``.
        """
        from tokenspeed_kernel.ops.attention.cute_dsl.dsa_topk import (
            has_cute_dsl_decode_topk,
        )

        if not has_cute_dsl_decode_topk():
            return None
        from tokenspeed_kernel.thirdparty.cute_dsl.topk import (
            CuteDSLTopKDecodeSinglePassMultiCTAClusterRunner as runner,
        )

        device = self.lm_head.weight.device
        probe = torch.zeros(
            self.max_rows, num_cols, dtype=torch.bfloat16, device=device
        )
        lens = torch.full((self.max_rows,), num_cols, dtype=torch.int32, device=device)
        try:
            runner._row_states_initialized = False
            indices, values = runner.forward(
                probe, lens, self.top_k, 1, return_val=True
            )
        except Exception as exc:  # noqa: BLE001
            logger.info("Radix top-k unavailable (%s); using torch.topk", exc)
            return None
        if indices is None or values is None:
            # Cluster capacity refused the shape; the base runner does not
            # earn a second code path here.
            logger.info(
                "Radix top-k declined rows=%d cols=%d; using torch.topk",
                self.max_rows,
                num_cols,
            )
            return None
        return runner

    def _shard_topk(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-row top-k over this rank's vocabulary shard.

        Args:
            logits: This rank's ``[rows, shard]`` logits.

        Returns:
            Unsorted ``(values, ids)``; the caller re-selects across ranks
            afterwards, so within-shard order does not matter.
        """
        if self._radix_topk is None:
            return torch.topk(logits, self.top_k, dim=-1, sorted=False)
        rows, cols = logits.shape
        lens = self._shard_seq_lens
        if lens is None or lens.shape[0] < rows:
            lens = torch.full(
                (self.max_rows,), cols, dtype=torch.int32, device=logits.device
            )
            self._shard_seq_lens = lens
        self._radix_topk._row_states_initialized = False
        ids, values = self._radix_topk.forward(
            logits, lens[:rows], self.top_k, 1, return_val=True
        )
        if ids is None:
            return torch.topk(logits, self.top_k, dim=-1, sorted=False)
        return values, ids

    def _ensure_gather_buffers(
        self, rows: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Resident staging and landing pad for one packed all-gather.

        Values and ids ride the same fp32 rows, so a call pays one collective
        rather than two; :attr:`enabled` checks the vocabulary is small enough
        for an id to be exact in fp32.

        Args:
            rows: Rows this call carries.
            device: Device the buffers must live on.

        Returns:
            ``(staged, gathered)`` views sized for ``rows``.
        """
        width = 2 * self.top_k
        buffers = self._gather_buffers
        if buffers is None or buffers[0].device != device:
            buffers = (
                torch.empty((self.max_rows, width), dtype=torch.float32, device=device),
                torch.empty(
                    (self.tp_size * self.max_rows, width),
                    dtype=torch.float32,
                    device=device,
                ),
            )
            self._gather_buffers = buffers
        return buffers[0][:rows], buffers[1][: rows * self.tp_size]

    def _shard_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """This rank's slice of the logits row, however the head computes it.

        Same dispatch as ``LogitsProcessor._get_logits``: a genuinely quantized
        head runs its own GEMM, because its ``weight`` is a packed tensor that
        must never be matmul'd.

        Args:
            hidden_states: ``[rows, hidden]`` activations to project.

        Returns:
            This rank's ``[rows, shard_padded]`` logits.
        """
        head = self.lm_head
        quant_method = getattr(head, "quant_method", None)
        if should_apply_lm_head_quant_method(head, quant_method):
            return quant_method.apply(head, hidden_states, None)
        return torch.matmul(hidden_states.to(head.weight.dtype), head.weight.T)

    def __call__(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Top-k over the whole vocabulary, gathering only each shard's best.

        Args:
            hidden_states: ``[rows, hidden]`` activations to project.

        Returns:
            ``(candidate_ids, values)``, both ``[rows, top_k]`` and sorted by
            descending value; ids are int64 global token ids and values are
            fp32.
        """
        shard = self.lm_head.shard_indices
        top_k = self.top_k

        logits = self._shard_logits(hidden_states)[:, : int(shard.num_org_elements)]
        if self.logit_scale is not None:
            logits.mul_(self.logit_scale)
        values, ids = self._shard_topk(logits)

        rows = int(values.shape[0])
        staged, gathered = self._ensure_gather_buffers(rows, values.device)
        staged[:, :top_k].copy_(values)
        staged[:, top_k:].copy_(ids)
        staged[:, top_k:].add_(float(shard.org_vocab_start_index))
        all_gather_into_tensor(gathered, staged, self.tp_group)

        # Rank-major to row-major, so one row's candidates from every rank sit
        # side by side for the final selection, values and ids each on a plane.
        planes = (
            gathered.view(self.tp_size, rows, 2, top_k)
            .permute(1, 2, 0, 3)
            .reshape(rows, 2, self.tp_size * top_k)
        )
        # Sorted, so the surviving candidate order does not depend on which
        # rank happened to contribute a value.
        values, lanes = torch.topk(planes[:, 0], top_k, dim=-1, sorted=True)
        # Everything downstream indexes codebooks with these.
        candidate_ids = torch.gather(planes[:, 1], 1, lanes).to(torch.int64)
        if self.softcapping:
            fused_softcap_generic(values, self.softcapping)
        return candidate_ids, values
