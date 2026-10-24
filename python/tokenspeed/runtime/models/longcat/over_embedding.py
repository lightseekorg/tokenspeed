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
"""LongCat branch-local over-embedding runtime layer."""

from __future__ import annotations

import re
from dataclasses import replace

import torch
from tokenspeed_kernel.ops.over_embedding import (
    OverEmbeddingSpec,
    append_packed_lookup_,
    longcat_pro_tp8_spec,
    project_add_word_,
)
from tokenspeed_kernel.platform import pdl_enabled
from torch import nn
from torch.nn.parameter import Parameter

from tokenspeed.runtime.distributed.comm_ops import all_reduce
from tokenspeed.runtime.execution.request_token_history import RequestTokenHistoryView
from tokenspeed.runtime.layers.vocab_parallel_embedding import VocabParallelEmbedding


def resolve_longcat_oe_hyperparameters(config) -> tuple[int, int]:
    """Return ``(max_ngram_order, hashes_per_order)`` from HF field aliases."""
    max_ngram_order = getattr(config, "emb_neighbor_num", None)
    if max_ngram_order is None:
        max_ngram_order = getattr(config, "oe_neighbor_num", None)
    hashes_per_order = getattr(config, "emb_split_num", None)
    if hashes_per_order is None:
        hashes_per_order = getattr(config, "oe_split_num", None)
    if max_ngram_order is None or hashes_per_order is None:
        raise ValueError(
            "LongCat OE requires neighbor_num (maximum n-gram order) and "
            "split_num (hash branches per order)"
        )
    return int(max_ngram_order), int(hashes_per_order)


def resolve_longcat_oe_spec(
    *,
    vocab_size: int,
    hidden_size: int,
    max_ngram_order: int,
    hashes_per_order: int,
    modulus0: int,
    tp_size: int,
    tp_rank: int,
) -> OverEmbeddingSpec:
    """Resolve and validate one supported LongCat OE ownership profile."""
    if (vocab_size, hidden_size, tp_size) == (163840, 8192, 8):
        spec = longcat_pro_tp8_spec(tp_rank)
    else:
        raise ValueError(
            "unsupported LongCat OE profile: "
            f"vocab_size={vocab_size}, hidden_size={hidden_size}, TP={tp_size}"
        )

    branch_count = (max_ngram_order - 1) * hashes_per_order
    expected = {
        "max_ngram_order": (max_ngram_order, spec.max_ngram_order),
        "branch_count": (branch_count, spec.branch_count),
        "modulus0": (
            modulus0,
            min(
                fragment.modulus - 2 * fragment.branch_id for fragment in spec.fragments
            ),
        ),
    }
    mismatches = [
        f"{name}={actual} (expected {wanted})"
        for name, (actual, wanted) in expected.items()
        if actual != wanted
    ]
    if mismatches:
        raise ValueError(
            f"checkpoint does not match {spec.profile} TP{tp_size}: "
            + ", ".join(mismatches)
        )
    return spec


def _branch_index(weight_name: str, *, projection: bool) -> int | None:
    patterns = (
        (
            r"(?:^|\.)oe_embed_proj(\d+)\.weight$",
            r"(?:^|\.)ngram_embeddings\.post_projs\.(\d+)\.weight$",
        )
        if projection
        else (
            r"(?:^|\.)oe_embed_tokens(\d+)\.weight$",
            r"(?:^|\.)ngram_embeddings\.embedders\.(\d+)\.weight$",
        )
    )
    for pattern in patterns:
        match = re.search(pattern, weight_name)
        if match is not None:
            return int(match.group(1))
    return None


class LongCatOverEmbedding(nn.Module):
    """Word embedding plus TP-local OE branches and one final TP reduction."""

    def __init__(
        self,
        *,
        num_embeddings: int,
        embedding_dim: int,
        over_embedding_m: int,
        hashes_per_order: int,
        max_ngram_order: int,
        tp_rank: int,
        tp_size: int,
        tp_group: tuple[int, ...] | None,
        params_dtype: torch.dtype | None = None,
        ignored_token_ids: tuple[int, ...] = (),
        eos_token_id: int | None = None,
    ) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.tp_group = tp_group
        dtype = params_dtype or torch.get_default_dtype()
        self.spec = resolve_longcat_oe_spec(
            vocab_size=num_embeddings,
            hidden_size=embedding_dim,
            max_ngram_order=max_ngram_order,
            hashes_per_order=hashes_per_order,
            modulus0=over_embedding_m + 1,
            tp_size=tp_size,
            tp_rank=tp_rank,
        )
        self.spec = replace(
            self.spec,
            ignored_token_ids=tuple(dict.fromkeys(ignored_token_ids)),
            eos_token_id=eos_token_id,
        )

        self.word_embedding = VocabParallelEmbedding(
            num_embeddings,
            embedding_dim,
            tp_rank=tp_rank,
            tp_size=tp_size,
            tp_group=tp_group,
            params_dtype=params_dtype,
        )
        self.oe_tables = nn.ParameterList(
            [
                Parameter(
                    torch.empty(
                        (fragment.modulus, fragment.feature_width),
                        dtype=dtype,
                    ),
                    requires_grad=False,
                )
                for fragment in self.spec.fragments
            ]
        )
        self.projection = Parameter(
            torch.empty(
                (self.spec.local_width, embedding_dim),
                dtype=dtype,
            ),
            requires_grad=False,
        )
        self._table_loaded = [False] * len(self.spec.fragments)
        self._projection_loaded = [False] * len(self.spec.fragments)

    def load_weight(self, weight_name: str, loaded_weight: torch.Tensor) -> bool:
        """Load one word, OE table, or projection checkpoint tensor.

        Returns:
            ``True`` when this layer owns the weight name. Non-local OE branches
            also return ``True`` because skipping them is intentional.
        """
        if weight_name.endswith("embed_tokens.weight") and not (
            "oe_embed_tokens" in weight_name
            or "ngram_embeddings.embedders" in weight_name
        ):
            self.word_embedding.weight_loader(
                self.word_embedding.weight,
                loaded_weight,
            )
            return True

        table_branch = _branch_index(weight_name, projection=False)
        if table_branch is not None:
            for local_index, fragment in enumerate(self.spec.fragments):
                if fragment.branch_id != table_branch:
                    continue
                expected = (fragment.modulus, self.spec.branch_width)
                if tuple(loaded_weight.shape) != expected:
                    raise ValueError(
                        f"OE table branch {table_branch} has shape "
                        f"{tuple(loaded_weight.shape)}, expected {expected}"
                    )
                self.oe_tables[local_index].data.copy_(
                    loaded_weight[
                        :,
                        fragment.feature_begin : fragment.feature_end,
                    ]
                )
                self._table_loaded[local_index] = True
            return True

        projection_branch = _branch_index(weight_name, projection=True)
        if projection_branch is not None:
            if tuple(loaded_weight.shape) == (
                self.embedding_dim,
                self.spec.branch_width,
            ):
                branch_projection = loaded_weight.t()
            elif tuple(loaded_weight.shape) == (
                self.spec.branch_width,
                self.embedding_dim,
            ):
                branch_projection = loaded_weight
            else:
                raise ValueError(
                    f"OE projection branch {projection_branch} has shape "
                    f"{tuple(loaded_weight.shape)}, expected "
                    f"({self.embedding_dim}, {self.spec.branch_width}) or "
                    f"({self.spec.branch_width}, {self.embedding_dim})"
                )
            output_begin = 0
            for local_index, fragment in enumerate(self.spec.fragments):
                output_end = output_begin + fragment.feature_width
                if fragment.branch_id == projection_branch:
                    self.projection.data[output_begin:output_end].copy_(
                        branch_projection[fragment.feature_begin : fragment.feature_end]
                    )
                    self._projection_loaded[local_index] = True
                output_begin = output_end
            return True
        return False

    def validate_loaded_weights(self) -> None:
        """Fail startup when any rank-local projection fragment is missing."""
        missing_tables = [
            fragment.branch_id
            for fragment, loaded in zip(
                self.spec.fragments,
                self._table_loaded,
            )
            if not loaded
        ]
        missing_projections = [
            fragment.branch_id
            for fragment, loaded in zip(
                self.spec.fragments,
                self._projection_loaded,
            )
            if not loaded
        ]
        if missing_tables or missing_projections:
            raise RuntimeError(
                "missing LongCat OE weights: "
                f"tables={missing_tables}, projections={missing_projections}"
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        request_token_history: RequestTokenHistoryView | None,
    ) -> torch.Tensor:
        if input_ids.numel() == 0:
            return self.word_embedding(input_ids)
        if request_token_history is None:
            raise RuntimeError("LongCat OE forward requires request token history")
        view = request_token_history
        if view.row_offset != 0:
            raise NotImplementedError(
                "LongCat OE requires the complete packed input batch"
            )
        word_partial = self.word_embedding(input_ids, reduce_results=False)
        activation = append_packed_lookup_(
            input_ids,
            view.input_start_offsets,
            view.req_pool_indices,
            view.active_request_mask,
            view.history_token_ids,
            view.committed_lengths,
            tuple(self.oe_tables),
            spec=self.spec,
            enable_pdl=pdl_enabled(),
        )
        local_partial = project_add_word_(
            word_partial,
            activation,
            self.projection,
            scale=self.spec.scale,
        )
        if self.tp_size > 1:
            local_partial = all_reduce(local_partial, self.tp_group)
        return local_partial

    @property
    def weight(self) -> Parameter:
        """Expose the tied regular word-embedding weight."""
        return self.word_embedding.weight

    @weight.setter
    def weight(self, value: Parameter) -> None:
        self.word_embedding.weight = value


__all__ = [
    "LongCatOverEmbedding",
    "resolve_longcat_oe_hyperparameters",
    "resolve_longcat_oe_spec",
]
