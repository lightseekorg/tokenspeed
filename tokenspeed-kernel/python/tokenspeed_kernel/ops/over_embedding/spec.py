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

"""Static LongCat over-embedding fragment ownership specifications."""

from __future__ import annotations

from dataclasses import dataclass

_PRO_PROFILE = "longcat-2.0"
_PRO_VOCAB_SIZE = 163840
_PRO_BRANCH_COUNT = 16
_PRO_BRANCH_WIDTH = 512
_PRO_HIDDEN_SIZE = 8192
_PRO_MAX_NGRAM_ORDER = 5
_PRO_MODULUS0 = 16476898


@dataclass(frozen=True)
class TableFragmentSpec:
    """Describe one compact, TP-local slice of an OE branch table.

    ``feature_begin`` is expressed in the full branch feature space. The GPU
    tensor itself is compact and therefore has row width ``feature_width``.
    """

    branch_id: int
    ngram_order: int
    modulus: int
    feature_begin: int
    feature_width: int

    def __post_init__(self) -> None:
        if self.branch_id < 0:
            raise ValueError(f"branch_id must be non-negative, got {self.branch_id}")
        if self.ngram_order < 2:
            raise ValueError(f"ngram_order must be at least 2, got {self.ngram_order}")
        if self.modulus <= 0:
            raise ValueError(f"modulus must be positive, got {self.modulus}")
        if self.feature_begin < 0:
            raise ValueError(
                f"feature_begin must be non-negative, got {self.feature_begin}"
            )
        if self.feature_width <= 0:
            raise ValueError(
                f"feature_width must be positive, got {self.feature_width}"
            )

    @property
    def feature_end(self) -> int:
        """Return the exclusive end in the full branch feature space."""
        return self.feature_begin + self.feature_width


@dataclass(frozen=True)
class OverEmbeddingSpec:
    """Describe one rank's immutable LongCat OE kernel geometry.

    ``eos_token_id`` is a sequence boundary rather than an ignored token. An
    EOS at the current position does not terminate the hash. When EOS appears
    in the look-back history, hashing stops before that EOS so tokens from the
    preceding sequence cannot affect the current n-gram. Explicit membership
    in ``ignored_token_ids`` remains effective at the current position.
    """

    profile: str
    tp_size: int
    rank: int
    vocab_size: int
    branch_count: int
    branch_width: int
    hidden_size: int
    max_ngram_order: int
    fragments: tuple[TableFragmentSpec, ...]
    ignored_token_ids: tuple[int, ...] = ()
    eos_token_id: int | None = None

    def __post_init__(self) -> None:
        if not self.profile:
            raise ValueError("profile must be non-empty")
        if self.tp_size <= 0:
            raise ValueError(f"tp_size must be positive, got {self.tp_size}")
        if not 0 <= self.rank < self.tp_size:
            raise ValueError(
                f"rank must be in [0, {self.tp_size}), got rank={self.rank}"
            )
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {self.vocab_size}")
        if self.branch_count <= 0:
            raise ValueError(f"branch_count must be positive, got {self.branch_count}")
        if self.branch_width <= 0:
            raise ValueError(f"branch_width must be positive, got {self.branch_width}")
        if self.hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {self.hidden_size}")
        if self.max_ngram_order < 2:
            raise ValueError(
                "max_ngram_order must be at least 2, got " f"{self.max_ngram_order}"
            )
        if not self.fragments:
            raise ValueError("fragments must be non-empty")
        if len(set(self.ignored_token_ids)) != len(self.ignored_token_ids):
            raise ValueError("ignored_token_ids must not contain duplicates")
        if any(
            token_id < 0 or token_id >= self.vocab_size
            for token_id in self.ignored_token_ids
        ):
            raise ValueError(f"ignored_token_ids must be inside [0, {self.vocab_size})")
        if self.eos_token_id is not None and not (
            0 <= self.eos_token_id < self.vocab_size
        ):
            raise ValueError(f"eos_token_id must be inside [0, {self.vocab_size})")

        intervals_by_branch: dict[int, list[tuple[int, int]]] = {}
        for fragment in self.fragments:
            if fragment.branch_id >= self.branch_count:
                raise ValueError(
                    f"fragment branch_id {fragment.branch_id} is outside "
                    f"[0, {self.branch_count})"
                )
            if fragment.ngram_order > self.max_ngram_order:
                raise ValueError(
                    f"fragment ngram_order {fragment.ngram_order} exceeds "
                    f"profile maximum {self.max_ngram_order}"
                )
            if fragment.feature_end > self.branch_width:
                raise ValueError(
                    "fragment feature range "
                    f"[{fragment.feature_begin}, {fragment.feature_end}) exceeds "
                    f"branch width {self.branch_width}"
                )
            intervals = intervals_by_branch.setdefault(fragment.branch_id, [])
            for begin, end in intervals:
                if fragment.feature_begin < end and begin < fragment.feature_end:
                    raise ValueError(
                        f"overlapping feature ranges for branch {fragment.branch_id}"
                    )
            intervals.append((fragment.feature_begin, fragment.feature_end))

    @property
    def history_lookback(self) -> int:
        """Number of stable tokens required before the current target step."""
        return self.max_ngram_order - 1

    @property
    def scale(self) -> int:
        """Divisor for word embedding plus every global OE branch."""
        return self.branch_count + 1

    @property
    def local_width(self) -> int:
        """Packed local activation width produced by the lookup kernel."""
        return sum(fragment.feature_width for fragment in self.fragments)

    @property
    def output_offsets(self) -> tuple[int, ...]:
        """Packed activation offsets derived from fragment tuple order."""
        offsets: list[int] = []
        offset = 0
        for fragment in self.fragments:
            offsets.append(offset)
            offset += fragment.feature_width
        return tuple(offsets)


def _modulus(modulus0: int, branch_id: int) -> int:
    return modulus0 + 2 * branch_id


def _ngram_order(branch_id: int) -> int:
    return branch_id // 4 + 2


def _full_fragment(
    *, branch_id: int, modulus0: int, branch_width: int
) -> TableFragmentSpec:
    return TableFragmentSpec(
        branch_id=branch_id,
        ngram_order=_ngram_order(branch_id),
        modulus=_modulus(modulus0, branch_id),
        feature_begin=0,
        feature_width=branch_width,
    )


def longcat_pro_tp8_spec(rank: int) -> OverEmbeddingSpec:
    """Return the official LongCat-2.0 TP8 ownership for ``rank``."""
    if not 0 <= rank < 8:
        raise ValueError(f"rank must be in [0, 8), got rank={rank}")
    branch0 = 2 * rank
    return OverEmbeddingSpec(
        profile=_PRO_PROFILE,
        tp_size=8,
        rank=rank,
        vocab_size=_PRO_VOCAB_SIZE,
        branch_count=_PRO_BRANCH_COUNT,
        branch_width=_PRO_BRANCH_WIDTH,
        hidden_size=_PRO_HIDDEN_SIZE,
        max_ngram_order=_PRO_MAX_NGRAM_ORDER,
        fragments=tuple(
            _full_fragment(
                branch_id=branch_id,
                modulus0=_PRO_MODULUS0,
                branch_width=_PRO_BRANCH_WIDTH,
            )
            for branch_id in (branch0, branch0 + 1)
        ),
    )


__all__ = [
    "OverEmbeddingSpec",
    "TableFragmentSpec",
    "longcat_pro_tp8_spec",
]
