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

"""CuTeDSL LongCat packed hash and compact-fragment lookup producer."""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cuda.bindings.driver import CUstream

_BLOCK_SIZE = 64


class CuteLongCatOEAppendPackedLookup:
    """Produce packed BF16 rows for variable-length request intervals."""

    def __init__(
        self,
        *,
        vocab_size: int,
        fragment_configs: tuple[tuple[int, int, int], ...],
        ignored_token_ids: tuple[int, ...] = (),
        eos_token_id: int | None = None,
        use_pdl: bool = False,
    ) -> None:
        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {vocab_size}")
        if len(fragment_configs) != 2:
            raise ValueError(
                f"local fragment count must be 2, got {len(fragment_configs)}"
            )
        for index, (ngram_order, modulus, feature_width) in enumerate(fragment_configs):
            if not 2 <= ngram_order <= 5:
                raise ValueError(
                    f"fragment {index} ngram_order must be in [2, 5], got {ngram_order}"
                )
            if modulus <= 0:
                raise ValueError(
                    f"fragment {index} modulus must be positive, got {modulus}"
                )
            if feature_width != 512:
                raise ValueError(
                    f"fragment {index} feature width must be 512, got {feature_width}"
                )
        if eos_token_id is not None and not 0 <= eos_token_id < vocab_size:
            raise ValueError(f"eos_token_id must be inside [0, {vocab_size})")

        self.ignored_token_ids = ignored_token_ids
        self.eos_token_id = -1 if eos_token_id is None else eos_token_id
        self.use_pdl = use_pdl
        self.ngram_order0, self.modulus0, self.feature_width0 = fragment_configs[0]
        self.ngram_order1, self.modulus1, self.feature_width1 = fragment_configs[1]

        self.output_offset0 = 0
        self.output_offset1 = self.feature_width0
        self.values_per_thread0 = self.feature_width0 // _BLOCK_SIZE
        self.values_per_thread1 = self.feature_width1 // _BLOCK_SIZE

        self.powers0 = tuple(
            pow(vocab_size, delta, self.modulus0) for delta in range(self.ngram_order0)
        )
        self.powers1 = tuple(
            pow(vocab_size, delta, self.modulus1) for delta in range(self.ngram_order1)
        )

    @cute.jit
    def __call__(
        self,
        input_ids: cute.Tensor,
        input_start_offsets: cute.Tensor,
        req_pool_indices: cute.Tensor,
        active_request_mask: cute.Tensor,
        history_token_ids: cute.Tensor,
        committed_lengths: cute.Tensor,
        table0: cute.Tensor,
        table1: cute.Tensor,
        out: cute.Tensor,
        stream: CUstream,
    ) -> None:
        token_count = cute.size(out, mode=[0])
        self.kernel(
            input_ids,
            input_start_offsets,
            req_pool_indices,
            active_request_mask,
            history_token_ids,
            committed_lengths,
            table0,
            table1,
            out,
        ).launch(
            grid=[token_count, 2, 1],
            block=[_BLOCK_SIZE, 1, 1],
            smem=8,
            stream=stream,
            use_pdl=self.use_pdl,
            min_blocks_per_mp=1,
        )

    @cute.jit
    def _find_request(
        self,
        input_start_offsets: cute.Tensor,
        token_index: cutlass.Int32,
    ) -> cutlass.Int32:
        lower = cutlass.Int32(0)
        upper = cutlass.Int32(cute.size(input_start_offsets, mode=[0]) - 1)
        while lower < upper:
            middle = (lower + upper) // 2
            if input_start_offsets[middle + 1] <= token_index:
                lower = middle + 1
            else:
                upper = middle
        return lower

    @cute.jit
    def _is_ignored(self, token: cutlass.Int32) -> cutlass.Boolean:
        ignored = cutlass.Boolean(False)
        for ignored_token_id in cutlass.range_constexpr(len(self.ignored_token_ids)):
            if token == self.ignored_token_ids[ignored_token_id]:
                ignored = cutlass.Boolean(True)
        return ignored

    @cute.jit
    def _hash_fragment(
        self,
        input_ids: cute.Tensor,
        history_token_ids: cute.Tensor,
        token_index: cutlass.Int32,
        position: cutlass.Int32,
        slot: cutlass.Int64,
        stable_len: cutlass.Int32,
        ngram_order: cutlass.Constexpr,
        modulus: cutlass.Constexpr,
        powers: cutlass.Constexpr,
    ) -> cutlass.Int32:
        invalid = cutlass.Boolean(False)
        boundary_reached = cutlass.Boolean(False)
        acc = cutlass.Int64(0)
        for delta in cutlass.range_constexpr(ngram_order):
            if delta <= position:
                token = input_ids[token_index - delta]
                if cutlass.const_expr(delta > 0):
                    if token == self.eos_token_id:
                        boundary_reached = cutlass.Boolean(True)
                if not boundary_reached:
                    ignored = self._is_ignored(token)
                    if ignored:
                        invalid = cutlass.Boolean(True)
                    else:
                        acc = acc + cutlass.Int64(token) * cutlass.Int64(powers[delta])
            else:
                stable_distance = delta - position
                if stable_distance <= stable_len:
                    token = history_token_ids[slot, stable_len - stable_distance]
                    if cutlass.const_expr(delta > 0):
                        if token == self.eos_token_id:
                            boundary_reached = cutlass.Boolean(True)
                    if not boundary_reached:
                        ignored = self._is_ignored(token)
                        if ignored:
                            invalid = cutlass.Boolean(True)
                        else:
                            acc = acc + cutlass.Int64(token) * cutlass.Int64(
                                powers[delta]
                            )

        row = cutlass.Int32(acc % cutlass.Int64(modulus))
        if invalid:
            row = cutlass.Int32(modulus - 1)
        return row

    @cute.kernel
    def kernel(
        self,
        input_ids: cute.Tensor,
        input_start_offsets: cute.Tensor,
        req_pool_indices: cute.Tensor,
        active_request_mask: cute.Tensor,
        history_token_ids: cute.Tensor,
        committed_lengths: cute.Tensor,
        table0: cute.Tensor,
        table1: cute.Tensor,
        out: cute.Tensor,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        token_index, local_fragment, _ = cute.arch.block_idx()

        if cutlass.const_expr(self.use_pdl):
            cute.arch.griddepcontrol_wait()

        smem = cutlass.utils.SmemAllocator()
        selected_row = smem.allocate_tensor(
            cutlass.Int32, cute.make_layout((1,)), byte_alignment=4
        )
        request_active = smem.allocate_tensor(
            cutlass.Int32, cute.make_layout((1,)), byte_alignment=4
        )

        if tidx == 0:
            selected_row[0] = cutlass.Int32(0)
            request = self._find_request(input_start_offsets, token_index)
            request_begin = input_start_offsets[request]
            position = token_index - request_begin
            slot = req_pool_indices[request]
            active = active_request_mask[request]
            request_active[0] = cutlass.Int32(0)
            stable_len = cutlass.Int32(0)
            if active:
                request_active[0] = cutlass.Int32(1)
                stable_len = committed_lengths[slot]
            if active and local_fragment == 0:
                history_token_ids[slot, stable_len + position] = input_ids[token_index]
                selected_row[0] = self._hash_fragment(
                    input_ids,
                    history_token_ids,
                    token_index,
                    position,
                    slot,
                    stable_len,
                    self.ngram_order0,
                    self.modulus0,
                    self.powers0,
                )
            if active and local_fragment == 1:
                selected_row[0] = self._hash_fragment(
                    input_ids,
                    history_token_ids,
                    token_index,
                    position,
                    slot,
                    stable_len,
                    self.ngram_order1,
                    self.modulus1,
                    self.powers1,
                )
        cute.arch.sync_threads()

        row = cutlass.Int64(selected_row[0])
        if local_fragment == 0:
            for value_index in cutlass.range_constexpr(self.values_per_thread0):
                column = tidx + value_index * _BLOCK_SIZE
                value = cutlass.BFloat16(0)
                if request_active[0] != 0:
                    value = table0[row, column]
                out[token_index, self.output_offset0 + column] = value
        if local_fragment == 1:
            for value_index in cutlass.range_constexpr(self.values_per_thread1):
                column = tidx + value_index * _BLOCK_SIZE
                value = cutlass.BFloat16(0)
                if request_active[0] != 0:
                    value = table1[row, column]
                out[token_index, self.output_offset1 + column] = value

        if cutlass.const_expr(self.use_pdl):
            cute.arch.sync_threads()
            cute.arch.griddepcontrol_launch_dependents()
