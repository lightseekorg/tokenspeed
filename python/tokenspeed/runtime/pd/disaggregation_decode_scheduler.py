# SPDX-License-Identifier: MIT AND Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 LightSeek Foundation
# SPDX-FileCopyrightText: Copyright contributors to the FluentLLM project
#
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

import logging
from typing import TYPE_CHECKING

import torch

from tokenspeed.runtime.cache.req_to_token_pool import ReqToTokenPoolInfo
from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.sampling.sampling_batch_info import SamplingBatchInfo

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from tokenspeed.runtime.configs.model_config import ModelConfig
    from tokenspeed.runtime.engine.schedule_batch import ScheduleBatch
    from tokenspeed.runtime.utils.server_args import ServerArgs


class DisaggregationDecodeScheduler:

    def prepare_for_prebuilt_extend(self: ScheduleBatch):
        """
        Prepare a prebuilt extend by populate metadata
        """

        self.forward_mode = ForwardMode.EXTEND
        reqs = self.reqs
        input_ids = [r.fill_ids[len(r.prefix_indices) :] for r in reqs]
        extend_num_tokens = sum(len(ids) for ids in input_ids)
        seq_lens = []
        pre_lens = []
        req_pool_indices = []

        # Pre-calculate total size
        total_size = sum(req.extend_input_len for req in reqs)
        out_cache_loc = torch.empty(total_size, dtype=torch.int64, device=self.device)

        # Fill the tensor in one pass
        offset = 0
        for i, req in enumerate(reqs):
            req_pool_indices.append(req.req_pool_idx)

            chunk = self.req_to_token_pool.req_to_token[req.req_pool_idx][
                : req.extend_input_len
            ]
            assert (
                offset + req.extend_input_len <= total_size
            ), f"Exceeds total size: offset={offset}, req.extend_input_len={req.extend_input_len}, total_size={total_size}"
            out_cache_loc[offset : offset + req.extend_input_len] = chunk
            offset += req.extend_input_len

            pre_len = len(req.prefix_indices)
            seq_len = len(req.origin_input_ids) + max(0, len(req.output_ids) - 1)
            seq_lens.append(seq_len)
            if len(req.output_ids) == 0:
                assert (
                    seq_len - pre_len == req.extend_input_len
                ), f"seq_len={seq_len}, pre_len={pre_len}, req.extend_input_len={req.extend_input_len}"

            req.cached_tokens += pre_len - req.already_computed
            req.already_computed = seq_len
            req.is_retracted = False
            pre_lens.append(pre_len)
            req.extend_logprob_start_len = 0

        extend_input_logprob_token_ids = None

        # Set fields
        self.input_ids = torch.tensor(
            sum(input_ids, []), dtype=torch.int32, device=self.device
        )
        self.req_pool_indices = torch.tensor(
            req_pool_indices, dtype=torch.int64, device=self.device
        )
        self.seq_lens = torch.tensor(seq_lens, dtype=torch.int64, device=self.device)
        self.seq_lens_cpu = torch.tensor(seq_lens, dtype=torch.int64, pin_memory=True)
        self.out_cache_loc = out_cache_loc
        self.seq_lens_sum = sum(seq_lens)

        if self.return_logprob:
            self.top_logprobs_nums = [r.top_logprobs_num for r in reqs]
            self.token_ids_logprobs = [r.token_ids_logprob for r in reqs]

        self.extend_num_tokens = extend_num_tokens
        self.prefix_lens = [len(r.prefix_indices) for r in reqs]
        self.extend_lens = [r.extend_input_len for r in reqs]
        self.extend_logprob_start_lens = [r.extend_logprob_start_len for r in reqs]
        self.extend_input_logprob_token_ids = extend_input_logprob_token_ids

        # Build sampling info
        self.sampling_info = SamplingBatchInfo.from_schedule_batch(
            self,
            self.model_config.vocab_size,
        )

    def process_prebuilt_extend(
        self: ScheduleBatch, server_args: ServerArgs, model_config: ModelConfig
    ):
        """Assign the buffered last input id to schedule batch"""
        self.output_ids = []
        for req in self.reqs:
            self.output_ids.append(req.output_ids[-1])
            alloced_len = len(req.fill_ids) - 1
            self.req_to_token_pool.set_req_pool_info(
                req.req_pool_idx,
                ReqToTokenPoolInfo(
                    alloced_len,
                    alloced_len,
                    self.req_to_token_pool.req_to_token[
                        req.req_pool_idx, :alloced_len
                    ].clone(),
                ),
            )
            # Cache the request in tree_cache with full sequence
            self.tree_cache.cache_unfinished_req(req)
            if req.grammar is not None:
                req.grammar.accept_token(req.output_ids[-1])
                req.grammar.finished = req.finished()
        self.output_ids = torch.tensor(
            self.output_ids, device=self.device, dtype=torch.int32
        )

        # Simulate the eagle run. We add mock data to hidden states for the
        # ease of implementation now meaning the first token will have acc rate
        # of 0.
        if not self.spec_algorithm.is_none():

            self.prealloc_for_draft_decode(is_disaggregation_decode=True)
            b, topk = len(self.reqs), server_args.speculative_eagle_topk
            assert topk == 1, "Tree attention is abandoned for now"
            last_verified_ids, token_list = self.output_ids, []

            for _ in range(server_args.speculative_num_steps):
                topk_index = torch.arange(
                    b * topk, device=self.device, dtype=torch.int32
                )
                topk_index = topk_index.reshape(b, topk)  # shape: (b, topk)
                token_list.append(topk_index)

            # local import to avoid circular importx
            from tokenspeed.runtime.spec_decode.eagle import EagleDraftOutput

            # use draft output to create verify input next
            spec_info = EagleDraftOutput(
                last_verified_ids=last_verified_ids,
                token_list=torch.cat(token_list, dim=-1),
            )
            self.spec_info = spec_info
