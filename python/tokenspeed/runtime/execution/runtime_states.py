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
"""Runtime state tensors shared by the model executor."""

import torch


class RuntimeStates:
    """Own runtime state tensors keyed by request-pool index."""

    def __init__(
        self,
        req_pool_size: int,
        vocab_size: int,
        output_length: int,
        device: str = "cuda",
    ):
        self.device = device
        self.vocab_size = vocab_size

        self.valid_cache_lengths = torch.zeros(
            req_pool_size + 1, dtype=torch.int32, device=device
        )
        # Resolve input ids from here when overlap scheduling.
        self.future_input_map = torch.empty(
            (req_pool_size + 1, output_length), dtype=torch.int32, device=device
        )
        self.remote_spec_candidate_ready = torch.zeros(
            req_pool_size + 1, dtype=torch.bool, device=device
        )

    def update_valid_cache_length(
        self, req_pool_indices: torch.Tensor, increment_lengths: torch.Tensor
    ) -> None:
        self.valid_cache_lengths.index_add_(0, req_pool_indices, increment_lengths)

    def reset_states(
        self,
        extend_request_pool_indices: torch.Tensor,
        extend_prefix_lens: torch.Tensor,
    ) -> None:
        self.valid_cache_lengths[extend_request_pool_indices] = extend_prefix_lens
        self.remote_spec_candidate_ready[extend_request_pool_indices] = False

    def write_remote_spec_candidate_ids(
        self, req_pool_idx: int, candidate_ids: list[int]
    ) -> None:
        width = self.future_input_map.shape[1]
        if len(candidate_ids) != width:
            raise RuntimeError(
                f"remote spec candidate width mismatch: got {len(candidate_ids)}, expected {width}"
            )
        ids = torch.tensor(
            candidate_ids,
            dtype=torch.int32,
            device="cpu",
            pin_memory=True,
        ).to(self.device, non_blocking=True)
        self.future_input_map[req_pool_idx, :width] = ids
        self.remote_spec_candidate_ready[req_pool_idx] = True
