# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import sys

import pytest
import torch
from tokenspeed_kernel.ops.attention.dsv4.cuda import (
    has_persistent_topk,
    persistent_topk,
)


@pytest.mark.skipif(not (torch.cuda.is_available()), reason=("CUDA is required"))
class TestDeepseekV4AttentionOpsTest:
    def _assert_persistent_topk_matches_torch(
        self,
        logits: torch.Tensor,
        lengths: torch.Tensor,
        output: torch.Tensor,
        topk: int,
    ) -> None:
        for row_idx, raw_len in enumerate(lengths.cpu().tolist()):
            row_output = output[row_idx].cpu()
            if raw_len <= topk:
                expected = torch.full((topk,), -1, dtype=torch.int32)
                if raw_len > 0:
                    expected[:raw_len] = torch.arange(raw_len, dtype=torch.int32)
                assert torch.equal(row_output, expected)
                continue

            expected = (
                torch.topk(
                    logits[row_idx, :raw_len],
                    k=topk,
                    dim=-1,
                    sorted=False,
                )
                .indices.to(torch.int32)
                .sort()
                .values.cpu()
            )
            assert torch.equal(row_output.sort().values, expected)

    def test_persistent_topk_matches_torch_across_decode_medium_large_paths(self):
        if not has_persistent_topk():
            pytest.skip("DeepSeek V4 persistent top-k op is not available")

        torch.manual_seed(6790)
        device = torch.device("cuda")
        topk = 512
        lengths = torch.tensor(
            [0, 7, 513, 9000, 33000], device=device, dtype=torch.int32
        )
        stride = 33024
        logits = torch.randn(
            (lengths.numel(), stride), device=device, dtype=torch.float32
        )
        for row_idx, raw_len in enumerate(lengths.cpu().tolist()):
            if raw_len < stride:
                logits[row_idx, raw_len:] = 1.0e6
        output = torch.full(
            (lengths.numel(), topk), -77, device=device, dtype=torch.int32
        )

        persistent_topk(
            logits,
            lengths,
            output,
            topk,
            int(lengths.max().item()),
            enable_pdl=False,
            q_len_per_req=1,
        )
        torch.cuda.synchronize()

        self._assert_persistent_topk_matches_torch(logits, lengths, output, topk)

    def test_persistent_topk_noncluster_fallback_matches_torch(self):
        if not has_persistent_topk():
            pytest.skip("DeepSeek V4 persistent top-k op is not available")

        torch.manual_seed(6791)
        device = torch.device("cuda")
        topk = 512
        num_rows = 36
        stride = 65568
        # More than 32 rows bypasses clusters on capable GPUs. The long live
        # row exercises the same single-CTA streaming kernel used when cluster
        # launch is unsupported; short rows verify graph-padded tail handling.
        lengths = torch.tensor(
            [0, 17, 513, 9000, 33000, 65537]
            + [520 + (idx % 24) for idx in range(num_rows - 6)],
            device=device,
            dtype=torch.int32,
        )
        logits = torch.randn((num_rows, stride), device=device, dtype=torch.float32)
        for row_idx, raw_len in enumerate(lengths.cpu().tolist()):
            if raw_len < stride:
                logits[row_idx, raw_len:] = 1.0e6
        output = torch.full((num_rows, topk), -77, device=device, dtype=torch.int32)

        persistent_topk(
            logits,
            lengths,
            output,
            topk,
            stride,
            enable_pdl=False,
            q_len_per_req=1,
        )
        torch.cuda.synchronize()

        self._assert_persistent_topk_matches_torch(logits, lengths, output, topk)

    def test_persistent_topk_cluster_path_matches_torch_for_all_supported_k(
        self, subtests
    ):
        if not has_persistent_topk():
            pytest.skip("DeepSeek V4 persistent top-k op is not available")

        torch.manual_seed(6792)
        device = torch.device("cuda")
        stride = 262144

        for topk in (512, 1024, 2048):
            with subtests.test(topk=topk):
                lengths = torch.tensor(
                    [topk + 1, 8192, 65537, 131071, 262143],
                    device=device,
                    dtype=torch.int32,
                )
                logits = torch.randn(
                    (lengths.numel(), stride),
                    device=device,
                    dtype=torch.float32,
                )
                output = torch.full(
                    (lengths.numel(), topk),
                    -77,
                    device=device,
                    dtype=torch.int32,
                )

                persistent_topk(
                    logits,
                    lengths,
                    output,
                    topk,
                    int(lengths.max().item()),
                    enable_pdl=False,
                    q_len_per_req=1,
                )
                torch.cuda.synchronize()

                self._assert_persistent_topk_matches_torch(
                    logits, lengths, output, topk
                )

    def test_persistent_topk_cluster_pdl_is_cuda_graph_capture_safe(self, subtests):
        if not has_persistent_topk():
            pytest.skip("DeepSeek V4 persistent top-k op is not available")

        torch.manual_seed(6794)
        device = torch.device("cuda")
        topk = 2048

        for stride, raw_lengths in (
            (131072, [1024, 8192, 65537, 131071]),
            (262144, [1024, 8192, 131073, 262143]),
        ):
            with subtests.test(stride=stride):
                lengths = torch.tensor(raw_lengths, device=device, dtype=torch.int32)
                logits = torch.randn(
                    (lengths.numel(), stride),
                    device=device,
                    dtype=torch.float32,
                )
                output = torch.full(
                    (lengths.numel(), topk),
                    -77,
                    device=device,
                    dtype=torch.int32,
                )

                persistent_topk(
                    logits,
                    lengths,
                    output,
                    topk,
                    stride,
                    enable_pdl=True,
                    q_len_per_req=1,
                )
                torch.cuda.synchronize()

                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    persistent_topk(
                        logits,
                        lengths,
                        output,
                        topk,
                        stride,
                        enable_pdl=True,
                        q_len_per_req=1,
                    )
                for _ in range(3):
                    graph.replay()
                torch.cuda.synchronize()

                self._assert_persistent_topk_matches_torch(
                    logits, lengths, output, topk
                )

    def test_persistent_topk_cluster_specialization_boundaries(self, subtests):
        if not has_persistent_topk():
            pytest.skip("DeepSeek V4 persistent top-k op is not available")

        torch.manual_seed(6795)
        device = torch.device("cuda")
        topk = 2048

        # Exercise both sides of the register-cluster and cluster-16 ceilings,
        # including the portable cluster-8 fallback above 256K.
        for max_seq_len in (65537, 131072, 131073, 262144, 262145):
            with subtests.test(max_seq_len=max_seq_len):
                actual_len = max_seq_len - 1
                lengths = torch.tensor([actual_len], device=device, dtype=torch.int32)
                logits = torch.randn(
                    (1, max_seq_len), device=device, dtype=torch.float32
                )
                logits[:, actual_len:] = 1.0e6
                output = torch.full((1, topk), -77, device=device, dtype=torch.int32)

                persistent_topk(
                    logits,
                    lengths,
                    output,
                    topk,
                    max_seq_len,
                    enable_pdl=False,
                    q_len_per_req=1,
                )
                torch.cuda.synchronize()

                self._assert_persistent_topk_matches_torch(
                    logits, lengths, output, topk
                )

    def test_persistent_topk_verify_mode_uses_causal_request_lengths(self):
        if not has_persistent_topk():
            pytest.skip("DeepSeek V4 persistent top-k op is not available")

        torch.manual_seed(6793)
        device = torch.device("cuda")
        topk = 512
        q_len_per_req = 4
        request_lengths = torch.tensor(
            [70000, 131071], device=device, dtype=torch.int32
        )
        effective_lengths = torch.tensor(
            [
                int(raw_len) - (q_len_per_req - 1) + query_idx
                for raw_len in request_lengths.cpu().tolist()
                for query_idx in range(q_len_per_req)
            ],
            device=device,
            dtype=torch.int32,
        )
        stride = 131072
        logits = torch.randn(
            (effective_lengths.numel(), stride),
            device=device,
            dtype=torch.float32,
        )
        output = torch.full(
            (effective_lengths.numel(), topk),
            -77,
            device=device,
            dtype=torch.int32,
        )

        persistent_topk(
            logits,
            request_lengths,
            output,
            topk,
            int(request_lengths.max().item()),
            q_len_per_req,
            enable_pdl=False,
        )
        torch.cuda.synchronize()

        self._assert_persistent_topk_matches_torch(
            logits, effective_lengths, output, topk
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
