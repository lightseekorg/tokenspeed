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
    has_indexer_mxfp4_paged_gather,
    has_persistent_topk,
    indexer_mxfp4_paged_gather,
    persistent_topk,
)
from tokenspeed_kernel.ops.attention.dsv4.triton import (
    _dsv4_fused_csa_indexer_mxfp4_cache_kernel,
    _dsv4_fused_sparse_compress_cache_kernel,
    _dsv4_gather_launch_config,
    _dsv4_use_serial_four_block_indexer_q,
    dsv4_compact_compressed_slot_mapping,
    dsv4_compressed_slot_mapping,
    dsv4_fused_indexer_q_rope_hadamard_mxfp4,
)

ROPE_DIM = 64


class TestDeepseekV4AttentionOpsCpuValidationTest:
    def test_indexer_q_serial_four_block_selector_is_fail_closed(self, subtests):
        valid = {
            "num_tokens": 8192,
            "head_dim": 128,
            "rope_dim": 64,
            "quant_block": 32,
            "is_cuda": True,
            "is_hip": False,
            "capability": (10, 0),
            "shapes_valid": True,
            "dtypes_valid": True,
            "devices_valid": True,
            "inner_strides": (1, 1, 1, 1),
        }
        assert _dsv4_use_serial_four_block_indexer_q(**valid)
        invalid_overrides = (
            {"num_tokens": 8191},
            {"head_dim": 64},
            {"rope_dim": 32},
            {"quant_block": 64},
            {"is_cuda": False},
            {"is_hip": True},
            {"capability": (9, 0)},
            {"shapes_valid": False},
            {"dtypes_valid": False},
            {"devices_valid": False},
            {"inner_strides": (2, 1, 1, 1)},
        )
        for override in invalid_overrides:
            case = valid | override
            with subtests.test(override=override):
                assert not (_dsv4_use_serial_four_block_indexer_q(**case))

    def test_v4_table_kernels_do_not_specialize_runtime_geometry(self, subtests):
        cases = (
            (
                _dsv4_fused_sparse_compress_cache_kernel,
                ("block_table_stride", "block_table_width"),
            ),
            (
                _dsv4_fused_csa_indexer_mxfp4_cache_kernel,
                ("block_table_stride", "block_table_width"),
            ),
        )
        for kernel, names in cases:
            parameters = {parameter.name: parameter for parameter in kernel.params}
            for name in names:
                with subtests.test(kernel=kernel.__name__, name=name):
                    parameter = parameters[name]
                    assert not (parameter.is_constexpr)
                    assert parameter.do_not_specialize

    def test_sparse_prefill_gather_launch_config(self, subtests):
        cases = (
            (1, 0, (128, 4)),
            (1, 512, (128, 4)),
            (1, 513, (512, 1)),
            (1, 3072, (512, 1)),
            (1, 3073, (1024, 1)),
            (1, 6144, (1024, 1)),
            (1, 6145, (2048, 1)),
            (3, 6145, (1024, 1)),
            (4, 12288, (1024, 1)),
            (4, 12289, (2048, 1)),
            (8, 2048, (512, 1)),
            (8, 8192, (2048, 1)),
        )
        for num_reqs, max_rows, expected in cases:
            with subtests.test(num_reqs=num_reqs, max_rows=max_rows):
                assert (_dsv4_gather_launch_config(num_reqs, max_rows)) == (expected)


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

    def test_indexer_mxfp4_paged_gather_matches_paged_layout(self):
        if not has_indexer_mxfp4_paged_gather():
            pytest.skip("DeepSeek V4 paged MXFP4 gather op is not available")

        device = torch.device("cuda")
        block_size = 4
        value_bytes = 64
        scale_bytes = 4
        num_blocks = 3
        kv_cache = torch.zeros(
            num_blocks,
            block_size * (value_bytes + scale_bytes),
            device=device,
            dtype=torch.uint8,
        )

        value_rows = {}
        scale_rows = {}
        for block_idx in range(num_blocks):
            for row_idx in range(block_size):
                values = (
                    (
                        torch.arange(value_bytes, device=device, dtype=torch.int16)
                        + block_idx * 37
                        + row_idx * 11
                    )
                    .remainder(251)
                    .to(torch.uint8)
                )
                scales = torch.tensor(
                    [block_idx, row_idx, block_idx * 17 + row_idx, 200 + block_idx],
                    device=device,
                    dtype=torch.uint8,
                )
                value_base = row_idx * value_bytes
                scale_base = block_size * value_bytes + row_idx * scale_bytes
                kv_cache[block_idx, value_base : value_base + value_bytes].copy_(values)
                kv_cache[block_idx, scale_base : scale_base + scale_bytes].copy_(scales)
                value_rows[(block_idx, row_idx)] = values
                scale_rows[(block_idx, row_idx)] = scales

        block_table = torch.tensor([[2, 0], [1, 0]], device=device, dtype=torch.int32)
        cu_seq_lens = torch.tensor([0, 5, 7], device=device, dtype=torch.int32)
        values_out = torch.full(
            (8, value_bytes), 0xCC, device=device, dtype=torch.uint8
        )
        scales_out = torch.full(
            (8, scale_bytes), 0xDD, device=device, dtype=torch.uint8
        )

        indexer_mxfp4_paged_gather(
            kv_cache,
            values_out,
            scales_out,
            block_table,
            cu_seq_lens,
            block_size,
        )
        torch.cuda.synchronize()

        expected_plan = [
            (2, 0),
            (2, 1),
            (2, 2),
            (2, 3),
            (0, 0),
            (1, 0),
            (1, 1),
        ]
        expected_values = torch.stack([value_rows[item] for item in expected_plan])
        expected_scales = torch.stack([scale_rows[item] for item in expected_plan])
        assert torch.equal(values_out[:7].cpu(), expected_values.cpu())
        assert torch.equal(scales_out[:7].cpu(), expected_scales.cpu())
        assert torch.equal(
            values_out[7].cpu(), torch.full((64,), 0xCC, dtype=torch.uint8)
        )
        assert torch.equal(
            scales_out[7].cpu(), torch.full((4,), 0xDD, dtype=torch.uint8)
        )

    def test_indexer_q_serial_four_block_is_bitwise_equal_at_production_shape(self):
        if torch.cuda.get_device_capability() != (10, 0):
            pytest.skip("exact serial-four-block specialization requires SM100")
        torch.manual_seed(9125)
        device = torch.device("cuda")
        num_tokens = 8192
        num_heads = 64
        positions = torch.arange(num_tokens, device=device, dtype=torch.int64)
        cos_sin = torch.randn(num_tokens, ROPE_DIM, device=device, dtype=torch.float32)
        q = torch.randn(num_tokens, num_heads, 128, device=device, dtype=torch.bfloat16)
        weights = torch.randn(num_tokens, num_heads, device=device, dtype=torch.float32)

        baseline = dsv4_fused_indexer_q_rope_hadamard_mxfp4(
            index_q=q,
            positions=positions,
            cos_sin_cache=cos_sin,
            weights=weights,
            softmax_scale=0.25,
            head_scale=num_heads**-0.5,
            prefer_serial_four_block=False,
        )
        serial = dsv4_fused_indexer_q_rope_hadamard_mxfp4(
            index_q=q,
            positions=positions,
            cos_sin_cache=cos_sin,
            weights=weights,
            softmax_scale=0.25,
            head_scale=num_heads**-0.5,
            prefer_serial_four_block=True,
        )
        torch.cuda.synchronize()

        (baseline_packed, baseline_scales), baseline_weights = baseline
        (serial_packed, serial_scales), serial_weights = serial
        assert torch.equal(serial_packed, baseline_packed)
        assert torch.equal(serial_scales, baseline_scales)
        assert torch.equal(serial_weights, baseline_weights)

    def test_compressed_slot_mapping_matches_page_reference(self):
        device = torch.device("cuda")
        query_start_loc = torch.tensor([0, 3, 5], device=device, dtype=torch.int32)
        seq_lens = torch.tensor([8, 6], device=device, dtype=torch.int32)
        block_table = torch.tensor(
            [
                [10, 11, 12],
                [20, 21, 22],
            ],
            device=device,
            dtype=torch.int32,
        )
        out = torch.empty(8, device=device, dtype=torch.int64)

        actual = dsv4_compressed_slot_mapping(
            num_tokens=5,
            query_start_loc=query_start_loc,
            seq_lens=seq_lens,
            block_table=block_table,
            block_size=4,
            compress_ratio=4,
            out=out,
        )
        torch.cuda.synchronize()

        assert (actual.data_ptr()) == (out.data_ptr())
        assert torch.equal(
            actual.cpu(),
            torch.tensor([-1, -1, 10 * 4 + 1, -1, -1], dtype=torch.int64),
        )
        assert torch.equal(out[5:].cpu(), torch.full((3,), -1))

    def test_grouped_compressed_slot_mapping_matches_full_table_reference(self):
        device = torch.device("cuda")
        token_to_req_indices = torch.tensor(
            [0, 0, 0, 1, 1, -1], device=device, dtype=torch.int32
        )
        query_start_loc = torch.tensor([0, 3, 5], device=device, dtype=torch.int32)
        seq_lens = torch.tensor([264, 520], device=device, dtype=torch.int32)
        block_table = torch.tensor(
            [[10, 20, 21], [30, 40, -1]], device=device, dtype=torch.int32
        )
        is_valid_token = torch.tensor(
            [True, True, True, True, False, True], device=device, dtype=torch.bool
        )
        out = torch.full((8,), 1234, device=device, dtype=torch.int64)

        actual = dsv4_compact_compressed_slot_mapping(
            num_tokens=6,
            token_to_req_indices=token_to_req_indices,
            query_start_loc=query_start_loc,
            seq_lens=seq_lens,
            block_table=block_table,
            block_size=64,
            compress_ratio=4,
            block_table_base_offsets=None,
            is_valid_token=is_valid_token,
            out=out,
        )
        torch.cuda.synchronize()

        assert (actual.data_ptr()) == (out.data_ptr())
        assert torch.equal(
            actual.cpu(),
            torch.tensor([-1, -1, 20 * 64 + 1, -1, -1, -1]),
        )
        assert torch.equal(out[6:].cpu(), torch.full((2,), -1))

    def test_grouped_compressed_slot_mapping_compact_table_graph_replay(self):
        device = torch.device("cuda")
        token_to_req_indices = torch.tensor(
            [0, 0, 0, 1, 1, -1], device=device, dtype=torch.int32
        )
        query_start_loc = torch.tensor([0, 3, 5], device=device, dtype=torch.int32)
        seq_lens = torch.tensor([264, 520], device=device, dtype=torch.int32)
        block_table = torch.tensor(
            [[20, 21], [30, -1]], device=device, dtype=torch.int32
        )
        base_offsets = torch.tensor([1, 2], device=device, dtype=torch.int32)
        is_valid_token = torch.tensor(
            [True, True, True, True, False, True], device=device, dtype=torch.bool
        )
        out = torch.full((8,), 1234, device=device, dtype=torch.int64)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            dsv4_compact_compressed_slot_mapping(
                num_tokens=6,
                token_to_req_indices=token_to_req_indices,
                query_start_loc=query_start_loc,
                seq_lens=seq_lens,
                block_table=block_table,
                block_size=64,
                compress_ratio=4,
                block_table_base_offsets=base_offsets,
                is_valid_token=is_valid_token,
                out=out,
            )
        seq_lens.copy_(torch.tensor([268, 524], device=device, dtype=torch.int32))
        block_table[0, 0] = 22
        is_valid_token[4] = True
        graph.replay()
        torch.cuda.synchronize()

        assert torch.equal(
            out[:6].cpu(),
            torch.tensor([-1, -1, 22 * 64 + 2, -1, 30 * 64 + 2, -1]),
        )
        assert torch.equal(out[6:].cpu(), torch.full((2,), -1))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))
