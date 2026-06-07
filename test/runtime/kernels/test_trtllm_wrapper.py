import unittest
from unittest.mock import patch

import torch


class TRTLLMWrapperTest(unittest.TestCase):
    def test_fast_topk_v2_decode_accepts_2d_lens(self):
        from tokenspeed_kernel.registry import error_fn
        from tokenspeed_kernel.thirdparty import trtllm

        if trtllm.fast_topk_v2 is None or trtllm.fast_topk_v2 is error_fn:
            self.skipTest("TRTLLM fast_topk_v2 is unavailable on this platform")

        captured = {}

        def fake_indexer_topk_decode(values, seq_lens, indices, next_n, topk):
            del values, indices
            captured["seq_lens"] = seq_lens
            captured["next_n"] = next_n
            captured["topk"] = topk

        with patch.object(
            torch.ops.trtllm,
            "indexer_topk_decode",
            fake_indexer_topk_decode,
            create=True,
        ):
            values = torch.empty((2, 4), dtype=torch.float32)
            seq_lens = torch.tensor([[3], [4]], dtype=torch.int64)
            indices = torch.empty((2, 2), dtype=torch.int32)

            trtllm.fast_topk_v2(
                values,
                seq_lens,
                indices,
                topk=2,
                next_n=1,
            )

        self.assertEqual(captured["next_n"], 1)
        self.assertEqual(captured["topk"], 2)
        self.assertEqual(captured["seq_lens"].dtype, torch.int32)
        self.assertEqual(captured["seq_lens"].dim(), 1)
        torch.testing.assert_close(
            captured["seq_lens"],
            torch.tensor([3, 4], dtype=torch.int32),
            atol=0,
            rtol=0,
        )

    def test_fast_topk_v2_prefill_uses_int32_row_offsets(self):
        from tokenspeed_kernel.registry import error_fn
        from tokenspeed_kernel.thirdparty import trtllm

        if trtllm.fast_topk_v2 is None or trtllm.fast_topk_v2 is error_fn:
            self.skipTest("TRTLLM fast_topk_v2 is unavailable on this platform")

        captured = {}

        def fake_indexer_topk_prefill(values, row_starts, row_ends, indices, topk):
            del values, indices
            captured["row_starts"] = row_starts
            captured["row_ends"] = row_ends
            captured["topk"] = topk

        with patch.object(
            torch.ops.trtllm,
            "indexer_topk_prefill",
            fake_indexer_topk_prefill,
            create=True,
        ):
            values = torch.empty((3, 4), dtype=torch.float32)
            seq_lens = torch.tensor([[1], [2]], dtype=torch.int64)
            indices = torch.empty((2, 2), dtype=torch.int32)

            trtllm.fast_topk_v2(
                values,
                seq_lens,
                indices,
                topk=2,
                next_n=2,
            )

        self.assertEqual(captured["topk"], 2)
        self.assertEqual(captured["row_starts"].dtype, torch.int32)
        self.assertEqual(captured["row_ends"].dtype, torch.int32)
        torch.testing.assert_close(
            captured["row_starts"],
            torch.tensor([0, 1], dtype=torch.int32),
            atol=0,
            rtol=0,
        )
        torch.testing.assert_close(
            captured["row_ends"],
            torch.tensor([1, 3], dtype=torch.int32),
            atol=0,
            rtol=0,
        )


if __name__ == "__main__":
    unittest.main()
