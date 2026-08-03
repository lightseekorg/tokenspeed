from __future__ import annotations

import os
import sys
import unittest
from types import SimpleNamespace

import torch

# CI Registration (parsed via AST, runtime no-op)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ci_system.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, suite="runtime-1gpu")

from tokenspeed.runtime.layers.attention.backends.tokenspeed_mla import (
    CuteDSLMLABackend,
)
from tokenspeed.runtime.layers.attention.backends.trtllm_mla import TRTLLMMLABackend

BACKENDS = (CuteDSLMLABackend, TRTLLMMLABackend)


def _stub(page_size: int, req_to_page_token_unit: int) -> SimpleNamespace:
    return SimpleNamespace(
        page_size=page_size,
        req_to_page_token_unit=req_to_page_token_unit,
        device=torch.device("cpu"),
    )


class DraftBlockTableUnitExpansionTest(unittest.TestCase):
    """MLA kernel block tables must expand logical scheduler page ids.

    The drafter's req_to_page is allocated in logical scheduler pages
    (Kimi-K3 LCM: 128 tokens) while the MLA decode kernels index the pool in
    their own smaller pages (64). Copying ids verbatim makes the kernel read
    rows the pool writes never touched (draft KV reads all zeros), which
    collapsed EAGLE3 draft logits.
    """

    def test_expands_logical_pages_when_units_differ(self):
        req_to_page = torch.tensor([[25, 51, 0], [7, 0, 0]], dtype=torch.int32)
        req_pool_indices = torch.tensor([0, 1])
        seq_lens = torch.tensor([98, 10])
        for cls in BACKENDS:
            table = cls._create_block_kv_indices(
                _stub(page_size=64, req_to_page_token_unit=128),
                2,
                6,
                req_pool_indices,
                seq_lens,
                req_to_page,
            )
            expected = torch.tensor(
                [[50, 51, 102, 103, 0, 1], [14, 15, 0, 1, 0, 1]],
                dtype=torch.int32,
            )
            self.assertTrue(
                torch.equal(table, expected), f"{cls.__name__}: {table.tolist()}"
            )

    def test_identity_copy_when_units_match(self):
        req_to_page = torch.tensor([[25, 51, 0], [7, 0, 0]], dtype=torch.int32)
        req_pool_indices = torch.tensor([0, 1])
        seq_lens = torch.tensor([98, 10])
        for cls in BACKENDS:
            table = cls._create_block_kv_indices(
                _stub(page_size=64, req_to_page_token_unit=64),
                2,
                4,
                req_pool_indices,
                seq_lens,
                req_to_page,
            )
            expected = torch.tensor([[25, 51, 0, 0], [7, 0, 0, 0]], dtype=torch.int32)
            self.assertTrue(
                torch.equal(table, expected), f"{cls.__name__}: {table.tolist()}"
            )

    def test_expansion_truncates_to_max_blocks(self):
        req_to_page = torch.tensor([[3, 5]], dtype=torch.int32)
        table = CuteDSLMLABackend._create_block_kv_indices(
            _stub(page_size=64, req_to_page_token_unit=128),
            1,
            3,
            torch.tensor([0]),
            torch.tensor([150]),
            req_to_page,
        )
        expected = torch.tensor([[6, 7, 10]], dtype=torch.int32)
        self.assertTrue(torch.equal(table, expected), table.tolist())


if __name__ == "__main__":
    unittest.main()
