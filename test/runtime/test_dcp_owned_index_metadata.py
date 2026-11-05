"""CPU checks of the owned-page plan's grouped, strided metadata view."""

import pytest
import torch

pytest.importorskip("tokenspeed_kernel_amd")
from tokenspeed_kernel_amd.ops.gfx950.attention.dsa.index_candidates import (
    _owned_query_group_metadata,
)


@pytest.mark.parametrize("group", [1, 4])
def test_grouped_owned_pages_are_a_live_view_of_each_groups_last_row(group):
    q = torch.empty(8, 32, 128, dtype=torch.float8_e4m3fn)
    pages = torch.arange(24, dtype=torch.int32).view(8, 3)
    lengths = torch.tensor([64, 64, 64, 65, 66, 66, 66, 66], dtype=torch.int32)
    table = _owned_query_group_metadata(q, pages, lengths, group)
    torch.testing.assert_close(table, pages[group - 1 :: group], atol=0, rtol=0)
    assert table.stride(0) == group * pages.stride(0)
    pages[group - 1, 0] = 987
    assert table[0, 0] == 987


def test_grouped_owned_pages_reject_incomplete_groups_and_nondense_plans():
    q = torch.empty(8, 32, 128, dtype=torch.float8_e4m3fn)
    pages = torch.empty(8, 4, dtype=torch.int32)
    lengths = torch.empty(8, dtype=torch.int32)
    with pytest.raises(ValueError, match="complete"):
        _owned_query_group_metadata(q[:7], pages[:7], lengths[:7], 4)
    with pytest.raises(ValueError, match="contiguous"):
        _owned_query_group_metadata(q, pages[:, ::2], lengths, 4)
