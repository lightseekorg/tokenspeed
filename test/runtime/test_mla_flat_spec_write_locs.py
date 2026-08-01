"""FlatKV write locations for MLA under speculative target verify.

Plain decode writes one latent row per request. Target verify decodes a whole
window per request and must write every row of it -- at positions
``seq-N .. seq-1``, flattened request-major to match the query layout the
verify read path builds. Getting the count wrong either trips the
select_out_cache_loc guard or, worse, folds N tokens' KV into one slot.
"""

from __future__ import annotations

import pytest
import torch

from tokenspeed.runtime.execution.forward_batch_info import ForwardMode
from tokenspeed.runtime.layers.attention.backends.mla import (
    MLAAttnBackend,
    MLADecodeMetadata,
)

PAGE = 64


def _backend(*, spec_num_tokens: int = 1, is_draft: bool = False) -> MLAAttnBackend:
    backend = MLAAttnBackend.__new__(MLAAttnBackend)
    backend.spec_num_tokens = spec_num_tokens
    backend.is_draft = is_draft
    backend.draft_block_decode = False
    backend.device = torch.device("cpu")
    backend._flat_bound = True
    backend._flat_contract_bound = True
    backend.max_context_len = 4096
    backend.max_num_pages = 8
    backend.decode_cuda_graph_metadata = {}
    return backend


def _table(rows: int = 2, cols: int = 8) -> torch.Tensor:
    """Page ids offset from 1 so no live row collides with the null page 0."""
    return (torch.arange(rows * cols, dtype=torch.int32) + 1).view(rows, cols)


# --------------------------------------------------------------------------
# Write-location math
# --------------------------------------------------------------------------


def test_plain_decode_writes_one_location_per_request() -> None:
    table = _table()
    seq_lens = torch.tensor([70, 130], dtype=torch.int32)

    locs = MLAAttnBackend._flat_decode_out_cache_loc(
        table, seq_lens, batch_size=2, flat_page_size=PAGE
    )

    assert locs.shape == (2,)
    # req0: pos 69 -> page col 1 (id 2), offset 5. req1: pos 129 -> col 2 (id 11), offset 1.
    assert locs.tolist() == [2 * PAGE + 5, 11 * PAGE + 1]


def test_verify_writes_the_whole_window_request_major() -> None:
    table = _table()
    seq_lens = torch.tensor([70, 130], dtype=torch.int32)

    locs = MLAAttnBackend._flat_decode_out_cache_loc(
        table, seq_lens, batch_size=2, flat_page_size=PAGE, q_len_per_req=4
    )

    assert locs.shape == (8,)
    # Request-major: all of req0's window, then all of req1's.
    expected = []
    for row, seq in enumerate((70, 130)):
        for pos in range(seq - 4, seq):
            page = int(table[row, pos // PAGE])
            expected.append(page * PAGE + pos % PAGE)
    assert locs.tolist() == expected


def test_verify_window_ends_at_the_last_token() -> None:
    """The final window row must equal what plain decode would have written."""
    table = _table()
    seq_lens = torch.tensor([70, 130], dtype=torch.int32)

    single = MLAAttnBackend._flat_decode_out_cache_loc(
        table, seq_lens, batch_size=2, flat_page_size=PAGE
    )
    window = MLAAttnBackend._flat_decode_out_cache_loc(
        table, seq_lens, batch_size=2, flat_page_size=PAGE, q_len_per_req=4
    )
    assert window.view(2, 4)[:, -1].tolist() == single.tolist()


def test_window_locations_are_distinct() -> None:
    """Folding the window into one slot is the failure this guards."""
    table = _table(rows=1)
    locs = MLAAttnBackend._flat_decode_out_cache_loc(
        table,
        torch.tensor([100], dtype=torch.int32),
        batch_size=1,
        flat_page_size=PAGE,
        q_len_per_req=8,
    )
    assert len(set(locs.tolist())) == 8


def test_window_spanning_a_page_boundary_follows_the_table() -> None:
    """Positions either side of a page edge must resolve to different pages."""
    table = _table(rows=1)
    # seq 66 with a window of 4 covers 62,63 (page col 0) and 64,65 (col 1).
    locs = MLAAttnBackend._flat_decode_out_cache_loc(
        table,
        torch.tensor([66], dtype=torch.int32),
        batch_size=1,
        flat_page_size=PAGE,
        q_len_per_req=4,
    )
    page0, page1 = int(table[0, 0]), int(table[0, 1])
    assert locs.tolist() == [
        page0 * PAGE + 62,
        page0 * PAGE + 63,
        page1 * PAGE + 0,
        page1 * PAGE + 1,
    ]


def test_short_sequences_clamp_instead_of_going_negative() -> None:
    table = _table(rows=1)
    locs = MLAAttnBackend._flat_decode_out_cache_loc(
        table,
        torch.tensor([2], dtype=torch.int32),
        batch_size=1,
        flat_page_size=PAGE,
        q_len_per_req=4,
    )
    assert all(loc >= 0 for loc in locs.tolist())


def test_out_buffer_is_filled_in_place() -> None:
    """Graph replay writes into the recorded buffer; a fresh tensor would be lost."""
    table = _table()
    seq_lens = torch.tensor([70, 130], dtype=torch.int32)
    buf = torch.zeros(2 * 4, dtype=torch.int64)

    returned = MLAAttnBackend._flat_decode_out_cache_loc(
        table,
        seq_lens,
        batch_size=2,
        flat_page_size=PAGE,
        out=buf,
        q_len_per_req=4,
    )
    assert returned.data_ptr() == buf.data_ptr()
    expected = MLAAttnBackend._flat_decode_out_cache_loc(
        table, seq_lens, batch_size=2, flat_page_size=PAGE, q_len_per_req=4
    )
    assert buf.tolist() == expected.tolist()


# --------------------------------------------------------------------------
# Window-width derivation
# --------------------------------------------------------------------------


def test_target_verify_decode_uses_the_full_window() -> None:
    backend = _backend(spec_num_tokens=8)
    assert backend._verify_q_len(ForwardMode.DECODE) == 8
    assert backend._graph_flat_q_len() == 8


def test_prefill_uses_a_single_location() -> None:
    """Extend tokens go through the extend path, not the verify window."""
    backend = _backend(spec_num_tokens=8)
    assert backend._verify_q_len(ForwardMode.EXTEND) == 1


def test_draft_never_takes_the_verify_window() -> None:
    """A chaining draft owns its own per-step write locations."""
    backend = _backend(spec_num_tokens=8, is_draft=True)
    assert backend._verify_q_len(ForwardMode.DECODE) == 1
    assert backend._graph_flat_q_len() == 1


def test_non_speculative_decode_is_unchanged() -> None:
    backend = _backend(spec_num_tokens=1)
    assert backend._verify_q_len(ForwardMode.DECODE) == 1
    assert backend._graph_flat_q_len() == 1


# --------------------------------------------------------------------------
# Mixed batches skip whole windows
# --------------------------------------------------------------------------


def test_mixed_batch_skips_whole_windows_not_rows() -> None:
    """num_extends counts requests; the loc buffer is strided by the window."""
    backend = _backend(spec_num_tokens=4)
    locs = torch.arange(3 * 4, dtype=torch.int64)
    backend.forward_decode_metadata = MLADecodeMetadata(
        num_extends=1,
        page_table=torch.zeros((3, 8), dtype=torch.int32),
        seq_lens=torch.zeros(3, dtype=torch.int32),
        flat_out_cache_loc=locs,
        flat_q_len_per_req=4,
    )

    selected = backend.select_out_cache_loc(
        layer=None, out_cache_loc=torch.zeros(8), forward_mode=ForwardMode.DECODE
    )
    # One extend request consumes a whole 4-wide window.
    assert selected.tolist() == list(range(4, 12))


def test_non_spec_mixed_batch_still_skips_single_rows() -> None:
    backend = _backend(spec_num_tokens=1)
    locs = torch.arange(3, dtype=torch.int64)
    backend.forward_decode_metadata = MLADecodeMetadata(
        num_extends=1,
        page_table=torch.zeros((3, 8), dtype=torch.int32),
        seq_lens=torch.zeros(3, dtype=torch.int32),
        flat_out_cache_loc=locs,
        flat_q_len_per_req=1,
    )
    selected = backend.select_out_cache_loc(
        layer=None, out_cache_loc=torch.zeros(2), forward_mode=ForwardMode.DECODE
    )
    assert selected.tolist() == [1, 2]


def test_count_mismatch_is_still_caught() -> None:
    backend = _backend(spec_num_tokens=4)
    backend.forward_decode_metadata = MLADecodeMetadata(
        num_extends=0,
        page_table=torch.zeros((2, 8), dtype=torch.int32),
        seq_lens=torch.zeros(2, dtype=torch.int32),
        flat_out_cache_loc=torch.arange(8, dtype=torch.int64),
        flat_q_len_per_req=4,
    )
    with pytest.raises(RuntimeError, match="write locations cover"):
        backend.select_out_cache_loc(
            layer=None,
            out_cache_loc=torch.zeros(2),
            forward_mode=ForwardMode.DECODE,
        )


# --------------------------------------------------------------------------
# Graph buffer sizing
# --------------------------------------------------------------------------


def test_graph_loc_buffer_is_sized_for_the_window() -> None:
    backend = _backend(spec_num_tokens=8)
    backend.init_cuda_graph_state(max_bs=4)
    assert backend.decode_cuda_graph_flat_out_cache_loc.shape == (32,)


def test_graph_loc_buffer_is_unexpanded_without_spec() -> None:
    backend = _backend(spec_num_tokens=1)
    backend.init_cuda_graph_state(max_bs=4)
    assert backend.decode_cuda_graph_flat_out_cache_loc.shape == (4,)


def test_capture_records_the_window_width_on_the_metadata() -> None:
    backend = _backend(spec_num_tokens=8)
    backend.init_cuda_graph_state(max_bs=4)
    backend.init_forward_metadata_capture_cuda_graph(
        bs=2,
        req_pool_indices=torch.tensor([0, 1]),
        seq_lens=torch.tensor([100, 200], dtype=torch.int32),
        forward_mode=ForwardMode.DECODE,
    )
    metadata = backend.decode_cuda_graph_metadata[2]
    assert metadata.flat_q_len_per_req == 8
    assert metadata.flat_out_cache_loc.shape == (16,)


def test_capture_clamps_seq_lens_to_the_window() -> None:
    """A request shorter than the window would resolve locations before its start."""
    backend = _backend(spec_num_tokens=8)
    backend.init_cuda_graph_state(max_bs=2)
    backend.init_forward_metadata_capture_cuda_graph(
        bs=2,
        req_pool_indices=torch.tensor([0, 1]),
        seq_lens=torch.tensor([3, 200], dtype=torch.int32),
        forward_mode=ForwardMode.DECODE,
    )
    assert backend.decode_cuda_graph_metadata[2].seq_lens.tolist() == [8, 200]


def test_draft_is_refused_on_the_flat_graph_path() -> None:
    backend = _backend(spec_num_tokens=8, is_draft=True)
    backend.init_cuda_graph_state(max_bs=2)
    with pytest.raises(NotImplementedError, match="draft worker"):
        backend.init_forward_metadata_capture_cuda_graph(
            bs=2,
            req_pool_indices=torch.tensor([0, 1]),
            seq_lens=torch.tensor([100, 200], dtype=torch.int32),
            forward_mode=ForwardMode.DECODE,
        )


def test_backend_declares_block_decode_capability() -> None:
    """The flat startup gate reads this off the class."""
    assert MLAAttnBackend.flat_block_decode_capable is True
