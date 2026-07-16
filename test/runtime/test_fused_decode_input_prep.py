from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
cache_loc_kernel = pytest.importorskip("tokenspeed.runtime.execution.cache_loc_kernel")


class _FakeKernelLauncher:
    def __init__(self) -> None:
        self.grid = None
        self.args = ()
        self.kwargs = {}

    def __getitem__(self, grid):
        self.grid = grid
        return self._launch

    def _launch(self, *args, **kwargs) -> None:
        self.args = args
        self.kwargs = kwargs


@pytest.mark.parametrize(
    ("mode_kwargs", "expected_dummy_slot", "expected_dummy_mode"),
    (
        pytest.param(
            {"use_dummy_cache_loc": True, "dummy_kv_slot": 23},
            23,
            True,
            id="group-keyed-dummy",
        ),
        pytest.param({}, 0, False, id="legacy"),
    ),
)
def test_fused_decode_wrapper_forwards_cache_loc_mode(
    monkeypatch,
    mode_kwargs,
    expected_dummy_slot,
    expected_dummy_mode,
):
    launcher = _FakeKernelLauncher()
    monkeypatch.setattr(
        cache_loc_kernel,
        "fused_decode_input_prep_kernel",
        launcher,
    )
    req_pool_indices = SimpleNamespace(shape=(2,))
    req_to_pages = SimpleNamespace(shape=(3, 7))

    cache_loc_kernel.fused_decode_input_prep(
        object(),
        object(),
        object(),
        req_pool_indices,
        object(),
        4,
        req_to_pages,
        16,
        **mode_kwargs,
    )

    assert launcher.grid == (2,)
    assert launcher.args[7] == expected_dummy_slot
    assert launcher.kwargs["USE_DUMMY_CACHE_LOC"] is expected_dummy_mode
    assert launcher.kwargs["max_pages"] == 7


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="fused decode input preparation requires CUDA",
)
@pytest.mark.parametrize(
    ("mode_kwargs", "expected_cache_locs"),
    (
        pytest.param(
            {"use_dummy_cache_loc": True, "dummy_kv_slot": 23},
            [23, 23],
            id="group-keyed-dummy",
        ),
        pytest.param({}, [11, 21], id="legacy"),
    ),
)
def test_fused_decode_input_prep_reuses_position_path(
    mode_kwargs,
    expected_cache_locs,
):
    req_pool_indices = torch.tensor([0, 1], dtype=torch.int64, device="cuda")
    valid_cache_lengths = torch.tensor([3, 5, 0], dtype=torch.int32, device="cuda")
    req_to_pages = torch.tensor(
        [
            [2, 3, 0, 0, 0, 0, 0],
            [4, 5, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=torch.int32,
        device="cuda",
    )
    cache_locs = torch.empty(2, dtype=torch.int32, device="cuda")
    positions = torch.empty(2, dtype=torch.int64, device="cuda")
    seq_lens = torch.empty(2, dtype=torch.int32, device="cuda")

    cache_loc_kernel.fused_decode_input_prep(
        cache_locs,
        positions,
        seq_lens,
        req_pool_indices,
        valid_cache_lengths,
        1,
        req_to_pages,
        4,
        **mode_kwargs,
    )

    assert cache_locs.tolist() == expected_cache_locs
    assert positions.tolist() == [3, 5]
    assert seq_lens.tolist() == [4, 6]
