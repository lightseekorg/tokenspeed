from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import ANY, MagicMock, call

import pytest
import torch


def _load_host_transfer_contract_module():
    module_name = "tokenspeed_kernel.ops.kvcache.host_transfer"
    module = sys.modules.get(module_name)
    if module is not None and hasattr(module, "HostTransferGeometry"):
        return module

    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError:
        root = Path(__file__).resolve().parents[2] / "python" / "tokenspeed_kernel"
        packages = (
            ("tokenspeed_kernel", root),
            ("tokenspeed_kernel.ops", root / "ops"),
            ("tokenspeed_kernel.ops.kvcache", root / "ops" / "kvcache"),
        )
        for name, path in packages:
            if name not in sys.modules:
                pkg = types.ModuleType(name)
                pkg.__path__ = [str(path)]
                sys.modules[name] = pkg
        triton_stub = sys.modules.setdefault(
            "tokenspeed_kernel.ops.kvcache.triton",
            MagicMock(name="kvcache_triton_stub"),
        )
        triton_stub.HOST_CACHE_TRANSFER_CHUNK_BYTES = 4096
        return importlib.import_module(module_name)


@pytest.fixture
def host_transfer_contract():
    module = _load_host_transfer_contract_module()
    return SimpleNamespace(
        HostTransferGeometry=module.HostTransferGeometry,
        HostTransferWorkspace=module.HostTransferWorkspace,
        build_host_transfer_geometry=module.build_host_transfer_geometry,
        _triton_is_unavailable=module._triton_is_unavailable,
        _transfer_cache_ranges=module._transfer_cache_ranges,
        transfer_cache_blocks=module.transfer_cache_blocks,
        wait_layer_ready=module.wait_layer_ready,
    )


requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA device"
)


def _sample_geometry(
    contract,
    *,
    num_host_lcm_blocks: int = 2,
    num_device_lcm_blocks: int = 16,
):
    return contract.build_host_transfer_geometry(
        rows=(
            (0, 0, 100, 96, 80, 0, 4, 40),
            (0, 0, 100, 96, 80, 40, 4, 40),
            (1, 1, 200, 2048, 1200, 0, 1, 1000),
        ),
        layer_slices=((0, 2), (2, 0), (2, 1)),
        group_packing=(4, 1),
        host_lcm_block_bytes=1280,
        num_host_lcm_blocks=num_host_lcm_blocks,
        num_device_lcm_blocks=num_device_lcm_blocks,
        num_device_buffers=2,
    )


def _committed_block_workspace(host_transfer, geometry, transfers):
    workspace = host_transfer.HostTransferWorkspace()
    num_blocks, _ = workspace.load_block_transfers(transfers, geometry=geometry)
    workspace.commit_block_transfers(num_blocks, torch.device("cpu"))
    return workspace, num_blocks


def _mock_triton_block_path(monkeypatch, host_transfer, workspace):
    triton_transfer = MagicMock()
    range_factory = MagicMock(side_effect=AssertionError("ranges must stay lazy"))
    monkeypatch.setattr(host_transfer, "_transfer_cache_blocks_triton", triton_transfer)
    monkeypatch.setattr(host_transfer, "_make_block_ranges", range_factory)
    monkeypatch.setattr(host_transfer, "_mapped_host_triton_available", None)
    workspace.bind_addresses = MagicMock(return_value="addresses")
    return triton_transfer, range_factory


def test_geometry_layer_slices_preserve_empty_layers(host_transfer_contract):
    geometry = _sample_geometry(host_transfer_contract)

    assert geometry.layer_slices == ((0, 2), (2, 0), (2, 1))
    assert geometry.num_field_rows == 3


def test_geometry_bind_uploads_device_rows_once(host_transfer_contract):
    geometry = _sample_geometry(host_transfer_contract)
    device = torch.device("cpu")

    first = geometry.bind(device)
    second = first.bind(device)

    assert first.device_rows is not None
    assert first.device_layer_slices is not None
    assert second.device_rows.data_ptr() == first.device_rows.data_ptr()
    assert second.device_layer_slices.data_ptr() == first.device_layer_slices.data_ptr()
    assert torch.equal(first.device_rows, geometry.host_rows)
    assert torch.equal(
        first.device_layer_slices,
        torch.tensor(((0, 2), (2, 0), (2, 1)), dtype=torch.int64),
    )


def test_build_geometry_validates_row_shape_and_fields(host_transfer_contract):
    build = host_transfer_contract.build_host_transfer_geometry
    base = dict(
        rows=((0, 0, 100, 96, 80, 0, 4, 40),),
        layer_slices=((0, 1),),
        group_packing=(4,),
        host_lcm_block_bytes=320,
        num_host_lcm_blocks=2,
        num_device_lcm_blocks=16,
        num_device_buffers=1,
    )

    with pytest.raises(ValueError, match="8 columns"):
        build(**{**base, "rows": ((0, 0, 100, 96, 80, 0, 4),)})

    with pytest.raises(IndexError, match="device_buffer_index"):
        build(**{**base, "rows": ((0, 1, 100, 96, 80, 0, 4, 40),)})

    with pytest.raises(ValueError, match="payload_bytes cannot exceed"):
        build(**{**base, "rows": ((0, 0, 100, 96, 80, 0, 4, 128),)})

    with pytest.raises(ValueError, match="group packing"):
        build(**{**base, "rows": ((0, 0, 100, 96, 80, 0, 8, 40),)})

    with pytest.raises(ValueError, match="outside host cache block"):
        build(**{**base, "rows": ((0, 0, 100, 96, 80, 60, 4, 40),)})

    with pytest.raises(ValueError, match="outside host LCM block"):
        build(**{**base, "rows": ((0, 0, 100, 96, 81, 0, 4, 40),)})


def test_load_block_transfers_buckets_by_group(host_transfer_contract):
    geometry = _sample_geometry(host_transfer_contract)
    workspace = host_transfer_contract.HostTransferWorkspace()
    transfers = (
        (1, 3, 2),
        (0, 5, 4),
        (0, 2, 1),
        (1, 7, 1),
    )

    num_blocks, group_offsets = workspace.load_block_transfers(
        transfers,
        geometry=geometry,
    )

    assert num_blocks == 4
    assert group_offsets == (0, 2, 4)
    assert torch.equal(
        workspace._block_host[:num_blocks],
        torch.tensor(
            ((5, 4), (2, 1), (3, 2), (7, 1)),
            dtype=torch.int64,
            device="cpu",
        ),
    )


def test_per_group_device_block_id_bounds_with_unequal_packing(host_transfer_contract):
    geometry = host_transfer_contract.build_host_transfer_geometry(
        rows=(
            (0, 0, 0, 64, 32, 0, 4, 32),
            (1, 0, 0, 128, 64, 0, 8, 64),
        ),
        layer_slices=((0, 2),),
        group_packing=(4, 8),
        host_lcm_block_bytes=512,
        num_host_lcm_blocks=2,
        num_device_lcm_blocks=8,
        num_device_buffers=1,
    )
    workspace = host_transfer_contract.HostTransferWorkspace()

    workspace.load_block_transfers(((0, 32, 1),), geometry=geometry)
    workspace.load_block_transfers(((1, 64, 1),), geometry=geometry)

    with pytest.raises(IndexError, match="device_block_id 33"):
        workspace.load_block_transfers(((0, 33, 1),), geometry=geometry)

    with pytest.raises(IndexError, match="device_block_id 65"):
        workspace.load_block_transfers(((1, 65, 1),), geometry=geometry)


def test_workspace_prepare_layer_ready_zeros_and_reuses(host_transfer_contract):
    workspace = host_transfer_contract.HostTransferWorkspace()
    device = torch.device("cpu")

    first = workspace.prepare_layer_ready(3, device)
    first[:] = 7
    workspace.layer_cta_counts()[:] = 4
    reused = workspace.prepare_layer_ready(3, device)

    assert reused.data_ptr() == first.data_ptr()
    assert reused.tolist() == [0, 0, 0]
    assert workspace.layer_cta_counts().tolist() == [0, 0, 0]


def test_workspace_reuses_block_mapping_storage(host_transfer_contract):
    geometry = _sample_geometry(host_transfer_contract)
    workspace = host_transfer_contract.HostTransferWorkspace()
    first = ((0, 2, 1), (1, 3, 2))
    workspace.load_block_transfers(first, geometry=geometry)
    host_ptr = workspace._block_host.data_ptr()
    offsets_ptr = workspace._block_group_offsets_host.data_ptr()

    workspace.load_block_transfers(((0, 4, 3),), geometry=geometry)

    assert workspace._block_host.data_ptr() == host_ptr
    assert workspace._block_group_offsets_host.data_ptr() == offsets_ptr


def test_group_offsets_use_valid_length_after_capacity_growth(host_transfer_contract):
    geometry = _sample_geometry(host_transfer_contract)
    workspace = host_transfer_contract.HostTransferWorkspace()
    device = torch.device("cpu")

    num_blocks, _ = workspace.load_block_transfers(((0, 2, 1),), geometry=geometry)
    workspace.commit_block_transfers(num_blocks, device)
    valid_count = geometry.num_groups + 1
    expanded = torch.zeros(valid_count + 4, dtype=torch.int64, device=device)
    expanded[:valid_count] = workspace._block_group_offsets_device[:valid_count]
    expanded[valid_count:] = 999
    workspace._block_group_offsets_device = expanded

    offsets = workspace.block_group_offsets_device()
    assert offsets.numel() == valid_count
    assert offsets.numel() < workspace._block_group_offsets_device.shape[0]
    assert 999 not in offsets.tolist()
    assert tuple(int(value) for value in offsets.tolist()) == (0, 1, 1)


@pytest.mark.parametrize(
    "transfers",
    [
        ((0, 0, 1),),
        ((0, 1, 0),),
        ((0, 2, 0),),
    ],
)
def test_load_block_transfers_rejects_zero_ids(host_transfer_contract, transfers):
    geometry = _sample_geometry(host_transfer_contract)
    workspace = host_transfer_contract.HostTransferWorkspace()

    with pytest.raises(ValueError, match="1-based"):
        workspace.load_block_transfers(transfers, geometry=geometry)


@pytest.mark.parametrize(
    "transfers,match",
    [
        (((2, 1, 1),), "group"),
        (((0, 65, 1),), "device_block_id 65"),
        (((1, 17, 1),), "device_block_id 17"),
        (((0, 1, 9),), "host_block_id"),
        (((1, 1, 3),), "host_block_id"),
    ],
)
def test_load_block_transfers_rejects_out_of_range_ids(
    host_transfer_contract, transfers, match
):
    geometry = _sample_geometry(host_transfer_contract)
    workspace = host_transfer_contract.HostTransferWorkspace()

    with pytest.raises(IndexError, match=match):
        workspace.load_block_transfers(transfers, geometry=geometry)


def test_empty_load_invalidates_committed_block_state(host_transfer_contract):
    geometry = _sample_geometry(host_transfer_contract)
    workspace = host_transfer_contract.HostTransferWorkspace()
    device = torch.device("cpu")

    num_blocks, _ = workspace.load_block_transfers(((0, 2, 1),), geometry=geometry)
    workspace.commit_block_transfers(num_blocks, device)

    workspace.load_block_transfers((), geometry=geometry)

    with pytest.raises(ValueError, match="not committed"):
        workspace.committed_block_tables(1)
    with pytest.raises(ValueError, match="not committed"):
        workspace.block_group_offsets_device()


def test_accessor_rejects_uncommitted_load(host_transfer_contract):
    geometry = _sample_geometry(host_transfer_contract)
    workspace = host_transfer_contract.HostTransferWorkspace()

    workspace.load_block_transfers(((0, 2, 1), (1, 3, 2)), geometry=geometry)

    with pytest.raises(ValueError, match="not committed"):
        workspace.committed_block_tables(2)
    with pytest.raises(ValueError, match="not committed"):
        workspace.block_group_offsets_device()


def test_new_load_invalidates_previous_commit(host_transfer_contract):
    geometry = _sample_geometry(host_transfer_contract)
    workspace = host_transfer_contract.HostTransferWorkspace()
    device = torch.device("cpu")

    first_count, _ = workspace.load_block_transfers(((0, 2, 1),), geometry=geometry)
    workspace.commit_block_transfers(first_count, device)

    workspace.load_block_transfers(((0, 4, 3),), geometry=geometry)

    with pytest.raises(ValueError, match="not committed"):
        workspace.committed_block_tables(first_count)


def test_commit_rejects_wrong_num_blocks(host_transfer_contract):
    geometry = _sample_geometry(host_transfer_contract)
    workspace = host_transfer_contract.HostTransferWorkspace()
    device = torch.device("cpu")

    num_blocks, _ = workspace.load_block_transfers(
        ((0, 2, 1), (1, 3, 2)),
        geometry=geometry,
    )

    with pytest.raises(ValueError, match="must equal"):
        workspace.commit_block_transfers(num_blocks - 1, device)


def test_commit_rejects_duplicate_upload(host_transfer_contract):
    geometry = _sample_geometry(host_transfer_contract)
    workspace = host_transfer_contract.HostTransferWorkspace()
    device = torch.device("cpu")

    num_blocks, _ = workspace.load_block_transfers(((0, 2, 1),), geometry=geometry)
    workspace.commit_block_transfers(num_blocks, device)

    with pytest.raises(ValueError, match="already committed"):
        workspace.commit_block_transfers(num_blocks, device)


def test_committed_tables_are_shared_without_reupload(host_transfer_contract):
    geometry = _sample_geometry(host_transfer_contract)
    workspace = host_transfer_contract.HostTransferWorkspace()
    num_blocks, _ = workspace.load_block_transfers(
        ((0, 2, 1), (0, 5, 4), (1, 3, 2)),
        geometry=geometry,
    )
    device = torch.device("cpu")
    block_table, returned_offsets = workspace.commit_block_transfers(num_blocks, device)
    block_ptr = workspace._block_device.data_ptr()
    offset_ptr = workspace._block_group_offsets_device.data_ptr()

    committed_blocks, offsets = workspace.committed_block_tables(num_blocks)

    assert committed_blocks.data_ptr() == block_table.data_ptr()
    assert offsets.data_ptr() == returned_offsets.data_ptr()
    assert workspace._block_device.data_ptr() == block_ptr
    assert workspace._block_group_offsets_device.data_ptr() == offset_ptr


def test_unrelated_triton_runtime_error_does_not_fall_back_to_dma(
    host_transfer_contract,
):
    assert not host_transfer_contract._triton_is_unavailable(
        RuntimeError("requested kernel specialization is not available")
    )


def test_block_transfer_rejects_unknown_direction():
    host_transfer = _load_host_transfer_contract_module()

    with pytest.raises(ValueError, match="direction"):
        host_transfer.transfer_cache_blocks(
            "sideways",
            (torch.empty(1, dtype=torch.uint8),),
            torch.empty(1, dtype=torch.uint8),
            _sample_geometry(host_transfer),
            host_transfer.HostTransferWorkspace(),
            stream=None,
            num_blocks=0,
            geometry_offset=0,
            num_geometry_rows=0,
        )


def test_block_h2d_triton_uses_committed_tables_without_expanding_ranges(monkeypatch):
    host_transfer = _load_host_transfer_contract_module()
    geometry = _sample_geometry(host_transfer).bind(torch.device("cpu"))
    workspace, num_blocks = _committed_block_workspace(
        host_transfer,
        geometry,
        (
            (0, 2, 1),
            (0, 3, 2),
            (0, 4, 3),
            (0, 5, 4),
            (1, 3, 2),
            (1, 4, 1),
        ),
    )
    triton_transfer, range_factory = _mock_triton_block_path(
        monkeypatch, host_transfer, workspace
    )
    device_buffer = torch.empty(1, dtype=torch.uint8)

    host_transfer.transfer_cache_blocks(
        "h2d",
        (device_buffer,),
        torch.empty(1, dtype=torch.uint8),
        geometry,
        workspace,
        stream=None,
        num_blocks=num_blocks,
        geometry_offset=2,
        num_geometry_rows=1,
        backend="triton",
        grid_cap=7,
    )

    workspace.bind_addresses.assert_called_once()
    range_factory.assert_not_called()
    triton_transfer.assert_called_once()
    args, kwargs = triton_transfer.call_args
    assert args[0] == "addresses"
    assert args[1].data_ptr() == geometry.device_rows.data_ptr()
    assert args[2].data_ptr() == workspace._block_device.data_ptr()
    assert args[3].numel() == geometry.num_groups + 1
    assert args[4] == 1
    assert kwargs == {
        "geometry_offset": 2,
        "num_geometry_rows": 1,
        "host_lcm_block_bytes": geometry.host_lcm_block_bytes,
        "work_items": 2,
        "num_device_buffers": 1,
        "grid_cap": 7,
    }


def test_block_h2d_triton_layered_flags_launch_full_geometry_once(monkeypatch):
    host_transfer = _load_host_transfer_contract_module()
    monkeypatch.setattr(host_transfer, "layer_ready_ptx_supported", lambda: True)
    geometry = _sample_geometry(host_transfer).bind(torch.device("cpu"))
    workspace, num_blocks = _committed_block_workspace(
        host_transfer,
        geometry,
        ((0, 2, 1), (1, 3, 2)),
    )
    flags = workspace.prepare_layer_ready(
        len(geometry.layer_slices), torch.device("cpu")
    )
    triton_transfer, range_factory = _mock_triton_block_path(
        monkeypatch, host_transfer, workspace
    )

    host_transfer.transfer_cache_blocks(
        "h2d",
        (torch.empty(1, dtype=torch.uint8),),
        torch.empty(1, dtype=torch.uint8),
        geometry,
        workspace,
        stream=None,
        num_blocks=num_blocks,
        geometry_offset=0,
        num_geometry_rows=geometry.num_field_rows,
        backend="triton",
        layer_ready_flags=flags,
    )

    range_factory.assert_not_called()
    triton_transfer.assert_called_once()
    args, kwargs = triton_transfer.call_args
    assert args[4] == 1
    assert kwargs["geometry_offset"] == 0
    assert kwargs["num_geometry_rows"] == geometry.num_field_rows
    assert kwargs["layer_ready_flags"].data_ptr() == flags.data_ptr()
    assert kwargs["layer_slices"].data_ptr() == geometry.device_layer_slices.data_ptr()
    assert (
        kwargs["layer_cta_counts"].data_ptr() == workspace.layer_cta_counts().data_ptr()
    )


def test_block_d2h_triton_reuses_geometry_without_expanding_ranges(monkeypatch):
    host_transfer = _load_host_transfer_contract_module()
    geometry = _sample_geometry(host_transfer).bind(torch.device("cpu"))
    workspace, num_blocks = _committed_block_workspace(
        host_transfer,
        geometry,
        ((0, 2, 1), (1, 3, 2)),
    )
    triton_transfer, range_factory = _mock_triton_block_path(
        monkeypatch, host_transfer, workspace
    )
    device_buffers = (
        torch.empty(1, dtype=torch.uint8),
        torch.empty(1, dtype=torch.uint8),
    )

    host_transfer.transfer_cache_blocks(
        "d2h",
        device_buffers,
        torch.empty(1, dtype=torch.uint8),
        geometry,
        workspace,
        stream=None,
        num_blocks=num_blocks,
        geometry_offset=0,
        num_geometry_rows=geometry.num_field_rows,
        backend="triton",
    )

    range_factory.assert_not_called()
    args = triton_transfer.call_args.args
    assert args[0] == "addresses"
    assert args[1].data_ptr() == geometry.device_rows.data_ptr()
    assert args[4] == 0


def test_block_h2d_triton_uses_max_real_work_across_layer_groups(monkeypatch):
    host_transfer = _load_host_transfer_contract_module()
    geometry = host_transfer.build_host_transfer_geometry(
        rows=(
            (0, 0, 0, 8192, 6000, 0, 1, 5003),
            (1, 0, 0, 64, 32, 0, 1, 32),
        ),
        layer_slices=((0, 2),),
        group_packing=(1, 1),
        host_lcm_block_bytes=6000,
        num_host_lcm_blocks=8,
        num_device_lcm_blocks=8,
        num_device_buffers=1,
    ).bind(torch.device("cpu"))
    workspace = host_transfer.HostTransferWorkspace()
    num_blocks, _ = workspace.load_block_transfers(
        ((0, 1, 1), (0, 2, 2), (1, 1, 1), (1, 2, 2), (1, 3, 3)),
        geometry=geometry,
    )
    workspace.commit_block_transfers(num_blocks, torch.device("cpu"))
    workspace.bind_addresses = MagicMock(return_value=object())
    triton_transfer = MagicMock()
    monkeypatch.setattr(host_transfer, "_transfer_cache_blocks_triton", triton_transfer)
    monkeypatch.setattr(host_transfer, "_mapped_host_triton_available", None)

    host_transfer.transfer_cache_blocks(
        "h2d",
        (torch.empty(1, dtype=torch.uint8),),
        torch.empty(1, dtype=torch.uint8),
        geometry,
        workspace,
        stream=None,
        num_blocks=num_blocks,
        geometry_offset=0,
        num_geometry_rows=2,
        backend="triton",
    )

    # Group 0: 2 blocks * 2 chunks = 4. Group 1: 3 blocks * 1 chunk = 3.
    assert triton_transfer.call_args.kwargs["work_items"] == 4


def test_block_h2d_triton_skips_layer_with_no_group_blocks(monkeypatch):
    host_transfer = _load_host_transfer_contract_module()
    geometry = _sample_geometry(host_transfer).bind(torch.device("cpu"))
    workspace = host_transfer.HostTransferWorkspace()
    num_blocks, _ = workspace.load_block_transfers(
        ((0, 2, 1),),
        geometry=geometry,
    )
    workspace.commit_block_transfers(num_blocks, torch.device("cpu"))
    triton_transfer = MagicMock()
    monkeypatch.setattr(host_transfer, "_transfer_cache_blocks_triton", triton_transfer)
    monkeypatch.setattr(host_transfer, "_mapped_host_triton_available", None)

    host_transfer.transfer_cache_blocks(
        "h2d",
        (torch.empty(1, dtype=torch.uint8),),
        torch.empty(1, dtype=torch.uint8),
        geometry,
        workspace,
        stream=None,
        num_blocks=num_blocks,
        geometry_offset=2,
        num_geometry_rows=1,
        backend="triton",
    )

    triton_transfer.assert_not_called()


@pytest.mark.parametrize("requested_num_blocks", [1, 3])
def test_block_h2d_triton_requires_exact_committed_block_count(
    monkeypatch,
    requested_num_blocks,
):
    host_transfer = _load_host_transfer_contract_module()
    geometry = _sample_geometry(host_transfer).bind(torch.device("cpu"))
    workspace = host_transfer.HostTransferWorkspace()
    committed_count, _ = workspace.load_block_transfers(
        ((0, 2, 1), (1, 3, 2)),
        geometry=geometry,
    )
    workspace.commit_block_transfers(committed_count, torch.device("cpu"))
    workspace.bind_addresses = MagicMock(return_value=object())
    monkeypatch.setattr(host_transfer, "_mapped_host_triton_available", None)
    monkeypatch.setattr(
        host_transfer,
        "_transfer_cache_blocks_triton",
        MagicMock(),
    )

    with pytest.raises(ValueError, match="committed block count"):
        host_transfer.transfer_cache_blocks(
            "h2d",
            (torch.empty(1, dtype=torch.uint8),),
            torch.empty(1, dtype=torch.uint8),
            geometry,
            workspace,
            stream=None,
            num_blocks=requested_num_blocks,
            geometry_offset=0,
            num_geometry_rows=2,
            backend="triton",
        )
    workspace.bind_addresses.assert_not_called()


@pytest.mark.parametrize("direction", ["d2h", "h2d"])
def test_block_dma_expands_ranges_without_binding_device_metadata(
    monkeypatch, direction
):
    host_transfer = _load_host_transfer_contract_module()
    geometry = _sample_geometry(host_transfer)
    workspace = host_transfer.HostTransferWorkspace()
    num_blocks, _ = workspace.load_block_transfers(
        ((0, 2, 1), (1, 3, 2)),
        geometry=geometry,
    )
    ranges = ((1, 6344, 1280, 1000),)
    range_factory = MagicMock(return_value=ranges)
    range_transfer = MagicMock()
    monkeypatch.setattr(host_transfer, "_make_block_ranges", range_factory)
    monkeypatch.setattr(host_transfer, "_transfer_cache_ranges", range_transfer)
    workspace.bind_addresses = MagicMock(
        side_effect=AssertionError("DMA must not bind addresses")
    )
    workspace.commit_block_transfers = MagicMock(
        side_effect=AssertionError("DMA must not commit blocks")
    )

    host_transfer.transfer_cache_blocks(
        direction,
        (torch.empty(1, dtype=torch.uint8),),
        torch.empty(1, dtype=torch.uint8),
        geometry,
        workspace,
        stream=None,
        num_blocks=num_blocks,
        geometry_offset=2,
        num_geometry_rows=1,
        backend="dma",
    )

    range_factory.assert_called_once_with(
        geometry,
        workspace,
        num_blocks=num_blocks,
        geometry_offset=2,
        num_geometry_rows=1,
    )
    range_transfer.assert_called_once_with(
        direction,
        (ANY,),
        ANY,
        ranges,
        None,
    )
    workspace.bind_addresses.assert_not_called()
    workspace.commit_block_transfers.assert_not_called()


def test_block_h2d_range_factory_expands_only_requested_geometry_slice():
    host_transfer = _load_host_transfer_contract_module()
    geometry = _sample_geometry(host_transfer)
    workspace = host_transfer.HostTransferWorkspace()
    num_blocks, _ = workspace.load_block_transfers(
        ((0, 2, 1), (0, 5, 4), (1, 3, 2)),
        geometry=geometry,
    )

    ranges = host_transfer._make_block_ranges(
        geometry,
        workspace,
        num_blocks=num_blocks,
        geometry_offset=0,
        num_geometry_rows=2,
    )

    assert ranges == (
        (0, 292, 0, 40),
        (0, 580, 240, 40),
        (0, 292, 40, 40),
        (0, 580, 280, 40),
    )


def test_block_h2d_range_factory_reads_only_referenced_group_rows():
    host_transfer = _load_host_transfer_contract_module()
    geometry = _sample_geometry(host_transfer)
    workspace = host_transfer.HostTransferWorkspace()
    num_blocks, _ = workspace.load_block_transfers(
        ((0, 2, 1), (0, 5, 4), (1, 3, 2)),
        geometry=geometry,
    )
    block_rows = workspace.host_block_rows(num_blocks)
    accesses = []

    class SliceOnlyRows:
        def __getitem__(self, key):
            accesses.append(key)
            return block_rows[key]

        def tolist(self):
            raise AssertionError("must not materialize the complete block table")

    workspace.host_block_rows = MagicMock(return_value=SliceOnlyRows())

    ranges = host_transfer._make_block_ranges(
        geometry,
        workspace,
        num_blocks=num_blocks,
        geometry_offset=2,
        num_geometry_rows=1,
    )

    assert ranges == ((1, 6344, 1280, 1000),)
    assert accesses == [slice(2, 3)]


def test_block_h2d_auto_only_expands_after_capability_failure(monkeypatch):
    host_transfer = _load_host_transfer_contract_module()
    geometry = _sample_geometry(host_transfer).bind(torch.device("cpu"))
    workspace = host_transfer.HostTransferWorkspace()
    num_blocks, _ = workspace.load_block_transfers(
        ((0, 2, 1),),
        geometry=geometry,
    )
    workspace.commit_block_transfers(num_blocks, torch.device("cpu"))
    workspace.bind_addresses = MagicMock(return_value=object())
    ranges = ((0, 292, 0, 40), (0, 332, 40, 40))
    range_factory = MagicMock(return_value=ranges)
    range_transfer = MagicMock()
    monkeypatch.setattr(host_transfer, "_make_block_ranges", range_factory)
    monkeypatch.setattr(host_transfer, "_transfer_cache_ranges", range_transfer)
    monkeypatch.setattr(host_transfer, "_mapped_host_triton_available", None)
    monkeypatch.setattr(
        host_transfer,
        "_transfer_cache_blocks_triton",
        MagicMock(side_effect=RuntimeError("mapped host access is not available")),
    )

    with pytest.warns(RuntimeWarning, match="falling back"):
        host_transfer.transfer_cache_blocks(
            "h2d",
            (torch.empty(1, dtype=torch.uint8),),
            torch.empty(1, dtype=torch.uint8),
            geometry,
            workspace,
            stream=None,
            num_blocks=num_blocks,
            geometry_offset=0,
            num_geometry_rows=2,
            backend="auto",
        )

    range_factory.assert_called_once()
    range_transfer.assert_called_once()

    monkeypatch.setattr(host_transfer, "_mapped_host_triton_available", None)
    monkeypatch.setattr(
        host_transfer,
        "_transfer_cache_blocks_triton",
        MagicMock(side_effect=RuntimeError("kernel launch failed")),
    )
    range_factory.reset_mock()
    range_transfer.reset_mock()
    with pytest.raises(RuntimeError, match="kernel launch failed"):
        host_transfer.transfer_cache_blocks(
            "h2d",
            (torch.empty(1, dtype=torch.uint8),),
            torch.empty(1, dtype=torch.uint8),
            geometry,
            workspace,
            stream=None,
            num_blocks=num_blocks,
            geometry_offset=0,
            num_geometry_rows=2,
            backend="auto",
        )
    range_factory.assert_not_called()
    range_transfer.assert_not_called()


def test_block_h2d_auto_capability_fallback_persists_across_layers(monkeypatch):
    host_transfer = _load_host_transfer_contract_module()
    geometry = _sample_geometry(host_transfer).bind(torch.device("cpu"))
    workspace = host_transfer.HostTransferWorkspace()
    num_blocks, _ = workspace.load_block_transfers(
        ((0, 2, 1), (1, 3, 2)),
        geometry=geometry,
    )
    workspace.commit_block_transfers(num_blocks, torch.device("cpu"))
    workspace.bind_addresses = MagicMock(return_value=object())
    triton_transfer = MagicMock(
        side_effect=RuntimeError("mapped host access is not available")
    )
    range_factory = MagicMock(
        side_effect=(
            ((0, 292, 0, 40),),
            ((0, 6344, 1280, 1000),),
        )
    )
    range_transfer = MagicMock()
    monkeypatch.setattr(host_transfer, "_transfer_cache_blocks_triton", triton_transfer)
    monkeypatch.setattr(host_transfer, "_make_block_ranges", range_factory)
    monkeypatch.setattr(host_transfer, "_transfer_cache_ranges", range_transfer)
    monkeypatch.setattr(host_transfer, "_mapped_host_triton_available", None)

    with pytest.warns(RuntimeWarning, match="falling back"):
        host_transfer.transfer_cache_blocks(
            "h2d",
            (torch.empty(1, dtype=torch.uint8),),
            torch.empty(1, dtype=torch.uint8),
            geometry,
            workspace,
            stream=None,
            num_blocks=num_blocks,
            geometry_offset=0,
            num_geometry_rows=2,
            backend="auto",
        )
    host_transfer.transfer_cache_blocks(
        "h2d",
        (torch.empty(1, dtype=torch.uint8),),
        torch.empty(1, dtype=torch.uint8),
        geometry,
        workspace,
        stream=None,
        num_blocks=num_blocks,
        geometry_offset=2,
        num_geometry_rows=1,
        backend="auto",
    )

    assert host_transfer._mapped_host_triton_available is False
    triton_transfer.assert_called_once()
    assert range_factory.call_args_list == [
        call(
            geometry,
            workspace,
            num_blocks=num_blocks,
            geometry_offset=0,
            num_geometry_rows=2,
        ),
        call(
            geometry,
            workspace,
            num_blocks=num_blocks,
            geometry_offset=2,
            num_geometry_rows=1,
        ),
    ]
    assert range_transfer.call_count == 2


@requires_cuda
def test_cache_ranges_dma_round_trip_across_multiple_device_buffers():
    transfer_cache_ranges = _load_host_transfer_contract_module()._transfer_cache_ranges

    first = torch.arange(64, dtype=torch.uint8, device="cuda")
    second = torch.arange(48, dtype=torch.bfloat16, device="cuda")
    second_bytes = second.view(torch.uint8)
    host = torch.zeros(96, dtype=torch.uint8, pin_memory=True)
    ranges = ((0, 8, 0, 24), (1, 16, 48, 32))
    stream = torch.cuda.Stream()

    transfer_cache_ranges("d2h", (first, second), host, ranges, stream)
    stream.synchronize()
    assert torch.equal(host[0:24], first[8:32].cpu())
    assert torch.equal(host[48:80], second_bytes[16:48].cpu())

    host[0:24].fill_(7)
    host[48:80].fill_(9)
    transfer_cache_ranges("h2d", (first, second), host, ranges, stream)
    stream.synchronize()
    assert torch.equal(first[8:32].cpu(), torch.full((24,), 7, dtype=torch.uint8))
    assert torch.equal(
        second_bytes[16:48].cpu(), torch.full((32,), 9, dtype=torch.uint8)
    )


@requires_cuda
def test_geometry_block_transfer_is_byte_exact_for_packed_multigroup_fields():
    module = _load_host_transfer_contract_module()
    geometry = module.build_host_transfer_geometry(
        rows=(
            (0, 0, 13, 6001, 5014, 0, 2, 5003),
            (0, 1, 5, 41, 5014, 5003, 2, 11),
            (1, 1, 17, 53, 19, 0, 3, 13),
            (1, 0, 29, 47, 19, 13, 3, 5),
        ),
        layer_slices=((0, 4),),
        group_packing=(2, 3),
        host_lcm_block_bytes=10028,
        num_host_lcm_blocks=2,
        num_device_lcm_blocks=4,
        num_device_buffers=2,
    )
    host = (
        torch.arange(20056, dtype=torch.int64)
        .remainder(251)
        .to(torch.uint8)
        .pin_memory()
    )
    first = torch.full((32768,), 0xA5, dtype=torch.uint8, device="cuda")
    second = torch.full((32768,), 0x5A, dtype=torch.uint8, device="cuda")
    workspace = module.HostTransferWorkspace()
    transfers = ((0, 1, 1), (0, 4, 4), (1, 2, 3), (1, 4, 6))
    num_blocks, _ = workspace.load_block_transfers(transfers, geometry=geometry)
    stream = torch.cuda.Stream()

    with torch.cuda.stream(stream):
        bound_geometry = geometry.bind(first.device, non_blocking=True)
        workspace.commit_block_transfers(
            num_blocks,
            first.device,
            non_blocking=True,
        )
    try:
        module.transfer_cache_blocks(
            "h2d",
            (first, second),
            host,
            bound_geometry,
            workspace,
            stream,
            num_blocks=num_blocks,
            geometry_offset=0,
            num_geometry_rows=4,
            backend="triton",
            grid_cap=2,
        )
    except RuntimeError as error:
        if (
            module._triton_is_unavailable(error)
            or "not device-mapped" in str(error).lower()
        ):
            pytest.skip(str(error))
        raise
    stream.synchronize()

    expected = (
        torch.full((32768,), 0xA5, dtype=torch.uint8),
        torch.full((32768,), 0x5A, dtype=torch.uint8),
    )
    rows = geometry.host_rows.tolist()
    grouped = (((1, 1), (4, 4)), ((2, 3), (4, 6)))
    for row in rows:
        (
            group,
            buffer,
            device_zero,
            stride,
            host_block_bytes,
            field_offset,
            packing,
            payload,
        ) = row
        for device_id, host_id in grouped[group]:
            parent, child = divmod(host_id - 1, packing)
            host_offset = (
                parent * geometry.host_lcm_block_bytes
                + child * host_block_bytes
                + field_offset
            )
            device_offset = device_zero + device_id * stride
            expected[buffer][device_offset : device_offset + payload] = host[
                host_offset : host_offset + payload
            ]
    assert torch.equal(first.cpu(), expected[0])
    assert torch.equal(second.cpu(), expected[1])

    host.zero_()
    module.transfer_cache_blocks(
        "d2h",
        (first, second),
        host,
        bound_geometry,
        workspace,
        stream,
        num_blocks=num_blocks,
        geometry_offset=0,
        num_geometry_rows=4,
        backend="triton",
        grid_cap=2,
    )
    stream.synchronize()

    expected_host = torch.zeros_like(host)
    for row in rows:
        (
            group,
            buffer,
            device_zero,
            stride,
            host_block_bytes,
            field_offset,
            packing,
            payload,
        ) = row
        for device_id, host_id in grouped[group]:
            parent, child = divmod(host_id - 1, packing)
            host_offset = (
                parent * geometry.host_lcm_block_bytes
                + child * host_block_bytes
                + field_offset
            )
            device_offset = device_zero + device_id * stride
            expected_host[host_offset : host_offset + payload] = expected[buffer][
                device_offset : device_offset + payload
            ]
    assert torch.equal(host, expected_host)


@requires_cuda
def test_layered_h2d_sets_flags_and_copies_each_layer():
    module = _load_host_transfer_contract_module()
    if not module.layer_ready_ptx_supported():
        pytest.skip("flagged Triton transfers require NVIDIA PTX")
    geometry = module.build_host_transfer_geometry(
        rows=(
            (0, 0, 0, 16, 16, 0, 1, 8),
            (0, 0, 8, 16, 16, 8, 1, 8),
        ),
        layer_slices=((0, 1), (1, 0), (1, 1)),
        group_packing=(1,),
        host_lcm_block_bytes=16,
        num_host_lcm_blocks=2,
        num_device_lcm_blocks=4,
        num_device_buffers=1,
    )
    host = torch.arange(32, dtype=torch.uint8).pin_memory()
    device = torch.full((64,), 0xA5, dtype=torch.uint8, device="cuda")
    workspace = module.HostTransferWorkspace()
    num_blocks, _ = workspace.load_block_transfers(((0, 1, 1),), geometry=geometry)
    stream = torch.cuda.Stream()
    compute = torch.cuda.Stream()

    with torch.cuda.stream(stream):
        bound = geometry.bind(device.device, non_blocking=True)
        workspace.commit_block_transfers(num_blocks, device.device, non_blocking=True)
        flags = workspace.prepare_layer_ready(3, device.device)
    try:
        module.transfer_cache_blocks(
            "h2d",
            (device,),
            host,
            bound,
            workspace,
            stream,
            num_blocks=num_blocks,
            geometry_offset=0,
            num_geometry_rows=2,
            backend="triton",
            layer_ready_flags=flags,
            grid_cap=2,
        )
        with torch.cuda.stream(compute):
            module.wait_layer_ready(flags, 0)
            module.wait_layer_ready(flags, 1)
            module.wait_layer_ready(flags, 2)
    except RuntimeError as error:
        if (
            module._triton_is_unavailable(error)
            or "not device-mapped" in str(error).lower()
        ):
            pytest.skip(str(error))
        raise
    compute.synchronize()

    assert flags.tolist() == [1, 1, 1]
    assert device[16:24].cpu().tolist() == host[0:8].tolist()
    assert device[24:32].cpu().tolist() == host[8:16].tolist()
