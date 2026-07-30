from __future__ import annotations

import json
import os
import sys
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

# CPU-only tests scheduled in runtime-1gpu because they import the full runtime.
sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=10, suite="runtime-1gpu")

from tokenspeed.runtime.pd.flatkv import (
    FLATKV_PD_PROTOCOL_VERSION,
    FlatKVPDGroup,
    FlatKVPDGroupPages,
    FlatKVPDLayout,
    FlatKVPDPageManifest,
    FlatKVPDPeerLayout,
    FlatKVPDProtocolError,
    FlatKVPDSLabRegistration,
    FlatKVPDTransferSegment,
    build_flatkv_page_manifest,
    build_lcm_flatkv_pd_contract,
    flatkv_manifest_page_ids,
    validate_flatkv_manifest_pair,
    validate_flatkv_peer_layout,
    validate_flatkv_slab_registrations,
)

_FINGERPRINT = "1" * 64


def _layout(capacity: int = 64) -> FlatKVPDLayout:
    return FlatKVPDLayout(
        version=FLATKV_PD_PROTOCOL_VERSION,
        layout_fingerprint=_FINGERPRINT,
        block_size=4,
        num_pages_with_null=capacity,
        physical_buffer_ids=("slab.0", "slab.1", "slab.2"),
        physical_page_bytes=128,
        groups=(
            FlatKVPDGroup("attention", "history", "full_suffix", (0, 1, 2)),
            FlatKVPDGroup("linear-a", "state", "latest_snapshot", (0, 1)),
            FlatKVPDGroup("linear-b", "state", "latest_snapshot", (0, 1)),
        ),
    )


def _op(page_offset: int = 0) -> SimpleNamespace:
    tables = {
        "attention": np.array([[1, 2, 3, 4, -1]], dtype=np.int32) + page_offset,
        "linear-a": np.array([[11, 12, 13, 14, -1]], dtype=np.int32) + page_offset,
        "linear-b": np.array([[21, 22, 23, 24, -1]], dtype=np.int32) + page_offset,
    }
    return SimpleNamespace(flat_block_tables_arrays=lambda: tables)


def _registrations(
    layout: FlatKVPDLayout, start: int
) -> tuple[FlatKVPDSLabRegistration, ...]:
    extent = layout.num_pages_with_null * layout.physical_page_bytes
    return tuple(
        FlatKVPDSLabRegistration(
            slot,
            layout.physical_buffer_ids[slot],
            start + slot * (extent + 1024),
            extent,
        )
        for slot in range(layout.physical_slot_count)
    )


def test_manifest_selects_history_suffix_and_only_final_state_slot() -> None:
    manifest = build_flatkv_page_manifest(
        _op(),
        layout=_layout(),
        request_row=0,
        prefix_len=4,
        prompt_len=14,
    )

    assert manifest.groups[0] == FlatKVPDGroupPages("attention", (2, 3, 4))
    assert manifest.groups[1:] == (
        FlatKVPDGroupPages("linear-a", (14,)),
        FlatKVPDGroupPages("linear-b", (24,)),
    )


def test_transfer_policy_is_model_defined_not_inferred_from_family() -> None:
    layout = _layout()
    groups = list(layout.groups)
    groups[1] = replace(groups[1], transfer_policy="full_suffix")
    layout = replace(layout, groups=tuple(groups))

    manifest = build_flatkv_page_manifest(
        _op(),
        layout=layout,
        request_row=0,
        prefix_len=4,
        prompt_len=14,
    )

    assert manifest.groups[1] == FlatKVPDGroupPages("linear-a", (12, 13, 14))


def test_layout_rejects_unbound_physical_slots() -> None:
    layout = _layout()

    assert layout.physical_slot_count == len(layout.physical_buffer_ids)
    with pytest.raises(FlatKVPDProtocolError, match="leaves physical slots unbound"):
        replace(
            layout,
            groups=tuple(
                replace(group, physical_slots=(0,)) for group in layout.groups
            ),
        )


def test_source_and_destination_page_ids_are_independent() -> None:
    source_layout = _layout(64)
    destination_layout = _layout(128)
    source = build_flatkv_page_manifest(
        _op(),
        layout=source_layout,
        request_row=0,
        prefix_len=4,
        prompt_len=14,
    )
    destination = build_flatkv_page_manifest(
        _op(40),
        layout=destination_layout,
        request_row=0,
        prefix_len=4,
        prompt_len=14,
    )

    validate_flatkv_peer_layout(source_layout, destination_layout.peer)
    validate_flatkv_manifest_pair(
        source,
        destination,
        source_layout,
        dst_num_pages_with_null=destination_layout.num_pages_with_null,
    )
    assert source.groups[0].page_ids != destination.groups[0].page_ids


@pytest.mark.parametrize("bad_page", [0, -1, 64])
def test_manifest_rejects_null_padding_and_out_of_capacity_page(
    bad_page: int,
) -> None:
    operation = _op()
    operation.flat_block_tables_arrays()["linear-a"][0, 3] = bad_page
    with pytest.raises(FlatKVPDProtocolError, match="invalid page ID"):
        build_flatkv_page_manifest(
            operation,
            layout=_layout(),
            request_row=0,
            prefix_len=4,
            prompt_len=14,
        )


def test_manifest_accepts_reordered_mapping_but_rejects_wrong_keys() -> None:
    with pytest.raises(FlatKVPDProtocolError, match="aligned"):
        build_flatkv_page_manifest(
            _op(),
            layout=_layout(),
            request_row=0,
            prefix_len=3,
            prompt_len=14,
        )

    reordered = _op()
    tables = reordered.flat_block_tables_arrays()
    reordered.flat_block_tables_arrays = lambda: {
        "linear-a": tables["linear-a"],
        "attention": tables["attention"],
        "linear-b": tables["linear-b"],
    }
    manifest = build_flatkv_page_manifest(
        reordered,
        layout=_layout(),
        request_row=0,
        prefix_len=4,
        prompt_len=14,
    )
    assert tuple(group.group_id for group in manifest.groups) == (
        "attention",
        "linear-a",
        "linear-b",
    )

    for wrong_tables in (
        {"attention": tables["attention"], "linear-a": tables["linear-a"]},
        {**tables, "extra": tables["attention"]},
    ):
        wrong = SimpleNamespace(flat_block_tables_arrays=lambda: wrong_tables)
        with pytest.raises(FlatKVPDProtocolError, match="group IDs disagree"):
            build_flatkv_page_manifest(
                wrong,
                layout=_layout(),
                request_row=0,
                prefix_len=4,
                prompt_len=14,
            )

    class DuplicateKeyMapping(dict):
        def __iter__(self):
            return iter(("attention", "attention", "linear-a", "linear-b"))

    duplicate = SimpleNamespace(
        flat_block_tables_arrays=lambda: DuplicateKeyMapping(tables)
    )
    with pytest.raises(FlatKVPDProtocolError, match="duplicate group IDs"):
        build_flatkv_page_manifest(
            duplicate,
            layout=_layout(),
            request_row=0,
            prefix_len=4,
            prompt_len=14,
        )


def test_manifest_rejects_selected_page_aliasing_a_live_unselected_slot() -> None:
    aliased = _op()
    aliased.flat_block_tables_arrays()["attention"][0, 0] = 2
    with pytest.raises(FlatKVPDProtocolError, match="aliases live unselected"):
        build_flatkv_page_manifest(
            aliased,
            layout=_layout(),
            request_row=0,
            prefix_len=4,
            prompt_len=14,
        )


def test_layout_and_manifest_wire_encoding_is_canonical_and_strict() -> None:
    layout = _layout()
    manifest = build_flatkv_page_manifest(
        _op(),
        layout=layout,
        request_row=0,
        prefix_len=4,
        prompt_len=14,
    )
    layout_wire = layout.peer.to_wire_bytes()
    manifest_wire = manifest.to_wire_bytes()
    assert json.loads(layout_wire) == {
        "layout_fingerprint": layout.layout_fingerprint,
        "num_pages_with_null": layout.num_pages_with_null,
        "version": FLATKV_PD_PROTOCOL_VERSION,
    }
    assert FlatKVPDPeerLayout.from_wire_bytes(layout_wire) == layout.peer
    assert FlatKVPDPageManifest.from_wire_bytes(manifest_wire) == manifest

    pretty = json.dumps(json.loads(layout_wire), indent=2).encode()
    with pytest.raises(FlatKVPDProtocolError, match="canonical"):
        FlatKVPDPeerLayout.from_wire_bytes(pretty)
    with pytest.raises(FlatKVPDProtocolError, match="duplicate key"):
        FlatKVPDPageManifest.from_wire_bytes(manifest_wire[:-1] + b',"version":1}')
    payload = json.loads(manifest_wire)
    payload["unexpected"] = True
    with pytest.raises(FlatKVPDProtocolError, match="schema keys"):
        FlatKVPDPageManifest.from_wire_bytes(
            json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
        )


def test_manifest_pair_rejects_layout_and_logical_mismatch() -> None:
    layout = _layout()
    source = build_flatkv_page_manifest(
        _op(),
        layout=layout,
        request_row=0,
        prefix_len=4,
        prompt_len=14,
    )
    destination = build_flatkv_page_manifest(
        _op(30),
        layout=layout,
        request_row=0,
        prefix_len=4,
        prompt_len=14,
    )
    with pytest.raises(FlatKVPDProtocolError, match="fingerprint"):
        validate_flatkv_peer_layout(
            layout, replace(layout.peer, layout_fingerprint="2" * 64)
        )
    with pytest.raises(FlatKVPDProtocolError, match="prompt_len"):
        validate_flatkv_manifest_pair(
            source,
            replace(destination, prompt_len=13),
            layout,
            dst_num_pages_with_null=layout.num_pages_with_null,
        )
    with pytest.raises(FlatKVPDProtocolError, match="page count"):
        validate_flatkv_manifest_pair(
            source,
            replace(
                destination,
                groups=(
                    replace(destination.groups[0], page_ids=(31, 32)),
                    *destination.groups[1:],
                ),
            ),
            layout,
            dst_num_pages_with_null=layout.num_pages_with_null,
        )
    # Kernel page ids are scoped by group in the two-level LCM layout.
    # Equal integers in different groups do not name the same CacheBlock.
    replace(
        source,
        groups=(
            source.groups[0],
            replace(source.groups[1], page_ids=(source.groups[0].page_ids[0],)),
            source.groups[2],
        ),
    )


def test_lcm_group_capacity_uses_its_cache_blocks_per_parent() -> None:
    layout = FlatKVPDLayout(
        version=1,
        layout_fingerprint=_FINGERPRINT,
        block_size=4,
        # Null parent plus three usable parents.
        num_pages_with_null=4,
        physical_buffer_ids=("arena",),
        physical_page_bytes=1024,
        groups=(
            FlatKVPDGroup(
                "history",
                "history",
                "full_suffix",
                (0,),
                cache_blocks_per_lcm_block=2,
                transfer_segments=(
                    FlatKVPDTransferSegment(
                        physical_slot=0,
                        field_id="layer.0.k",
                        dtype="bfloat16",
                        page_zero_offset=384,
                        page_stride_bytes=128,
                        payload_bytes=128,
                    ),
                ),
            ),
        ),
    )
    tables = {"history": np.array([[5, 6]], dtype=np.int32)}
    operation = SimpleNamespace(flat_block_tables_arrays=lambda: tables)

    build_flatkv_page_manifest(
        operation,
        layout=layout,
        request_row=0,
        prefix_len=0,
        prompt_len=8,
    )
    tables["history"][0, 1] = 7
    with pytest.raises(FlatKVPDProtocolError, match="invalid page ID"):
        build_flatkv_page_manifest(
            operation,
            layout=layout,
            request_row=0,
            prefix_len=0,
            prompt_len=8,
        )


def test_manifest_flattens_to_shared_mooncake_page_vector_order() -> None:
    source_layout = _layout(64)
    destination_layout = _layout(128)
    source = build_flatkv_page_manifest(
        _op(),
        layout=source_layout,
        request_row=0,
        prefix_len=8,
        prompt_len=14,
    )
    destination = build_flatkv_page_manifest(
        _op(40),
        layout=destination_layout,
        request_row=0,
        prefix_len=8,
        prompt_len=14,
    )
    assert flatkv_manifest_page_ids(
        source,
        layout=source_layout,
        peer="source",
    ) == (3, 4, 14, 24)
    assert flatkv_manifest_page_ids(
        destination,
        layout=source_layout,
        num_pages_with_null=destination_layout.num_pages_with_null,
        peer="destination",
    ) == (43, 44, 54, 64)


def test_registration_validation_rejects_bad_extents() -> None:
    layout = _layout()
    registrations = _registrations(layout, 10_000)
    extent = layout.num_pages_with_null * layout.physical_page_bytes
    assert (
        validate_flatkv_slab_registrations(registrations, layout=layout, peer="test")
        == registrations
    )
    with pytest.raises(FlatKVPDProtocolError, match="exact physical-slot"):
        validate_flatkv_slab_registrations(
            tuple(reversed(registrations)), layout=layout, peer="test"
        )
    with pytest.raises(FlatKVPDProtocolError, match="extent"):
        validate_flatkv_slab_registrations(
            (replace(registrations[0], length=extent - 1), *registrations[1:]),
            layout=layout,
            peer="test",
        )
    with pytest.raises(FlatKVPDProtocolError, match="buffer ID"):
        validate_flatkv_slab_registrations(
            (
                replace(registrations[0], buffer_id="wrong.raw.0"),
                *registrations[1:],
            ),
            layout=layout,
            peer="test",
        )
    with pytest.raises(FlatKVPDProtocolError, match="overlap"):
        validate_flatkv_slab_registrations(
            (
                registrations[0],
                replace(registrations[1], base_addr=registrations[0].base_addr),
                registrations[2],
            ),
            layout=layout,
            peer="test",
        )
    with pytest.raises(FlatKVPDProtocolError, match="uint64"):
        FlatKVPDSLabRegistration(0, "slab.0", (1 << 64) - extent + 1, extent)


def test_lcm_contract_registers_one_arena_and_preserves_group_geometry() -> None:
    class _Dtype:
        def __str__(self):
            return "torch.uint8"

    class _Backing:
        dtype = _Dtype()
        nbytes = 4096

        @staticmethod
        def is_contiguous():
            return True

        @staticmethod
        def storage_offset():
            return 0

        @staticmethod
        def data_ptr():
            return 0x10000

        @staticmethod
        def untyped_storage():
            return SimpleNamespace(data_ptr=lambda: 0x10000)

    plan = SimpleNamespace(
        logical_block_tokens=4,
        lcm_block_bytes=1024,
        num_lcm_blocks=3,
        arena_bytes=4096,
        groups=(
            SimpleNamespace(
                group_id="history",
                cache_blocks_per_lcm_block=2,
                page_count=7,
            ),
        ),
        planes=(
            SimpleNamespace(
                plane_id="k",
                bytes_per_lcm_block=512,
                arena_offset_bytes=0,
            ),
        ),
        fields=(
            SimpleNamespace(
                group_id="history",
                field_id="layer.0.k",
                plane_id="k",
                shape=(4, 8),
                element_size=2,
                field_offset_bytes=0,
                page_stride_bytes=256,
                payload_bytes=64,
            ),
        ),
    )
    specs = (
        SimpleNamespace(
            group_id="history",
            family="history",
            transfer_policy="full_suffix",
            retention="full_history",
            sliding_window_tokens=None,
        ),
    )

    layout, registrations = build_lcm_flatkv_pd_contract(
        plan=plan,
        backing=_Backing(),
        group_specs=specs,
        field_dtypes={"layer.0.k": "bfloat16"},
    )

    assert layout.num_pages_with_null == 4
    assert layout.groups[0].cache_blocks_per_lcm_block == 2
    assert layout.groups[0].transfer_segments[0].page_zero_offset == 256
    assert registrations == (FlatKVPDSLabRegistration(0, "lcm_arena", 0x10000, 4096),)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
