from __future__ import annotations

import os
import sys
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

from tokenspeed.runtime.layers.attention.kv_cache.recipes.plan import (  # noqa: E402
    CacheFieldLayout,
    CacheGroupLayout,
    CacheMemoryPlan,
    CachePlaneLayout,
)
from tokenspeed.runtime.layers.attention.kv_cache.recipes.spec import (  # noqa: E402
    PagedCacheGroupSpec,
)
from tokenspeed.runtime.pd.cache_protocol import (  # noqa: E402
    CachePDGroupPages,
    CachePDPageManifest,
    CacheTransferContract,
)
from tokenspeed.runtime.pd.mooncake.entities import (  # noqa: E402
    TransferInfo,
    TransferKVChunk,
)


def _group_spec(
    group_id: str,
    family: str,
    policy: str,
    *,
    packing: int = 1,
) -> PagedCacheGroupSpec:
    return PagedCacheGroupSpec(
        group_id=group_id,
        retention="full_history",
        rows_per_page=2,
        entry_stride_tokens=1,
        sliding_window_tokens=None,
        family=family,
        cache_blocks_per_lcm_block=packing,
        transfer_policy=policy,
    )


def _layout(*, capacity: int = 16) -> CacheTransferContract:
    specs = (
        _group_spec("history", "history", "full_suffix"),
        _group_spec("state", "state", "latest_snapshot"),
    )
    plan = CacheMemoryPlan(
        logical_block_tokens=2,
        lcm_block_bytes=32,
        num_lcm_blocks=capacity - 1,
        groups=tuple(CacheGroupLayout(spec.group_id, 1, capacity) for spec in specs),
        planes=(CachePlaneLayout("cache", 32, 0),),
        fields=(
            CacheFieldLayout("history", "layer.0.latent", "cache", (16,), 1, 0, 32),
            CacheFieldLayout("state", "layer.1.state", "cache", (16,), 1, 16, 32),
        ),
    )
    return CacheTransferContract(plan, specs, ("uint8", "uint8"))


def _op(*, state_page: int = 6):
    tables = {
        "history": np.asarray([[1, 2, 3]], dtype=np.int32),
        "state": np.asarray([[4, 5, state_page]], dtype=np.int32),
    }
    return SimpleNamespace(
        request_ids=["request-0"],
        request_pool_indices=[7],
        extend_prefix_lens=[2],
        prefill_lengths=[5],
        num_extends=lambda: 1,
        block_tables_arrays=lambda: tables,
    )


def _destination_manifest() -> CachePDPageManifest:
    return CachePDPageManifest(
        groups=(
            CachePDGroupPages("history", (10, 11)),
            CachePDGroupPages("state", (12,)),
        ),
        prefix_len=2,
        prompt_len=5,
    )


def _destination_transfer_info(layout: CacheTransferContract) -> TransferInfo:
    manifest = _destination_manifest()
    frames = [
        b"9",
        b"127.0.0.1",
        b"9000",
        b"127.0.0.1:9001",
        np.asarray([10, 11, 12], dtype=np.int64).tobytes(),
        b"7",
        b"1",
        b"2",
        b"",
        b"1",
        b"[]",
        manifest.to_wire_bytes(),
        layout.to_wire_bytes(),
        b"1",
        b"0",
    ]
    return TransferInfo.from_zmq(frames)


class _FakeTensor:
    def __init__(self, values) -> None:
        self.values = list(values)

    def to(self, _device, non_blocking: bool = False):
        assert non_blocking
        return self


class _FakeCudaStreamContext:
    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc, _tb):
        return False


def test_cache_decode_destination_seeds_remote_prompt_cache_length(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tokenspeed.runtime.pd.decode_executor as decode_module

    executor = object.__new__(decode_module.DisaggDecodeExecutor)
    forward_op = SimpleNamespace(
        num_extends=lambda: 2,
        request_pool_indices=[7, 11],
        prefill_lengths=[1535, 3073],
    )
    tensor_calls = []

    def fake_tensor(values, *, dtype, device, pin_memory):
        tensor_calls.append((list(values), dtype, device, pin_memory))
        return _FakeTensor(values)

    current_stream = object()
    stream_context = _FakeCudaStreamContext()
    execution_stream = SimpleNamespace(
        wait_stream=lambda stream: stream is current_stream
    )
    resets = []
    runtime_states = SimpleNamespace(
        reset_states=lambda indices, lengths: resets.append(
            (indices.values, lengths.values)
        )
    )
    fake_torch = SimpleNamespace(
        int64="int64",
        int32="int32",
        tensor=fake_tensor,
        cuda=SimpleNamespace(
            current_stream=lambda: current_stream,
            stream=lambda stream: (
                stream_context
                if stream is execution_stream
                else pytest.fail("unexpected execution stream")
            ),
        ),
    )
    monkeypatch.setattr(decode_module, "torch", fake_torch)

    executor.reset_valid_cache_length(
        forward_op,
        runtime_states,
        execution_stream,
        device="cuda:0",
    )

    assert tensor_calls == [
        ([7, 11], "int64", "cpu", True),
        ([1535, 3073], "int32", "cpu", True),
    ]
    assert resets == [([7, 11], [1535, 3073])]


def test_cache_factory_exposes_one_typed_arena() -> None:
    from tokenspeed.runtime.cache.transfer.layout import (
        CacheField,
        CacheGroupLayout,
        CacheTransferLayout,
    )
    from tokenspeed.runtime.pd.factory import get_kv_args

    class _Dtype:
        def __str__(self):
            return "torch.uint8"

    class _Buffer:
        dtype = _Dtype()
        nbytes = _layout().plan.arena_bytes

        @staticmethod
        def is_contiguous():
            return True

        @staticmethod
        def storage_offset():
            return 0

        @staticmethod
        def data_ptr():
            return 0x1000

        @staticmethod
        def untyped_storage():
            return SimpleNamespace(data_ptr=lambda: 0x1000)

    layout = _layout()
    transfer_layout = CacheTransferLayout(
        num_lcm_blocks=layout.plan.num_lcm_blocks,
        groups=tuple(
            CacheGroupLayout(
                group_id=group.group_id,
                cache_blocks_per_lcm_block=group.cache_blocks_per_lcm_block,
                fields=(
                    CacheField(
                        field_id=field.field_id,
                        device_buffer_index=0,
                        device_block_zero_offset_bytes=field.field_offset_bytes,
                        block_stride_bytes=field.page_stride_bytes,
                        payload_bytes=field.payload_bytes,
                        dtype=dtype,
                    ),
                ),
            )
            for group, field, dtype in zip(
                layout.plan.groups,
                layout.plan.fields,
                layout.field_dtypes,
                strict=True,
            )
        ),
        buffers=(_Buffer(),),
        consumers=(tuple(field.field_id for field in layout.plan.fields),),
    )
    pool = SimpleNamespace(
        supports_disaggregation=True,
        plan=layout.plan,
        paged_cache_group_specs=layout.group_specs,
        cache_transfer_layout=lambda *, scope: transfer_layout,
        cache_contract_binding=lambda: (
            _Buffer(),
            {
                field.field_id: dtype
                for field, dtype in zip(
                    layout.plan.fields, layout.field_dtypes, strict=True
                )
            },
        ),
    )

    model_config = SimpleNamespace(
        num_attention_layers=2,
        num_key_value_heads=1,
        hf_config=SimpleNamespace(),
    )
    kv_args = get_kv_args(
        0,
        0,
        "mlx5_0",
        pool,
        None,
        model_config=model_config,
    )

    assert kv_args.target_layer_num == 1
    assert kv_args.kv_layer_ids == [0]
    assert kv_args.offsets == [(0,)]
    assert kv_args.kv_item_lens == [32]
    assert kv_args.kv_unit_lens == [32]
    assert kv_args.cache_layout == layout


def test_decode_publishes_manifest_through_legacy_receiver() -> None:
    import tokenspeed.runtime.pd.decode_executor as decode_module

    calls = []
    receiver = SimpleNamespace(
        prefill=lambda *args, **kwargs: calls.append((args, kwargs))
    )
    executor = object.__new__(decode_module.DisaggDecodeExecutor)
    executor.cache_layout = _layout()
    executor.receivers = {"request-0": receiver}
    executor._request_pool_indices = {}

    executor._cache_prefill(_op())

    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args[0].tolist() == [2, 3, 6]
    assert args[1:] == (7, 2, None)
    assert kwargs["page_manifest"].groups[0].page_ids == (2, 3)
    assert executor._request_pool_indices == {"request-0": 7}


def test_prefill_submits_manifest_through_legacy_sender() -> None:
    import tokenspeed.runtime.pd.prefill_executor as prefill_module

    layout = _layout()
    destination = _destination_transfer_info(layout)
    calls = []

    class _Sender:
        bootstrap_room = 9

        def has_layerwise_final(self):
            return False

        def has_layerwise_transfer(self):
            return False

        def send(self, *args, **kwargs):
            calls.append((args, kwargs))

    executor = object.__new__(prefill_module.DisaggPrefillExecutor)
    executor.cache_layout = layout
    executor.senders = {"request-0": _Sender()}
    executor.kv_manager = SimpleNamespace(
        transfer_infos={9: {destination.mooncake_session_id: destination}}
    )
    executor._request_token = {"request-0": 42}
    executor._layerwise_enabled = False

    executor._cache_decode(_op())

    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args[0].tolist() == [2, 3, 6]
    assert args[1:] == (7, True)
    assert kwargs["bootstrap_token"] == 42
    assert kwargs["page_manifest"].prompt_len == 5
    assert executor._request_token == {}


def test_cache_contract_epd_abort_notifies_decode() -> None:
    import tokenspeed.runtime.pd.prefill_executor as prefill_module

    calls = []
    executor = object.__new__(prefill_module.DisaggPrefillExecutor)
    executor.uses_cache_contract = True
    executor.kv_manager = SimpleNamespace(
        abort_room=lambda room, reason: calls.append((room, reason))
    )
    bootstrap = SimpleNamespace(bootstrap_room=9)

    executor.abort("request-0", bootstrap)

    assert calls == [
        (9, "EPD: prefill aborted request request-0 (embedding receive timed out)")
    ]


def test_shared_manager_validates_manifest_before_dma() -> None:
    from tokenspeed.runtime.pd.mooncake.prefill import MooncakeKVManagerPrefill

    layout = _layout()
    destination = _destination_transfer_info(layout)
    source_manifest = CachePDPageManifest(
        groups=(
            CachePDGroupPages("history", (2, 3)),
            CachePDGroupPages("state", (6,)),
        ),
        prefix_len=2,
        prompt_len=5,
    )
    chunk = TransferKVChunk(
        room=9,
        prefill_kv_indices=np.asarray([2, 3, 6], dtype=np.int64),
        index_slice=slice(0, 3),
        is_last=True,
        prefill_aux_index=7,
        page_manifest=source_manifest,
    )
    manager = object.__new__(MooncakeKVManagerPrefill)
    manager.kv_args = SimpleNamespace(cache_layout=layout)
    manager.world_size = 1
    manager.dp_size = 1
    manager.attn_tp_rank = 0

    manager._validate_cache_transfer(chunk, destination)

    chunk.prefill_kv_indices[2] = 5
    with pytest.raises(ValueError, match="page vector disagree"):
        manager._validate_cache_transfer(chunk, destination)


def _segmented_layout(
    *,
    capacity: int,
    history_k_offset: int,
    history_v_offset: int,
    state_offset: int,
) -> CacheTransferContract:
    specs = (
        _group_spec("history", "history", "full_suffix", packing=2),
        _group_spec("state", "state", "latest_snapshot"),
    )
    plan = CacheMemoryPlan(
        logical_block_tokens=2,
        lcm_block_bytes=1024,
        num_lcm_blocks=capacity - 1,
        groups=(
            CacheGroupLayout("history", 2, 1 + (capacity - 1) * 2),
            CacheGroupLayout("state", 1, capacity),
        ),
        planes=(
            CachePlaneLayout("history_k", 256, history_k_offset),
            CachePlaneLayout("history_v", 256, history_v_offset),
            CachePlaneLayout("state", 512, state_offset),
        ),
        fields=(
            CacheFieldLayout("history", "layer.0.k", "history_k", (128,), 2, 0, 256),
            CacheFieldLayout("history", "layer.0.v", "history_v", (128,), 2, 0, 256),
            CacheFieldLayout("state", "layer.1.ssm", "state", (96,), 4, 0, 512),
        ),
    )
    return CacheTransferContract(
        plan,
        specs,
        ("bfloat16", "bfloat16", "float32"),
    )


def test_shared_manager_resolves_lcm_group_segments() -> None:
    from tokenspeed.runtime.pd.mooncake.prefill import MooncakeKVManagerPrefill

    layout = _segmented_layout(
        capacity=5,
        history_k_offset=768,
        history_v_offset=2816,
        state_offset=512,
    )
    source_manifest = CachePDPageManifest(
        groups=(
            CachePDGroupPages("history", (1, 2)),
            CachePDGroupPages("state", (4,)),
        ),
        prefix_len=0,
        prompt_len=4,
    )
    destination_manifest = CachePDPageManifest(
        groups=(
            CachePDGroupPages("history", (5, 6)),
            CachePDGroupPages("state", (3,)),
        ),
        prefix_len=0,
        prompt_len=4,
    )
    calls = []
    manager = object.__new__(MooncakeKVManagerPrefill)
    manager.kv_args = SimpleNamespace(
        cache_layout=layout,
        kv_data_ptrs=[0x10000],
        kv_data_lens=[layout.plan.arena_bytes],
        kv_item_lens=[1024],
    )
    manager.engine = SimpleNamespace(
        batch_transfer_sync=lambda session, src, dst, lengths: (
            calls.append((session, src, dst, lengths)) or 0
        )
    )

    assert (
        manager.send_kvcache(
            "session",
            np.asarray([1, 2, 4], dtype=np.int64),
            [0x20000],
            np.asarray([5, 6, 3], dtype=np.int64),
            None,
            src_page_manifest=source_manifest,
            dst_page_manifest=destination_manifest,
            dst_cache_layout=layout,
        )
        == 0
    )
    assert calls == [
        (
            "session",
            [
                0x10000 + 768 + 1 * 256,
                0x10000 + 768 + 2 * 256,
                0x10000 + 2816 + 1 * 256,
                0x10000 + 2816 + 2 * 256,
                0x10000 + 512 + 4 * 512,
            ],
            [
                0x20000 + 768 + 5 * 256,
                0x20000 + 768 + 6 * 256,
                0x20000 + 2816 + 5 * 256,
                0x20000 + 2816 + 6 * 256,
                0x20000 + 512 + 3 * 512,
            ],
            [256, 256, 256, 256, 384],
        )
    ]


def test_shared_manager_uses_destination_page_zero_offsets() -> None:
    from tokenspeed.runtime.pd.mooncake.prefill import MooncakeKVManagerPrefill

    source_layout = _segmented_layout(
        capacity=5,
        history_k_offset=768,
        history_v_offset=2816,
        state_offset=512,
    )
    destination_layout = _segmented_layout(
        capacity=7,
        history_k_offset=768,
        history_v_offset=3840,
        state_offset=512,
    )
    source_manifest = CachePDPageManifest(
        groups=(
            CachePDGroupPages("history", (1, 2)),
            CachePDGroupPages("state", (4,)),
        ),
        prefix_len=0,
        prompt_len=4,
    )
    destination_manifest = CachePDPageManifest(
        groups=(
            CachePDGroupPages("history", (5, 6)),
            CachePDGroupPages("state", (3,)),
        ),
        prefix_len=0,
        prompt_len=4,
    )
    calls = []
    manager = object.__new__(MooncakeKVManagerPrefill)
    manager.kv_args = SimpleNamespace(
        cache_layout=source_layout,
        kv_data_ptrs=[0x10000],
        kv_data_lens=[source_layout.plan.arena_bytes],
        kv_item_lens=[1024],
    )
    manager.engine = SimpleNamespace(
        batch_transfer_sync=lambda session, src, dst, lengths: (
            calls.append((session, src, dst, lengths)) or 0
        )
    )

    assert (
        manager.send_kvcache(
            "session",
            np.asarray([1, 2, 4], dtype=np.int64),
            [0x20000],
            np.asarray([5, 6, 3], dtype=np.int64),
            None,
            src_page_manifest=source_manifest,
            dst_page_manifest=destination_manifest,
            dst_cache_layout=destination_layout,
        )
        == 0
    )
    assert calls[0][1] == [
        0x10000 + 768 + 1 * 256,
        0x10000 + 768 + 2 * 256,
        0x10000 + 2816 + 1 * 256,
        0x10000 + 2816 + 2 * 256,
        0x10000 + 512 + 4 * 512,
    ]
    assert calls[0][2] == [
        0x20000 + 768 + 5 * 256,
        0x20000 + 768 + 6 * 256,
        0x20000 + 3840 + 5 * 256,
        0x20000 + 3840 + 6 * 256,
        0x20000 + 512 + 3 * 512,
    ]


def test_cache_uses_equal_tp_identity_route() -> None:
    from tokenspeed.runtime.pd.mooncake.decode import PrefillParallelInfo
    from tokenspeed.runtime.pd.mooncake.receiver import _calc

    layout = _layout()
    manager = SimpleNamespace(
        world_size=8,
        dp_size=1,
        kv_args=SimpleNamespace(
            engine_rank=3,
            cache_layout=layout,
            kv_item_lens=[32],
            kv_unit_lens=[32],
        ),
    )
    prefill = PrefillParallelInfo(
        tp_size=8,
        dp_size=1,
        enable_mla_l1_5_cache=False,
        kv_item_lens=(32,),
        kv_unit_lens=(32,),
        cache_layout=layout,
    )

    route = _calc(manager, prefill)

    assert route.target_tp_rank == 3
    assert route.target_tp_ranks == (3,)
    assert route.required_prefill_response_num == 1
    assert route.default_required_dst_info_num == 1

    prefill.tp_size = 4
    route = _calc(manager, prefill)
    assert route.target_tp_rank is None
    assert route.target_tp_ranks == (1,)
    assert route.required_prefill_response_num == 1
    assert route.default_required_dst_info_num == 2


def test_pool_without_disaggregation_does_not_require_cache_contract() -> None:
    from tokenspeed.runtime.pd.factory import _get_cache_contract

    assert _get_cache_contract(SimpleNamespace(supports_disaggregation=False)) is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
