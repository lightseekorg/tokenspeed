from types import SimpleNamespace

import msgspec
import pytest

from tokenspeed.runtime.pd.kv_events import (
    AllBlocksCleared,
    BlockRemoved,
    BlockStored,
    EventIdAllocator,
    EventPublisherFactory,
    KVEventBatch,
    KVEventsConfig,
    NullEventPublisher,
    apply_envelope,
    assign_event_ids,
    drain_scheduler_kv_events,
    scheduler_kv_event_to_wire_event,
    scheduler_kv_events_to_wire_events,
)


class _FakePublisher(NullEventPublisher):
    def __init__(self, attn_dp_rank: int = 0, **kwargs):
        super().__init__(attn_dp_rank=attn_dp_rank)
        self.kwargs = kwargs


def test_backend_id_default_reads_envs(monkeypatch) -> None:
    import sys
    import types

    from tokenspeed.runtime.pd import kv_events as kv_events_mod

    utils_pkg = types.ModuleType("tokenspeed.runtime.utils")
    utils_pkg.__path__ = []  # mark as package
    env_mod = types.ModuleType("tokenspeed.runtime.utils.env")

    class _Field:
        def get(self) -> str:
            return "from-envs"

    env_mod.envs = types.SimpleNamespace(TOKENSPEED_KV_EVENTS_BACKEND_ID=_Field())
    monkeypatch.setitem(sys.modules, "tokenspeed.runtime.utils", utils_pkg)
    monkeypatch.setitem(sys.modules, "tokenspeed.runtime.utils.env", env_mod)

    assert kv_events_mod._default_kv_events_backend_id() == "from-envs"
    assert KVEventsConfig().backend_id == "from-envs"


def test_kv_events_config_defaults_hash_mode_to_fnv() -> None:
    config = KVEventsConfig()

    assert config.hash_mode == "fnv"


def test_vllm_style_enable_kv_cache_events_config_is_accepted() -> None:
    config = KVEventsConfig.from_cli(
        '{"publisher":"zmq","endpoint":"tcp://*:5557","enable_kv_cache_events":true}'
    )

    assert config.enable_kv_cache_events is True
    assert EventPublisherFactory.is_enabled(
        '{"publisher":"zmq","endpoint":"tcp://*:5557","enable_kv_cache_events":true}'
    )


def test_factory_returns_null_publisher_when_events_are_disabled() -> None:
    publisher = EventPublisherFactory.create(
        '{"publisher":"zmq","endpoint":"tcp://*:5557","enable_kv_cache_events":false}',
        attn_dp_rank=3,
    )

    assert isinstance(publisher, NullEventPublisher)


def test_enable_only_config_defaults_to_zmq_publisher() -> None:
    original = EventPublisherFactory._registry["zmq"]
    EventPublisherFactory._registry["zmq"] = _FakePublisher
    try:
        publisher = EventPublisherFactory.create(
            '{"enable_kv_cache_events":true}',
            attn_dp_rank=4,
        )
    finally:
        EventPublisherFactory._registry["zmq"] = original

    assert isinstance(publisher, _FakePublisher)


def test_block_stored_carries_token_ids_for_xxh3_mode() -> None:
    event = SimpleNamespace(
        kind="BlockStored",
        block_hashes=[123],
        parent_block_hash=None,
        token_ids=[1, 2, 3, 4],
        block_size=4,
    )

    wire_event = scheduler_kv_event_to_wire_event(event)

    assert wire_event.token_ids == [1, 2, 3, 4]

    wire_event_xxh3 = scheduler_kv_event_to_wire_event(event, hash_mode="xxh3")

    assert wire_event_xxh3.token_ids == [1, 2, 3, 4]

    bad = SimpleNamespace(
        kind="BlockStored",
        block_hashes=[123],
        parent_block_hash=None,
        token_ids=[],
        block_size=4,
    )
    with pytest.raises(ValueError, match="token_ids"):
        scheduler_kv_event_to_wire_event(bad, hash_mode="xxh3")

    bad_none = SimpleNamespace(
        kind="BlockStored",
        block_hashes=[123],
        parent_block_hash=None,
        token_ids=None,
        block_size=4,
    )
    with pytest.raises(ValueError, match="token_ids"):
        scheduler_kv_event_to_wire_event(bad_none, hash_mode="xxh3")


def test_scheduler_block_stored_translation() -> None:
    event = SimpleNamespace(
        kind="BlockStored",
        block_hashes=[123],
        parent_block_hash=None,
        token_ids=[1, 2, 3, 4],
        block_size=4,
    )

    wire_event = scheduler_kv_event_to_wire_event(event)

    assert wire_event == BlockStored(
        block_hashes=[123],
        parent_block_hash=None,
        token_ids=[1, 2, 3, 4],
        block_size=4,
    )


def test_scheduler_block_removed_translation() -> None:
    event = SimpleNamespace(kind="BlockRemoved", block_hashes=[123, 456])

    wire_event = scheduler_kv_event_to_wire_event(event)

    assert wire_event == BlockRemoved(block_hashes=[123, 456])


def test_scheduler_translation_uses_event_kind_not_shape() -> None:
    event = SimpleNamespace(
        kind="FutureSchedulerEvent",
        block_hashes=[123],
        token_ids=[1, 2],
    )

    with pytest.raises(TypeError, match="FutureSchedulerEvent"):
        scheduler_kv_event_to_wire_event(event)


def test_drain_scheduler_kv_events_skips_binding_when_disabled() -> None:
    assert drain_scheduler_kv_events(object(), enabled=False) == []


def test_drain_scheduler_kv_events_errors_clearly_when_binding_is_missing() -> None:
    with pytest.raises(RuntimeError, match="Scheduler.drain_kv_events"):
        drain_scheduler_kv_events(object(), enabled=True)


def test_drain_scheduler_kv_events_returns_scheduler_events() -> None:
    event = SimpleNamespace(block_hashes=[123])
    scheduler = SimpleNamespace(drain_kv_events=lambda: [event])

    assert drain_scheduler_kv_events(scheduler, enabled=True) == [event]


def test_rfc1527_envelope_on_block_stored() -> None:
    batch = KVEventBatch(
        ts=1.0,
        events=[
            BlockStored(
                block_hashes=[123],
                parent_block_hash=None,
                token_ids=[1, 2],
                block_size=2,
                backend_id="worker-0",
                medium="gpu",
                dp_rank=0,
                model_name="test-model",
            )
        ],
        attn_dp_rank=0,
    )
    decoded = msgspec.msgpack.decode(msgspec.msgpack.encode(batch))

    # Tagged BlockStored with new optional fields; unset envelope fields omitted.
    assert decoded == [
        1.0,
        [
            {
                "type": "BlockStored",
                "block_hashes": [123],
                "parent_block_hash": None,
                "token_ids": [1, 2],
                "block_size": 2,
                "backend_id": "worker-0",
                "medium": "gpu",
                "dp_rank": 0,
                "model_name": "test-model",
            },
        ],
        0,
    ]


def test_kv_events_config_rfc1527_defaults() -> None:
    config = KVEventsConfig()

    assert config.backend_id == "tokenspeed-worker"
    assert config.tenant_id == "default"
    assert config.model_name is None
    assert config.wire_format == "legacy"
    assert config.publish_medium is True


def test_factory_pops_rfc1527_config_fields() -> None:
    original = EventPublisherFactory._registry["zmq"]
    EventPublisherFactory._registry["zmq"] = _FakePublisher
    try:
        publisher = EventPublisherFactory.create(
            '{"enable_kv_cache_events":true,"backend_id":"worker-1",'
            '"tenant_id":"t1","wire_format":"rfc1527","publish_medium":false}',
            attn_dp_rank=0,
        )
    finally:
        EventPublisherFactory._registry["zmq"] = original

    assert isinstance(publisher, _FakePublisher)
    assert "backend_id" not in publisher.kwargs
    assert "tenant_id" not in publisher.kwargs
    assert "wire_format" not in publisher.kwargs
    assert "publish_medium" not in publisher.kwargs
    assert "model_name" not in publisher.kwargs
    assert "hash_mode" not in publisher.kwargs


def test_apply_envelope_legacy_leaves_fields_unset() -> None:
    event = BlockStored(
        block_hashes=[123],
        parent_block_hash=None,
        token_ids=[1, 2],
        block_size=2,
    )
    config = KVEventsConfig(
        wire_format="legacy",
        backend_id="worker-0",
        tenant_id="t1",
        model_name="m",
    )

    annotated = apply_envelope(event, config)

    assert annotated.backend_id is None
    assert annotated.medium is None
    assert annotated.model_name is None
    assert annotated.tenant_id is None
    # omit_defaults must keep Dynamo map payload free of envelope keys
    decoded = msgspec.msgpack.decode(msgspec.msgpack.encode(annotated))
    assert "backend_id" not in decoded
    assert "medium" not in decoded


def test_apply_envelope_rfc1527_sets_fields_from_config() -> None:
    event = BlockStored(
        block_hashes=[123],
        parent_block_hash=None,
        token_ids=[1, 2],
        block_size=2,
    )
    config = KVEventsConfig(
        wire_format="rfc1527",
        backend_id="worker-0",
        tenant_id="t1",
        model_name="test-model",
        publish_medium=True,
    )

    annotated = apply_envelope(event, config, medium="gpu")

    assert annotated.backend_id == "worker-0"
    assert annotated.tenant_id == "t1"
    assert annotated.model_name == "test-model"
    assert annotated.medium == "gpu"


def test_apply_envelope_medium_gpu_under_rfc1527() -> None:
    event = BlockStored(
        block_hashes=[1],
        parent_block_hash=None,
        token_ids=[1],
        block_size=1,
    )
    config = KVEventsConfig(wire_format="rfc1527", backend_id="w", publish_medium=True)

    annotated = apply_envelope(event, config, medium="gpu")

    assert annotated.medium == "gpu"


def test_scheduler_kv_events_to_wire_events_tags_medium_gpu_for_rfc1527() -> None:
    """Device-tier scheduler events are GPU; wire path must set medium=gpu."""
    raw = SimpleNamespace(
        kind="BlockStored",
        block_hashes=[123],
        parent_block_hash=None,
        token_ids=[1, 2, 3, 4],
        block_size=4,
    )
    config = KVEventsConfig(
        wire_format="rfc1527",
        backend_id="worker-0",
        tenant_id="t1",
        model_name="m",
        publish_medium=True,
    )

    events = scheduler_kv_events_to_wire_events(
        [raw], hash_mode=config.hash_mode, config=config, medium="gpu"
    )

    assert len(events) == 1
    assert isinstance(events[0], BlockStored)
    assert events[0].medium == "gpu"
    assert events[0].backend_id == "worker-0"
    assert events[0].tenant_id == "t1"
    assert events[0].model_name == "m"


def test_scheduler_kv_events_tier_host_maps_to_medium_cpu() -> None:
    """KvEventTier.kHost / 1 → wire medium=cpu under rfc1527."""
    raw = SimpleNamespace(
        kind="BlockStored",
        block_hashes=[456],
        parent_block_hash=None,
        token_ids=[1, 2],
        block_size=2,
        tier=1,  # KvEventTier.kHost
    )
    config = KVEventsConfig(
        wire_format="rfc1527",
        backend_id="worker-0",
        publish_medium=True,
    )

    events = scheduler_kv_events_to_wire_events([raw], config=config, medium="gpu")

    assert events[0].medium == "cpu"


def test_scheduler_kv_events_tier_device_maps_to_medium_gpu() -> None:
    """KvEventTier.kDevice / 0 → wire medium=gpu when caller passes medium=gpu."""
    raw = SimpleNamespace(
        kind="BlockStored",
        block_hashes=[789],
        parent_block_hash=None,
        token_ids=[3, 4],
        block_size=2,
        tier=0,  # KvEventTier.kDevice
    )
    config = KVEventsConfig(
        wire_format="rfc1527",
        backend_id="worker-0",
        publish_medium=True,
    )

    events = scheduler_kv_events_to_wire_events([raw], config=config, medium="gpu")

    assert events[0].medium == "gpu"


def test_scheduler_kv_events_missing_tier_uses_caller_medium_fallback() -> None:
    """Older bindings without tier fall back to the caller's medium= argument."""
    raw = SimpleNamespace(
        kind="BlockRemoved",
        block_hashes=[1, 2],
        # no tier attribute
    )
    config = KVEventsConfig(
        wire_format="rfc1527",
        backend_id="worker-0",
        publish_medium=True,
    )

    events = scheduler_kv_events_to_wire_events([raw], config=config, medium="gpu")

    assert events[0].medium == "gpu"


def test_scheduler_kv_events_mixed_device_and_host_tiers() -> None:
    """Batch of device + host stored events get correct per-event mediums."""

    class _FakeTier:
        """Stand-in for nanobind enum that is not an int but coerces via int()."""

        def __init__(self, value: int) -> None:
            self._value = value

        def __int__(self) -> int:
            return self._value

    device = SimpleNamespace(
        kind="BlockStored",
        block_hashes=[111],
        parent_block_hash=None,
        token_ids=[1],
        block_size=1,
        tier=0,
    )
    host = SimpleNamespace(
        kind="BlockStored",
        block_hashes=[222],
        parent_block_hash=111,
        token_ids=[2],
        block_size=1,
        tier=1,
    )
    host_enum = SimpleNamespace(
        kind="BlockRemoved",
        block_hashes=[222],
        tier=_FakeTier(1),
    )
    config = KVEventsConfig(
        wire_format="rfc1527",
        backend_id="worker-0",
        publish_medium=True,
    )

    events = scheduler_kv_events_to_wire_events(
        [device, host, host_enum], config=config, medium="gpu"
    )

    assert [e.medium for e in events] == ["gpu", "cpu", "cpu"]


def test_scheduler_kv_events_to_wire_events_default_medium_is_none() -> None:
    """Callers must pass medium explicitly; default leaves the field unset."""
    raw = SimpleNamespace(
        kind="BlockStored",
        block_hashes=[123],
        parent_block_hash=None,
        token_ids=[1, 2],
        block_size=2,
    )
    config = KVEventsConfig(
        wire_format="rfc1527",
        backend_id="worker-0",
        publish_medium=True,
    )

    events = scheduler_kv_events_to_wire_events([raw], config=config)

    assert events[0].backend_id == "worker-0"
    assert events[0].medium is None


def test_scheduler_kv_events_to_wire_events_legacy_omits_medium() -> None:
    raw = SimpleNamespace(
        kind="BlockStored",
        block_hashes=[123],
        parent_block_hash=None,
        token_ids=[1, 2],
        block_size=2,
    )
    config = KVEventsConfig(wire_format="legacy", backend_id="worker-0")

    events = scheduler_kv_events_to_wire_events([raw], config=config, medium="gpu")

    assert events[0].medium is None
    assert events[0].backend_id is None
    decoded = msgspec.msgpack.decode(msgspec.msgpack.encode(events[0]))
    assert "medium" not in decoded
    assert "backend_id" not in decoded


def test_apply_envelope_rfc1527_skips_medium_when_disabled() -> None:
    event = BlockRemoved(block_hashes=[99])
    config = KVEventsConfig(
        wire_format="rfc1527",
        backend_id="worker-1",
        publish_medium=False,
    )

    annotated = apply_envelope(event, config, medium="cpu")

    assert annotated.backend_id == "worker-1"
    assert annotated.medium is None


def test_apply_envelope_rfc1527_cleared_event() -> None:
    event = AllBlocksCleared()
    config = KVEventsConfig(wire_format="rfc1527", backend_id="w", tenant_id="t")

    annotated = apply_envelope(event, config, medium="gpu")

    assert annotated.backend_id == "w"
    assert annotated.tenant_id == "t"
    assert annotated.medium == "gpu"


def test_event_id_allocator_starts_at_zero_and_increments() -> None:
    """RFC #1527 event_id is monotonic per stream; Dynamo-style start at 0."""
    allocator = EventIdAllocator()
    key = ("m", 16, "backend-0", "gpu", 0)

    assert allocator.next(key) == 0
    assert allocator.next(key) == 1
    assert allocator.next(key) == 2


def test_event_id_allocator_independent_streams() -> None:
    allocator = EventIdAllocator()
    stream_a = ("m", 16, "backend-0", "gpu", 0)
    stream_b = ("m", 16, "backend-0", "cpu", 0)

    assert allocator.next(stream_a) == 0
    assert allocator.next(stream_b) == 0
    assert allocator.next(stream_a) == 1
    assert allocator.next(stream_b) == 1


def _rfc1527_stored(*, block_size: int = 16) -> BlockStored:
    return BlockStored(
        block_hashes=[123],
        parent_block_hash=None,
        token_ids=[1, 2],
        block_size=block_size,
    )


def test_assign_event_ids_two_publishes_increment() -> None:
    """Two successive assign/publish cycles on the same stream increment event_id."""
    allocator = EventIdAllocator()
    config = KVEventsConfig(
        wire_format="rfc1527",
        backend_id="worker-0",
        model_name="test-model",
        publish_medium=True,
    )

    first = apply_envelope(_rfc1527_stored(), config, medium="gpu", dp_rank=0)
    assign_event_ids([first], config, allocator)
    assert first.event_id == 0

    second = apply_envelope(_rfc1527_stored(), config, medium="gpu", dp_rank=0)
    assign_event_ids([second], config, allocator)
    assert second.event_id == 1


def test_assign_event_ids_legacy_leaves_event_id_unset() -> None:
    allocator = EventIdAllocator()
    config = KVEventsConfig(wire_format="legacy", backend_id="worker-0")
    event = _rfc1527_stored()

    assign_event_ids([event], config, allocator)

    assert event.event_id is None
    assert allocator.next(("x", None, None, None, None)) == 0  # unused


def test_assign_event_ids_block_removed_uses_none_block_size() -> None:
    """BlockRemoved has no block_size; stream key uses None for that dimension."""
    allocator = EventIdAllocator()
    config = KVEventsConfig(
        wire_format="rfc1527",
        backend_id="worker-0",
        model_name="m",
    )
    removed = apply_envelope(
        BlockRemoved(block_hashes=[1]), config, medium="gpu", dp_rank=0
    )
    stored = apply_envelope(
        _rfc1527_stored(block_size=16), config, medium="gpu", dp_rank=0
    )

    assign_event_ids([removed], config, allocator)
    assign_event_ids([stored], config, allocator)

    # Different block_size dimension → independent streams both start at 0
    assert removed.event_id == 0
    assert stored.event_id == 0

    assign_event_ids(
        [
            apply_envelope(
                BlockRemoved(block_hashes=[2]), config, medium="gpu", dp_rank=0
            )
        ],
        config,
        allocator,
    )
    # Same (model, None, backend, medium, dp) stream as first removed
    assert allocator.next(("m", None, "worker-0", "gpu", 0)) == 2


def test_apply_envelope_sets_dp_rank() -> None:
    event = _rfc1527_stored()
    config = KVEventsConfig(wire_format="rfc1527", backend_id="w")

    annotated = apply_envelope(event, config, medium="gpu", dp_rank=3)

    assert annotated.dp_rank == 3


def test_scheduler_kv_events_to_wire_events_passes_dp_rank() -> None:
    raw = SimpleNamespace(
        kind="BlockStored",
        block_hashes=[123],
        parent_block_hash=None,
        token_ids=[1, 2],
        block_size=2,
    )
    config = KVEventsConfig(wire_format="rfc1527", backend_id="worker-0")

    events = scheduler_kv_events_to_wire_events(
        [raw], config=config, medium="gpu", dp_rank=2
    )

    assert events[0].dp_rank == 2


def test_kv_event_batch_msgpack_shape_is_dynamo_compatible() -> None:
    """Dynamo ZMQ relay accepts tagged **map** events (vLLM-style), not
    TokenSpeed's older ``array_like`` positional arrays.

    ``array_like`` was dropped because msgspec ``omit_defaults`` cannot strip
    trailing ``None`` defaults from positional arrays; map encoding lets unset
    RFC #1527 envelope fields disappear so legacy payloads stay Dynamo-safe.
    """
    payload = msgspec.msgpack.encode(
        KVEventBatch(
            ts=1.5,
            events=[
                BlockStored(
                    block_hashes=[123],
                    parent_block_hash=None,
                    token_ids=[1, 2],
                    block_size=2,
                ),
                BlockRemoved(block_hashes=[123]),
            ],
            attn_dp_rank=2,
        )
    )

    decoded = msgspec.msgpack.decode(payload)

    # Batch remains array-like; events are tagged maps without envelope keys.
    assert decoded == [
        1.5,
        [
            {
                "type": "BlockStored",
                "block_hashes": [123],
                "parent_block_hash": None,
                "token_ids": [1, 2],
                "block_size": 2,
            },
            {
                "type": "BlockRemoved",
                "block_hashes": [123],
            },
        ],
        2,
    ]
