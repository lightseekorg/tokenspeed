from types import SimpleNamespace

import msgspec
import pytest

from tokenspeed.runtime.pd.kv_events import (
    BlockRemoved,
    BlockStored,
    EventPublisherFactory,
    KVEventBatch,
    KVEventsConfig,
    NullEventPublisher,
    drain_scheduler_kv_events,
    scheduler_kv_event_to_wire_event,
)


class _FakePublisher(NullEventPublisher):
    def __init__(self, attn_dp_rank: int = 0, **kwargs):
        super().__init__(attn_dp_rank=attn_dp_rank)
        self.kwargs = kwargs


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


def test_kv_event_batch_msgpack_shape_is_dynamo_compatible() -> None:
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

    assert decoded == [
        1.5,
        [
            ["BlockStored", [123], None, [1, 2], 2],
            ["BlockRemoved", [123]],
        ],
        2,
    ]
