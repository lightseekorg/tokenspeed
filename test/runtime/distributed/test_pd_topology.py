from types import SimpleNamespace

import pytest

from tokenspeed.runtime.pd.topology import PDParallelTopology


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ({"tp_size": 0}, "tp_size must be greater than 0"),
        ({"dp_size": 0}, "dp_size must be greater than 0"),
        ({"tp_rank": -1}, "tp_rank must be in \\[0, 2\\)"),
        ({"tp_rank": 2}, "tp_rank must be in \\[0, 2\\)"),
        ({"dp_rank": -1}, "dp_rank must be in \\[0, 4\\)"),
        ({"dp_rank": 4}, "dp_rank must be in \\[0, 4\\)"),
        ({"world_size": 7}, "world_size must equal tp_size \\* dp_size"),
        ({"global_rank": -1}, "global_rank must be in \\[0, 8\\)"),
        ({"global_rank": 8}, "global_rank must be in \\[0, 8\\)"),
    ],
)
def test_direct_constructor_rejects_invalid_topology(
    override: dict[str, int], match: str
) -> None:
    values = {
        "tp_size": 2,
        "tp_rank": 1,
        "dp_size": 4,
        "dp_rank": 1,
        "world_size": 8,
        "global_rank": 7,
    }
    values.update(override)

    with pytest.raises(ValueError, match=match):
        PDParallelTopology(**values)


def _mapping(
    *,
    tp_size: int,
    dp_size: int,
    world_size: int,
    global_rank: int,
) -> SimpleNamespace:
    tp_rank = global_rank % tp_size
    dp_rank = global_rank // tp_size % dp_size
    return SimpleNamespace(
        rank=global_rank,
        world_size=world_size,
        attn=SimpleNamespace(
            tp_size=tp_size,
            tp_rank=tp_rank,
            dp_size=dp_size,
            dp_rank=dp_rank,
        ),
    )


def test_topology_preserves_typed_attention_coordinates() -> None:
    mapping = _mapping(
        tp_size=2,
        dp_size=4,
        world_size=8,
        global_rank=3,
    )

    topology = PDParallelTopology.from_mapping(mapping)

    assert topology == PDParallelTopology(
        tp_size=2,
        tp_rank=1,
        dp_size=4,
        dp_rank=1,
        world_size=8,
        global_rank=3,
    )


def test_cache_pd_accepts_heterogeneous_tp() -> None:
    topology = PDParallelTopology.from_mapping(
        _mapping(
            tp_size=4,
            dp_size=2,
            world_size=8,
            global_rank=7,
        )
    )

    assert (topology.tp_size, topology.tp_rank) == (4, 3)


@pytest.mark.parametrize("role", ["prefill", "decode"])
def test_the_device_builds_the_pd_transfer_peer_from_the_mapping(monkeypatch, role):
    """The engine's PD start-up path: ``_build_kv_transfer`` derives the
    topology from the real ``Mapping`` and hands it to the transfer factory.
    Runs the device code up to the factory call, so a topology method the
    device still names but the class no longer has fails here, not at the
    first PD engine start."""
    from tokenspeed.runtime.distributed.mapping import Mapping
    from tokenspeed.runtime.execution import device
    from tokenspeed.runtime.pd import factory
    from tokenspeed.runtime.pd.mooncake import entities

    mapping = Mapping(rank=3, world_size=4, attn_tp_size=4, attn_qcp_size=4)
    seen = {}

    def fake_create(mode, backend, args, kv_args, gloo_group):
        seen.update(mode=mode, args=args, kv_args=kv_args, gloo_group=gloo_group)
        return SimpleNamespace(kind="peer")

    monkeypatch.setattr(factory, "create_kv_transfer", fake_create)
    monkeypatch.setattr(factory, "get_kv_args", lambda *a, **kw: "kv-args")
    monkeypatch.setattr(
        "tokenspeed.runtime.distributed.process_group_manager.process_group_manager.get_process_group",
        lambda backend, group: ("gloo", group),
    )
    server_args = SimpleNamespace(
        disaggregation_mode=role,
        disaggregation_transfer_backend="mooncake",
        disaggregation_bootstrap_port=8998,
        dist_init_addr="127.0.0.1:1",
        served_model_name="m",
        app_key="k",
        metrics_reporters="",
        disaggregation_ib_device=None,
        disaggregation_layerwise_interval=0,
        mapping=mapping,
    )
    peer = device._build_kv_transfer(
        server_args,
        SimpleNamespace(token_to_kv_pool=None),
        cache_fields_by_stage=((),),
        producer_fields_by_step=((),),
        logical_plan=None,
        model_config=None,
        draft_model_config=None,
        gpu_id=0,
        global_rank=3,
    )
    assert peer.kind == "peer" and seen["mode"] == role
    assert isinstance(seen["args"], entities.KVManagerArgs)
    assert seen["args"].topology == PDParallelTopology.from_mapping(mapping)
    assert seen["args"].topology.tp_rank == 3 and seen["args"].topology.world_size == 4
    assert seen["gloo_group"] == ("gloo", mapping.attn.tp_group)
