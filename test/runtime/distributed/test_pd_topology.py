from types import SimpleNamespace

import pytest

from tokenspeed.runtime.pd.topology import PDParallelTopology


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ({"tp_size": 0}, "tp_size must be greater than 0"),
        ({"cp_size": 0}, "cp_size must be greater than 0"),
        ({"dp_size": 0}, "dp_size must be greater than 0"),
        ({"tp_rank": -1}, "tp_rank must be in \\[0, 2\\)"),
        ({"tp_rank": 2}, "tp_rank must be in \\[0, 2\\)"),
        ({"cp_rank": -1}, "cp_rank must be in \\[0, 2\\)"),
        ({"cp_rank": 2}, "cp_rank must be in \\[0, 2\\)"),
        ({"dp_rank": -1}, "dp_rank must be in \\[0, 2\\)"),
        ({"dp_rank": 2}, "dp_rank must be in \\[0, 2\\)"),
        ({"world_size": 7}, "world_size must equal tp_size \\* cp_size \\* dp_size"),
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
        "cp_size": 2,
        "cp_rank": 1,
        "dp_size": 2,
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
    cp_size: int,
    dp_size: int,
    world_size: int,
    global_rank: int,
) -> SimpleNamespace:
    tp_rank = global_rank % tp_size
    cp_rank = global_rank // tp_size % cp_size
    dp_rank = global_rank // (tp_size * cp_size) % dp_size
    return SimpleNamespace(
        rank=global_rank,
        world_size=world_size,
        attn=SimpleNamespace(
            tp_size=tp_size,
            tp_rank=tp_rank,
            cp_size=cp_size,
            cp_rank=cp_rank,
            dp_size=dp_size,
            dp_rank=dp_rank,
        ),
    )


def test_topology_preserves_typed_attention_coordinates() -> None:
    mapping = _mapping(
        tp_size=2,
        cp_size=2,
        dp_size=2,
        world_size=8,
        global_rank=3,
    )

    topology = PDParallelTopology.from_mapping(mapping)

    assert topology == PDParallelTopology(
        tp_size=2,
        tp_rank=1,
        cp_size=2,
        cp_rank=1,
        dp_size=2,
        dp_rank=0,
        world_size=8,
        global_rank=3,
    )
    assert topology.tp_rank != topology.global_rank % (
        topology.world_size // topology.dp_size
    )


def test_cache_pd_rejects_context_parallel_topology() -> None:
    topology = PDParallelTopology.from_mapping(
        _mapping(
            tp_size=2,
            cp_size=2,
            dp_size=2,
            world_size=8,
            global_rank=6,
        )
    )

    with pytest.raises(ValueError, match="context parallelism.*cp_size=2"):
        topology.require_cache_pd_supported()


def test_cache_pd_accepts_heterogeneous_tp_with_cp_one() -> None:
    topology = PDParallelTopology.from_mapping(
        _mapping(
            tp_size=4,
            cp_size=1,
            dp_size=2,
            world_size=8,
            global_rank=7,
        )
    )

    topology.require_cache_pd_supported()
    assert (topology.tp_size, topology.tp_rank) == (4, 3)
