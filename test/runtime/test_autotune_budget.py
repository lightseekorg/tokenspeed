import argparse
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tokenspeed.runtime.execution.model_executor import (
    _resolve_autotune_num_tokens,
)
from tokenspeed.runtime.utils import server_args as server_args_module
from tokenspeed.runtime.utils.server_args import ServerArgs


@pytest.mark.parametrize(
    ("chunked_prefill", "cap", "expected"),
    [
        (8192, 8192, 8192),
        (16384, 8192, 8192),
        (4096, 8192, 4096),
        (16384, 0, 0),
        (-1, 8192, 0),
    ],
)
def test_autotune_budget_is_independent_of_scheduler_budget(
    chunked_prefill: int, cap: int, expected: int
) -> None:
    assert _resolve_autotune_num_tokens(chunked_prefill, cap) == expected


def test_autotune_budget_cli() -> None:
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    args = parser.parse_args(["--autotune-max-num-tokens", "4096"])
    assert args.autotune_max_num_tokens == 4096


def test_amd_dspark_graph_capture_is_capped_before_allocation() -> None:
    with patch.object(ServerArgs, "__post_init__"):
        args = ServerArgs(
            model="test/model",
            speculative_algorithm="DSPARK",
            max_cudagraph_capture_size=48,
            gpu_memory_utilization=0.88,
        )
    args.mapping = SimpleNamespace(world_size=8)
    platform = SimpleNamespace(is_amd=True, is_nvidia=False)
    with (
        patch.object(server_args_module, "current_platform", return_value=platform),
        patch.object(
            server_args_module, "get_amdgpu_memory_capacity", return_value=288_000
        ),
    ):
        args.resolve_memory_and_scheduling()
    assert args.max_cudagraph_capture_size == 16
    assert args.max_num_seqs == 16
