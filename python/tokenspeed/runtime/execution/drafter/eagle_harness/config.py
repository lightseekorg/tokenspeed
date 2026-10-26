# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""EAGLE-Harness configuration, built from the server arguments."""

from __future__ import annotations

import dataclasses
import json
import os
from collections.abc import Callable
from typing import TYPE_CHECKING, TypeVar

import torch.distributed as dist

from tokenspeed.runtime.execution.drafter.eagle_harness.cost import (
    CHAIN_RULE_MAX_BATCH_SIZE,
    max_batch_size,
    measured_batch_sizes,
)

if TYPE_CHECKING:
    from tokenspeed.runtime.configs.model_config import ModelConfig
    from tokenspeed.runtime.utils.server_args import ServerArgs

__all__ = [
    "ChainWidthEntry",
    "ChainWidthTable",
    "EagleHarnessConfig",
    "chain_table_width",
    "chain_width_table",
    "check_chain_cost_domain",
    "decode_graph_floors",
    "first_rank_value",
    "load_run_config",
    "validate_server_args",
]


@dataclasses.dataclass(frozen=True)
class ChainWidthEntry:
    """One ``MIN_BS:WIDTH`` entry of ``--speculative-chain-width-table``."""

    min_bs: int  # live requests of a round from which the entry decides
    width: int  # verify width: width - 1 drafted tokens and the root


# Entries by increasing min_bs; empty when the confidence rule decides every round.
ChainWidthTable = tuple[ChainWidthEntry, ...]


def _parse_chain_width_table(text: str, full_width: int) -> ChainWidthTable:
    """``--speculative-chain-width-table``: ``"MIN_BS:WIDTH,..."``, or ``"off"``
    (the confidence rule decides every round). Widths above the chain's full
    width ``full_width`` (``--speculative-num-draft-tokens``) are capped to it.

    Raises:
        ValueError: An entry that is not two integers, a batch size below 1, a
            width below 2 (one draft token), or a batch size given twice.
    """
    if text.strip().lower() == "off":
        return ()
    entries: dict[int, int] = {}
    for item in text.split(","):
        parts = item.strip().split(":")
        try:
            if len(parts) != 2:
                raise ValueError
            min_bs, width = int(parts[0]), int(parts[1])
        except ValueError:
            raise ValueError(
                f"--speculative-chain-width-table entry {item.strip()!r} is not "
                f"MIN_BS:WIDTH (two integers), in {text!r}; 'off' disables the table"
            ) from None
        if min_bs < 1 or width < 2:
            raise ValueError(
                f"--speculative-chain-width-table entry {item.strip()!r} needs a "
                "batch size of at least 1 and a width of at least 2 (one draft token)"
            )
        if min_bs in entries:
            raise ValueError(
                f"--speculative-chain-width-table gives batch size {min_bs} twice"
            )
        entries[min_bs] = min(width, full_width)
    return tuple(ChainWidthEntry(b, w) for b, w in sorted(entries.items()))


def chain_table_width(table: ChainWidthTable, bs: int) -> int | None:
    """The verify width the table fixes for a round of ``bs`` live requests:
    the width of the entry with the largest ``min_bs <= bs``; None where the
    confidence rule decides (below the first entry, or no table)."""
    width = None
    for entry in table:
        if bs < entry.min_bs:
            break
        width = entry.width
    return width


def _chain_rule_max_batch_size(table: ChainWidthTable, max_requests: int) -> int:
    """The most live requests of a round the confidence rule decides (0: none)."""
    return min(max_requests, table[0].min_bs - 1) if table else max_requests


def check_chain_cost_domain(
    measured: list[int], table: ChainWidthTable, max_requests: int, where: str
) -> None:
    """Refuse cost data ``where`` measured at the batch sizes ``measured`` when
    the confidence rule decides rounds above the largest of them: the nearest
    measured batch would price them silently.

    Raises:
        ValueError: The rule decides past the measured batch sizes.
    """
    rule_max = _chain_rule_max_batch_size(table, max_requests)
    if measured and rule_max > max(measured):
        raise ValueError(
            f"EAGLE-Harness chain cost data in {where} is measured up to "
            f"{max(measured)} requests, but the confidence rule decides rounds of "
            f"up to {rule_max}: start --speculative-chain-width-table at "
            f"{max(measured) + 1} or below, lower --max-num-seqs, or measure the "
            "cost data at larger batches"
        )


@dataclasses.dataclass(frozen=True)
class EagleHarnessConfig:
    """``--speculative-exec-mode``, ``--speculative-data-path``,
    ``--speculative-compute-intensity`` and ``--speculative-chain-width-table``;
    ``sampled_trees`` with ``--enable-speculative-sampling``."""

    exec_mode: int
    data_path: str
    compute_intensity: str
    is_moe_target: bool
    # Temperature > 0 requests sample children (a chain: one) from the draft q, accepted by rejection.
    sampled_trees: bool
    # The TP-synced seed of the children's draws.
    seed: int
    # The token exec mode 2 commits to end a measured batch (None: the model declares no EOS).
    eos_token_id: int | None
    # Chain verify width per live batch size, exec modes 0/1; empty: the confidence rule decides.
    chain_width_table: ChainWidthTable
    # --max-num-seqs: the most live requests of a round.
    max_requests: int
    # (padded batch, floor) per decode graph: it serves floor + 1 to padded batch live requests.
    decode_graph_floors: tuple[tuple[int, int], ...]

    @classmethod
    def from_server_args(
        cls, server_args: ServerArgs, model_config: ModelConfig
    ) -> EagleHarnessConfig | None:
        if not server_args.eagle_harness:
            return None
        eos_token_ids = model_config.hf_eos_token_id
        return cls(
            exec_mode=server_args.speculative_exec_mode,
            data_path=server_args.speculative_data_path,
            compute_intensity=server_args.speculative_compute_intensity,
            is_moe_target=_routes_experts(model_config.hf_text_config),
            sampled_trees=server_args.enable_speculative_sampling,
            seed=server_args.seed,
            eos_token_id=min(eos_token_ids) if eos_token_ids else None,
            chain_width_table=chain_width_table(server_args),
            max_requests=server_args.max_num_seqs,
            decode_graph_floors=decode_graph_floors(server_args),
        )


@dataclasses.dataclass(frozen=True)
class _DecodeLadderArgs:
    """The executor config fields the decode graph ladder is built from
    (``forward_step.get_batch_sizes_to_capture``), as
    ``ModelExecutorConfig.from_server_args`` takes them."""

    cudagraph_capture_sizes: list[int] | None
    max_num_seqs: int
    data_parallel_size: int
    disable_cuda_graph_padding: bool
    max_cudagraph_capture_size: int


def decode_graph_floors(server_args: ServerArgs) -> tuple[tuple[int, int], ...]:
    """Each decode graph's padded batch and the largest live batch size below
    the forwards replaying it: the ladder entry before it when batches pad up
    to the ladder, else one below it. Empty under attention data parallelism,
    where any smaller live batch can replay a graph (a rank pads to the
    largest rank's batch)."""
    from tokenspeed.runtime.execution.forward_step import get_batch_sizes_to_capture

    dp_size = server_args.mapping.attn.dp_size
    if dp_size > 1:
        return ()
    ladder = get_batch_sizes_to_capture(
        _DecodeLadderArgs(
            cudagraph_capture_sizes=server_args.cudagraph_capture_sizes,
            max_num_seqs=server_args.max_num_seqs,
            data_parallel_size=dp_size,
            disable_cuda_graph_padding=server_args.disable_cuda_graph_padding,
            max_cudagraph_capture_size=server_args.max_cudagraph_capture_size,
        )
    )
    if server_args.disable_cuda_graph_padding:
        return tuple((bs, bs - 1) for bs in ladder)
    return tuple(zip(ladder, [0, *ladder[:-1]]))


def chain_width_table(server_args: ServerArgs) -> ChainWidthTable:
    """The width table a server runs: ``--speculative-chain-width-table`` for
    EAGLE-Harness chains in exec modes 0 and 1, empty otherwise (trees, and the
    cost measurement, exec mode 2, which measures every width)."""
    if (
        server_args.speculative_compute_intensity != "chain_intensity"
        or server_args.speculative_exec_mode not in (0, 1)
    ):
        return ()
    return _parse_chain_width_table(
        server_args.speculative_chain_width_table,
        server_args.speculative_num_draft_tokens,
    )


def _routes_experts(hf_config) -> bool:
    """Whether a target routes tokens to experts (the EAGLE-Harness doubles
    its depth loss for MoE targets). Configs may set the expert counts to
    None for dense variants."""
    return (
        (getattr(hf_config, "num_experts", None) or 0) > 0
        or (getattr(hf_config, "num_local_experts", None) or 0) > 0
        or (getattr(hf_config, "n_routed_experts", None) or 0) > 0
        or any(
            "moe" in arch.lower()
            for arch in getattr(hf_config, "architectures", None) or ()
        )
    )


_T = TypeVar("_T")


def first_rank_value(load: Callable[[], _T], group: dist.ProcessGroup | None) -> _T:
    """``load()`` run by the first rank of ``group`` alone, its value (or the
    exception it raised) the same on every rank: the harness files decide the
    collectives a round runs, so the ranks must not read them apart.

    Args:
        load: Reads a harness file (cost data, a run config).
        group: The ranks' CPU group of their own (None: this rank alone).

    Returns:
        ``load``'s value on the first rank.
    """
    if group is None:
        return load()
    payload: list = [None]
    if dist.get_rank(group) == 0:
        try:
            payload[0] = load()
        except Exception as exc:  # noqa: BLE001 — raised on every rank below
            payload[0] = exc
    dist.broadcast_object_list(payload, src=dist.get_global_rank(group, 0), group=group)
    if isinstance(payload[0], Exception):
        raise payload[0]
    return payload[0]


def load_run_config(path: str) -> dict:
    """A ``<data path>/config/*.json`` file the measurement or benchmark writes."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    # One mapping, or a one-element JSON list holding it.
    return data[0] if isinstance(data, list) else data


def validate_server_args(server_args: ServerArgs) -> None:
    """Refuse EAGLE-4 settings it cannot serve. Exec mode -1 (static chains
    and trees) only checks speculative sampling; exec modes 0 to 2
    (EAGLE-Harness) check every limit."""
    mode = server_args.speculative_exec_mode
    if mode < 0:
        if (
            server_args.speculative_algorithm == "EAGLE4"
            and server_args.enable_speculative_sampling
        ):
            raise ValueError(
                "--enable-speculative-sampling with EAGLE4 needs EAGLE-Harness "
                "(--speculative-exec-mode 0 or 1): "
                "a static draft's candidate head scores no full-vocabulary draft "
                "distribution"
            )
        return
    if mode > 2:
        raise ValueError(f"--speculative-exec-mode must be -1, 0, 1 or 2, got {mode}")
    intensity = server_args.speculative_compute_intensity
    if intensity not in ("high_intensity", "low_intensity", "chain_intensity"):
        raise ValueError(f"unknown --speculative-compute-intensity {intensity!r}")
    # Refused at startup when malformed, also where it does not apply.
    _parse_chain_width_table(
        server_args.speculative_chain_width_table,
        server_args.speculative_num_draft_tokens,
    )
    if server_args.speculative_algorithm != "EAGLE4":
        raise ValueError(
            "--speculative-exec-mode >= 0 (EAGLE-Harness) needs "
            f"--speculative-algorithm EAGLE4, got {server_args.speculative_algorithm}"
        )
    if mode == 2 and server_args.enable_speculative_sampling:
        raise ValueError(
            "EAGLE-Harness cost measurement (exec mode 2) times greedy rounds; "
            "drop --enable-speculative-sampling"
        )
    if server_args.enable_speculative_sampling and (
        server_args.sampling_backend != "flashinfer"
    ):
        raise ValueError(
            "EAGLE-Harness verifies sampled drafts by tree rejection sampling, "
            "which only --sampling-backend flashinfer has; got "
            f"{server_args.sampling_backend}"
        )
    topk = server_args.speculative_eagle_topk
    if topk not in (1, 2, 4, 8):
        raise ValueError(
            "EAGLE-Harness widths are powers of two (EAGLE-4 rounds them up): "
            f"--speculative-eagle-topk must be 1, 2, 4 or 8, got {topk}"
        )
    if server_args.speculative_data_path is None:
        raise ValueError(
            "EAGLE-Harness needs its cost data (or, measuring it in exec mode "
            "2, its run directory): --speculative-data-path"
        )
    chain = server_args.speculative_compute_intensity == "chain_intensity"
    if (topk == 1) != chain:
        raise ValueError(
            "EAGLE-Harness chains (--speculative-eagle-topk 1) run "
            "--speculative-compute-intensity chain_intensity, and trees "
            f"high_intensity or low_intensity: {server_args.speculative_eagle_topk=}, "
            f"{server_args.speculative_compute_intensity=}"
        )
    if chain:
        if (
            server_args.speculative_num_steps > 15
            or server_args.speculative_num_draft_tokens > 16
        ):
            raise ValueError(
                "EAGLE-Harness chains draft at most 15 steps and verify at most "
                f"16 tokens: {server_args.speculative_num_steps=}, "
                f"{server_args.speculative_num_draft_tokens=}"
            )
    cap = max_batch_size(server_args.speculative_compute_intensity)
    if mode == 2:
        # The measurement samples the cost data's batch sizes.
        cap = min(cap, CHAIN_RULE_MAX_BATCH_SIZE)
    if server_args.max_num_seqs > cap:
        raise ValueError(
            f"EAGLE-Harness {'chains' if chain else 'trees'} serve at most {cap} "
            f"requests per rank{' in exec mode 2' if mode == 2 else ''}; got "
            f"--max-num-seqs {server_args.max_num_seqs}"
        )
    if chain and mode != 2:
        _validate_chain_rule_domain(server_args)
    steps = server_args.speculative_num_steps
    # Kernel limits: ancestor masks <= two 64-bit words, ranking <= 1024 nodes, sampled depths < 64.
    if (
        not 2 <= steps <= 64
        or (steps - 1) * topk > 128
        or 1 + topk + (steps - 1) * topk * topk > 1024
    ):
        raise ValueError(
            "EAGLE-Harness trees need 2..64 steps, "
            "(steps - 1) * topk <= 128 lane slots and 1 + topk + (steps - 1) "
            f"* topk^2 <= 1024 drafted nodes: {topk=}, {steps=}"
        )
    if mode != 2 and steps > 32:
        raise ValueError(
            "EAGLE-Harness decides its rounds on the device for at most 32 "
            f"steps (exec mode 2 measures up to 64): {steps=}"
        )


def _validate_chain_rule_domain(server_args: ServerArgs) -> None:
    """Refuse a chain whose confidence rule would decide rounds of more than
    ``CHAIN_RULE_MAX_BATCH_SIZE`` requests (its cost data and the measurement
    stop there), or of more requests than its cost data measured (checked here
    when the cost directory exists, and again when the drafter loads it)."""
    table = chain_width_table(server_args)
    rule_max = _chain_rule_max_batch_size(table, server_args.max_num_seqs)
    if rule_max > CHAIN_RULE_MAX_BATCH_SIZE:
        raise ValueError(
            "the EAGLE-Harness chain confidence rule decides rounds of at most "
            f"{CHAIN_RULE_MAX_BATCH_SIZE} requests, but would decide up to "
            f"{rule_max} (--max-num-seqs {server_args.max_num_seqs}): start "
            f"--speculative-chain-width-table at {CHAIN_RULE_MAX_BATCH_SIZE + 1} "
            "or below"
        )
    cost_dir = os.path.join(server_args.speculative_data_path, "cost")
    if os.path.isdir(cost_dir):
        check_chain_cost_domain(
            measured_batch_sizes(cost_dir),
            table,
            server_args.max_num_seqs,
            cost_dir,
        )
