#!/usr/bin/env python3
"""Reproduce FlashKDA recurrent-state NaNs from a padded prefill bucket.

The launch configuration selects ``--kda-backend flashkda``. Its runtime call
chain is::

    hybrid_linear_attn.kda_paged_prefill
      -> flashkda_nvidia_kda_paged_prefill
      -> flash_kda_chunk_prefill
      -> flash_kda.fwd

Edit the constants in the configuration block below, then run this file with
the same Python environment as ``tokenspeed``. This is an eager operator demo;
it does not capture or replay a CUDA graph.
"""

from __future__ import annotations

import torch
from tokenspeed_kernel.ops.attention import kda_paged_prefill

from tokenspeed.runtime.layers.attention.backends.kda import (
    _slice_kda_prefill_inputs,
)

# ---------------------------------------------------------------------------
# Configuration: edit these values to try another packed prefill shape.
# ---------------------------------------------------------------------------

BUCKET_TOKENS = 384
CU_SEQLENS = (0, 2, 287, 382)
NAN_START_TOKEN = 382

GLOBAL_NUM_HEADS = 96
TENSOR_PARALLEL_SIZE = 8
HEAD_DIM = 128
LOWER_BOUND = -5.0
DTYPE = torch.bfloat16
SEED = 20260806

# Run both the problematic bucket-shaped call and the current runtime behavior.
RUN_UNTRIMMED_BUCKET = True
RUN_RUNTIME_TRIMMED = False


def _validate_config() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if CU_SEQLENS[0] != 0:
        raise ValueError("CU_SEQLENS must start at 0")
    if any(left > right for left, right in zip(CU_SEQLENS, CU_SEQLENS[1:])):
        raise ValueError("CU_SEQLENS must be nondecreasing")
    if CU_SEQLENS[-1] > BUCKET_TOKENS:
        raise ValueError("the real-token count cannot exceed BUCKET_TOKENS")
    if not CU_SEQLENS[-1] <= NAN_START_TOKEN <= BUCKET_TOKENS:
        raise ValueError(
            "NAN_START_TOKEN must be between the real-token count and bucket size"
        )
    if GLOBAL_NUM_HEADS % TENSOR_PARALLEL_SIZE:
        raise ValueError("GLOBAL_NUM_HEADS must be divisible by TENSOR_PARALLEL_SIZE")


def _make_inputs():
    local_num_heads = GLOBAL_NUM_HEADS // TENSOR_PARALLEL_SIZE
    num_sequences = len(CU_SEQLENS) - 1

    torch.manual_seed(SEED)
    query = torch.randn(
        1,
        BUCKET_TOKENS,
        local_num_heads,
        HEAD_DIM,
        device="cuda",
        dtype=DTYPE,
    )
    key = torch.randn_like(query)
    value = (torch.randn_like(query) * 0.1).to(DTYPE)
    gate = torch.randn_like(query)
    beta = torch.randn(
        1,
        BUCKET_TOKENS,
        local_num_heads,
        device="cuda",
        dtype=DTYPE,
    )
    A_log = torch.randn(local_num_heads, device="cuda", dtype=torch.float32) * 0.25
    dt_bias = (
        torch.randn(
            local_num_heads,
            HEAD_DIM,
            device="cuda",
            dtype=torch.float32,
        )
        * 1.4
        - 4.6
    ).clamp_(-9.0, -0.7)
    recurrent_state = torch.zeros(
        num_sequences,
        local_num_heads,
        HEAD_DIM,
        HEAD_DIM,
        device="cuda",
        dtype=torch.float32,
    )
    cu_seqlens = torch.tensor(CU_SEQLENS, device="cuda", dtype=torch.int32)

    value[:, NAN_START_TOKEN:].fill_(float("nan"))
    packed_inputs = (query, key, value, gate, beta)
    parameters = (A_log, dt_bias)
    return packed_inputs, parameters, recurrent_state, cu_seqlens


def _run_case(
    label: str,
    call_tokens: int,
    packed_inputs,
    parameters,
    recurrent_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> None:
    operator_inputs = _slice_kda_prefill_inputs(call_tokens, *packed_inputs)
    result = kda_paged_prefill(
        *operator_inputs,
        *parameters,
        initial_state=recurrent_state,
        cu_seqlens=cu_seqlens,
        lower_bound=LOWER_BOUND,
        solution="flashkda",
    )
    torch.cuda.synchronize()

    state_nan = torch.isnan(result.final_state).flatten(1)
    state_nan_by_sequence = state_nan.any(dim=1).tolist()
    state_nan_count_by_sequence = state_nan.sum(dim=1).tolist()

    final_state = result.final_state

    print(f"\n[{label}]")
    print(f"operator input shape: {tuple(operator_inputs[2].shape)}")
    print(f"{final_state.shape=}")
    print(f"output NaN count: {torch.isnan(result.out).sum().item()}")
    print(f"recurrent_state has NaN: {any(state_nan_by_sequence)}")
    print(f"NaN by sequence: {state_nan_by_sequence}")
    print(f"NaN count by sequence: {state_nan_count_by_sequence}")


def main() -> None:
    _validate_config()
    packed_inputs, parameters, recurrent_state, cu_seqlens = _make_inputs()
    real_tokens = CU_SEQLENS[-1]

    print(f"bucket shape: {BUCKET_TOKENS}")
    print(f"effective tokens: {real_tokens}")
    print(f"cu_seqlens: {list(CU_SEQLENS)}")
    print(f"value[{NAN_START_TOKEN}:] is NaN")

    if RUN_UNTRIMMED_BUCKET:
        _run_case(
            "untrimmed bucket",
            BUCKET_TOKENS,
            packed_inputs,
            parameters,
            recurrent_state,
            cu_seqlens,
        )
    if RUN_RUNTIME_TRIMMED:
        _run_case(
            "runtime-trimmed real prefix",
            real_tokens,
            packed_inputs,
            parameters,
            recurrent_state,
            cu_seqlens,
        )


if __name__ == "__main__":
    main()
