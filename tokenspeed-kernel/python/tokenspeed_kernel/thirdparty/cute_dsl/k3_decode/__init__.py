# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Vendored from vLLM's Kimi-K3 specialized decode (CuTe DSL decode kernels).
# PreparedTail takes the rank's up-projection shard rather than narrowing the full
# weight; OprojMailbox lives in common.host; to_cute comes from latent_moe_tail.

"""CuTe DSL decode kernels for Kimi-K3: MoE front, experts and tail."""
