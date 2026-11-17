# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Vendored from vLLM's Kimi-K3 specialized decode (CuTe DSL decode kernels).
# PreparedTail takes the rank's up-projection shard rather than narrowing the full
# weight; OprojMailbox lives in common.host; to_cute comes from latent_moe_tail.
# MoEScratch gives the B<=8 and B>8 experts separate workspaces: a B>8 call left
# state the next B<=8 call misread. The experts (synced with the Kimi-K3.1 release)
# keep the TP8 floored hidden scale and the pre-multiplied SiTU scalars. The routing,
# tail and AttnRes kernels take only the release's scheduling changes (early routing
# release with an urgent shared-expert stream, AttnRes re-arm after its store), not its
# numerics changes nor its TP8 B5-8 tail release at entry; the MoE provider takes its
# in-kernel prefetch tables. The attention input projection takes the release's warp
# split-K (gemm/warp_splitk.py, kernel unchanged, compiled from fake operands like the
# cluster split-K) and its per-row-count dispatch (gemm/decode.py) without the cuBLAS
# engine: the cluster split-K stays prepared at every row count, for the rows above
# 16 and the operands the warp split-K cannot take.
# The o-projection producers take their output rows (``n``, K3's 7168 by default)
# rather than a fixed 112-CTA grid.

"""CuTe DSL decode kernels for Kimi-K3: MoE front, experts and tail."""
