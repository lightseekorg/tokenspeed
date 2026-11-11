# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Vendored from vLLM's Kimi-K3.1 specialized decode (cutedsl/dsa): the fused decode
# indexer, adapted to FP32 logits and an FP32 top-k (see indexer.py), and the
# cluster index front, extended to the q_b rows and TokenSpeed's norm, quantization
# and index-K write arithmetic (see front.py).

"""CuTe DSL decode front and indexer kernels for Kimi-K3.1 DSA layers."""
