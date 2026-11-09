# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Vendored from vLLM's Kimi-K3.1 specialized decode (cutedsl/dsa): the fused decode
# indexer, adapted to FP32 logits and an FP32 top-k (see indexer.py).

"""CuTe DSL decode indexer kernels for Kimi-K3.1 DSA layers."""
