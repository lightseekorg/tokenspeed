# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Vendored from vLLM's Kimi-K3.1 specialized decode (cutedsl/mla): the value
# projection with its output gate, adapted to runtime row counts (see value.py).

"""CuTe DSL MLA decode kernels for Kimi-K3.1 DSA layers."""
