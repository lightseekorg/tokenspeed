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

"""CPU checks of the Gluon Petit kernel-variant enumeration startup compiles.

Imports only the vendored ``petit_kernel`` package (no device needed).
"""

import sys
from pathlib import Path

import pytest

_VENDOR = (
    Path(__file__).resolve().parents[4]
    / "python"
    / "tokenspeed_kernel"
    / "thirdparty"
    / "gluon_petit"
)
if str(_VENDOR) not in sys.path:
    sys.path.insert(0, str(_VENDOR))

import petit_kernel as pk  # noqa: E402


def _config(world_size: int, num_experts: int, top_k: int, capacity: int):
    # DSv4.1 Flash (384 experts, top 6) and its DSpark draft (128, top 3).
    return pk.MegaMoeConfig(
        world_size=world_size,
        num_experts=num_experts,
        topk=top_k,
        model_dim=5120,
        activation=pk.MegaMoeActivation.mxfp4,
        activation_function=pk.MegaMoeActivationFunction("silu_clamped"),
        stages=pk.MegaMoeStages.two_stage,
        inter_dim=2560,
        has_bias=False,
        max_tokens_per_rank=capacity,
    )


@pytest.mark.parametrize("experts,top_k", [(384, 6), (128, 3)])
@pytest.mark.parametrize(
    "world_size,capacity", [(4, 1024), (4, 2048), (4, 4096), (4, 7808), (8, 1024)]
)
def test_variant_counts_cover_every_launchable_key(
    experts, top_k, world_size, capacity
) -> None:
    config = _config(world_size, experts, top_k, capacity)
    counts = config.variant_token_counts()
    every_key = {config.variant_key(n) for n in range(capacity + 1)}
    covered = [config.variant_key(n) for n in counts]
    # One count per key, every key a launch can reach, all within capacity.
    assert len(covered) == len(set(covered))
    assert set(covered) == every_key
    assert all(0 <= n <= capacity for n in counts)
    assert counts == tuple(sorted(counts))


@pytest.mark.parametrize(
    "experts,top_k,expected",
    [
        # Target: m32 up to 479 rows, m64/w4 at 480-1007 (cta192 below 512),
        # m64/w8 from 1008, each in Triton's 1 / multiple-of-16 / other class.
        (384, 6, (0, 1, 2, 480, 481, 512, 513, 1008, 1009)),
        # Draft: cta128 below 12 rows, cta64 below 24, m64/w4 at 352-703,
        # m64/w8 from 704.
        (128, 3, (0, 1, 2, 12, 16, 24, 32, 352, 353, 704, 705)),
    ],
)
@pytest.mark.parametrize("capacity", [1024, 4096])
def test_ep4_dsv41_variant_counts(experts, top_k, expected, capacity) -> None:
    assert _config(4, experts, top_k, capacity).variant_token_counts() == expected


def test_prefill_tail_variant_is_compiled() -> None:
    # A 50,000-token prompt leaves an 848-row prefill tail at 1,024, 2,048 and
    # 4,096-token chunks; its m64/w4 stage-1 variant compiled for 27-28 s
    # inside a serving step before startup compiled every variant.
    config = _config(4, 384, 6, 4096)
    assert config.variant_key(848) == ("cta56", "m64_w4", "div16")
    assert config.variant_key(848) in {
        config.variant_key(n) for n in config.variant_token_counts()
    }


@pytest.mark.parametrize(
    "value,expected",
    [
        (0, "div16"),
        (1, "one"),
        (2, "plain"),
        (16, "div16"),
        (848, "div16"),
        (849, "plain"),
    ],
)
def test_triton_int_class(value, expected) -> None:
    assert pk._triton_int_class(value) == expected
