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

"""Structural contracts of the sonic-sampler patch set and facade."""

import subprocess
import sys
import textwrap

import pytest

pytest.importorskip("sonic_sampler")

# The facade must bind sonic to the vendored Triton before anything imports sonic.
from tokenspeed_kernel.ops.sampling.sonic import measured_dispatch  # noqa: E402
from tokenspeed_kernel.thirdparty import sonic as patchset  # noqa: E402
from tokenspeed_kernel.thirdparty.sonic import noise  # noqa: E402


def _bucket(block_n: int):
    from sonic_sampler.interface.dispatch import (
        BatchBucket,
        PriorityBucket,
        RuntimeConfig,
        VocabBucket,
    )

    config = RuntimeConfig(
        priority=0, block_n=block_n, strategy="bitonic", first_warps=8, second_warps=2
    )
    return VocabBucket(
        size=262144, batch=PriorityBucket([BatchBucket(size=1 << 16, config=[config])])
    )


def test_patch_anchor_drift_is_an_import_error() -> None:
    """The facade gates on ``except ImportError``: an upstream source change
    that moves an anchor must surface as that, not as a RuntimeError that
    would break every sampling backend's registry import."""
    with pytest.raises(ImportError, match="anchor not found"):
        patchset._patched(
            noise,
            "gumbel_noise",
            [("no-such-anchor\n", "x\n")],
            patchset._namespace(noise),
        )


def test_missing_patch_target_is_an_import_error() -> None:
    with pytest.raises(ImportError, match="does not define"):
        patchset._patched(noise, "no_such_kernel", [], patchset._namespace(noise))


def test_namespace_refuses_unbound_patch_target() -> None:
    with pytest.raises(ImportError, match="does not bind"):
        patchset._namespace(noise, no_such_kernel=object())


def test_measured_dispatch_guards_the_scratchpad_size() -> None:
    """The tiling sizes its scratchpad for the packaged bucket's smallest
    block_n; a measured entry below it would write out of bounds."""
    assert measured_dispatch(100, _bucket(4096)).block_n == 4096
    assert measured_dispatch(90, _bucket(4096)) is None
    with pytest.raises(ValueError, match="undersized"):
        measured_dispatch(100, _bucket(8192))


def _run(code: str) -> None:
    subprocess.run([sys.executable, "-c", textwrap.dedent(code)], check=True)


def test_import_order_binds_sonic_to_vendored_triton() -> None:
    """Importing the patch set first still binds sonic's kernels to
    tokenspeed_triton (the helpers it re-jits against)."""
    _run("""
        import tokenspeed_kernel.thirdparty.sonic  # noqa: F401
        import sonic_sampler.ops.prologue as prologue
        import tokenspeed_triton
        assert prologue.jit is tokenspeed_triton.jit, prologue.jit
        """)


@pytest.mark.parametrize(
    "leaf", ["sonic_sampler.ops.prologue", "sonic_sampler.core.flags"]
)
def test_sonic_bound_to_another_triton_is_refused(leaf: str) -> None:
    """A sonic already bound to stock triton (through any of its modules)
    cannot host the vendored noise helpers: the patch set refuses it at import
    instead of failing at launch."""
    _run(f"""
        import {leaf}  # noqa: F401
        try:
            import tokenspeed_kernel.thirdparty.sonic  # noqa: F401
        except ImportError as exc:
            assert "tokenspeed_triton" in str(exc), exc
        else:
            raise AssertionError("patch set accepted a sonic bound to another Triton")
        """)
