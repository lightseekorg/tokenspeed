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

# Single point of indirection for the triton vendor release package used by
# tokenspeed-kernel. All in-tree code imports triton symbols from here so the
# underlying distribution can be swapped in one place.

import contextlib
import importlib
import importlib.abc
import importlib.util
import os
import sys


def _use_native_triton() -> bool:
    """Return True when in-tree kernels should bind to the native ``triton``.

    The default vendor Triton distribution is ``tokenspeed_triton`` (an
    NVIDIA/AMD-focused fork with gluon/proton extensions). Intel XPU has no
    ``tokenspeed_triton`` build, so on XPU we bind to the platform's native
    ``triton`` (e.g. ``triton-xpu``) instead. Override with
    ``TOKENSPEED_KERNEL_TRITON=triton|tokenspeed_triton``.
    """
    override = os.environ.get("TOKENSPEED_KERNEL_TRITON", "").strip().lower()
    if override == "triton":
        return True
    if override == "tokenspeed_triton":
        return False
    try:
        import torch

        if (
            hasattr(torch, "xpu")
            and torch.xpu.is_available()
            and not torch.cuda.is_available()
        ):
            return True
    except Exception:
        pass
    return False


if _use_native_triton():
    # Intel XPU path: use the native Triton (triton-xpu). Extensions that only
    # exist in tokenspeed_triton (gluon, proton) are optional here because the
    # kernels that use them are gated to other vendors and are not imported on
    # XPU.
    import triton
    from triton import language as tl

    try:
        import triton.experimental.gluon.language as gl
        from triton.experimental import gluon
    except Exception:  # pragma: no cover - gluon is NVIDIA/AMD-only
        gl = None
        gluon = None
    try:
        import triton.profiler as proton
    except Exception:  # pragma: no cover - proton not present on triton-xpu
        proton = None
    try:
        from triton.language.core import _aggregate as aggregate
    except Exception:  # pragma: no cover
        aggregate = None
    try:
        from triton.language.extra import libdevice
    except Exception:  # pragma: no cover
        libdevice = None
    try:
        from triton.tools.tensor_descriptor import TensorDescriptor
    except Exception:  # pragma: no cover
        TensorDescriptor = None
else:
    import tokenspeed_triton as triton
    import tokenspeed_triton.experimental.gluon.language as gl
    import tokenspeed_triton.profiler as proton
    from tokenspeed_triton import language as tl
    from tokenspeed_triton.experimental import gluon
    from tokenspeed_triton.language.core import _aggregate as aggregate
    from tokenspeed_triton.language.extra import libdevice
    from tokenspeed_triton.tools.tensor_descriptor import TensorDescriptor

__all__ = [
    "aggregate",
    "TensorDescriptor",
    "gl",
    "gluon",
    "libdevice",
    "proton",
    "redirect_triton_to_tokenspeed_triton",
    "tl",
    "triton",
]


_TRITON_SRC = "triton"
_TRITON_DST = "tokenspeed_triton"


class _ReuseModuleLoader(importlib.abc.Loader):
    """Loader that re-uses an already-loaded module under an alias name.

    Used by :class:`_TritonRedirectFinder` so Python's import machinery
    reuses the existing ``tokenspeed_triton.*`` module object instead of
    creating a fresh one (which would yield duplicate classes and break
    ``isinstance`` checks).
    """

    def __init__(self, module):
        self._module = module

    def create_module(self, spec):
        return self._module

    def exec_module(self, module):
        return None


class _TritonRedirectFinder(importlib.abc.MetaPathFinder):
    """Lazy ``triton[.x.y]`` -> ``tokenspeed_triton[.x.y]`` redirect finder."""

    def find_spec(self, fullname, path, target=None):
        if fullname != _TRITON_SRC and not fullname.startswith(_TRITON_SRC + "."):
            return None
        target_name = _TRITON_DST + fullname[len(_TRITON_SRC) :]
        try:
            target_module = importlib.import_module(target_name)
        except ImportError:
            return None
        is_pkg = hasattr(target_module, "__path__")
        spec = importlib.util.spec_from_loader(
            fullname, _ReuseModuleLoader(target_module), is_package=is_pkg
        )
        if is_pkg:
            spec.submodule_search_locations = target_module.__path__
        return spec


@contextlib.contextmanager
def redirect_triton_to_tokenspeed_triton():
    """Make ``triton[.x.y]`` resolve to ``tokenspeed_triton[.x.y]`` in scope.

    Use as a context manager around imports of third-party packages that
    bind ``triton`` at module load time::

        from tokenspeed_kernel._triton import redirect_triton_to_tokenspeed_triton

        with redirect_triton_to_tokenspeed_triton():
            import some_third_party_package_that_uses_triton

    Outside the ``with`` block ``sys.modules`` is restored to its prior
    state, so unrelated code is unaffected.
    """
    saved = {
        name: sys.modules[name]
        for name in list(sys.modules)
        if name == _TRITON_SRC or name.startswith(_TRITON_SRC + ".")
    }
    for name in saved:
        del sys.modules[name]

    # Redirect every ``tokenspeed_triton.*`` already in ``sys.modules`` so the
    # protected imports hit the cache directly (no spec/loader machinery, no
    # risk of accidentally instantiating a duplicate module).
    for name, mod in list(sys.modules.items()):
        if name == _TRITON_DST or name.startswith(_TRITON_DST + "."):
            sys.modules[_TRITON_SRC + name[len(_TRITON_DST) :]] = mod

    finder = _TritonRedirectFinder()
    sys.meta_path.insert(0, finder)
    try:
        yield
    finally:
        try:
            sys.meta_path.remove(finder)
        except ValueError:
            pass
        for name in list(sys.modules):
            if (
                name == _TRITON_SRC or name.startswith(_TRITON_SRC + ".")
            ) and name not in saved:
                del sys.modules[name]
        for name, mod in saved.items():
            sys.modules[name] = mod
