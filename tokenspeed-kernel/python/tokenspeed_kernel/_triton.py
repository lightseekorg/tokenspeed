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
#
# On Ascend platforms, use triton-ascend (3.2.0) directly instead of
# tokenspeed_triton (3.7.10) to avoid JIT API version mismatches.

import contextlib
import importlib
import importlib.abc
import importlib.util
import sys


def _is_ascend_env() -> bool:
    try:
        import torch

        return hasattr(torch, "npu") and torch.npu.is_available()
    except Exception:
        return False


if _is_ascend_env():
    import triton

    from triton import language as tl

    # triton-ascend may not have all submodules that tokenspeed_triton has;
    # provide safe fallbacks.
    try:
        import triton.profiler as proton
    except (ImportError, AttributeError):
        proton = None

    try:
        import triton.experimental.gluon.language as gl
        from triton.experimental import gluon
        from triton.language.core import _aggregate as aggregate
    except (ImportError, AttributeError):
        gl = None
        gluon = None
        aggregate = None

    try:
        from triton.language.extra import libdevice
    except (ImportError, AttributeError):
        libdevice = None

    try:
        from triton.tools.tensor_descriptor import TensorDescriptor
    except (ImportError, AttributeError):
        TensorDescriptor = None

    _TRITON_SRC = "tokenspeed_triton"
    _TRITON_DST = "triton"
else:
    import tokenspeed_triton as triton
    import tokenspeed_triton.experimental.gluon.language as gl
    import tokenspeed_triton.profiler as proton
    from tokenspeed_triton import language as tl
    from tokenspeed_triton.experimental import gluon
    from tokenspeed_triton.language.core import _aggregate as aggregate
    from tokenspeed_triton.language.extra import libdevice
    from tokenspeed_triton.tools.tensor_descriptor import TensorDescriptor

    _TRITON_SRC = "triton"
    _TRITON_DST = "tokenspeed_triton"

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
    """Lazy redirect finder between triton package namespaces."""

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
    """Make one triton namespace resolve to the other in scope.

    On CUDA/ROCm: ``triton[.x.y]`` -> ``tokenspeed_triton[.x.y]``
    On Ascend:    ``tokenspeed_triton[.x.y]`` -> ``triton[.x.y]``
    """
    saved = {
        name: sys.modules[name]
        for name in list(sys.modules)
        if name == _TRITON_SRC or name.startswith(_TRITON_SRC + ".")
    }
    for name in saved:
        del sys.modules[name]

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
