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

"""Loader boundary for the unchanged vendored Gluon Petit source."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType


def _is_below(path: str, root: Path) -> bool:
    try:
        Path(path).resolve().relative_to(root)
    except ValueError:
        return False
    return True


def load_petit_kernel() -> ModuleType:
    """Load Petit's upstream-rooted packages without editing their imports."""
    vendor_root = Path(__file__).resolve().parent
    for package_name in ("petit_kernel", "lib"):
        module = sys.modules.get(package_name)
        module_file = getattr(module, "__file__", None)
        if module_file is not None and not _is_below(module_file, vendor_root):
            raise RuntimeError(
                f"Cannot load vendored Gluon Petit: {package_name!r} is already "
                f"loaded from {module_file}"
            )

    vendor_path = str(vendor_root)
    if vendor_path not in sys.path:
        sys.path.insert(0, vendor_path)
    importlib.invalidate_caches()
    return importlib.import_module("petit_kernel")


petit_kernel = load_petit_kernel()

__all__ = ["load_petit_kernel", "petit_kernel"]
