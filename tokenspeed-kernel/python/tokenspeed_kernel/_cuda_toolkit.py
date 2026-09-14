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

import os
import site
import sys
from pathlib import Path


def prepare_cuda_toolkit_env() -> None:
    """Expose a wheel-provided CUDA toolkit to extensions that require nvcc."""
    site_paths = []
    try:
        site_paths.extend(site.getsitepackages())
    except Exception:
        pass
    site_paths.extend(sys.path)

    candidates = []
    requested_cuda_home = os.environ.get("CUDA_HOME")
    if requested_cuda_home:
        candidates.append(Path(requested_cuda_home))
    for base in site_paths:
        candidates.extend(sorted((Path(base) / "nvidia").glob("cu*"), reverse=True))

    for candidate in candidates:
        candidate = candidate.expanduser().resolve()
        if not (
            (candidate / "include" / "cuda_runtime.h").exists()
            and (candidate / "bin" / "nvcc").exists()
        ):
            continue

        os.environ["CUDA_HOME"] = str(candidate)
        _prepend_env_path("CPATH", candidate / "include")
        _prepend_env_path("PATH", candidate / "bin")
        return


def _prepend_env_path(name: str, path: Path) -> None:
    value = str(path)
    entries = [entry for entry in os.environ.get(name, "").split(os.pathsep) if entry]
    if value not in entries:
        os.environ[name] = os.pathsep.join([value, *entries])
