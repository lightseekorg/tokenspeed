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

"""Check the installed FlashMLA wheel before publishing a CUDA kernel wheel."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import re
import subprocess
from pathlib import Path

from packaging.requirements import Requirement


def verify_torch_cuda(torch_cuda: str | None, cuda_major: int) -> None:
    """Reject a CPU Torch build or a different CUDA major before loading kernels."""
    if torch_cuda is None or int(torch_cuda.split(".", maxsplit=1)[0]) != cuda_major:
        raise ValueError(f"Expected Torch CUDA {cuda_major}, found {torch_cuda!r}")


def verify_binary_reports(
    elf_listing: str, dynamic_section: str, cuda_major: int
) -> None:
    """Require both legacy Hopper sparse kernels and the selected CUDA runtime."""
    if cuda_major not in (12, 13):
        raise ValueError(f"Unsupported CUDA major version: {cuda_major}")
    missing = [
        name
        for name in ("v32_persistent_h64", "v32_persistent_h128")
        if not re.search(rf"\b{name}\.sm_90a?\.cubin\b", elf_listing)
    ]
    if missing:
        raise ValueError(
            f"FlashMLA wheel is missing Hopper kernels: {', '.join(missing)}"
        )
    runtimes = set(re.findall(r"\(NEEDED\).*\[libcudart\.so\.(\d+)\]", dynamic_section))
    if runtimes != {str(cuda_major)}:
        raise ValueError(
            f"FlashMLA wheel must link libcudart.so.{cuda_major}; "
            f"found CUDA runtime majors {sorted(runtimes)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda-major", type=int, choices=(12, 13), required=True)
    parser.add_argument("--cuobjdump", type=Path, required=True)
    args = parser.parse_args()
    import torch

    verify_torch_cuda(torch.version.cuda, args.cuda_major)
    distribution = importlib.metadata.distribution("tokenspeed-flashmla")
    for raw in distribution.requires or ():
        requirement = Requirement(raw)
        if (
            requirement.name == "torch"
            and torch.__version__ not in requirement.specifier
        ):
            raise ValueError(
                f"Installed Torch {torch.__version__} does not satisfy {raw}"
            )
    libraries = [
        Path(distribution.locate_file(path))
        for path in distribution.files or ()
        if str(path).startswith("flash_mla/") and str(path).endswith(".so")
    ]
    if len(libraries) != 1:
        raise ValueError(f"Expected one FlashMLA native library; found {libraries}")
    library = libraries[0]
    elf_listing = subprocess.check_output(
        [str(args.cuobjdump), "--list-elf", str(library)], text=True
    )
    dynamic_section = subprocess.check_output(
        ["readelf", "-d", str(library)], text=True
    )
    verify_binary_reports(elf_listing, dynamic_section, args.cuda_major)
    print(
        json.dumps(
            {
                "distribution": distribution.metadata["Name"],
                "version": distribution.version,
                "cuda_major": args.cuda_major,
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "hopper_sparse_kernels": ["v32_persistent_h64", "v32_persistent_h128"],
                "library": str(library),
                "sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
            }
        )
    )


if __name__ == "__main__":
    main()
