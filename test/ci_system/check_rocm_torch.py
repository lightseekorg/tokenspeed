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

"""Check the installed ROCm Torch stack without requiring a visible GPU."""

import argparse
import importlib
import importlib.metadata

from packaging.requirements import Requirement


def check_version(requirement: str, installed: str) -> None:
    expected = Requirement(requirement)
    if not expected.specifier.contains(installed, prereleases=True):
        raise SystemExit(
            f"Installed {expected.name}=={installed} does not satisfy {expected}"
        )
    print(f"Installed {expected.name}=={installed}", flush=True)


def check_stack(
    torch_version: str,
    torchvision_version: str | None,
    device_package: str | None,
) -> None:
    torch = importlib.import_module("torch")
    check_version(f"torch=={torch_version}", torch.__version__)
    if not torch.version.hip or torch.version.cuda is not None:
        raise SystemExit("Expected a ROCm PyTorch build")
    print(f"Torch HIP runtime: {torch.version.hip}", flush=True)
    if torchvision_version:
        torchvision = importlib.import_module("torchvision")
        check_version(f"torchvision=={torchvision_version}", torchvision.__version__)
    if device_package:
        requirement = Requirement(device_package)
        check_version(device_package, importlib.metadata.version(requirement.name))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--torch-version", required=True)
    parser.add_argument("--torchvision-version")
    parser.add_argument("--device-package")
    args = parser.parse_args()
    check_stack(args.torch_version, args.torchvision_version, args.device_package)


if __name__ == "__main__":
    main()
