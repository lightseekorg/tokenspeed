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

"""Acquire official Qwen3.5 checkpoints with per-file immutable revisions."""

import argparse
import json
from pathlib import Path
from urllib.request import urlopen

from download_qwen35 import download_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sizes", nargs="+", required=True, choices=["4B", "9B", "27B"]
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifests", type=Path, required=True)
    args = parser.parse_args()
    args.manifests.mkdir(parents=True, exist_ok=True)
    for size in args.sizes:
        source = f"Qwen/Qwen3.5-{size}"
        url = f"https://modelscope.cn/api/v1/models/{source}/repo/files?Revision=master&Recursive=true"
        with urlopen(url, timeout=60) as response:
            listing = json.load(response)
        assert listing["Success"], listing
        entries = [
            dict(
                file=f["Path"],
                sha256=f["Sha256"],
                size=f["Size"],
                revision=f["Revision"],
            )
            for f in listing["Data"]["Files"]
            if f["Type"] == "blob" and f["Path"] not in ("README.md", ".gitattributes")
        ]
        manifest = dict(source=source, endpoint="https://modelscope.cn", files=entries)
        path = args.manifests / f"qwen35-{size.lower()}-model-manifest.json"
        if path.exists():
            manifest = json.loads(path.read_text())
        else:
            path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(
            f"Acquiring {source}: {sum(f['size'] for f in manifest['files']) / 2**30:.2f} GiB",
            flush=True,
        )
        download_model(path, args.output / f"Qwen3.5-{size}")


if __name__ == "__main__":
    main()
