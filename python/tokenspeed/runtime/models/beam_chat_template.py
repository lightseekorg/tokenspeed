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

"""Create an explicit SMG template override for the qualified Beam tokenizer.

Only the exact released Beam template is supported. Its generation markers
track assistant spans in Transformers but are unknown to the pinned gateway.
The replacement is qualified for this template's scopes and byte rendering;
it is not a general transformation for arbitrary Jinja templates.
"""

import argparse
import hashlib
import json
from pathlib import Path

BEAM_CHAT_TEMPLATE_SHA256 = (
    "24371da068468a5d15940c9af90dbf3d2ce61a6aa99fc122384aed49fc6c3c68"
)


def prepare_beam_gateway_template(source: Path, output: Path) -> dict[str, str]:
    """Write a new gateway-only template and return paths and SHA256 digests.

    ``source`` must be the qualified released Beam chat template. ``output``
    must be a new file outside its checkpoint directory. Source aliases and
    existing outputs are refused; the checkpoint and engine template stay
    unchanged. Unknown, edited or malformed templates require qualification
    before their source hash can be admitted.
    """
    checkpoint_directory = source.parent.resolve(strict=True)
    source = source.resolve(strict=True)
    resolved_output = output.resolve()
    if resolved_output == source or (output.exists() and output.samefile(source)):
        raise ValueError("The gateway template output must not alias its source")
    if any(
        resolved_output.is_relative_to(parent)
        for parent in (checkpoint_directory, source.parent)
    ):
        raise ValueError(
            "The gateway template output must be outside the checkpoint directory"
        )
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"The gateway template output already exists: {output}")

    original = source.read_bytes()
    source_sha = hashlib.sha256(original).hexdigest()
    if source_sha != BEAM_CHAT_TEMPLATE_SHA256:
        raise ValueError(
            f"Unsupported Beam chat template SHA256 {source_sha}; only the qualified released template is supported"
        )
    compatible = original.replace(b"{%- generation %}", b"{%- if true %}").replace(
        b"{%- endgeneration %}", b"{%- endif %}"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        stream.write(compatible)
    return {
        "source": str(source),
        "source_sha256": source_sha,
        "output": str(output.resolve()),
        "output_sha256": hashlib.sha256(compatible).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare_beam_gateway_template(args.source, args.output), indent=2))


if __name__ == "__main__":
    main()
