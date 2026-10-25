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

"""The serialized tokenizer info the ts-grammar tests compile against.

A nine-token raw vocabulary (`TOY_VOCAB`), its last token the stop token,
serialized by the xgrammar the Python package pins. Rewrite the fixture after
an xgrammar upgrade:

    python tokenspeed-engine/crates/ts-grammar/tests/fixtures/generate.py
"""

from pathlib import Path

import xgrammar

TOY_VOCAB = ["a", "b", "{", "}", '"', "x", ":", ",", "</s>"]


def main() -> None:
    info = xgrammar.TokenizerInfo(
        TOY_VOCAB, vocab_type=xgrammar.VocabType.RAW, stop_token_ids=[8]
    )
    path = Path(__file__).resolve().parent / "toy_tokenizer_info.json"
    path.write_text(info.serialize_json())
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
