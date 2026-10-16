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

"""CPU-only safety tests for the explicitly qualified Beam gateway template."""

import hashlib
import os
import sys
from pathlib import Path

import pytest

from tokenspeed.runtime.models import beam_chat_template
from tokenspeed.runtime.models.beam_chat_template import prepare_beam_gateway_template

_TEST_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_TEST_DIR))
from ci_system.ci_register import register_cuda_ci  # noqa: E402

register_cuda_ci(est_time=5, suite="runtime-1gpu")


@pytest.mark.parametrize("prefix,suffix", [("header \n", "\n footer"), ("\t", "\r\n")])
def test_qualified_rewrite_preserves_every_other_byte(
    tmp_path, monkeypatch, prefix, suffix
):
    original = (
        prefix + '{%- generation %}\n{{ "assistant" }}\n{%- endgeneration %}' + suffix
    ).encode()
    expected = (
        prefix + '{%- if true %}\n{{ "assistant" }}\n{%- endif %}' + suffix
    ).encode()
    # Unit-only admission of a small fixture exercises the mechanics in CI;
    # production acceptance remains pinned to the released Beam asset hash.
    monkeypatch.setattr(
        beam_chat_template,
        "BEAM_CHAT_TEMPLATE_SHA256",
        hashlib.sha256(original).hexdigest(),
    )
    source = tmp_path / "checkpoint" / "chat_template.jinja"
    source.parent.mkdir()
    source.write_bytes(original)
    output = tmp_path / "gateway.jinja"
    result = prepare_beam_gateway_template(source, output)
    assert source.read_bytes() == original
    assert output.read_bytes() == expected
    assert result["source_sha256"] == hashlib.sha256(original).hexdigest()
    assert result["output_sha256"] == hashlib.sha256(expected).hexdigest()


@pytest.mark.parametrize(
    "content",
    [
        "{% generation %}assistant{% endgeneration %}",
        "{%- generation %}assistant{%- endgeneration %}",
        '{{ "{%- generation %}" }}',
        "{% raw %}{%- generation %}{%- endgeneration %}{% endraw %}",
        "{# {%- generation %}{%- endgeneration %} #}",
        "{%- generation %}unclosed",
        "{%- endgeneration %}unopened",
        "{%- generation %}{%- generation %}nested{%- endgeneration %}{%- endgeneration %}",
        "{%- generation %}{% set outer = 1 %}{%- endgeneration %}",
    ],
)
def test_unknown_templates_are_refused_without_editing_literal_or_other_content(
    tmp_path, content
):
    source = tmp_path / "checkpoint" / "chat_template.jinja"
    source.parent.mkdir()
    source.write_bytes(content.encode())
    output = tmp_path / "derived" / "gateway.jinja"
    with pytest.raises(ValueError, match="only the qualified released template"):
        prepare_beam_gateway_template(source, output)
    assert source.read_bytes() == content.encode()
    assert not output.parent.exists()


@pytest.mark.parametrize("alias", ["same", "symlink", "hardlink"])
def test_source_aliases_are_never_overwritten(tmp_path, alias):
    source = tmp_path / "checkpoint" / "chat_template.jinja"
    source.parent.mkdir()
    source.write_bytes(b"original")
    output = source if alias == "same" else tmp_path / "alias.jinja"
    if alias == "symlink":
        output.symlink_to(source)
    elif alias == "hardlink":
        output.hardlink_to(source)
    with pytest.raises(ValueError, match="must not alias its source"):
        prepare_beam_gateway_template(source, output)
    assert source.read_bytes() == b"original"


def test_existing_output_is_never_overwritten(tmp_path):
    source = tmp_path / "checkpoint" / "chat_template.jinja"
    source.parent.mkdir()
    source.write_bytes(b"original")
    output = tmp_path / "gateway.jinja"
    output.write_bytes(b"existing output")
    with pytest.raises(FileExistsError, match="already exists"):
        prepare_beam_gateway_template(source, output)
    assert output.read_bytes() == b"existing output"


@pytest.mark.parametrize("symlink_parent", [False, True])
def test_output_cannot_be_placed_in_the_checkpoint(tmp_path, symlink_parent):
    source = tmp_path / "checkpoint" / "chat_template.jinja"
    source.parent.mkdir()
    source.write_bytes(b"original")
    parent = source.parent
    if symlink_parent:
        parent = tmp_path / "linked-checkpoint"
        parent.symlink_to(source.parent, target_is_directory=True)
    with pytest.raises(ValueError, match="outside the checkpoint"):
        prepare_beam_gateway_template(source, parent / "gateway.jinja")
    assert source.read_bytes() == b"original"


def test_qualified_asset_produces_the_native_gateway_checked_template(tmp_path):
    source_path = os.environ.get("TOKENSPEED_TEST_BEAM_CHAT_TEMPLATE")
    if source_path is None:
        pytest.skip("requires the pinned Beam tokenizer-only asset; no model weights")
    source = Path(source_path)
    before = source.read_bytes()
    output = tmp_path / "gateway.jinja"
    result = prepare_beam_gateway_template(source, output)
    assert source.read_bytes() == before
    assert result["source_sha256"] == hashlib.sha256(before).hexdigest()
    assert result["output_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert (
        result["output_sha256"]
        == "f33d82f534550ddc13969720850d6809f42c1572813269906056580cb3b510b6"
    )
    assert output.read_bytes() == before.replace(
        b"{%- generation %}", b"{%- if true %}"
    ).replace(b"{%- endgeneration %}", b"{%- endif %}")


def test_symlink_source_keeps_its_logical_checkpoint_directory_protected(tmp_path):
    blob = tmp_path / "blobs" / "template"
    blob.parent.mkdir()
    blob.write_bytes(b"original")
    source = tmp_path / "checkpoint" / "chat_template.jinja"
    source.parent.mkdir()
    source.symlink_to(blob)
    with pytest.raises(ValueError, match="outside the checkpoint"):
        prepare_beam_gateway_template(source, source.parent / "gateway.jinja")
    assert blob.read_bytes() == b"original"
    assert not (source.parent / "gateway.jinja").exists()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
