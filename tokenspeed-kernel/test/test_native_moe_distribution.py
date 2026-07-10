from pathlib import Path

import pytest
from check_native_moe_distribution import (
    _PACKAGE_PREFIX,
    _kernel_meta_symbols,
    _validate_cubin_inventory,
    _validate_source_cubin_inventory,
)

_TEST_ARCHIVE_STEMS = {
    f"Bmm_MxE4m3_MxE2m1MxE4m3_Fp32_test_{index:02d}" for index in range(85)
}


def _kernel_metadata(stems: set[str]) -> str:
    sorted_stems = sorted(stems)
    declarations = [
        f"extern unsigned char const {stem}_cubin[];" for stem in sorted_stems
    ]
    length_declarations = [
        f"extern unsigned int const {stem}_cubin_len;" for stem in sorted_stems
    ]
    config_entries = [f"{{{stem}_cubin, {stem}_cubin_len," for stem in sorted_stems]
    return "\n".join(declarations + length_declarations + config_entries)


def test_native_moe_source_cubin_inventory() -> None:
    source_root = Path(__file__).resolve().parents[1] / "python"
    _validate_source_cubin_inventory(source_root / _PACKAGE_PREFIX)


@pytest.mark.parametrize(
    ("missing_line", "expected_error"),
    (
        (
            "extern unsigned int const {stem}_cubin_len;\n",
            "archive/length symbols mismatch",
        ),
        ("{{{stem}_cubin, {stem}_cubin_len,\n", "archive/config entries mismatch"),
    ),
)
def test_native_moe_inventory_rejects_incomplete_metadata(
    missing_line: str, expected_error: str
) -> None:
    stem = min(_TEST_ARCHIVE_STEMS)
    contents = _kernel_metadata(_TEST_ARCHIVE_STEMS).replace(
        missing_line.format(stem=stem), "", 1
    )

    with pytest.raises(RuntimeError, match=expected_error):
        _validate_cubin_inventory(
            _TEST_ARCHIVE_STEMS,
            *_kernel_meta_symbols(contents),
            origin="test metadata",
        )


def test_native_moe_inventory_rejects_unvendored_family() -> None:
    archive_stems = set(_TEST_ARCHIVE_STEMS)
    archive_stems.remove(min(archive_stems))
    archive_stems.add("Bmm_MxE4m3_MxE2m1MxE4m3_test_unvendored")
    metadata = _kernel_metadata(archive_stems)

    with pytest.raises(RuntimeError, match="unexpected cubin families"):
        _validate_cubin_inventory(
            archive_stems,
            *_kernel_meta_symbols(metadata),
            origin="test metadata",
        )
