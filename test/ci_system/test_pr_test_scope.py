from pr_test_scope import extra_changes


def test_description_lines_name_extra_changed_paths():
    description = (
        "## Summary\n"
        "Touches the MoE dispatch.\n"
        "CI-Extra-Changes: tokenspeed-kernel/python/tokenspeed_kernel/ops/moe\n"
        "  CI-Extra-Changes:   tokenspeed-kernel/test/ops/test_gemm.py  \n"
        "CI-Extra-Changes:\n"
        "Not CI-Extra-Changes: python/tokenspeed/x.py\n"
    )

    assert extra_changes(description) == [
        "tokenspeed-kernel/python/tokenspeed_kernel/ops/moe",
        "tokenspeed-kernel/test/ops/test_gemm.py",
    ]
