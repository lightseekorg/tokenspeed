import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_kernel_sources_use_tokenspeed_triton_adapter():
    source_root = ROOT / "tokenspeed-kernel/python/tokenspeed_kernel"
    adapter = source_root / "_triton.py"
    # These helpers intentionally interact with framework-owned stock Triton;
    # neither defines TokenSpeed Triton kernels.
    stock_triton_interop = {
        source_root / "thirdparty/fla.py",
        source_root / "thirdparty/msa/cute/src/common/cute_dsl_utils.py",
    }
    direct_imports = {}

    for path in source_root.rglob("*.py"):
        if path == adapter or path in stock_triton_interop:
            continue
        tree = ast.parse(path.read_text(), filename=str(path))
        lines = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = (alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                names = (node.module,)
            else:
                continue
            if any(
                name == "triton"
                or name.startswith("triton.")
                or name == "tokenspeed_triton"
                or name.startswith("tokenspeed_triton.")
                for name in names
            ):
                lines.append(node.lineno)
        if lines:
            direct_imports[str(path.relative_to(ROOT))] = lines

    assert direct_imports == {}, (
        "TokenSpeed kernel sources must import Triton through "
        f"tokenspeed_kernel._triton: {direct_imports}"
    )
