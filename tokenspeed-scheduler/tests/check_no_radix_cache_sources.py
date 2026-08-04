from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_DIRECTORIES = (
    ROOT / "csrc/resource/radix_tree",
    ROOT / "csrc/resource/kv_prefix_cache",
    ROOT / "csrc/resource/hybrid_prefix_cache",
)
FORBIDDEN_SYMBOLS = ("RadixTree", "KVPrefixCache", "HybridPrefixCache")


def main() -> None:
    failures = [
        str(path.relative_to(ROOT)) for path in FORBIDDEN_DIRECTORIES if path.exists()
    ]
    for root in (ROOT / "csrc", ROOT / "bindings"):
        for path in root.rglob("*"):
            if path.suffix not in {".h", ".cpp"}:
                continue
            text = path.read_text()
            if any(symbol in text for symbol in FORBIDDEN_SYMBOLS):
                failures.append(str(path.relative_to(ROOT)))

    if failures:
        raise SystemExit(
            "Radix prefix-cache sources remain:\n"
            + "\n".join(f"  {path}" for path in sorted(set(failures)))
        )


if __name__ == "__main__":
    main()
