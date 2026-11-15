#!/usr/bin/env bash
set -euo pipefail

# Run from a full checkout inside the matching manylinux_2_28 image.
dnf install -y openssl-devel zstd
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
export PATH="${CARGO_HOME:-$HOME/.cargo}/bin:$PATH"

python_tag="${PYTHON_VERSION//./}"
build_python="/opt/python/cp${python_tag}-cp${python_tag}/bin/python"
uv venv --python "$build_python" /tmp/engine-build
export PATH="/tmp/engine-build/bin:$PATH"
uv pip install --python /tmp/engine-build/bin/python build wheel twine
python -m build --wheel --outdir /tmp/engine-raw tokenspeed-engine
auditwheel repair --plat "manylinux_2_28_${ARCH}" --wheel-dir dist /tmp/engine-raw/*.whl
python -m twine check --strict dist/*.whl
