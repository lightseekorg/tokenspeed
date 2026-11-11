#!/usr/bin/env bash
set -euo pipefail

# Run from a full checkout inside the matching manylinux_2_28 image.
dnf install -y openssl-devel zstd
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
export PATH="${CARGO_HOME:-$HOME/.cargo}/bin:$PATH"

# The image's own interpreters omit libpython, which ts-host must embed.
export UV_PYTHON_INSTALL_DIR=/opt/build-python
uv python install "$PYTHON_VERSION"
build_python=$(uv python find --managed-python "$PYTHON_VERSION")
uv venv --python "$build_python" /tmp/engine-build
export PATH="/tmp/engine-build/bin:$PATH"
uv pip install --python /tmp/engine-build/bin/python build wheel twine
python -m build --wheel --outdir /tmp/engine-raw tokenspeed-engine
python test/ci_system/repair_engine_wheel.py /tmp/engine-raw/*.whl
auditwheel repair --plat "manylinux_2_28_${ARCH}" --wheel-dir dist /tmp/engine-vendored/*.whl
python -m twine check --strict dist/*.whl
