#!/bin/bash
# Build and install tokenspeed-engine (the ts-host rank binary and the
# Python frontend's connection to it) from this checkout, for ${PYTHON}
# (python3 by default). Every engine rank runs ts-host, so every serving
# install needs it.
set -euo pipefail

WORKSPACE=${WORKSPACE:-$(pwd)}
PYTHON=${PYTHON:-python3}

# The workspace's minimum Rust (`rust-version` in tokenspeed-engine/Cargo.toml).
if ! command -v cargo > /dev/null 2>&1; then
    if [ ! -x "${HOME}/.cargo/bin/cargo" ]; then
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs |
            sh -s -- -y --profile minimal --default-toolchain stable
    fi
    export PATH="${HOME}/.cargo/bin:${PATH}"
fi
cargo --version

# Engine changes accumulate without a version bump, like the scheduler's:
# reinstall the checkout instead of accepting an earlier build.
"${PYTHON}" -m pip install --force-reinstall --no-deps \
    "${WORKSPACE}/tokenspeed-engine"
"${PYTHON}" -c 'import os, tokenspeed_engine; assert os.access(tokenspeed_engine.host_binary(), os.X_OK)'
