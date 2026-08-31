#!/usr/bin/env bash
set -euo pipefail

KIMI_CODE_VERSION=0.29.0

install_root=${KIMI_CODE_ROOT:-/raid/cache/kimi-code/${KIMI_CODE_VERSION}}
binary=${install_root}/bin/kimi

if [[ -x "$binary" ]] && [[ $("$binary" --version) == "$KIMI_CODE_VERSION" ]]; then
  printf 'Using cached Kimi Code %s at %s\n' "$KIMI_CODE_VERSION" "$binary"
  exit 0
fi

mkdir -p "$install_root"
curl -fsSL https://code.kimi.com/kimi-code/install.sh | \
  KIMI_VERSION="$KIMI_CODE_VERSION" \
  KIMI_INSTALL_DIR="$install_root" \
  KIMI_NO_MODIFY_PATH=1 \
  bash

actual_version=$("$binary" --version)
[[ "$actual_version" == "$KIMI_CODE_VERSION" ]] || {
  echo "Expected Kimi Code $KIMI_CODE_VERSION, got $actual_version" >&2
  exit 1
}

printf 'Installed Kimi Code %s at %s\n' "$actual_version" "$binary"
