#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

if ! xcrun --find metal >/dev/null 2>&1; then
  echo "Metal compiler not found. Install Xcode 26.6 and the MetalToolchain component." >&2
  exit 1
fi

cmake -S . -B build-mac-metal -DCMAKE_BUILD_TYPE=Release -DSTATIC_RUNTIME=OFF
cmake --build build-mac-metal --parallel
./build-mac-metal/tron_vanity_generator --selftest
./build-mac-metal/tron_vanity_generator --matchtest
./build-mac-metal/tron_vanity_generator --list
