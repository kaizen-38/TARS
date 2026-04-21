#!/usr/bin/env bash
# Build the VAL (Validate) binary from KCL-Planning/VAL.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VAL_DIR="$REPO_ROOT/third_party/VAL"

if [ ! -d "$VAL_DIR" ]; then
    echo "ERROR: $VAL_DIR not found."
    echo "Run: git submodule update --init third_party/VAL"
    exit 1
fi

cd "$VAL_DIR"

echo "==> Building VAL..."

# VAL uses CMake
if [ -f "CMakeLists.txt" ]; then
    mkdir -p build
    cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release
    make -j2
    echo "==> VAL built at: $VAL_DIR/build/bin/Validate"
elif [ -f "Makefile" ]; then
    make -j2
    echo "==> VAL built."
else
    echo "ERROR: No CMakeLists.txt or Makefile found in $VAL_DIR"
    exit 1
fi
