#!/usr/bin/env bash
# Build AI-Planning/pddl-generators C binaries and Fast Downward.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GEN_DIR="$REPO_ROOT/third_party/pddl-generators"
FD_DIR="$REPO_ROOT/third_party/downward"

if [ ! -d "$GEN_DIR" ]; then
    echo "ERROR: $GEN_DIR not found. Run: bash scripts/setup_third_party.sh" >&2
    exit 1
fi

echo "==> Installing pddl-generators Python deps..."
if [ -f "$GEN_DIR/requirements.txt" ]; then
    pip install -r "$GEN_DIR/requirements.txt" --quiet
fi

# ---------------------------------------------------------------------------
# Build each domain that has a C binary (has a Makefile or compile script)
# ---------------------------------------------------------------------------
echo "==> Building pddl-generator C binaries..."

DOMAINS_WITH_MAKE=(blocksworld gripper ferry miconic rovers satellite)

for domain in "${DOMAINS_WITH_MAKE[@]}"; do
    d="$GEN_DIR/$domain"
    if [ ! -d "$d" ]; then
        echo "  WARNING: $d not found, skipping" >&2
        continue
    fi
    if [ -f "$d/Makefile" ]; then
        echo "  Building $domain..."
        make -C "$d" -j"$(nproc 2>/dev/null || echo 4)" -k 2>&1 | grep -v "^make\[" | tail -8 || true
    elif ls "$d"/*.c &>/dev/null 2>&1; then
        echo "  Compiling $domain (no Makefile, using gcc)..."
        (cd "$d" && gcc -O2 -o "$domain" *.c -lm 2>&1 | tail -5) || true
    else
        echo "  $domain: no C source / Makefile found, skipping" >&2
    fi
done

# sokoban has a different sub-path
SOK_DIR="$GEN_DIR/sokoban/random"
if [ -d "$SOK_DIR" ]; then
    echo "  Building sokoban..."
    if [ -f "$SOK_DIR/Makefile" ]; then
        make -C "$SOK_DIR" -j"$(nproc 2>/dev/null || echo 4)" -k 2>&1 | grep -v "^make\[" | tail -8 || true
    elif ls "$SOK_DIR"/*.c &>/dev/null 2>&1; then
        (cd "$SOK_DIR" && gcc -O2 -o sokoban-generator-typed *.c -lm 2>&1 | tail -5) || true
    fi
fi

# Verify key binaries exist
echo ""
echo "==> Checking generator binaries..."
MISSING=0
for check in \
    "$GEN_DIR/blocksworld/blocksworld" \
    "$GEN_DIR/gripper/gripper" \
    "$GEN_DIR/ferry/ferry" \
    "$GEN_DIR/miconic/miconic" \
    "$GEN_DIR/rovers/rovgen" \
    "$GEN_DIR/satellite/satgen" \
    "$GEN_DIR/sokoban/random/sokoban-generator-typed"; do
    if [ -f "$check" ]; then
        echo "  [ok] $check"
    else
        echo "  [MISSING] $check" >&2
        MISSING=$(( MISSING + 1 ))
    fi
done

if [ "$MISSING" -gt 0 ]; then
    echo ""
    echo "WARNING: $MISSING generator binaries missing. Check build output above." >&2
fi

# ---------------------------------------------------------------------------
# Build Fast Downward
# ---------------------------------------------------------------------------
if [ ! -d "$FD_DIR" ]; then
    echo "WARNING: $FD_DIR not found. Teacher planner unavailable." >&2
else
    echo ""
    echo "==> Building Fast Downward (this takes ~5 min)..."
    cd "$FD_DIR"
    python3 build.py release 2>&1 | tail -20
    echo "==> Fast Downward built."
fi

echo ""
echo "==> build_generators.sh done."
