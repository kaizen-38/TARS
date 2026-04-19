#!/usr/bin/env bash
# Initialize all third-party dependencies under third_party/.
# Works whether or not git submodule gitlinks are committed.
# Run once on the Sol login node before any build or Slurm jobs.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Helper: clone or pull a repo into a given path
clone_or_update() {
    local url="$1"
    local dest="$2"
    local depth="${3:-1}"
    if [ -d "$dest/.git" ]; then
        echo "  [ok] $dest already exists, skipping clone"
    elif [ -d "$dest" ] && [ "$(ls -A "$dest" 2>/dev/null)" ]; then
        echo "  [ok] $dest non-empty, skipping clone"
    else
        echo "  --> cloning $url -> $dest (depth=$depth)"
        mkdir -p "$(dirname "$dest")"
        git clone --depth "$depth" "$url" "$dest"
    fi
}

echo "==> Setting up third_party dependencies..."

# Try the git submodule path first (works if gitlinks are committed)
if git submodule status 2>/dev/null | grep -q '^'; then
    echo "  Found registered submodules — running git submodule update..."
    git submodule update --init --recursive
else
    echo "  No registered gitlinks — using direct git clone fallback"

    clone_or_update \
        "https://github.com/AI-Planning/pddl-generators.git" \
        "third_party/pddl-generators"

    clone_or_update \
        "https://github.com/aibasel/downward.git" \
        "third_party/downward" 1

    clone_or_update \
        "https://github.com/KCL-Planning/VAL.git" \
        "third_party/VAL"

    clone_or_update \
        "https://github.com/hiyouga/LLaMAFactory.git" \
        "third_party/LLaMAFactory"

    clone_or_update \
        "https://github.com/harshakokel/PlanBench.git" \
        "third_party/PlanBench"

    clone_or_update \
        "https://github.com/ValerioBelcamino/PDDL_encoder.git" \
        "third_party/PDDL_encoder"
fi

echo ""
echo "==> Checking third_party contents..."
for d in third_party/pddl-generators third_party/downward third_party/VAL third_party/LLaMAFactory; do
    if [ -d "$d" ]; then
        echo "  [ok] $d"
    else
        echo "  [MISSING] $d — clone may have failed"
    fi
done

# Install LLaMA-Factory as editable package
if [ -d "third_party/LLaMAFactory" ]; then
    echo ""
    echo "==> Installing LLaMA-Factory..."
    pip install -e third_party/LLaMAFactory --quiet
else
    echo ""
    echo "WARNING: third_party/LLaMAFactory not found — training steps will fail."
    echo "         Re-run: bash scripts/setup_third_party.sh"
fi

echo ""
echo "==> Done. Next: bash scripts/build_generators.sh && bash scripts/build_val.sh"
echo "     or just:  make build-tools"
