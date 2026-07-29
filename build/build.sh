#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$root_dir/dist"
mojo build --emit shared-lib -I "$root_dir/src" \
    "$root_dir/src/simplify.mojo" \
    -o "$root_dir/dist/libmojo-fast-simplification.so"
