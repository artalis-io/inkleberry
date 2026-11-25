#!/bin/sh
# Format only the C port; pinned CUDA/Zig and generated ABI files stay untouched.
set -eu
cd "$(dirname "$0")/.."
fmt=${CLANG_FORMAT:-clang-format-18}
"$fmt" --version | grep -q 'version 18\.' || { echo 'clang-format 18 is required' >&2; exit 1; }
if [ "${1:-}" = --check ]; then
    set -- --dry-run --Werror
else
    set -- -i
fi
"$fmt" "$@" include/sq.h src/c/internal.h src/c/foundation.c src/c/safetensors.c cli/main.c test/unit.c test/concurrency.c test/installed_consumer.c test/cuda_smoke.c fuzz/fuzz_safetensors.c
