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
"$fmt" "$@" include/sq.h include/sq_gpu.h include/sq_weights.h src/c/internal.h src/c/weights_internal.h src/c/weights_recipe.c src/c/weights_runtime.c src/c/gpu_cuda.c src/c/checked_cuda.h src/cuda/checked_weights.cu src/c/json.h src/c/json.c src/c/tokenizer.h src/c/tokenizer_profile.h src/c/tokenizer_load.c src/c/tokenizer.c src/c/inkling.h src/c/inkling.c src/c/foundation.c src/c/safetensors.c cli/main.c test/unit.c test/packing.cu test/gpu_context.cu test/gpu_weights_compare.c test/weights_runtime.c test/tokenizer.c test/inkling.c test/concurrency.c test/installed_gpu_consumer.c test/installed_consumer.c test/cuda_smoke.c fuzz/fuzz_safetensors.c fuzz/fuzz_tokenizer.c
