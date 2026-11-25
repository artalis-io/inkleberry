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
"$fmt" "$@" include/sq.h include/sq_engine.h src/c/checked_prefill.h src/cuda/checked_prefill.cu src/c/engine_prefill.c src/c/engine_internal.h src/c/engine_bind.c src/c/engine_layout.c src/c/engine_forward.c src/c/engine_runtime.c include/sq_gpu.h include/sq_weights.h src/c/internal.h src/c/weights_internal.h src/c/weights_recipe.c src/c/weights_runtime.c src/c/gpu_cuda.c src/c/checked_cuda.h src/cuda/checked_weights.cu src/c/json.h src/c/json.c src/c/tokenizer.h src/c/tokenizer_profile.h src/c/tokenizer_load.c src/c/tokenizer.c src/c/inkling.h src/c/inkling.c src/c/foundation.c src/c/safetensors.c cli/generate.c cli/main.c test/session_context.cu test/mock_gpu.h test/engine.c test/ar_compare.c test/unit.c test/packing.cu test/gpu_context.cu test/gpu_weights_compare.c test/weights_runtime.c test/tokenizer.c test/inkling.c test/concurrency.c test/installed_gpu_consumer.c test/installed_consumer.c test/cuda_smoke.c fuzz/fuzz_safetensors.c fuzz/fuzz_tokenizer.c
