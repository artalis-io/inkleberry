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
"$fmt" "$@" test/infer_consumer.c fuzz/fuzz_infer.c include/sq_infer.h include/sq_infer_squid.h src/c/infer_internal.h src/c/infer.c src/c/infer_model.c src/c/infer_session.c src/c/infer_request.c src/c/engine_control.h test/infer.c test/infer_mock.c test/infer_native.c include/sq_chat.h src/c/chat.c src/c/stream.c include/sq.h include/sq_engine.h src/c/checked_prefill.h src/cuda/checked_prefill.cu src/c/engine_prefill.c src/c/lookup.h src/c/lookup.c src/c/engine_adaptive.c src/c/engine_prefix.c src/c/prefix.h src/c/sampling.c src/c/engine_spec.c src/c/engine_internal.h src/c/engine_bind.c src/c/engine_layout.c src/c/engine_forward.c src/c/engine_runtime.c include/sq_gpu.h include/sq_weights.h src/c/internal.h src/c/weights_internal.h src/c/weights_recipe.c src/c/weights_runtime.c src/c/gpu_cuda.c src/c/checked_cuda.h src/cuda/checked_weights.cu src/c/json.h src/c/json.c src/c/tokenizer.h src/c/tokenizer_profile.h src/c/tokenizer_load.c src/c/tokenizer.c src/c/inkling.h src/c/inkling.c src/c/foundation.c src/c/safetensors.c cli/generate.c cli/main.c test/sampling_kernels.cu test/session_context.cu test/lookup.c test/sampling.c test/mock_gpu.h test/engine.c test/adaptive_graph.cu test/adaptive_compare.c test/prefix_compare.c test/prefix_recovery.c test/decode_recovery.c test/decode_compare.c test/ar_compare.c test/chat_native.c fuzz/fuzz_stream.c test/chat.c test/stream_driver.c test/unit.c test/packing.cu test/gpu_context.cu test/gpu_weights_compare.c test/weights_runtime.c test/tokenizer.c test/inkling.c test/concurrency.c test/installed_gpu_consumer.c test/installed_consumer.c test/cuda_smoke.c fuzz/fuzz_safetensors.c fuzz/fuzz_tokenizer.c
