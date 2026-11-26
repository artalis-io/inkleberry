# Canonical source groups shared by modular and generated builds.
HOST = src/foundation.c src/json.c src/safetensors.c src/tokenizer_load.c src/tokenizer.c src/inkling.c src/weights_runtime.c src/weights_recipe.c src/sampling.c src/lookup.c src/chat.c src/stream.c
GPU_HOST = src/engine_bind.c src/engine_layout.c src/engine_forward.c src/engine_spec.c src/engine_adaptive.c src/engine_prefix.c src/engine_prefill.c src/engine_runtime.c src/gpu_cuda.c
INFER = src/infer.c src/infer_model.c src/infer_session.c src/infer_request.c
PUBLIC_HEADERS = include/sq.h include/sq_chat.h include/sq_weights.h include/sq_gpu.h include/sq_engine.h include/sq_infer.h include/sq_infer_squid.h
