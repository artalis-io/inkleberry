# Canonical source groups shared by modular and generated builds.
HOST = src/foundation.c src/json.c src/safetensors.c src/tokenizer_load.c src/tokenizer.c src/inkling.c src/weights_runtime.c src/weights_recipe.c src/sampling.c src/lookup.c src/chat.c src/stream.c
GPU_HOST = src/engine_bind.c src/engine_layout.c src/engine_forward.c src/engine_spec.c src/engine_adaptive.c src/engine_prefix.c src/engine_prefill.c src/engine_runtime.c src/gpu_cuda.c
INFER = src/infer.c src/infer_model.c src/infer_session.c src/infer_request.c
PUBLIC_HEADERS = include/ifn.h include/ifn_chat.h include/ifn_weights.h include/ifn_gpu.h include/ifn_engine.h include/ifn_infer.h include/ifn_infer_inkling.h
