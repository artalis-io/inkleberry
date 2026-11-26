# Optional pinned implementation comparisons; run from the repository root.
ZIG ?= zig
.PHONY: gpu-weight-oracle
gpu-weight-oracle: reference-check .build/gpu-context.o .build/gpu-weight-compare.o libsq-gpu.a libsq.a
	$(ZIG) build-exe -O ReleaseFast --dep weights -Mroot=reference/oracle/weights.zig -Mweights=reference/upstream/src/weights.zig .build/gpu-context.o .build/gpu-weight-compare.o libsq-gpu.a libsq.a -L$(CUDA_ROOT)/lib64 -L$(CUDA_ROOT)/lib64/stubs -lcudart -lcublas -lcuda -lstdc++ -lm -lpthread -lc -femit-bin=.build/gpu-weight-oracle

.PHONY: ar-oracle
ar-oracle: reference-check .build/session-context.o .build/ar-compare.o libsq-gpu.a libsq.a
	$(ZIG) build-exe -O ReleaseFast --dep engine -Mroot=reference/oracle/engine.zig -Mengine=reference/upstream/src/engine.zig .build/session-context.o .build/ar-compare.o libsq-gpu.a libsq.a -L$(CUDA_ROOT)/lib64 -L$(CUDA_ROOT)/lib64/stubs -lcudart -lcublas -lcuda -lstdc++ -lm -lpthread -lc -femit-bin=.build/ar-oracle

.PHONY: decode-oracle
decode-oracle: reference-check .build/session-context.o .build/decode-compare.o libsq-gpu.a libsq.a
	$(ZIG) build-exe -O ReleaseFast --dep engine -Mroot=reference/oracle/decode.zig -Mengine=reference/upstream/src/engine.zig .build/session-context.o .build/decode-compare.o libsq-gpu.a libsq.a -L$(CUDA_ROOT)/lib64 -L$(CUDA_ROOT)/lib64/stubs -lcudart -lcublas -lcuda -lstdc++ -lm -lpthread -lc -femit-bin=.build/decode-oracle

.PHONY: adaptive-oracle
adaptive-oracle: reference-check .build/session-context.o .build/adaptive-compare.o libsq-gpu.a libsq.a
	$(ZIG) build-exe -O ReleaseFast --dep engine -Mroot=reference/oracle/adaptive.zig -Mengine=reference/upstream/src/engine.zig .build/session-context.o .build/adaptive-compare.o libsq-gpu.a libsq.a -L$(CUDA_ROOT)/lib64 -L$(CUDA_ROOT)/lib64/stubs -lcudart -lcublas -lcuda -lstdc++ -lm -lpthread -lc -femit-bin=.build/adaptive-oracle

.PHONY: prefix-oracle
prefix-oracle: reference-check .build/session-context.o .build/prefix-compare.o libsq-gpu.a libsq.a
	$(ZIG) build-exe -O ReleaseFast --dep engine -Mroot=reference/oracle/prefix.zig -Mengine=reference/upstream/src/engine.zig .build/session-context.o .build/prefix-compare.o libsq-gpu.a libsq.a -L$(CUDA_ROOT)/lib64 -L$(CUDA_ROOT)/lib64/stubs -lcudart -lcublas -lcuda -lstdc++ -lm -lpthread -lc -femit-bin=.build/prefix-oracle

.PHONY: chat-oracle
chat-oracle: reference-check
	python3 reference/validation/prepare_chat_oracle.py
	$(ZIG) build-exe -O ReleaseFast --dep server -Mroot=reference/oracle/chat.zig -Mserver=.build/chat-oracle-src/server.zig -lc -femit-bin=.build/chat-oracle
