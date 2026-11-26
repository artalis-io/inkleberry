# Optional pinned implementation comparisons; run from the repository root.
ZIG ?= zig
.PHONY: native-oracle-sources
native-oracle-sources: reference-check
	python3 reference/validation/prepare_native_oracle.py
.PHONY: gpu-weight-oracle
gpu-weight-oracle: native-oracle-sources .build/gpu-context.o .build/gpu-weight-compare.o libinkleberry-gpu.a libinkleberry.a
	$(ZIG) build-exe -O ReleaseFast --dep weights -Mroot=.build/native-oracle-src/oracle/weights.zig -Mweights=.build/native-oracle-src/upstream/weights.zig .build/gpu-context.o .build/gpu-weight-compare.o libinkleberry-gpu.a libinkleberry.a -L$(CUDA_ROOT)/lib64 -L$(CUDA_ROOT)/lib64/stubs -lcudart -lcublas -lcuda -lstdc++ -lm -lpthread -lc -femit-bin=.build/gpu-weight-oracle

.PHONY: ar-oracle
ar-oracle: native-oracle-sources .build/session-context.o .build/ar-compare.o libinkleberry-gpu.a libinkleberry.a
	$(ZIG) build-exe -O ReleaseFast --dep engine -Mroot=.build/native-oracle-src/oracle/engine.zig -Mengine=.build/native-oracle-src/upstream/engine.zig .build/session-context.o .build/ar-compare.o libinkleberry-gpu.a libinkleberry.a -L$(CUDA_ROOT)/lib64 -L$(CUDA_ROOT)/lib64/stubs -lcudart -lcublas -lcuda -lstdc++ -lm -lpthread -lc -femit-bin=.build/ar-oracle

.PHONY: decode-oracle
decode-oracle: native-oracle-sources .build/session-context.o .build/decode-compare.o libinkleberry-gpu.a libinkleberry.a
	$(ZIG) build-exe -O ReleaseFast --dep engine -Mroot=.build/native-oracle-src/oracle/decode.zig -Mengine=.build/native-oracle-src/upstream/engine.zig .build/session-context.o .build/decode-compare.o libinkleberry-gpu.a libinkleberry.a -L$(CUDA_ROOT)/lib64 -L$(CUDA_ROOT)/lib64/stubs -lcudart -lcublas -lcuda -lstdc++ -lm -lpthread -lc -femit-bin=.build/decode-oracle

.PHONY: adaptive-oracle
adaptive-oracle: native-oracle-sources .build/session-context.o .build/adaptive-compare.o libinkleberry-gpu.a libinkleberry.a
	$(ZIG) build-exe -O ReleaseFast --dep engine -Mroot=.build/native-oracle-src/oracle/adaptive.zig -Mengine=.build/native-oracle-src/upstream/engine.zig .build/session-context.o .build/adaptive-compare.o libinkleberry-gpu.a libinkleberry.a -L$(CUDA_ROOT)/lib64 -L$(CUDA_ROOT)/lib64/stubs -lcudart -lcublas -lcuda -lstdc++ -lm -lpthread -lc -femit-bin=.build/adaptive-oracle

.PHONY: prefix-oracle
prefix-oracle: native-oracle-sources .build/session-context.o .build/prefix-compare.o libinkleberry-gpu.a libinkleberry.a
	$(ZIG) build-exe -O ReleaseFast --dep engine -Mroot=.build/native-oracle-src/oracle/prefix.zig -Mengine=.build/native-oracle-src/upstream/engine.zig .build/session-context.o .build/prefix-compare.o libinkleberry-gpu.a libinkleberry.a -L$(CUDA_ROOT)/lib64 -L$(CUDA_ROOT)/lib64/stubs -lcudart -lcublas -lcuda -lstdc++ -lm -lpthread -lc -femit-bin=.build/prefix-oracle

.PHONY: chat-oracle
chat-oracle: reference-check
	python3 reference/validation/prepare_chat_oracle.py
	$(ZIG) build-exe -O ReleaseFast --dep server -Mroot=.build/native-oracle-src/oracle/chat.zig -Mserver=.build/chat-oracle-src/server.zig -lc -femit-bin=.build/chat-oracle
