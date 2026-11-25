# C library/installed-consumer conventions adapted from artalis-io/jev-bush (0BSD).
.DEFAULT_GOAL := all
CC ?= cc
CXX ?= c++
AR ?= ar
SAN_CC ?= clang
SAN_CXX ?= clang++
CFLAGS ?= -O2
LDLIBS += -lm
WARNFLAGS = -std=c11 -Wall -Wextra -Wpedantic -Wshadow -Wformat=2 -Werror
CPPFLAGS += -Iinclude -Isrc/c -D_FILE_OFFSET_BITS=64
PREFIX ?= /usr/local
CUDA_ROOT ?= /usr/local/cuda
NVCC = $(CUDA_ROOT)/bin/nvcc
CUDA_ARCH ?= 120a
CUDAFLAGS = -O3 --use_fast_math -std=c++20 -gencode arch=compute_$(CUDA_ARCH),code=sm_$(CUDA_ARCH) --expt-relaxed-constexpr -Icuda
CUDA_LIBS = -L$(CUDA_ROOT)/lib64 -lcudart -lcublas -L$(CUDA_ROOT)/lib64/stubs -lcuda -lstdc++ -lm -lpthread
HOST = src/c/foundation.c src/c/json.c src/c/safetensors.c src/c/tokenizer_load.c src/c/tokenizer.c src/c/inkling.c src/c/weights_runtime.c src/c/weights_recipe.c src/c/sampling.c src/c/lookup.c src/c/chat.c src/c/stream.c
OBJ = $(HOST:src/c/%.c=.build/%.o)
PIC = $(HOST:src/c/%.c=.build/%.pic.o)
CUDA_HEADERS = $(wildcard cuda/*.hpp cuda/*.cuh)
CUDA_MATH = kernels_gemv kernels_inkling kernels_dspark kernels_prefill
CUDA_OBJECTS = $(addprefix .build/,$(addsuffix .cuda.o,$(CUDA_MATH)))
ifeq ($(shell uname -s),Darwin)
SHARED = libsq.dylib
SHARED_FLAGS = -dynamiclib -Wl,-install_name,@rpath/libsq.dylib
else
SHARED = libsq.so
SHARED_FLAGS = -shared
endif
.PHONY: all check clean sanitize thread-sanitize fuzz-smoke generated-check format lint install install-check cuda-abi-check cuda-check FORCE
all: squid-c libsq.a $(SHARED)
.build:
	mkdir -p $@
FORCE:
.build/config: FORCE | .build
	@printf '%s\n' '$(CC) $(CXX) $(AR) $(CPPFLAGS) $(CFLAGS) $(WARNFLAGS) $(LDFLAGS)' '$(CUDA_ROOT) $(CUDAFLAGS)' > $@.tmp
	@cmp -s $@.tmp $@ || mv $@.tmp $@
	@rm -f $@.tmp
.build/%.o: src/c/%.c include/sq.h src/c/internal.h .build/config
	$(CC) $(CPPFLAGS) $(CFLAGS) $(WARNFLAGS) -MMD -MP -c $< -o $@
.build/%.pic.o: src/c/%.c include/sq.h src/c/internal.h .build/config
	$(CC) $(CPPFLAGS) $(CFLAGS) $(WARNFLAGS) -fPIC -fvisibility=hidden -DSQ_SHARED -DSQ_BUILD_SHARED -MMD -MP -c $< -o $@
libsq.a: $(OBJ)
	$(AR) rcs $@ $^
$(SHARED): $(PIC)
	$(CC) $(SHARED_FLAGS) $(LDFLAGS) $^ $(LDLIBS) -o $@
squid-c: cli/main.c libsq.a .build/config
	$(CC) $(CPPFLAGS) $(CFLAGS) $(WARNFLAGS) $< libsq.a $(LDFLAGS) $(LDLIBS) -o $@
.build/unit: test/unit.c libsq.a .build/config
	$(CC) $(CPPFLAGS) $(CFLAGS) $(WARNFLAGS) $< libsq.a $(LDFLAGS) $(LDLIBS) -o $@
.build/concurrency: test/concurrency.c libsq.a .build/config
	$(CC) $(CPPFLAGS) $(CFLAGS) $(WARNFLAGS) $< libsq.a -pthread $(LDFLAGS) $(LDLIBS) -o $@
generated-check:
	python3 tools/verify_upstream.py
	python3 tools/generate_cuda_abi.py --check
.build/tokenizer-fixture.json: test/tokenizer_fixtures.py | .build
	python3 $< $@
.build/tokenizer-test: test/tokenizer.c libsq.a .build/config
	$(CC) $(CPPFLAGS) $(CFLAGS) $(WARNFLAGS) $< libsq.a -pthread $(LDFLAGS) $(LDLIBS) -o $@
.build/inkling-test: test/inkling.c libsq.a .build/config
	$(CC) $(CPPFLAGS) $(CFLAGS) $(WARNFLAGS) $< libsq.a $(LDFLAGS) $(LDLIBS) -o $@
.build/weights-test: test/weights_runtime.c test/mock_gpu.h libsq.a .build/config
	$(CC) $(CPPFLAGS) $(CFLAGS) $(WARNFLAGS) $< libsq.a -pthread $(LDFLAGS) $(LDLIBS) -o $@
check: .build/chat-native.o .build/chat-test .build/stream-driver .build/chat-fixture.json .build/prefix-recovery.o .build/prefix-compare.o .build/adaptive-compare.o .build/adaptive-recovery.o .build/lookup-test .build/sampling-test .build/decode-compare.o .build/engine_forward.o .build/engine_spec.o .build/engine_adaptive.o .build/engine_prefix.o .build/engine_prefill.o .build/generate.o .build/check-ar-cpp.o .build/engine-test .build/weights-test all generated-check .build/unit .build/concurrency .build/tokenizer-test .build/tokenizer-fixture.json .build/inkling-test
	.build/chat-test .build/chat-fixture.json
	python3 test/stream_cases.py .build/stream-driver
	.build/unit
	.build/engine-test
	.build/sampling-test
	.build/lookup-test
	python3 test/weights_plan.py ./squid-c
	.build/inkling-test
	python3 test/inkling_cases.py .build/inkling-test ./squid-c .build/weights-test
	.build/tokenizer-test .build/tokenizer-fixture.json
	python3 test/tokenizer_cases.py .build/tokenizer-test ./squid-c
	.build/concurrency
	python3 test/safetensors_cases.py ./squid-c
	$(CC) $(CPPFLAGS) $(CFLAGS) $(WARNFLAGS) -c test/cuda_abi_c.c -o .build/cuda_abi_c.o
	$(CXX) -std=c++11 -Wall -Wextra -Werror -Iinclude -x c++ -c test/installed_consumer.c -o .build/cpp-consumer.o
	$(CC) .build/cpp-consumer.o libsq.a $(LDFLAGS) $(LDLIBS) -o .build/cpp-consumer
	.build/cpp-consumer
sanitize:
	$(MAKE) clean
	$(MAKE) CC=$(SAN_CC) CXX=$(SAN_CXX) CFLAGS='-O1 -g -fno-omit-frame-pointer -fsanitize=address,undefined -fno-sanitize-recover=all' LDFLAGS='-fsanitize=address,undefined' check
thread-sanitize:
	$(MAKE) clean
	$(MAKE) CC=$(SAN_CC) CXX=$(SAN_CXX) CFLAGS='-O1 -g -fsanitize=thread' LDFLAGS='-fsanitize=thread' .build/concurrency .build/tokenizer-test .build/tokenizer-fixture.json .build/weights-test .build/inkling-test .build/engine-test .build/sampling-test .build/lookup-test .build/chat-test .build/chat-fixture.json squid-c
	.build/concurrency
	.build/tokenizer-test .build/tokenizer-fixture.json
	python3 test/inkling_cases.py .build/inkling-test ./squid-c .build/weights-test
	.build/engine-test
	.build/sampling-test
	.build/lookup-test
	.build/chat-test .build/chat-fixture.json
fuzz-smoke: | .build
	$(SAN_CC) -O1 -g $(CPPFLAGS) -std=c11 -fsanitize=fuzzer,address,undefined -fno-sanitize-recover=all $(HOST) fuzz/fuzz_stream.c -o .build/fuzz-stream
	.build/fuzz-stream -runs=3000 -max_len=32768
	python3 test/safetensors_cases.py --corpus .build/fuzz-corpus
	$(SAN_CC) -O1 -g $(CPPFLAGS) -std=c11 -fsanitize=fuzzer,address,undefined -fno-sanitize-recover=all $(HOST) fuzz/fuzz_safetensors.c -o .build/fuzz-safetensors
	.build/fuzz-safetensors -runs=3000 -max_len=262144 .build/fuzz-corpus
	python3 test/tokenizer_cases.py --corpus .build/tokenizer-corpus
	$(SAN_CC) -O1 -g $(CPPFLAGS) -std=c11 -fsanitize=fuzzer,address,undefined -fno-sanitize-recover=all $(HOST) fuzz/fuzz_tokenizer.c -o .build/fuzz-tokenizer
	.build/fuzz-tokenizer -runs=3000 -max_len=32768 .build/tokenizer-corpus
format:
	tools/format.sh
lint: generated-check
	tools/format.sh --check
	python3 tools/lint.py
install: all
	mkdir -p $(DESTDIR)$(PREFIX)/include $(DESTDIR)$(PREFIX)/lib/pkgconfig $(DESTDIR)$(PREFIX)/bin
	cp include/sq.h include/sq_chat.h include/sq_weights.h include/sq_gpu.h include/sq_engine.h $(DESTDIR)$(PREFIX)/include/
	cp libsq.a $(SHARED) $(DESTDIR)$(PREFIX)/lib/
	cp squid-c $(DESTDIR)$(PREFIX)/bin/
	sed 's|@PREFIX@|$(PREFIX)|g' sq.pc.in > $(DESTDIR)$(PREFIX)/lib/pkgconfig/squid-c.pc
install-check: DESTDIR = $(CURDIR)/.build/stage
install-check: PREFIX = /usr/local
install-check: install
	$(CC) $(CFLAGS) $(WARNFLAGS) -I.build/stage/usr/local/include test/installed_consumer.c .build/stage/usr/local/lib/libsq.a $(LDFLAGS) $(LDLIBS) -o .build/installed-static
	.build/installed-static
	$(CC) $(CFLAGS) $(WARNFLAGS) -I.build/stage/usr/local/include test/installed_consumer.c -L.build/stage/usr/local/lib -lsq -Wl,-rpath,'$(CURDIR)/.build/stage/usr/local/lib' $(LDFLAGS) $(LDLIBS) -o .build/installed-shared
	.build/installed-shared
.build/%.cuda.o: cuda/%.cu $(CUDA_HEADERS) .build/config
	$(NVCC) $(CUDAFLAGS) -c $< -o $@
.build/abi-probe.o: test/cuda_abi_probe.cu src/c/cuda_abi.h cuda/cuda_abi.cu $(CUDA_HEADERS) .build/config
	$(NVCC) $(CUDAFLAGS) -Isrc/c -c $< -o $@
.build/abi-c.o: test/cuda_abi_c.c src/c/cuda_abi.h .build/config
	$(CC) $(CPPFLAGS) $(CFLAGS) $(WARNFLAGS) -c $< -o $@
.build/abi-probe: .build/abi-probe.o .build/abi-c.o $(CUDA_OBJECTS)
	$(CXX) $^ $(CUDA_LIBS) -o $@
cuda-abi-check: generated-check .build/abi-probe
	LD_LIBRARY_PATH='$(CUDA_ROOT)/lib64' .build/abi-probe
.build/cuda-smoke.o: test/cuda_smoke.c src/c/cuda_abi.h .build/config
	$(CC) $(CPPFLAGS) $(CFLAGS) $(WARNFLAGS) -c $< -o $@
.build/cuda-smoke: .build/cuda-smoke.o .build/cuda_abi.cuda.o $(CUDA_OBJECTS)
	$(CXX) $^ $(CUDA_LIBS) -o $@
cuda-check: cuda-abi-check .build/cuda-smoke
	python3 tools/run_cuda_smoke.py '$(CUDA_ROOT)/lib64' .build/cuda-smoke
clean:
	rm -rf .build squid-c libsq.a libsq.so libsq.dylib libsq-gpu.a squid-c-weights squid-c-generate
-include $(wildcard .build/*.d)

# GPU backend is opt-in; the host library still builds without CUDA.
.PHONY: gpu
gpu: generated-check libsq-gpu.a
.build/checked-weights.cuda.o: src/cuda/checked_weights.cu src/c/checked_cuda.h cuda/cuda_abi.cu $(CUDA_HEADERS) .build/config
	$(NVCC) $(CUDAFLAGS) -c $< -o $@
libsq-gpu.a: .build/checked-prefill.cuda.o .build/engine_bind.o .build/engine_layout.o .build/engine_forward.o .build/engine_spec.o .build/engine_adaptive.o .build/engine_prefix.o .build/engine_prefill.o .build/engine_runtime.o .build/gpu_cuda.o .build/checked-weights.cuda.o $(filter-out .build/kernels_prefill.cuda.o,$(CUDA_OBJECTS))
	rm -f $@
	$(AR) rcs $@ $^
.build/gpu-weight-compare.o: test/gpu_weights_compare.c include/sq_gpu.h .build/config
	$(CC) $(CPPFLAGS) $(CFLAGS) $(WARNFLAGS) -c $< -o $@
.build/gpu-weight-main.o: test/gpu_weights_compare.c include/sq_gpu.h .build/config
	$(CC) $(CPPFLAGS) $(CFLAGS) $(WARNFLAGS) -DSQ_STANDALONE_WEIGHTS -c $< -o $@
squid-c-weights: .build/gpu-context.o .build/gpu-weight-main.o libsq-gpu.a libsq.a
	$(CXX) $^ $(CUDA_LIBS) $(LDFLAGS) -o $@
ZIG ?= zig
.PHONY: gpu-weight-oracle
gpu-weight-oracle: generated-check .build/gpu-context.o .build/gpu-weight-compare.o libsq-gpu.a libsq.a
	$(ZIG) build-exe -O ReleaseFast --dep weights -Mroot=test/oracle/weights.zig -Mweights=src/weights.zig .build/gpu-context.o .build/gpu-weight-compare.o libsq-gpu.a libsq.a -L$(CUDA_ROOT)/lib64 -L$(CUDA_ROOT)/lib64/stubs -lcudart -lcublas -lcuda -lstdc++ -lm -lpthread -lc -femit-bin=.build/gpu-weight-oracle

.build/gpu-context.o: test/gpu_context.cu include/sq_gpu.h .build/config
	$(NVCC) $(CUDAFLAGS) -Iinclude -c $< -o $@
.build/packing.o: test/packing.cu src/c/checked_cuda.h .build/config
	$(NVCC) $(CUDAFLAGS) -Isrc/c -c $< -o $@
.build/packing: .build/packing.o libsq-gpu.a
	$(CXX) $^ $(CUDA_LIBS) -o $@
.PHONY: gpu-packing-check install-gpu
gpu-packing-check: generated-check .build/packing
	python3 tools/run_cuda_smoke.py '$(CUDA_ROOT)/lib64' .build/packing
install-gpu: install gpu
	cp libsq-gpu.a $(DESTDIR)$(PREFIX)/lib/

.PHONY: gpu-install-check
gpu-install-check: DESTDIR = $(CURDIR)/.build/gpu-stage
gpu-install-check: PREFIX = /usr/local
gpu-install-check: install-gpu
	$(CC) $(CFLAGS) $(WARNFLAGS) -I.build/gpu-stage/usr/local/include -c test/installed_gpu_consumer.c -o .build/installed-gpu.o
	$(CXX) .build/installed-gpu.o .build/gpu-stage/usr/local/lib/libsq-gpu.a .build/gpu-stage/usr/local/lib/libsq.a $(CUDA_LIBS) -o .build/installed-gpu
	LD_LIBRARY_PATH='$(CUDA_ROOT)/lib64' .build/installed-gpu
	$(CXX) -std=c++11 -Wall -Wextra -Werror -I.build/gpu-stage/usr/local/include -x c++ -c test/installed_gpu_consumer.c -o .build/installed-gpu-cpp.o
	$(CXX) .build/installed-gpu-cpp.o .build/gpu-stage/usr/local/lib/libsq-gpu.a .build/gpu-stage/usr/local/lib/libsq.a $(CUDA_LIBS) -o .build/installed-gpu-cpp
	LD_LIBRARY_PATH='$(CUDA_ROOT)/lib64' .build/installed-gpu-cpp
.build/generate.o: cli/generate.c include/sq_engine.h .build/config
	$(CC) $(CPPFLAGS) $(CFLAGS) $(WARNFLAGS) -c $< -o $@
squid-c-generate: .build/generate.o libsq-gpu.a libsq.a
	$(CXX) $^ $(CUDA_LIBS) $(LDFLAGS) -o $@
.build/ar-compare.o: test/ar_compare.c include/sq_engine.h .build/config
	$(CC) $(CPPFLAGS) $(CFLAGS) $(WARNFLAGS) -c $< -o $@
.PHONY: ar-oracle
ar-oracle: generated-check .build/session-context.o .build/ar-compare.o libsq-gpu.a libsq.a
	$(ZIG) build-exe -O ReleaseFast --dep engine -Mroot=test/oracle/engine.zig -Mengine=src/engine.zig .build/session-context.o .build/ar-compare.o libsq-gpu.a libsq.a -L$(CUDA_ROOT)/lib64 -L$(CUDA_ROOT)/lib64/stubs -lcudart -lcublas -lcuda -lstdc++ -lm -lpthread -lc -femit-bin=.build/ar-oracle

.build/checked-prefill.cuda.o: src/cuda/checked_prefill.cu src/c/checked_prefill.h cuda/kernels_prefill.cu $(CUDA_HEADERS) .build/config
	$(NVCC) $(CUDAFLAGS) -c $< -o $@

.build/engine-test: test/engine.c test/mock_gpu.h .build/engine_bind.o .build/engine_layout.o .build/engine_runtime.o .build/engine_adaptive.o .build/engine_prefix.o libsq.a .build/config
	$(CC) $(CPPFLAGS) $(CFLAGS) $(WARNFLAGS) test/engine.c .build/engine_bind.o .build/engine_layout.o .build/engine_runtime.o .build/engine_adaptive.o .build/engine_prefix.o libsq.a -pthread $(LDFLAGS) $(LDLIBS) -o $@

.build/session-context.o: test/session_context.cu include/sq_engine.h .build/config
	$(NVCC) $(CUDAFLAGS) -Iinclude -c $< -o $@

.build/check-ar-cpp.o: test/installed_gpu_consumer.c include/sq_engine.h .build/config
	$(CXX) -std=c++11 -Wall -Wextra -Werror -Iinclude -x c++ -c $< -o $@
.build/decode-compare.o: test/decode_compare.c test/ar_compare.c include/sq_engine.h .build/config
	$(CC) $(CPPFLAGS) $(CFLAGS) $(WARNFLAGS) -c $< -o $@
.PHONY: decode-oracle
decode-oracle: generated-check .build/session-context.o .build/decode-compare.o libsq-gpu.a libsq.a
	$(ZIG) build-exe -O ReleaseFast --dep engine -Mroot=test/oracle/decode.zig -Mengine=src/engine.zig .build/session-context.o .build/decode-compare.o libsq-gpu.a libsq.a -L$(CUDA_ROOT)/lib64 -L$(CUDA_ROOT)/lib64/stubs -lcudart -lcublas -lcuda -lstdc++ -lm -lpthread -lc -femit-bin=.build/decode-oracle
.build/sampling-kernels.o: test/sampling_kernels.cu cuda/kernels_gemv.cu $(CUDA_HEADERS) .build/config
	$(NVCC) $(CUDAFLAGS) -c $< -o $@
.build/sampling-kernels: .build/sampling-kernels.o
	$(CXX) $^ $(CUDA_LIBS) -o $@
.PHONY: sampling-kernel-check
sampling-kernel-check: generated-check .build/sampling-kernels
	python3 tools/run_cuda_smoke.py '$(CUDA_ROOT)/lib64' .build/sampling-kernels

.build/sampling-test: test/sampling.c libsq.a .build/config
	$(CC) $(CPPFLAGS) $(CFLAGS) $(WARNFLAGS) $< libsq.a $(LDFLAGS) $(LDLIBS) -o $@
.build/decode-recovery.o: test/decode_recovery.c test/decode_compare.c test/ar_compare.c include/sq_engine.h .build/config
	$(CC) $(CPPFLAGS) $(CFLAGS) $(WARNFLAGS) -c $< -o $@
.build/decode-recovery: .build/decode-recovery.o .build/session-context.o libsq-gpu.a libsq.a
	$(CXX) $^ $(CUDA_LIBS) $(LDFLAGS) -o $@
.build/adaptive-compare.o: test/adaptive_compare.c test/decode_compare.c test/ar_compare.c include/sq_engine.h .build/config
	$(CC) $(CPPFLAGS) $(CFLAGS) $(WARNFLAGS) -c $< -o $@
.PHONY: adaptive-oracle
adaptive-oracle: generated-check .build/session-context.o .build/adaptive-compare.o libsq-gpu.a libsq.a
	$(ZIG) build-exe -O ReleaseFast --dep engine -Mroot=test/oracle/adaptive.zig -Mengine=src/engine.zig .build/session-context.o .build/adaptive-compare.o libsq-gpu.a libsq.a -L$(CUDA_ROOT)/lib64 -L$(CUDA_ROOT)/lib64/stubs -lcudart -lcublas -lcuda -lstdc++ -lm -lpthread -lc -femit-bin=.build/adaptive-oracle

.build/lookup-test: test/lookup.c libsq.a .build/config
	$(CC) $(CPPFLAGS) $(CFLAGS) $(WARNFLAGS) $< libsq.a $(LDFLAGS) $(LDLIBS) -o $@
.build/adaptive-graph.o: test/adaptive_graph.cu src/c/checked_cuda.h .build/config
	$(NVCC) $(CUDAFLAGS) -Isrc/c -c $< -o $@
.build/adaptive-graph: .build/adaptive-graph.o libsq-gpu.a libsq.a
	$(CXX) $^ $(CUDA_LIBS) -o $@

.build/adaptive-recovery.o: test/decode_recovery.c test/decode_compare.c test/ar_compare.c include/sq_engine.h .build/config
	$(CC) $(CPPFLAGS) $(CFLAGS) $(WARNFLAGS) -DSQ_TEST_ADAPTIVE=1 -c $< -o $@
.build/adaptive-recovery: .build/adaptive-recovery.o .build/session-context.o libsq-gpu.a libsq.a
	$(CXX) $^ $(CUDA_LIBS) $(LDFLAGS) -o $@

.build/prefix-compare.o: test/prefix_compare.c test/decode_compare.c test/ar_compare.c include/sq_engine.h .build/config
	$(CC) $(CPPFLAGS) $(CFLAGS) $(WARNFLAGS) -c $< -o $@
.PHONY: prefix-oracle
prefix-oracle: generated-check .build/session-context.o .build/prefix-compare.o libsq-gpu.a libsq.a
	$(ZIG) build-exe -O ReleaseFast --dep engine -Mroot=test/oracle/prefix.zig -Mengine=src/engine.zig .build/session-context.o .build/prefix-compare.o libsq-gpu.a libsq.a -L$(CUDA_ROOT)/lib64 -L$(CUDA_ROOT)/lib64/stubs -lcudart -lcublas -lcuda -lstdc++ -lm -lpthread -lc -femit-bin=.build/prefix-oracle
.build/prefix-recovery.o: test/prefix_recovery.c test/prefix_compare.c test/decode_compare.c test/ar_compare.c include/sq_engine.h .build/config
	$(CC) $(CPPFLAGS) $(CFLAGS) $(WARNFLAGS) -c $< -o $@
.build/prefix-recovery: .build/prefix-recovery.o .build/session-context.o libsq-gpu.a libsq.a
	$(CXX) $^ $(CUDA_LIBS) $(LDFLAGS) -o $@

.build/chat-fixture.json: test/chat_fixtures.py test/tokenizer_fixtures.py | .build
	python3 test/chat_fixtures.py $@
.build/chat-test: test/chat.c include/sq_chat.h libsq.a .build/config
	$(CC) $(CPPFLAGS) $(CFLAGS) $(WARNFLAGS) $< libsq.a $(LDFLAGS) $(LDLIBS) -o $@
.build/stream-driver: test/stream_driver.c include/sq_chat.h libsq.a .build/config
	$(CC) $(CPPFLAGS) $(CFLAGS) $(WARNFLAGS) $< libsq.a $(LDFLAGS) $(LDLIBS) -o $@
.PHONY: chat-oracle
chat-oracle: generated-check
	python3 tools/prepare_chat_oracle.py
	$(ZIG) build-exe -O ReleaseFast --dep server -Mroot=test/oracle/chat.zig -Mserver=.build/chat-oracle-src/server.zig -lc -femit-bin=.build/chat-oracle
.build/chat-native.o: test/chat_native.c test/decode_compare.c test/ar_compare.c include/sq_chat.h .build/config
	$(CC) $(CPPFLAGS) $(CFLAGS) $(WARNFLAGS) -c $< -o $@
.build/chat-native: .build/chat-native.o .build/session-context.o libsq-gpu.a libsq.a
	$(CXX) $^ $(CUDA_LIBS) $(LDFLAGS) -o $@
