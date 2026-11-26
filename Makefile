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
CPPFLAGS += -Iinclude -Isrc -D_FILE_OFFSET_BITS=64
PREFIX ?= /usr/local
CUDA_ROOT ?= /usr/local/cuda
NVCC = $(CUDA_ROOT)/bin/nvcc
CUDA_ARCH ?= 120a
CUDAFLAGS = -O3 --use_fast_math -std=c++20 -gencode arch=compute_$(CUDA_ARCH),code=sm_$(CUDA_ARCH) --expt-relaxed-constexpr -Icuda
CUDA_LIBS = -L$(CUDA_ROOT)/lib64 -lcudart -lcublas -L$(CUDA_ROOT)/lib64/stubs -lcuda -lstdc++ -lm -lpthread
include config/sources.mk
OBJ = $(HOST:src/%.c=.build/%.o)
PIC = $(HOST:src/%.c=.build/%.pic.o)
CUDA_HEADERS = $(wildcard cuda/*.hpp cuda/*.cuh)
CUDA_MATH = kernels_gemv kernels_inkling kernels_dspark kernels_prefill
CUDA_OBJECTS = $(addprefix .build/,$(addsuffix .cuda.o,$(CUDA_MATH)))
ifeq ($(shell uname -s),Darwin)
SHARED = libinkleberry.dylib
SHARED_FLAGS = -dynamiclib -Wl,-install_name,@rpath/libinkleberry.dylib
else
SHARED = libinkleberry.so
SHARED_FLAGS = -shared
endif
.PHONY: all check clean sanitize thread-sanitize fuzz-smoke generated-check format lint install install-check cuda-abi-check cuda-check FORCE
all: inkleberry libinkleberry.a $(SHARED)
.build:
	mkdir -p $@
FORCE:
.build/config: FORCE | .build
	@printf '%s\n' '$(CC) $(CXX) $(AR) $(CPPFLAGS) $(CFLAGS) $(WARNFLAGS) $(LDFLAGS)' '$(CUDA_ROOT) $(CUDAFLAGS)' > $@.tmp
	@cmp -s $@.tmp $@ || mv $@.tmp $@
	@rm -f $@.tmp
.build/%.o: src/%.c include/ifn.h src/internal.h .build/config
	$(CC) $(CPPFLAGS) $(CFLAGS) $(WARNFLAGS) -MMD -MP -c $< -o $@
.build/%.pic.o: src/%.c include/ifn.h src/internal.h .build/config
	$(CC) $(CPPFLAGS) $(CFLAGS) $(WARNFLAGS) -fPIC -fvisibility=hidden -DIFN_SHARED -DIFN_BUILD_SHARED -MMD -MP -c $< -o $@
libinkleberry.a: $(OBJ)
	$(AR) rcs $@ $^
$(SHARED): $(PIC)
	$(CC) $(SHARED_FLAGS) $(LDFLAGS) $^ $(LDLIBS) -o $@
inkleberry: cli/main.c libinkleberry.a .build/config
	$(CC) $(CPPFLAGS) $(CFLAGS) $(WARNFLAGS) $< libinkleberry.a $(LDFLAGS) $(LDLIBS) -o $@
.build/unit: test/unit.c libinkleberry.a .build/config
	$(CC) $(CPPFLAGS) $(CFLAGS) $(WARNFLAGS) $< libinkleberry.a $(LDFLAGS) $(LDLIBS) -o $@
.build/concurrency: test/concurrency.c libinkleberry.a .build/config
	$(CC) $(CPPFLAGS) $(CFLAGS) $(WARNFLAGS) $< libinkleberry.a -pthread $(LDFLAGS) $(LDLIBS) -o $@
generated-check:
	python3 tools/verify_upstream.py --production-only
	python3 tools/generate_cuda_abi.py --check
	python3 test/cuda_abi_schema.py
	python3 test/native_namespace.py
.PHONY: reference-check
reference-check: generated-check
	python3 tools/verify_upstream.py
.build/tokenizer-fixture.json: test/tokenizer_fixtures.py | .build
	python3 $< $@
.build/tokenizer-test: test/tokenizer.c libinkleberry.a .build/config
	$(CC) $(CPPFLAGS) $(CFLAGS) $(WARNFLAGS) $< libinkleberry.a -pthread $(LDFLAGS) $(LDLIBS) -o $@
.build/inkling-test: test/inkling.c libinkleberry.a .build/config
	$(CC) $(CPPFLAGS) $(CFLAGS) $(WARNFLAGS) $< libinkleberry.a $(LDFLAGS) $(LDLIBS) -o $@
.build/weights-test: test/weights_runtime.c test/mock_gpu.h libinkleberry.a .build/config
	$(CC) $(CPPFLAGS) $(CFLAGS) $(WARNFLAGS) $< libinkleberry.a -pthread $(LDFLAGS) $(LDLIBS) -o $@
check: .build/chat-native.o .build/chat-test .build/stream-driver .build/chat-fixture.json .build/prefix-recovery.o .build/prefix-compare.o .build/adaptive-compare.o .build/adaptive-recovery.o .build/lookup-test .build/sampling-test .build/decode-compare.o .build/engine_forward.o .build/engine_spec.o .build/engine_adaptive.o .build/engine_prefix.o .build/engine_prefill.o .build/generate.o .build/check-ar-cpp.o .build/engine-test .build/weights-test all generated-check .build/unit .build/concurrency .build/tokenizer-test .build/tokenizer-fixture.json .build/inkling-test
	.build/chat-test .build/chat-fixture.json
	python3 test/stream_cases.py .build/stream-driver
	.build/unit
	python3 test/namespace.py libinkleberry.a $(SHARED)
	.build/engine-test
	.build/sampling-test
	.build/lookup-test
	python3 test/weights_plan.py ./inkleberry
	.build/inkling-test
	python3 test/inkling_cases.py .build/inkling-test ./inkleberry .build/weights-test .build/infer-test .build/chat-fixture.json
	.build/tokenizer-test .build/tokenizer-fixture.json
	python3 test/tokenizer_cases.py .build/tokenizer-test ./inkleberry
	.build/concurrency
	python3 test/safetensors_cases.py ./inkleberry
	$(CC) $(CPPFLAGS) $(CFLAGS) $(WARNFLAGS) -c test/cuda_abi_c.c -o .build/cuda_abi_c.o
	$(CXX) -std=c++11 -Wall -Wextra -Werror -Iinclude -x c++ -c test/installed_consumer.c -o .build/cpp-consumer.o
	$(CC) .build/cpp-consumer.o libinkleberry.a $(LDFLAGS) $(LDLIBS) -o .build/cpp-consumer
	.build/cpp-consumer
sanitize:
	$(MAKE) clean
	$(MAKE) CC='$(SAN_CC)' CXX='$(SAN_CXX)' CFLAGS='-O1 -g -fno-omit-frame-pointer -fsanitize=address,undefined -fno-sanitize-recover=all' LDFLAGS='-fsanitize=address,undefined' check
thread-sanitize:
	$(MAKE) clean
	$(MAKE) CC='$(SAN_CC)' CXX='$(SAN_CXX)' CFLAGS='-O1 -g -fsanitize=thread' LDFLAGS='-fsanitize=thread' .build/infer-test .build/concurrency .build/tokenizer-test .build/tokenizer-fixture.json .build/weights-test .build/inkling-test .build/engine-test .build/sampling-test .build/lookup-test .build/chat-test .build/chat-fixture.json inkleberry
	.build/infer-test .build/chat-fixture.json
	.build/concurrency
	.build/tokenizer-test .build/tokenizer-fixture.json
	python3 test/inkling_cases.py .build/inkling-test ./inkleberry .build/weights-test .build/infer-test .build/chat-fixture.json
	.build/engine-test
	.build/sampling-test
	.build/lookup-test
	.build/chat-test .build/chat-fixture.json
fuzz-smoke: | .build
	$(SAN_CC) -O1 -g $(CPPFLAGS) -std=c11 -fsanitize=fuzzer,address,undefined -fno-sanitize-recover=all $(HOST) $(INFER) test/infer_mock.c fuzz/fuzz_infer.c -pthread -o .build/fuzz-infer
	.build/fuzz-infer -runs=3000 -max_len=4096
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
install: all infer
	mkdir -p $(DESTDIR)$(PREFIX)/include $(DESTDIR)$(PREFIX)/lib/pkgconfig $(DESTDIR)$(PREFIX)/bin
	cp include/ifn_infer.h include/ifn_infer_inkling.h include/ifn.h include/ifn_chat.h include/ifn_weights.h include/ifn_gpu.h include/ifn_engine.h $(DESTDIR)$(PREFIX)/include/
	cp libinkleberry-infer.a libinkleberry.a $(SHARED) $(DESTDIR)$(PREFIX)/lib/
	cp inkleberry $(DESTDIR)$(PREFIX)/bin/
	sed 's|@PREFIX@|$(PREFIX)|g' inkleberry.pc.in > $(DESTDIR)$(PREFIX)/lib/pkgconfig/inkleberry.pc
install-check: DESTDIR = $(CURDIR)/.build/stage
install-check: PREFIX = /usr/local
install-check: install
	$(CC) $(CFLAGS) $(WARNFLAGS) -I.build/stage/usr/local/include test/installed_consumer.c .build/stage/usr/local/lib/libinkleberry.a $(LDFLAGS) $(LDLIBS) -o .build/installed-static
	.build/installed-static
	$(CC) $(CFLAGS) $(WARNFLAGS) -I.build/stage/usr/local/include test/installed_consumer.c -L.build/stage/usr/local/lib -linkleberry -Wl,-rpath,'$(CURDIR)/.build/stage/usr/local/lib' $(LDFLAGS) $(LDLIBS) -o .build/installed-shared
	.build/installed-shared
.build/%.cuda.o: cuda/%.cu $(CUDA_HEADERS) .build/config
	$(NVCC) $(CUDAFLAGS) -c $< -o $@
.build/abi-probe.o: test/cuda_abi_probe.cu src/cuda_abi.h cuda/cuda_abi.cu $(CUDA_HEADERS) .build/config
	$(NVCC) $(CUDAFLAGS) -Isrc -c $< -o $@
.build/abi-c.o: test/cuda_abi_c.c src/cuda_abi.h .build/config
	$(CC) $(CPPFLAGS) $(CFLAGS) $(WARNFLAGS) -c $< -o $@
.build/abi-probe: .build/abi-probe.o .build/abi-c.o $(CUDA_OBJECTS)
	$(CXX) $^ $(CUDA_LIBS) -o $@
cuda-abi-check: generated-check .build/abi-probe
	LD_LIBRARY_PATH='$(CUDA_ROOT)/lib64' .build/abi-probe
.build/cuda-smoke.o: test/cuda_smoke.c src/cuda_abi.h .build/config
	$(CC) $(CPPFLAGS) $(CFLAGS) $(WARNFLAGS) -c $< -o $@
.build/cuda-smoke: .build/cuda-smoke.o .build/cuda_abi.cuda.o $(CUDA_OBJECTS)
	$(CXX) $^ $(CUDA_LIBS) -o $@
cuda-check: cuda-abi-check .build/cuda-smoke
	python3 tools/run_cuda_smoke.py '$(CUDA_ROOT)/lib64' .build/cuda-smoke
clean:
	rm -rf .build inkleberry libinkleberry.a libinkleberry.so libinkleberry.dylib libinkleberry-gpu.a libinkleberry-infer.a libinkleberry_backend.so.1 libinkleberry-cuda.a inkleberry-weights inkleberry-generate
-include $(wildcard .build/*.d)

# GPU backend is opt-in; the host library still builds without CUDA.
.PHONY: gpu
gpu: generated-check libinkleberry-gpu.a
.build/checked-weights.cuda.o: src/cuda/checked_weights.cu src/checked_cuda.h cuda/cuda_abi.cu $(CUDA_HEADERS) .build/config
	$(NVCC) $(CUDAFLAGS) -c $< -o $@
libinkleberry-gpu.a: .build/checked-prefill.cuda.o .build/engine_bind.o .build/engine_layout.o .build/engine_forward.o .build/engine_spec.o .build/engine_adaptive.o .build/engine_prefix.o .build/engine_prefill.o .build/engine_runtime.o .build/gpu_cuda.o .build/checked-weights.cuda.o $(filter-out .build/kernels_prefill.cuda.o,$(CUDA_OBJECTS))
	rm -f $@
	$(AR) rcs $@ $^
.build/gpu-weight-compare.o: test/gpu_weights_compare.c include/ifn_gpu.h .build/config
	$(CC) $(CPPFLAGS) $(CFLAGS) $(WARNFLAGS) -c $< -o $@
.build/gpu-weight-main.o: test/gpu_weights_compare.c include/ifn_gpu.h .build/config
	$(CC) $(CPPFLAGS) $(CFLAGS) $(WARNFLAGS) -DIFN_STANDALONE_WEIGHTS -c $< -o $@
inkleberry-weights: .build/gpu-context.o .build/gpu-weight-main.o libinkleberry-gpu.a libinkleberry.a
	$(CXX) $^ $(CUDA_LIBS) $(LDFLAGS) -o $@

.build/gpu-context.o: test/gpu_context.cu include/ifn_gpu.h .build/config
	$(NVCC) $(CUDAFLAGS) -Iinclude -c $< -o $@
.build/packing.o: test/packing.cu src/checked_cuda.h .build/config
	$(NVCC) $(CUDAFLAGS) -Isrc -c $< -o $@
.build/packing: .build/packing.o libinkleberry-gpu.a
	$(CXX) $^ $(CUDA_LIBS) -o $@
.PHONY: gpu-packing-check install-gpu
gpu-packing-check: generated-check .build/packing
	python3 tools/run_cuda_smoke.py '$(CUDA_ROOT)/lib64' .build/packing
install-gpu: install gpu
	cp libinkleberry-gpu.a $(DESTDIR)$(PREFIX)/lib/

.PHONY: gpu-install-check
gpu-install-check: DESTDIR = $(CURDIR)/.build/gpu-stage
gpu-install-check: PREFIX = /usr/local
gpu-install-check: install-gpu
	$(CC) $(CFLAGS) $(WARNFLAGS) -I.build/gpu-stage/usr/local/include -c test/installed_gpu_consumer.c -o .build/installed-gpu.o
	$(CXX) .build/installed-gpu.o .build/gpu-stage/usr/local/lib/libinkleberry-gpu.a .build/gpu-stage/usr/local/lib/libinkleberry.a $(CUDA_LIBS) -o .build/installed-gpu
	LD_LIBRARY_PATH='$(CUDA_ROOT)/lib64' .build/installed-gpu
	$(CXX) -std=c++11 -Wall -Wextra -Werror -I.build/gpu-stage/usr/local/include -x c++ -c test/installed_gpu_consumer.c -o .build/installed-gpu-cpp.o
	$(CXX) .build/installed-gpu-cpp.o .build/gpu-stage/usr/local/lib/libinkleberry-gpu.a .build/gpu-stage/usr/local/lib/libinkleberry.a $(CUDA_LIBS) -o .build/installed-gpu-cpp
	LD_LIBRARY_PATH='$(CUDA_ROOT)/lib64' .build/installed-gpu-cpp
.build/generate.o: cli/generate.c include/ifn_engine.h .build/config
	$(CC) $(CPPFLAGS) $(CFLAGS) $(WARNFLAGS) -c $< -o $@
inkleberry-generate: .build/generate.o libinkleberry-gpu.a libinkleberry.a
	$(CXX) $^ $(CUDA_LIBS) $(LDFLAGS) -o $@
.build/ar-compare.o: test/ar_compare.c include/ifn_engine.h .build/config
	$(CC) $(CPPFLAGS) $(CFLAGS) $(WARNFLAGS) -c $< -o $@

.build/checked-prefill.cuda.o: src/cuda/checked_prefill.cu src/checked_prefill.h cuda/kernels_prefill.cu $(CUDA_HEADERS) .build/config
	$(NVCC) $(CUDAFLAGS) -c $< -o $@

.build/engine-test: test/engine.c test/mock_gpu.h .build/engine_bind.o .build/engine_layout.o .build/engine_runtime.o .build/engine_adaptive.o .build/engine_prefix.o libinkleberry.a .build/config
	$(CC) $(CPPFLAGS) $(CFLAGS) $(WARNFLAGS) test/engine.c .build/engine_bind.o .build/engine_layout.o .build/engine_runtime.o .build/engine_adaptive.o .build/engine_prefix.o libinkleberry.a -pthread $(LDFLAGS) $(LDLIBS) -o $@

.build/session-context.o: test/session_context.cu include/ifn_engine.h .build/config
	$(NVCC) $(CUDAFLAGS) -Iinclude -c $< -o $@

.build/check-ar-cpp.o: test/installed_gpu_consumer.c include/ifn_engine.h .build/config
	$(CXX) -std=c++11 -Wall -Wextra -Werror -Iinclude -x c++ -c $< -o $@
.build/decode-compare.o: test/decode_compare.c test/ar_compare.c include/ifn_engine.h .build/config
	$(CC) $(CPPFLAGS) $(CFLAGS) $(WARNFLAGS) -c $< -o $@
.build/sampling-kernels.o: test/sampling_kernels.cu cuda/kernels_gemv.cu $(CUDA_HEADERS) .build/config
	$(NVCC) $(CUDAFLAGS) -c $< -o $@
.build/sampling-kernels: .build/sampling-kernels.o
	$(CXX) $^ $(CUDA_LIBS) -o $@
.PHONY: sampling-kernel-check
sampling-kernel-check: generated-check .build/sampling-kernels
	python3 tools/run_cuda_smoke.py '$(CUDA_ROOT)/lib64' .build/sampling-kernels

.build/sampling-test: test/sampling.c libinkleberry.a .build/config
	$(CC) $(CPPFLAGS) $(CFLAGS) $(WARNFLAGS) $< libinkleberry.a $(LDFLAGS) $(LDLIBS) -o $@
.build/decode-recovery.o: test/decode_recovery.c test/decode_compare.c test/ar_compare.c include/ifn_engine.h .build/config
	$(CC) $(CPPFLAGS) $(CFLAGS) $(WARNFLAGS) -c $< -o $@
.build/decode-recovery: .build/decode-recovery.o .build/session-context.o libinkleberry-gpu.a libinkleberry.a
	$(CXX) $^ $(CUDA_LIBS) $(LDFLAGS) -o $@
.build/adaptive-compare.o: test/adaptive_compare.c test/decode_compare.c test/ar_compare.c include/ifn_engine.h .build/config
	$(CC) $(CPPFLAGS) $(CFLAGS) $(WARNFLAGS) -c $< -o $@

.build/lookup-test: test/lookup.c libinkleberry.a .build/config
	$(CC) $(CPPFLAGS) $(CFLAGS) $(WARNFLAGS) $< libinkleberry.a $(LDFLAGS) $(LDLIBS) -o $@
.build/adaptive-graph.o: test/adaptive_graph.cu src/checked_cuda.h .build/config
	$(NVCC) $(CUDAFLAGS) -Isrc -c $< -o $@
.build/adaptive-graph: .build/adaptive-graph.o libinkleberry-gpu.a libinkleberry.a
	$(CXX) $^ $(CUDA_LIBS) -o $@

.build/adaptive-recovery.o: test/decode_recovery.c test/decode_compare.c test/ar_compare.c include/ifn_engine.h .build/config
	$(CC) $(CPPFLAGS) $(CFLAGS) $(WARNFLAGS) -DIFN_TEST_ADAPTIVE=1 -c $< -o $@
.build/adaptive-recovery: .build/adaptive-recovery.o .build/session-context.o libinkleberry-gpu.a libinkleberry.a
	$(CXX) $^ $(CUDA_LIBS) $(LDFLAGS) -o $@

.build/prefix-compare.o: test/prefix_compare.c test/decode_compare.c test/ar_compare.c include/ifn_engine.h .build/config
	$(CC) $(CPPFLAGS) $(CFLAGS) $(WARNFLAGS) -c $< -o $@
.build/prefix-recovery.o: test/prefix_recovery.c test/prefix_compare.c test/decode_compare.c test/ar_compare.c include/ifn_engine.h .build/config
	$(CC) $(CPPFLAGS) $(CFLAGS) $(WARNFLAGS) -c $< -o $@
.build/prefix-recovery: .build/prefix-recovery.o .build/session-context.o libinkleberry-gpu.a libinkleberry.a
	$(CXX) $^ $(CUDA_LIBS) $(LDFLAGS) -o $@

.build/chat-fixture.json: test/chat_fixtures.py test/tokenizer_fixtures.py | .build
	python3 test/chat_fixtures.py $@
.build/chat-test: test/chat.c include/ifn_chat.h libinkleberry.a .build/config
	$(CC) $(CPPFLAGS) $(CFLAGS) $(WARNFLAGS) $< libinkleberry.a $(LDFLAGS) $(LDLIBS) -o $@
.build/stream-driver: test/stream_driver.c include/ifn_chat.h libinkleberry.a .build/config
	$(CC) $(CPPFLAGS) $(CFLAGS) $(WARNFLAGS) $< libinkleberry.a $(LDFLAGS) $(LDLIBS) -o $@
.build/chat-native.o: test/chat_native.c test/decode_compare.c test/ar_compare.c include/ifn_chat.h .build/config
	$(CC) $(CPPFLAGS) $(CFLAGS) $(WARNFLAGS) -c $< -o $@
.build/chat-native: .build/chat-native.o .build/session-context.o libinkleberry-gpu.a libinkleberry.a
	$(CXX) $^ $(CUDA_LIBS) $(LDFLAGS) -o $@

# Serving ABI: separate from libinkleberry's CPU-only parser/tokenizer contract.
INFER_OBJ = $(INFER:src/%.c=.build/%.o)
.PHONY: infer infer-check infer-shared
infer: libinkleberry-infer.a
libinkleberry-infer.a: $(INFER_OBJ)
	$(AR) rcs $@ $^
.build/infer-test: test/infer.c test/infer_mock.c libinkleberry-infer.a libinkleberry.a .build/config
	$(CC) $(CPPFLAGS) $(CFLAGS) $(WARNFLAGS) test/infer.c test/infer_mock.c libinkleberry-infer.a libinkleberry.a -pthread $(LDFLAGS) $(LDLIBS) -o $@
infer-check: .build/infer-test .build/chat-fixture.json
	.build/infer-test .build/chat-fixture.json
check: infer-check
BACKEND_C = $(INFER) $(GPU_HOST)
BACKEND_OBJ = $(BACKEND_C:src/%.c=.build/backend-%.o)
BACKEND_CUDA = $(addprefix .build/backend-,$(addsuffix .cuda.o,kernels_gemv kernels_inkling kernels_dspark)) .build/backend-checked-weights.cuda.o .build/backend-checked-prefill.cuda.o
.build/backend-%.o: src/%.c .build/config
	$(CC) $(CPPFLAGS) $(CFLAGS) $(WARNFLAGS) -fPIC -fvisibility=hidden -DIFN_SHARED -MMD -MP -c $< -o $@
.build/backend-%.cuda.o: cuda/%.cu $(CUDA_HEADERS) .build/config
	$(NVCC) $(CUDAFLAGS) -Xcompiler=-fPIC,-fvisibility=hidden -c $< -o $@
.build/backend-checked-weights.cuda.o: src/cuda/checked_weights.cu src/checked_cuda.h cuda/cuda_abi.cu $(CUDA_HEADERS) .build/config
	$(NVCC) $(CUDAFLAGS) -Xcompiler=-fPIC,-fvisibility=hidden -c $< -o $@
.build/backend-checked-prefill.cuda.o: src/cuda/checked_prefill.cu src/checked_prefill.h cuda/kernels_prefill.cu $(CUDA_HEADERS) .build/config
	$(NVCC) $(CUDAFLAGS) -Xcompiler=-fPIC,-fvisibility=hidden -c $< -o $@
libinkleberry_backend.so.1: $(BACKEND_OBJ) $(BACKEND_CUDA) $(PIC) src/infer.exports
	$(CXX) -shared -Wl,-soname,libinkleberry_backend.so.1 -Wl,--version-script=src/infer.exports -Wl,-z,defs $(BACKEND_OBJ) $(BACKEND_CUDA) $(PIC) $(CUDA_LIBS) $(LDFLAGS) -o $@
infer-shared: generated-check libinkleberry_backend.so.1
.build/infer-native.o: test/infer_native.c include/ifn_infer.h .build/config
	$(CC) $(CPPFLAGS) $(CFLAGS) $(WARNFLAGS) -c $< -o $@
.build/infer-native: .build/infer-native.o libinkleberry-infer.a libinkleberry-gpu.a libinkleberry.a
	$(CXX) $^ $(CUDA_LIBS) $(LDFLAGS) -o $@

.build/infer-consumer: test/infer_consumer.c .build/stage/usr/local/include/ifn_infer.h libinkleberry-infer.a libinkleberry.a .build/config
	$(CC) $(CFLAGS) $(WARNFLAGS) -I.build/stage/usr/local/include -Iinclude -Isrc test/infer_consumer.c test/infer_mock.c .build/stage/usr/local/lib/libinkleberry-infer.a .build/stage/usr/local/lib/libinkleberry.a -pthread $(LDFLAGS) $(LDLIBS) -o $@
.build/stage/usr/local/include/ifn_infer.h: install
.build/infer-mock.pic.o: test/infer_mock.c .build/config
	$(CC) $(CPPFLAGS) $(CFLAGS) $(WARNFLAGS) -fPIC -MMD -MP -c $< -o $@
.build/infer-mock.so: $(INFER:src/%.c=.build/%.pic.o) $(PIC) .build/infer-mock.pic.o
	$(CC) $(SHARED_FLAGS) $^ -pthread $(LDFLAGS) $(LDLIBS) -o $@
.PHONY: infer-consumer-check
infer-consumer-check: .build/infer-consumer .build/infer-mock.so
	.build/infer-consumer
	$(CXX) -std=c++11 -Wall -Wextra -Werror -I.build/stage/usr/local/include -x c++ -c test/infer_consumer.c -o .build/infer-consumer-cpp.o
	$(CC) .build/infer-consumer-cpp.o .build/infer-mock.pic.o libinkleberry-infer.a libinkleberry.a -pthread $(LDFLAGS) $(LDLIBS) -o .build/infer-consumer-cpp
	.build/infer-consumer-cpp
	python3 test/infer_ffi.py ./.build/infer-mock.so
install-check: infer-consumer-check
.build/infer-native-shared: .build/infer-native.o libinkleberry_backend.so.1
	$(CC) $^ -Wl,-rpath,'$(CURDIR)' -pthread $(LDFLAGS) -o $@
.PHONY: install-infer-backend
install-infer-backend: install-gpu infer-shared
	cp libinkleberry_backend.so.1 $(DESTDIR)$(PREFIX)/lib/
	ln -sf libinkleberry_backend.so.1 $(DESTDIR)$(PREFIX)/lib/libinkleberry_backend.so
	sed 's|@PREFIX@|$(PREFIX)|g' inkleberry-infer.pc.in > $(DESTDIR)$(PREFIX)/lib/pkgconfig/inkleberry-infer.pc
.PHONY: infer-provider-check
infer-provider-check: infer-shared
	python3 test/infer_exports.py ./libinkleberry_backend.so.1
	LD_LIBRARY_PATH='$(CUDA_ROOT)/lib64' python3 test/infer_ffi.py ./libinkleberry_backend.so.1

.PHONY: cuda-abi-negative-check reference-free-check
cuda-abi-negative-check: cuda-abi-check
	python3 test/cuda_abi_negative.py --cc '$(CC)' --cxx '$(CXX)' --nvcc '$(NVCC)' --cuda-root '$(CUDA_ROOT)' --cuda-flags '$(CUDAFLAGS)' --cuda-objects '$(CUDA_OBJECTS)'
reference-free-check:
	python3 test/reference_free.py --cc '$(CC)' --cxx '$(CXX)'

# Optional comparison tools are excluded from production source packages.
-include reference/oracles.mk

.PHONY: amalgamation amalgamation-check amalgamation-test
amalgamation:
	python3 tools/amalgamate.py
amalgamation-check:
	python3 tools/amalgamate.py --check
	python3 test/amalgamate_generator.py
.build/amalgam.o: dist/ifn.c dist/ifn.h .build/config
	$(CC) $(CFLAGS) $(WARNFLAGS) -Idist -c $< -o $@
.build/amalgam-gpu.o: dist/ifn.c dist/ifn.h .build/config
	$(CC) $(CFLAGS) $(WARNFLAGS) -Idist -DIFN_ENABLE_CUDA -c $< -o $@
.build/amalgam-%: test/%.c .build/amalgam.o .build/config
	$(CC) $(CPPFLAGS) $(CFLAGS) $(WARNFLAGS) $< .build/amalgam.o -pthread $(LDFLAGS) $(LDLIBS) -o $@
.build/embedding-consumer: examples/inspect.c .build/amalgam.o .build/config
	$(CC) $(CFLAGS) $(WARNFLAGS) -Idist $< .build/amalgam.o $(LDFLAGS) $(LDLIBS) -o $@
.build/embedding-consumer-cpp.o: examples/inspect.c dist/ifn.h .build/config
	$(CXX) -std=c++11 -Wall -Wextra -Werror -Idist -x c++ -c $< -o $@
.build/mmap-test: test/mmap.c libinkleberry.a .build/config
	$(CC) $(CPPFLAGS) $(CFLAGS) $(WARNFLAGS) $< libinkleberry.a $(LDFLAGS) $(LDLIBS) -o $@
amalgamation-test: amalgamation-check .build/embedding-generate.o .build/amalgam-unit .build/amalgam-tokenizer .build/amalgam-chat .build/amalgam-stream_driver .build/amalgam-concurrency .build/amalgam-mmap .build/embedding-consumer .build/embedding-consumer-cpp.o .build/tokenizer-fixture.json .build/chat-fixture.json libinkleberry.a
	.build/amalgam-unit
	.build/amalgam-tokenizer .build/tokenizer-fixture.json
	.build/amalgam-chat .build/chat-fixture.json
	python3 test/stream_cases.py .build/amalgam-stream_driver
	.build/amalgam-concurrency
	.build/amalgam-mmap
	python3 test/amalgamation.py libinkleberry.a .build/amalgam.o
check: amalgamation-test mmap-check
.PHONY: mmap-check
mmap-check: .build/mmap-test
	.build/mmap-test
lint: amalgamation-check
# CUDA-only archive supports the single C host without duplicate host objects.
libinkleberry-cuda.a: .build/checked-prefill.cuda.o .build/checked-weights.cuda.o $(filter-out .build/kernels_prefill.cuda.o,$(CUDA_OBJECTS))
	$(AR) rcs $@ $^
.build/infer-native-amalgam: .build/infer-native.o .build/amalgam-gpu.o libinkleberry-cuda.a
	$(CXX) $^ $(CUDA_LIBS) $(LDFLAGS) -o $@

.PHONY: source-package embedding-package runtime-package runtime-gpu-package packages release-check
VERSION ?= 0.1.0-dev
source-package: generated-check amalgamation-check
	python3 tools/release.py source --version '$(VERSION)'
embedding-package: amalgamation-check
	python3 tools/release.py embedding --version '$(VERSION)'
runtime-package: all
	python3 tools/release.py runtime --version '$(VERSION)'
runtime-gpu-package: all inkleberry-generate infer-shared libinkleberry-cuda.a
	python3 tools/release.py runtime --gpu --version '$(VERSION)'
packages: source-package embedding-package runtime-package
release-check: all amalgamation-check
	python3 test/package_policy.py
	python3 test/release.py --cc '$(CC)' --cxx '$(CXX)'

.build/embedding-generate.o: examples/generate.c dist/ifn.h .build/config
	$(CC) $(CFLAGS) $(WARNFLAGS) -Idist -c $< -o $@
.build/embedding-generate: .build/embedding-generate.o .build/amalgam-gpu.o libinkleberry-cuda.a
	$(CXX) $^ $(CUDA_LIBS) $(LDFLAGS) -o $@

.PHONY: native-release-check
native-release-check: generated-check all amalgamation-check
	python3 test/native_release.py --model '$(MODEL_DIR)' --cuda-root '$(CUDA_ROOT)' --cc '$(CC)' --cxx '$(CXX)'

.PHONY: amalgamation-fuzz-smoke
amalgamation-fuzz-smoke: | .build
	$(SAN_CC) -O1 -g $(CPPFLAGS) -std=c11 -fsanitize=fuzzer,address,undefined -fno-sanitize-recover=all dist/ifn.c fuzz/fuzz_stream.c -o .build/fuzz-amalgam-stream
	.build/fuzz-amalgam-stream -runs=3000 -max_len=32768
	python3 test/safetensors_cases.py --corpus .build/fuzz-corpus
	$(SAN_CC) -O1 -g $(CPPFLAGS) -std=c11 -fsanitize=fuzzer,address,undefined -fno-sanitize-recover=all dist/ifn.c fuzz/fuzz_safetensors.c -o .build/fuzz-amalgam-safetensors
	.build/fuzz-amalgam-safetensors -runs=3000 -max_len=262144 .build/fuzz-corpus
	python3 test/tokenizer_cases.py --corpus .build/tokenizer-corpus
	$(SAN_CC) -O1 -g $(CPPFLAGS) -std=c11 -fsanitize=fuzzer,address,undefined -fno-sanitize-recover=all dist/ifn.c fuzz/fuzz_tokenizer.c -o .build/fuzz-amalgam-tokenizer
	.build/fuzz-amalgam-tokenizer -runs=3000 -max_len=32768 .build/tokenizer-corpus

.PHONY: native-namespace-check
native-namespace-check: libinkleberry-cuda.a libinkleberry_backend.so.1
	python3 test/native_symbols.py libinkleberry-cuda.a libinkleberry_backend.so.1
