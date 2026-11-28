# C library/installed-consumer conventions adapted from artalis-io/jev-bush (0BSD).
.DEFAULT_GOAL := all
.DELETE_ON_ERROR:
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
LIBDIR ?= $(PREFIX)/lib
INCLUDEDIR ?= $(PREFIX)/include
BINDIR ?= $(PREFIX)/bin
PKGCONFIGDIR ?= $(LIBDIR)/pkgconfig
IFN_VERSION := $(shell cat VERSION)
FUZZ_RUNS ?= 3000
FUZZ_SECONDS ?= 0
HARDEN ?= 1
ifeq ($(HARDEN),1)
CFLAGS += -fstack-protector-strong
ifeq ($(shell uname -s),Linux)
CPPFLAGS += -D_FORTIFY_SOURCE=3
PROGRAM_COMPILE = -fPIE
PROGRAM_HARDEN = -pie
LDFLAGS += -Wl,-z,relro,-z,now,-z,noexecstack
endif
endif
CUDA_ROOT ?= $(if $(CUDA_PATH),$(subst \,/,$(CUDA_PATH)),/usr/local/cuda)
NVCC = $(CUDA_ROOT)/bin/nvcc
CUDA_ARCH ?= 120a
CUDAFLAGS = -O3 --use_fast_math -std=c++20 -gencode arch=compute_$(CUDA_ARCH),code=sm_$(CUDA_ARCH) --expt-relaxed-constexpr -Icuda
CUDA_RUNTIME_PATH ?= $(CUDA_ROOT)/lib64
CUDA_LIBS = -L$(CUDA_ROOT)/lib64 -lcudart -lcublas -L$(CUDA_ROOT)/lib64/stubs -lcuda -lstdc++ -lm -lpthread
include config/sources.mk
OBJ = $(HOST:src/%.c=$(BUILD_DIR)/%.o)
PIC = $(HOST:src/%.c=$(BUILD_DIR)/%.pic.o)
IFN_HEADER_DEPS = $(wildcard include/*.h src/*.h)
CUDA_HEADERS = $(wildcard cuda/*.hpp cuda/*.cuh src/cuda/*.h src/cuda/*.hpp) src/inkling_geometry.h
CUDA_MATH = kernels_gemv kernels_inkling kernels_dspark kernels_prefill
CUDA_OBJECTS = $(addprefix $(BUILD_DIR)/,$(addsuffix .cuda.o,$(CUDA_MATH)))
ifeq ($(shell uname -s),Darwin)
SHARED = libinkleberry.dylib
SHARED_REAL = libinkleberry.$(IFN_VERSION).dylib
SHARED_ABI = libinkleberry.1.dylib
SHARED_FLAGS = -dynamiclib -Wl,-install_name,@rpath/$(SHARED_ABI)
PLAIN_SHARED_FLAGS = -dynamiclib
else
SHARED = libinkleberry.so
SHARED_REAL = libinkleberry.so.$(IFN_VERSION)
SHARED_ABI = libinkleberry.so.1
SHARED_FLAGS = -shared -Wl,-soname,$(SHARED_ABI)
PLAIN_SHARED_FLAGS = -shared
endif
include mk/toolchain.mk
include mk/cuda-profile.mk
export CC CXX PYTHON BUILD_DIR IFN_WINDOWS

.PHONY: all check clean sanitize thread-sanitize fuzz-smoke generated-check format lint install install-check cuda-abi-check cuda-check FORCE
all: $(CLI) $(HOST_STATIC) $(SHARED)
$(BUILD_DIR):
	mkdir -p $@
FORCE:
$(BUILD_DIR)/config: FORCE | $(BUILD_DIR)
	@printf '%s\n' '$(CC) $(CXX) $(AR) $(CPPFLAGS) $(CFLAGS) $(WARNFLAGS) $(LDFLAGS)' '$(CUDA_ROOT) $(CUDAFLAGS) $(CUDA_TARGET_FLAGS) $(CUDA_PIC_FLAGS) $(CXX_WARNFLAGS) $(CUDA_LIBS) $(IFN_LINKER) $(PROVIDER_FLAGS) $(IFN_VERSION) $(HARDEN) $(PROGRAM_COMPILE) $(PROGRAM_HARDEN)' > $@.tmp
	@cmp -s $@.tmp $@ || mv $@.tmp $@
	@rm -f $@.tmp
$(BUILD_DIR)/%.o: src/%.c include/ifn.h src/internal.h $(IFN_HEADER_DEPS) $(BUILD_DIR)/config
	$(call ifn_compile,$(CC),$(CPPFLAGS) $(CFLAGS) $(WARNFLAGS) $(DEPFLAGS),$<,$@)
$(BUILD_DIR)/%.pic.o: src/%.c include/ifn.h src/internal.h $(IFN_HEADER_DEPS) $(BUILD_DIR)/config
	$(call ifn_compile,$(CC),$(CPPFLAGS) $(CFLAGS) $(WARNFLAGS) $(PIC_FLAGS) $(HIDDEN_FLAGS) -DIFN_SHARED -DIFN_BUILD_SHARED $(DEPFLAGS),$<,$@)
$(HOST_STATIC): $(OBJ)
	$(call ifn_archive,$@,$^)
$(SHARED_REAL): $(PIC)
	$(call ifn_shared,$(CC),,$^,$(SHARED_FLAGS) $(LDFLAGS) $(LDLIBS),$@)
ifneq ($(SHARED),$(SHARED_REAL))
$(SHARED): $(SHARED_REAL)
	ln -sf $(notdir $(SHARED_REAL)) $(SHARED)
	ln -sf $(notdir $(SHARED_REAL)) $(SHARED_ABI)
endif
$(CLI): cli/main.c $(CLI_HOST) $(BUILD_DIR)/config
	$(call ifn_program,$(CC),$(CPPFLAGS) $(CFLAGS) $(WARNFLAGS) $(PROGRAM_COMPILE),$< $(CLI_HOST),$(LDFLAGS) $(PROGRAM_HARDEN) $(LDLIBS),$@)
$(BUILD_DIR)/unit$(EXE): test/unit.c $(HOST_STATIC) $(BUILD_DIR)/config
	$(call ifn_program,$(CC),$(CPPFLAGS) $(CFLAGS) $(WARNFLAGS),$< $(HOST_STATIC),$(LDFLAGS) $(LDLIBS),$@)
$(BUILD_DIR)/concurrency$(EXE): test/concurrency.c $(HOST_STATIC) $(BUILD_DIR)/config
	$(call ifn_program,$(CC),$(CPPFLAGS) $(CFLAGS) $(WARNFLAGS),$< $(HOST_STATIC),$(THREAD_LIBS) $(LDFLAGS) $(LDLIBS),$@)
generated-check:
	$(PYTHON) tools/version.py --check
	$(PYTHON) tools/verify_upstream.py
	$(PYTHON) tools/generate_cuda_abi.py --check
	$(PYTHON) test/cuda_abi_schema.py
	$(PYTHON) test/native_namespace.py
	$(PYTHON) test/cuda_guard.py
$(BUILD_DIR)/tokenizer-fixture.json: test/tokenizer_fixtures.py | $(BUILD_DIR)
	$(PYTHON) $< $@
$(BUILD_DIR)/tokenizer-test$(EXE): test/tokenizer.c $(HOST_STATIC) $(BUILD_DIR)/config
	$(call ifn_program,$(CC),$(CPPFLAGS) $(CFLAGS) $(WARNFLAGS),$< $(HOST_STATIC),$(THREAD_LIBS) $(LDFLAGS) $(LDLIBS),$@)
$(BUILD_DIR)/inkling-test$(EXE): test/inkling.c $(HOST_STATIC) $(BUILD_DIR)/config
	$(call ifn_program,$(CC),$(CPPFLAGS) $(CFLAGS) $(WARNFLAGS),$< $(HOST_STATIC),$(LDFLAGS) $(LDLIBS),$@)
$(BUILD_DIR)/weights-test$(EXE): test/weights_runtime.c test/mock_gpu.h $(HOST_STATIC) $(BUILD_DIR)/config
	$(call ifn_program,$(CC),$(CPPFLAGS) $(CFLAGS) $(WARNFLAGS),$< $(HOST_STATIC),$(THREAD_LIBS) $(LDFLAGS) $(LDLIBS),$@)
check: $(BUILD_DIR)/chat-native.o $(BUILD_DIR)/chat-test$(EXE) $(BUILD_DIR)/stream-driver$(EXE) $(BUILD_DIR)/chat-fixture.json $(BUILD_DIR)/prefix-recovery.o $(BUILD_DIR)/prefix-compare.o $(BUILD_DIR)/adaptive-compare.o $(BUILD_DIR)/adaptive-recovery.o $(BUILD_DIR)/lookup-test$(EXE) $(BUILD_DIR)/sampling-test$(EXE) $(BUILD_DIR)/decode-compare.o $(BUILD_DIR)/engine_forward.o $(BUILD_DIR)/engine_spec.o $(BUILD_DIR)/engine_adaptive.o $(BUILD_DIR)/engine_prefix.o $(BUILD_DIR)/engine_prefill.o $(BUILD_DIR)/generate.o $(BUILD_DIR)/check-ar-cpp.o $(BUILD_DIR)/engine-test$(EXE) $(BUILD_DIR)/weights-test$(EXE) all generated-check $(BUILD_DIR)/unit$(EXE) $(BUILD_DIR)/concurrency$(EXE) $(BUILD_DIR)/tokenizer-test$(EXE) $(BUILD_DIR)/tokenizer-fixture.json $(BUILD_DIR)/inkling-test$(EXE)
	$(BUILD_DIR)/chat-test$(EXE) $(BUILD_DIR)/chat-fixture.json
	$(PYTHON) test/stream_cases.py $(BUILD_DIR)/stream-driver$(EXE)
	$(BUILD_DIR)/unit$(EXE)
	$(PYTHON) test/namespace.py $(HOST_STATIC) $(SHARED)
	$(BUILD_DIR)/engine-test$(EXE)
	$(BUILD_DIR)/sampling-test$(EXE)
	$(BUILD_DIR)/lookup-test$(EXE)
	$(PYTHON) test/weights_plan.py ./$(CLI)
	$(BUILD_DIR)/inkling-test$(EXE)
	$(PYTHON) test/inkling_cases.py $(BUILD_DIR)/inkling-test$(EXE) ./$(CLI) $(BUILD_DIR)/weights-test$(EXE) $(BUILD_DIR)/infer-test$(EXE) $(BUILD_DIR)/chat-fixture.json
	$(BUILD_DIR)/tokenizer-test$(EXE) $(BUILD_DIR)/tokenizer-fixture.json
	$(PYTHON) test/tokenizer_cases.py $(BUILD_DIR)/tokenizer-test$(EXE) ./$(CLI)
	$(BUILD_DIR)/concurrency$(EXE)
	$(PYTHON) test/safetensors_cases.py ./$(CLI)
	$(call ifn_compile,$(CC),$(CPPFLAGS) $(CFLAGS) $(WARNFLAGS),test/cuda_abi_c.c,$(BUILD_DIR)/cuda_abi_c.o)
	$(call ifn_compile,$(CXX),$(CXX_WARNFLAGS) -Iinclude $(CXX_LANGUAGE),test/installed_consumer.c,$(BUILD_DIR)/cpp-consumer.o)
	$(call ifn_program,$(CC),,$(BUILD_DIR)/cpp-consumer.o $(HOST_STATIC),$(LDFLAGS) $(LDLIBS),$(BUILD_DIR)/cpp-consumer$(EXE))
	$(BUILD_DIR)/cpp-consumer$(EXE)
sanitize:
	$(MAKE) clean
	$(MAKE) CC='$(SAN_CC)' CXX='$(SAN_CXX)' CFLAGS='-O1 -g -fno-omit-frame-pointer -fsanitize=address,undefined -fno-sanitize-recover=all' LDFLAGS='-fsanitize=address,undefined' check
thread-sanitize:
	$(MAKE) clean
	$(MAKE) CC='$(SAN_CC)' CXX='$(SAN_CXX)' CFLAGS='-O1 -g -fsanitize=thread' LDFLAGS='-fsanitize=thread' $(BUILD_DIR)/infer-test$(EXE) $(BUILD_DIR)/concurrency$(EXE) $(BUILD_DIR)/tokenizer-test$(EXE) $(BUILD_DIR)/tokenizer-fixture.json $(BUILD_DIR)/weights-test$(EXE) $(BUILD_DIR)/inkling-test$(EXE) $(BUILD_DIR)/engine-test$(EXE) $(BUILD_DIR)/sampling-test$(EXE) $(BUILD_DIR)/lookup-test$(EXE) $(BUILD_DIR)/chat-test$(EXE) $(BUILD_DIR)/chat-fixture.json $(CLI)
	$(BUILD_DIR)/infer-test$(EXE) $(BUILD_DIR)/chat-fixture.json
	$(BUILD_DIR)/concurrency$(EXE)
	$(BUILD_DIR)/tokenizer-test$(EXE) $(BUILD_DIR)/tokenizer-fixture.json
	$(PYTHON) test/inkling_cases.py $(BUILD_DIR)/inkling-test$(EXE) ./$(CLI) $(BUILD_DIR)/weights-test$(EXE) $(BUILD_DIR)/infer-test$(EXE) $(BUILD_DIR)/chat-fixture.json
	$(BUILD_DIR)/engine-test$(EXE)
	$(BUILD_DIR)/sampling-test$(EXE)
	$(BUILD_DIR)/lookup-test$(EXE)
	$(BUILD_DIR)/chat-test$(EXE) $(BUILD_DIR)/chat-fixture.json
fuzz-smoke: | $(BUILD_DIR)
	$(SAN_CC) -O1 -g $(CPPFLAGS) -std=c11 -fsanitize=fuzzer,address,undefined -fno-sanitize-recover=all $(HOST) $(INFER) test/infer_mock.c fuzz/fuzz_infer.c $(THREAD_LIBS) -o $(BUILD_DIR)/fuzz-infer
	$(BUILD_DIR)/fuzz-infer -runs=$(FUZZ_RUNS) -max_total_time=$(FUZZ_SECONDS) -max_len=4096
	$(SAN_CC) -O1 -g $(CPPFLAGS) -std=c11 -fsanitize=fuzzer,address,undefined -fno-sanitize-recover=all $(HOST) fuzz/fuzz_stream.c -o $(BUILD_DIR)/fuzz-stream
	$(BUILD_DIR)/fuzz-stream -runs=$(FUZZ_RUNS) -max_total_time=$(FUZZ_SECONDS) -max_len=32768
	$(PYTHON) test/safetensors_cases.py --corpus $(BUILD_DIR)/fuzz-corpus
	$(SAN_CC) -O1 -g $(CPPFLAGS) -std=c11 -fsanitize=fuzzer,address,undefined -fno-sanitize-recover=all $(HOST) fuzz/fuzz_safetensors.c -o $(BUILD_DIR)/fuzz-safetensors
	$(BUILD_DIR)/fuzz-safetensors -runs=$(FUZZ_RUNS) -max_total_time=$(FUZZ_SECONDS) -max_len=262144 $(BUILD_DIR)/fuzz-corpus
	$(PYTHON) test/tokenizer_cases.py --corpus $(BUILD_DIR)/tokenizer-corpus
	$(SAN_CC) -O1 -g $(CPPFLAGS) -std=c11 -fsanitize=fuzzer,address,undefined -fno-sanitize-recover=all $(HOST) fuzz/fuzz_tokenizer.c -o $(BUILD_DIR)/fuzz-tokenizer
	$(BUILD_DIR)/fuzz-tokenizer -runs=$(FUZZ_RUNS) -max_total_time=$(FUZZ_SECONDS) -max_len=32768 $(BUILD_DIR)/tokenizer-corpus
format:
	tools/format.sh
lint: generated-check
	tools/format.sh --check
	$(PYTHON) tools/lint.py
	$(PYTHON) test/lint_boundaries.py
install: all infer
	mkdir -p $(DESTDIR)$(INCLUDEDIR) $(DESTDIR)$(LIBDIR) $(DESTDIR)$(PKGCONFIGDIR) $(DESTDIR)$(BINDIR)
	cp $(PUBLIC_HEADERS) $(DESTDIR)$(INCLUDEDIR)/
	cp $(INFER_STATIC) $(HOST_STATIC) $(SHARED_REAL) $(DESTDIR)$(LIBDIR)/
ifneq ($(IFN_WINDOWS),1)
	ln -sf $(notdir $(SHARED_REAL)) $(DESTDIR)$(LIBDIR)/$(notdir $(SHARED))
	ln -sf $(notdir $(SHARED_REAL)) $(DESTDIR)$(LIBDIR)/$(notdir $(SHARED_ABI))
else
	cp $(BUILD_DIR)/inkleberry-1.lib $(DESTDIR)$(LIBDIR)/
	cp $(SHARED_REAL) $(DESTDIR)$(BINDIR)/
endif
	cp $(CLI) $(DESTDIR)$(BINDIR)/
ifneq ($(IFN_WINDOWS),1)
	sed -e 's|@PREFIX@|$(PREFIX)|g' -e 's|@LIBDIR@|$(LIBDIR)|g' -e 's|@INCLUDEDIR@|$(INCLUDEDIR)|g' -e 's|@VERSION@|$(IFN_VERSION)|g' inkleberry.pc.in > $(DESTDIR)$(PKGCONFIGDIR)/inkleberry.pc
endif
.PHONY: uninstall uninstall-infer-backend
uninstall-infer-backend:
	rm -f $(addprefix $(DESTDIR)$(LIBDIR)/,libinkleberry_backend.so $(notdir $(PROVIDER)) $(BACKEND_IMPORT))
	rm -f $(DESTDIR)$(PKGCONFIGDIR)/inkleberry-infer.pc
uninstall: uninstall-infer-backend
	rm -f $(addprefix $(DESTDIR)$(INCLUDEDIR)/,$(notdir $(PUBLIC_HEADERS)))
	rm -f $(addprefix $(DESTDIR)$(LIBDIR)/,$(notdir $(HOST_STATIC) $(INFER_STATIC) $(GPU_STATIC) $(SHARED) $(SHARED_REAL) $(SHARED_ABI) $(HOST_IMPORT)))
	rm -f $(addprefix $(DESTDIR)$(BINDIR)/,$(notdir $(CLI) $(GENERATE) $(WEIGHTS_CLI)) $(INSTALLED_DLLS))
	rm -f $(addprefix $(DESTDIR)$(PKGCONFIGDIR)/,inkleberry.pc inkleberry-infer.pc inkleberry-static-infer.pc)
install-check: DESTDIR = $(CURDIR)/$(BUILD_DIR)/stage
install-check: PREFIX = /usr/local
install-check: install
	$(call ifn_program,$(CC),$(CFLAGS) $(WARNFLAGS) -I$(BUILD_DIR)/stage/usr/local/include,test/installed_consumer.c $(BUILD_DIR)/stage/usr/local/lib/$(notdir $(HOST_STATIC)),$(LDFLAGS) $(LDLIBS),$(BUILD_DIR)/installed-static$(EXE))
	$(BUILD_DIR)/installed-static$(EXE)
	$(call ifn_program,$(CC),$(CFLAGS) $(WARNFLAGS) -I$(BUILD_DIR)/stage/usr/local/include,test/installed_consumer.c $(RPATH_STAGE),$(INSTALLED_SHARED_LINK) $(LDFLAGS) $(LDLIBS),$(INSTALLED_SHARED_PROGRAM))
	$(INSTALLED_SHARED_PROGRAM)
$(BUILD_DIR)/%.cuda.o: cuda/%.cu $(CUDA_HEADERS) $(BUILD_DIR)/config
	$(NVCC) $(CUDAFLAGS) $(CUDA_TARGET_FLAGS) -c $< -o $@
$(BUILD_DIR)/abi-probe.o: test/cuda_abi_probe.cu src/cuda_abi.h cuda/cuda_abi.cu $(CUDA_HEADERS) $(BUILD_DIR)/config
	$(NVCC) $(CUDAFLAGS) $(CUDA_TARGET_FLAGS) -Isrc -c $< -o $@
$(BUILD_DIR)/abi-c.o: test/cuda_abi_c.c src/cuda_abi.h $(BUILD_DIR)/config
	$(call ifn_compile,$(CC),$(CPPFLAGS) $(CFLAGS) $(WARNFLAGS),$<,$@)
$(BUILD_DIR)/abi-probe$(EXE): $(BUILD_DIR)/abi-probe.o $(BUILD_DIR)/abi-c.o $(CUDA_OBJECTS)
	$(call ifn_program,$(CXX),,$^,$(CUDA_LIBS),$@)
cuda-abi-check: generated-check $(BUILD_DIR)/abi-probe$(EXE)
	LD_LIBRARY_PATH='$(CUDA_ROOT)/lib64' $(BUILD_DIR)/abi-probe$(EXE)
$(BUILD_DIR)/cuda-smoke.o: test/cuda_smoke.c src/cuda_abi.h $(BUILD_DIR)/config
	$(call ifn_compile,$(CC),$(CPPFLAGS) $(CFLAGS) $(WARNFLAGS),$<,$@)
$(BUILD_DIR)/cuda-smoke$(EXE): $(BUILD_DIR)/cuda-smoke.o $(BUILD_DIR)/cuda_abi.cuda.o $(CUDA_OBJECTS)
	$(call ifn_program,$(CXX),,$^,$(CUDA_LIBS),$@)
cuda-check: cuda-abi-check $(BUILD_DIR)/cuda-smoke$(EXE)
	$(PYTHON) tools/run_cuda_smoke.py '$(CUDA_RUNTIME_PATH)' $(BUILD_DIR)/cuda-smoke$(EXE)
clean:
	rm -rf $(BUILD_DIR) $(SHARED_REAL) $(SHARED_ABI) $(CLI) $(HOST_STATIC) $(SHARED) $(GPU_STATIC) $(INFER_STATIC) $(PROVIDER) $(CUDA_STATIC) $(WEIGHTS_CLI) $(GENERATE)
-include $(wildcard $(BUILD_DIR)/*.d)

# GPU backend is opt-in; the host library still builds without CUDA.
.PHONY: gpu
gpu: generated-check $(GPU_STATIC)
$(BUILD_DIR)/checked-weights.cuda.o: src/cuda/checked_weights.cu src/cuda/capability.h src/checked_cuda.h cuda/cuda_abi.cu $(CUDA_HEADERS) $(BUILD_DIR)/config
	$(NVCC) $(CUDAFLAGS) $(CUDA_TARGET_FLAGS) -c $< -o $@
$(GPU_STATIC): $(BUILD_DIR)/checked-prefill.cuda.o $(BUILD_DIR)/engine_bind.o $(BUILD_DIR)/engine_layout.o $(BUILD_DIR)/engine_forward.o $(BUILD_DIR)/engine_spec.o $(BUILD_DIR)/engine_adaptive.o $(BUILD_DIR)/engine_prefix.o $(BUILD_DIR)/engine_prefill.o $(BUILD_DIR)/engine_runtime.o $(BUILD_DIR)/gpu_cuda.o $(BUILD_DIR)/checked-weights.cuda.o $(filter-out $(BUILD_DIR)/kernels_prefill.cuda.o,$(CUDA_OBJECTS))
	rm -f $@
	$(call ifn_archive,$@,$^)
$(BUILD_DIR)/gpu-weight-compare.o: test/gpu_weights_compare.c include/ifn_gpu.h $(BUILD_DIR)/config
	$(call ifn_compile,$(CC),$(CPPFLAGS) $(CFLAGS) $(WARNFLAGS),$<,$@)
$(BUILD_DIR)/gpu-weight-main.o: test/gpu_weights_compare.c include/ifn_gpu.h $(BUILD_DIR)/config
	$(call ifn_compile,$(CC),$(CPPFLAGS) $(CFLAGS) $(WARNFLAGS) -DIFN_STANDALONE_WEIGHTS,$<,$@)
$(WEIGHTS_CLI): $(BUILD_DIR)/gpu-context.o $(BUILD_DIR)/gpu-weight-main.o $(GPU_STATIC) $(HOST_STATIC)
	$(call ifn_program,$(CXX),,$^,$(CUDA_LIBS) $(LDFLAGS),$@)

$(BUILD_DIR)/gpu-context.o: test/gpu_context.cu include/ifn_gpu.h $(BUILD_DIR)/config
	$(NVCC) $(CUDAFLAGS) $(CUDA_TARGET_FLAGS) -Iinclude -c $< -o $@
$(BUILD_DIR)/packing.o: test/packing.cu src/checked_cuda.h $(BUILD_DIR)/config
	$(NVCC) $(CUDAFLAGS) $(CUDA_TARGET_FLAGS) -Isrc -c $< -o $@
$(BUILD_DIR)/packing$(EXE): $(BUILD_DIR)/packing.o $(GPU_STATIC)
	$(call ifn_program,$(CXX),,$^,$(CUDA_LIBS),$@)
.PHONY: gpu-packing-check install-gpu
gpu-packing-check: generated-check $(BUILD_DIR)/packing$(EXE)
	$(PYTHON) tools/run_cuda_smoke.py '$(CUDA_RUNTIME_PATH)' $(BUILD_DIR)/packing$(EXE)
install-gpu: install gpu
ifneq ($(IFN_WINDOWS),1)
	sed -e 's|@PREFIX@|$(PREFIX)|g' -e 's|@LIBDIR@|$(LIBDIR)|g' -e 's|@INCLUDEDIR@|$(INCLUDEDIR)|g' -e 's|@VERSION@|$(IFN_VERSION)|g' -e 's|@CUDA_ROOT@|$(CUDA_ROOT)|g' inkleberry-static-infer.pc.in > $(DESTDIR)$(PKGCONFIGDIR)/inkleberry-static-infer.pc
endif
	cp $(GPU_STATIC) $(DESTDIR)$(LIBDIR)/

.PHONY: gpu-install-check
gpu-install-check: DESTDIR = $(CURDIR)/$(BUILD_DIR)/gpu-stage
gpu-install-check: PREFIX = /usr/local
gpu-install-check: install-gpu
	$(call ifn_compile,$(CC),$(CFLAGS) $(WARNFLAGS) -I$(BUILD_DIR)/gpu-stage/usr/local/include,test/installed_gpu_consumer.c,$(BUILD_DIR)/installed-gpu.o)
	$(call ifn_program,$(CXX),,$(BUILD_DIR)/installed-gpu.o $(BUILD_DIR)/gpu-stage/usr/local/lib/$(notdir $(GPU_STATIC)) $(BUILD_DIR)/gpu-stage/usr/local/lib/$(notdir $(HOST_STATIC)),$(CUDA_LIBS),$(BUILD_DIR)/installed-gpu$(EXE))
	LD_LIBRARY_PATH='$(CUDA_ROOT)/lib64' $(BUILD_DIR)/installed-gpu
	$(call ifn_compile,$(CXX),$(CXX_WARNFLAGS) -I$(BUILD_DIR)/gpu-stage/usr/local/include $(CXX_LANGUAGE),test/installed_gpu_consumer.c,$(BUILD_DIR)/installed-gpu-cpp.o)
	$(call ifn_program,$(CXX),,$(BUILD_DIR)/installed-gpu-cpp.o $(BUILD_DIR)/gpu-stage/usr/local/lib/$(notdir $(GPU_STATIC)) $(BUILD_DIR)/gpu-stage/usr/local/lib/$(notdir $(HOST_STATIC)),$(CUDA_LIBS),$(BUILD_DIR)/installed-gpu-cpp$(EXE))
	LD_LIBRARY_PATH='$(CUDA_ROOT)/lib64' $(BUILD_DIR)/installed-gpu-cpp
$(BUILD_DIR)/generate.o: cli/generate.c include/ifn_engine.h $(BUILD_DIR)/config
	$(call ifn_compile,$(CC),$(CPPFLAGS) $(CFLAGS) $(WARNFLAGS) $(PROGRAM_COMPILE),$<,$@)
$(BUILD_DIR)/ar-compare.o: test/ar_compare.c include/ifn_engine.h $(BUILD_DIR)/config
	$(call ifn_compile,$(CC),$(CPPFLAGS) $(CFLAGS) $(WARNFLAGS),$<,$@)

$(BUILD_DIR)/checked-prefill.cuda.o: src/cuda/checked_prefill.cu src/checked_prefill.h cuda/kernels_prefill.cu $(CUDA_HEADERS) $(BUILD_DIR)/config
	$(NVCC) $(CUDAFLAGS) $(CUDA_TARGET_FLAGS) -c $< -o $@

$(BUILD_DIR)/engine-test$(EXE): test/engine.c test/mock_gpu.h $(BUILD_DIR)/engine_bind.o $(BUILD_DIR)/engine_layout.o $(BUILD_DIR)/engine_runtime.o $(BUILD_DIR)/engine_adaptive.o $(BUILD_DIR)/engine_prefix.o $(HOST_STATIC) $(BUILD_DIR)/config
	$(call ifn_program,$(CC),$(CPPFLAGS) $(CFLAGS) $(WARNFLAGS),test/engine.c $(BUILD_DIR)/engine_bind.o $(BUILD_DIR)/engine_layout.o $(BUILD_DIR)/engine_runtime.o $(BUILD_DIR)/engine_adaptive.o $(BUILD_DIR)/engine_prefix.o $(HOST_STATIC),$(THREAD_LIBS) $(LDFLAGS) $(LDLIBS),$@)

$(BUILD_DIR)/session-context.o: test/session_context.cu include/ifn_engine.h $(BUILD_DIR)/config
	$(NVCC) $(CUDAFLAGS) $(CUDA_TARGET_FLAGS) -Iinclude -c $< -o $@

$(BUILD_DIR)/check-ar-cpp.o: test/installed_gpu_consumer.c include/ifn_engine.h $(BUILD_DIR)/config
	$(call ifn_compile,$(CXX),$(CXX_WARNFLAGS) -Iinclude $(CXX_LANGUAGE),$<,$@)
$(BUILD_DIR)/decode-compare.o: test/decode_compare.c test/ar_compare.c include/ifn_engine.h $(BUILD_DIR)/config
	$(call ifn_compile,$(CC),$(CPPFLAGS) $(CFLAGS) $(WARNFLAGS),$<,$@)
$(BUILD_DIR)/sampling-kernels.o: test/sampling_kernels.cu cuda/kernels_gemv.cu $(CUDA_HEADERS) $(BUILD_DIR)/config
	$(NVCC) $(CUDAFLAGS) $(CUDA_TARGET_FLAGS) -c $< -o $@
$(BUILD_DIR)/sampling-kernels$(EXE): $(BUILD_DIR)/sampling-kernels.o
	$(call ifn_program,$(CXX),,$^,$(CUDA_LIBS),$@)
.PHONY: sampling-kernel-check
sampling-kernel-check: generated-check $(BUILD_DIR)/sampling-kernels$(EXE)
	$(PYTHON) tools/run_cuda_smoke.py '$(CUDA_RUNTIME_PATH)' $(BUILD_DIR)/sampling-kernels$(EXE)

$(BUILD_DIR)/sampling-test$(EXE): test/sampling.c $(HOST_STATIC) $(BUILD_DIR)/config
	$(call ifn_program,$(CC),$(CPPFLAGS) $(CFLAGS) $(WARNFLAGS),$< $(HOST_STATIC),$(LDFLAGS) $(LDLIBS),$@)
$(BUILD_DIR)/decode-recovery.o: test/decode_recovery.c test/decode_compare.c test/ar_compare.c include/ifn_engine.h $(BUILD_DIR)/config
	$(call ifn_compile,$(CC),$(CPPFLAGS) $(CFLAGS) $(WARNFLAGS),$<,$@)
$(BUILD_DIR)/decode-recovery$(EXE): $(BUILD_DIR)/decode-recovery.o $(BUILD_DIR)/session-context.o $(GPU_STATIC) $(HOST_STATIC)
	$(call ifn_program,$(CXX),,$^,$(CUDA_LIBS) $(LDFLAGS),$@)
$(BUILD_DIR)/adaptive-compare.o: test/adaptive_compare.c test/decode_compare.c test/ar_compare.c include/ifn_engine.h $(BUILD_DIR)/config
	$(call ifn_compile,$(CC),$(CPPFLAGS) $(CFLAGS) $(WARNFLAGS),$<,$@)

$(BUILD_DIR)/lookup-test$(EXE): test/lookup.c $(HOST_STATIC) $(BUILD_DIR)/config
	$(call ifn_program,$(CC),$(CPPFLAGS) $(CFLAGS) $(WARNFLAGS),$< $(HOST_STATIC),$(LDFLAGS) $(LDLIBS),$@)
$(BUILD_DIR)/adaptive-graph.o: test/adaptive_graph.cu src/checked_cuda.h $(BUILD_DIR)/config
	$(NVCC) $(CUDAFLAGS) $(CUDA_TARGET_FLAGS) -Isrc -c $< -o $@
$(BUILD_DIR)/adaptive-graph$(EXE): $(BUILD_DIR)/adaptive-graph.o $(GPU_STATIC) $(HOST_STATIC)
	$(call ifn_program,$(CXX),,$^,$(CUDA_LIBS),$@)

$(BUILD_DIR)/adaptive-recovery.o: test/decode_recovery.c test/decode_compare.c test/ar_compare.c include/ifn_engine.h $(BUILD_DIR)/config
	$(call ifn_compile,$(CC),$(CPPFLAGS) $(CFLAGS) $(WARNFLAGS) -DIFN_TEST_ADAPTIVE=1,$<,$@)
$(BUILD_DIR)/adaptive-recovery$(EXE): $(BUILD_DIR)/adaptive-recovery.o $(BUILD_DIR)/session-context.o $(GPU_STATIC) $(HOST_STATIC)
	$(call ifn_program,$(CXX),,$^,$(CUDA_LIBS) $(LDFLAGS),$@)

$(BUILD_DIR)/prefix-compare.o: test/prefix_compare.c test/decode_compare.c test/ar_compare.c include/ifn_engine.h $(BUILD_DIR)/config
	$(call ifn_compile,$(CC),$(CPPFLAGS) $(CFLAGS) $(WARNFLAGS),$<,$@)
$(BUILD_DIR)/prefix-recovery.o: test/prefix_recovery.c test/prefix_compare.c test/decode_compare.c test/ar_compare.c include/ifn_engine.h $(BUILD_DIR)/config
	$(call ifn_compile,$(CC),$(CPPFLAGS) $(CFLAGS) $(WARNFLAGS),$<,$@)
$(BUILD_DIR)/prefix-recovery$(EXE): $(BUILD_DIR)/prefix-recovery.o $(BUILD_DIR)/session-context.o $(GPU_STATIC) $(HOST_STATIC)
	$(call ifn_program,$(CXX),,$^,$(CUDA_LIBS) $(LDFLAGS),$@)

$(BUILD_DIR)/chat-fixture.json: test/chat_fixtures.py test/tokenizer_fixtures.py | $(BUILD_DIR)
	$(PYTHON) test/chat_fixtures.py $@
$(BUILD_DIR)/chat-test$(EXE): test/chat.c include/ifn_chat.h $(HOST_STATIC) $(BUILD_DIR)/config
	$(call ifn_program,$(CC),$(CPPFLAGS) $(CFLAGS) $(WARNFLAGS),$< $(HOST_STATIC),$(LDFLAGS) $(LDLIBS),$@)
$(BUILD_DIR)/stream-driver$(EXE): test/stream_driver.c include/ifn_chat.h $(HOST_STATIC) $(BUILD_DIR)/config
	$(call ifn_program,$(CC),$(CPPFLAGS) $(CFLAGS) $(WARNFLAGS),$< $(HOST_STATIC),$(LDFLAGS) $(LDLIBS),$@)
$(BUILD_DIR)/chat-native.o: test/chat_native.c test/decode_compare.c test/ar_compare.c include/ifn_chat.h $(BUILD_DIR)/config
	$(call ifn_compile,$(CC),$(CPPFLAGS) $(CFLAGS) $(WARNFLAGS),$<,$@)
$(BUILD_DIR)/chat-native$(EXE): $(BUILD_DIR)/chat-native.o $(BUILD_DIR)/session-context.o $(GPU_STATIC) $(HOST_STATIC)
	$(call ifn_program,$(CXX),,$^,$(CUDA_LIBS) $(LDFLAGS),$@)

# Serving ABI: separate from libinkleberry's CPU-only parser/tokenizer contract.
INFER_OBJ = $(INFER:src/%.c=$(BUILD_DIR)/%.o)
.PHONY: infer infer-check infer-shared
infer: $(INFER_STATIC)
$(INFER_STATIC): $(INFER_OBJ)
	$(call ifn_archive,$@,$^)
$(BUILD_DIR)/infer-test$(EXE): test/infer.c test/infer_mock.c $(INFER_STATIC) $(HOST_STATIC) $(BUILD_DIR)/config
	$(call ifn_program,$(CC),$(CPPFLAGS) $(CFLAGS) $(WARNFLAGS),test/infer.c test/infer_mock.c $(INFER_STATIC) $(HOST_STATIC),$(THREAD_LIBS) $(LDFLAGS) $(LDLIBS),$@)
infer-check: $(BUILD_DIR)/infer-test$(EXE) $(BUILD_DIR)/chat-fixture.json
	$(BUILD_DIR)/infer-test$(EXE) $(BUILD_DIR)/chat-fixture.json
check: infer-check
BACKEND_C = $(INFER) $(GPU_HOST)
BACKEND_OBJ = $(BACKEND_C:src/%.c=$(BUILD_DIR)/backend-%.o)
BACKEND_CUDA = $(addprefix $(BUILD_DIR)/backend-,$(addsuffix .cuda.o,kernels_gemv kernels_inkling kernels_dspark)) $(BUILD_DIR)/backend-checked-weights.cuda.o $(BUILD_DIR)/backend-checked-prefill.cuda.o
$(BUILD_DIR)/backend-%.o: src/%.c $(IFN_HEADER_DEPS) $(BUILD_DIR)/config
	$(call ifn_compile,$(CC),$(CPPFLAGS) $(CFLAGS) $(WARNFLAGS) $(PIC_FLAGS) $(HIDDEN_FLAGS) $(BACKEND_DEFS) $(DEPFLAGS),$<,$@)
$(BUILD_DIR)/backend-%.cuda.o: cuda/%.cu $(CUDA_HEADERS) $(BUILD_DIR)/config
	$(NVCC) $(CUDAFLAGS) $(CUDA_TARGET_FLAGS) $(CUDA_PIC_FLAGS) -c $< -o $@
$(BUILD_DIR)/backend-checked-weights.cuda.o: src/cuda/checked_weights.cu src/checked_cuda.h cuda/cuda_abi.cu $(CUDA_HEADERS) $(BUILD_DIR)/config
	$(NVCC) $(CUDAFLAGS) $(CUDA_TARGET_FLAGS) $(CUDA_PIC_FLAGS) -c $< -o $@
$(BUILD_DIR)/backend-checked-prefill.cuda.o: src/cuda/checked_prefill.cu src/checked_prefill.h cuda/kernels_prefill.cu $(CUDA_HEADERS) $(BUILD_DIR)/config
	$(NVCC) $(CUDAFLAGS) $(CUDA_TARGET_FLAGS) $(CUDA_PIC_FLAGS) -c $< -o $@
# Reuse the existing PIC host/kernel objects; static SDK code generation is unchanged.
$(GENERATE): $(BUILD_DIR)/generate.o $(GPU_HOST:src/%.c=$(BUILD_DIR)/backend-%.o) $(BACKEND_CUDA) $(PIC)
	$(call ifn_program,$(CXX),,$^,$(CUDA_LIBS) $(LDFLAGS) $(PROGRAM_HARDEN),$@)
$(PROVIDER): $(BACKEND_OBJ) $(BACKEND_CUDA) $(PROVIDER_HOST) src/infer.exports src/infer.def
	$(call ifn_shared,$(CXX),,$(BACKEND_OBJ) $(BACKEND_CUDA) $(PROVIDER_HOST),$(PROVIDER_FLAGS) $(CUDA_LIBS) $(LDFLAGS),$@)
infer-shared: generated-check $(PROVIDER)
$(BUILD_DIR)/infer-native.o: test/infer_native.c include/ifn_infer.h $(BUILD_DIR)/config
	$(call ifn_compile,$(CC),$(CPPFLAGS) $(CFLAGS) $(WARNFLAGS),$<,$@)
$(BUILD_DIR)/infer-native$(EXE): $(BUILD_DIR)/infer-native.o $(INFER_STATIC) $(GPU_STATIC) $(HOST_STATIC)
	$(call ifn_program,$(CXX),,$^,$(CUDA_LIBS) $(LDFLAGS),$@)

$(BUILD_DIR)/infer-consumer$(EXE): test/infer_consumer.c $(BUILD_DIR)/stage/usr/local/include/ifn_infer.h $(INFER_STATIC) $(HOST_STATIC) $(BUILD_DIR)/config
	$(call ifn_program,$(CC),$(CFLAGS) $(WARNFLAGS) -I$(BUILD_DIR)/stage/usr/local/include -Iinclude -Isrc,test/infer_consumer.c test/infer_mock.c $(BUILD_DIR)/stage/usr/local/lib/$(notdir $(INFER_STATIC)) $(BUILD_DIR)/stage/usr/local/lib/$(notdir $(HOST_STATIC)),$(THREAD_LIBS) $(LDFLAGS) $(LDLIBS),$@)
$(BUILD_DIR)/stage/usr/local/include/ifn_infer.h: install
$(BUILD_DIR)/infer-mock.pic.o: test/infer_mock.c $(BUILD_DIR)/config
	$(call ifn_compile,$(CC),$(CPPFLAGS) $(CFLAGS) $(WARNFLAGS) $(PIC_FLAGS) $(DEPFLAGS),$<,$@)
$(MOCK_PROVIDER): $(INFER:src/%.c=$(BUILD_DIR)/%.pic.o) $(PROVIDER_HOST) $(BUILD_DIR)/infer-mock.pic.o
	$(call ifn_shared,$(CC),,$^,$(PLAIN_SHARED_FLAGS) $(MOCK_EXPORT_FLAGS) $(THREAD_LIBS) $(LDFLAGS) $(LDLIBS),$@)
.PHONY: infer-consumer-check
infer-consumer-check: $(BUILD_DIR)/infer-consumer$(EXE) $(MOCK_PROVIDER)
	$(BUILD_DIR)/infer-consumer$(EXE)
	$(call ifn_compile,$(CXX),$(CXX_WARNFLAGS) -I$(BUILD_DIR)/stage/usr/local/include $(CXX_LANGUAGE),test/infer_consumer.c,$(BUILD_DIR)/infer-consumer-cpp.o)
	$(call ifn_program,$(CC),,$(BUILD_DIR)/infer-consumer-cpp.o $(BUILD_DIR)/infer-mock.pic.o $(INFER_STATIC) $(HOST_STATIC),$(THREAD_LIBS) $(LDFLAGS) $(LDLIBS),$(BUILD_DIR)/infer-consumer-cpp$(EXE))
	$(BUILD_DIR)/infer-consumer-cpp$(EXE)
	$(PYTHON) test/infer_ffi.py ./$(MOCK_PROVIDER)
install-check: infer-consumer-check
$(BUILD_DIR)/infer-native-shared$(EXE): $(BUILD_DIR)/infer-native.o $(PROVIDER)
	$(call ifn_program,$(CC),,$^ $(RPATH_ROOT),$(THREAD_LIBS) $(LDFLAGS),$@)
.PHONY: install-infer-backend
install-infer-backend: install-gpu infer-shared
	cp $(PROVIDER) $(DESTDIR)$(LIBDIR)/
ifneq ($(IFN_WINDOWS),1)
	ln -sf $(PROVIDER) $(DESTDIR)$(LIBDIR)/libinkleberry_backend.so
else
	cp $(BUILD_DIR)/inkleberry_backend-1.lib $(DESTDIR)$(LIBDIR)/
endif
ifneq ($(IFN_WINDOWS),1)
	sed -e 's|@PREFIX@|$(PREFIX)|g' -e 's|@LIBDIR@|$(LIBDIR)|g' -e 's|@INCLUDEDIR@|$(INCLUDEDIR)|g' -e 's|@VERSION@|$(IFN_VERSION)|g' inkleberry-infer.pc.in > $(DESTDIR)$(PKGCONFIGDIR)/inkleberry-infer.pc
endif
.PHONY: infer-provider-check
infer-provider-check: infer-shared
	$(PYTHON) test/infer_exports.py ./$(PROVIDER)
	LD_LIBRARY_PATH='$(CUDA_ROOT)/lib64' $(PYTHON) test/infer_ffi.py ./$(PROVIDER)

.PHONY: cuda-abi-negative-check reference-free-check
cuda-abi-negative-check: cuda-abi-check
	$(PYTHON) test/cuda_abi_negative.py --cc '$(CC)' --cxx '$(CXX)' --nvcc '$(NVCC)' --cuda-root '$(CUDA_ROOT)' --cuda-flags '$(CUDAFLAGS) $(CUDA_TARGET_FLAGS)' --cuda-objects '$(CUDA_OBJECTS)'
reference-free-check:
	$(PYTHON) test/reference_free.py --cc '$(CC)' --cxx '$(CXX)'


.PHONY: amalgamation amalgamation-check amalgamation-test
amalgamation:
	$(PYTHON) tools/amalgamate.py
amalgamation-check:
	$(PYTHON) tools/amalgamate.py --check
	$(PYTHON) test/amalgamate_generator.py
$(BUILD_DIR)/amalgam.o: dist/ifn.c dist/ifn.h $(BUILD_DIR)/config
	$(call ifn_compile,$(CC),$(CFLAGS) $(WARNFLAGS) -Idist,$<,$@)
$(BUILD_DIR)/amalgam-gpu.o: dist/ifn.c dist/ifn.h $(BUILD_DIR)/config
	$(call ifn_compile,$(CC),$(CFLAGS) $(WARNFLAGS) -Idist -DIFN_ENABLE_CUDA,$<,$@)
$(BUILD_DIR)/amalgam-%$(EXE): test/%.c $(BUILD_DIR)/amalgam.o $(BUILD_DIR)/config
	$(call ifn_program,$(CC),$(CPPFLAGS) $(CFLAGS) $(WARNFLAGS),$< $(BUILD_DIR)/amalgam.o,$(THREAD_LIBS) $(LDFLAGS) $(LDLIBS),$@)
$(BUILD_DIR)/embedding-consumer$(EXE): examples/inspect.c $(BUILD_DIR)/amalgam.o $(BUILD_DIR)/config
	$(call ifn_program,$(CC),$(CFLAGS) $(WARNFLAGS) -Idist,$< $(BUILD_DIR)/amalgam.o,$(LDFLAGS) $(LDLIBS),$@)
$(BUILD_DIR)/embedding-consumer-cpp.o: examples/inspect.c dist/ifn.h $(BUILD_DIR)/config
	$(call ifn_compile,$(CXX),$(CXX_WARNFLAGS) -Idist $(CXX_LANGUAGE),$<,$@)
$(BUILD_DIR)/mmap-test$(EXE): test/mmap.c $(HOST_STATIC) $(BUILD_DIR)/config
	$(call ifn_program,$(CC),$(CPPFLAGS) $(CFLAGS) $(WARNFLAGS),$< $(HOST_STATIC),$(LDFLAGS) $(LDLIBS),$@)
amalgamation-test: amalgamation-check $(BUILD_DIR)/embedding-generate.o $(BUILD_DIR)/amalgam-unit$(EXE) $(BUILD_DIR)/amalgam-tokenizer$(EXE) $(BUILD_DIR)/amalgam-chat$(EXE) $(BUILD_DIR)/amalgam-stream_driver$(EXE) $(BUILD_DIR)/amalgam-concurrency$(EXE) $(BUILD_DIR)/amalgam-mmap$(EXE) $(BUILD_DIR)/embedding-consumer$(EXE) $(BUILD_DIR)/embedding-consumer-cpp.o $(BUILD_DIR)/tokenizer-fixture.json $(BUILD_DIR)/chat-fixture.json $(HOST_STATIC)
	$(BUILD_DIR)/amalgam-unit$(EXE)
	$(BUILD_DIR)/amalgam-tokenizer$(EXE) $(BUILD_DIR)/tokenizer-fixture.json
	$(BUILD_DIR)/amalgam-chat$(EXE) $(BUILD_DIR)/chat-fixture.json
	$(PYTHON) test/stream_cases.py $(BUILD_DIR)/amalgam-stream_driver$(EXE)
	$(BUILD_DIR)/amalgam-concurrency$(EXE)
	$(BUILD_DIR)/amalgam-mmap$(EXE)
	$(PYTHON) test/amalgamation.py $(HOST_STATIC) $(BUILD_DIR)/amalgam.o
check: amalgamation-test mmap-check
.PHONY: mmap-check
mmap-check: $(BUILD_DIR)/mmap-test$(EXE)
	$(BUILD_DIR)/mmap-test$(EXE)
	$(BUILD_DIR)/mmap-test$(EXE) --large
lint: amalgamation-check
# CUDA-only archive supports the single C host without duplicate host objects.
$(CUDA_STATIC): $(BUILD_DIR)/checked-prefill.cuda.o $(BUILD_DIR)/checked-weights.cuda.o $(filter-out $(BUILD_DIR)/kernels_prefill.cuda.o,$(CUDA_OBJECTS))
	$(call ifn_archive,$@,$^)
$(BUILD_DIR)/infer-native-amalgam$(EXE): $(BUILD_DIR)/infer-native.o $(BUILD_DIR)/amalgam-gpu.o $(CUDA_STATIC)
	$(call ifn_program,$(CXX),,$^,$(CUDA_LIBS) $(LDFLAGS),$@)

.PHONY: source-package embedding-package runtime-package runtime-gpu-package packages release-check
VERSION ?= $(IFN_VERSION)
source-package: generated-check amalgamation-check
	$(PYTHON) tools/release.py source --version '$(VERSION)'
embedding-package: amalgamation-check
	$(PYTHON) tools/release.py embedding --version '$(VERSION)'
runtime-package: all
	$(PYTHON) tools/release.py runtime --build-root '$(IFN_PACKAGE_BUILD_ROOT)' --version '$(VERSION)'
runtime-gpu-package: all $(GENERATE) infer-shared $(CUDA_STATIC) $(GPU_STATIC) $(INFER_STATIC)
	$(PYTHON) tools/release.py runtime --gpu --build-root '$(IFN_PACKAGE_BUILD_ROOT)' --cuda-profile '$(BUILD_DIR)/cuda-profile.json' --version '$(VERSION)'
packages: source-package embedding-package runtime-package
release-check: all amalgamation-check
	$(PYTHON) test/package_policy.py
	$(PYTHON) test/release.py --cc '$(CC)' --cxx '$(CXX)'

$(BUILD_DIR)/embedding-generate.o: examples/generate.c dist/ifn.h $(BUILD_DIR)/config
	$(call ifn_compile,$(CC),$(CFLAGS) $(WARNFLAGS) -Idist,$<,$@)
$(BUILD_DIR)/embedding-generate$(EXE): $(BUILD_DIR)/embedding-generate.o $(BUILD_DIR)/amalgam-gpu.o $(CUDA_STATIC)
	$(call ifn_program,$(CXX),,$^,$(CUDA_LIBS) $(LDFLAGS),$@)

.PHONY: native-release-check
native-release-check: generated-check all amalgamation-check
	$(PYTHON) test/native_release.py --model '$(MODEL_DIR)' --cuda-root '$(CUDA_ROOT)' --cc '$(CC)' --cxx '$(CXX)'

.PHONY: amalgamation-fuzz-smoke
amalgamation-fuzz-smoke: | $(BUILD_DIR)
	$(SAN_CC) -O1 -g $(CPPFLAGS) -std=c11 -fsanitize=fuzzer,address,undefined -fno-sanitize-recover=all dist/ifn.c fuzz/fuzz_stream.c -o $(BUILD_DIR)/fuzz-amalgam-stream
	$(BUILD_DIR)/fuzz-amalgam-stream -runs=$(FUZZ_RUNS) -max_total_time=$(FUZZ_SECONDS) -max_len=32768
	$(PYTHON) test/safetensors_cases.py --corpus $(BUILD_DIR)/fuzz-corpus
	$(SAN_CC) -O1 -g $(CPPFLAGS) -std=c11 -fsanitize=fuzzer,address,undefined -fno-sanitize-recover=all dist/ifn.c fuzz/fuzz_safetensors.c -o $(BUILD_DIR)/fuzz-amalgam-safetensors
	$(BUILD_DIR)/fuzz-amalgam-safetensors -runs=$(FUZZ_RUNS) -max_total_time=$(FUZZ_SECONDS) -max_len=262144 $(BUILD_DIR)/fuzz-corpus
	$(PYTHON) test/tokenizer_cases.py --corpus $(BUILD_DIR)/tokenizer-corpus
	$(SAN_CC) -O1 -g $(CPPFLAGS) -std=c11 -fsanitize=fuzzer,address,undefined -fno-sanitize-recover=all dist/ifn.c fuzz/fuzz_tokenizer.c -o $(BUILD_DIR)/fuzz-amalgam-tokenizer
	$(BUILD_DIR)/fuzz-amalgam-tokenizer -runs=$(FUZZ_RUNS) -max_total_time=$(FUZZ_SECONDS) -max_len=32768 $(BUILD_DIR)/tokenizer-corpus

.PHONY: native-namespace-check
native-namespace-check: $(CUDA_STATIC) $(PROVIDER)
	$(PYTHON) test/native_symbols.py $(CUDA_STATIC) $(PROVIDER)

.PHONY: host-error-check
host-error-check: | $(BUILD_DIR)
	$(call ifn_program,$(CXX),$(CXX20_WARNFLAGS) -Itest/cuda-stubs -Icuda,test/host_cuda_errors.cpp,$(THREAD_LIBS),$(BUILD_DIR)/host-cuda-errors$(EXE))
	$(BUILD_DIR)/host-cuda-errors$(EXE)
check: host-error-check

$(BUILD_DIR)/hash-test$(EXE): test/hash.c $(HOST_STATIC) $(BUILD_DIR)/config
	$(call ifn_program,$(CC),$(CPPFLAGS) $(CFLAGS) $(WARNFLAGS),$< $(HOST_STATIC),$(LDFLAGS) $(LDLIBS),$@)
.PHONY: identity-check
identity-check: $(BUILD_DIR)/hash-test$(EXE) $(BUILD_DIR)/infer-test$(EXE) $(BUILD_DIR)/chat-fixture.json
	$(PYTHON) test/hashes.py $(BUILD_DIR)/hash-test$(EXE)
	$(PYTHON) test/identity_admission.py $(BUILD_DIR)/infer-test$(EXE) $(BUILD_DIR)/chat-fixture.json
check: identity-check
$(BUILD_DIR)/native-benchmark.o: test/native_benchmark.c $(PUBLIC_HEADERS) $(BUILD_DIR)/config
	$(call ifn_compile,$(CC),$(CPPFLAGS) $(CFLAGS) $(WARNFLAGS),$<,$@)
$(BUILD_DIR)/native-benchmark$(EXE): $(BUILD_DIR)/native-benchmark.o $(GPU_STATIC) $(HOST_STATIC)
	$(call ifn_program,$(CXX),,$^,$(CUDA_LIBS) $(LDFLAGS),$@)
$(BUILD_DIR)/verify.o: examples/verify.c $(PUBLIC_HEADERS) $(BUILD_DIR)/config
	$(call ifn_compile,$(CC),$(CPPFLAGS) $(CFLAGS) $(WARNFLAGS),$<,$@)
$(BUILD_DIR)/verify$(EXE): $(BUILD_DIR)/verify.o $(PROVIDER) $(HOST_STATIC)
	$(call ifn_program,$(CC),,$^ $(RPATH_ROOT),$(THREAD_LIBS) $(LDFLAGS),$@)

.PHONY: public-example-check
public-example-check:
	$(CC) $(WARNFLAGS) -Iinclude $(SYNTAX_ONLY) examples/generate.c examples/verify.c
	$(CC) $(WARNFLAGS) -Idist $(SYNTAX_ONLY) examples/generate.c examples/verify.c
check: public-example-check

$(BUILD_DIR)/abi-contract$(EXE): test/abi_contract.c $(PUBLIC_HEADERS) $(BUILD_DIR)/config
	$(call ifn_program,$(CC),$(C_STD) -Iinclude,$<,,$@)
.PHONY: abi-contract-check install-contract-check harden-check performance-gate-check
abi-contract-check: $(BUILD_DIR)/abi-contract$(EXE)
	$(BUILD_DIR)/abi-contract$(EXE) > $(BUILD_DIR)/abi-current.json
	$(PYTHON) -c 'import json; assert json.load(open("$(BUILD_DIR)/abi-current.json")) == json.load(open("test/fixtures/abi-1.0.json")), "ABI 1.0 layout/value regression"'
install-contract-check: all infer
ifeq ($(IFN_WINDOWS),1)
	$(PYTHON) test/windows_install.py
else
	$(PYTHON) test/install_contract.py
endif
harden-check: all
	$(PYTHON) tools/check_hardening.py $(CLI) $(SHARED)
performance-gate-check:
	$(PYTHON) test/performance_gate.py
check: abi-contract-check performance-gate-check release-gate-check
$(BUILD_DIR)/facade-benchmark.o: test/native_benchmark.c $(PUBLIC_HEADERS) $(BUILD_DIR)/config
	$(call ifn_compile,$(CC),$(CPPFLAGS) $(CFLAGS) $(WARNFLAGS) -DIFN_BENCH_FACADE,$<,$@)
$(BUILD_DIR)/facade-benchmark$(EXE): $(BUILD_DIR)/facade-benchmark.o $(INFER_STATIC) $(GPU_STATIC) $(HOST_STATIC)
	$(call ifn_program,$(CXX),,$^,$(CUDA_LIBS) $(LDFLAGS),$@)

.PHONY: release-gate-check harden-regression-check
release-gate-check:
	$(PYTHON) test/release_gate.py
harden-regression-check:
	$(PYTHON) test/hardening.py --cc '$(CC)' --cxx '$(CXX)'

$(BUILD_DIR)/host-clock$(EXE): test/host_clock.c $(HOST_STATIC) $(BUILD_DIR)/config
	$(call ifn_program,$(CC),$(CPPFLAGS) $(CFLAGS) $(WARNFLAGS),$< $(HOST_STATIC),$(LDFLAGS) $(LDLIBS),$@)
platform-check: $(BUILD_DIR)/host-clock$(EXE)
check: platform-check
$(BUILD_DIR)/platform-map$(EXE): test/platform_map.c $(HOST_STATIC) $(BUILD_DIR)/config
	$(call ifn_program,$(CC),$(CPPFLAGS) $(CFLAGS) $(WARNFLAGS),$< $(HOST_STATIC),$(LDFLAGS) $(LDLIBS),$@)
platform-check: $(BUILD_DIR)/platform-map$(EXE)
	$(PYTHON) test/host_clock.py $(BUILD_DIR)/host-clock$(EXE)
	$(BUILD_DIR)/platform-map$(EXE)
.PHONY: windows-sdk-check
windows-sdk-check: all infer
	$(PYTHON) test/windows_sdk.py
ifeq ($(IFN_WINDOWS),1)
windows-pe-check: all $(MOCK_PROVIDER)
	$(PYTHON) test/windows_pe.py $(CLI) $(SHARED) $(MOCK_PROVIDER)
check: windows-pe-check
endif

ifeq ($(IFN_WINDOWS),1)
.PHONY: windows-cuda-build-check
windows-cuda-build-check: gpu infer-shared $(GENERATE) $(CUDA_STATIC) $(BUILD_DIR)/abi-probe$(EXE)
	$(PYTHON) test/infer_exports.py $(PROVIDER)
	$(PYTHON) test/windows_pe.py $(GENERATE) $(PROVIDER)
	$(PYTHON) test/infer_ffi.py $(PROVIDER)
	PATH="$$(cygpath -u '$(CUDA_ROOT)/bin/x64'):$$(cygpath -u '$(CUDA_ROOT)/bin'):$$PATH" $(BUILD_DIR)/abi-probe$(EXE)
endif

.PHONY: gpu-build
gpu-build: all gpu infer-shared $(GENERATE) $(CUDA_STATIC) $(INFER_STATIC)
$(BUILD_DIR)/cuda-profile.json: tools/cuda_profile.py $(BUILD_DIR)/config
	$(PYTHON) tools/cuda_profile.py --arch '$(CUDA_ARCH)' $(if $(filter 1,$(EXPERIMENTAL_AMPERE)),--experimental,) --out $@
$(GPU_STATIC) $(CUDA_STATIC) $(PROVIDER): | $(BUILD_DIR)/cuda-profile.json

$(BUILD_DIR)/cuda-policy-120$(EXE): test/cuda_policy.cpp src/cuda/geometry_contract.hpp src/inkling_geometry.h test/cuda-stubs/cuda_runtime.h src/cuda/launch_policy.hpp src/cuda/resource_policy.hpp src/cuda/prefill_resources.hpp src/cuda/target_profile.h $(BUILD_DIR)/config
	$(call ifn_program,$(CXX),$(CXX20_WARNFLAGS) $(filter -fsanitize% -fno-sanitize% -fno-omit-frame-pointer -g,$(CFLAGS)) -Itest/cuda-stubs -Icuda -DIFN_CUDA_TARGET_SM=120,$<,$(THREAD_LIBS) $(LDFLAGS),$@)
$(BUILD_DIR)/cuda-policy-121$(EXE): test/cuda_policy.cpp src/cuda/geometry_contract.hpp src/inkling_geometry.h test/cuda-stubs/cuda_runtime.h src/cuda/launch_policy.hpp src/cuda/resource_policy.hpp src/cuda/prefill_resources.hpp src/cuda/target_profile.h $(BUILD_DIR)/config
	$(call ifn_program,$(CXX),$(CXX20_WARNFLAGS) $(filter -fsanitize% -fno-sanitize% -fno-omit-frame-pointer -g,$(CFLAGS)) -Itest/cuda-stubs -Icuda -DIFN_CUDA_TARGET_SM=121,$<,$(THREAD_LIBS) $(LDFLAGS),$@)
$(BUILD_DIR)/cuda-policy-86$(EXE): test/cuda_policy.cpp src/cuda/geometry_contract.hpp src/inkling_geometry.h test/cuda-stubs/cuda_runtime.h src/cuda/launch_policy.hpp src/cuda/resource_policy.hpp src/cuda/prefill_resources.hpp src/cuda/target_profile.h $(BUILD_DIR)/config
	$(call ifn_program,$(CXX),$(CXX20_WARNFLAGS) $(filter -fsanitize% -fno-sanitize% -fno-omit-frame-pointer -g,$(CFLAGS)) -Itest/cuda-stubs -Icuda -DIFN_CUDA_TARGET_SM=86 -DIFN_CUDA_ENABLE_EXPERIMENTAL_AMPERE=1,$<,$(THREAD_LIBS) $(LDFLAGS),$@)
.PHONY: cuda-policy-check
cuda-policy-check: $(BUILD_DIR)/cuda-policy-120$(EXE) $(BUILD_DIR)/cuda-policy-121$(EXE) $(BUILD_DIR)/cuda-policy-86$(EXE)
	$(BUILD_DIR)/cuda-policy-120$(EXE) 12 0 1
	$(BUILD_DIR)/cuda-policy-121$(EXE) 12 1 1
	$(BUILD_DIR)/cuda-policy-86$(EXE) 8 6 0
	$(PYTHON) test/cuda_profiles.py --cxx '$(CXX)' --flags '$(CXX20_WARNFLAGS)'
check: cuda-policy-check
.PHONY: print-build-dir
print-build-dir:
	@printf '%s\n' '$(BUILD_DIR)'

# Compile the real native acceptance fixture in CPU CI; execution stays GPU-only.
check: $(BUILD_DIR)/infer-native.o

$(BUILD_DIR)/conversion-oracles.o: test/conversion_oracles.cu $(CUDA_HEADERS) $(BUILD_DIR)/config
	$(NVCC) $(CUDAFLAGS) $(CUDA_TARGET_FLAGS) -c $< -o $@
$(BUILD_DIR)/conversion-oracles$(EXE): $(BUILD_DIR)/conversion-oracles.o
	$(call ifn_program,$(CXX),,$^,$(CUDA_LIBS),$@)
.PHONY: conversion-oracle-check
conversion-oracle-check: generated-check $(BUILD_DIR)/conversion-oracles$(EXE)
	$(PYTHON) tools/run_cuda_smoke.py '$(CUDA_RUNTIME_PATH)' $(BUILD_DIR)/conversion-oracles$(EXE)

.PHONY: release-build-info
release-build-info:
	@$(info IFN_RELEASE_BUILD_DIR=$(BUILD_DIR))
	@$(info IFN_RELEASE_EXE=$(EXE))
	@$(info IFN_RELEASE_PROVIDER=$(abspath $(PROVIDER)))
	@$(info IFN_RELEASE_CC=$(CC))
	@$(info IFN_RELEASE_NVCC=$(NVCC))
	@$(info IFN_RELEASE_CUDA_RUNTIME_PATH=$(CUDA_RUNTIME_PATH))
	@:

$(BUILD_DIR)/linear-shapes.o: test/linear_shapes.cu $(CUDA_HEADERS) src/checked_prefill.h $(BUILD_DIR)/config
	$(NVCC) $(CUDAFLAGS) $(CUDA_TARGET_FLAGS) -c $< -o $@
$(BUILD_DIR)/linear-shapes$(EXE): $(BUILD_DIR)/linear-shapes.o $(CUDA_STATIC)
	$(call ifn_program,$(CXX),,$^,$(CUDA_LIBS) $(LDFLAGS),$@)
.PHONY: linear-shape-check
linear-shape-check: generated-check $(BUILD_DIR)/linear-shapes$(EXE)
	$(PYTHON) tools/run_cuda_smoke.py '$(CUDA_RUNTIME_PATH)' $(BUILD_DIR)/linear-shapes$(EXE)

# Compile both benchmark clients in CPU CI; linking/execution remains GPU-only.
check: $(BUILD_DIR)/native-benchmark.o $(BUILD_DIR)/facade-benchmark.o

# Test instrumentation replaces only this archive member; production builds do not enable taps.
$(BUILD_DIR)/numerical-forward.o: src/engine_forward.c $(IFN_HEADER_DEPS) $(BUILD_DIR)/config
	$(call ifn_compile,$(CC),$(CPPFLAGS) $(CFLAGS) $(WARNFLAGS) -DIFN_TEST_NUMERICS,$<,$@)
$(BUILD_DIR)/numerical.o: test/numerical.c $(IFN_HEADER_DEPS) $(BUILD_DIR)/config
	$(call ifn_compile,$(CC),$(CPPFLAGS) $(CFLAGS) $(WARNFLAGS),$<,$@)
$(BUILD_DIR)/numerical$(EXE): $(BUILD_DIR)/numerical.o $(BUILD_DIR)/numerical-forward.o $(BUILD_DIR)/numerical-prefill.o $(GPU_STATIC) $(HOST_STATIC)
	$(call ifn_program,$(CXX),,$^,$(CUDA_LIBS) $(LDFLAGS),$@)
check: $(BUILD_DIR)/numerical.o $(BUILD_DIR)/numerical-forward.o

.PHONY: numerical-compare-check
numerical-compare-check:
	$(PYTHON) test/numerical_compare.py
	$(PYTHON) test/first_divergence.py
check: numerical-compare-check

$(BUILD_DIR)/numerical-prefill.o: src/engine_prefill.c $(IFN_HEADER_DEPS) $(BUILD_DIR)/config
	$(call ifn_compile,$(CC),$(CPPFLAGS) $(CFLAGS) $(WARNFLAGS) -DIFN_TEST_NUMERICS,$<,$@)
check: $(BUILD_DIR)/numerical-prefill.o

# Semantic acceptance is separate from timed/golden generation benchmarks.
$(BUILD_DIR)/semantic.o: test/semantic.c $(PUBLIC_HEADERS) $(BUILD_DIR)/config
	$(call ifn_compile,$(CC),$(CPPFLAGS) $(CFLAGS) $(WARNFLAGS),$<,$@)
$(BUILD_DIR)/semantic$(EXE): $(BUILD_DIR)/semantic.o $(GPU_STATIC) $(HOST_STATIC)
	$(call ifn_program,$(CXX),,$^,$(CUDA_LIBS) $(LDFLAGS),$@)
.PHONY: semantic-validator-check
semantic-validator-check:
	$(PYTHON) test/semantic_acceptance.py
check: $(BUILD_DIR)/semantic.o semantic-validator-check

.PHONY: audit-boundary-check audit-check
audit-boundary-check:
	$(PYTHON) tools/check_audit_boundaries.py
	$(PYTHON) test/audit_boundaries.py
check: audit-boundary-check
audit-check: generated-check audit-boundary-check
	$(PYTHON) tools/lint.py

.PHONY: qualification-safety-check
qualification-safety-check:
	$(PYTHON) test/qualification_safety.py
check: qualification-safety-check

# Worker-lifetime client intentionally compiles against the installed ABI only.
# Install the native provider first; PREFIX identifies that installation.
.PHONY: worker-lifetime worker-lifetime-check
worker-lifetime: $(BUILD_DIR)/config
	$(CC) $(CFLAGS) $(WARNFLAGS) -I$(PREFIX)/include -Itest test/worker_lifetime.c -L$(PREFIX)/lib -Wl,-rpath,$(PREFIX)/lib -linkleberry_backend -lpthread -o $(BUILD_DIR)/worker-lifetime
$(BUILD_DIR)/worker-lifetime-mock$(EXE): test/worker_lifetime.c test/worker_lifetime_mock.c $(BUILD_DIR)/config
	$(call ifn_program,$(CC),$(CFLAGS) $(WARNFLAGS) -Iinclude -Itest,test/worker_lifetime.c test/worker_lifetime_mock.c,$(THREAD_LIBS) $(LDFLAGS) $(LDLIBS),$@)
worker-lifetime-check: $(BUILD_DIR)/worker-lifetime-mock$(EXE)
	$(PYTHON) test/worker_lifetime.py ./$(BUILD_DIR)/worker-lifetime-mock$(EXE)
	$(PYTHON) test/worker_runner.py ./$(BUILD_DIR)/worker-lifetime-mock$(EXE)
ifeq ($(shell uname -s),Linux)
check: worker-lifetime-check
endif
