# C library/installed-consumer conventions adapted from artalis-io/jev-bush (0BSD).
.DEFAULT_GOAL := all
CC ?= cc
CXX ?= c++
AR ?= ar
SAN_CC ?= clang
SAN_CXX ?= clang++
CFLAGS ?= -O2
WARNFLAGS = -std=c11 -Wall -Wextra -Wpedantic -Wshadow -Wformat=2 -Werror
CPPFLAGS += -Iinclude -Isrc/c -D_FILE_OFFSET_BITS=64
PREFIX ?= /usr/local
CUDA_ROOT ?= /usr/local/cuda
NVCC = $(CUDA_ROOT)/bin/nvcc
CUDA_ARCH ?= 120a
CUDAFLAGS = -O3 --use_fast_math -std=c++20 -gencode arch=compute_$(CUDA_ARCH),code=sm_$(CUDA_ARCH) --expt-relaxed-constexpr -Icuda
CUDA_LIBS = -L$(CUDA_ROOT)/lib64 -lcudart -lcublas -lstdc++ -lm -lpthread
HOST = src/c/foundation.c src/c/safetensors.c
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
	$(CC) $(SHARED_FLAGS) $(LDFLAGS) $^ -o $@
squid-c: cli/main.c libsq.a .build/config
	$(CC) $(CPPFLAGS) $(CFLAGS) $(WARNFLAGS) $< libsq.a $(LDFLAGS) -o $@
.build/unit: test/unit.c libsq.a .build/config
	$(CC) $(CPPFLAGS) $(CFLAGS) $(WARNFLAGS) $< libsq.a $(LDFLAGS) -o $@
.build/concurrency: test/concurrency.c libsq.a .build/config
	$(CC) $(CPPFLAGS) $(CFLAGS) $(WARNFLAGS) $< libsq.a -pthread $(LDFLAGS) -o $@
generated-check:
	python3 tools/verify_upstream.py
	python3 tools/generate_cuda_abi.py --check
check: all generated-check .build/unit .build/concurrency
	.build/unit
	.build/concurrency
	python3 test/safetensors_cases.py ./squid-c
	$(CC) $(CPPFLAGS) $(CFLAGS) $(WARNFLAGS) -c test/cuda_abi_c.c -o .build/cuda_abi_c.o
	$(CXX) -std=c++11 -Wall -Wextra -Werror -Iinclude -x c++ -c test/installed_consumer.c -o .build/cpp-consumer.o
	$(CC) .build/cpp-consumer.o libsq.a $(LDFLAGS) -o .build/cpp-consumer
	.build/cpp-consumer
sanitize:
	$(MAKE) clean
	$(MAKE) CC=$(SAN_CC) CXX=$(SAN_CXX) CFLAGS='-O1 -g -fno-omit-frame-pointer -fsanitize=address,undefined -fno-sanitize-recover=all' LDFLAGS='-fsanitize=address,undefined' check
thread-sanitize:
	$(MAKE) clean
	$(MAKE) CC=$(SAN_CC) CXX=$(SAN_CXX) CFLAGS='-O1 -g -fsanitize=thread' LDFLAGS='-fsanitize=thread' .build/concurrency
	.build/concurrency
fuzz-smoke: | .build
	python3 test/safetensors_cases.py --corpus .build/fuzz-corpus
	$(SAN_CC) -O1 -g $(CPPFLAGS) -std=c11 -fsanitize=fuzzer,address,undefined -fno-sanitize-recover=all $(HOST) fuzz/fuzz_safetensors.c -o .build/fuzz-safetensors
	.build/fuzz-safetensors -runs=3000 -max_len=262144 .build/fuzz-corpus
format:
	tools/format.sh
lint: generated-check
	tools/format.sh --check
	python3 tools/lint.py
install: all
	mkdir -p $(DESTDIR)$(PREFIX)/include $(DESTDIR)$(PREFIX)/lib/pkgconfig $(DESTDIR)$(PREFIX)/bin
	cp include/sq.h $(DESTDIR)$(PREFIX)/include/
	cp libsq.a $(SHARED) $(DESTDIR)$(PREFIX)/lib/
	cp squid-c $(DESTDIR)$(PREFIX)/bin/
	sed 's|@PREFIX@|$(PREFIX)|g' sq.pc.in > $(DESTDIR)$(PREFIX)/lib/pkgconfig/squid-c.pc
install-check: DESTDIR = $(CURDIR)/.build/stage
install-check: PREFIX = /usr/local
install-check: install
	$(CC) $(CFLAGS) $(WARNFLAGS) -I.build/stage/usr/local/include test/installed_consumer.c .build/stage/usr/local/lib/libsq.a $(LDFLAGS) -o .build/installed-static
	.build/installed-static
	$(CC) $(CFLAGS) $(WARNFLAGS) -I.build/stage/usr/local/include test/installed_consumer.c -L.build/stage/usr/local/lib -lsq -Wl,-rpath,'$(CURDIR)/.build/stage/usr/local/lib' $(LDFLAGS) -o .build/installed-shared
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
	rm -rf .build squid-c libsq.a libsq.so libsq.dylib
-include $(wildcard .build/*.d)
