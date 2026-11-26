# Embedding distribution

`dist/ifn.h` exposes the existing public C interfaces in one header. `dist/ifn.c`
contains the C implementation in one translation unit. The modular sources remain
authoritative. Regenerate with `make amalgamation`; CI verifies it with
`make amalgamation-check`. Do not edit generated files directly.

## Host APIs

The two files and a C11 compiler are sufficient for checkpoint readers, tokenizer,
conversation/stream filtering and CPU allocation planning:

```sh
cc -O2 -std=c11 -Idist dist/ifn.c examples/inspect.c -lm -o inspect
./inspect /external/checkpoint.safetensors
```

Only standard system headers are included. No Python, Zig, source tree or generator
is needed by the embedding consumer. A C++ application includes the same header
and links the implementation compiled as C. Do not compile the implementation as
C++, and do not link both modular and amalgamated definitions into one application.
This host build does not implement CPU model generation.

## GPU inference

Enable the C native host and provider with `IFN_ENABLE_CUDA`, then link the separately
compiled CUDA implementation and the existing NVIDIA/system libraries:

```sh
make libinkleberry-cuda.a
cc -O2 -std=c11 -DIFN_ENABLE_CUDA -Idist -c dist/ifn.c -o inference.o
cc -O2 -std=c11 -Idist -c examples/generate.c -o example.o
c++ example.o inference.o libinkleberry-cuda.a \
  -L/usr/local/cuda/lib64 -L/usr/local/cuda/lib64/stubs \
  -lcudart -lcublas -lcuda -lstdc++ -lm -lpthread -o generate
CUDA_VISIBLE_DEVICES=GPU-APPROVED-ID LD_LIBRARY_PATH=/usr/local/cuda/lib64 \
  ./generate /external/model-directory 'What is 2 + 2?'
```

The CUDA archive is built with NVCC. The existing kernel mathematics, graph
execution and precision behavior are retained. This is a single C host file,
not an ISO-C replacement for the CUDA implementation. The archive is native to
its build architecture; it is also included in the GPU runtime/SDK package.

The [inference interface](inference-abi.md) defines lifecycle, request ownership,
limits and cancellation. Current execution capacity is one active generation per
provider process. Output callbacks borrow their spans and must return promptly.
Keep checkpoint inputs immutable while mapped, and close children before parents.

## Generator contract

The source groups in `config/sources.mk` drive modular builds and generation.
Local headers are expanded once, public headers are combined, and file-private
helpers are mechanically namespaced to preserve their translation-unit scope.
Comments/string literals and member accesses are preserved. File-local macros
are cleared after each module. Canonical C source and algorithm expressions are
not edited. The generator follows the single-file distribution pattern while
keeping the modules reviewable.

Host regressions and parser fuzz smoke run against both implementations; public/global symbol sets must
match. The native qualification target exercises the amalgamated GPU host; the
extracted modular and amalgamated paths passed (see [validation](distribution-validation.md)). The generated source
and its licenses/attribution travel together in the embedding archive.
