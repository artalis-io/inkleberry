# squid-c

A C11 host port of [tensorfish/squid5090](https://github.com/tensorfish/squid5090), retaining its Blackwell CUDA kernels and Apache-2.0 attribution.

The port follows [Jev Bush](https://github.com/artalis-io/jev-bush)'s explicit ownership, checked errors, allocation fault injection, installed-consumer tests and separate model-backed validation practices. It is GPU-first; Jev's CPU model implementation is separate.

## Implemented

- C11 static/shared library with a versioned options structure, injected allocator, opaque checkpoint ownership, typed statuses and thread-local diagnostics.
- Read-only safetensors mapping and bounded memory parsing; checked dtype/shape/offset arithmetic, UTF-8, duplicate tensor/field rejection, complete non-overlapping payload coverage and immutable concurrent lookup.
- `squid-c inspect`: real checkpoint metadata inspection, including the official Inkling main and MTP files.
- C byte-level BPE encoding, counted raw-byte decoding, special tokens, reusable scratch and the pinned single-user prompt template. No allocation during encode/decode/render.
- Architecture-specific readonly checkpoint binding: exact role/dtype/rank/shape validation for 822 main text tensors and 54 optional MTP tensors. Vision tensors are not bound. An explicitly requested missing MTP is an error.
- Generated declarations for all 76 retained CUDA functions. A two-translation-unit probe compares the actual C and CUDA C++ record layouts and checks all function prototypes.
- A C-hosted CUDA kernel smoke test with eager execution and graph replay; no Zig host required for that test.
- Host CI, malformed-input tests, allocator fault injection, sanitizer/fuzzer targets, source provenance checks, formatting and installed-library consumers.
- Optional owned CUDA weight upload/packing with exact/FP8/NVFP4, full-vocabulary MTP, bounded readback, recoverable errors and retryable cleanup; full payload parity against the pinned Zig loader.

**C model inference is not implemented yet.** Forward/prefill/decode, KV-cache/session resources, sampling/speculation and serving are subsequent port slices. Kernel smoke tests and checkpoint inspection do not establish C generation correctness or performance. [Design and acceptance gates](docs/c-port-design.md); [first-slice evidence](docs/first-port-validation.md); [tokenizer/binding evidence](docs/input-port-validation.md); [GPU weight contract and evidence](docs/gpu-weights-validation.md).

## Build and use

Host-only tools need a C11 compiler, a C++ compiler for the header consumer test, Make and Python 3. There are no third-party host-library dependencies. Linux AMD64 and macOS ARM64 host tests are exercised; CUDA execution targets Linux AMD64 / Blackwell `sm_120a`.

```sh
make -j8
./squid-c --version
./squid-c weight-plan nvfp4 mtp
./squid-c inspect /external/path/model.safetensors
./squid-c tokenize /external/path/tokenizer.json 'Hello, world!'
./squid-c validate-model /external/path/model-00001-of-00001.safetensors /external/path/mtp.safetensors
make check install-check
make lint                 # requires clang-format 18
make sanitize             # Clang ASan + UBSan; includes failure paths
make fuzz-smoke           # Clang libFuzzer + ASan + UBSan, 3,000 iterations per parser
make thread-sanitize      # immutable lookups, TLS and simulated GPU lifecycle concurrency
```

`make install-check` stages inside `.build/stage`, never in a system prefix. To install explicitly: `make install DESTDIR=/staging/path PREFIX=/usr/local`. The public API is [include/sq.h](include/sq.h). It is an initial development API; no stable ABI is promised yet. Files supplied to the reader must remain immutable for the lifetime of the mapping. The bounded reader supports U8, BF16, F32 and F8_E4M3, 2,048 tensors, rank 8, 255-byte UTF-8 names and a 16 MiB header; it is not a universal safetensors implementation or a model-shape validator.

For an external real-checkpoint descriptor comparison (no payload copying):

```sh
python3 tools/check_checkpoint.py ./libsq.so /external/path/model.safetensors /external/path/mtp.safetensors
```

On macOS use `libsq.dylib`. Use a normal, non-sanitized shared library under ordinary Python. `SAN_CC` and `SAN_CXX` select sanitizer compilers. On core-1 the installed Clang selects an incomplete GCC 14 tree; its libFuzzer link needs `SAN_CC='clang --gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/13'`. No global toolchain changes are needed.

## Tokenizer and binding contracts

The tokenizer accepts the pinned Inkling ByteLevel/BPE JSON profile, not arbitrary Hugging Face normalizers, regexes or added-token whitespace rules. Unsupported profiles and malformed vocabulary/merges fail closed. Limits: 64 MiB JSON, 204,800 token IDs, 600,000 merges and 8 MiB of token bytes. `ignore_merges` retains the pinned Zig merge-graph behavior. Unicode categories and long-input chunk boundaries intentionally match Zig; representative reference parity is tested, not exhaustive Unicode equivalence.

Tokenizers own their tables and permit concurrent immutable use with a distinct scratch handle per active call. Scratch owns about 8.25 MiB of working arrays. Invalid UTF-8 returns an error. Text inputs are counted and may contain NUL; decoded individual tokens can contain partial UTF-8 bytes. Buffer sizing returns the full required count with `SQ_BUFFER_TOO_SMALL`, including for a NULL/zero-capacity query; encoding never silently truncates. Rendering uses six typed thinking-effort presets, matching Zig's named levels; arbitrary numeric effort and general multi-turn/tool rendering are not ported yet.

A checkpoint binding owns readonly mappings, with views borrowed until close. It validates architecture layout and excludes vision, but does not scan every payload for non-finite values, validate model `config.json`, authenticate the weight contents. The optional GPU API consumes this binding and performs upload/packing separately. Auxiliary ablation/pruned-draft-vocabulary loading remains later work. A caller's files must stay immutable while mapped. Omit the MTP argument for AR-only validation; supplying a missing/incompatible MTP never silently falls back.

Real input parity tools need external files and a normal non-sanitized shared library:

```sh
python3 tools/check_inkling.py ./libsq.so /external/path/model-directory
python3 tools/check_tokenizer.py ./libsq.so /external/path/model-directory \
  --zig-oracle /external/path/pinned-squid5090 --reference
```

`--reference` requires an independently installed Python `tokenizers` package in the invoking environment. It never downloads a model; omit it to run the standard-library-only C/Zig comparison. Default `make check` uses synthetic byte-complete tokenizer fixtures and sparse, metadata-only model files, not provider weights or a GPU.

## CUDA checks

CUDA toolkit 13.5 and a Blackwell GPU are used for the retained kernels. The build retains upstream's C++20, `sm_120a` and fast-math flags. CPU-only CI does not qualify the GPU path. These commands compile locally, download nothing, and do not alter GPU configuration:

```sh
make -j8 CUDA_ROOT=/usr/local/cuda-13.5 cuda-abi-check
SQ_TEST_GPU_UUID=GPU-REPLACE-WITH-APPROVED-IDLE-UUID \
  make -j8 CUDA_ROOT=/usr/local/cuda-13.5 cuda-check
```

The GPU runner refuses a missing UUID, active compute processes or more than 256 MiB of used memory on the selected device. This is a point-in-time guard, not a scheduler reservation; coordinate exclusive access for release qualification. It never terminates other workloads. A full model/session lifecycle and the upstream cuBLAS ownership issue remain to be ported before generation can pass release gates.

## Provenance and scope

The evaluated source is pinned at `d10c0abb2c8990be81c08ba84c66a8333a691b33` in [UPSTREAM.json](UPSTREAM.json). `make generated-check` verifies every pinned CUDA/Zig/build/license input and the generated ABI. Original kernels and Zig host remain unchanged as the reference.

The `artalis` default branch is the C project line. `main` preserves the newer upstream snapshot inherited when the fork was created; kernel upgrades are reviewed separately from the language port. The architectural reference is Jev Bush revision `85fd51f8c99b1b8b7ea26d33e02413e681a9271d` (0BSD). We adopt its C engineering conventions without importing its CPU inference algorithms or accelerator design.

Scope: C host/public API, retained CUDA C++20/NVCC, official Inkling-2-Mini safetensors. No GGUF or CPU generation backend is promised. Start with the C library and CLI; serving follows correctness and lifecycle qualification.

## License

Apache-2.0: retain [LICENSE](LICENSE), [NOTICE](NOTICE) and modification attribution. Model materials and NVIDIA SDK/runtime terms are separate. No weights or credentials are included. The original README and NOTICE are archived under `docs/upstream/`.
