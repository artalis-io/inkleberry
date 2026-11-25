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
- Optional C greedy AR sessions with retained weights, owned KV/recurrent state, recoverable cuBLAS dispatch, graph/eager execution and cancellation/reset.
- Optional owned CUDA weight upload/packing with exact/FP8/NVFP4, full-vocabulary MTP, bounded readback, recoverable errors and retryable cleanup; full payload parity against the pinned Zig loader.

**C autoregressive, fixed-width and adaptive speculative generation are implemented.** It includes the 48-visit forward path, Tensor Core batched prefill, BF16/FP8/Q4 KV, eager/graph decode, teacher-forced logits and cooperative cancellation. Temperature/top-p/top-k sampling and fixed-width DSpark speculation are also implemented. Adaptive 2/4/8-token verification and verified prompt lookup are implemented. Session-private prefix caching is available by explicit opt-in. Serving remains a subsequent slice. Model-backed validation is documented below. [Design and acceptance gates](docs/c-port-design.md); [first-slice evidence](docs/first-port-validation.md); [tokenizer/binding evidence](docs/input-port-validation.md); [GPU weight contract and evidence](docs/gpu-weights-validation.md); [AR execution contract and evidence](docs/ar-validation.md); [sampling/speculation contract](docs/speculation-validation.md); [adaptive/lookup contract](docs/adaptive-validation.md); [prefix-cache contract](docs/prefix-validation.md).

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

The GPU runner refuses a missing UUID, active compute processes or more than 256 MiB of used memory on the selected device. This is a point-in-time guard, not a scheduler reservation; coordinate exclusive access for release qualification. It never terminates other workloads. The C AR path owns its cuBLAS handle and uses checked prefill dispatch; it never calls upstream's global cuBLAS initializer. One live C session per process is supported initially. See the AR validation contract for remaining release gates.

## C autoregressive generation

```sh
make -j8 CUDA_ROOT=/usr/local/cuda-13.5 squid-c-generate
SQ_TEST_GPU_UUID=GPU-REPLACE-WITH-APPROVED-IDLE-UUID \
  python3 tools/run_cuda_smoke.py /usr/local/cuda-13.5/lib64 \
  ./squid-c-generate /external/model-directory 'What is 2 + 2?' 128 nvfp4 fp8 graph
```

The CLI uses the C tokenizer and single-user template with `SQ_THINK_NONE`, and
streams raw decoded token bytes. It supports `exact|fp8|nvfp4`, `bf16|fp8|q4` KV
and `eager|graph`. Its context limit is 8,192 tokens; the typed API accepts an
explicit limit up to 1,048,576, subject to available memory. This upper bound is
input validation, not a claim of qualification at every context length.
The embedding API is [include/sq_engine.h](include/sq_engine.h). No networking or
Zig dependency is required for the C library/CLI; Zig is used only as a test oracle.

For deterministic stochastic generation and fixed-width DSpark, append
`--temperature 0.8 --top-p 0.95 --top-k 40 --seed 42 --drafts 7` to the CLI
command. Positive drafts require the external `mtp.safetensors`; there is no
silent AR fallback. The typed API is `sq_session_generate_ex`. See the
[sampling/speculation contract](docs/speculation-validation.md) for verification,
acceptance counters and the fixed-width scope. For adaptive verification append
`--adaptive-threshold 0.8 --prompt-lookup 1` alongside `--drafts 7` and `graph`.
Both options are explicit; existing commands keep their previous behavior. See
[adaptive validation](docs/adaptive-validation.md) for graph ownership, overlap,
lookup verification and qualification commands.

## Prefix reuse

The embedding API can opt in with `sq_session_prefix_cache(session, 1)`. It
allocates four host-RAM local/recurrent checkpoints and a bounded global-KV
side-request shield, with no additional device arena. At the tested 8K FP8/MTP
configuration this uses about 984 MiB of host RAM. Defaults allocate none of it.
Use `sq_decode_options.system_prefix_tokens` for an optional shared-prefix
checkpoint boundary, and `sq_session_prefix_info` for completed-request reuse,
restore, shield and host-memory counts. Reset or disable between separate trust
scopes; this is a session cache, not a multi-tenant storage layer.

The CLI accepts `--prefix-cache 1 --system-prefix 128`, but each invocation is
one request: the cache does not persist across processes. Repeated embedding
calls and the [prefix oracle/recovery tests](docs/prefix-validation.md) exercise
actual reuse. Cancellation, errors, explicit reset and direct logits mutation
invalidate cached history. See the contract for memory costs and limitations.

## Text conversations and streaming

The host-only [sq_chat.h](include/sq_chat.h) API encodes multi-turn text messages
with separate trusted markers and untrusted text, optional reasoning history and
prefix-cache hints. Its bounded stream filter separates content/reasoning,
handles UTF-8 across token boundaries and suppresses custom stops without leaking
partial matches. Streaming and collected results share the same filter.

See [the contract and executed tests](docs/conversation-streaming.md). The legacy
single-user CLI stays compatible. Tools, multimodal messages and HTTP serving
remain separate; the HTTP adapter will resolve and pin the latest Keel release.

## Provenance and scope

The evaluated source is pinned at `d10c0abb2c8990be81c08ba84c66a8333a691b33` in [UPSTREAM.json](UPSTREAM.json). `make generated-check` verifies every pinned CUDA/Zig/build/license input and the generated ABI. Original kernels and Zig host remain unchanged as the reference.

The `artalis` default branch is the C project line. `main` preserves the newer upstream snapshot inherited when the fork was created; kernel upgrades are reviewed separately from the language port. The architectural reference is Jev Bush revision `85fd51f8c99b1b8b7ea26d33e02413e681a9271d` (0BSD). We adopt its C engineering conventions without importing its CPU inference algorithms or accelerator design.

Scope: C host/public API, retained CUDA C++20/NVCC, official Inkling-2-Mini safetensors. No GGUF or CPU generation backend is promised. Start with the C library and CLI; serving follows correctness and lifecycle qualification.

## License

Apache-2.0: retain [LICENSE](LICENSE), [NOTICE](NOTICE) and modification attribution. Model materials and NVIDIA SDK/runtime terms are separate. No weights or credentials are included. The original README and NOTICE are archived under `docs/upstream/`.

The experimental [inference ABI](docs/inference-abi.md) separates model execution
from protocol adapters for open-source and proprietary harnesses. It supports
independent request streams with request-scoped cancellation; this backend still
executes one generation at a time per process. See the
[design rationale](docs/inference-abi-design.md). HTTP serving is not implemented.
