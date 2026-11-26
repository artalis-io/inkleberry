# Inkleberry Finn

![Inkleberry Finn logo](logo.png)

Inkleberry Finn is a C11 inference library for Inkling-2-Mini, with checkpoint
loading, tokenization and native Blackwell GPU execution.
It provides explicit resource ownership, recoverable errors, streaming generation
and a versioned C interface for embedding in applications and serving layers.
See [library integration and qualification](docs/library-integration.md) for ABI,
diagnostics, artifact verification and model-backed release checks.

Current model support is Inkling-2-Mini. Native execution is qualified on Linux
AMD64 with an RTX PRO 6000 Blackwell; host tooling is tested on Linux AMD64 and
macOS ARM64. GPU kernels use CUDA C++20. The C runtime needs neither Python nor Zig.

## Build

```sh
git clone https://github.com/artalis-io/inkleberry.git
cd inkleberry
make -j8
make check install-check
make lint                         # clang-format 18
```

Development checks and generators use Python's standard library. CUDA builds
require the NVIDIA toolkit; the tested configuration is CUDA 13.5, `sm_120a`.

```sh
make -j8 inkleberry-generate infer-shared
./inkleberry inspect /external/model.safetensors
./inkleberry-generate /external/model-directory 'What is 2 + 2?' 128 nvfp4 fp8 graph
```

Use a reserved GPU when running inference. The public namespace is `ifn_*` / `IFN_*`, with `inkleberry` executables
and `libinkleberry` libraries. This replaces the pre-release names; no compatibility aliases are provided. Build, installation and validation
commands are in [the build guide](docs/build.md).

## Embed

Use the modular headers under `include/`, or the generated two-file distribution:

```sh
cc -O2 -std=c11 -Idist dist/ifn.c examples/inspect.c -lm -o inspect
./inspect /external/model.safetensors
```

`dist/ifn.h` and `dist/ifn.c` are generated from the canonical modules. Host input
APIs work with a C compiler alone; GPU inference additionally links the separately
compiled CUDA implementation. See [embedding](docs/embedding.md) and the
[versioned inference interface](docs/inference-abi.md).

## Capabilities

- Read-only memory-mapped safetensors, checked tensor layouts and borrowed payloads.
- BPE tokenization and typed text conversations, with bounded UTF-8/stop filtering.
- Autoregressive and fixed/adaptive speculative generation; deterministic sampling.
- BF16/FP8/Q4 KV, CUDA graphs and explicit prefix reuse with isolation scopes.
- Request-owned cancellation, terminal results, capability discovery and budgets.

The current backend supports one prepared GPU session and one active generation
per process. Independent streams have separate state; scheduling belongs to the
caller. CPU generation, multimodal input and HTTP serving are not implemented.

## Distribution

```sh
make packages                    # source, embedding and host runtime archives
make runtime-gpu-package         # architecture-specific GPU runtime/SDK
make release-check               # build and exercise extracted artifacts
```

Packages use explicit file allowlists and checksum manifests. Model files,
reference material, experiments and private state are excluded. Runtime archives
contain no Python/Zig code. GPU packages declare their required driver/CUDA/cuBLAS
libraries; proprietary SDK/driver binaries are supplied by the deployment host.
See [release contents](docs/releases.md).

## Documentation

- [Contributor and agent guide](AGENTS.md) · [Claude Code guide](CLAUDE.md)
- [Naming and pre-release API transition](docs/naming.md)
- [Single-GPU RTX Blackwell / GB10 measurements](docs/validation/blackwell-gb10-2026-11-26.md)
- [Build and testing](docs/build.md)
- [Embedding and amalgamation](docs/embedding.md)
- [Checkpoint storage and memory mapping](docs/storage.md)
- [Inference API](docs/inference-abi.md)
- [Serving contract design](docs/serving-contract.md)
- [Conversations and streaming](docs/conversation-streaming.md)
- [Native ABI schema](abi/README.md)

Historical designs and qualification receipts live
under `reference/`, outside the production distribution. Attribution and source
provenance are retained in [LICENSE](LICENSE), [NOTICE](NOTICE),
[THIRD_PARTY.md](THIRD_PARTY.md) and `UPSTREAM.json`.
