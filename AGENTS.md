# C inference library engineering contract

Read `reference/design/c-port-design.md` and `UPSTREAM.json` first. This project is a C11 inference
library retaining the pinned CUDA C++20 implementation. `artalis` is the project
branch; `main` preserves the upstream snapshot. Do not silently update kernels
during library maintenance.

- Preserve Apache-2.0 LICENSE/NOTICE and translation attribution.
- Public C interfaces use the `sq_` prefix, opaque owned handles, explicit status
  returns, documented lifetime/threading contracts and cleared error outputs.
- No exits, aborts, assertions on external input, process-global device ownership,
  or hidden allocation outside the injected allocator in new library code.
- Check integer/shape/offset/allocation arithmetic before use. Bound parsers and
  fuzz malformed input. Validate tensor roles/shapes before future device upload.
- Keep the pinned Zig host as an optional oracle. `make generated-check` verifies
  production pins and schema-generated artifacts; `make reference-check` also
  verifies pinned reference bytes. Generate CUDA declarations from `abi/cuda.json`
  through `tools/generate_cuda_abi.py`; never hand-edit generated declarations.
- Follow Jev Bush's C engineering practices, not its model algorithms. Keep
  modular sources canonical; generate the embedding distribution with `make amalgamation`.
- Run `make check install-check lint`, ASan/UBSan and fuzz smoke for host changes.
  clang-format 18 applies only to handwritten implementation files. CPU CI cannot prove
  numerical inference correctness or CUDA resource safety.
- GPU tests require an explicitly selected idle device via `SQ_TEST_GPU_UUID`.
  Do not terminate other workloads, alter power/clock settings, install global
  toolchains or modify existing inference services. Use isolated build paths.
- Never check in checkpoints, keys or generated bulk evaluation outputs.
- Do not claim C generation, speedups or quality parity until the actual C
  inference path is implemented and model-backed gates have run.

Production C sources are directly under `src/`, with checked adapters in
`src/cuda/`. Pinned Zig sources/build script live under `reference/upstream/`;
resolve original upstream names through `UPSTREAM.json`'s `relocated_files` map.
Keep upstream file hashes and historical provenance unchanged when moving files.

Release artifacts use explicit allowlists in `config/package.json`. Keep reference
material out of production packages and regenerate `dist/` after canonical changes.
