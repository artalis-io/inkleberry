# squid-c

A C11 host port of [tensorfish/squid5090](https://github.com/tensorfish/squid5090), retaining its Blackwell CUDA kernels and Apache-2.0 attribution.

The port follows [Jev Bush](https://github.com/artalis-io/jev-bush)'s explicit C API ownership, checked errors, fault injection, installed-consumer tests and separate model-backed validation practices. It is GPU-first; Jev's CPU inference implementation is a different model and algorithm.

## Status

Implementation has started from the evaluated upstream revision `d10c0abb2c8990be81c08ba84c66a8333a691b33`. [Design and acceptance gates](docs/c-port-design.md).

The C inference loop is not implemented yet. The initial foundation slice is developed separately; it must pass host and real CUDA ABI/kernel checks before expansion. No C generation performance or model parity is claimed at this stage.

The `artalis` default branch is the C project line. `main` preserves the newer upstream snapshot inherited when the GitHub fork was created. Kernel upgrades are separate from the language port.

## Scope

C host and public API, retained CUDA C++20 compiled with NVCC, Linux AMD64 / Blackwell `sm_120a` first. Official Inkling-2-Mini safetensors; no GGUF or CPU generation backend is promised. The original Zig sources and `build.zig` remain the pinned reference during transition.

## License

Apache-2.0: retain [LICENSE](LICENSE), [NOTICE](NOTICE) and modification attribution. Model materials and NVIDIA SDK/runtime terms are separate. No weights or credentials are included. The original upstream README and NOTICE are archived under `docs/upstream/`.
