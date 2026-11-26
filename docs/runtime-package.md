# Native runtime/SDK

Extract this archive into a deployment-owned directory. `MANIFEST.json` records
its architecture, package version, contents and SHA-256 digests. The `bin/`
commands run without Python or Zig. Public C headers and native libraries are
under `include/` and `lib/`; development tests/reference material are excluded.

```sh
./bin/inkleberry --version
./bin/inkleberry inspect /external/checkpoint.safetensors
```

GPU packages additionally contain the generation command, versioned provider and
CUDA implementation archive. Supply a compatible NVIDIA driver, CUDA runtime,
cuBLAS (including its dependencies) and C++ runtime on the host. SDK and driver
binaries are not included. See `DEPENDENCIES.json`. Keep their library paths
available through the normal system loader configuration or a process-local
`LD_LIBRARY_PATH`; do not change machine configuration as part of extraction.

```sh
CUDA_VISIBLE_DEVICES=GPU-APPROVED-ID LD_LIBRARY_PATH=/usr/local/cuda/lib64 \
  ./bin/inkleberry-generate /external/model-directory 'What is 2 + 2?' 32 nvfp4 fp8 graph
```

Model files are external and must remain immutable while mapped. This is a native
architecture-specific package, not a cross-architecture image. No service is
installed or started by extraction. Select a reserved GPU before inference.
