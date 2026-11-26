# Release contents

Release creation is local and does not publish artifacts or start services.
Existing command/library and public symbol names remain compatible.

| Artifact | Contents | Excludes |
| --- | --- | --- |
| Source | Explicitly allowlisted maintained C/CUDA, headers, schema, tools, tests, docs and generated embedding files | Reference implementations, experiments, model files, build outputs and private state |
| Embedding | `dist/ifn.h`, `dist/ifn.c`, C examples and notices | Python/Zig, source tree, generators and test fixtures |
| Host runtime/SDK | Host executable, shared library, public headers and dependency manifest | Development tools, sources, reference files and models |
| GPU runtime/SDK | Host runtime plus native generation command, provider and CUDA implementation archive | Proprietary driver/SDK libraries and model files |

```sh
make packages VERSION=0.1.0-dev
make runtime-gpu-package VERSION=0.1.0-dev
make release-check
```

Artifacts are written under `.build/releases/`. `config/package.json` is an explicit
source/header allowlist; unrelated local files cannot enter an archive by directory
scanning. Inputs must be regular files, not symlinks to external data. Only the
known versioned provider link is created in GPU packages.

Every archive includes LICENSE, NOTICE, THIRD_PARTY.md and a checksum manifest.
Ordering, ownership, modes and compression timestamps are deterministic. Set
`SOURCE_DATE_EPOCH` to control timestamps (default zero). The same source inputs
and version produce the same source archive. Native artifacts differ by build
architecture/toolchain; architecture is recorded in the runtime manifest.

Runtime packages contain no Python or Zig. Compatible NVIDIA driver/CUDA/cuBLAS
libraries are supplied separately by the deployment host and declared in
`DEPENDENCIES.json`. The package does not install a toolkit, modify system services
or distribute an entire SDK. Model data, credentials and customer state are never
release inputs.

`make release-check` creates and verifies archives in a temporary directory,
rebuilds and tests the extracted production sources with the reference tree absent,
compiles/runs C and C++ consumers from the two-file embedding distribution and
executes the packaged host command. GPU qualification separately runs extracted
GPU artifacts on a reserved device. Packaging alone does not establish numerical
GPU correctness.
