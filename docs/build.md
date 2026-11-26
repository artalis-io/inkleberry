# Build and validation

The host library needs a C11 compiler and `libm`. Development targets also use
Make, Python 3 standard library, a C++ compiler for header-consumer checks and
clang-format 18 for formatting. No Python package installation is required.
CUDA execution needs a compatible NVIDIA driver, CUDA toolkit and cuBLAS.
The current GPU qualification target is Linux AMD64/Blackwell `sm_120a` with
CUDA 13.5. Host checks also run on macOS ARM64; that is not a macOS GPU backend.

```sh
make -j8
make check install-check
make lint
make sanitize
make fuzz-smoke
make thread-sanitize
```

`make check` tests modular and amalgamated host implementations, including mapped
inputs, malformed files, ownership, tokenizer/conversation/streaming behavior,
allocation faults, concurrency and symbol compatibility. `make install-check`
stages its files under `.build/stage` and tests real installed consumers.

```sh
make install DESTDIR=/staging/path PREFIX=/usr/local
make -j8 gpu infer-shared inkleberry-generate
make install-infer-backend DESTDIR=/staging/path PREFIX=/usr/local
```

`libinkleberry` supplies host input/validation APIs, `libinkleberry-gpu.a` the modular native host,
and `libinkleberry_backend.so.1` the versioned provider. `libinkleberry-cuda.a` supplies only the
CUDA implementation for the amalgamated C host. Keep model files external.

## GPU validation

Reserve the selected device; the test guard is a point-in-time idle check, not a
scheduler. These tests never terminate unrelated workloads or change GPU settings.

```sh
make cuda-abi-negative-check
make .build/infer-native-shared .build/infer-native-amalgam
INKLEBERRY_TEST_GPU_UUID=GPU-APPROVED-ID INKLEBERRY_TEST_TIMEOUT_SECONDS=600 \
  python3 tools/run_cuda_smoke.py /usr/local/cuda/lib64 \
  .build/infer-native-amalgam /external/model-directory
```

The guard rejects absent UUIDs, active compute processes and devices with more
than 256 MiB in use. GB10 reports dedicated VRAM as unavailable because it uses
unified memory; that case additionally requires zero reported GPU utilization,
while still rejecting any compute process. Unavailable counters on other devices
fail closed. Check system-memory headroom separately before model-backed GB10
tests. C/CUDA declaration and layout checks do not execute kernels.
Native model workflows do. A CPU-only CI result does not qualify GPU execution.

```sh
make generated-check       # production pins and native ABI schema
make amalgamation-check    # exact generated distribution freshness
make reference-free-check  # build with reference files absent
make release-check         # build/exercise the actual extracted source and SDK
```

The retired Zig host and comparison harnesses are no longer included. Native
model qualification uses fixed golden tokens, recovery/cancellation tests and
measured performance gates; see [releases](releases.md).

`SAN_CC`/`SAN_CXX` select sanitizer compilers. On the qualified core host, selecting
its installed GCC 13 runtime directory avoids an incomplete GCC search path:
`SAN_CC='clang --gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/13'`, with matching
`SAN_CXX`. This is a local compiler argument, not a system/toolchain modification.


## Required and optional CI

Pull requests and pushes run Linux GCC/Clang host checks, ASan/UBSan with timed
fuzzing, lint and TSan. All five are required merge checks on `main`, bound to
the GitHub Actions app. Branches must be up to date; administrators are subject
to the same checks. The desired configuration is checked in at
[.github/branch-protection.json](../.github/branch-protection.json). Changes to
these job names must update the required-check configuration as well.

macOS tests only the portable host C library. It is an optional manual workflow,
not a PR/push dependency and not CUDA qualification:

```sh
gh workflow run macos-portability.yml --repo artalis-io/inkleberry --ref main
```

The workflow becomes available on the default branch once its PR is merged.
The trusted, manually initiated model-backed GPU workflow remains separate.
Native release qualification still requires a reserved device and external model
assets; Linux CPU CI does not replace it.

## Native Windows x64

Use Visual Studio 2022 Build Tools with the x64 C++ workload and Windows SDK,
LLVM `clang-cl` for the second compiler profile, and GNU Make/MSYS2 Bash for
build recipes. Compilation, linking and runtime use the native Windows ABI and
dynamic Microsoft CRT, not a MinGW or MSYS runtime. The environment helper only
imports the installed toolchain into the current shell; it installs nothing.

From MSYS2 Bash in the checkout:

```sh
source scripts/msvc-env.sh
make -j4 CC=cl PYTHON=python check install-check
make -j4 CC=clang-cl PYTHON=python check install-check
```

Put native Python and LLVM on PATH if they are not already available. The native
profile sets Python UTF-8 mode and C/C++ source UTF-8 explicitly. MSVC uses
`/std:c11 /experimental:c11atomics`; clang-cl uses its native C11 atomics. Both use
`/W4 /WX /MD`. Output trees are separate:
`.build/windows-msvc-x64-release-cuda120a` and
`.build/windows-clang-cl-x64-release-cuda120a`. Do not mix archives/objects from
different CRT, compiler, architecture or CUDA target configurations.

With an existing CUDA installation (`CUDA_PATH`, or an explicit `CUDA_ROOT`):

```sh
make -j2 CC=cl PYTHON=python gpu infer-shared
make -j2 CC=clang-cl PYTHON=python gpu infer-shared
```

NVCC continues to compile the retained CUDA C++20 implementation using its
supported MSVC host compiler. `CC=clang-cl` selects the C host compiler, not
Clang-CUDA. The default target remains `sm_120a`; Windows portability does not
qualify Ampere or change the device capability gate. A successful build is not
a model-backed numerical or throughput qualification.

The native host SDK produces `inkleberry.lib`, `inkleberry-1.dll` and its distinct
`inkleberry-1.lib` import library. The inference DLL is
`inkleberry_backend-1.dll`; its `.def` exports only `ifn_infer_get_api`. Keep the
DLLs beside the executable or on the operator-controlled DLL search path.
CUDA/driver runtime dependencies are external and are not bundled.

For Windows runtime packages, pass the selected native output directory:

```sh
python tools/release.py runtime --build-root .build/windows-msvc-x64-release-cuda120a
# Add --gpu only after building and validating the native CUDA targets.
```

`make install` and `make install-contract-check` support native Windows custom
prefixes without requiring symlink privileges. Use the installed `.lib` files
explicitly with MSVC/clang-cl; GNU pkg-config metadata is emitted only on POSIX.
`make windows-sdk-check` verifies deterministic archives and extracted C/C++
consumers. `make windows-cuda-build-check` compiles/links CUDA, inspects PE exports
and imports, negotiates the provider ABI and runs the C/C++ layout probe without
loading model weights or executing inference.
