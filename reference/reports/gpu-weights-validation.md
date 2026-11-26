# C GPU weight upload and packing

This slice ports the pinned Zig weight loader, not forward/prefill/decode or
serving. Original CUDA and Zig files remain byte-for-byte pinned at
`d10c0abb2c8990be81c08ba84c66a8333a691b33`. The previous C tokenizer and binding
slice is the base of this stacked change.

## Implemented contract

`include/sq_weights.h` provides a host-only allocation planner. `include/sq_gpu.h`
adds an optional CUDA backend, linked as `libsq-gpu.a` before `libsq.a`. There is
no simulated production backend. Fault injection uses a private test vtable.

The recipe uses the original 256-byte arena order, query/KV concatenation and
alias slicing, expert scale calculation, BF16/FP8/NVFP4 choices, RNN splitting,
full-vocabulary MTP fold and draft layers. Planning and execution share the C
recipe, but validation uses independent Zig allocation accounting and the actual
unchanged Zig loader. All payload bytes and descriptor fields are compared;
uninitialized alignment padding is deliberately excluded.

Supported options: exact, FP8 and NVFP4, AR or MTP, and forced BF16 MTP. “Exact”
retains the checkpoint's quantized expert banks; it does not reconstruct original
full-precision weights. Device embeddings and full draft vocabulary are required
for now. Ablation, host embeddings and pruned draft vocabulary are not ported.
The library does not inspect `SQUID_*` environment variables; only the test
harness uses them to configure the Zig oracle and the equivalent explicit C
options. Vision tensors remain excluded.

Create allocates host bookkeeping through the injected allocator. Load borrows a
validated readonly checkpoint only until return, preflights free device memory,
allocates an owned arena and temporary buffers, and uses a dedicated stream.
Closing the checkpoint after load does not invalidate the GPU image. Requested
peak bytes exclude driver/context overhead, so free-memory preflight is not an
allocation guarantee. Fixed Inkling shapes are validated before upload; numeric
expert auxiliaries and MTP fusion metadata are checked for finite valid values.
There is no exhaustive scan or authentication of all model payload bytes.

Load errors retain the handle: always call destroy. Failed frees retain the
allocation pointers for a subsequent destroy attempt. The first operation error
survives cleanup errors. A failed device/context restoration is returned as an
error. No reset or caller-owned context/stream destruction occurs. Each GPU call
selects the configured primary context and restores the caller's device and
current driver context, including a previously NULL context. Driver failures can
still require caller intervention; recovery never resets the whole device.

Per-handle operations reject overlap with `SQ_BUSY`. The caller must keep the
handle and checkpoint alive for every in-flight call and prevent new calls once
destruction can release the handle. The busy flag is not memory reclamation.
A completed/failed image must be destroyed before loading another checkpoint.
There are no writable public GPU pointers. Span readback is bounded to payload
bytes and synchronizes the owned stream.

The additive checked CUDA bridge compiles the unchanged upstream ABI source in
one translation unit and invokes its original helper kernels. It replaces
aborting packing trampolines with error-returning launches on the owned stream;
it does not copy or alter quantization math. Do not also link the original
`cuda_abi` object into a consumer of `libsq-gpu.a`.

## Reproduce

Host-only checks need no model or CUDA:

```sh
make -j8 check install-check
make lint                         # clang-format 18
make sanitize
make fuzz-smoke
make thread-sanitize
./squid-c weight-plan nvfp4 mtp
```

On an explicitly reserved Blackwell GPU, with external official Inkling files:

```sh
make -j8 CUDA_ROOT=/usr/local/cuda-13.5 gpu squid-c-weights
make CUDA_ROOT=/usr/local/cuda-13.5 gpu-install-check
make CUDA_ROOT=/usr/local/cuda-13.5 ZIG=/path/to/zig gpu-weight-oracle
export SQ_TEST_GPU_UUID=GPU-REPLACE-WITH-APPROVED-IDLE-UUID
python3 tools/validate_gpu_weights.py \
  --model /external/model-directory --cuda-lib /usr/local/cuda-13.5/lib64
make CUDA_ROOT=/usr/local/cuda-13.5 gpu-packing-check
python3 tools/run_cuda_smoke.py /usr/local/cuda-13.5/lib64 \
  /usr/local/cuda-13.5/bin/compute-sanitizer --tool memcheck \
  --leak-check full --error-exitcode 99 .build/packing
python3 tools/run_cuda_smoke.py /usr/local/cuda-13.5/lib64 \
  /usr/local/cuda-13.5/bin/compute-sanitizer --tool memcheck \
  --leak-check full --error-exitcode 99 ./squid-c-weights \
  /external/model-directory exact ar
```

The directory must contain `model-00001-of-00001.safetensors` and, for MTP,
`mtp.safetensors`. No download occurs. The oracle harness targets the audited
Zig 0.17.1 toolchain used by the pinned source. The matrix runner selects full
vocabulary and both MTP precision variants explicitly. Each case gets the idle
UUID guard; it refuses an existing compute workload. The guard is not a GPU
reservation and cannot eliminate scheduler races.

`make install-gpu DESTDIR=/staging PREFIX=/usr/local` installs both archives and
public headers. Compile C consumers against those headers, then link with C++:

```sh
cc -std=c11 -I/staging/usr/local/include -c consumer.c -o consumer.o
c++ consumer.o /staging/usr/local/lib/libsq-gpu.a \
  /staging/usr/local/lib/libsq.a -L/usr/local/cuda-13.5/lib64 \
  -lcudart -lcublas -lcuda -lm -lpthread -o consumer
```

A toolkit-only link environment may require its `lib64/stubs` search directory
for `-lcuda`; never put that stubs directory in the runtime library path. Runtime
needs the NVIDIA driver and CUDA libraries. Optional GPU install tests link and
execute a C consumer using only staged headers/archives, without loading a model.

## Executed evidence — core-1, 25 November 2026

Hardware: RTX PRO 6000 Blackwell Max-Q, GPU0, driver 595.104.02, 97,887 MiB,
CUDA 13.5.37, `sm_120a`; source toolchain flags retain upstream fast math.
GPU1 and unrelated services were not modified. Every GPU run passed the idle
preflight. Model file identities and machine-readable results are in
`benchmarks/gpu-weights-2026-11-25.json`.

Eight distinct layouts passed exact payload and descriptor comparison: three
modes times AR/MTP, plus FP8/NVFP4 with forced BF16 MTP. A separate CPU test
checked all nine mode/MTP-precision planner combinations, including the redundant
exact/BF16 variant, against a closed-form translation of Zig byte accounting.
Payload comparison covers roughly 15–20 GB per configuration and every active
view (646 AR, 693 MTP). This is complete payload comparison, not sampled hashes.
Each native parity run also verifies preservation of a test-owned custom driver
context and sentinel allocation during readback/destruction, plus survival of
the original Zig allocation after C teardown.

Compute Sanitizer memcheck with full leak checking passed with zero errors and
zero leaked allocations for five small checked/original kernel comparisons:
BF16→FP8, BF16→NVFP4, MXFP8 expert→NVFP4, RNN splitting and MTP folding. The
fold fixture uses eight vocabulary rows and the actual hidden/rank dimensions.
A separate complete C-only exact/AR upload/readback/teardown also passed memcheck
and leak checking. Full-model MTP was compared normally, not under memcheck.

Linux GCC/Clang host checks, installed consumers, ASan/UBSan with leak detection,
two 3,000-iteration fuzz smoke campaigns and TSan passed. The TSan target includes
blocked concurrent GPU-image operations through the private backend. Fault tests
cover all allocation sites, stream lifecycle, memory preflight, early/late copy
and quantization failures, expert/RNN/fold errors, sync and restoration failures,
retryable cleanup, sticky original diagnostics and invalid numeric auxiliaries.
macOS ARM64 host checks, installed consumers and formatting/lint passed. These
CPU tests do not emulate GPU numerical execution.

Load timings are single warmed-cache validation observations, not benchmarks or
speedup claims. VRAM samples include **both** the Zig and C images and context
fixtures; they are not standalone serving requirements. The planner reports
requested C allocations precisely, independently of sampled runtime overhead.

## Next gate

C forward/prefill/decode, KV-cache/session ownership, cuBLAS lifecycle,
sampling/speculation, cancellation and serving remain unimplemented. This slice
proves upload/packing parity and exercised resource safety, not generated-token
agreement, quality, long-context support or inference throughput.
