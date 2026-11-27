# Persistent-worker qualification

The Linux client `test/worker_lifetime.c` consumes only installed public inference
headers and the installed provider. It loads the model once, establishes two
same-device greedy raw-token baselines using fresh sessions, closes those sessions,
and keeps a third session prepared throughout the campaign. It never clears that
session's cache between ordinary requests. Kernels, goldens and numerical policy
are unchanged. The client explicitly selects `nvfp4-mtp`, the default
`fp8-graph` session, adaptive seven-token speculation with lookup, threshold 0.8,
greedy decoding, seed 42, and a 128-token output budget. It qualifies this profile;
it does not claim the full profile/context matrix.

Each seeded cycle alternates scope A/B and randomly selects a suffix and one of
three disruptions. Five requests exercise: cross-scope miss, same-scope/shared-
system-prefix hit with changed suffix, cancellation, recovery miss, recovery hit.
All uninterrupted requests must return OK, match the corresponding clean-session
baseline tokens exactly, and report the expected cache reuse. Correct tokens with
forbidden cross-scope reuse fail qualification. Late cancellation must report
TOO_LATE. JSONL is flushed before and after each request, preserving the failing
sequence even if the worker hangs. The independent history validator recomputes
the seeded schedule and rejects missing results, unexpected errors, token changes,
wrong cache behavior and unsuccessful/incomplete campaigns.

## Failure disposition

| Injection | Required outcome | Caller action |
| --- | --- | --- |
| Callback cancellation during token emission | CANCELLED, cancellation finish, at least one token | Submit next request on the same session; probe must pass with cache miss |
| Cancellation after RUNNING, before first output | Accepted, CANCELLED, no emitted tokens | Same session, recovery probe |
| Cancellation before run | Accepted, CANCELLED, no emitted tokens | Same session, recovery probe |
| Any uninjected request error, token/cache mismatch, cleanup failure | Qualification failure | Stop campaign; retain evidence; do not hide it by recreating a session |
| Timeout, interrupt, native crash | Qualification failure, missing/failed terminal record | Supervisor terminates its owned process group; retain history before restarting |

**Prefill coverage limit:** public `prompt_tokens_processed` is updated after the
native call returns. RUNNING with zero output does not prove execution entered
prefill. This client therefore reports pre-output cancellation, not proven mid-
prefill cancellation. Existing lower-level prefill recovery tests remain necessary.
No poisoned-session/device-loss recovery is injected or claimed here. Those cases
require dedicated fault injection and explicit session/process recovery tests;
ordinary failures in this campaign always fail, regardless of reported disposition.

## Build and run on Linux

From a clean pinned source checkout, build/install locally into an isolated prefix:

```sh
PREFIX="$PWD/.build/worker-sdk"
make -j4 CUDA_ARCH=120a install-infer-backend PREFIX="$PREFIX"
make worker-lifetime PREFIX="$PREFIX"
make worker-lifetime-check
```

The prefix is an absolute path. No installation into `/usr/local` is necessary.
Use the installed provider file appropriate to the platform (Linux `.so.1`).
Check its actual filename with `ls "$PREFIX/lib"`.

Short native campaign (12 cycles / 60 stress requests, plus two clean baselines):

```sh
python3 tools/run_worker_lifetime.py \
  --source-revision "$(git rev-parse HEAD)" \
  --executable .build/worker-lifetime \
  --provider "$PREFIX/lib/libinkleberry_backend.so.1" \
  --model /data/models/safetensors/inkling-2-mini-c9d028302cd0 \
  --gpu GPU-5a215cb3-57b0-0a6e-d511-e90bb05c0e05 \
  --library-path /usr/local/cuda/lib64 \
  --seed 42 --cycles 12 --output /data/scratch/ifn-worker-short
```

Check the UUID and workload inventory immediately before running. The guard refuses
allocated GPUs. It does not reserve a device against an external scheduler; arrange
exclusive availability for the campaign. Never stop unrelated services.

After a successful short native run, use the same command with
`--cycles 10000000 --seconds 28800` and a new output directory for an eight-hour
campaign. Duration starts after baseline preparation and stops after the current
five-request cycle. The supervisor allows 20 minutes for loading/cleanup and kills
only its owned process group on timeout/interruption. Keep the controller attached
or use an operator-managed session; this command does not install cron/systemd jobs.
The ordinary smoke guard's one-hour maximum is preserved; only explicit
`INKLEBERRY_TEST_LONG_RUN=1` permits a bounded maximum of 24 hours.

For source archives, capture `qualification_source.capture` from the clean checkout
and pass `--source-manifest PATH`. This is operator-attested source inventory, not a
signed build attestation. The runner hashes client, provider and three model files,
records the command, and verifies source/client/provider identity after the run.
Inputs must remain immutable; the runner does not protect model files from another
process modifying them during execution.

## Evidence and interpretation

- `manifest.json`: source inventory/revision, executable/provider/model hashes,
  seed, requested cycles and duration, GPU UUID.
- `history.jsonl`: clean baseline tokens; ordered request beginnings/results;
  cache counts, status, tokens, process RSS and callback timing.
- `device.jsonl`: five-second samples of UUID, GPU name, driver, allocated memory,
  utilization and power. Allocation is device-wide, not per-client accounting.
- `stderr.log`: native diagnostics/guard failures, never discarded.
- `summary.json`: success/failure, exercised modes, comparable warm probes grouped
  by suffix and scope, with min/median/max RSS, run time, TTFT and decode duration.

```sh
python3 tools/check_worker_lifetime.py /data/scratch/ifn-worker-short/history.jsonl
```

Reproduce the failing cycle by using the recorded seed and setting cycles to the
failing cycle index plus one. Earlier cycles are intentionally retained: the bug
may depend on the full ordered history. Duration-limited campaigns record their
actual completed cycle count. Interrupted campaigns are failures, not partial passes.

Compare warm step-four probes only within the same suffix/scope. Raw outputs allow
computing `(tokens - 1) / decode_seconds`; one-token probes have no meaningful decode
rate. RSS includes model mappings and allocator/runtime behavior, so a maximum is
not proof of a leak. No arbitrary memory/performance acceptance threshold is imposed:
review temporal growth and like-for-like distributions before release. Numerical
identity here is same-device regression evidence, not cross-device quality evidence.

## Execution record

Implemented CPU adversarial tests pass. Native GPU and overnight qualification have
**not run**: discovery found both core-1 RTX PRO 6000 GPUs occupied by a two-GPU vLLM
worker (~55 GB each). No workload was stopped. Proven mid-prefill cancellation,
poisoned-session recovery and device-loss recovery remain unqualified in this client.
Spark first-divergent-token/logit analysis is a separate next milestone.
