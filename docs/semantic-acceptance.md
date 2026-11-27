# Semantic acceptance

This small model-backed suite measures useful task outcomes independently of
bit-identical token goldens. Kernels, quantization, math policy and existing
benchmark fixtures stay unchanged. Default coverage is greedy AR and adaptive
speculation; select all four profiles for wider coverage. Every case gets a
2,048-token output budget, seed 42, 32K context, NVFP4/MTP weights, FP8 global KV
and graph execution. There are no automatic retries or prompt adjustments.

The seven cases cover two exact arithmetic answers, typed JSON extraction,
a C11 array-maximum function, exact identifier copying, retrieval from a padded
multi-record prompt, and a short autumn paragraph. Retrieval is a small smoke
case, not a maximum-context or multi-needle qualification.

Every response must finish with EOS and contain one complete final-answer
channel. The validator rejects incomplete/truncated responses, malformed text,
duplicate JSON keys, incorrect answers, and obvious repeated eight-word loops
in either reasoning or final text. A prose pass checks word count, paragraph
structure, topic words and lexical diversity. **These are structural smoke
checks, not proof of general coherence or semantic quality.** Review the saved
prose manually; do not turn those heuristics into a broad quality claim.

Generated C must compile with warnings as errors and pass negative, singleton,
extreme-integer and duplicate-value tests. It runs only in a disposable Docker
container with no network, read-only root and inputs, non-root user, dropped
capabilities, bounded CPU/memory/processes, and a writable temporary filesystem.
Timeout cleanup removes the container. Assessment can run on a separate
controller; the GPU host does not need Docker. Only standard-library Python is
used. Public compiler base images are allowed; no model/application image is
published. Do not run generated code directly on the host.

## Run

Use a new external output directory each time, an approved idle GPU and the
existing guard. Native Windows additionally requires the reviewed desktop
reservation described in `experimental-ampere.md`. Paths below are examples;
use the actual CUDA runtime directory (`bin/x64` for the tested Windows toolkit),
selected build directory and `.exe` suffix where applicable.

```sh
make -j4 .build/semantic
python3 tools/semantic_acceptance.py run \
  --binary .build/semantic --model /external/model \
  --gpu GPU-APPROVED-ID --cuda-runtime /usr/local/cuda/lib64 \
  --source-revision "$(git rev-parse HEAD)" \
  --profiles ar fixed adaptive sampled --out /external/new-semantic-run
```

Record a clean exact source revision when building. The receipt records that
revision, executable/model checksums, prompt hashes and selected profiles.
Raw special-token output and termination statistics are retained per case. A
failed GPU process stops the campaign; it cannot become a successful assessment.
Run completion means collection finished, **not** that answers passed.

```sh
# Explicit preparation on the assessment controller; assessment never pulls.
docker pull gcc:14.2.0
python3 tools/semantic_acceptance.py assess \
  --input /external/new-semantic-run --out /external/semantic-assessment.json \
  --code-image gcc:14.2.0
```

The assessment records the resolved compiler-image ID. Pin `--code-image` to a
repository digest for repeated formal qualification. No application images,
customer data or model bytes belong in this repository. Copy completed receipts
and text outputs between machines without copying the model. Do not claim a GPU
passed based on CPU CI: `make check` compiles the client and tests validator
failure paths without running inference or generated code. Separate task success
from existing numerical, performance and distribution-preservation gates.
