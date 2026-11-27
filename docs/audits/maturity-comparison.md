# Software maturity comparison

Reviewed 27 November 2026 against Jev Bush
`85fd51f8c99b1b8b7ea26d33e02413e681a9271d` and the local bitnet.c source at
`ab9e8e8661371ee33fb086980b675aae88246de5`. bitnet.c's working tree already
contained unrelated development changes; those were left untouched. This is a
source/tooling comparison, not a new performance or model-quality benchmark.

Inkleberry already has opaque ownership, allocator/failure tests, explicit ABI
versioning, portable host boundaries, generated amalgamation, C/C++ installed
consumers, deterministic packages, Linux hardening, sanitizer/fuzz gates and
MSVC/clang-cl checks. Rebuilding those systems would add duplication. Jev Bush
and bitnet.c are stronger in several specific areas below.

| Priority | What the other project does better | Inkleberry gap and proportionate next step |
| --- | --- | --- |
| 1 | Jev Bush evaluates a public typed-decisions dataset using `run_openjev_eval.py`, `score_openjev.py`, fixed fixtures and strict-mode checks | The seven semantic tasks are useful smoke coverage, not a representative quality evaluation. Add a modest held-out suite with multiple seeds, longer completed answers, executable code tests and position-varied retrieval; keep task success separate from exact tokens. |
| 2 | bitnet.c has explicit backend/quant/model matrices (`test/backend_matrix.sh`, `test/model_matrix.sh`) and `test_gpu_validate.c` across quant formats | Consolidate the existing IFN execution/precision/device coverage into one capability-and-evidence matrix: supported, experimental, build-only, runtime-tested, pending. Preserve fixed Inkling geometry; don't add a multi-model framework just to match bitnet.c's breadth. |
| 3 | bitnet.c validates many quant/operator combinations against reference calculations; Jev Bush has strict execution-shape invariance tests | IFN has packing probes, selected FP64 checks, graph tests and 84 semantic passes, but independent operator references and adversarial numeric coverage remain uneven. Add targeted MoE/attention/quant scale and boundary-value references, plus longer-context/cache tests. Do not demand cross-architecture bit identity or change kernels before measured evidence. |
| 4 | Jev Bush's concurrency and error-frame checks connect its documented execution model to explicit tests | IFN's process-wide single-active-session contract is already explicit and tested. Extend real GPU fault/recovery coverage (OOM, failed graph capture, stream/context errors, cancellation under long prefill) and compute-sanitizer runs on reserved hardware. CPU mocks and TSan do not certify CUDA safety. |
| 5 | Jev Bush's long-lived `eval` process amortizes model load and exercises state reuse | The semantic runner reloads weights for every case: correct, but expensive on GB10. A small persistent test client could run the same cases with explicit reset and compare fresh-session versus reused-session results. Keep it in tests, separate from scheduling/serving. |

## Architecture judgment

bitnet.c's backend checks enforce useful independent axes: quantization code must
not dispatch on model-family names; core model ownership must not include runtime
implementations; platform runtimes consume policy rather than environment reads.
IFN already separates model geometry/quant policy, host execution, checked CUDA
adapters and launch/resource preparation. Strengthen those narrow boundaries as
features are added rather than importing a generic GPU IR or backend hierarchy.

Jev Bush's CPU backends and concurrent independent sessions are useful capabilities,
but not evidence that IFN should add CPU inference or parallel GPU execution.
Those would be product decisions with memory/scheduling implications. Likewise,
more model families, formats or platforms do not automatically mean better
software engineering for a single-model CUDA library.

Release tooling is not an obvious deficit against these two repositories: IFN
already has versioned ABI fixtures, allowlisted/checksummed source and runtime
artifacts, and extracted consumers. `VERSION` is still `0.2.0-dev`; neither IFN nor
Jev Bush had published GitHub releases in the read-only check. A tagged, fully
qualified release is an operational milestone, not a reason to redesign packaging.

## Recommended order

Finish the four audit fixes, keep their regressions mandatory, then prioritize
held-out semantic coverage and the device/precision evidence matrix. Follow with
focused numerical/failure tests. No new framework, backend abstraction, kernel
rewrite or deployment work is needed for these steps.
