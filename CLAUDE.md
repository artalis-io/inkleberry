# Inkleberry Finn — Claude Code

Read and follow [AGENTS.md](AGENTS.md). It is the shared engineering contract and
source of truth for architecture, naming, ownership, dependency pins and testing.

Start by checking the working tree and current branch. Preserve existing work.
Use `rg` for discovery and inspect the actual headers/build rules before editing.
Keep changes focused and explain the behavior, validation and remaining limits.

- Public namespace: `ifn_` / `IFN_`; project: Inkleberry Finn / `inkleberry`.
- Edit modular sources; regenerate the two-file `dist/ifn.h` + `dist/ifn.c` SDK.
- Preserve pinned CUDA/reference bytes and legal notices. Internal upstream CUDA
  names are intentional; do not mass-rename retained dependencies.
- Check C and C++ consumers, install behavior and extracted release artifacts.
- Use injected allocators, checked arithmetic and recoverable errors.
- Run the relevant checks in AGENTS.md; distinguish host passes from native GPU
  qualification. Never claim generation correctness from a successful compilation.
- Do not disturb unrelated GPU workloads or system services. Native tests require
  an approved idle device through `INKLEBERRY_TEST_GPU_UUID` and the guard script.
- No private checkpoints, credentials, generated binaries or bulk results in Git.

No model or service deployment is implied by a source-code maintenance request.
