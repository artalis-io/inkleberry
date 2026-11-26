# Pinned upstream reference

`upstream/` contains the unmodified Zig host and build script from
`tensorfish/squid5090` at `d10c0abb2c8990be81c08ba84c66a8333a691b33`.
These files are an optional test oracle, not a second maintained runtime.
Production sources are C11 under `src/`; normal C builds do not invoke Zig.

Original upstream names and hashes remain in `UPSTREAM.json`. Its
`relocated_files` map records where the original files now live. The verifier,
CUDA ABI generator and oracle preparation tools resolve those names through that
map. Original source paths in attribution comments, NOTICE and historical
benchmark receipts refer to their recorded source revision/layout and remain unchanged.

There is only one kernel tree: `upstream/cuda` is a relative symlink to the
repository's `cuda/`. Neither the CUDA kernels nor the reference Zig files were
edited during relocation. The repository [LICENSE](../LICENSE),
[NOTICE](../NOTICE) and [third-party inventory](../THIRD_PARTY.md) still apply.

From the repository root, optional comparison builds use the usual targets:

```sh
make ZIG=/path/to/zig chat-oracle
make ZIG=/path/to/zig ar-oracle
make ZIG=/path/to/zig gpu-weight-oracle
make ZIG=/path/to/zig decode-oracle adaptive-oracle prefix-oracle
```

To build the original upstream executable itself, run from its reference root:

```sh
cd reference/upstream
/path/to/zig build
```

This is for explicit oracle work; do not start its HTTP server as part of the C
runtime. Model files stay external. GPU tests still require the documented idle-
device guard. The current upstream Zig version/build requirements are unchanged.
