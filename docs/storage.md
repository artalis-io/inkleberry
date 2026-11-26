# Checkpoint storage

The file-based safetensors reader already uses `mmap` with `PROT_READ` and
`MAP_PRIVATE`. It maps a validated regular file and closes the descriptor after
mapping. Tensor lookup returns borrowed spans into that mapping; it does not copy
the complete checkpoint or individual payloads into an additional host buffer.
The mapping stays alive until `sq_safetensors_close` or its owning checkpoint closes.

This avoids payload-sized host allocations and uses the operating system's page
cache. It does not guarantee a particular cold-load time: storage, page faults,
GPU transfers and quantization/packing still contribute to loading. GPU upload is
a separate operation and still transfers/represents the weights in device memory.
This is not a GPU-direct-storage path or zero-copy device inference.

Keep source files immutable while mapped, including their size. Modifying or
truncating a mapped file can invalidate borrowed views and cause an OS fault.
Do not retain spans after closing their owner. The memory-input reader borrows
caller-owned bytes instead; its lifetime is explicit in the public header.

The tokenizer also maps its input while loading, then owns its decoded tables and
releases the input mapping. Model loading validates roles/shapes before upload;
failed uploads retain checkpoint mappings until native cleanup succeeds.

`make mmap-check` exercises a sparse 64 MiB synthetic tensor with less than 2 MiB
of injected heap available. It checks payload access, lifetime after unlink and
complete allocation cleanup. Linux additionally checks the live mapping's inode
and `r--p` protection using `/proc/self/maps`. The same regression runs against the
modular library and generated embedding implementation. It uses no private model
or dataset and is a memory/lifetime check, not a throughput benchmark.
