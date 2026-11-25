#!/usr/bin/env python3
"""Compare every C tensor descriptor with independent Python JSON parsing.

External immutable checkpoint paths only. Never loads payloads into host RAM or
prints weights. Run a non-sanitized shared library under ordinary Python.
"""
import argparse
import ctypes as c
import hashlib
import json
import struct
from pathlib import Path

class Tensor(c.Structure):
    _fields_ = [('name', c.c_char * 256), ('dtype', c.c_uint32),
                ('rank', c.c_uint32), ('shape', c.c_uint64 * 8),
                ('offset', c.c_uint64), ('bytes', c.c_uint64)]

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('library', type=Path)
    p.add_argument('checkpoints', type=Path, nargs='+')
    args = p.parse_args()
    lib = c.CDLL(str(args.library.resolve()))
    lib.sq_safetensors_open.argtypes = [c.c_char_p, c.c_void_p, c.POINTER(c.c_void_p)]
    lib.sq_safetensors_open.restype = c.c_uint32
    lib.sq_safetensors_count.argtypes = [c.c_void_p]
    lib.sq_safetensors_count.restype = c.c_size_t
    lib.sq_safetensors_at.argtypes = [c.c_void_p, c.c_size_t, c.POINTER(c.POINTER(Tensor))]
    lib.sq_safetensors_at.restype = c.c_uint32
    lib.sq_safetensors_find.argtypes = [c.c_void_p, c.c_char_p, c.c_size_t, c.POINTER(c.POINTER(Tensor))]
    lib.sq_safetensors_find.restype = c.c_uint32
    lib.sq_safetensors_close.argtypes = [c.c_void_p]
    lib.sq_last_error.restype = c.c_char_p
    dtype = {'U8': 0, 'BF16': 1, 'F32': 2, 'F8_E4M3': 3}
    results = []
    for path in args.checkpoints:
        with path.open('rb') as f:
            n, = struct.unpack('<Q', f.read(8))
            if n > 16 * 1024 * 1024:
                raise SystemExit('Header exceeds bounded reader limit')
            raw = f.read(n)
            header = json.loads(raw)
        expected = {k: v for k, v in header.items() if k != '__metadata__'}
        handle = c.c_void_p()
        if lib.sq_safetensors_open(bytes(path), None, c.byref(handle)):
            raise SystemExit(lib.sq_last_error().decode())
        try:
            if lib.sq_safetensors_count(handle) != len(expected):
                raise SystemExit('Tensor count mismatch')
            seen = set()
            for i in range(len(expected)):
                item = c.POINTER(Tensor)()
                if lib.sq_safetensors_at(handle, i, c.byref(item)):
                    raise SystemExit('Enumeration failed')
                t = item.contents
                name = t.name.decode('utf-8')
                e = expected[name]
                begin, end = e['data_offsets']
                if (t.dtype, list(t.shape[:t.rank]), t.offset, t.bytes) != (
                        dtype[e['dtype']], e['shape'], begin, end - begin):
                    raise SystemExit('Descriptor mismatch: ' + name)
                found = c.POINTER(Tensor)()
                key = name.encode()
                if lib.sq_safetensors_find(handle, key, len(key), c.byref(found)) or c.addressof(found.contents) != c.addressof(t):
                    raise SystemExit('Lookup mismatch: ' + name)
                seen.add(name)
            if seen != set(expected):
                raise SystemExit('Enumeration coverage mismatch')
        finally:
            lib.sq_safetensors_close(handle)
        results.append({'file': path.name, 'file_bytes': path.stat().st_size,
                        'header_sha256': hashlib.sha256(raw).hexdigest(),
                        'tensor_descriptors_matched': len(expected)})
    print(json.dumps({'checkpoint_metadata_validation': results}, indent=2))

if __name__ == '__main__':
    main()
