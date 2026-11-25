#!/usr/bin/env python3
"""An independent ctypes consumer of ABI negotiation/discovery; no CUDA claim."""
import ctypes as c
import sys

class Bytes(c.Structure):
    _fields_ = [('data', c.c_void_p), ('bytes', c.c_uint64)]

class Out(c.Structure):
    _fields_ = [('struct_size', c.c_uint32), ('written_size', c.c_uint32)]

class Info(c.Structure):
    _fields_ = [('hdr', Out), ('backend_id', Bytes), ('engine_revision', Bytes),
                ('compiled_features', c.c_uint64), ('pointer_bits', c.c_uint32),
                ('limit_scope', c.c_uint32), ('max_prepared_sessions', c.c_uint32),
                ('max_active_runs', c.c_uint32)]

InfoFn = c.CFUNCTYPE(c.c_uint32, c.POINTER(Info), c.c_void_p)
class TablePrefix(c.Structure):
    _fields_ = [('struct_size', c.c_uint32), ('major', c.c_uint32),
                ('minor', c.c_uint32), ('reserved', c.c_uint32), ('backend_info', InfoFn)]

lib = c.CDLL(sys.argv[1])
get = lib.sq_infer_get_api
get.argtypes = [c.c_uint32, c.c_uint32, c.POINTER(c.c_void_p)]
get.restype = c.c_uint32
ptr = c.c_void_p(1)
assert get(2, 0, c.byref(ptr)) == 12 and ptr.value is None
assert get(1, 0, c.byref(ptr)) == 0 and ptr.value
api = c.cast(ptr, c.POINTER(TablePrefix)).contents
assert api.major == 1 and api.struct_size >= c.sizeof(TablePrefix)
info = Info(Out(c.sizeof(Info), 0))
assert api.backend_info(c.byref(info), None) == 0
assert info.hdr.written_size == c.sizeof(Info)
assert c.string_at(info.backend_id.data, info.backend_id.bytes) == b'artalis.squid'
assert info.pointer_bits == 64 and info.max_active_runs == 1
print('ctypes ABI negotiation and capability discovery passed')
