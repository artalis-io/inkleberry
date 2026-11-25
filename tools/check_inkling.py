#!/usr/bin/env python3
"""Verify bound text roles against an independent external checkpoint header.

Does not dereference/copy weight payloads. Vision tensors must stay unbound.
"""
import argparse
import ctypes as c
import json
from pathlib import Path
import struct
from check_checkpoint import Tensor

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('library',type=Path)
    p.add_argument('model',type=Path)
    args=p.parse_args()
    lib=c.CDLL(str(args.library.resolve()))
    lib.sq_inkling_checkpoint_open.argtypes=[c.c_char_p,c.c_char_p,c.c_void_p,c.POINTER(c.c_void_p)]
    lib.sq_inkling_checkpoint_open.restype=c.c_uint32
    lib.sq_inkling_checkpoint_close.argtypes=[c.c_void_p]
    lib.sq_inkling_checkpoint_count.argtypes=[c.c_void_p]
    lib.sq_inkling_checkpoint_count.restype=c.c_size_t
    lib.sq_inkling_checkpoint_at.argtypes=[c.c_void_p,c.c_size_t,c.POINTER(c.c_uint32),c.POINTER(c.POINTER(Tensor))]
    lib.sq_inkling_checkpoint_at.restype=c.c_uint32
    lib.sq_inkling_checkpoint_data.argtypes=[c.c_void_p,c.c_size_t,c.POINTER(c.c_void_p),c.POINTER(c.c_size_t)]
    lib.sq_inkling_checkpoint_data.restype=c.c_uint32
    lib.sq_last_error.restype=c.c_char_p
    paths=[args.model/'model-00001-of-00001.safetensors',args.model/'mtp.safetensors']
    expected={};ignored=0
    for source,path in enumerate(paths):
        with path.open('rb') as f:
            n,=struct.unpack('<Q',f.read(8));header=json.loads(f.read(n))
        for name,entry in header.items():
            if name=='__metadata__':continue
            if source==0 and name.startswith('model.visual.'):
                ignored+=1;continue
            expected[source,name]=entry
    handle=c.c_void_p()
    if lib.sq_inkling_checkpoint_open(bytes(paths[0]),bytes(paths[1]),None,c.byref(handle)):
        raise SystemExit(lib.sq_last_error().decode())
    seen=set();count=[0,0];dtype=['U8','BF16','F32','F8_E4M3']
    try:
        for i in range(lib.sq_inkling_checkpoint_count(handle)):
            source=c.c_uint32();tensor=c.POINTER(Tensor)();data=c.c_void_p();size=c.c_size_t()
            if lib.sq_inkling_checkpoint_at(handle,i,c.byref(source),c.byref(tensor)):
                raise SystemExit('Missing bound metadata')
            t=tensor.contents;name=t.name.decode();key=source.value,name;e=expected[key]
            begin,end=e['data_offsets']
            if (dtype[t.dtype],list(t.shape[:t.rank]),t.offset,t.bytes)!=(e['dtype'],e['shape'],begin,end-begin):
                raise SystemExit('Bound metadata mismatch')
            if lib.sq_inkling_checkpoint_data(handle,i,c.byref(data),c.byref(size)) or not data or size.value!=t.bytes:
                raise SystemExit('Invalid borrowed payload span')
            if key in seen:raise SystemExit('Duplicate binding')
            seen.add(key);count[source.value]+=1
        if seen!=set(expected):raise SystemExit('Role coverage mismatch')
    finally:
        lib.sq_inkling_checkpoint_close(handle)
    print(json.dumps({'main_bound':count[0],'mtp_bound':count[1],
                      'vision_unbound':ignored,'all_metadata_matched':True,
                      'payloads_copied':False,'gpu_used':False},indent=2))

if __name__=='__main__':main()
