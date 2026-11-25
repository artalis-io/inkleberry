#!/usr/bin/env python3
"""Independent byte-accounting oracle transcribed from pinned Zig's closed-form
compute_total_vram_bytes, not from the C allocation recipe. No GPU/model needed."""
import json
import subprocess
import sys

def aligned(n):return (n+255)//256*256

def linear(r,c,k):
    if k==0:return aligned(r*c*2)
    if k==1:return aligned(r*c)+aligned(r*4)
    return aligned(r*c//2)+aligned(r*c//16)

def expected(mode,mtp,force):
    h,q,kv,v=2304,3072,1024,200058
    total=aligned(201024*h*2)+2*aligned(h*2)+linear(v,h,mode)
    total+=sum(aligned((3 if (i+1)%6==0 else 1)*h*4) for i in range(1,24))
    total+=25*aligned(h*4)+2*aligned(24*h*4)
    for i in range(24):
        global_layer=(i+1)%6==0
        total+=(4 if global_layer else 2)*aligned(h*2)
        total+=linear(6552,h,mode)+linear(2*kv,h,mode)+linear(h,q,mode)
        total+=2*aligned(4*kv*2)
        if global_layer:total+=2*aligned(32*h*2)
        total+=2*aligned(h*4*2)+aligned(q*2)+aligned(384*2)+aligned(16*1024*2)
        if 6<=i<18:
            total+=4*(linear(64,2*h,2)+2*linear(h,64,2) if mode==2 else aligned(64*2*h*2)+2*aligned(h*64*2))
        total+=linear(128,h,2) if mode==2 else aligned(128*h*2)
        total+=aligned(128*4)
        for cols in [2304,1152]:
            n=128*h*cols
            total+=aligned(n//2)+aligned(n//16)+aligned(128*4) if i<20 or mode==2 else aligned(n)+aligned(n//32)
    if mtp:
        k=0 if mode==0 or force else 2
        total+=aligned(64)+3*aligned(h*2)+aligned(201024*256*2)+aligned(768*2)+aligned(h*256*2)+aligned(h*2)
        total+=aligned(768*512*2)+2*aligned(768*h*2)+aligned(v*4)+linear(v,256,k)
        total+=3*(3*aligned(h*2)+3*aligned(128*2)+linear(5120,h,k)+linear(h,q,k)+linear(18432,h,k)+linear(h,9216,k)+linear(2048,h,k))
    return total

rows=[]
for mode,name in enumerate(['exact','fp8','nvfp4']):
    for variant in ['ar','mtp','mtp-bf16']:
        mtp=variant!='ar';force=variant=='mtp-bf16'
        p=subprocess.run([sys.argv[1],'weight-plan',name,variant],text=True,capture_output=True,check=True)
        row=json.loads(p.stdout)
        assert row['arena_bytes']==expected(mode,mtp,force),(name,variant,row)
        assert row['staging_bytes']==200058*2304*2 and row['amax_bytes']==4
        assert row['fold_temporary_bytes']==(200058*256*2 if mtp else 0)
        assert row['peak_device_bytes']==sum(row[k] for k in ['arena_bytes','staging_bytes','amax_bytes','fold_temporary_bytes'])
        rows.append({'mode':name,'variant':variant,**row})
print('9 weight layouts match independent pinned-Zig byte accounting')
if len(sys.argv)>2:
    from pathlib import Path
    Path(sys.argv[2]).write_text(json.dumps(rows,indent=2)+'\n')
