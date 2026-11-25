#!/usr/bin/env python3
"""Synthetic sparse payloads exercise real file admission without model bytes.

The files have the architecture's logical sizes, but holes consume no payload
storage and are never read by metadata binding. No real weights are generated.
"""
from copy import deepcopy
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

binary = sys.argv[1]
specs = json.loads(subprocess.check_output([binary,'--schema'],text=True))
names = ['U8','BF16','F32','F8_E4M3']; widths=[1,2,4,1]
headers=[{},{}]; sizes=[0,0]
for s in specs:
    n=widths[s['dtype']]
    for d in s['shape']:n*=d
    source=s['source'];start=sizes[source];sizes[source]+=n
    headers[source][s['name']]={'dtype':names[s['dtype']], 'shape':s['shape'], 'data_offsets':[start,start+n]}

def write(path, header, payload):
    data=json.dumps(header,separators=(',',':')).encode()
    with path.open('wb') as f:
        f.write(struct.pack('<Q',len(data)));f.write(data)
        f.truncate(8+len(data)+payload)
        # Numeric auxiliaries for the private GPU-lifecycle simulation only.
        for name,entry in header.items():
            if name.endswith(('_weight_amax','_weight_outer_scale_denominator')):
                start,end=entry['data_offsets']
                f.seek(8+len(data)+start)
                f.write(struct.pack('<f',1.0)*((end-start)//4))
    # Fail if the platform materializes holes rather than creating sparse files.
    if path.stat().st_blocks*512>2*1024*1024:
        raise SystemExit('Filesystem did not create a sparse fixture; refusing bulk fixture storage')

def run(args):
    r=subprocess.run(args,text=True,capture_output=True,timeout=30)
    if r.returncode or 'Sanitizer' in r.stderr or 'runtime error:' in r.stderr:
        raise SystemExit(r.stdout+r.stderr)

with tempfile.TemporaryDirectory(prefix='sq-schema-') as folder:
    probe=Path(folder)/'sparse-probe'
    with probe.open('wb') as f:f.truncate(16*1024*1024)
    if probe.stat().st_blocks*512>4096:
        raise SystemExit('Sparse files are required for metadata-only model fixtures')
    probe.unlink()
    paths=[Path(folder)/'main',Path(folder)/'mtp']
    for i in range(2):write(paths[i],headers[i],sizes[i])
    run([binary,*map(str,paths)])
    if len(sys.argv)>2:
        for mtp,expected in [(True,876),(False,822)]:
            args=[sys.argv[2],'validate-model',str(paths[0])]+([str(paths[1])] if mtp else [])
            value=json.loads(subprocess.check_output(args,text=True))
            if value['bound_tensors']!=expected or value['mtp']!=mtp or value['device_upload']:
                raise SystemExit('CLI binding contract mismatch')
    if len(sys.argv)>3:
        run([sys.argv[3],*map(str,paths)])
        # Corrupt only synthetic auxiliary scalars; original data is never used.
        for field,value in [('model.llm.layers.0.mlp.experts.w13_weight_outer_scale_denominator',0.0),
                            ('model.llm.layers.0.mlp.experts.w13_weight_amax',float('nan'))]:
            data=json.dumps(headers[0],separators=(',',':')).encode()
            offset=headers[0][field]['data_offsets'][0]
            with paths[0].open('r+b') as f:f.seek(8+len(data)+offset);f.write(struct.pack('<f',value))
            run([sys.argv[3],*map(str,paths),'--invalid-numerics'])
            write(paths[0],headers[0],sizes[0])
    # Same-byte-count permutations must reach, and fail, model-role validation.
    cases=[]
    def change(name,source,mutate):
        h=deepcopy(headers);mutate(h[source]);cases.append((name,h))
    change('embed-transpose',0,lambda h:h['model.llm.embed.weight'].update(shape=[2304,201024]))
    change('embed-rank',0,lambda h:h['model.llm.embed.weight'].update(shape=[2304*201024]))
    change('expert-dtype',0,lambda h:h['model.llm.layers.20.mlp.experts.w13_weight'].update(dtype='U8'))
    change('missing-role',0,lambda h:h.update({'model.visual.not_an_embedding':h.pop('model.llm.embed.weight')}))
    change('unknown-role',0,lambda h:h.update({'model.llm.unknown':{'dtype':'U8','shape':[0],'data_offsets':[0,0]}}))
    change('extra-mtp',1,lambda h:h.update({'unknown':{'dtype':'U8','shape':[0],'data_offsets':[0,0]}}))
    change('mtp-transpose',1,lambda h:h['markov_head.rnn_proj.weight'].update(shape=[5120,768]))
    change('local-convolution-layout',0,lambda h:h['model.llm.layers.0.attn.k_sconv.weight'].update(shape=[4,1,1024]))
    change('global-convolution-layout',0,lambda h:h['model.llm.layers.5.attn.k_sconv.static_w'].update(shape=[1024,4]))
    change('controller-shape',0,lambda h:h['model.llm.layers.6.controllers.attn.down_weight'].update(shape=[4608,64]))
    for name,h in cases:
        for i in range(2):write(paths[i],h[i],sizes[i])
        run([binary,'--invalid',*map(str,paths)])
    # Vision tensors are deliberately ignored, including zero-byte extra ones.
    valid=deepcopy(headers)
    valid[0]['model.visual.unused']={'dtype':'U8','shape':[0],'data_offsets':[0,0]}
    for i in range(2):write(paths[i],dict(reversed(list(valid[i].items()))),sizes[i])
    run([binary,*map(str,paths)])
    run([binary,'--invalid',str(paths[0]),str(Path(folder)/'missing-mtp')])
print('876-role binding, sparse payloads, 10 same-size drift cases, optional MTP failure, reordered headers and vision exclusion passed')
