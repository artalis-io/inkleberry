#!/usr/bin/env python3
"""Compare external diagnostic traces without declaring numerical tolerances."""
import argparse
import array
import json
import math
from pathlib import Path
import sys

STAGES = {'embedding':2304, 'attention_input':2304, 'qrgs':6552,
          'key_projection':1024, 'attention_output_projection':2304,
          'router_logits':128, 'router_weights':8, 'moe_output':2304,
          'final_norm':2304, 'logits':200058}

def load(root, batch):
    run=json.loads((root/'run.json').read_text()) if (root/'run.json').exists() else {'token_count':8,'trace_from':0}
    count,start=run['token_count'],run['trace_from']
    if type(count) is not int or type(start) is not int or not 0<=start<count<=128:raise ValueError('Invalid trace range')
    rows = [json.loads(s) for s in (root/'trace.jsonl').read_text().splitlines()]
    index = {}
    for row in rows:
        if row['batch'] != batch: continue
        if row['stage'] not in STAGES or row['width'] != STAGES[row['stage']]:
            raise ValueError('Unknown stage/shape')
        key = (row['position'], row['visit'], row['stage'])
        if key in index: raise ValueError('Duplicate trace stage')
        if not start<=key[0]<count or not 0<=key[1]<=48: raise ValueError('Invalid trace position/visit')
        data = (root/f"b{batch}-p{key[0]}-v{key[1]}-{key[2]}.bin").read_bytes()
        if len(data) != 4*row['width']: raise ValueError('Incomplete tensor payload')
        index[key] = data
    prefill=((count-1)//8)*8 if run.get('generation_probe') else 0
    expected={(p,v,stage) for p in range(max(start,prefill),count) for v in range(48)
              for stage in STAGES if stage not in ('embedding','final_norm','logits')}
    expected.update((p,0,'embedding') for p in range(start,count))
    expected.update((p,48,'final_norm') for p in range(start,count))
    if prefill:
        expected.update((p,v,stage) for p in range(prefill) for v in range(48)
                        for stage in STAGES if stage not in ('embedding','final_norm','logits','moe_output'))
    else:
        expected.update((p,48,'logits') for p in range(start,count))
    if index.keys()!=expected: raise ValueError('Incomplete trace coverage')
    return index

def floats(data):
    values=array.array('f'); values.frombytes(data)
    if sys.byteorder!='little':values.byteswap()
    if not all(math.isfinite(x) for x in values): raise ValueError('Non-finite trace')
    return values

def compare(left, right, left_batch=1, right_batch=1):
    left_run,right_run=left/'run.json',right/'run.json'
    if left_run.exists()!=right_run.exists():raise ValueError('Run identity availability mismatch')
    if left_run.exists():
        x,y=json.loads(left_run.read_text()),json.loads(right_run.read_text())
        x.pop('generated_token',None);y.pop('generated_token',None)
        if x!=y:raise ValueError('Input/configuration identity mismatch')
    a,b=load(left,left_batch),load(right,right_batch)
    if a.keys()!=b.keys():raise ValueError('Trace identity mismatch')
    order={stage:i for i,stage in enumerate(STAGES)}
    results=[]
    for key in sorted(a,key=lambda k:(k[0],k[1],order[k[2]])):
        x,y=floats(a[key]),floats(b[key])
        if a[key]==b[key]:continue
        delta=[abs(u-v) for u,v in zip(x,y)]
        worst=max(range(len(delta)),key=delta.__getitem__)
        row=dict(zip(('position','visit','stage'),key))
        row.update(max_abs=delta[worst],rms=math.sqrt(sum(v*v for v in delta)/len(delta)),
                   different_values=sum(u!=v for u,v in zip(x,y)),worst_index=worst,
                   left=x[worst],right=y[worst])
        if key[2]=='logits':
            ia=max(range(len(x)),key=x.__getitem__);ib=max(range(len(y)),key=y.__getitem__)
            row.update(left_top1=ia,right_top1=ib,top1_equal=ia==ib)
        results.append(row)
    span_a=[json.loads(s) for s in (left/'spans.jsonl').read_text().splitlines()]
    span_b=[json.loads(s) for s in (right/'spans.jsonl').read_text().splitlines()]
    if len(span_a)!=len(span_b):raise ValueError('Packed span/view count mismatch')
    packing=[{'left':x,'right':y} for x,y in zip(span_a,span_b) if x!=y]
    return {'left_batch':left_batch,'right_batch':right_batch,'packed_records':len(span_a),
            'packing_differences':packing,'compared_tensors':len(a),'different_tensors':len(results),
            'first_difference':results[0] if results else None,'differences':results,
            'numerical_acceptance_established':False}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('left',type=Path);p.add_argument('right',type=Path)
    p.add_argument('--left-batch',type=int,choices=[0,1,2,4,8],default=1)
    p.add_argument('--right-batch',type=int,choices=[0,1,2,4,8],default=1)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    report=compare(a.left,a.right,a.left_batch,a.right_batch)
    a.out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='differences' and k!='packing_differences'}))
if __name__=='__main__':main()
