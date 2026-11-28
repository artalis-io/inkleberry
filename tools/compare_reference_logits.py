#!/usr/bin/env python3
"""Three conditional-prefix reference comparisons, explicitly not precision-equivalence scores."""
import argparse
import hashlib
import json
import struct
from pathlib import Path

from collect_reference_logits import contexts
from compare_boundary_logits import floats, probabilities, vector_metrics


def compare(reference,control,boundaries,divergences):
    manifest=json.loads((reference/'manifest.json').read_text())
    summary=json.loads((reference/'summary.json').read_text())
    expected=contexts(boundaries/'rtx',divergences['differences'])
    if manifest['contexts']!=expected:raise ValueError('Reference input IDs/prefixes differ')
    if summary.get('passed') is not True:raise ValueError('Reference run did not pass')
    index={}
    for row in summary['runs']:
        key=row['case'],row['run']
        if key in index:raise ValueError('Duplicate reference repetition')
        index[key]=row
    if set(index)!={(c,r) for c in (4,5,6) for r in (0,1)}:
        raise ValueError('Missing reference repetitions')
    controls=[json.loads(line) for line in (control/'control.jsonl').read_text().splitlines()
              if not line.startswith('Idle GPU preflight:')]
    control_index={}
    for row in controls:
        key=row['case'],row['run']
        if key in control_index:raise ValueError('Duplicate control repetition')
        control_index[key]=row
    if set(control_index)!=set(index):raise ValueError('Incomplete native control')
    for name in ('rtx','gb10'):
        meta=json.loads((boundaries/name/'metadata.json').read_text())
        if meta['model_sha256']!=manifest['model_sha256']:raise ValueError('Model files differ')
    comparisons=[]
    for item in expected:
        c=item['case']
        vectors=[]
        for run in (0,1):
            path=reference/f'case-{c}-run-{run}.logits.f32'
            v=floats(path,200058)
            if hashlib.sha256(path.read_bytes()).hexdigest()!=index[c,run]['sha256']:
                raise ValueError('Reference logit hash mismatch')
            if max(range(len(v)),key=v.__getitem__)!=index[c,run]['top1']:
                raise ValueError('Reference top token mismatch')
            vectors.append(v)
        if vectors[0]!=vectors[1]:raise ValueError('Nonrepeatable reference')
        ref=vectors[0];ref_top=sorted(range(len(ref)),key=lambda i:(-ref[i],i))[:10]
        p=probabilities(ref,1)
        row=dict(case=c,prompt_tokens=len(item['prompt']),preceding_output_tokens=len(item['prefix']),
                 reference_top10=[dict(token=i,logit=ref[i],probability=p[i]) for i in ref_top],
                 reference_margin=ref[ref_top[0]]-ref[ref_top[1]],configurations={})
        for name,path in [('production_rtx',boundaries/f'rtx/case-{c}-profile-0.logits.f32'),
                          ('production_gb10',boundaries/f'gb10/case-{c}-profile-0.logits.f32'),
                          ('source_weight_bf16kv_rtx',control/f'case-{c}.logits.f32')]:
            v=floats(path,200058);top=max(range(len(v)),key=v.__getitem__)
            if name=='source_weight_bf16kv_rtx':
                expected_input_sha=hashlib.sha256(b''.join(struct.pack('<i',t) for t in item['prompt']+item['prefix'])).hexdigest()
                for run in (0,1):
                    r=control_index[c,run]
                    if (r['top1']!=top or r['prompt_tokens']!=len(item['prompt']) or
                            r['total_tokens']!=len(item['prompt'])+len(item['prefix']) or
                            r['input_sha256']!=expected_input_sha):
                        raise ValueError('Control boundary or selected token mismatch')
            row['configurations'][name]=dict(top1=top,top1_matches_reference=top==ref_top[0],
                                             metrics=vector_metrics(v,ref))
        comparisons.append(row)
    return dict(format_version=1,comparison_validation_passed=True,cases=comparisons,
                reference_revision=manifest['reference_revision'],
                boundary='Three preselected divergent AR prefixes; not a held-out quality benchmark. '
                         'Reference dequantized BF16, production additionally quantized/FP8 KV. '
                         'Control changes both linears and KV; expert arithmetic remains different. '
                         'Do not infer exact equivalence, isolated cause or overall device accuracy ranking.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('reference','control','boundaries','divergences','out'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    result=compare(a.reference,a.control,a.boundaries,json.loads(a.divergences.read_text()))
    with a.out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')


if __name__=='__main__':main()
