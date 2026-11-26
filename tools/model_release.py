#!/usr/bin/env python3
"""Explicit model identities, deterministic golden tokens and performance gates.
Uses external local assets only; never downloads, mutates models or selects GPUs.
"""
import argparse,hashlib,json,math,platform,statistics,struct,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
FILES=['tokenizer.json','model-00001-of-00001.safetensors','mtp.safetensors']
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def records(path):
    rows=[json.loads(s) for s in path.read_text().splitlines() if s.startswith('{')]
    groups={}
    for r in rows:
        if r['warmup']:continue
        assert r['generated']==len(r['tokens']) and r['generated']>0
        assert 0<r['first_token_ns']<=r['elapsed_ns']
        groups.setdefault(str(r['case'])+'/'+r['profile'],[]).append(r)
    if len(groups)!=28:raise ValueError('Expected all 28 workload/profile groups')
    for key,group in groups.items():
        if len(group)<5:raise ValueError('At least five measured repetitions required: '+key)
        if len({tuple(r['tokens']) for r in group})!=1:raise ValueError('Non-deterministic generation: '+key)
    return groups

def summary(group):
    result={}
    metrics={'elapsed_ns':[r['elapsed_ns'] for r in group],
             'first_token_ns':[r['first_token_ns'] for r in group]}
    if group[0]['generated']>1:
        metrics['decode_tps']=[(r['generated']-1)*1e9/(r['elapsed_ns']-r['first_token_ns']) for r in group if r['elapsed_ns']>r['first_token_ns']]
    for k,values in metrics.items():
        result[k]={'median':statistics.median(values),'min':min(values),'max':max(values),
                   'cv':statistics.pstdev(values)/statistics.mean(values)}
    return result

def compare(before,after,latency_tolerance=.10,decode_tolerance=.05):
    base,candidate=records(before),records(after);report={'quality_equal':True,'performance_passed':True,'cases':{}}
    for key,group in base.items():
        other=candidate[key]
        if group[0]['tokens']!=other[0]['tokens'] or group[0]['prompt_tokens']!=other[0]['prompt_tokens']:
            raise ValueError('Output/prompt token regression: '+key)
        a,b=summary(group),summary(other);checks={}
        for metric in a:
            ratio=b[metric]['median']/a[metric]['median']
            # Very short generations are TTFT tests, not stable decode throughput benchmarks.
            checked=metric!='decode_tps' or group[0]['generated']>=16
            passed=not checked or (ratio>=1-decode_tolerance if metric=='decode_tps' else ratio<=1+latency_tolerance)
            checks[metric]={'ratio':ratio,'checked':checked,'passed':passed}
            report['performance_passed'] &= passed
        report['cases'][key]={'baseline':a,'candidate':b,'checks':checks,'generated':group[0]['generated']}
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    m=sub.add_parser('manifest');m.add_argument('model',type=Path);m.add_argument('--out',type=Path,required=True)
    g=sub.add_parser('golden');g.add_argument('records',type=Path);g.add_argument('--manifest',type=Path,required=True);g.add_argument('--out',type=Path,required=True)
    v=sub.add_parser('verify');v.add_argument('records',type=Path);v.add_argument('--golden',type=Path,required=True)
    c=sub.add_parser('compare');c.add_argument('before',type=Path);c.add_argument('after',type=Path);c.add_argument('--out',type=Path,required=True);c.add_argument('--latency-tolerance',type=float,default=.10);c.add_argument('--decode-tolerance',type=float,default=.05)
    a=p.parse_args()
    if a.command=='manifest':
        files={n:digest(a.model/n) for n in FILES}
        identity=hashlib.sha256(b'ifn-artifacts-v1'+b''.join(bytes.fromhex(files[n]) for n in FILES)).hexdigest()
        a.out.write_text(json.dumps({'schema_version':1,'model_family':'inkling-2-mini','files':files,'artifact_fingerprint':identity},indent=2)+'\n')
    elif a.command=='golden':
        groups=records(a.records)
        manifest=json.loads(a.manifest.read_text())
        manifest['cases']={k:{'prompt_tokens':v[0]['prompt_tokens'],'tokens':v[0]['tokens']} for k,v in sorted(groups.items())}
        manifest['workload_revision']='native-benchmark-v1';manifest['warmup_per_case']=1;manifest['measured_per_case']=5
        a.out.write_text(json.dumps(manifest,indent=2)+'\n')
    elif a.command=='verify':
        golden=json.loads(a.golden.read_text())
        for key,group in records(a.records).items():
            expected=golden['cases'][key]
            if group[0]['tokens']!=expected['tokens'] or group[0]['prompt_tokens']!=expected['prompt_tokens']:raise SystemExit('Golden mismatch: '+key)
        print('All 28 profile/workload golden token sequences match')
    else:
        if not all(math.isfinite(x) and 0<=x<1 for x in [a.latency_tolerance,a.decode_tolerance]):p.error('Tolerances must be finite and in [0,1)')
        report=compare(a.before,a.after,a.latency_tolerance,a.decode_tolerance)
        a.out.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({k:v for k,v in report.items() if k!='cases'}))
        if not report['performance_passed']:raise SystemExit('Performance gate requires investigation; do not waive noisy results silently')
if __name__=='__main__':main()
