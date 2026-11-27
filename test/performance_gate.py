#!/usr/bin/env python3
import copy,importlib.util,json,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('model_release',ROOT/'tools/model_release.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
rows=[{'case':case,'profile':profile,'warmup':False,'run':run,'generated':20,'tokens':[17]*20,'prompt_tokens':32,'first_token_ns':100,'elapsed_ns':1000} for case in range(7) for profile in ['ar','fixed','adaptive','sampled'] for run in range(5)]
with tempfile.TemporaryDirectory() as tmp:
    a=Path(tmp)/'a';b=Path(tmp)/'b'
    def write(p,r):p.write_text('\n'.join(json.dumps(x) for x in r))
    write(a,rows);write(b,rows);assert m.compare(a,b)['performance_passed']
    bad=copy.deepcopy(rows)
    for r in bad:r['elapsed_ns']*=2
    write(b,bad);assert not m.compare(a,b)['performance_passed']
    for r in bad:r['tokens'][0]=18
    write(b,bad)
    try:m.compare(a,b);raise AssertionError('quality regression accepted')
    except ValueError:pass
    write(b,rows[:-1])
    try:m.records(b);raise AssertionError('incomplete repetitions accepted')
    except ValueError:pass
print('Performance/quality gate rejects slower, changed-output and incomplete campaigns')

# Non-first repetitions must have the same inputs/settings, not only output.
with tempfile.TemporaryDirectory() as tmp:
    path=Path(tmp)/'records.jsonl'
    def rejected(changed):
        path.write_text('\n'.join(json.dumps(r) for r in changed))
        try:m.records(path);raise AssertionError('invalid campaign accepted')
        except ValueError:pass
    for field,value in [('prompt_tokens',33),('run',0),('case',7),('elapsed_ns',0)]:
        bad=copy.deepcopy(rows);bad[1][field]=value;rejected(bad)
    identified=copy.deepcopy(rows)
    for r in identified:r.update(workload_revision='native-benchmark-v2',input_sha256='a'*64,config_sha256='b'*64)
    path.write_text('\n'.join(json.dumps(r) for r in identified));m.records(path,require_identity=True)
    for field,value in [('input_sha256','c'*64),('config_sha256','c'*64),('workload_revision','other')]:
        bad=copy.deepcopy(identified);bad[1][field]=value;rejected(bad)
    bad=copy.deepcopy(identified);del bad[1]['config_sha256'];rejected(bad)
print('Mixed prompts/settings, duplicate repetitions and malformed identities rejected')
