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
