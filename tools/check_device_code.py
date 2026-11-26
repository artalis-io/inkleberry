#!/usr/bin/env python3
"""Compare actual sm_120a machine instruction/control words by kernel function."""
import argparse,json,re,subprocess
from pathlib import Path
OBJECTS=['kernels_gemv.cuda.o','kernels_inkling.cuda.o','kernels_dspark.cuda.o','checked-prefill.cuda.o','checked-weights.cuda.o']
def kernels(tool,path):
    text=subprocess.check_output([tool,'--dump-sass',str(path)],text=True)
    result={};current=None
    for line in text.splitlines():
        m=re.search(r'Function : (\S+)',line)
        if m:
            current=re.sub(r'_GLOBAL__N__[0-9a-f]+_[0-9]+_[A-Za-z0-9_]+?_cu_[0-9a-f]+','_GLOBAL__N__',m[1])
            if current in result:raise ValueError('Ambiguous kernel identity')
            result[current]=[]
        elif current:
            result[current]+=re.findall(r'/\*\s*(0x[0-9a-f]+)\s*\*/',line)
    if not result or any(not v for v in result.values()):raise ValueError('No complete device code found: '+str(path))
    return result
p=argparse.ArgumentParser(description=__doc__);p.add_argument('before',type=Path);p.add_argument('after',type=Path);p.add_argument('--cuobjdump',default='/usr/local/cuda/bin/cuobjdump');p.add_argument('--out',type=Path,required=True);a=p.parse_args()
report={}
for name in OBJECTS:
    before=kernels(a.cuobjdump,a.before/name);after=kernels(a.cuobjdump,a.after/name)
    if before!=after:raise SystemExit('Device machine-code difference requires review: '+name)
    report[name]={'functions':len(before),'instruction_words':sum(map(len,before.values())),'identical':True}
a.out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
