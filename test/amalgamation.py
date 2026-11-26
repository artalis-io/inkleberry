#!/usr/bin/env python3
"""The embedding distribution must preserve every host library global symbol."""
import subprocess
import sys

def symbols(path):
    result=set()
    for line in subprocess.check_output(['nm','-g',path],text=True).splitlines():
        fields=line.split()
        if len(fields)<3 or fields[-2] in ['U','w','v']:continue
        name=fields[-1]
        if name.startswith('_sq_'):name=name[1:]
        if name.startswith('sq_'):result.add(name)
    return result

left,right=map(symbols,sys.argv[1:])
assert left and left==right, {'missing':sorted(left-right),'extra':sorted(right-left)}
print('Host modular/amalgamated symbol surfaces match:',len(left))
