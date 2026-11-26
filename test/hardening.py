#!/usr/bin/env python3
"""Build without compiler-default PIE; enforce it explicitly and reject ET_EXEC."""
import argparse
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--cc', default='cc')
p.add_argument('--cxx', default='c++')
a = p.parse_args()
with tempfile.TemporaryDirectory(prefix='ifn-pie-') as tmp:
    base = Path(tmp)
    source = base / 'source'
    source.mkdir()
    for directory in ['src', 'include', 'cli', 'config', 'mk']:
        shutil.copytree(ROOT / directory, source / directory)
    for name in ['Makefile', 'VERSION']:
        shutil.copy2(ROOT / name, source / name)
    # Simulate a compiler configured without default PIE. Final explicit -fPIE/-pie
    # must override these defaults. Do not pass link-only flags to compile-only calls.
    wrappers = []
    for name, command in [('cc', a.cc), ('cxx', a.cxx)]:
        wrapper = base / name
        wrapper.write_text('#!' + sys.executable + '\nimport os,sys\n'
                           'base=' + repr(shlex.split(command)) + '\n'
                           'args=sys.argv[1:]\n'
                           'defaults=["-fno-PIE"]\n'
                           'if "-c" not in args and "-shared" not in args: defaults += ["-no-pie"]\n'
                           'os.execvp(base[0],base+defaults+args)\n')
        wrapper.chmod(0o755)
        wrappers.append(str(wrapper))
    build = ['make', '-j4', 'CC='+wrappers[0], 'CXX='+wrappers[1]]
    subprocess.run(build+['all'], cwd=source, check=True)
    checker = [sys.executable, str(ROOT/'tools/check_hardening.py')]
    subprocess.run(checker+[str(source/'inkleberry'), str(source/'libinkleberry.so')], check=True)
    # Keep the other protections, intentionally omit PIE: the checker must fail.
    probe = base/'nonpie.c'
    probe.write_text('#include <stdio.h>\nint main(int n,char **v){char b[32];'
                     'snprintf(b,sizeof b,"%s",n>1?v[1]:"probe");return puts(b)<0;}\n')
    executable = base/'nonpie'
    subprocess.run(shlex.split(a.cc)+['-O2','-fno-PIE','-no-pie','-fstack-protector-strong',
                   str(probe),'-Wl,-z,relro,-z,now,-z,noexecstack','-o',str(executable)],check=True)
    result = subprocess.run(checker+[str(executable)],capture_output=True,text=True)
    assert result.returncode and 'executable must be PIE' in result.stderr, result.stderr
    subprocess.run(build+['HARDEN=0', 'all'], cwd=source, check=True)
    elf = subprocess.check_output(['readelf','-W','-h',str(source/'inkleberry')],text=True)
    assert any('Type:' in line and 'EXEC' in line for line in elf.splitlines()), elf
print('Explicit PIE overrides non-PIE compiler defaults; checker rejects ET_EXEC; HARDEN=0 remains usable')
