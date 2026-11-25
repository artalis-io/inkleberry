#!/usr/bin/env python3
"""Linux provider export allowlist and SONAME are part of the public contract."""
import subprocess
import sys

path = sys.argv[1]
lines = subprocess.check_output(['nm', '-D', '--defined-only', '-P', path], text=True).splitlines()
names = {line.split()[0].split('@')[0] for line in lines}
assert names == {'SQ_INFER_1.0', 'sq_infer_get_api'}, names
elf = subprocess.check_output(['readelf', '-d', path], text=True)
assert '(SONAME)' in elf and '[libsq_backend.so.1]' in elf
print('provider export allowlist and versioned SONAME passed')
