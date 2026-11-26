#!/usr/bin/env python3
"""Verify ELF hardening on actual host artifacts, independent of compiler defaults."""
import subprocess,sys
from pathlib import Path
for name in sys.argv[1:]:
    p=Path(name)
    headers=subprocess.check_output(['readelf','-W','-l',str(p)],text=True)
    dynamic=subprocess.check_output(['readelf','-W','-d',str(p)],text=True)
    assert 'GNU_RELRO' in headers,p
    stack=next(s for s in headers.splitlines() if 'GNU_STACK' in s)
    assert 'RWE' not in stack and ' RW ' in stack,p
    assert 'BIND_NOW' in dynamic or 'NOW' in dynamic,p
    symbols=subprocess.check_output(['nm','-u',str(p)],text=True)
    assert '__stack_chk_fail' in symbols,p
    if '(SONAME)' not in dynamic:
        elf=subprocess.check_output(['readelf','-W','-h',str(p)],text=True)
        assert any('Type:' in line and 'DYN' in line for line in elf.splitlines()), (p, 'executable must be PIE')
        assert any('FLAGS_1' in line and 'PIE' in line for line in dynamic.splitlines()), (p, 'missing PIE flag')
    if p.name.startswith('libinkleberry.so'):
        assert '(SONAME)' in dynamic and '[libinkleberry.so.1]' in dynamic,p
print('Host artifacts have non-executable stack, full RELRO/NOW, stack protection, executable PIE and versioned host SONAME')
