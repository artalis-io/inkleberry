#!/usr/bin/env python3
"""Check process/error/allocation boundaries across C, CUDA and included headers."""
import argparse
from pathlib import Path
import re
from amalgamate import TOKEN
p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1]); a = p.parse_args()
root = a.root
sources = [f for directory in ['src', 'cuda', 'include', 'dist'] for f in (root / directory).rglob('*')
           if f.suffix in ('.c', '.h', '.cpp', '.cu', '.cuh', '.hpp')]
if not list((root / 'src').glob('*.c')): raise SystemExit('No production C sources found')
for path in sorted(sources):
    code = ''.join(t if not t.startswith(('/*', '//', '"', "'")) else ' ' for t in TOKEN.findall(path.read_text()))
    if re.search(r'\b(exit|_Exit|quick_exit|abort|assert|system|popen)\s*\(', code):
        raise SystemExit(f'{path.name}: process termination/commands forbidden in library')
    if path.name not in ('foundation.c', 'ifn.c') and re.search(r'\b(malloc|calloc|realloc|free)\s*\(', code):
        raise SystemExit(f'{path.name}: use injected allocator')
print('C/CUDA library error and allocation boundaries passed')
