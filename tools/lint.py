#!/usr/bin/env python3
"""Small explicit invariants for the new library, not upstream kernel policy."""
import re
from pathlib import Path
root = Path(__file__).resolve().parents[1]
for p in (root / 'src/c').glob('*.c'):
    text = p.read_text()
    if re.search(r'\b(exit|abort|assert|system|popen)\s*\(', text):
        raise SystemExit(f'{p.name}: process termination/commands forbidden in library')
    if p.name != 'foundation.c' and re.search(r'\b(malloc|calloc|realloc|free)\s*\(', text):
        raise SystemExit(f'{p.name}: use injected allocator')
print('C library source invariants passed')
