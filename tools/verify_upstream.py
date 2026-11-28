#!/usr/bin/env python3
"""Verify retained dependency bytes and reversible CUDA adaptations."""
import hashlib,json
from pathlib import Path
from native_namespace import original_bytes
root=Path(__file__).resolve().parents[1]
manifest=json.loads((root/'UPSTREAM.json').read_text())
for name,expected in manifest['files'].items():
    payload=(root / name).read_bytes()
    if name.startswith('cuda/'):
        payload=original_bytes(payload.decode(), root, name)
    actual=hashlib.sha256(payload).hexdigest()
    if actual!=expected:raise SystemExit('Unreviewed upstream drift: '+name)
print('Pinned dependency sources verified:',manifest['revision'])
