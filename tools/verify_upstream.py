#!/usr/bin/env python3
"""The first port slice may not alter pinned CUDA/ABI/oracle inputs."""
import hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
manifest=json.loads((root/'UPSTREAM.json').read_text())
for name,expected in manifest['files'].items():
    actual=hashlib.sha256((root/name).read_bytes()).hexdigest()
    if actual!=expected:raise SystemExit('Unreviewed upstream drift: '+name)
print('Pinned upstream source verified:',manifest['revision'])
