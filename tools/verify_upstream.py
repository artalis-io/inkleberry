#!/usr/bin/env python3
"""Verify pinned bytes; production checks do not require the optional Zig oracle."""
import argparse,hashlib,json
from pathlib import Path
from upstream import source_path
from native_namespace import original_bytes
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--production-only', action='store_true', help='verify non-Zig pinned sources only')
args=parser.parse_args()
root=Path(__file__).resolve().parents[1]
manifest=json.loads((root/'UPSTREAM.json').read_text())
for name,expected in manifest['files'].items():
    if args.production_only and name.endswith('.zig'):
        continue
    payload=source_path(root, name).read_bytes()
    if name.startswith('cuda/'):
        payload=original_bytes(payload.decode(), root)
    actual=hashlib.sha256(payload).hexdigest()
    if actual!=expected:raise SystemExit('Unreviewed upstream drift: '+name)
print('Pinned production source verified:' if args.production_only else 'Pinned upstream source verified:',manifest['revision'])
