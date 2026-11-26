#!/usr/bin/env python3
"""Adapt only link names in disposable Zig oracle copies; originals stay frozen."""
from pathlib import Path
import re
import sys

root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root / 'tools'))
from native_namespace import names

mapping = {k:v for k,v in names(root).items() if k.startswith('squid_')}
out = root / '.build/native-oracle-src'
for source_dir, target in [(root / 'reference/upstream/src', out / 'upstream'),
                           (root / 'reference/oracle', out / 'oracle')]:
    target.mkdir(parents=True, exist_ok=True)
    for source in sorted(source_dir.glob('*.zig')):
        # All changed identifiers are C link references, not model algorithms.
        def link_name(match):
            name = match[0]
            return mapping.get(name, 'ifn_' + name[3:] if name.startswith('sq_') else name)
        text = re.sub(r'\b(?:squid_|sq_)\w+', link_name, source.read_text())
        text = text.replace('SQUID_MTP_BF16', 'INKLEBERRY_MTP_BF16')
        (target / source.name).write_text(text)
print('Prepared disposable Zig copies with current native link names; originals unchanged')
