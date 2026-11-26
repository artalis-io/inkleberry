#!/usr/bin/env python3
"""Verify actual native archive/provider symbol names after CUDA compilation."""
import json
from pathlib import Path
import re
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
bridge = {f['name'] for f in json.loads((root / 'abi/cuda.json').read_text())['functions']}
seen = set()
for filename in sys.argv[1:]:
    text = subprocess.check_output(['nm', '-C', '-g', '--defined-only', filename], text=True)
    for line in text.splitlines():
        match = re.match(r'^\s*[0-9a-fA-F]+\s+[A-Za-z]\s+(.+)$', line)
        if not match:
            continue
        name = match[1].replace('(anonymous namespace)::', '')
        assert not re.search(r'\b(?:squid|sq_|squid_|si_)', name), name
        # C++/CUDA compiler support symbols and standard-library instantiations
        # belong to their toolchains, not the project's source namespace.
        assert name.startswith(('ifn_', 'IFN_', '__', '_GLOBAL_', 'std::',
                                'DW.ref.', 'typeinfo ', 'vtable ', 'guard variable ')), name
        seen.add(name.split('@@')[0])
assert bridge <= seen, sorted(bridge - seen)
assert 'ifn_infer_get_api' in seen
print('Native CUDA archive/provider symbols use the project namespace:', len(bridge), 'bridge functions')
