#!/usr/bin/env python3
"""Keep the renamed public SDK and host binary free of pre-release API names."""
from pathlib import Path
import re
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
headers = sorted((root / 'include').glob('*.h'))
assert len(headers) == 7
for header in headers + [root / 'dist/ifn.h']:
    assert header.name.startswith('ifn')
    text = header.read_text()
    assert not re.search(r'\b(?:sq_|SQ_)[A-Za-z0-9_]*', text), header
    assert 'squid' not in text.lower(), header
assert not list((root / 'dist').glob('sq.*'))
for filename in sys.argv[1:]:
    output = subprocess.check_output(['nm', '-g', filename], text=True)
    defined = []
    for line in output.splitlines():
        fields = line.split()
        if len(fields) >= 3 and fields[-2] not in ('U', 'w', 'v'):
            name = fields[-1].removeprefix('_')
            assert not name.startswith(('sq_', 'SQ_', 'squid_')), name
            defined.append(name)
    assert 'ifn_status_string' in defined, filename
print('Public headers and host symbols use the ifn namespace exclusively')
