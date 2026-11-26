#!/usr/bin/env python3
"""Keep the renamed public SDK and host binary free of pre-release API names."""
from pathlib import Path
import re
import subprocess
import os
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from amalgamate import TOKEN
from native_namespace import kernel_names

root = Path(__file__).resolve().parents[1]
headers = sorted((root / 'include').glob('*.h'))
assert len(headers) == 9
for header in headers + [root / 'dist/ifn.h']:
    assert header.name.startswith('ifn')
    text = header.read_text()
    assert not re.search(r'\b(?:sq_|SQ_)[A-Za-z0-9_]*', text), header
    assert 'squid' not in text.lower(), header
assert not list((root / 'dist').glob('sq.*'))
# Include internal C/CUDA declarations: public headers alone cannot guard this.
for directory in ['src', 'cuda', 'include', 'cli', 'examples', 'test', 'fuzz']:
    for path in sorted((root / directory).rglob('*')):
        if path.suffix not in ('.c', '.h', '.cu', '.cuh', '.hpp'):
            continue
        code = ''.join(t if not t.startswith(('/*', '//', '"', "'")) else ' '
                       for t in TOKEN.findall(path.read_text()))
        assert not re.search(r'\b(?:squid|squid_\w*|SQUID_\w*|sq_\w*|SQ_\w*|si_\w*|SI_\w*)\b', code), path
        # CUDA attributes may separate the return type and kernel identifier.
        for name in kernel_names(code):
            assert name.startswith('ifn_'), (path, name)
        for line in code.splitlines():
            if not line or line[0].isspace() or line.startswith(('#', 'template', 'namespace', 'using', 'extern')):
                continue
            match = re.match(r'(?:[A-Za-z_][A-Za-z_0-9:<>,* &]*?\s+|\*+)([A-Za-z_]\w*)\s*\(', line)
            if match and not line.startswith('typedef '):
                name = match[1]
                assert name.startswith(('ifn_', 'IFN_', '__')) or name in ('main', 'LLVMFuzzerTestOneInput', 'static_assert', 'alignas'), (path, name)
        for name in re.findall(r'^#define\s+(\w+)', code, re.M):
            assert name.startswith(('IFN_', '_')), (path, name)
        for name in re.findall(r'^(?:struct|class|namespace)\s+(\w+)', code, re.M):
            assert name.startswith('ifn_'), (path, name)
        for name in re.findall(r'^}\s*(\w+)\s*;', code, re.M):
            assert name.startswith('ifn_'), (path, name)
for filename in sys.argv[1:]:
    if os.name == 'nt' and filename.lower().endswith('.dll'):
        output = subprocess.check_output(['llvm-readobj', '--coff-exports', filename], text=True)
        names = re.findall(r'^  Name: (.+)$', output, re.M)
        assert names and all(n.startswith('ifn_') for n in names), names
        assert 'ifn_status_string' in names
        continue
    output = subprocess.check_output(['llvm-nm' if os.name == 'nt' else 'nm', '-g', filename], text=True)
    defined = []
    for line in output.splitlines():
        fields = line.split()
        if len(fields) >= 3 and fields[-2] not in ('U', 'w', 'v'):
            name = fields[-1]
            if name.startswith('_ifn_'): name = name[1:]
            if os.name == 'nt' and (name in ('@feat.00', '@comp.id', 'snprintf', 'vsnprintf', '_Atomic_exchange8', '_Atomic_store8', '_Check_memory_order', '?_OptionsStorage@?1??__local_stdio_printf_options@@9@9') or name.startswith('??_C@')): continue
            assert name.startswith('ifn_') or name.startswith(('__', '.')), name
            defined.append(name)
    assert 'ifn_status_string' in defined, filename
print('Public/internal C/CUDA declarations and host symbols use the ifn namespace')
